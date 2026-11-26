"""Replace AMOS OFF final/best validation only after both no-TTA evaluations succeed."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import workflow as w


def run(config_path):
    import fcntl
    c = w.validate(w.read(config_path))
    if sys.platform != 'linux':
        raise RuntimeError('Run on the Linux GPU server only')
    os.environ.update(w.environment(c))
    p = w.locations(c)
    snapshot = w.read(p['suite']/'snapshot.json')
    if snapshot['configuration'] != c or snapshot['splits'] != w.sha(p['pre']/'splits_final.json'):
        raise ValueError('Use the original run configuration and frozen validation split')
    with (p['root']/'amos-workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        from nnunetv2.utilities.checkpoints import find_checkpoint_file
        from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerAMOS250 import AMOSPredictor, AMOSAMPPredictor
        from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = w.model_dir(c, 'off')
        fold = model/'fold_0'
        ids = w.read(p['pre']/'splits_final.json')[0]['val']
        checkpoints = {}
        for name in ('final', 'best'):
            checkpoint = find_checkpoint_file(str(fold/f'checkpoint_{name}.safetensors'))
            if not checkpoint:
                raise FileNotFoundError(f'Missing {name} checkpoint')
            checkpoints[name] = Path(checkpoint)
        # Temporary predictions are on the same filesystem, permitting rename on publication.
        with tempfile.TemporaryDirectory(prefix='no-tta-evaluation-', dir=fold) as temporary:
            staging = Path(temporary)
            provenance = {}
            for name, checkpoint in checkpoints.items():
                folder = 'validation' if name == 'final' else 'validation_best'
                output = staging/folder
                output.mkdir()
                predictor = (AMOSAMPPredictor if c['amp'] else AMOSPredictor)(
                    device=torch.device('cuda'), perform_everything_on_device=False, use_mirroring=False)
                predictor.initialize_from_trained_model_folder(
                    str(model), use_folds=(0,), checkpoint_name=checkpoint.name)
                print(f'Evaluating {name}: TTA OFF, {len(ids)} cases', flush=True)
                predictor.predict_from_files_sequential(
                    [[str(p['raw']/'imagesTr'/f'{case}_0000.nii.gz')] for case in ids],
                    [str(output/case) for case in ids], save_probabilities=False, overwrite=True)
                compute_metrics_on_folder2(str(p['raw']/'labelsTr'), str(output),
                    str(p['raw']/'dataset.json'), str(p['pre']/f"{p['plans']}.json"),
                    num_processes=c['workers'], chill=True)
                summary = w.read(output/'summary.json')
                if {f.name for f in output.glob('*.nii.gz')} != {f'{case}.nii.gz' for case in ids} or len(summary['metric_per_case']) != len(ids):
                    raise ValueError('Prediction cases do not match the frozen split')
                provenance[name] = dict(checkpoint_sha256=w.sha(checkpoint), use_mirroring=False,
                                        validation_ids=ids, amp=c['amp'])
                w.write(output/'evaluation_settings.json', provenance[name])
                del predictor
                torch.cuda.empty_cache()
            # Roll back publication if either directory replacement fails.
            moved, installed = [], []
            try:
                for folder in ('validation', 'validation_best'):
                    old = fold/folder
                    if old.exists():
                        old.rename(staging/(folder+'.previous'))
                        moved.append(folder)
                    (staging/folder).rename(old)
                    installed.append(folder)
            except BaseException:
                for folder in installed:
                    shutil.rmtree(fold/folder)
                for folder in moved:
                    (staging/(folder+'.previous')).rename(fold/folder)
                raise
            # Context exit deletes the previous TTA predictions only after both replacements succeed.
        # Remove the earlier one-case probe, which is no longer needed.
        probe = fold/'validation_best_no_tta_probe'
        if probe.exists():
            shutil.rmtree(probe)
        w.write(p['suite']/'evaluation_no_tta.json', provenance)
        # Delete stale derived reports; regenerate numerical reports now, plots in the notebook.
        for name in ('comparison.json', 'comparison.csv', 'comparison.png', 'per-organ-comparison.csv'):
            (p['suite']/name).unlink(missing_ok=True)
        rows = w.comparison(c)
        import csv
        names = {str(v): k for k, v in w.read(p['raw']/'dataset.json')['labels'].items()}
        with (p['suite']/'per-organ-comparison.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['augmentation','checkpoint','organ','Dice','IoU'])
            writer.writeheader()
            for checkpoint, folder in [('final','validation'),('best','validation_best')]:
                for label, metrics in w.read(fold/folder/'summary.json')['mean'].items():
                    writer.writerow(dict(augmentation='off', checkpoint=checkpoint, organ=names.get(label,label),
                                         Dice=metrics['Dice'], IoU=metrics['IoU']))
        print(json.dumps(rows, indent=2), flush=True)
        print('Replaced final/best validation and reports with TTA OFF results.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    run(parser.parse_args().config)
