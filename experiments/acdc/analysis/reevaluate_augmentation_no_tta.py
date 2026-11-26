"""Re-evaluate ACDC augmentation checkpoints with mirror TTA disabled; no training."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'experiments/acdc/scripts'))
from workflow import digest, write_json, model_dir, environment
from run_augmentation_suite import TRAINERS
from compare_best import mask_metrics, aggregate, write_csv


def models_and_cases(config):
    root = Path(config['storage_root'])
    records, reference = [], None
    for name, run_id in config['runs'].items():
        if name not in [*TRAINERS, 'baseline_1000'] or not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
            raise ValueError('Invalid experiment name or run ID')
        run = root / 'runs' / run_id
        saved = json.loads((run/'config.snapshot.json').read_text())['configuration']
        expected = 'nnUNetTrainer' if name == 'baseline_1000' else TRAINERS[name]
        if saved['trainer'] != expected or saved['dataset_id'] != 27:
            raise ValueError(f'{name}: checkpoint run is not the expected ACDC trainer')
        split = json.loads((run/'splits_final.json').read_text())[0]
        canonical = {k: sorted(split[k]) for k in ('train','val')}
        if len(set(canonical['train'])) != 160 or len(set(canonical['val'])) != 40:
            raise ValueError(f'{name}: expected 160/40 unique volumes')
        patients = lambda ids: {case.rsplit('_frame',1)[0] for case in ids}
        if patients(canonical['train']) & patients(canonical['val']):
            raise ValueError('Patient leakage across train/validation')
        if reference is None:
            reference = canonical
        elif canonical != reference:
            raise ValueError(f'{name}: different train/validation split')
        folder = run/'nnUNet_results'/'Dataset027_ACDC'/f"{expected}__{saved['plans']}__{saved['configuration']}"
        dataset=json.loads((folder/'dataset.json').read_text())
        if dataset['labels'] != {'background':0,'RV':1,'MLV':2,'LVC':3}:
            raise ValueError('Unexpected ACDC labels')
        for kind in config['checkpoints']:
            if kind not in ('best','final'):
                raise ValueError('Choose best and/or final')
            checkpoint=next((folder/'fold_0'/f'checkpoint_{kind}.{ext}' for ext in ('safetensors','pth')
                             if (folder/'fold_0'/f'checkpoint_{kind}.{ext}').is_file()),None)
            if checkpoint is None:
                raise FileNotFoundError(f'{name}: checkpoint_{kind} missing in {folder}')
            records.append(dict(name=f'{name}/{kind}', experiment=name, kind=kind,run_id=run_id,
                                model=folder,checkpoint=checkpoint))
    if not records:
        raise ValueError('Select at least one run and checkpoint')
    return records,reference['val']


def run(config):
    output_id = config["output_id"]
    if sys.platform != 'linux' or not config.get('execution_enabled', False):
        raise RuntimeError('Run checkpoint inference on the Linux GPU server only')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', output_id):
        raise ValueError('Invalid output ID')
    c = dict(storage_root=config['storage_root'],run_id='evaluation-only',gpu=config['gpu'],
             augmentation_workers=2,compile=False)
    os.environ.update(environment(c))
    import fcntl
    import torch
    import nnunetv2
    if Path(nnunetv2.__file__).resolve().parent != REPO / 'nnunetv2':
        raise RuntimeError('Install this checkout in the current kernel environment')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    records, cases = models_and_cases(config)
    root = Path(config['storage_root']).expanduser().resolve()
    raw = root / 'nnUNet_raw/Dataset027_ACDC'
    output = root / 'augmentation_reevaluation' / output_id
    output.mkdir(parents=True, exist_ok=True)
    with (root / f"gpu-{config['gpu']}.lock").open('a') as gpu_lock, (output / 'evaluation.lock').open('a') as eval_lock:
        try:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(eval_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('GPU runner or best-checkpoint evaluation is still active') from None
        from evaluate_regions import read_masks
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.utilities.checkpoints import load_checkpoint_file
        code = hashlib.sha256()
        for file in sorted((REPO / 'nnunetv2').rglob('*.py')):
            code.update(str(file.relative_to(REPO)).encode())
            code.update(file.read_bytes())
        provenance = dict(
            cases=cases, checkpoints=config['checkpoints'], step_size=0.5, gaussian=True, use_mirroring=False, mirror_axes=[],
            models={r['name']: dict(run_id=r['run_id'], checkpoint_sha256=digest(r['checkpoint']),
                    plans_sha256=digest(r['model'] / 'plans.json'), dataset_sha256=digest(r['model'] / 'dataset.json'))
                    for r in records},
            inputs={case: {kind: digest(raw / folder / f'{case}{suffix}.nii.gz')
                          for kind, folder, suffix in [('image', 'imagesTr', '_0000'), ('label', 'labelsTr', '')]}
                    for case in cases},
            code_sha256=code.hexdigest(), evaluator_sha256=digest(__file__),
            metrics_sha256=digest(Path(__file__).with_name('compare_best.py')),
            geometry_reader_sha256=digest(REPO / 'experiments/acdc/scripts/evaluate_regions.py'),
            packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']})
        manifest = output / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != provenance:
            raise ValueError('Evaluation inputs/code/environment changed; choose a new OUTPUT_ID')
        if not manifest.exists() and any(p.name != 'evaluation.lock' for p in output.iterdir()):
            raise ValueError('Nonempty output without provenance; choose a new OUTPUT_ID')
        write_json(manifest, provenance)
        comparison = []
        for record in records:
            name = record['name']
            dest = output / name
            dest.mkdir(parents=True, exist_ok=True)
            print(f'=== {name}: TTA OFF, same fold 0 validation ===', flush=True)
            pending = []
            for case in cases:
                if not (dest / f'{case}.nii.gz').is_file():
                    pending.append(case)
                    continue
                try:
                    gt, pred, _ = read_masks(raw / 'labelsTr' / f'{case}.nii.gz', dest / f'{case}.nii.gz')
                    mask_metrics(gt, pred)
                except (RuntimeError, FileNotFoundError, ValueError):
                    pending.append(case)
            if pending:
                predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                            device=torch.device('cuda', 0), perform_everything_on_device=False)
                predictor.initialize_from_trained_model_folder(str(record['model']), use_folds=(0,),
                                                               checkpoint_name=record['checkpoint'].name)
                predictor.predict_from_files_sequential(
                    [[str(raw / 'imagesTr' / f'{case}_0000.nii.gz')] for case in pending],
                    [str(dest / case) for case in pending], save_probabilities=False, overwrite=True)
                del predictor
                torch.cuda.empty_cache()
            rows = []
            for case in cases:
                gt, pred, _ = read_masks(raw / 'labelsTr' / f'{case}.nii.gz', dest / f'{case}.nii.gz')
                rows.extend(dict(case=case, **m) for m in mask_metrics(gt, pred))
            meta = load_checkpoint_file(str(record['checkpoint']), map_location=torch.device('cpu'),
                                        keys=('current_epoch',))
            summary = dict(experiment=record['experiment'], run_id=record['run_id'], checkpoint=record['kind'], tta=False,
                           checkpoint_completed_epochs=meta.get('current_epoch'), validation_cases=len(cases),
                           training_budget=1000 if record['experiment'] == 'baseline_1000' else 250, **aggregate(rows))
            write_csv(dest / 'case-metrics.csv', rows)
            write_json(dest / 'summary.json', summary)
            comparison.append(summary)
            # Partial progress is explicitly tagged; notebook checks expected row count.
            write_json(output / 'comparison.json', dict(complete=len(comparison) == len(records), results=comparison))
            write_csv(output / 'comparison.csv', comparison)
            print(summary, flush=True)
        print(f'TTA-OFF evaluation saved: {output}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    run(config)
