"""Evaluate best checkpoints without touching training snapshots or final predictions."""
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
from run_augmentation_suite import read_suite, theme_config, suite_dir


def models_and_cases(suite, baseline_run):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', baseline_run):
        raise ValueError('Invalid baseline run ID')
    root = Path(suite['storage_root']).expanduser().resolve()
    c = theme_config(suite, suite['themes'][0])
    c.update(run_id=baseline_run, trainer='nnUNetTrainer')
    entries = [('baseline_1000', c)] + [(t, theme_config(suite, t)) for t in suite['themes']]
    records, reference = [], None
    for name, config in entries:
        folder = model_dir(config)
        run = root / 'runs' / config['run_id']
        split = json.loads((run / 'splits_final.json').read_text())[0]
        canonical = {key: sorted(split[key]) for key in ('train', 'val')}
        if len(canonical['val']) != 40 or len(set(canonical['val'])) != 40:
            raise ValueError(f'{name}: expected 40 unique validation cases')
        if set(canonical['train']) & set(canonical['val']):
            raise ValueError('Train/validation overlap')
        if reference is None:
            reference = canonical
        elif reference != canonical:
            raise ValueError(f'{name}: training/validation split differs from baseline')
        fold = folder / 'fold_0'
        checkpoint = next((fold / n for n in ('checkpoint_best.safetensors', 'checkpoint_best.pth')
                           if (fold / n).is_file()), None)
        if checkpoint is None:
            raise FileNotFoundError(f'{name}: checkpoint_best missing in {fold}')
        if not any(fold.glob('checkpoint_final.*')) or not (fold / 'validation/summary.json').exists():
            raise RuntimeError(f'{name}: training and final validation must finish first')
        for path in (folder / 'plans.json', folder / 'dataset.json'):
            if not path.is_file():
                raise FileNotFoundError(path)
        records.append(dict(name=name, run_id=config['run_id'], model=folder, checkpoint=checkpoint))
    return records, reference['val']


def mask_metrics(gt, pred):
    import numpy as np
    if gt.shape != pred.shape or not np.isin(gt, [0, 1, 2, 3]).all() or not np.isin(pred, [0, 1, 2, 3]).all():
        raise ValueError('Invalid mask shape or labels')
    values = []
    for label, region in [(1, 'RV'), (2, 'MLV'), (3, 'LVC')]:
        a, b = gt == label, pred == label
        tp = int(np.count_nonzero(a & b))
        total = int(a.sum()) + int(b.sum())
        union = total - tp
        values.append(dict(region=region, dice=2 * tp / total if total else float('nan'),
                           iou=tp / union if union else float('nan')))
    return values


def aggregate(rows):
    import numpy as np
    values = {}
    for region in ('RV', 'MLV', 'LVC'):
        for metric in ('dice', 'iou'):
            v = [r[metric] for r in rows if r['region'] == region and np.isfinite(r[metric])]
            values[f'{metric}_{region}_percent'] = float(np.mean(v) * 100) if v else float('nan')
    for metric, name in [('dice', 'dice_percent'), ('iou', 'miou_percent')]:
        v = [values[f'{metric}_{region}_percent'] for region in ('RV', 'MLV', 'LVC')]
        values[name] = float(np.nanmean(v))
    return values


def write_csv(path, rows):
    path = Path(path)
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def run(suite, baseline_run, output_id):
    if sys.platform != 'linux' or not suite.get('execution_enabled', False):
        raise RuntimeError('Run checkpoint inference on the Linux GPU server only')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', output_id):
        raise ValueError('Invalid output ID')
    c = theme_config(suite, suite['themes'][0])
    os.environ.update(environment(c))
    import fcntl
    import torch
    import nnunetv2
    if Path(nnunetv2.__file__).resolve().parent != REPO / 'nnunetv2':
        raise RuntimeError('Install this checkout in the current kernel environment')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    records, cases = models_and_cases(suite, baseline_run)
    root = Path(suite['storage_root']).expanduser().resolve()
    raw = root / 'nnUNet_raw/Dataset027_ACDC'
    output = suite_dir(suite) / output_id
    output.mkdir(parents=True, exist_ok=True)
    with (root / f"gpu-{suite['gpu']}.lock").open('a') as gpu_lock, (output / 'evaluation.lock').open('a') as eval_lock:
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
            cases=cases, checkpoint='best', step_size=0.5, gaussian=True, mirror_axes=[0, 1, 2],
            models={r['name']: dict(run_id=r['run_id'], checkpoint_sha256=digest(r['checkpoint']),
                    plans_sha256=digest(r['model'] / 'plans.json'), dataset_sha256=digest(r['model'] / 'dataset.json'))
                    for r in records},
            inputs={case: {kind: digest(raw / folder / f'{case}{suffix}.nii.gz')
                          for kind, folder, suffix in [('image', 'imagesTr', '_0000'), ('label', 'labelsTr', '')]}
                    for case in cases},
            code_sha256=code.hexdigest(), evaluator_sha256=digest(__file__),
            geometry_reader_sha256=digest(REPO / 'experiments/acdc/scripts/evaluate_regions.py'),
            packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']})
        manifest = output / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != provenance:
            raise ValueError('Evaluation inputs/code/environment changed; choose a new BEST_OUTPUT_ID')
        if not manifest.exists() and any(p.name != 'evaluation.lock' for p in output.iterdir()):
            raise ValueError('Nonempty output without provenance; choose a new BEST_OUTPUT_ID')
        write_json(manifest, provenance)
        comparison = []
        for record in records:
            name = record['name']
            dest = output / name
            dest.mkdir(exist_ok=True)
            print(f'=== {name}: checkpoint_best, same fold 0 validation ===', flush=True)
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
                predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                            device=torch.device('cuda', 0), perform_everything_on_device=True)
                predictor.initialize_from_trained_model_folder(str(record['model']), use_folds=(0,),
                                                               checkpoint_name=record['checkpoint'].name)
                if tuple(predictor.allowed_mirroring_axes or ()) != (0, 1, 2):
                    raise ValueError(f'{name}: checkpoint TTA axes differ from common protocol')
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
            summary = dict(experiment=name, run_id=record['run_id'], checkpoint='best',
                           checkpoint_completed_epochs=meta.get('current_epoch'), validation_cases=len(cases),
                           training_budget=1000 if name == 'baseline_1000' else 250, **aggregate(rows))
            write_csv(dest / 'case-metrics.csv', rows)
            write_json(dest / 'summary.json', summary)
            comparison.append(summary)
            # Partial progress is explicitly tagged; notebook checks expected row count.
            write_json(output / 'comparison.json', dict(complete=len(comparison) == len(records), results=comparison))
            write_csv(output / 'comparison.csv', comparison)
            print(summary, flush=True)
        print(f'Best-checkpoint comparison saved: {output}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--baseline-run', default='acdc-baseline-v1')
    p.add_argument('--output-id', default='best-comparison-v1')
    a = p.parse_args()
    run(read_suite(a.config), a.baseline_run, a.output_id)
