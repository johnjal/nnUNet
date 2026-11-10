"""Out-of-fold whole-volume region metrics; no training or inference on import."""
import argparse
import json
from workflow import guard, paths, model_dir, read_config, write_json, freeze_run

REGIONS = {'kidney': (1, 2, 3), 'masses': (2, 3), 'tumor': (2,)}


def region_metrics(gt, pred, spacing, tolerance):
    import numpy as np
    from surface_distance import metrics
    total = int(gt.sum()) + int(pred.sum())
    if not total:
        # Exclude both-empty cases rather than reward absent anatomy.
        return float('nan'), float('nan')
    dice = 2 * np.count_nonzero(gt & pred) / total
    if not gt.any() or not pred.any():
        return float(dice), 0.0
    distances = metrics.compute_surface_distances(gt, pred, spacing)
    return float(dice), float(metrics.compute_surface_dice_at_tolerance(distances, tolerance))


def evaluate(c):
    guard(c)
    freeze_run(c)  # Refuse evaluating changed plans/splits against an existing run.
    import nibabel as nib
    import numpy as np
    import pandas as pd
    p = paths(c)
    splits = json.loads((p['run']/'splits_final.json').read_text())
    rows = []
    for fold in c['folds']:
        folder = model_dir(c)/f'fold_{fold}'
        if not (folder/'validation'/'summary.json').exists():
            raise FileNotFoundError(f'Final validation not complete: {folder}')
        for case in splits[fold]['val']:
            truth = nib.load(p['raw']/c['dataset_name']/'labelsTr'/f'{case}.nii.gz')
            prediction = nib.load(folder/'validation'/f'{case}.nii.gz')
            if truth.shape != prediction.shape or not np.allclose(truth.affine, prediction.affine, atol=1e-4):
                raise ValueError(f'Geometry mismatch: {case}; do not silently resample evaluation masks')
            gt, pred = np.asanyarray(truth.dataobj), np.asanyarray(prediction.dataobj)
            if not np.isin(gt, [0, 1, 2, 3]).all() or not np.isin(pred, [0, 1, 2, 3]).all():
                raise ValueError(f'Unexpected labels in {case}')
            spacing = tuple(float(x) for x in truth.header.get_zooms()[:3])
            for region, labels in REGIONS.items():
                dice, nsd = region_metrics(np.isin(gt, labels), np.isin(pred, labels), spacing, c['nsd_tolerance_mm'])
                rows.append(dict(fold=fold, case=case, region=region, dice=dice, nsd=nsd))
            print(f'Fold {fold}: {case}', flush=True)
    frame = pd.DataFrame(rows)
    per_region = frame.groupby(['fold', 'region'])[['dice', 'nsd']].mean()
    per_fold = per_region.groupby('fold').mean()
    summary = per_fold.mean() * 100
    report = p['run']/'reports'
    report.mkdir(exist_ok=True)
    suffix = '-'.join(map(str, sorted(c['folds'])))
    frame.to_csv(report/f'cases-folds-{suffix}.csv', index=False)
    per_region.to_csv(report/f'regions-folds-{suffix}.csv')
    per_fold.to_csv(report/f'folds-{suffix}.csv')
    comparison = {
        'complete_5fold': sorted(c['folds']) == list(range(5)),
        'folds': c['folds'], 'dice_percent': float(summary['dice']), 'nsd_percent': float(summary['nsd']),
        'paper_dice_percent': 86.04, 'paper_nsd_percent': 82.91,
        'aggregation': 'case mean per region, then region mean per fold, then equal fold mean; both-empty excluded',
        'checkpoint': 'final', 'nsd_tolerance_mm': c['nsd_tolerance_mm'],
        'limitations': c['reproduction_notes'],
    }
    write_json(report/f'comparison-folds-{suffix}.json', comparison)
    return frame, per_fold, comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    evaluate(read_config(parser.parse_args().config))
