"""Out-of-fold whole-volume region metrics; no training or inference on import."""
import argparse
import json
from workflow import guard, paths, model_dir, read_config, write_json, freeze_run

REGIONS = {'RV': (1,), 'MLV': (2,), 'LVC': (3,)}


def read_masks(truth_path, prediction_path):
    """Use nnU-Net's SimpleITK geometry interpretation, without resampling."""
    import SimpleITK as sitk
    import numpy as np
    truth = sitk.ReadImage(str(truth_path))
    prediction = sitk.ReadImage(str(prediction_path))
    if truth.GetDimension() != 3 or prediction.GetDimension() != 3:
        raise ValueError(f'Geometry mismatch: {truth_path}; expected two 3D masks')
    differences = []
    for name in ('Dimension', 'Size', 'Spacing', 'Origin', 'Direction'):
        actual = getattr(truth, f'Get{name}')()
        predicted = getattr(prediction, f'Get{name}')()
        equal = actual == predicted if name in ('Dimension', 'Size') else np.allclose(
            actual, predicted, rtol=0, atol=1e-4)
        if not equal:
            differences.append(f'{name}: reference={actual}, prediction={predicted}')
    if differences:
        raise ValueError(f'Geometry mismatch: {truth_path}; ' + '; '.join(differences)
                         + '; expected matching 3D masks; do not silently resample evaluation masks')
    # SimpleITK arrays are z,y,x; physical spacing must follow the same order for NSD.
    return (sitk.GetArrayFromImage(truth), sitk.GetArrayFromImage(prediction),
            tuple(reversed(truth.GetSpacing())))


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
    provenance = freeze_run(c, evaluation=True)
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
            gt, pred, spacing = read_masks(
                p['raw']/c['dataset_name']/'labelsTr'/f'{case}.nii.gz',
                folder/'validation'/f'{case}.nii.gz')
            if not np.isin(gt, [0, 1, 2, 3]).all() or not np.isin(pred, [0, 1, 2, 3]).all():
                raise ValueError(f'Unexpected labels in {case}')
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
        'paper_dice_percent': 91.54, 'paper_nsd_percent': 94.93,
        'aggregation': 'case mean per region, then region mean per fold, then equal fold mean; both-empty excluded',
        'checkpoint': 'final', 'nsd_tolerance_mm': c['nsd_tolerance_mm'],
        'geometry_reader': 'SimpleITK; arrays and spacing in z,y,x order; no resampling',
        'evaluation_provenance': provenance,
        'limitations': c['reproduction_notes'],
    }
    write_json(report/f'comparison-folds-{suffix}.json', comparison)
    return frame, per_fold, comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    evaluate(read_config(parser.parse_args().config))
