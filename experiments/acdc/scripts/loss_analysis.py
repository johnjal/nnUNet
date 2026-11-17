"""Whole-volume probability analysis. Inference is explicit and Linux/CUDA-only."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

REGIONS = {1: 'RV', 2: 'MLV', 3: 'LVC'}
EPS = 1e-7
SMOOTH = 1e-5


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def context(storage, run_id='acdc-baseline-v1', fold=0, checkpoint='checkpoint_final.safetensors'):
    storage = Path(storage).expanduser().resolve()
    run = storage / 'runs' / run_id
    snapshot = json.loads((run / 'config.snapshot.json').read_text())
    # Old absolute server paths in the snapshot are deliberately not used.
    config = snapshot.get('configuration', snapshot)
    dataset = config.get('dataset_name', 'Dataset027_ACDC')
    model = run / 'nnUNet_results' / dataset / (
        f"{config.get('trainer', 'nnUNetTrainer')}__{config.get('plans', 'nnUNetPlans')}__"
        f"{config.get('configuration', '3d_fullres')}")
    raw = storage / 'nnUNet_raw' / dataset
    splits = json.loads((run / 'splits_final.json').read_text())
    split = splits[fold]
    if set(split['train']) & set(split['val']):
        raise ValueError('Train/validation overlap')
    if {c.split('_')[0] for c in split['train']} & {c.split('_')[0] for c in split['val']}:
        raise ValueError('Patient leakage between train and validation')
    cases = [(s, c) for s in ('train', 'val') for c in sorted(split[s])]
    expected = {p.name.removesuffix('.nii.gz') for p in (raw / 'labelsTr').glob('*.nii.gz')}
    if set(c for _, c in cases) != expected or len(cases) != len(expected):
        raise ValueError('Split does not cover each raw case exactly once')
    for file in [model / 'plans.json', model / 'dataset.json', model / f'fold_{fold}' / checkpoint]:
        if not file.is_file():
            raise FileNotFoundError(file)
    for _, case in cases:
        for file in [raw / 'imagesTr' / f'{case}_0000.nii.gz', raw / 'labelsTr' / f'{case}.nii.gz']:
            if not file.is_file():
                raise FileNotFoundError(file)
    out = run / 'loss_analysis' / f'fold_{fold}' / Path(checkpoint).stem
    return dict(storage=storage, run=run, raw=raw, model=model, fold=fold,
                checkpoint=checkpoint, out=out, cases=cases)


def matching_geometry(a, b):
    import numpy as np
    if a.GetDimension() != 3 or b.GetDimension() != 3:
        raise ValueError('Expected 3D data')
    for key in ('Size', 'Spacing', 'Origin', 'Direction'):
        if not np.allclose(getattr(a, 'Get' + key)(), getattr(b, 'Get' + key)(), rtol=0, atol=1e-4):
            raise ValueError(f'Geometry mismatch: {key}; no automatic resampling')


def validate_probabilities(gt, prob):
    import numpy as np
    if prob.shape != (4, *gt.shape) or not np.isin(gt, [0, 1, 2, 3]).all():
        raise ValueError('Unexpected probability shape or ground-truth labels')
    if not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1.00001:
        raise ValueError('Invalid probabilities')
    if not np.allclose(prob.sum(axis=0), 1, rtol=0, atol=2e-4):
        raise ValueError('Probabilities do not sum to 1')


def load_case(ctx, subset, case, with_image=False):
    import numpy as np
    import SimpleITK as sitk
    if (subset, case) not in ctx['cases']:
        raise ValueError('Unknown case or subset')
    gt_img = sitk.ReadImage(str(ctx['raw'] / 'labelsTr' / f'{case}.nii.gz'))
    base = ctx['out'] / subset / case
    pred_img = sitk.ReadImage(str(base) + '.nii.gz')
    matching_geometry(gt_img, pred_img)
    gt, pred = sitk.GetArrayFromImage(gt_img), sitk.GetArrayFromImage(pred_img)
    with np.load(str(base) + '.npz', allow_pickle=False) as archive:
        prob = archive['probabilities'].astype(np.float32)
    validate_probabilities(gt, prob)
    if not np.array_equal(prob.argmax(axis=0), pred):
        raise ValueError('Mask differs from probability argmax')
    image = None
    if with_image:
        img = sitk.ReadImage(str(ctx['raw'] / 'imagesTr' / f'{case}_0000.nii.gz'))
        matching_geometry(gt_img, img)
        image = sitk.GetArrayFromImage(img)
    return gt, pred, prob, tuple(reversed(gt_img.GetSpacing())), image


def scores(gt, prob):
    """All-voxel CE and foreground macro soft Dice; no deep supervision/augmentation."""
    import numpy as np
    validate_probabilities(gt, prob)
    pred = prob.argmax(axis=0)
    p_true = np.take_along_axis(prob, gt.astype(np.int64)[None], axis=0)[0]
    ce_map = -np.log(np.clip(p_true, EPS, 1))
    result = {'ce': float(ce_map.mean(dtype=np.float64)),
              'foreground_ce': float(ce_map[gt > 0].mean()) if (gt > 0).any() else float('nan'),
              'error_fraction': float(np.mean(pred != gt)),
              'foreground_fraction': float(np.mean(gt > 0))}
    soft, hard = [], []
    for label, name in REGIONS.items():
        target, mask = gt == label, pred == label
        value = (2 * prob[label][target].sum(dtype=np.float64) + SMOOTH) / (
            prob[label].sum(dtype=np.float64) + target.sum() + SMOOTH)
        denom = int(target.sum()) + int(mask.sum())
        dice = 2 * np.count_nonzero(target & mask) / denom if denom else float('nan')
        soft.append(value)
        hard.append(dice)
        result[f'soft_dice_{name}'] = float(value)
        result[f'hard_dice_{name}'] = float(dice)
        result[f'ce_{name}'] = float(ce_map[target].mean()) if target.any() else float('nan')
    result['soft_dice'] = float(np.mean(soft))
    result['soft_dice_loss'] = 1 - result['soft_dice']
    result['eval_loss'] = result['ce'] - result['soft_dice']
    result['hard_dice'] = float(np.nanmean(hard)) if np.isfinite(hard).any() else float('nan')
    return result, ce_map


def infer(ctx, minimum_free_gib=2):
    if platform.system() != 'Linux':
        raise RuntimeError('Inference is restricted to the Linux GPU server')
    import torch
    import SimpleITK as sitk
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; install a CUDA-compatible PyTorch in this kernel environment')
    out = ctx['out']
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    code_hash = hashlib.sha256()
    for file in sorted((repo / 'nnunetv2').rglob('*.py')):
        code_hash.update(str(file.relative_to(repo)).encode())
        code_hash.update(file.read_bytes())
    provenance = dict(
        fold=ctx['fold'], checkpoint=ctx['checkpoint'],
        checkpoint_sha256=sha256(ctx['model'] / f"fold_{ctx['fold']}" / ctx['checkpoint']),
        plans_sha256=sha256(ctx['model'] / 'plans.json'),
        dataset_sha256=sha256(ctx['model'] / 'dataset.json'),
        split_sha256=sha256(ctx['run'] / 'splits_final.json'),
        code_sha256=code_hash.hexdigest(), analysis_sha256=sha256(__file__),
        packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']},
        tile_step_size=0.5, mirroring=True, gaussian=True,
        inputs={case: {kind: sha256(ctx['raw'] / folder / f'{case}{suffix}.nii.gz')
                       for kind, folder, suffix in [('image', 'imagesTr', '_0000'), ('label', 'labelsTr', '')]}
                for _, case in ctx['cases']})
    manifest = out / 'inference-manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise ValueError('Inference inputs/code/environment changed. Use a new output directory; do not mix results.')
    if not manifest.exists() and any(out.iterdir()):
        raise ValueError('Output directory is nonempty but has no manifest')
    write_json(manifest, provenance)
    pending = []
    for subset, case in ctx['cases']:
        marker = out / subset / f'{case}.complete.json'
        if marker.exists():
            load_case(ctx, subset, case)  # Detect incomplete/corrupt probability exports before skipping.
        else:
            pending.append((subset, case))
    # Conservative uncompressed probability+mask estimate, plus scratch headroom. Not a guarantee.
    estimated = 0
    for _, case in pending:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(ctx['raw'] / 'labelsTr' / f'{case}.nii.gz'))
        reader.ReadImageInformation()
        import math
        estimated += math.prod(reader.GetSize()) * 17
    free = shutil.disk_usage(out).free
    print(f'Pending {len(pending)}/{len(ctx["cases"])}; estimate {estimated / 2**30:.2f} GiB + headroom; free {free / 2**30:.2f} GiB', flush=True)
    if free < estimated * 1.3 + minimum_free_gib * 2**30:
        raise RuntimeError('Insufficient free disk space for probability exports')
    if not pending:
        return
    predictor = nnUNetPredictor(device=torch.device('cuda', 0), tile_step_size=0.5,
                                use_gaussian=True, use_mirroring=True)
    predictor.initialize_from_trained_model_folder(str(ctx['model']), use_folds=(ctx['fold'],),
                                                  checkpoint_name=ctx['checkpoint'])
    for i, (subset, case) in enumerate(pending, 1):
        if shutil.disk_usage(out).free < minimum_free_gib * 2**30:
            raise RuntimeError('Disk headroom exhausted; completed cases can be resumed')
        folder = out / subset
        folder.mkdir(exist_ok=True)
        print(f'[{i}/{len(pending)}] {subset}: {case}', flush=True)
        predictor.predict_from_files_sequential(
            [[str(ctx['raw'] / 'imagesTr' / f'{case}_0000.nii.gz')]],
            [str(folder / case)], save_probabilities=True, overwrite=True)
        load_case(ctx, subset, case)
        write_json(folder / f'{case}.complete.json', {'case': case, 'subset': subset})


def analyse(ctx):
    import pandas as pd
    missing = [c for s, c in ctx['cases'] if not (ctx['out'] / s / f'{c}.complete.json').exists()]
    if missing:
        raise RuntimeError(f'{len(missing)} cases have no completed inference. Run inference first: {missing[:5]}')
    rows, slices = [], []
    for subset, case in ctx['cases']:
        gt, pred, prob, spacing, _ = load_case(ctx, subset, case)
        info = dict(subset=subset, case=case, patient=case.split('_')[0], frame=case.split('_')[1])
        row, _ = scores(gt, prob)
        rows.append(dict(**info, **row, slices=gt.shape[0], voxel_volume_mm3=float(__import__('math').prod(spacing))))
        for z in range(gt.shape[0]):
            row, _ = scores(gt[z], prob[:, z])
            slices.append(dict(**info, z=z, **row))
    cases, slices = pd.DataFrame(rows), pd.DataFrame(slices)
    report = ctx['out'] / 'reports'
    report.mkdir(exist_ok=True)
    cases.to_csv(report / 'case-losses.csv', index=False)
    slices.to_csv(report / 'slice-losses.csv', index=False)
    return cases, slices


def show_slice(ctx, subset, case, z, loaded=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    gt, pred, prob, spacing, image = loaded or load_case(ctx, subset, case, with_image=True)
    if not 0 <= z < gt.shape[0]:
        raise ValueError(f'z must be 0..{gt.shape[0]-1}')
    metrics, ce = scores(gt[z], prob[:, z])
    errors = np.zeros(gt[z].shape, dtype=np.uint8)
    errors[(gt[z] == 0) & (pred[z] > 0)] = 1
    errors[(gt[z] > 0) & (pred[z] == 0)] = 2
    errors[(gt[z] > 0) & (pred[z] > 0) & (gt[z] != pred[z])] = 3
    fig, axes = plt.subplots(1, 5, figsize=(19, 4), constrained_layout=True)
    lo, hi = np.percentile(image, [1, 99])
    aspect = spacing[1] / spacing[2]
    labels = ListedColormap(['black', '#46cf87', '#ffad42', '#48a9ff'])
    error_colors = ListedColormap(['black', '#ffd43b', '#f03e3e', '#cc5de8'])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4)
    for ax in axes:
        ax.imshow(image[z], cmap='gray', vmin=lo, vmax=hi, aspect=aspect)
        ax.axis('off')
    axes[1].imshow(np.ma.masked_equal(gt[z], 0), cmap=labels, norm=norm, alpha=.65, aspect=aspect)
    axes[2].imshow(np.ma.masked_equal(pred[z], 0), cmap=labels, norm=norm, alpha=.65, aspect=aspect)
    axes[3].imshow(np.ma.masked_equal(errors, 0), cmap=error_colors, norm=norm, alpha=.85, aspect=aspect)
    # Fixed volume-wide scale permits comparison between slices; clipping is only visual.
    p_true = np.take_along_axis(prob, gt.astype(np.int64)[None], axis=0)[0]
    vmax = max(float(np.percentile(-np.log(np.clip(p_true, EPS, 1)), 99.5)), .1)
    heat = axes[4].imshow(ce, cmap='magma', vmin=0, vmax=vmax, aspect=aspect)
    fig.colorbar(heat, ax=axes[4], shrink=.65, label='CE (display clipped at volume P99.5)')
    for ax, title in zip(axes, ['MRI', 'GT: green RV / orange MLV / blue LVC', 'Prediction',
                               'Yellow FP / red FN / purple wrong class', 'Voxel cross-entropy']):
        ax.set_title(title, fontsize=9)
    fig.suptitle(f'{case} | {subset} | z={z}/{gt.shape[0]-1} | slice CE={metrics["ce"]:.4f} '
                 f'| slice Dice={metrics["hard_dice"]:.3f} | eval loss={metrics["eval_loss"]:.4f}')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--storage', required=True)
    p.add_argument('--run-id', default='acdc-baseline-v1')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--checkpoint', default='checkpoint_final.safetensors')
    args = p.parse_args()
    infer(context(args.storage, args.run_id, args.fold, args.checkpoint))
