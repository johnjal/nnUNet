"""Notebook 07 queue. Importing only supports monitoring; execution is Linux/CUDA gated."""
import argparse
import copy
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'experiments/acdc/scripts'))
from workflow import guard, check_space, paths, model_dir, environment, freeze_run, write_json, digest, validate_splits
from run_folds import run as run_folds

TRAINERS = {
    'swin_v2': 'nnUNetTrainerACDCSwinV2NoDS250',
    'unet': 'nnUNetTrainerACDCUNetNoDS250',
}


def validate_suite(s):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', s['suite_id']):
        raise ValueError('Invalid suite ID')
    if not re.fullmatch(r'[0-9]+', s['gpu']):
        raise ValueError('Select one numeric GPU index')
    if s['models'] != list(TRAINERS) or s['fold'] != 0 or s['epochs'] != 250:
        raise ValueError('This experiment runs Swin V2 then U-Net, fold 0, 250 epochs')
    if type(s['batch_size']) is not int or s['batch_size'] not in (1, 2, 4):
        raise ValueError('Common batch_size must be 1, 2 or 4')
    if not Path(s['storage_root']).is_absolute() or s['augmentation_workers'] < 1:
        raise ValueError('Use absolute storage root and positive worker count')
    if type(s['evaluate_best']) is not bool:
        raise ValueError('evaluate_best must be Boolean')
    return s


def read_suite(path):
    return validate_suite(json.loads(Path(path).read_text()))


def suite_dir(s):
    return Path(s['storage_root']) / 'architecture_suites' / s['suite_id']


def model_config(s, name):
    c = json.loads((REPO / 'experiments/acdc/config.json').read_text())
    c.update(storage_root=s['storage_root'], source_dataset=str(Path(s['storage_root']) / 'metadata/acdc'),
             execution_enabled=s['execution_enabled'], expected_git_commit=s['expected_git_commit'],
             gpu=s['gpu'], augmentation_workers=s['augmentation_workers'], compile=False,
             run_id=f"{s['suite_id']}-{name}", trainer=TRAINERS[name], folds=[0], num_epochs=250,
             plans=f"nnUNetPlansNoDS250BS{s['batch_size']}", deep_supervision=False,
             reproduction_notes='Architecture comparison; scratch; no DS; fold 0; 250 epochs. '
             'Common nnU-Net optimizer/loss/full augmentation. Swin MONAI 1.5.1 feature24 v2 '
             'with symmetric input padding, cropped logits and gradient checkpointing. '
             'Not an exact recreation of the historical paper; no fixed training seed.')
    return c


def make_plans(base, batch):
    result = copy.deepcopy(base)
    result['plans_name'] = f'nnUNetPlansNoDS250BS{batch}'
    result['configurations']['3d_fullres']['batch_size'] = batch
    return result


def preflight(s):
    validate_suite(s)
    c = model_config(s, 'swin_v2')
    guard(c)
    check_space(c)
    for package, version in [('monai', '1.5.1'), ('einops', '0.8.1'), ('batchgeneratorsv2', '0.3.8'),
                             ('dynamic-network-architectures', '0.4.4')]:
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f'Install {package}=={version} in this kernel environment')
    p = paths(c)
    pre = p['pre'] / c['dataset_name']
    for f in [pre / 'preparation.complete.json', pre / 'nnUNetPlans.json',
              pre / 'dataset_fingerprint.json', pre / 'splits_final.json',
              p['raw'] / c['dataset_name'] / 'dataset.json',
              Path(c['source_dataset']) / 'download_manifest.json']:
        if not f.is_file():
            raise FileNotFoundError(f'Copy prepared acdc-data or finish notebook 01 first: {f}')
    base = json.loads((pre / 'nnUNetPlans.json').read_text())
    conf = base['configurations']['3d_fullres']
    if (conf['patch_size'] != [20, 256, 224] or conf['spacing'] != [5.0, 1.5625, 1.5625]
            or conf['preprocessor_name'] != 'DefaultPreprocessor' or conf['batch_dice']):
        raise ValueError('Prepared ACDC plans differ from the reviewed baseline')
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
    data = pre / conf['data_identifier']
    cases = infer_dataset_class(str(data)).get_identifiers(str(data))
    if len(cases) != 200:
        raise ValueError('Expected 200 prepared cases')
    splits = json.loads((pre / 'splits_final.json').read_text())
    validate_splits(splits, cases)
    if len(splits[0]['train']) != 160 or len(splits[0]['val']) != 40:
        raise ValueError('Expected fold 0 with 160 training and 40 validation cases')
    dataset = json.loads((p['raw'] / c['dataset_name'] / 'dataset.json').read_text())
    if dataset['labels'] != {'background': 0, 'RV': 1, 'MLV': 2, 'LVC': 3} or len(dataset['channel_names']) != 1:
        raise ValueError('Expected single-channel ACDC and labels background/RV/MLV/LVC=0/1/2/3')
    for case in splits[0]['val']:
        for directory, suffix in [('imagesTr', '_0000'), ('labelsTr', '')]:
            if not (p['raw'] / c['dataset_name'] / directory / f'{case}{suffix}.nii.gz').is_file():
                raise FileNotFoundError(f'Missing raw validation input: {case}, {directory}')
    for name in s['models']:
        recursive_find_trainer_class_by_name(TRAINERS[name])
    plans = make_plans(base, s['batch_size'])
    target = pre / f"{c['plans']}.json"
    if target.exists() and json.loads(target.read_text()) != plans:
        raise ValueError(f'Existing comparison plans differ: {target}')
    if not target.exists():
        write_json(target, plans)
    return plans


def freeze_suite(s):
    folder = suite_dir(s)
    sources = [Path(__file__), Path(__file__).with_name('requirements.txt'),
               REPO / 'experiments/acdc/analysis/compare_best.py']
    snapshot = {'suite': s, 'driver_hashes': {str(p.relative_to(REPO)): digest(p) for p in sources}}
    target = folder / 'suite.snapshot.json'
    if target.exists() and json.loads(target.read_text()) != snapshot:
        raise ValueError('Suite configuration/code changed; choose a new SUITE_ID')
    write_json(target, snapshot)
    for name in s['models']:
        freeze_run(model_config(s, name))


def gpu_lock(s):
    import fcntl
    handle = (Path(s['storage_root']) / f"gpu-{s['gpu']}.lock").open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('Another experiment is using this GPU') from None
    return handle


def smoke(s):
    """Full planned batch, synthetic data, one forward/backward/update per network on server."""
    plans = preflight(s)
    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    pm = PlansManager(plans)
    cm = pm.get_configuration('3d_fullres')
    reports = []
    with gpu_lock(s):
        for name in s['models']:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            net = recursive_find_trainer_class_by_name(TRAINERS[name]).build_network_architecture(
                pm, cm, 1, 4, False).cuda().train()
            optimizer = torch.optim.SGD(net.parameters(), lr=.01, momentum=.99, nesterov=True, weight_decay=3e-5)
            loss_fn = DC_and_CE_loss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': False}, {},
                                    weight_ce=1, weight_dice=1, dice_class=MemoryEfficientSoftDiceLoss)
            x = torch.randn(s['batch_size'], 1, *cm.patch_size, device='cuda')
            target = torch.randint(0, 4, (s['batch_size'], 1, *cm.patch_size), device='cuda')
            scaler = torch.amp.GradScaler('cuda')
            with torch.autocast('cuda'):
                output = net(x)
                if tuple(output.shape) != (s['batch_size'], 4, *cm.patch_size):
                    raise ValueError('Incorrect logits shape')
                loss = loss_fn(output, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f'{name}: nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
            if not torch.isfinite(norm):
                raise RuntimeError(f'{name}: nonfinite gradient')
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            reports.append(dict(model=name, batch=s['batch_size'], patch=list(cm.patch_size),
                                loss=float(loss.detach()), seconds=time.monotonic() - start,
                                parameters=sum(p.numel() for p in net.parameters()),
                                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30))
            print(reports[-1], flush=True)
            del net, optimizer, x, target, output, loss, scaler, loss_fn
        torch.cuda.empty_cache()
    write_json(suite_dir(s) / 'smoke.json', reports)


def evaluate_best(s, name):
    """Keep training's final/validation folder intact; save best full-volume masks separately."""
    c = model_config(s, name)
    os.environ.update(environment(c))
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.checkpoints import find_checkpoint_file, load_checkpoint_file
    from evaluate_regions import read_masks
    sys.path.insert(0, str(REPO / 'experiments/acdc/analysis'))
    from compare_best import mask_metrics, aggregate, write_csv
    p = paths(c)
    fold = model_dir(c) / 'fold_0'
    checkpoint_file = find_checkpoint_file(str(fold / 'checkpoint_best.safetensors'))
    if checkpoint_file is None:
        raise FileNotFoundError(f'Best checkpoint missing: {fold}')
    checkpoint = Path(checkpoint_file)
    cases = json.loads((p['run'] / 'splits_final.json').read_text())[0]['val']
    dest = suite_dir(s) / 'best' / name
    dest.mkdir(parents=True, exist_ok=True)
    raw = p['raw'] / c['dataset_name']
    provenance = {'checkpoint_sha256': digest(checkpoint),
                  'inputs': {case: {kind: digest(raw / directory / f'{case}{suffix}.nii.gz')
                             for kind, directory, suffix in [('image', 'imagesTr', '_0000'), ('label', 'labelsTr', '')]}
                             for case in cases}, 'protocol': 'step0.5_gaussian_mirror012_no_postprocessing'}
    with gpu_lock(s):
        manifest = dest / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != provenance:
            raise ValueError('Best checkpoint or validation inputs changed; use a new suite ID')
        if not manifest.exists() and any(dest.iterdir()):
            raise ValueError('Untracked best outputs without manifest; choose a new suite ID')
        write_json(manifest, provenance)
        pending = []
        for case in cases:
            try:
                if not (dest / f'{case}.nii.gz').exists():
                    raise FileNotFoundError(case)
                gt, pred, _ = read_masks(raw / 'labelsTr' / f'{case}.nii.gz', dest / f'{case}.nii.gz')
                mask_metrics(gt, pred)
            except (FileNotFoundError, RuntimeError, ValueError):
                pending.append(case)
        if pending:
            predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=True,
                                        device=torch.device('cuda', 0))
            predictor.initialize_from_trained_model_folder(str(model_dir(c)), use_folds=(0,), checkpoint_name=checkpoint.name)
            if tuple(predictor.allowed_mirroring_axes or ()) != (0, 1, 2):
                raise ValueError('Unexpected inference mirroring axes')
            predictor.predict_from_files_sequential(
                [[str(raw / 'imagesTr' / f'{case}_0000.nii.gz')] for case in pending],
                [str(dest / case) for case in pending], save_probabilities=False, overwrite=True)
            del predictor
            torch.cuda.empty_cache()
        rows = []
        for case in cases:
            gt, pred, _ = read_masks(raw / 'labelsTr' / f'{case}.nii.gz', dest / f'{case}.nii.gz')
            rows.extend(dict(case=case, **metric) for metric in mask_metrics(gt, pred))
        meta = load_checkpoint_file(str(checkpoint), keys=('current_epoch',))
        write_csv(dest / 'case-metrics.csv', rows)
        write_json(dest / 'summary.json', dict(checkpoint_completed_epochs=meta.get('current_epoch'),
                                              validation_cases=len(cases), **aggregate(rows)))


def comparison_rows(s):
    rows = []
    for name in s['models']:
        c = model_config(s, name)
        for checkpoint in (['final', 'best'] if s['evaluate_best'] else ['final']):
            row = dict(model=name, run_id=c['run_id'], checkpoint=checkpoint, fold=0,
                       epochs=250, batch_size=s['batch_size'], deep_supervision=False, status='pending')
            if checkpoint == 'final':
                file = model_dir(c) / 'fold_0/validation/summary.json'
                if file.exists() and any((model_dir(c) / 'fold_0').glob('checkpoint_final.*')):
                    result = json.loads(file.read_text())
                    row.update(status='completed', dice_percent=100 * result['foreground_mean']['Dice'],
                               miou_percent=100 * result['foreground_mean']['IoU'])
                    for label, region in [('1', 'RV'), ('2', 'MLV'), ('3', 'LVC')]:
                        row[f'dice_{region}_percent'] = 100 * result['mean'][label]['Dice']
            else:
                file = suite_dir(s) / 'best' / name / 'summary.json'
                if file.exists():
                    row.update(json.loads(file.read_text()), status='completed')
            rows.append(row)
    return rows


def write_comparison(s):
    rows = comparison_rows(s)
    write_json(suite_dir(s) / 'comparison.json', rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path = suite_dir(s) / 'comparison.csv'
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def run_suite(s, resume=False):
    import fcntl
    preflight(s)
    folder = suite_dir(s)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'suite.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This suite already has an active runner') from None
        freeze_suite(s)
        state = dict(status='running', active_model=None, models={}, started=time.time(), pid=os.getpid())
        write_json(folder / 'status.json', state)
        try:
            # A separate process releases all synthetic-test CUDA allocations before real training.
            subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--config',
                            str(folder / 'suite.server.json'), '--smoke-only'], check=True,
                           env=environment(model_config(s, 'swin_v2')))
            for name in s['models']:
                state.update(active_model=name)
                state['models'][name] = 'training'
                write_json(folder / 'status.json', state)
                print(f'=== {name}: no deep supervision, fold 0, 250 epochs ===', flush=True)
                run_folds(model_config(s, name), resume=resume)
                write_comparison(s)
                if s['evaluate_best']:
                    state['models'][name] = 'evaluating_best'
                    write_json(folder / 'status.json', state)
                    evaluate_best(s, name)
                    write_comparison(s)
                state['models'][name] = 'completed'
                write_json(folder / 'status.json', state)
            state.update(status='completed', active_model=None)
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
            raise
        finally:
            state['ended'] = time.time()
            write_json(folder / 'status.json', state)


def interrupt(signum, frame):
    raise KeyboardInterrupt(f'Signal {signum}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    suite = read_suite(args.config)
    if args.smoke_only:
        smoke(suite)
    else:
        run_suite(suite, args.resume)
