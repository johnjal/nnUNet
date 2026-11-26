"""Notebook 08 independent jobs: no augmentation, encoder pretraining comparison. Importing only supports monitoring; execution is Linux/CUDA gated."""
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
    'scratch': 'nnUNetTrainerACDCSwinScratchNoAug250',
    'voco': 'nnUNetTrainerACDCSwinVoCoNoAug250',
}


def validate_suite(s):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', s['suite_id']):
        raise ValueError('Invalid suite ID')
    if not re.fullmatch(r'[0-9]+', s['gpu']):
        raise ValueError('Select one numeric GPU index')
    if s['models'] != list(TRAINERS) or s['fold'] != 0 or s['epochs'] != 250:
        raise ValueError('Expected scratch/VoCo comparison, fold 0, 250 epochs')
    if type(s['batch_size']) is not int or s['batch_size'] != 4:
        raise ValueError('Keep batch 4 to match the existing baseline; do not silently reduce it')
    if not Path(s['storage_root']).is_absolute() or s['augmentation_workers'] < 1:
        raise ValueError('Use absolute storage root and positive worker count')
    if s['evaluate_best'] is not True:
        raise ValueError('Both final and best evaluations are required')
    if type(s['seed']) is not int or not 0 <= s['seed'] < 2**32:
        raise ValueError('Seed must be a uint32 integer')
    return s


def read_suite(path):
    return validate_suite(json.loads(Path(path).read_text()))


def suite_dir(s):
    return Path(s['storage_root']) / 'pretraining_suites' / s['suite_id']


def model_config(s, name):
    c = json.loads((REPO / 'experiments/acdc/config.json').read_text())
    c.update(storage_root=s['storage_root'], source_dataset=str(Path(s['storage_root']) / 'metadata/acdc'),
             execution_enabled=s['execution_enabled'], expected_git_commit=s['expected_git_commit'],
             gpu=s['gpu'], augmentation_workers=s['augmentation_workers'], compile=False,
             run_id=f"{s['suite_id']}-{name}", trainer=TRAINERS[name], folds=[0], num_epochs=250,
             plans=f"nnUNetPlansNoAugVoCo250BS{s['batch_size']}", deep_supervision=False,
             seed=s['seed'], augmentation_theme='none', use_mirroring=False,
             voco_sha256=s['voco_sha256'] if name == 'voco' else None,
             reproduction_notes='ACDC no augmentation; DS OFF; AMP; fold0;250 epochs; '
             'same nnU-Net SGD/Dice+CE; Swin v2 feature48; complete VoCo encoder transfer only; '
             'decoder random; fixed seed (not bitwise deterministic); TTA OFF; historical U-Net DS ON.')
    return c


def make_plans(base, batch):
    result = copy.deepcopy(base)
    result['plans_name'] = f'nnUNetPlansNoAugVoCo250BS{batch}'
    result['configurations']['3d_fullres']['batch_size'] = batch
    return result


def preflight(s, task):
    validate_suite(s)
    c = model_config(s, 'scratch')
    os.environ['ACDC_SEED'] = str(s['seed'])
    if task == 'voco':
        os.environ['ACDC_VOCO_CHECKPOINT'] = s['voco_checkpoint']
        if digest(Path(s['voco_checkpoint'])) != s['voco_sha256']:
            raise ValueError('VoCo checkpoint hash changed')
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
    if task in TRAINERS:
        recursive_find_trainer_class_by_name(TRAINERS[task])
    plans = make_plans(base, s['batch_size'])
    target = pre / f"{c['plans']}.json"
    if target.exists() and json.loads(target.read_text()) != plans:
        raise ValueError(f'Existing comparison plans differ: {target}')
    if not target.exists():
        write_json(target, plans)
    if task == 'unet_existing':
        baseline_record(s)
    return plans


def task_dir(s, task):
    if task not in ['unet_existing', *TRAINERS]:
        raise ValueError('Unknown task')
    return suite_dir(s) / 'jobs' / task


def freeze_task(s, task):
    folder = task_dir(s, task)
    sources = [Path(__file__), Path(__file__).with_name('requirements.txt'),
               REPO / 'experiments/acdc/analysis/compare_best.py']
    snapshot = {'task': task, 'suite': s,
                'driver_hashes': {str(p.relative_to(REPO)): digest(p) for p in sources}}
    if task == 'unet_existing':
        snapshot['baseline'] = baseline_record(s)['provenance']
    target = folder / 'snapshot.json'
    if target.exists() and json.loads(target.read_text()) != snapshot:
        raise ValueError('Task configuration/code changed; choose a new SUITE_ID')
    write_json(target, snapshot)
    if task in TRAINERS:
        freeze_run(model_config(s, task))


def gpu_lock(s):
    import fcntl
    handle = (Path(s['storage_root']) / f"gpu-{s['gpu']}.lock").open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('Another experiment is using this GPU') from None
    return handle


def smoke(s, task):
    """Full planned batch, synthetic data, one forward/backward/update per network on server."""
    plans = preflight(s, task)
    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    pm = PlansManager(plans)
    cm = pm.get_configuration('3d_fullres')
    reports = []
    with gpu_lock(s):
        for name in [task]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            net = recursive_find_trainer_class_by_name(TRAINERS[name]).build_network_architecture(
                pm, cm, 1, 4, False).cuda().train()
            if name == 'voco':
                from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerACDCVoCo250 import load_voco
                report = load_voco(net, s['voco_checkpoint'])
                write_json(task_dir(s, task) / 'voco-load-report.json', report)
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
    write_json(task_dir(s, task) / 'smoke.json', reports)


def baseline_record(s):
    """Validate historical training provenance without rewriting its frozen snapshot."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', s['baseline_run_id']):
        raise ValueError('Invalid baseline run ID')
    run = Path(s['storage_root']) / 'runs' / s['baseline_run_id']
    snapshot = json.loads((run / 'config.snapshot.json').read_text())
    c = snapshot['configuration']
    if (c.get('trainer') != 'nnUNetTrainerACDCNoAug250' or c.get('augmentation_theme') != 'none'
            or c.get('num_epochs') != 250 or c.get('dataset_id') != 27):
        raise ValueError('Baseline must be ACDC none, 250 epochs, original DS ON trainer')
    current = paths(model_config(s, 'scratch'))['pre'] / 'Dataset027_ACDC'
    splits = json.loads((run / 'splits_final.json').read_text())
    if splits != json.loads((current / 'splits_final.json').read_text()):
        raise ValueError('Historical and current patient splits differ')
    historical_plans = json.loads((run/'nnUNetPlans.json').read_text())
    if historical_plans['configurations']['3d_fullres']['batch_size'] != 4:
        raise ValueError('Historical baseline must use batch 4')
    for file in ('nnUNetPlans.json', 'dataset_fingerprint.json'):
        if digest(run / file) != digest(current / file):
            raise ValueError(f'Historical preprocessing differs: {file}')
    raw_json = Path(s['storage_root']) / 'nnUNet_raw/Dataset027_ACDC/dataset.json'
    if digest(raw_json) != digest(run / 'dataset.json'):
        raise ValueError('Historical dataset labels differ')
    model = run / 'nnUNet_results/Dataset027_ACDC' / f"{c['trainer']}__{c['plans']}__3d_fullres"
    from nnunetv2.utilities.checkpoints import find_checkpoint_file
    checkpoints = {}
    for kind in ('final', 'best'):
        cp = find_checkpoint_file(str(model / 'fold_0' / f'checkpoint_{kind}.safetensors'))
        if not cp:
            raise FileNotFoundError(f'Baseline {kind} checkpoint missing: {model}')
        checkpoints[kind] = Path(cp)
    return dict(model=model, cases=splits[0]['val'], checkpoints=checkpoints,
                provenance={'snapshot_sha256': digest(run / 'config.snapshot.json'),
                            'checkpoints': {k: digest(v) for k,v in checkpoints.items()}})


def evaluate(s, name, kind):
    """All three models use identical raw-volume inference and metric aggregation, TTA OFF."""
    c = model_config(s, 'scratch' if name == 'unet_existing' else name)
    os.environ.update(environment(c))
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.checkpoints import find_checkpoint_file, load_checkpoint_file
    from evaluate_regions import read_masks
    sys.path.insert(0, str(REPO / 'experiments/acdc/analysis'))
    from compare_best import mask_metrics, aggregate, write_csv
    model = baseline_record(s)['model'] if name == 'unet_existing' else model_dir(c)
    cp = find_checkpoint_file(str(model / 'fold_0' / f'checkpoint_{kind}.safetensors'))
    if cp is None:
        raise FileNotFoundError(f'{name}/{kind}: checkpoint missing')
    cp = Path(cp)
    split_file = paths(c)['pre'] / c['dataset_name'] / 'splits_final.json'
    cases = json.loads(split_file.read_text())[0]['val']
    raw = paths(c)['raw'] / c['dataset_name']
    dest = suite_dir(s) / 'evaluation_no_tta' / name / kind
    dest.mkdir(parents=True, exist_ok=True)
    provenance = {'checkpoint_sha256': digest(cp), 'use_mirroring': False,
                  'tile_step_size': .5, 'gaussian': True, 'postprocessing': False,
                  'inputs': {case: {d: digest(raw / d / f'{case}{suffix}.nii.gz')
                           for d,suffix in [('imagesTr','_0000'),('labelsTr','')]} for case in cases}}
    with gpu_lock(s):
        manifest = dest / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != provenance:
            raise ValueError('Evaluation inputs/checkpoint changed; use a new suite ID')
        if not manifest.exists() and any(dest.iterdir()):
            raise ValueError('Untracked predictions in evaluation directory')
        write_json(manifest, provenance)
        # Resumable: validate each mask before reusing it; never read old TTA directories.
        pending = []
        for case in cases:
            try:
                gt, pred, _ = read_masks(raw/'labelsTr'/f'{case}.nii.gz', dest/f'{case}.nii.gz')
                mask_metrics(gt, pred)
            except (FileNotFoundError, RuntimeError, ValueError):
                pending.append(case)
        if pending:
            predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=False,
                                       perform_everything_on_device=False, device=torch.device('cuda', 0))
            predictor.initialize_from_trained_model_folder(str(model), use_folds=(0,), checkpoint_name=cp.name)
            predictor.predict_from_files_sequential(
                [[str(raw/'imagesTr'/f'{case}_0000.nii.gz')] for case in pending],
                [str(dest/case) for case in pending], save_probabilities=False, overwrite=True)
            del predictor
            torch.cuda.empty_cache()
        rows = []
        for case in cases:
            gt,pred,_ = read_masks(raw/'labelsTr'/f'{case}.nii.gz', dest/f'{case}.nii.gz')
            rows.extend(dict(case=case, **metric) for metric in mask_metrics(gt,pred))
        meta = load_checkpoint_file(str(cp), keys=('current_epoch',))
        write_csv(dest/'case-metrics.csv', rows)
        write_json(dest/'summary.json', dict(model=name, checkpoint=kind, deep_supervision=name=='unet_existing',
            augmentation=False, tta=False, validation_cases=len(cases),
            comparison_protocol=comparison_protocol(s),
            checkpoint_completed_epochs=meta.get('current_epoch'), **aggregate(rows)))


def comparison_protocol(s):
    """Path-independent fingerprint for safely merging results from different machines."""
    c = model_config(s, 'scratch')
    p = paths(c)
    pre = p['pre'] / c['dataset_name']
    return dict(dataset='ACDC', fold=0, epochs=250, batch_size=4, augmentation=False,
                tta=False, patch=[20,256,224], seed=s['seed'],
                input_hashes={f: digest(pre/f) for f in
                              ['splits_final.json','nnUNetPlans.json','dataset_fingerprint.json']},
                dataset_hash=digest(p['raw']/c['dataset_name']/'dataset.json'),
                metric_code_hash=digest(REPO/'experiments/acdc/analysis/compare_best.py'))


def write_comparison(s, roots=None):
    """Merge small evaluation folders copied from independent servers; no GPU needed."""
    sys.path.insert(0, str(REPO / 'experiments/acdc/analysis'))
    from compare_best import write_csv
    roots = [Path(p) for p in roots] if roots is not None else [suite_dir(s)]
    records = {}
    reference = None
    for root in roots:
        for name in ['unet_existing', *s['models']]:
            for kind in ('final','best'):
                path=root/'evaluation_no_tta'/name/kind/'summary.json'
                if not path.exists():
                    continue
                row=json.loads(path.read_text())
                protocol=row.get('comparison_protocol')
                if protocol is None:
                    raise ValueError(f'Missing cross-instance protocol: {path}')
                if reference is None:
                    reference=protocol
                elif reference!=protocol:
                    raise ValueError(f'Experiment conditions differ: {path}')
                key=(name,kind)
                if key in records and records[key]!=row:
                    raise ValueError(f'Conflicting duplicate results: {name}/{kind}')
                records[key]=row
    rows=list(records.values())
    if rows:
        write_json(suite_dir(s)/'comparison.json',rows)
        write_csv(suite_dir(s)/'comparison.csv',rows)
    return rows


def run_task(s, task, resume=False):
    import fcntl
    folder=task_dir(s, task)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'task.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state=dict(status='running', stage='preflight', model=task, started=time.time())
        write_json(folder/'status.json',state)
        try:
            preflight(s, task)
            freeze_task(s, task)
            if task in TRAINERS:
                state.update(stage='smoke')
                write_json(folder/'status.json',state)
                subprocess.run([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--config',
                    str(folder/'config.json'),'--task',task,'--smoke-only'], check=True,
                    env=environment(model_config(s,task)))
                state.update(stage='training')
                write_json(folder/'status.json',state)
                run_folds(model_config(s,task), resume=resume)
            state.update(stage='evaluation')
            write_json(folder/'status.json',state)
            for kind in ('final','best'):
                evaluate(s,task,kind)
            # Only the notebook merges reports, avoiding cross-job report write races.
            state.update(status='completed',stage='complete')
        except BaseException as error:
            state.update(status='failed',error=str(error))
            raise
        finally:
            state['ended']=time.time()
            write_json(folder/'status.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--task', required=True, choices=['unet_existing', *TRAINERS])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    def interrupt(signum,frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupt)
    signal.signal(signal.SIGINT,interrupt)
    suite=read_suite(args.config)
    if args.smoke_only:
        if args.task not in TRAINERS:
            raise ValueError("Existing U-Net needs evaluation only, no smoke training")
        smoke(suite, args.task)
    else:
        run_task(suite,args.task,args.resume)
