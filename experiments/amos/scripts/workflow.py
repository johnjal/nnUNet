"""Server-only AMOS MRI preparation and sequential augmentation comparison."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import random
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
DATASET = 'Dataset620_AMOSMRI'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def validate(c):
    if c.get('split_mode') not in (None, 'train50_val10'):
        raise ValueError('Unknown split mode')
    if c.get('split_mode') == 'train50_val10' and type(c.get('split_seed')) is not int:
        raise ValueError('split_seed must be an integer')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', c['run_id']):
        raise ValueError('Invalid run_id')
    if not re.fullmatch(r'\d+', c['gpu']):
        raise ValueError('Select one GPU index')
    if type(c['evaluate_best']) is not bool or type(c['seed']) is not int:
        raise ValueError('evaluate_best must be Boolean and seed must be an integer')
    if type(c['amp']) is not bool or c['batch_size'] not in (1, 2, 4):
        raise ValueError('amp must be Boolean; batch_size must be 1, 2 or 4')
    if not 1 <= c['workers'] <= 8 or not 2 <= c['planner_gb'] <= 16:
        raise ValueError('Invalid worker count or planning memory target')
    root = Path(c['storage_root'])
    if not root.is_absolute() or root == REPO or REPO in root.parents:
        raise ValueError('Use absolute storage_root outside the repo')
    return c


def themes(c):
    return ('off',) if c.get('split_mode') == 'train50_val10' else ('off', 'on')


def locations(c):
    root = Path(c['storage_root'])
    plans = f"nnUNetPlansAMOSMRI{c['planner_gb']:g}GBBS{c['batch_size']}"
    suite = root / 'experiments' / c['run_id']
    pre_root = root/'nnUNet_preprocessed'
    if c.get('split_mode') == 'train50_val10':
        pre_root = root/f"nnUNet_preprocessed_train50val10_seed{c['split_seed']}"
    return dict(root=root, source=root/'amos22', raw=root/'nnUNet_raw'/DATASET,
                pre=pre_root/DATASET, suite=suite,
                results=suite/'nnUNet_results', plans=plans)


def environment(c):
    p = locations(c)
    return dict(os.environ, nnUNet_raw=str(p['raw'].parent), nnUNet_preprocessed=str(p['pre'].parent),
                nnUNet_results=str(p['results']), CUDA_VISIBLE_DEVICES=c['gpu'],
                nnUNet_compile='false', nnUNet_n_proc_DA=str(c['workers']),
                nnUNet_def_n_proc=str(c['workers']), AMOS_SEED=str(c['seed']),
                PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(p['root']/'cache/matplotlib'),
                XDG_CACHE_HOME=str(p['root']/'cache'), WANDB_MODE='disabled',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                PYTHONPATH=str(REPO) + os.pathsep + os.environ.get('PYTHONPATH', ''),
                PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''))


def trainer(c, theme):
    if theme not in ('off', 'on'):
        raise ValueError(theme)
    return 'nnUNetTrainerAMOS' + ('Aug' if theme == 'on' else 'NoAug') + ('AMP' if c['amp'] else 'FP32')


def model_dir(c, theme):
    p = locations(c)
    return p['results']/DATASET/f"{trainer(c, theme)}__{p['plans']}__3d_fullres"


def call(c, command):
    print('Running:', ' '.join(map(str,command)), flush=True)
    proc = subprocess.Popen(list(map(str,command)), env=environment(c))
    try:
        code = proc.wait()
        if code:
            raise RuntimeError(f'Command failed (exit {code}): {command[0]}. See the preceding error in runner.log; this does not necessarily indicate CUDA OOM.')
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise


def cli(c, name, *args):
    executable = Path(sys.executable).parent/name
    if not executable.is_file():
        raise FileNotFoundError(f'Install this checkout in the kernel venv: {executable}')
    call(c, [executable, *args])


def free_space(c, gib=5):
    root = locations(c)['root']
    root.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(root).free / 2**30
    print(f'Free disk: {available:.1f} GiB', flush=True)
    if available < gib:
        raise RuntimeError(f'Need at least {gib} GiB free; this is a guard, not a preprocessing size estimate')


def frozen(path, value):
    if Path(path).exists() and read(path) != value:
        raise ValueError(f'Existing configuration/source differs: {path}. Use a new run_id or separate storage; never overwrite provenance.')
    write(path, value)


def verify_source(c):
    p = locations(c)
    manifest = read(p['root']/'download_manifest.json')
    expected = {'imagesTr':40, 'labelsTr':40, 'imagesVa':20, 'labelsVa':20}
    if manifest['counts'] != expected:
        raise ValueError('Expected original MRI 40/20 labeled split')
    for item in manifest['files']:
        path = (p['root']/item['path']).resolve()
        if p['root'].resolve() not in path.parents or sha(path) != item['sha256']:
            raise ValueError(f'Source hash mismatch: {path}')
    for folder, count in expected.items():
        files = list((p['source']/folder).glob('*.nii.gz'))
        if len(files) != count or any(int(f.name[5:9]) < 500 for f in files):
            raise ValueError('Source must contain MRI subset only')
    return manifest


def link_original(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha(src) != sha(dst):
            raise ValueError(f'Existing converted file differs: {dst}')
        return
    # Both are inside storage_root. Hard links avoid duplicating NIfTI data.
    os.link(src, dst)


def prepare(c):
    p = locations(c)
    if not (p['root']/'download_manifest.json').exists():
        call(c, [sys.executable, '-B', Path(__file__).with_name('download_mri.py'),
                 '--output', p['root'], '--download', '--workers', '4'])
    verify_source(c)
    spec = dict(source_sha256=sha(p['root']/'download_manifest.json'),
                planner_gb=c['planner_gb'], batch_size=c['batch_size'], plans=p['plans'],
                split='official_train40_validation20', reader='NibabelIOWithReorient')
    frozen(p['root']/f"prepare-{p['plans']}.json", spec)
    splits = {'train':[], 'val':[]}
    for original, split in [('Tr','train'),('Va','val')]:
        for image in sorted((p['source']/f'images{original}').glob('*.nii.gz')):
            case = image.name[:-7]
            label = p['source']/f'labels{original}'/image.name
            link_original(image, p['raw']/'imagesTr'/f'{case}_0000.nii.gz')
            link_original(label, p['raw']/'labelsTr'/image.name)
            splits[split].append(case)
    original = read(p['source']/'dataset.json')
    dataset = dict(channel_names={'0':'MRI'}, labels={v:int(k) for k,v in original['labels'].items()},
                   numTraining=60, file_ending='.nii.gz', overwrite_image_reader_writer='NibabelIOWithReorient')
    frozen(p['raw']/'dataset.json', dataset)
    if c.get('split_mode') == 'train50_val10':
        # Keep the original 40 training cases; move a fixed random 10 of 20 validation cases.
        pool = sorted(splits['val'])
        random.Random(c['split_seed']).shuffle(pool)
        splits = dict(train=sorted(splits['train'] + pool[:10]), val=sorted(pool[10:]))
        # Separate split directory: never change the old experiment's split.
        # Completed preprocessing is independent of the training split; reuse via hard links.
        old = p['root']/'nnUNet_preprocessed'/DATASET
        old_marker = old/f"{p['plans']}.complete.json"
        if old_marker.exists() and not (p['pre']/f"{p['plans']}.complete.json").exists():
            saved = read(old_marker)
            if saved['spec'] != spec or saved['plans_sha256'] != sha(old/f"{p['plans']}.json"):
                raise ValueError('Original preprocessing provenance differs; cannot reuse')
            for source in sorted(old.rglob('*')):
                if source.is_file() and source.name != 'splits_final.json' and not source.name.endswith('.complete.json'):
                    link_original(source, p['pre']/source.relative_to(old))
            write(p['pre']/old_marker.name, saved)
    frozen(p['pre']/'splits_final.json', [splits])
    marker = p['pre']/f"{p['plans']}.complete.json"
    if marker.exists():
        saved = read(marker)
        if saved['spec'] != spec or saved['plans_sha256'] != sha(p['pre']/f"{p['plans']}.json"):
            raise ValueError('Prepared plan changed')
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        plan = read(p['pre']/f"{p['plans']}.json")
        folder = p['pre']/plan['configurations']['3d_fullres']['data_identifier']
        if len(infer_dataset_class(str(folder)).get_identifiers(str(folder))) != 60:
            raise ValueError('Prepared cases missing')
        print('Reusing verified prepared data', flush=True)
        return
    free_space(c, 10)
    cli(c, 'nnUNetv2_plan_and_preprocess', '-d','620','--verify_dataset_integrity','--no_pp',
        '-npfp',c['workers'],'-gpu_memory_target',c['planner_gb'],'-overwrite_plans_name',p['plans'])
    planfile = p['pre']/f"{p['plans']}.json"
    plan = read(planfile)
    plan['configurations']['3d_fullres']['batch_size'] = c['batch_size']
    write(planfile, plan)
    cli(c, 'nnUNetv2_preprocess', '-d','620','-plans_name',p['plans'],'-c','3d_fullres','-np',c['workers'])
    write(marker, dict(spec=spec, plans_sha256=sha(planfile)))
    print('Prepared patch/spacing/batch:', plan['configurations']['3d_fullres']['patch_size'],
          plan['configurations']['3d_fullres']['spacing'], c['batch_size'], flush=True)
    free_space(c)


def recipe(c):
    p = locations(c)
    files = sorted((REPO/'nnunetv2').rglob('*.py')) + sorted(Path(__file__).parent.glob('*.py')) + [REPO/'pyproject.toml']
    versions = subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    frozen(p['suite']/'snapshot.json', dict(configuration=c, code_sha256=hashlib.sha256(
        ''.join(str(f.relative_to(REPO))+sha(f) for f in files).encode()).hexdigest(),
        packages=versions, plans=sha(p['pre']/f"{p['plans']}.json"),
        splits=sha(p['pre']/'splits_final.json'), source=sha(p['root']/'download_manifest.json')))


def smoke(c, theme):
    import torch
    from nnunetv2.run.run_training import get_trainer_from_args
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    t = get_trainer_from_args('620','3d_fullres',0,trainer(c,theme),locations(c)['plans'],False,device=torch.device('cuda'))
    print('Smoke:', theme, '| AMP:', c['amp'], '| DS: ON', flush=True)
    t.on_train_start()
    t.network.train()
    try:
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            result = t.train_step(next(t.dataloader_train))
            if not __import__('numpy').isfinite(result['loss']):
                raise RuntimeError('Non-finite training loss')
        t.network.eval()
        with torch.no_grad():
            result = t.validation_step(next(t.dataloader_val))
        if not __import__('numpy').isfinite(result['loss']):
            raise RuntimeError('Non-finite validation loss')
        write(locations(c)['suite']/f'smoke-{theme}.json', dict(amp=c['amp'],
              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, training_steps=3,
              note='Real augmented patches and backward pass; fresh training process starts from scratch'))
    finally:
        for loader in (t.dataloader_train,t.dataloader_val):
            if hasattr(loader,'_finish'):
                loader._finish()


def best(c, theme):
    import torch
    from nnunetv2.utilities.checkpoints import find_checkpoint_file
    from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerAMOS250 import AMOSPredictor, AMOSAMPPredictor
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2
    p = locations(c)
    output = model_dir(c,theme)/'fold_0'/'validation_best'
    if (output/'summary.json').exists():
        return
    checkpoint = find_checkpoint_file(str(model_dir(c,theme)/'fold_0'/'checkpoint_best.safetensors'))
    if not checkpoint:
        raise FileNotFoundError('Best checkpoint missing')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pred = (AMOSAMPPredictor if c['amp'] else AMOSPredictor)(device=torch.device('cuda'),perform_everything_on_device=False)
    pred.initialize_from_trained_model_folder(str(model_dir(c,theme)), use_folds=(0,), checkpoint_name=Path(checkpoint).name)
    ids = read(p['pre']/'splits_final.json')[0]['val']
    output.mkdir(parents=True,exist_ok=True)
    pred.predict_from_files_sequential([[str(p['raw']/'imagesTr'/f'{i}_0000.nii.gz')] for i in ids],
                                      [str(output/i) for i in ids],save_probabilities=False,overwrite=True)
    compute_metrics_on_folder2(str(p['raw']/'labelsTr'),str(output),str(p['raw']/'dataset.json'),
                               str(p['pre']/f"{p['plans']}.json"),num_processes=c['workers'],chill=True)


def repair_foreground_mean(summary):
    """Macro average over evaluable organs, preserving missing-organ NaNs and real zeros."""
    organs = [metrics for label, metrics in summary['mean'].items() if str(label) != '0']
    counts = {}
    for metric in summary['foreground_mean']:
        values = [organ[metric] for organ in organs if not math.isnan(organ[metric])]
        summary['foreground_mean'][metric] = sum(values)/len(values) if values else float('nan')
        counts[metric] = len(values)
    summary['foreground_valid_class_counts'] = counts
    return summary


def comparison(c):
    rows=[]
    for theme in themes(c):
        for checkpoint,folder in [('final','validation'),('best','validation_best')]:
            path=model_dir(c,theme)/'fold_0'/folder/'summary.json'
            if path.exists():
                summary=repair_foreground_mean(read(path))
                if len(summary['metric_per_case']) != len(read(locations(c)['pre']/'splits_final.json')[0]['val']):
                    raise ValueError('Validation case count differs from frozen split')
                write(path, summary)
                rows.append(dict(augmentation=theme,checkpoint=checkpoint,
                                 Dice_valid_organs=summary['foreground_valid_class_counts']['Dice'],
                                 IoU_valid_organs=summary['foreground_valid_class_counts']['IoU'],
                                 Dice=summary['foreground_mean']['Dice'],IoU=summary['foreground_mean']['IoU']))
    p=locations(c)['suite']
    write(p/'comparison.json',rows)
    with (p/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['augmentation','checkpoint','Dice','IoU','Dice_valid_organs','IoU_valid_organs'])
        writer.writeheader();writer.writerows(rows)
    return rows


def run(c, resume=False):
    import fcntl
    validate(c)
    if sys.platform != 'linux':
        raise RuntimeError('Run only on Linux GPU server; no local preprocessing/training')
    os.environ.update(environment(c))
    p=locations(c)
    p['suite'].mkdir(parents=True,exist_ok=True)
    def stop(*args):
        raise KeyboardInterrupt('Stopped')
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGHUP,stop)
    with (p['root']/'amos-workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen(p['suite']/'config.json',c)
        try:
            import torch
            import nnunetv2
            if Path(nnunetv2.__file__).resolve().parent != REPO/'nnunetv2':
                raise RuntimeError('Install this checkout with pip install -e .')
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA GPU unavailable; check notebook kernel, driver and PyTorch wheel')
            for name, version in [('batchgeneratorsv2','0.3.8'),('dynamic-network-architectures','0.4.4')]:
                if importlib.metadata.version(name) != version:
                    raise RuntimeError(f'Install {name}=={version} using the notebook setup cell')
            if torch.__version__ != '2.8.0+cu128':
                raise RuntimeError('Use the notebook torch 2.8.0+cu128 environment')
            print('GPU:',torch.cuda.get_device_name(0), 'Total GiB:',torch.cuda.get_device_properties(0).total_memory/2**30, flush=True)
            write(p['suite']/'status.json',dict(status='preparing'))
            prepare(c)
            recipe(c)
            # Probe selected variants before training; isolated output avoids resumable fake checkpoints.
            for theme in themes(c):
                if not (p['suite']/f'smoke-{theme}.json').exists():
                    write(p['suite']/'status.json',dict(status='smoke',theme=theme))
                    call(c,[sys.executable,'-B',__file__,'--config',p['suite']/'config.json','--smoke',theme])
            for theme in themes(c):
                free_space(c)
                write(p['suite']/'status.json',dict(status='training',theme=theme))
                fold=model_dir(c,theme)/'fold_0'
                final_exists = bool(list(fold.glob('checkpoint_final.safetensors')) + list(fold.glob('checkpoint_final.pth')))
                if not (final_exists and (fold/'validation'/'summary.json').exists()):
                    checkpoints=list(fold.glob('checkpoint_*.safetensors'))+list(fold.glob('checkpoint_*.pth'))
                    if checkpoints and not resume:
                        raise RuntimeError('Existing checkpoint: set RESUME=True for identical settings')
                    cli(c,'nnUNetv2_train','620','3d_fullres','0','-tr',trainer(c,theme),'-p',p['plans'],
                        *(['--c'] if checkpoints else []))
                if c['evaluate_best']:
                    call(c,[sys.executable,'-B',__file__,'--config',p['suite']/'config.json','--best',theme])
                comparison(c)
            write(p['suite']/'status.json',dict(status='completed'))
        except BaseException as error:
            state = read(p['suite']/'status.json') if (p['suite']/'status.json').exists() else {}
            write(p['suite']/'status.json',dict(status='failed',stage=state.get('status'),error=str(error),
                  help='Errno 28 / No space left: free disk space or increase storage; AMP does not help. Only CUDA out of memory: AMP=True + NEW RUN_ID. See runner.log.'))
            raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',choices=['on','off'])
    parser.add_argument('--best',choices=['on','off'])
    args=parser.parse_args()
    c=validate(read(args.config))
    os.environ.update(environment(c))
    if sys.platform != 'linux':
        raise RuntimeError('Server execution only')
    if args.smoke:
        # Separate output area: smoke weights are discarded, never resumed for real training.
        os.environ['nnUNet_results']=str(locations(c)['suite']/'smoke-results')
        smoke(c,args.smoke)
    elif args.best:
        best(c,args.best)
    else:
        run(c,args.resume)
