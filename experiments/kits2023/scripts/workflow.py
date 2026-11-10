"""Server workflow helpers. Importing this module never starts work or writes files."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]


def read_config(path):
    c = json.loads(Path(path).read_text())
    if not re.fullmatch(r'[A-Za-z0-9_-]+', c['run_id']):
        raise ValueError('run_id must contain only letters, digits, underscores or hyphens')
    if not c['folds'] or len(set(c['folds'])) != len(c['folds']) or any(f not in range(5) for f in c['folds']):
        raise ValueError('folds must be unique integers in 0..4')
    if not re.fullmatch(r'[0-9]+', c['gpu']):
        raise ValueError('gpu must be one numeric NVIDIA device index')
    if (c['dataset_id'], c['dataset_name'], c['expected_cases'], c['configuration'], c['plans'], c['trainer']) != (
            220, 'Dataset220_KiTS2023', 489, '3d_fullres', 'nnUNetPlans', 'nnUNetTrainer'):
        raise ValueError('This recipe supports the fixed KiTS2023 baseline; extend it explicitly for other experiments')
    return c


def paths(c):
    root = Path(c['storage_root']).expanduser().resolve()
    run = root / 'runs' / c['run_id']
    return dict(root=root, run=run, raw=root/'nnUNet_raw',
                pre=root/'nnUNet_preprocessed', results=run/'nnUNet_results')


def environment(c):
    p = paths(c)
    return dict(os.environ, nnUNet_raw=str(p['raw']), nnUNet_preprocessed=str(p['pre']),
                nnUNet_results=str(p['results']), CUDA_VISIBLE_DEVICES=c['gpu'],
                PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''),
                nnUNet_n_proc_DA=str(c['augmentation_workers']),
                nnUNet_compile='true' if c['compile'] else 'false', nnUNet_wandb_enabled='0',
                XDG_CACHE_HOME=str(p['root']/'cache'), MPLCONFIGDIR=str(p['root']/'cache'/'matplotlib'),
                PYTHONDONTWRITEBYTECODE='1')


def guard(c):
    if not c['execution_enabled'] or sys.platform != 'linux':
        raise RuntimeError('Execution disabled. Enable only in config.server.json on the Linux GPU server.')
    for key in ('storage_root', 'source_dataset'):
        path = Path(c[key]).expanduser()
        if not path.is_absolute() or 'CHANGE_ME' in str(path):
            raise ValueError(f'Configure an absolute server path: {key}')
    root = paths(c)['root']
    source = Path(c['source_dataset']).expanduser().resolve()
    if any(path == REPO or REPO in path.parents for path in (root, source)):
        raise ValueError('Store datasets/results outside the Git checkout')
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if actual != c['expected_git_commit']:
        raise RuntimeError(f'Code revision changed: {actual}; review and explicitly update config')
    os.environ.update(environment(c))  # Set before importing nnUNet or torch.
    import importlib.util
    spec = importlib.util.find_spec('nnunetv2')
    if spec is None or Path(spec.origin).resolve().parent != REPO/'nnunetv2':
        raise RuntimeError('Install this checkout with pip install -e . in the notebook kernel environment')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required; CPU/MPS fallback is intentionally disabled')
    print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30, 2), 'GiB')


def run_command(args, c):
    print(' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), env=environment(c), cwd=REPO, check=True)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_splits(splits, cases):
    expected, seen = set(cases), []
    if len(splits) != 5:
        raise ValueError('Expected five folds')
    for split in splits:
        train, val = set(split['train']), set(split['val'])
        if (not train or not val or train & val or train | val != expected
                or len(train) != len(split['train']) or len(val) != len(split['val'])):
            raise ValueError('Invalid train/validation split')
        seen.extend(split['val'])
    if len(seen) != len(expected) or set(seen) != expected:
        raise ValueError('Each case must appear in validation exactly once')


def prepare(c):
    guard(c)
    p = paths(c)
    source = Path(c['source_dataset'])
    cases = sorted(x.name for x in source.glob('case_*') if x.is_dir())
    from download_data import TRAINING_CASES
    if cases != TRAINING_CASES:
        raise ValueError(f'Expected {c["expected_cases"]} labeled training cases, got {len(cases)}')
    for case in cases:
        for name in ('imaging.nii.gz', 'segmentation.nii.gz'):
            if not (source/case/name).is_file():
                raise FileNotFoundError(source/case/name)
    raw = p['raw']/c['dataset_name']
    marker = raw/'conversion.complete.json'
    source_manifest = source/'download_manifest.json'
    if not source_manifest.exists():
        raise ValueError('Run the download stage first to create the verified data manifest')
    source_hash = digest(source_manifest)
    if marker.exists():
        if json.loads(marker.read_text())['source_manifest_sha256'] != source_hash:
            raise ValueError('Source manifest changed; do not reuse this conversion')
        if (len(list((raw/'imagesTr').glob('*.nii.gz'))) != len(cases)
                or len(list((raw/'labelsTr').glob('*.nii.gz'))) != len(cases)):
            raise ValueError('Converted data is incomplete')
        print('Verified conversion already exists; skipping')
        return
    # Re-running the official converter repairs an interrupted conversion.
    # A pre-existing unmarked dataset is only adopted after checking every file.
    if raw.exists():
        for case in cases:
            for target, original in ((raw/'imagesTr'/f'{case}_0000.nii.gz', source/case/'imaging.nii.gz'),
                                     (raw/'labelsTr'/f'{case}.nii.gz', source/case/'segmentation.nii.gz')):
                if target.exists() and target.stat().st_size >= original.stat().st_size and digest(target) != digest(original):
                    raise ValueError(f'Existing conversion differs: {target}')
    from nnunetv2.dataset_conversion.Dataset220_KiTS2023 import convert_kits2023
    convert_kits2023(str(source), c['dataset_id'])
    write_json(marker, {'source_manifest_sha256': source_hash})


def preprocess(c):
    guard(c)
    p = paths(c)
    labels = p['raw']/c['dataset_name']/'labelsTr'
    cases = sorted(x.name[:-7] for x in labels.glob('*.nii.gz'))
    if len(cases) != c['expected_cases']:
        raise ValueError('Converted dataset count mismatch')
    run_command(['nnUNetv2_plan_and_preprocess', '-d', c['dataset_id'], '-c', c['configuration'],
                 '-np', c['preprocessing_workers'], '--verify_dataset_integrity'], c)
    from nnunetv2.utilities.crossval_split import generate_crossval_split
    split_path = p['pre']/c['dataset_name']/'splits_final.json'
    splits = json.loads(split_path.read_text()) if split_path.exists() else generate_crossval_split(cases)
    validate_splits(splits, cases)
    if not split_path.exists():
        write_json(split_path, splits)
    print('Five folds frozen:', split_path)


def model_dir(c):
    return paths(c)['results']/c['dataset_name']/f'{c["trainer"]}__{c["plans"]}__{c["configuration"]}'


def training_command(c, fold, resume=False):
    args = ['nnUNetv2_train', str(c['dataset_id']), c['configuration'], str(fold),
            '-tr', c['trainer'], '-p', c['plans'], '-device', 'cuda']
    if resume:
        args.append('--c')
    return args


def parse_losses(folder):
    """Read logs without loading large/untrusted checkpoints; newer retries replace epochs."""
    rows = {}
    for log in sorted(Path(folder).glob('training_log_*.txt'), key=lambda x: (x.stat().st_mtime_ns, x.name)):
        epoch = None
        for line in log.read_text(errors='replace').splitlines():
            match = re.search(r'\bEpoch\s+(\d+)\s*$', line)
            if match:
                epoch = int(match[1])
                rows[epoch] = {'epoch': epoch}  # Drop stale values from a retried epoch.
            match = re.search(r'\b(train_loss|val_loss)\s+([-+\deE.]+)', line)
            if match and epoch is not None:
                rows[epoch][match[1]] = float(match[2])
            match = re.search(r'Epoch time:\s*([\deE.+-]+)\s*s', line)
            if match and epoch is not None:
                rows[epoch]['epoch_seconds'] = float(match[1])
    return [rows[k] for k in sorted(rows) if 'train_loss' in rows[k] and 'val_loss' in rows[k]]


def freeze_run(c):
    p = paths(c)
    pre = p['pre']/c['dataset_name']
    files = [Path(c['source_dataset'])/'download_manifest.json', pre/'splits_final.json', pre/f'{c["plans"]}.json', pre/'dataset_fingerprint.json',
             p['raw']/c['dataset_name']/'dataset.json']
    hashes = {x.name: digest(x) for x in files}
    # Folds are a queue selection, not an experimental hyperparameter.
    fixed = {k: v for k, v in c.items() if k not in ('folds', 'execution_enabled')}
    code_files = sorted((REPO/'nnunetv2').rglob('*.py')) + [REPO/'pyproject.toml']
    code_sha = hashlib.sha256(''.join(str(x.relative_to(REPO)) + digest(x) for x in code_files).encode()).hexdigest()
    recipe_sha = hashlib.sha256(''.join(x.name + digest(x) for x in sorted(Path(__file__).parent.glob('*.py'))).encode()).hexdigest()
    packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    frozen = {'configuration': fixed, 'input_sha256': hashes,
              'code_sha256': code_sha, 'recipe_sha256': recipe_sha,
              'packages_sha256': hashlib.sha256(packages.encode()).hexdigest()}
    snapshot = p['run']/'config.snapshot.json'
    if snapshot.exists():
        if json.loads(snapshot.read_text()) != frozen:
            raise ValueError('Run configuration or prepared inputs changed. Choose a new run_id.')
        return
    p['run'].mkdir(parents=True, exist_ok=True)
    write_json(snapshot, frozen)
    for file in files:
        (p['run']/file.name).write_bytes(file.read_bytes())
    manifest = {
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
        'git_status': subprocess.check_output(['git','status','--short'], cwd=REPO, text=True),
        'python': sys.version,
        'packages': packages,
        'gpu': subprocess.check_output(['nvidia-smi'], text=True),
        'notes': c['reproduction_notes'],
    }
    write_json(p['run']/'manifest.json', manifest)
