"""Sequential theme queue for notebook 06. No work on import; GPU server only."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

from workflow import REPO, guard, freeze_run, paths, model_dir, write_json, check_space, validate_splits
from run_folds import run as run_folds

TRAINERS = {
    'spatial': 'nnUNetTrainerACDCSpatial250',
    'quality': 'nnUNetTrainerACDCQuality250',
    'intensity': 'nnUNetTrainerACDCIntensity250',
    'none': 'nnUNetTrainerACDCNoAug250',
}


def read_suite(path):
    suite = json.loads(Path(path).read_text())
    if not re.fullmatch(r'[A-Za-z0-9_-]+', suite['suite_id']):
        raise ValueError('suite_id must use letters, numbers, underscores or hyphens')
    if not re.fullmatch(r'[0-9]+', suite['gpu']):
        raise ValueError('gpu must be one numeric device index')
    if suite['themes'] not in [list(TRAINERS)[:3], list(TRAINERS)]:
        raise ValueError('Expected spatial, quality, intensity, optionally followed by none')
    if suite['fold'] != 0 or suite['epochs'] != 250:
        raise ValueError('This recipe is fixed to fold 0 and 250 epochs')
    return suite


def theme_config(suite, theme):
    c = json.loads((REPO / 'experiments/acdc/config.json').read_text())
    c.update(execution_enabled=suite['execution_enabled'], storage_root=suite['storage_root'],
             source_dataset=str(Path(suite['storage_root']) / 'metadata/acdc'),
             expected_git_commit=suite['expected_git_commit'], gpu=suite['gpu'],
             augmentation_workers=suite['augmentation_workers'], compile=False,
             run_id=f"{suite['suite_id']}-{theme}", trainer=TRAINERS[theme], folds=[0],
             num_epochs=250, augmentation_theme=theme,
             reproduction_notes='Single augmentation theme; fold 0; 250 epochs; base optimizer/loss/plans. '
                                'Gamma and elastic off. Common default mirroring TTA. Random initialization per run; '
                                'single runs do not isolate seed variance.')
    return c


def suite_dir(suite):
    return Path(suite['storage_root']).expanduser().resolve() / 'augmentation_suites' / suite['suite_id']


def preflight(suite):
    c = theme_config(suite, suite['themes'][0])
    guard(c)
    check_space(c)
    p = paths(c)
    pre = p['pre'] / c['dataset_name']
    for file in [pre / 'preparation.complete.json', pre / 'nnUNetPlans.json',
                 pre / 'dataset_fingerprint.json', pre / 'splits_final.json',
                 p['raw'] / c['dataset_name'] / 'dataset.json',
                 Path(c['source_dataset']) / 'download_manifest.json']:
        if not file.is_file():
            raise FileNotFoundError(f'Copy the prepared acdc-data folder or complete notebook 01: {file}')
    plans = json.loads((pre / 'nnUNetPlans.json').read_text())
    conf = plans['configurations']['3d_fullres']
    if conf['preprocessor_name'] != 'DefaultPreprocessor' or conf['batch_size'] != 4 or conf['patch_size'] != [20, 256, 224]:
        raise ValueError('Prepared plans differ from the ACDC baseline; review before running this recipe')
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
    dataset_folder = pre / conf['data_identifier']
    dataset_class = infer_dataset_class(str(dataset_folder))
    cases = dataset_class.get_identifiers(str(dataset_folder))
    if len(cases) != 200:
        raise ValueError(f'Expected 200 prepared cases, found {len(cases)}')
    validate_splits(json.loads((pre / 'splits_final.json').read_text()), cases)
    for theme in suite['themes']:
        trainer = recursive_find_trainer_class_by_name(TRAINERS[theme])
        if trainer.augmentation_theme != theme:
            raise ValueError('Trainer/theme mismatch')
    return c


def comparison_rows(suite):
    rows = []
    for theme in suite['themes']:
        c = theme_config(suite, theme)
        folder = model_dir(c) / 'fold_0'
        summary_file = folder / 'validation/summary.json'
        row = {'theme': theme, 'run_id': c['run_id'], 'fold': 0, 'epochs': 250,
               'checkpoint': 'final', 'status': 'not_completed'}
        if summary_file.exists() and any(folder.glob('checkpoint_final.*')):
            result = json.loads(summary_file.read_text())
            row.update(status='completed', dice_percent=100 * result['foreground_mean']['Dice'],
                       miou_percent=100 * result['foreground_mean']['IoU'])
            for label, name in [('1', 'RV'), ('2', 'MLV'), ('3', 'LVC')]:
                row[f'dice_{name}_percent'] = 100 * result['mean'][label]['Dice']
                row[f'iou_{name}_percent'] = 100 * result['mean'][label]['IoU']
        rows.append(row)
    return rows


def write_comparison(suite):
    rows = comparison_rows(suite)
    folder = suite_dir(suite)
    write_json(folder / 'comparison.json', rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    tmp = folder / 'comparison.csv.tmp'
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(folder / 'comparison.csv')


def _interrupt(signum, frame):
    # Unwind run_folds.run so it terminates/waits for its training subprocess and releases locks.
    raise KeyboardInterrupt(f'Signal {signum}: stopping suite')


def run_suite(suite, resume=False):
    # Import here so monitoring helpers remain usable on non-Linux machines.
    import fcntl
    preflight(suite)
    folder = suite_dir(suite)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'suite.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This augmentation suite already has an active runner') from None
        saved = folder / 'suite.snapshot.json'
        if saved.exists() and json.loads(saved.read_text()) != suite:
            raise ValueError('Suite settings changed. Choose a new SUITE_ID.')
        write_json(saved, suite)
        # Freeze every theme before starting any training; later code/package drift is rejected.
        for theme in suite['themes']:
            freeze_run(theme_config(suite, theme))
        state = dict(status='running', pid=os.getpid(), started=time.time(), themes={}, active_theme=None)
        write_json(folder / 'status.json', state)
        try:
            for theme in suite['themes']:
                state['active_theme'] = theme
                state['themes'][theme] = 'running'
                write_json(folder / 'status.json', state)
                print(f'=== {theme}: fold 0, 250 epochs ===', flush=True)
                run_folds(theme_config(suite, theme), resume=resume)
                state['themes'][theme] = 'completed'
                write_comparison(suite)
                write_json(folder / 'status.json', state)
            state.update(status='completed', active_theme=None)
        except BaseException as error:
            state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
            state['error'] = str(error)
            if state['active_theme']:
                state['themes'][state['active_theme']] = state['status']
            raise
        finally:
            state['ended'] = time.time()
            write_json(folder / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    run_suite(read_suite(args.config), resume=args.resume)
