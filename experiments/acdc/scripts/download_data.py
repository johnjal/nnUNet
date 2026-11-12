"""Download the official labeled ED/ES frames directly into nnU-Net raw folders.

No 4D images, test patients, archives or second copy of the original images.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import re
import shutil
import time
import urllib.request
from workflow import digest, guard, read_config, write_json, raw_target, check_space

INVENTORY = Path(__file__).resolve().parents[1]/'data_manifest.json'


def validate_inventory(inventory):
    files = inventory['files']
    by_patient = {}
    names = set()
    for entry in files:
        name = entry['name']
        match = re.fullmatch(r'(patient\d{3})_frame\d{2}(_gt)?\.nii\.gz', name)
        if not match or name in names or entry['patient'] != match[1]:
            raise ValueError(f'Invalid or duplicate 3D frame: {name}')
        if entry['size'] <= 0 or not entry['url'].startswith(
                'https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/item/'):
            raise ValueError('Invalid size or nonofficial download URL')
        names.add(name)
        by_patient.setdefault(match[1], set()).add(name)
    if set(by_patient) != {f'patient{i:03d}' for i in range(1, 101)} or len(files) != 400:
        raise ValueError('Expected 100 training patients and 400 image/mask files')
    for patient, patient_files in by_patient.items():
        images = {x for x in patient_files if not x.endswith('_gt.nii.gz')}
        masks = {x.replace('_gt.nii.gz', '.nii.gz') for x in patient_files if x.endswith('_gt.nii.gz')}
        if len(images) != 2 or images != masks:
            raise ValueError(f'Expected two paired phases for {patient}')
    if sum(x['size'] for x in files) != inventory['total_bytes']:
        raise ValueError('Inventory total size mismatch')


def load_inventory():
    inventory = json.loads(INVENTORY.read_text())
    validate_inventory(inventory)
    return inventory


def download_file(url, target, expected_size):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.partial')
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'acdc-reproduction'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as out:
                shutil.copyfileobj(response, out)
            if temporary.stat().st_size != expected_size:
                raise ValueError(f'Download size changed for {target.name}')
            with gzip.open(temporary, 'rb') as stream:
                while stream.read(1024*1024):
                    pass
            temporary.replace(target)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
        finally:
            temporary.unlink(missing_ok=True)


def download(c):
    guard(c)
    check_space(c)
    inventory = load_inventory()
    inventory_sha = digest(INVENTORY)
    path = Path(c['source_dataset'])/'download_manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest['inventory_sha256'] != inventory_sha:
            raise ValueError('Official file inventory changed; do not mix download versions')
    else:
        manifest = {'inventory_sha256': inventory_sha, 'files': {}}
        write_json(path, manifest)
    workers = c.get('download_workers', 10)  # Existing server configs also default to ten.
    if type(workers) is not int or not 1 <= workers <= 10:
        raise ValueError('download_workers must be an integer from 1 to 10')

    def fetch(entry, expected):
        name = entry['name']
        target = raw_target(c, name)
        if target.exists() and expected and digest(target) == expected:
            return name, expected, True
        if target.exists() and not expected:
            # Recover a completed file not yet recorded when a previous run stopped.
            if target.stat().st_size != entry['size']:
                raise ValueError(f'Untracked file has unexpected size: {target}')
            with gzip.open(target, 'rb') as stream:
                while stream.read(1024*1024):
                    pass
        else:
            download_file(entry['url'], target, entry['size'])
        actual = digest(target)
        if expected and actual != expected:
            raise ValueError(f'File content changed since the first download: {name}')
        return name, actual, False

    total = len(inventory['files'])
    errors = []
    print(f'Download concurrency: {workers}; checking {total} files', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, entry, manifest['files'].get(entry['name'])): entry['name']
                   for entry in inventory['files']}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                name, actual, reused = future.result()
            except Exception as error:
                name = futures[future]
                errors.append((name, error))
                print(f'[{index}/{total}] FAILED {name}: {error}', flush=True)
                continue
            if manifest['files'].get(name) != actual:
                manifest['files'][name] = actual
                # Only the main thread writes; sorting makes the file independent of completion order.
                manifest['files'] = dict(sorted(manifest['files'].items()))
                write_json(path, manifest)
            print(f'[{index}/{total}] {"REUSED" if reused else "OK"} {name}', flush=True)
    if errors:
        name, error = errors[0]
        raise RuntimeError(f'{len(errors)} download(s) failed; successful files saved. First failure: {name}') from error
    print(f'Download verified: {inventory["total_bytes"]/1e6:.2f} MB, no 4D files.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    download(read_config(parser.parse_args().config))
