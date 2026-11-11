"""Download the official labeled ED/ES frames directly into nnU-Net raw folders.

No 4D images, test patients, archives or second copy of the original images.
"""
import argparse
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
    for index, entry in enumerate(inventory['files'], 1):
        name = entry['name']
        target = raw_target(c, name)
        expected = manifest['files'].get(name)
        if target.exists() and expected and digest(target) == expected:
            continue
        if target.exists() and not expected:
            # Recover a download completed just before a crash, without a second copy.
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
        manifest['files'][name] = actual
        write_json(path, manifest)
        print(f'[{index}/400] {name}', flush=True)
    print(f'Download verified: {inventory["total_bytes"]/1e6:.2f} MB, no 4D files.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    download(read_config(parser.parse_args().config))
