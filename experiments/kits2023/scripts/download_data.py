"""Download official KiTS23 training images and labels only, with restart support."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import time
import urllib.request
from workflow import digest, guard, read_config, write_json

ANNOTATION_REVISION = 'c1088353084c17b8882a11db71429e7c022b7785'
TRAINING_CASES = [f'case_{i:05d}' for i in list(range(300)) + list(range(400, 589))]


def download_file(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.partial')
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'kits2023-reproduction'})
            with urllib.request.urlopen(request, timeout=120) as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            # Stream gzip CRC verification without allocating a complete CT volume.
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
    root = Path(c['source_dataset'])
    manifest_path = root/'download_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['annotation_revision'] != ANNOTATION_REVISION:
            raise ValueError('Annotation revision mismatch; use a new source_dataset directory')
    else:
        # Freeze HF main once. Copy this manifest to another server BEFORE downloading
        # to use the exact same resolved image revision and hashes there.
        url = 'https://huggingface.co/api/datasets/neheller/KiTS-Challenge-Imaging/revision/main'
        with urllib.request.urlopen(url, timeout=60) as response:
            revision = json.load(response)['sha']
        manifest = {'annotation_revision': ANNOTATION_REVISION, 'image_revision': revision, 'files': {}}
        write_json(manifest_path, manifest)
    for index, case in enumerate(TRAINING_CASES, 1):
        urls = {
            'imaging.nii.gz': f'https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/{manifest["image_revision"]}/images/{case}.nii.gz',
            'segmentation.nii.gz': f'https://raw.githubusercontent.com/neheller/kits23/{ANNOTATION_REVISION}/dataset/{case}/segmentation.nii.gz',
        }
        for name, url in urls.items():
            key = f'{case}/{name}'
            target = root/key
            expected = manifest['files'].get(key)
            if target.exists() and expected and digest(target) == expected:
                continue
            download_file(url, target)
            actual = digest(target)
            if expected and actual != expected:
                raise ValueError(f'Download checksum changed: {key}')
            manifest['files'][key] = actual
            write_json(manifest_path, manifest)
        print(f'[{index}/{len(TRAINING_CASES)}] {case}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    download(read_config(parser.parse_args().config))
