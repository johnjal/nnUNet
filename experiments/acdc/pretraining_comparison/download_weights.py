"""Download only the official VoCo Base SSL checkpoint; verify published LFS SHA256."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

MODEL = 'Luffy503/VoCo'
FILENAME = 'VoCo_B_SSL_head.pt'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download(folder):
    if sys.platform != 'linux':
        raise RuntimeError('Download on the Linux server only')
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/FILENAME
    manifest=folder/(FILENAME+'.json')
    if manifest.exists():
        meta=json.loads(manifest.read_text())
        if target.exists() and sha256(target)==meta['sha256']:
            return target,meta
    revision=get_json(f'https://huggingface.co/api/models/{MODEL}')['sha']
    files=get_json(f'https://huggingface.co/api/models/{MODEL}/tree/{revision}')
    entry=next(f for f in files if f['path']==FILENAME)
    expected=entry['lfs']['oid']
    url=f'https://huggingface.co/{MODEL}/resolve/{revision}/{FILENAME}'
    tmp=target.with_suffix('.pt.part')
    try:
        with urllib.request.urlopen(url,timeout=120) as response, tmp.open('wb') as f:
            while block:=response.read(1024*1024):
                f.write(block)
        if tmp.stat().st_size!=entry['size'] or sha256(tmp)!=expected:
            raise ValueError('VoCo download size/SHA256 mismatch')
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    meta=dict(model=MODEL,filename=FILENAME,revision=revision,sha256=expected,url=url,
              release='Large-Scale-Medical / PreCT-160K SSL; not original CVPR checkpoint')
    manifest.write_text(json.dumps(meta,indent=2)+'\n')
    return target,meta
