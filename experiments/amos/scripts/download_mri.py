"""Download selected original AMOS ZIP entries via HTTP Range; never fetch the full ZIP."""
import argparse
import io
import json
import os
import re
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import struct
import time
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
import zlib

URL = 'https://zenodo.org/records/7262581/files/amos22.zip?download=1'
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST = 0.0


def request_range(start, end):
    # Bound aggregate request rate across workers (Zenodo enforces a per-minute limit).
    global _LAST_REQUEST
    with _REQUEST_LOCK:
        time.sleep(max(0, 0.6 - (time.monotonic() - _LAST_REQUEST)))
        _LAST_REQUEST = time.monotonic()
    response = urllib.request.urlopen(urllib.request.Request(
        URL, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}), timeout=90)
    expected = f'bytes {start}-{end}/'
    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(expected):
        response.close()
        raise RuntimeError('Server did not honor Range; full ZIP download refused')
    return response


class RemoteZip(io.RawIOBase):
    def __init__(self):
        with request_range(0, 0) as r:
            self.size = int(r.headers['Content-Range'].split('/')[-1])
        self.pos = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        size = self.size - self.pos if size < 0 else min(size, self.size - self.pos)
        if size <= 0:
            return b''
        if size > 16 * 1024**2:
            raise RuntimeError('Unexpected large metadata read refused')
        with request_range(self.pos, self.pos + size - 1) as r:
            data = r.read(size + 1)
        if len(data) != size:
            raise IOError('Incomplete range')
        self.pos += size
        return data


def download_entry(info, root):
    name = PurePosixPath(info.filename)
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('Unsafe ZIP path')
    dest = root.joinpath(*name.parts)
    if dest.exists():
        crc = 0
        with dest.open('rb') as f:
            for block in iter(lambda: f.read(1024**2), b''):
                crc = zlib.crc32(block, crc)
        if dest.stat().st_size == info.file_size and crc == info.CRC:
            return 'verified existing'
        raise ValueError(f'Existing file does not match original: {dest}')
    if info.flag_bits & 1 or info.compress_type not in (0, 8):
        raise ValueError('Unsupported ZIP entry')
    with request_range(info.header_offset, info.header_offset + 29) as r:
        header = r.read(30)
    if header[:4] != b'PK\x03\x04':
        raise ValueError('Invalid local ZIP header')
    name_len, extra_len = struct.unpack_from('<HH', header, 26)
    start = info.header_offset + 30 + name_len + extra_len
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + '.part')
    dec = zlib.decompressobj(-15) if info.compress_type == 8 else None
    crc = count = received = 0
    try:
        with request_range(start, start + info.compress_size - 1) as r, partial.open('wb') as f:
            while received < info.compress_size:
                block = r.read(min(1024**2, info.compress_size - received))
                if not block:
                    raise IOError('Incomplete entry download')
                received += len(block)
                data = dec.decompress(block) if dec else block
                f.write(data)
                crc = zlib.crc32(data, crc)
                count += len(data)
            if dec:
                data = dec.flush()
                f.write(data)
                crc = zlib.crc32(data, crc)
                count += len(data)
                if not dec.eof:
                    raise IOError('Incomplete compressed entry')
        if count != info.file_size or crc != info.CRC:
            raise IOError('ZIP size/CRC verification failed')
        os.replace(partial, dest)
    finally:
        partial.unlink(missing_ok=True)
    return 'downloaded and verified'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--workers', type=int, choices=range(1, 11), default=8)
    args = parser.parse_args()
    with zipfile.ZipFile(RemoteZip()) as archive:
        infos = archive.infolist()
        print('ZIP entries:', len(infos), flush=True)
        # Official archive readme: MRI IDs >= 500. Restrict to labeled splits;
        # imagesTs has no public ground truth and is unnecessary for this experiment.
        selected = []
        counts = {}
        for info in infos:
            match = re.fullmatch(r'amos22/(imagesTr|labelsTr|imagesVa|labelsVa)/amos_(\d+)\.nii\.gz', info.filename)
            if match and int(match[2]) >= 500:
                selected.append(info)
                counts[match[1]] = counts.get(match[1], 0) + 1
        if counts != {'imagesTr': 40, 'labelsTr': 40, 'imagesVa': 20, 'labelsVa': 20}:
            raise ValueError(f'Unexpected MRI split: {counts}')
        names = {i.filename for i in selected}
        for info in selected:
            if '/images' in info.filename and info.filename.replace('/images', '/labels') not in names:
                raise ValueError('Missing paired ground truth')
        for name in ('amos22/dataset.json', 'amos22/readme.md'):
            selected.append(archive.getinfo(name))
        selected.sort(key=lambda i: i.filename)
        compressed = sum(i.compress_size for i in selected)
        extracted = sum(i.file_size for i in selected)
        print('MRI labeled splits:', counts, flush=True)
        print(f'Download: {compressed / 1e9:.3f} GB; disk: {extracted / 1e9:.3f} GB', flush=True)
        if not args.download:
            return
        root = args.output.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < extracted + 1024**3:
            raise RuntimeError('Insufficient free disk space')
        def transfer(info):
            for attempt in range(4):
                try:
                    return download_entry(info, root)
                except (OSError, urllib.error.URLError) as error:
                    if attempt == 3:
                        raise
                    print(f'Retrying {info.filename}: {error}', flush=True)
                    time.sleep(60 if isinstance(error, urllib.error.HTTPError) and error.code == 429 else 15 * (attempt + 1))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(transfer, info): info for info in selected}
            for number, future in enumerate(as_completed(futures), 1):
                info = futures[future]
                print(f'[{number}/{len(selected)}] {info.filename}: {future.result()}', flush=True)
        manifest = {
            'source': URL, 'archive_bytes': archive.fp.size,
            'selection': 'MRI ID >= 500; original train/validation images and labels only; test excluded',
            'counts': counts, 'compressed_bytes': compressed, 'extracted_bytes': extracted,
            'files': []}
        for info in selected:
            path = root / info.filename
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1024**2), b''):
                    digest.update(block)
            manifest['files'].append(dict(path=info.filename, size=info.file_size,
                                         crc32=f'{info.CRC:08x}', sha256=digest.hexdigest()))
        (root / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('Complete:', root, flush=True)



if __name__ == '__main__':
    main()
