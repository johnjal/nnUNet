"""Restartable download -> conversion -> planning pipeline, suitable for tmux."""
import argparse
import fcntl
import json
from workflow import guard, paths, read_config, prepare, preprocess, digest, write_json
from download_data import download


def main(c):
    guard(c)
    p = paths(c)
    p['root'].mkdir(parents=True, exist_ok=True)
    with (p['root']/'prepare.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        download(c)
        prepare(c)
        pre = p['pre']/c['dataset_name']
        marker = pre/'preparation.complete.json'
        signature = {
            'commit': c['expected_git_commit'], 'configuration': c['configuration'],
            'source_sha256': digest(p['raw']/c['dataset_name']/'conversion.complete.json'),
        }
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved['signature'] != signature:
                raise RuntimeError('Preparation settings changed: use a separate storage_root')
            for name, sha in saved['hashes'].items():
                if digest(pre/name) != sha:
                    raise RuntimeError(f'Prepared metadata changed: {name}')
            print('Preparation already complete; skipping preprocessing', flush=True)
            return
        preprocess(c)
        names = ['nnUNetPlans.json', 'splits_final.json', 'dataset_fingerprint.json']
        write_json(marker, {'signature': signature, 'hashes': {name: digest(pre/name) for name in names}})
        print('PREPARATION COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(read_config(parser.parse_args().config))
