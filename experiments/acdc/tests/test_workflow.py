"""Offline tests use synthetic bytes and temporary folders; no GPU or medical data."""
import ast
import copy
import gzip
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from workflow import guard, parse_losses, validate_splits, raw_target, read_config, prepare
from run_folds import action_for
from download_data import download_file, load_inventory, validate_inventory, download


class SmokeTests(unittest.TestCase):
    def test_notebooks_and_scripts_parse_without_saved_outputs(self):
        for path in (ROOT/'notebooks').glob('*.ipynb'):
            nb = json.loads(path.read_text())
            for cell in nb['cells']:
                if cell['cell_type'] == 'code':
                    ast.parse(''.join(cell['source']))
                    self.assertIsNone(cell['execution_count'])
                    self.assertEqual(cell['outputs'], [])
        for path in (ROOT/'scripts').glob('*.py'):
            ast.parse(path.read_text())

    def test_disabled_guard_does_not_execute_or_write(self):
        with patch('subprocess.check_output', side_effect=AssertionError('should not execute')):
            with self.assertRaisesRegex(RuntimeError, 'Execution disabled'):
                guard(read_config(ROOT/'config.json'))

    def test_official_inventory_is_100_patients_with_two_pairs(self):
        inventory = load_inventory()
        self.assertEqual(len(inventory['files']), 400)
        self.assertEqual(inventory['total_bytes'], 115805231)
        invalid = copy.deepcopy(inventory)
        invalid['files'][0]['name'] = 'patient001_4d.nii.gz'
        with self.assertRaises(ValueError):
            validate_inventory(invalid)

    def test_patient_leakage_detected_even_when_case_split_valid(self):
        cases = [f'patient{i:03d}_frame{f:02d}' for i in range(1, 6) for f in (1, 10)]
        splits = []
        for i in range(1, 6):
            val = [x for x in cases if x.startswith(f'patient{i:03d}_')]
            splits.append(dict(train=[x for x in cases if x not in val], val=val))
        validate_splits(splits, cases)
        # Disjoint cases and perfect coverage are not enough: swap one phase across folds.
        a, b = splits[0]['val'][1], splits[1]['val'][1]
        splits[0]['val'][1], splits[1]['val'][1] = b, a
        for split in splits:
            split['train'] = [x for x in cases if x not in split['val']]
        with self.assertRaisesRegex(ValueError, 'Patient leakage'):
            validate_splits(splits, cases)

    def test_output_mapping_rejects_4d_and_traversal(self):
        c = read_config(ROOT/'config.json')
        self.assertEqual(raw_target(c, 'patient001_frame01.nii.gz').name, 'patient001_frame01_0000.nii.gz')
        self.assertEqual(raw_target(c, 'patient001_frame01_gt.nii.gz').name, 'patient001_frame01.nii.gz')
        for name in ['patient001_4d.nii.gz', '../bad.nii.gz']:
            with self.assertRaises(ValueError):
                raw_target(c, name)

    def test_download_retry_preserves_old_file_and_cleans_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp)/'source.gz', Path(tmp)/'target.gz'
            data = gzip.compress(b'synthetic')
            src.write_bytes(data)
            download_file(src.as_uri(), dst, len(data))
            self.assertEqual(dst.read_bytes(), data)
            src.write_bytes(b'bad gzip')
            with patch('download_data.time.sleep'):
                with self.assertRaises(ValueError):
                    download_file(src.as_uri(), dst, len(data))
            self.assertEqual(dst.read_bytes(), data)
            self.assertFalse(dst.with_name(dst.name+'.partial').exists())

    def test_direct_download_resumes_without_source_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = read_config(ROOT/'config.json')
            c.update(storage_root=tmp, source_dataset=str(Path(tmp)/'metadata'))
            src = Path(tmp)/'synthetic.gz'
            src.write_bytes(gzip.compress(b'synthetic'))
            entries = [dict(name=f'patient001_frame01{suffix}.nii.gz', url=src.as_uri(), size=src.stat().st_size)
                       for suffix in ('', '_gt')]
            inventory = dict(files=entries, total_bytes=sum(x['size'] for x in entries))
            with patch('download_data.guard'), patch('download_data.check_space'), patch('download_data.load_inventory', return_value=inventory), redirect_stdout(io.StringIO()):
                download(c)
                with patch('download_data.download_file', side_effect=AssertionError('redownload')):
                    download(c)
            for entry in entries:
                self.assertEqual(raw_target(c, entry['name']).read_bytes(), src.read_bytes())
            self.assertEqual([x.name for x in Path(c['source_dataset']).iterdir()], ['download_manifest.json'])

    def test_resumed_loss_logs_use_latest_complete_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'training_log_1.txt').write_text('Epoch 0\ntrain_loss -0.4\nval_loss -0.3\nEpoch 1\ntrain_loss -0.5\nval_loss -0.4\n')
            (p/'training_log_2.txt').write_text('Epoch 1\ntrain_loss -0.6\nval_loss -0.45\nEpoch 2\ntrain_loss -0.7\n')
            os.utime(p/'training_log_1.txt', (1, 1))
            os.utime(p/'training_log_2.txt', (2, 2))
            self.assertEqual(parse_losses(p)[-1], {'epoch': 1, 'train_loss': -.6, 'val_loss': -.45})

    def test_runner_restart_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'fold_0'
            self.assertEqual(action_for(folder, False), 'start')
            folder.mkdir()
            (folder/'checkpoint_latest.safetensors').touch()
            self.assertEqual(action_for(folder, True), 'resume')
            with self.assertRaises(RuntimeError):
                action_for(folder, False)
            (folder/'checkpoint_final.safetensors').touch()
            (folder/'validation').mkdir()
            (folder/'validation'/'summary.json').write_text('{}')
            self.assertEqual(action_for(folder, True), 'skip')


if __name__ == '__main__':
    unittest.main()
