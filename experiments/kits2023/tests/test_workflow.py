"""Offline smoke tests: only synthetic files in TemporaryDirectory, no nnU-Net imports."""
import ast
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from workflow import guard, parse_losses, validate_splits, training_command, read_config
from run_folds import action_for
from download_data import download_file, TRAINING_CASES


class SmokeTests(unittest.TestCase):
    def test_all_notebook_cells_compile_and_are_clean(self):
        for path in (ROOT/'notebooks').glob('*.ipynb'):
            nb = json.loads(path.read_text())
            self.assertEqual(nb['nbformat'], 4)
            for cell in nb['cells']:
                if cell['cell_type'] == 'code':
                    ast.parse(''.join(cell['source']))
                    self.assertIsNone(cell['execution_count'])
                    self.assertEqual(cell['outputs'], [])

    def test_disabled_guard_has_no_side_effects(self):
        cfg = read_config(ROOT/'config.json')
        with patch('subprocess.check_output', side_effect=AssertionError('should not execute')):
            with self.assertRaisesRegex(RuntimeError, 'Execution disabled'):
                guard(cfg)

    def test_resume_logs_replace_stale_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'training_log_1.txt').write_text('2026: Epoch 0\ntrain_loss -0.4\nval_loss -0.3\nEpoch 1\ntrain_loss -0.5\nval_loss -0.4\n')
            (p/'training_log_2.txt').write_text('2026: Epoch 1\ntrain_loss -0.6\nval_loss -0.45\nEpoch 2\ntrain_loss -0.7\n')
            self.assertEqual(parse_losses(p), [
                {'epoch': 0, 'train_loss': -.4, 'val_loss': -.3},
                {'epoch': 1, 'train_loss': -.6, 'val_loss': -.45}])

    def test_split_leakage_and_duplicate_detection(self):
        cases = list('abcde')
        splits = [dict(train=[x for x in cases if x != v], val=[v]) for v in cases]
        validate_splits(splits, cases)
        splits[0]['train'].append('a')
        with self.assertRaises(ValueError):
            validate_splits(splits, cases)

    def test_fold_restart_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'fold_0'
            self.assertEqual(action_for(folder, False), 'start')
            folder.mkdir()
            (folder/'training_log_0.txt').touch()
            with self.assertRaises(RuntimeError):
                action_for(folder, True)
            (folder/'checkpoint_latest.safetensors').touch()
            self.assertEqual(action_for(folder, True), 'resume')
            with self.assertRaises(RuntimeError):
                action_for(folder, False)
            (folder/'checkpoint_final.safetensors').touch()
            (folder/'validation').mkdir()
            (folder/'validation'/'summary.json').write_text('{}')
            self.assertEqual(action_for(folder, True), 'skip')

    def test_download_atomic_and_corrupt_gzip_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp)/'source.gz', Path(tmp)/'destination.gz'
            src.write_bytes(gzip.compress(b'synthetic test data'))
            download_file(src.as_uri(), dst)
            self.assertEqual(gzip.decompress(dst.read_bytes()), b'synthetic test data')
            src.write_bytes(b'not gzip')
            with patch('download_data.time.sleep'):
                with self.assertRaises(gzip.BadGzipFile):
                    download_file(src.as_uri(), dst)
            self.assertEqual(gzip.decompress(dst.read_bytes()), b'synthetic test data')
            self.assertFalse(dst.with_name(dst.name+'.partial').exists())

    def test_training_command_and_cases(self):
        c = read_config(ROOT/'config.json')
        self.assertEqual(len(TRAINING_CASES), 489)
        self.assertNotIn('case_00300', TRAINING_CASES)
        self.assertNotIn('case_00589', TRAINING_CASES)
        cmd = training_command(c, 0, True)
        self.assertEqual(cmd[-1], '--c')
        self.assertIn('3d_fullres', cmd)
        self.assertNotIn('--val_best', cmd)


if __name__ == '__main__':
    unittest.main()
