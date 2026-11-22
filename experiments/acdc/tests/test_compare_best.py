"""CPU-only checks for best-checkpoint comparison; no medical data needed."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('compare_best', Path(__file__).resolve().parents[1] / 'analysis/compare_best.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class BestComparisonTests(unittest.TestCase):
    def test_metrics_and_absent_class(self):
        rows = m.mask_metrics(np.array([0, 1, 1, 2]), np.array([0, 1, 2, 2]))
        self.assertAlmostEqual(rows[0]['dice'], 2 / 3)
        self.assertAlmostEqual(rows[0]['iou'], 1 / 2)
        self.assertTrue(np.isnan(rows[2]['dice']))
        scores = m.aggregate(rows)
        self.assertAlmostEqual(scores['dice_percent'], 200 / 3)
        self.assertAlmostEqual(scores['miou_percent'], 50)
        with self.assertRaises(ValueError):
            m.mask_metrics(np.array([1]), np.array([4]))

    def test_inventory_requires_matching_splits_and_best(self):
        with tempfile.TemporaryDirectory() as root:
            suite = dict(storage_root=root, suite_id='test', themes=list(m.read_suite.__globals__['TRAINERS']),
                         execution_enabled=True, expected_git_commit='test', gpu='0', augmentation_workers=1)
            split = [dict(train=['train_case'], val=[f'case{i:03d}' for i in range(40)])]
            configs = [m.theme_config(suite, t) for t in suite['themes']]
            baseline = dict(configs[0], run_id='baseline', trainer='nnUNetTrainer')
            for c in [baseline, *configs]:
                folder = m.model_dir(c)
                fold = folder / 'fold_0'
                (fold / 'validation').mkdir(parents=True)
                for f in [fold / 'checkpoint_best.safetensors', fold / 'checkpoint_final.safetensors',
                          fold / 'validation/summary.json', folder / 'plans.json', folder / 'dataset.json']:
                    f.write_text('{}')
                (Path(root) / 'runs' / c['run_id'] / 'splits_final.json').write_text(json.dumps(split))
            records, cases = m.models_and_cases(suite, 'baseline')
            self.assertEqual(len(records), 5)
            self.assertEqual(len(cases), 40)
            changed = Path(root) / 'runs' / configs[-1]['run_id'] / 'splits_final.json'
            altered = [dict(train=['different_training_case'], val=split[0]['val'])]
            changed.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, 'split differs'):
                m.models_and_cases(suite, 'baseline')
            changed.write_text(json.dumps(split))
            records[-1]['checkpoint'].unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'checkpoint_best missing'):
                m.models_and_cases(suite, 'baseline')


if __name__ == '__main__':
    unittest.main()
