"""Optional small synthetic metric tests; no real images and no GPU."""
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
AVAILABLE = all(importlib.util.find_spec(m) for m in ('numpy', 'nibabel', 'pandas', 'surface_distance'))


@unittest.skipUnless(AVAILABLE, 'Optional metric dependencies are not installed')
class MetricTests(unittest.TestCase):
    def test_acdc_classes_are_disjoint(self):
        from evaluate_regions import REGIONS
        self.assertEqual(REGIONS, {'RV': (1,), 'MLV': (2,), 'LVC': (3,)})

    def test_surface_distance_uses_mm_and_empty_policy(self):
        import numpy as np
        from evaluate_regions import region_metrics
        gt = np.zeros((8, 8, 8), dtype=bool)
        gt[2, 2, 2] = True
        pred = np.zeros_like(gt)
        pred[4, 2, 2] = True
        self.assertEqual(region_metrics(gt, gt, (1, 1, 1), 2), (1., 1.))
        self.assertEqual(region_metrics(gt, np.zeros_like(gt), (1, 1, 1), 2), (0., 0.))
        self.assertTrue(np.isnan(region_metrics(np.zeros_like(gt), np.zeros_like(gt), (1, 1, 1), 2)[0]))
        close = region_metrics(gt, pred, (1, 1, 1), 2)[1]
        far = region_metrics(gt, pred, (4, 1, 1), 2)[1]
        self.assertGreater(close, far)

    def test_fivefold_report_and_geometry_rejection(self):
        import nibabel as nib
        import numpy as np
        from evaluate_regions import evaluate
        from workflow import paths, model_dir
        with tempfile.TemporaryDirectory() as tmp:
            c = json.loads((ROOT/'config.json').read_text())
            c.update(storage_root=tmp, folds=list(range(5)))
            p = paths(c)
            labels = p['raw']/c['dataset_name']/'labelsTr'
            labels.mkdir(parents=True)
            p['run'].mkdir(parents=True)
            splits = []
            mask = np.zeros((8, 8, 8), dtype=np.uint8)
            mask[1:6, 1:6, 1:6] = 1
            mask[2:3, 2:3, 2:3] = 2
            mask[4:5, 4:5, 4:5] = 3
            for fold in range(5):
                case = f'patient{fold+1:03d}_frame01'
                splits.append({'train': [], 'val': [case]})
                val = model_dir(c)/f'fold_{fold}'/'validation'
                val.mkdir(parents=True)
                (val/'summary.json').write_text('{}')
                image = nib.Nifti1Image(mask, np.diag([1., 1., 2., 1.]))
                nib.save(image, labels/f'{case}.nii.gz')
                nib.save(image, val/f'{case}.nii.gz')
            (p['run']/'splits_final.json').write_text(json.dumps(splits))
            with patch('evaluate_regions.guard'), patch('evaluate_regions.freeze_run'), redirect_stdout(io.StringIO()):
                rows, folds, report = evaluate(c)
                self.assertEqual(len(rows), 15)
                self.assertEqual(len(folds), 5)
                self.assertTrue(report['complete_5fold'])
                self.assertEqual(report['dice_percent'], 100.)
                self.assertEqual(report['nsd_percent'], 100.)
                wrong = nib.Nifti1Image(mask, np.eye(4))
                nib.save(wrong, model_dir(c)/'fold_0'/'validation'/'patient001_frame01.nii.gz')
                with self.assertRaisesRegex(ValueError, 'Geometry mismatch'):
                    evaluate(c)


if __name__ == '__main__':
    unittest.main()
