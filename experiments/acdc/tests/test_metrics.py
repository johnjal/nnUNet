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
AVAILABLE = all(importlib.util.find_spec(m) for m in ('numpy', 'nibabel', 'pandas', 'surface_distance', 'SimpleITK'))


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
            with patch('evaluate_regions.guard'), patch('evaluate_regions.freeze_run', return_value=None), redirect_stdout(io.StringIO()):
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

    def test_acdc_conflicting_sform_and_real_geometry_mismatches(self):
        import nibabel as nib
        import numpy as np
        import SimpleITK as sitk
        from evaluate_regions import read_masks, region_metrics
        with tempfile.TemporaryDirectory() as tmp:
            truth_path = Path(tmp)/'truth.nii.gz'
            pred_path = Path(tmp)/'prediction.nii.gz'
            mask = np.zeros((6, 7, 8), dtype=np.uint8)
            mask[2, 3, 4] = 1
            image = nib.Nifti1Image(mask, np.diag([-1.3, -1.3, 10., 1.]))
            image.set_qform(np.diag([-1.3, -1.3, 10., 1.]), code=1)
            image.set_sform(np.diag([-1., -1., 1., 1.]), code=1)
            nib.save(image, truth_path)
            # Match nnU-Net's export: retain the geometry interpreted by SimpleITK.
            prediction = sitk.ReadImage(str(truth_path))
            sitk.WriteImage(prediction, str(pred_path))
            self.assertFalse(np.allclose(nib.load(truth_path).affine, nib.load(pred_path).affine))
            gt, pred, spacing = read_masks(truth_path, pred_path)
            np.testing.assert_array_equal(gt, mask.transpose(2, 1, 0))
            np.testing.assert_allclose(spacing, (10., 1.3, 1.3))
            self.assertEqual(region_metrics(gt == 1, pred == 1, spacing, 2), (1., 1.))
            # Two slices apart is 20 mm, not 2.6 mm: guard against reversed NSD axes.
            shifted = np.roll(pred, 2, axis=0)
            self.assertEqual(region_metrics(gt == 1, shifted == 1, spacing, 2)[1], 0.)
            for name, value in [('Spacing', (2., 1.3, 10.)), ('Origin', (1., 0., 0.)),
                                ('Direction', (-1., 0., 0., 0., -1., 0., 0., 0., 1.))]:
                with self.subTest(name=name):
                    changed = sitk.Image(prediction)
                    getattr(changed, f'Set{name}')(value)
                    sitk.WriteImage(changed, str(pred_path))
                    with self.assertRaisesRegex(ValueError, 'Geometry mismatch'):
                        read_masks(truth_path, pred_path)


if __name__ == '__main__':
    unittest.main()
