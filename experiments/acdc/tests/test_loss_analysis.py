"""Small synthetic tests: no model, GPU, or real dataset inference."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import loss_analysis as la
try:
    import numpy as np
    import SimpleITK as sitk
    import pandas
    import matplotlib
    matplotlib.use("Agg")
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "numpy/pandas/SimpleITK/matplotlib required")
class LossAnalysisTests(unittest.TestCase):
    def test_perfect_predictions(self):
        gt = np.array([[[0, 1], [2, 3]]], dtype=np.uint8)
        prob = np.eye(4, dtype=np.float32)[gt].transpose(3, 0, 1, 2)
        score, ce = la.scores(gt, prob)
        self.assertEqual(score['ce'], 0)
        self.assertEqual(score['soft_dice'], 1)
        self.assertEqual(score['hard_dice'], 1)
        self.assertEqual(score['eval_loss'], -1)
        self.assertTrue((ce == 0).all())

    def test_uniform_and_confident_errors(self):
        gt = np.array([[[0, 1], [2, 3]]], dtype=np.uint8)
        prob = np.full((4, *gt.shape), .25, dtype=np.float32)
        score, _ = la.scores(gt, prob)
        self.assertAlmostEqual(score['ce'], np.log(4), places=6)
        expected_dice = (.5 + la.SMOOTH) / (2 + la.SMOOTH)
        self.assertAlmostEqual(score['soft_dice'], expected_dice)
        wrong = np.eye(4, dtype=np.float32)[(gt + 1) % 4].transpose(3, 0, 1, 2)
        wrong_score, _ = la.scores(gt, wrong)
        self.assertEqual(wrong_score['hard_dice'], 0)
        self.assertTrue(np.isfinite(wrong_score['ce']))
        self.assertGreater(wrong_score['eval_loss'], score['eval_loss'])

    def test_invalid_probabilities(self):
        gt = np.zeros((2, 2), dtype=np.uint8)
        for prob in [np.zeros((4, 2, 2)), np.full((4, 2, 2), np.nan), np.ones((3, 2, 2))]:
            with self.assertRaises(ValueError):
                la.scores(gt, prob)

    def test_empty_foreground(self):
        gt = np.zeros((2, 2), dtype=np.uint8)
        prob = np.zeros((4, 2, 2), dtype=np.float32)
        prob[0] = 1
        score, _ = la.scores(gt, prob)
        self.assertTrue(np.isnan(score['hard_dice']))
        self.assertTrue(np.isnan(score['foreground_ce']))
        self.assertEqual(score['soft_dice'], 1)

    def test_relocated_dataset_and_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / 'runs/acdc-baseline-v1'
            model = run / 'nnUNet_results/Dataset027_ACDC/nnUNetTrainer__nnUNetPlans__3d_fullres'
            (model / 'fold_0').mkdir(parents=True)
            (model / 'fold_0/checkpoint_final.safetensors').write_bytes(b'synthetic-placeholder')
            for name in ['plans.json', 'dataset.json']:
                (model / name).write_text('{}')
            (run / 'config.snapshot.json').write_text(json.dumps({'configuration': {'storage_root': '/old/server'}}))
            split = {'train': ['patient001_frame01'], 'val': ['patient002_frame01']}
            (run / 'splits_final.json').write_text(json.dumps([split]))
            raw = root / 'nnUNet_raw/Dataset027_ACDC'
            gt = np.array([[[0, 1], [2, 3]], [[0, 1], [2, 3]]], dtype=np.uint8)
            img = sitk.GetImageFromArray(gt)
            img.SetSpacing((1.2, 1.5, 5))
            for folder in ['imagesTr', 'labelsTr']:
                (raw / folder).mkdir(parents=True)
            for case in split['train'] + split['val']:
                sitk.WriteImage(img, str(raw / 'imagesTr' / f'{case}_0000.nii.gz'))
                sitk.WriteImage(img, str(raw / 'labelsTr' / f'{case}.nii.gz'))
            ctx = la.context(root)
            with self.assertRaisesRegex(RuntimeError, 'no completed inference'):
                la.analyse(ctx)
            for subset, case in ctx['cases']:
                folder = ctx['out'] / subset
                folder.mkdir(parents=True)
                sitk.WriteImage(img, str(folder / f'{case}.nii.gz'))
                prob = np.eye(4, dtype=np.float32)[gt].transpose(3, 0, 1, 2)
                np.savez_compressed(folder / f'{case}.npz', probabilities=prob)
                (folder / f'{case}.complete.json').write_text('{}')
            cases, slices = la.analyse(ctx)
            self.assertEqual(len(cases), 2)
            self.assertEqual(len(slices), 4)
            self.assertTrue((cases.eval_loss == -1).all())
            data = la.load_case(ctx, 'train', 'patient001_frame01', with_image=True)
            self.assertTrue(np.allclose(data[3], (5, 1.5, 1.2)))
            la.show_slice(ctx, 'train', 'patient001_frame01', 0, loaded=data)
            wrong = sitk.GetImageFromArray(gt)
            wrong.SetSpacing((2, 2, 2))
            sitk.WriteImage(wrong, str(ctx['out'] / 'train/patient001_frame01.nii.gz'))
            with self.assertRaisesRegex(ValueError, 'Geometry mismatch'):
                la.load_case(ctx, 'train', 'patient001_frame01')
            # ED/ES leakage must be rejected even when case names differ.
            (run / 'splits_final.json').write_text(json.dumps([
                {'train': ['patient001_frame01'], 'val': ['patient001_frame02']}]))
            with self.assertRaisesRegex(ValueError, 'Patient leakage'):
                la.context(root)


if __name__ == '__main__':
    unittest.main()
