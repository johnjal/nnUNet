"""CPU synthetic tests. Never use the real ACDC data or start training."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('architecture_queue', REPO / 'experiments/acdc/architecture_comparison/runner.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def suite(root):
    return dict(suite_id='test-nods', storage_root=str(root), gpu='0', batch_size=4,
                augmentation_workers=1, models=list(queue.TRAINERS), fold=0, epochs=250,
                evaluate_best=True, execution_enabled=True, expected_git_commit='test')


class QueueTests(unittest.TestCase):
    def test_plans_preserve_preprocessing_and_original(self):
        base = json.loads((REPO / 'experiments/acdc/docs/artifacts/runs/acdc-baseline-v1/nnUNetPlans.json').read_text())
        saved = copy.deepcopy(base)
        new = queue.make_plans(base, 2)
        self.assertEqual(base, saved)
        conf = new['configurations']['3d_fullres']
        self.assertEqual(conf['batch_size'], 2)
        self.assertEqual(conf['data_identifier'], base['configurations']['3d_fullres']['data_identifier'])
        conf['batch_size'] = 4
        self.assertEqual(conf, base['configurations']['3d_fullres'])

    def test_fixed_budget_and_shared_batch(self):
        s = suite('/tmp/test-acdc')
        queue.validate_suite(s)
        a, b = [queue.model_config(s, n) for n in s['models']]
        self.assertNotEqual(a['run_id'], b['run_id'])
        self.assertEqual(a['plans'], b['plans'])
        self.assertEqual(a['folds'], [0])
        self.assertFalse(a['deep_supervision'])
        for key, val in [('epochs', 1000), ('fold', 1), ('batch_size', 3)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                queue.validate_suite(dict(s, **{key: val}))

    def test_queue_order_resume_and_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            s = suite(temp)
            calls = []
            def train(c, resume):
                calls.append((c['trainer'], resume))
                folder = queue.model_dir(c) / 'fold_0'
                (folder / 'validation').mkdir(parents=True)
                (folder / 'checkpoint_final.safetensors').touch()
                result = {'foreground_mean': {'Dice': .9, 'IoU': .82},
                          'mean': {str(i): {'Dice': .9} for i in (1, 2, 3)}}
                (folder / 'validation/summary.json').write_text(json.dumps(result))
            with patch.object(queue, 'preflight'), patch.object(queue, 'freeze_suite'), \
                 patch.object(queue.subprocess, 'run') as smoke, patch.object(queue, 'run_folds', train), \
                 patch.object(queue, 'evaluate_best', side_effect=lambda s, n: calls.append(('best', n))):
                queue.run_suite(s, resume=True)
            self.assertEqual(calls, [(queue.TRAINERS['swin_v2'], True), ('best', 'swin_v2'),
                                     (queue.TRAINERS['unet'], True), ('best', 'unet')])
            self.assertIn('--smoke-only', smoke.call_args.args[0])
            state = json.loads((queue.suite_dir(s) / 'status.json').read_text())
            self.assertEqual(state['status'], 'completed')
            final = [r for r in queue.comparison_rows(s) if r['checkpoint'] == 'final']
            self.assertEqual([r['dice_percent'] for r in final], [90, 90])

    def test_failure_stops_before_second_model(self):
        with tempfile.TemporaryDirectory() as temp:
            s = suite(temp)
            with patch.object(queue, 'preflight'), patch.object(queue, 'freeze_suite'), \
                 patch.object(queue.subprocess, 'run'), \
                 patch.object(queue, 'run_folds', side_effect=RuntimeError('synthetic failure')) as train:
                with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                    queue.run_suite(s)
            self.assertEqual(train.call_count, 1)
            self.assertEqual(json.loads((queue.suite_dir(s) / 'status.json').read_text())['status'], 'failed')

    def test_notebook_is_clean_and_syntactically_valid(self):
        n = json.loads((REPO / 'experiments/acdc/notebooks/07_swinunetr_v2_no_deep_supervision.ipynb').read_text())
        for c in n['cells']:
            if c['cell_type'] == 'code':
                ast.parse(''.join(c['source']))
                self.assertIsNone(c['execution_count'])
                self.assertEqual(c['outputs'], [])


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.torch = torch
        from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerACDCNoDS250 import (
            PaddedSwinUNETRV2, nnUNetTrainerACDCUNetNoDS250, nnUNetTrainerACDCSwinV2NoDS250)
        cls.wrapper = PaddedSwinUNETRV2
        cls.unet = nnUNetTrainerACDCUNetNoDS250
        cls.swin = nnUNetTrainerACDCSwinV2NoDS250

    def test_padding_crops_exact_original_coordinates(self):
        torch = self.torch
        wrapper = self.wrapper.__new__(self.wrapper)
        torch.nn.Module.__init__(wrapper)
        wrapper.network = torch.nn.Identity()
        for shape in [(20, 33, 65), (32, 64, 64)]:
            x = torch.randn(1, 1, *shape, requires_grad=True)
            y = wrapper(x)
            self.assertTrue(torch.equal(x, y))
            y.sum().backward()
            self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_trainer_defaults_and_discovery(self):
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
        def init_stub(obj, plans, configuration, fold, dataset_json, device):
            obj.num_epochs = 1000
            obj.enable_deep_supervision = True
        for cls in [self.unet, self.swin]:
            with patch.object(nnUNetTrainer, '__init__', init_stub):
                trainer = cls({}, '3d_fullres', 0, {}, self.torch.device('cpu'))
            self.assertEqual(trainer.num_epochs, 250)
            self.assertFalse(trainer.enable_deep_supervision)
            self.assertIs(cls.configure_optimizers, nnUNetTrainer.configure_optimizers)
            self.assertIs(cls.get_training_transforms, nnUNetTrainer.get_training_transforms)
            self.assertIs(cls._build_loss, nnUNetTrainer._build_loss)
            trainer.set_deep_supervision_enabled(False)
            trainer.set_deep_supervision_enabled(True)  # base validation's exit hook
            self.assertFalse(trainer.enable_deep_supervision)
            self.assertIs(recursive_find_trainer_class_by_name(cls.__name__), cls)

    def test_best_metrics_preserve_final_outputs_and_reject_changed_inputs(self):
        import numpy as np
        import SimpleITK as sitk
        from nnunetv2.utilities.checkpoints import save_checkpoint_file
        with tempfile.TemporaryDirectory() as temp:
            s = suite(temp)
            c = queue.model_config(s, 'unet')
            fold = queue.model_dir(c) / 'fold_0'
            (fold / 'validation').mkdir(parents=True)
            final_file = fold / 'validation/summary.json'
            final_file.write_text('original final metrics')
            save_checkpoint_file({'current_epoch': 123}, str(fold / 'checkpoint_best.safetensors'))
            raw = Path(temp) / 'nnUNet_raw/Dataset027_ACDC'
            for d in ['imagesTr', 'labelsTr']:
                (raw / d).mkdir(parents=True)
            dest = queue.suite_dir(s) / 'best/unet'
            dest.mkdir(parents=True)
            cases = ['patient001_frame01', 'patient001_frame02']
            for case in cases:
                array = np.tile(np.arange(4, dtype=np.uint8), (2, 3, 1))
                img = sitk.GetImageFromArray(array)
                for pth in [raw / 'imagesTr' / f'{case}_0000.nii.gz',
                            raw / 'labelsTr' / f'{case}.nii.gz']:
                    sitk.WriteImage(img, str(pth))
            split = Path(temp) / 'runs' / c['run_id'] / 'splits_final.json'
            split.write_text(json.dumps([{'train': [], 'val': cases}]))
            def predict(inputs, outputs, **kwargs):
                for output in outputs:
                    case = Path(output).name
                    (dest / f'{case}.nii.gz').write_bytes((raw / 'labelsTr' / f'{case}.nii.gz').read_bytes())
            # Mock only the GPU prediction; actual geometry/metrics/checkpoint I/O run.
            with patch.dict(os.environ, {}, clear=False), patch(
                    'nnunetv2.inference.predict_from_raw_data.nnUNetPredictor') as predictor:
                predictor.return_value.allowed_mirroring_axes = (0, 1, 2)
                predictor.return_value.predict_from_files_sequential.side_effect = predict
                queue.evaluate_best(s, 'unet')
                self.assertEqual(predictor.return_value.initialize_from_trained_model_folder.call_args.kwargs[
                    'checkpoint_name'], 'checkpoint_best.safetensors')
            result = json.loads((dest / 'summary.json').read_text())
            self.assertEqual(result['checkpoint_completed_epochs'], 123)
            self.assertEqual(result['dice_percent'], 100)
            self.assertEqual(result['miou_percent'], 100)
            self.assertEqual(final_file.read_text(), 'original final metrics')
            save_checkpoint_file({'current_epoch': 124}, str(fold / 'checkpoint_best.safetensors'))
            with patch.dict(os.environ, {}, clear=False), self.assertRaisesRegex(ValueError, 'changed'):
                queue.evaluate_best(s, 'unet')

    def test_real_unet_single_output_and_backward(self):
        torch = self.torch
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        base = json.loads((REPO / 'experiments/acdc/docs/artifacts/runs/acdc-baseline-v1/nnUNetPlans.json').read_text())
        pm = PlansManager(queue.make_plans(base, 4))
        net = self.unet.build_network_architecture(pm, pm.get_configuration('3d_fullres'), 1, 4, False)
        x = torch.randn(1, 1, 20, 64, 64)
        y = net(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(tuple(y.shape), (1, 4, 20, 64, 64))
        self.assertFalse(net.decoder.deep_supervision)
        y.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_real_swin_v2_forward_backward_and_inference_restore(self):
        torch = self.torch
        from nnunetv2.utilities.checkpoints import save_checkpoint_file
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        net = self.wrapper(1, 4).train()
        self.assertTrue(net.network.swinViT.use_v2)
        x = torch.randn(1, 1, 20, 32, 64)
        y = net(x)
        self.assertEqual(tuple(y.shape), (1, 4, 20, 32, 64))
        loss = y.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
        net.zero_grad(set_to_none=True)
        net.eval()
        with torch.no_grad():
            expected = net(x)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'fold_0').mkdir()
            source = REPO / 'experiments/acdc/docs/artifacts/runs/acdc-baseline-v1'
            (folder / 'plans.json').write_bytes((source / 'nnUNetPlans.json').read_bytes())
            (folder / 'dataset.json').write_bytes((source / 'dataset.json').read_bytes())
            save_checkpoint_file({'network_weights': net.state_dict(), 'trainer_name': self.swin.__name__,
                                  'init_args': {'configuration': '3d_fullres'},
                                  'inference_allowed_mirroring_axes': (0, 1, 2)},
                                 str(folder / 'fold_0/checkpoint_best.safetensors'))
            with patch.dict(os.environ, {'nnUNet_compile': 'false'}):
                predictor = nnUNetPredictor(device=torch.device('cpu'))
                predictor.initialize_from_trained_model_folder(str(folder), (0,), 'checkpoint_best.safetensors')
            predictor.network.eval()
            with torch.no_grad():
                actual = predictor.network(x)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
