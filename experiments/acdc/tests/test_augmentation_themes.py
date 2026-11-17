"""Theme isolation and queue lifecycle; no actual training or dataset access."""
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / 'nnunetv2/training/nnUNetTrainer/variants/data_augmentation/nnUNetTrainerACDCTheme250.py'
sys.path.insert(0, str(REPO / 'experiments/acdc/scripts'))
import run_augmentation_suite as queue


class BaseTrainerStub:
    def __init__(self, plans, configuration, fold, dataset_json, device):
        # Parent's checkpoint metadata introspects the subclass signature.
        parameters = inspect.signature(self.__init__).parameters
        local = dict(plans=plans, configuration=configuration, fold=fold,
                     dataset_json=dataset_json, device=device)
        self.my_init_kwargs = {k: local[k] for k in parameters}
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50


class Record:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


def load_variants(real=False):
    tree = ast.parse(SOURCE.read_text())
    ns = {'nnUNetTrainer': BaseTrainerStub}
    if real:
        tree.body = [n for n in tree.body if not (
            isinstance(n, ast.ImportFrom) and n.module == 'nnunetv2.training.nnUNetTrainer.nnUNetTrainer')]
    else:
        ns['torch'] = types.SimpleNamespace(device=lambda value: value)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module.startswith('batchgeneratorsv2'):
                for item in node.names:
                    ns[item.name] = type(item.name, (Record,), {})
        tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    exec(compile(tree, str(SOURCE), 'exec'), ns)
    return ns


def transform_for(ns, theme, **overrides):
    args = dict(patch_size=(4, 32, 32), rotation_for_DA=(-3.14159, 3.14159),
                deep_supervision_scales=[[1, 1, 1], [1, .5, .5]], mirror_axes=(0, 1, 2),
                do_dummy_2d_data_aug=True, use_mask_for_norm=[False])
    args.update(overrides)
    return ns[queue.TRAINERS[theme]].get_training_transforms(**args)


class ThemeContractTests(unittest.TestCase):
    def test_exclusive_transforms_and_probabilities(self):
        ns = load_variants()
        expected = {'spatial': [], 'quality': ['GaussianNoiseTransform', 'GaussianBlurTransform', 'SimulateLowResolutionTransform'],
                    'intensity': ['MultiplicativeBrightnessTransform', 'ContrastTransform'], 'none': []}
        for theme in expected:
            transforms = transform_for(ns, theme).args[0]
            names = [type(t).__name__ for t in transforms]
            wrapped = [t for t in transforms if type(t).__name__ == 'RandomTransform']
            self.assertEqual([type(t.args[0]).__name__ for t in wrapped], expected[theme])
            self.assertEqual('MirrorTransform' in names, theme == 'spatial')
            spatial = next(t for t in transforms if type(t).__name__ == 'SpatialTransform')
            self.assertEqual(spatial.kwargs['p_rotation'], .2 if theme == 'spatial' else 0)
            self.assertEqual(spatial.kwargs['p_scaling'], .2 if theme == 'spatial' else 0)
            self.assertEqual(spatial.kwargs['p_elastic_deform'], 0)
            self.assertEqual(spatial.args[0], (32, 32))
            self.assertEqual(names[-2:], ['RemoveLabelTansform', 'DownsampleSegForDSTransform'])
            self.assertEqual(names[0], 'Convert3DTo2DTransform')
            self.assertIn('Convert2DTo3DTransform', names)
            if theme == 'quality':
                self.assertEqual([t.kwargs['apply_probability'] for t in wrapped], [.1, .2, .25])
                self.assertEqual(wrapped[-1].args[0].kwargs['ignore_axes'], (0,))
            if theme == 'intensity':
                self.assertEqual([t.kwargs['apply_probability'] for t in wrapped], [.15, .15])
            trainer = ns[queue.TRAINERS[theme]]({}, '3d_fullres', 0, {}, 'cpu')
            self.assertEqual(trainer.num_epochs, 250)
            self.assertEqual(trainer.num_iterations_per_epoch, 250)
            self.assertEqual(trainer.num_val_iterations_per_epoch, 50)
            self.assertEqual(set(trainer.my_init_kwargs), {'plans', 'configuration', 'fold', 'dataset_json', 'device'})

    def test_enabled_transforms_match_base_parameters(self):
        ns = load_variants()
        base_source = REPO / 'nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py'
        tree = ast.parse(base_source.read_text())
        base = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == 'nnUNetTrainer')
        fn = next(m for m in base.body if isinstance(m, ast.FunctionDef) and m.name == 'get_training_transforms')
        for arg in fn.args.args:
            arg.annotation = None
        fn.returns = None
        fn.decorator_list = []
        for name in ['GammaTransform', 'ApplyRandomBinaryOperatorTransform',
                     'RemoveRandomConnectedComponentFromOneHotEncodingTransform',
                     'MoveSegAsOneHotToDataTransform', 'ConvertSegmentationToRegionsTransform']:
            ns[name] = type(name, (Record,), {})
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(base_source), 'exec'), ns)
        baseline = ns['get_training_transforms']((4, 32, 32), (-3.14159, 3.14159),
                    [[1, 1, 1], [1, .5, .5]], (0, 1, 2), True, [False]).args[0]
        def kind(t):
            return type(t.args[0]).__name__ if type(t).__name__ == 'RandomTransform' else type(t).__name__
        def serial(value):
            if isinstance(value, Record):
                return (type(value).__name__, serial(value.args), serial(value.kwargs))
            if isinstance(value, (tuple, list)):
                return [serial(v) for v in value]
            if isinstance(value, dict):
                return {k: serial(v) for k, v in value.items()}
            return value
        lookup = {kind(t): t for t in baseline}
        for theme in queue.TRAINERS:
            for transform in transform_for(ns, theme).args[0]:
                expected = copy.deepcopy(lookup[kind(transform)])
                if kind(transform) == 'SpatialTransform' and theme != 'spatial':
                    expected.kwargs.update(p_rotation=0, p_scaling=0)
                self.assertEqual(serial(transform), serial(expected))

    def test_inference_and_validation_are_inherited(self):
        tree = ast.parse(SOURCE.read_text())
        methods = {n.name for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(methods, {'__init__', 'get_training_transforms'})
        ns = load_variants()
        with self.assertRaises(ValueError):
            transform_for(ns, 'spatial', is_cascaded=True)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('batchgeneratorsv2'),
                         'Optional CPU transform dependencies are absent')
    def test_real_transforms_preserve_shape_and_label_alignment(self):
        import torch
        torch.set_num_threads(1)
        ns = load_variants(real=True)
        label = torch.zeros((1, 4, 48, 48), dtype=torch.int16)
        label[:, :, 8:20, 8:20] = 1
        label[:, :, 20:35, 8:20] = 2
        label[:, :, 16:32, 24:36] = 3
        for theme in queue.TRAINERS:
            transform = transform_for(ns, theme)
            for seed in range(3):
                torch.manual_seed(seed)
                out = transform(image=label.float().clone(), segmentation=label.clone())
                self.assertEqual(tuple(out['image'].shape), (1, 4, 32, 32))
                targets = out['segmentation']
                self.assertEqual(tuple(targets[0].shape), (1, 4, 32, 32))
                self.assertEqual(tuple(targets[1].shape), (1, 4, 16, 16))
                self.assertTrue(torch.isfinite(out['image']).all())
                self.assertTrue(set(torch.unique(targets[0]).tolist()) <= {0, 1, 2, 3})
                if theme == 'none':
                    torch.testing.assert_close(out['image'], label[:, :, 8:40, 8:40].float())
                    torch.testing.assert_close(out['image'], targets[0].float())
                if theme in ('quality', 'intensity'):
                    torch.testing.assert_close(targets[0], label[:, :, 8:40, 8:40])


class SuiteTests(unittest.TestCase):
    def fixture(self, root):
        return dict(suite_id='test-suite', storage_root=str(root), gpu='0', augmentation_workers=4,
                    themes=['spatial', 'quality', 'intensity', 'none'], fold=0, epochs=250,
                    execution_enabled=True, expected_git_commit='test')

    def test_schema_and_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = self.fixture(tmp)
            p = Path(tmp) / 'config.json'
            p.write_text(json.dumps(suite))
            self.assertEqual(queue.read_suite(p), suite)
            configs = [queue.theme_config(suite, t) for t in suite['themes']]
            ignore = {'run_id', 'trainer', 'augmentation_theme'}
            normalized = [{k: v for k, v in c.items() if k not in ignore} for c in configs]
            self.assertTrue(all(c == normalized[0] for c in normalized))
            self.assertEqual(len({c['run_id'] for c in configs}), 4)
            suite['epochs'] = 1000
            p.write_text(json.dumps(suite))
            with self.assertRaises(ValueError):
                queue.read_suite(p)

    def test_sequential_queue_resume_and_settings_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = self.fixture(tmp)
            with patch.object(queue, 'preflight'), patch.object(queue, 'freeze_run'), patch.object(queue, 'run_folds') as run:
                queue.run_suite(suite, resume=True)
                self.assertEqual([c.args[0]['augmentation_theme'] for c in run.call_args_list], suite['themes'])
                self.assertTrue(all(c.kwargs['resume'] for c in run.call_args_list))
                status = json.loads((queue.suite_dir(suite) / 'status.json').read_text())
                self.assertEqual(status['status'], 'completed')
                altered = copy.deepcopy(suite)
                altered['gpu'] = '1'
                with self.assertRaises(ValueError):
                    queue.run_suite(altered)

    def test_failure_stops_remaining_themes(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = self.fixture(tmp)
            with patch.object(queue, 'preflight'), patch.object(queue, 'freeze_run'), \
                    patch.object(queue, 'run_folds', side_effect=[None, RuntimeError('synthetic failure')]) as run:
                with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                    queue.run_suite(suite)
                self.assertEqual(run.call_count, 2)
                status = json.loads((queue.suite_dir(suite) / 'status.json').read_text())
                self.assertEqual(status['status'], 'failed')
                self.assertEqual(status['themes']['quality'], 'failed')
                self.assertNotIn('intensity', status['themes'])


if __name__ == '__main__':
    unittest.main()
