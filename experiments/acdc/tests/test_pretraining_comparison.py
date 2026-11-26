"""CPU-only protocol/weight-loading tests, no torch or dataset installation needed."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from contextlib import ExitStack

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO/'experiments/acdc/pretraining_comparison'
spec=importlib.util.spec_from_file_location('acdc_voco_runner_test',TOOLS/'runner.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
TRAINER=REPO/'nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerACDCVoCo250.py'


def standalone_function(path,name):
    tree=ast.parse(path.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    scope={}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]


class Tensor:
    def __init__(self,shape): self.shape=shape


class ProtocolTests(unittest.TestCase):
    def test_checkpoint_mapping_requires_complete_encoder(self):
        select=standalone_function(TRAINER,'select_encoder_weights')
        target={'swinViT.a':Tensor((2,3)), 'swinViT.b':Tensor((4,)), 'decoder.weight':Tensor((8,))}
        source={'module.swin_vit.a':Tensor((2,3)), 'module.swin_vit.b':Tensor((4,)),
                'decoder.weight':Tensor((8,))}
        loaded,ignored=select({'state_dict':source},target)
        self.assertEqual(set(loaded),{'swinViT.a','swinViT.b'})
        self.assertEqual(ignored,['decoder.weight'])
        with self.assertRaisesRegex(ValueError,'incompatible'):
            select({'swinViT.a':Tensor((2,3))},target)
        with self.assertRaisesRegex(ValueError,'shape_mismatch'):
            select({'swinViT.a':Tensor((9,)), 'swinViT.b':Tensor((4,))},target)

    def test_batch_cannot_silently_differ_from_baseline(self):
        suite=dict(suite_id='test',gpu='0',models=['scratch','voco'],fold=0,epochs=250,
                   seed=12345,batch_size=4,storage_root='/tmp/acdc',augmentation_workers=2,evaluate_best=True)
        runner.validate_suite(suite)
        for batch in (1,2):
            with self.assertRaises(ValueError): runner.validate_suite(dict(suite,batch_size=batch))

    def test_plans_keep_existing_preprocessed_data_and_patch(self):
        base={'plans_name':'nnUNetPlans','configurations':{'3d_fullres':{
            'data_identifier':'existing-data','batch_size':4,'patch_size':[20,256,224]}}}
        derived=runner.make_plans(base,4)
        self.assertEqual(derived['configurations'],base['configurations'])
        self.assertEqual(base['plans_name'],'nnUNetPlans')
        self.assertNotEqual(derived['plans_name'],base['plans_name'])

    def test_noaug_transform_has_no_random_augmentation(self):
        source=REPO/'nnunetv2/training/nnUNetTrainer/variants/data_augmentation/nnUNetTrainerACDCTheme250.py'
        tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_training_transforms')
        fn.decorator_list=[]
        def constructor(name):
            return lambda *args,**kwargs: dict(name=name,args=args,kwargs=kwargs)
        scope={name:constructor(name) for name in ['SpatialTransform','Convert3DTo2DTransform',
            'Convert2DTo3DTransform','RemoveLabelTansform']}
        scope['ComposeTransforms']=lambda transforms:transforms
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),scope)
        fake=type('NoAug',(),{'augmentation_theme':'none'})
        transforms=scope['get_training_transforms'](fake,[20,256,224],(-3.14,3.14),None,None,True)
        spatial=next(t for t in transforms if t['name']=='SpatialTransform')['kwargs']
        self.assertEqual((spatial['p_rotation'],spatial['p_scaling'],spatial['p_elastic_deform']),(0,0,0))
        self.assertEqual([t['name'] for t in transforms],['Convert3DTo2DTransform','SpatialTransform',
            'Convert2DTo3DTransform','RemoveLabelTansform'])

    def test_trainer_constructor_preserves_checkpoint_metadata_signature(self):
        tree=ast.parse(TRAINER.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='nnUNetTrainerACDCSwinScratchNoAug250')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        self.assertEqual([a.arg for a in fn.args.args],
                         ['self','plans','configuration','fold','dataset_json','device'])
        self.assertIsNone(fn.args.vararg)
        self.assertIsNone(fn.args.kwarg)

    def test_each_job_runs_only_the_selected_model(self):
        for task in ('scratch','voco','unet_existing'):
            with TemporaryDirectory() as tmp, ExitStack() as stack:
                suite={'suite_id':'separate','storage_root':tmp}
                mocks={name:stack.enter_context(patch.object(runner,name)) for name in
                       ('preflight','freeze_task','model_config','environment','run_folds','evaluate')}
                mocks['environment'].return_value={}
                process=stack.enter_context(patch.object(runner.subprocess,'run'))
                runner.run_task(suite,task)
                self.assertEqual([call.args[1:] for call in mocks['evaluate'].call_args_list],
                                 [(task,'final'),(task,'best')])
                if task=='unet_existing':
                    process.assert_not_called()
                    mocks['run_folds'].assert_not_called()
                else:
                    mocks['run_folds'].assert_called_once()
                    command=process.call_args.args[0]
                    self.assertEqual(command[command.index('--task')+1],task)
                state=json.loads((runner.task_dir(suite,task)/'status.json').read_text())
                self.assertEqual(state['status'],'completed')

    def test_cross_instance_merge_and_conflict_detection(self):
        with TemporaryDirectory() as tmp:
            suite={'suite_id':'combined','storage_root':tmp,'models':['scratch','voco']}
            roots=[Path(tmp)/'server1',Path(tmp)/'server2']
            protocol={'split':'same','seed':12345}
            for root,model in zip(roots,['scratch','voco']):
                file=root/'evaluation_no_tta'/model/'final'/'summary.json'
                runner.write_json(file,dict(model=model,checkpoint='final',comparison_protocol=protocol,dice_percent=90))
            self.assertEqual(len(runner.write_comparison(suite,roots)),2)
            file=roots[1]/'evaluation_no_tta/voco/final/summary.json'
            row=json.loads(file.read_text()); row['comparison_protocol']['seed']=999
            runner.write_json(file,row)
            with self.assertRaisesRegex(ValueError,'conditions differ'):
                runner.write_comparison(suite,roots)

    def test_notebook_is_clean_and_cells_compile(self):
        nb=json.loads((REPO/'experiments/acdc/notebooks/08_noaug_swin_voco.ipynb').read_text())
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                self.assertEqual(cell['outputs'],[])
                self.assertIsNone(cell['execution_count'])
                compile(''.join(cell['source']),'<notebook>','exec')


if __name__=='__main__': unittest.main()
