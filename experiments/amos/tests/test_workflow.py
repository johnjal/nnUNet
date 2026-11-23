import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('amos_workflow',REPO/'experiments/amos/scripts/workflow.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def config(root):
    return dict(storage_root=str(root),run_id='test-fp32',gpu='0',amp=False,
                batch_size=2,planner_gb=8,workers=1,seed=12345,evaluate_best=True)


class WorkflowTests(unittest.TestCase):
    def test_precision_changes_trainers_not_preprocessing(self):
        c=config('/tmp/amos-example')
        d=dict(c,amp=True,run_id='test-amp')
        self.assertEqual(w.locations(c)['pre'],w.locations(d)['pre'])
        self.assertEqual(w.locations(c)['plans'],w.locations(d)['plans'])
        self.assertNotEqual(w.model_dir(c,'off'),w.model_dir(d,'off'))
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'config.json';w.frozen(f,c)
            with self.assertRaises(ValueError):w.frozen(f,d)

    def test_prepare_official_split_links_and_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=config(tmp);p=w.locations(c)
            files=[]
            for split,ids in [('Tr',range(507,547)),('Va',range(547,567))]:
                for folder in ['images'+split,'labels'+split]:
                    for i in ids:
                        f=p['source']/folder/f'amos_{i:04d}.nii.gz'
                        f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(str(i).encode())
                        files.append(dict(path=str(f.relative_to(p['root'])),sha256=w.sha(f)))
            w.write(p['source']/'dataset.json',{'labels':{'0':'background','1':'organ'}})
            w.write(p['root']/'download_manifest.json',dict(counts={'imagesTr':40,'labelsTr':40,'imagesVa':20,'labelsVa':20},files=files))
            def cli(c,name,*args):
                if name=='nnUNetv2_plan_and_preprocess':
                    w.write(p['pre']/f"{p['plans']}.json",{'configurations':{'3d_fullres':{'batch_size':9,'patch_size':[16]*3,'spacing':[1]*3}}})
            with patch.object(w,'cli',cli),patch.object(w,'free_space'):
                w.prepare(c)
            split=w.read(p['pre']/'splits_final.json')[0]
            self.assertEqual((len(split['train']),len(split['val'])),(40,20))
            self.assertFalse(set(split['train']) & set(split['val']))
            f=p['raw']/'imagesTr/amos_0507_0000.nii.gz'
            self.assertEqual(f.stat().st_ino,(p['source']/'imagesTr/amos_0507.nii.gz').stat().st_ino)
            self.assertEqual(w.read(p['raw']/'dataset.json')['channel_names'],{'0':'MRI'})
            self.assertEqual(w.read(p['pre']/f"{p['plans']}.json")['configurations']['3d_fullres']['batch_size'],2)

    def test_prepare_50_10_isolated_and_off_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=dict(config(tmp),split_mode='train50_val10',split_seed=12345);p=w.locations(c)
            self.assertEqual(w.themes(c),('off',))
            self.assertNotEqual(p['pre'],w.locations(config(tmp))['pre'])
            files=[]
            for split,ids in [('Tr',range(507,547)),('Va',range(547,567))]:
                for folder in ['images'+split,'labels'+split]:
                    for i in ids:
                        f=p['source']/folder/f'amos_{i:04d}.nii.gz'
                        f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(str(i).encode())
                        files.append(dict(path=str(f.relative_to(p['root'])),sha256=w.sha(f)))
            w.write(p['source']/'dataset.json',{'labels':{'0':'background','1':'organ'}})
            w.write(p['root']/'download_manifest.json',dict(counts={'imagesTr':40,'labelsTr':40,'imagesVa':20,'labelsVa':20},files=files))
            def cli(c,name,*args):
                if name=='nnUNetv2_plan_and_preprocess':
                    w.write(p['pre']/f"{p['plans']}.json",{'configurations':{'3d_fullres':{'batch_size':9,'patch_size':[16]*3,'spacing':[1]*3}}})
            with patch.object(w,'cli',cli),patch.object(w,'free_space'):
                w.prepare(c)
            split=w.read(p['pre']/'splits_final.json')[0]
            self.assertEqual((len(split['train']),len(split['val'])),(50,10))
            self.assertTrue({f'amos_{i:04d}' for i in range(507,547)} <= set(split['train']))
            self.assertFalse(set(split['train']) & set(split['val']))
            f=p['raw']/'imagesTr/amos_0507_0000.nii.gz'
            self.assertEqual(f.stat().st_ino,(p['source']/'imagesTr/amos_0507.nii.gz').stat().st_ino)
            self.assertEqual(w.read(p['raw']/'dataset.json')['channel_names'],{'0':'MRI'})
            self.assertEqual(w.read(p['pre']/f"{p['plans']}.json")['configurations']['3d_fullres']['batch_size'],2)

    def test_queue_probes_both_before_training_and_stops_on_failure(self):
        import torch
        for fail in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                c=config(tmp);events=[]
                def call(c,command):
                    stage='smoke' if '--smoke' in command else 'best'
                    events.append((stage,command[-1]))
                    if fail and stage=='smoke':
                        raise RuntimeError('synthetic CUDA OOM')
                def cli(c,name,*args):
                    events.append(('train',args[4]))
                with patch.object(w.sys,'platform','linux'), patch.object(w.signal,'signal'), \
                     patch.dict(os.environ,{},clear=False), patch.object(torch,'__version__','2.8.0+cu128'), \
                     patch.object(torch.cuda,'is_available',return_value=True), \
                     patch.object(torch.cuda,'get_device_name',return_value='test GPU'), \
                     patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(total_memory=24*2**30)), \
                     patch.object(w.importlib.metadata,'version',side_effect=lambda name:'0.3.8' if name=='batchgeneratorsv2' else '0.4.4'), \
                     patch.object(w,'prepare'),patch.object(w,'recipe'),patch.object(w,'free_space'), \
                     patch.object(w,'call',call),patch.object(w,'cli',cli),patch.object(w,'comparison'):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'OOM'):w.run(c)
                        self.assertEqual(events,[('smoke','off')])
                    else:
                        w.run(c)
                        self.assertEqual(events[:2],[('smoke','off'),('smoke','on')])
                        self.assertEqual([e[0] for e in events],['smoke','smoke','train','best','train','best'])
                self.assertEqual(w.read(w.locations(c)['suite']/'status.json')['status'], 'failed' if fail else 'completed')

    def test_notebook_clean_and_valid(self):
        nb=json.loads((REPO/'experiments/amos/notebooks/01_mri_augmentation_comparison.ipynb').read_text())
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'],[])


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.torch=torch
        import nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerAMOS250 as module
        cls.m=module

    def test_classes_discoverable_and_defaults(self):
        from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
        def init(t,*args,**kwargs):t.grad_scaler='enabled'
        for theme in ('off','on'):
            for amp in (False,True):
                name=w.trainer(dict(config('/tmp/amos'),amp=amp),theme)
                cls=recursive_find_trainer_class_by_name(name)
                with patch.object(self.m.nnUNetTrainer,'__init__',init):
                    t=cls({},'3d_fullres',0,{},self.torch.device('cpu'))
                self.assertEqual(t.num_epochs,250)
                self.assertTrue(t.enable_deep_supervision)
                self.assertEqual(t.use_amp,amp)
                self.assertEqual(t.grad_scaler,'enabled' if amp else None)

    def test_no_aug_preserves_values_and_ds(self):
        torch=self.torch
        transform=self.m.nnUNetTrainerAMOSNoAugFP32.get_training_transforms(
            [8,16,16],(-.3,.3),[[1,1,1],[.5,.5,.5]],(0,1,2),False)
        image=torch.randn(1,8,16,16)
        seg=torch.randint(0,16,(1,8,16,16),dtype=torch.int16)
        for _ in range(3):
            out=transform(image=image.clone(),segmentation=seg.clone())
            self.assertTrue(torch.equal(out['image'],image))
            self.assertTrue(torch.equal(out['segmentation'][0],seg))
            self.assertEqual(list(out['segmentation'][1].shape),[1,4,8,8])

    def test_real_unet_ds_train_validation(self):
        torch=self.torch
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.utilities.label_handling.label_handling import LabelManager
        plan=json.loads((REPO/'experiments/acdc/docs/artifacts/runs/acdc-baseline-v1/nnUNetPlans.json').read_text())
        conf=plan['configurations']['3d_fullres'];conf['patch_size']=[16,32,32]
        kw=conf['architecture']['arch_kwargs']
        kw.update(n_stages=3,features_per_stage=[4,8,16],kernel_sizes=[[3]*3]*3,
                  strides=[[1]*3,[2]*3,[2]*3],n_conv_per_stage=[2]*3,n_conv_per_stage_decoder=[2]*2)
        pm=PlansManager(plan);cm=pm.get_configuration('3d_fullres')
        for use_amp in (False,True):
            cls=self.m.nnUNetTrainerAMOSAugAMP if use_amp else self.m.nnUNetTrainerAMOSAugFP32
            t=cls.__new__(cls)
            t.device=torch.device('cpu');t.is_ddp=False;t.enable_deep_supervision=True
            t.configuration_manager=cm
            t.label_manager=LabelManager({'background':0,**{f'organ{i}':i for i in range(1,16)}},regions_class_order=None)
            t.grad_scaler=None;t._do_i_compile=lambda:False
            t.network=t.build_network_architecture(pm,cm,1,16,True)
            t.loss=t._build_loss()
            t.optimizer=torch.optim.SGD(t.network.parameters(),lr=.01)
            x=torch.randn(2,1,16,32,32)
            y=torch.randint(0,16,(2,1,16,32,32))
            targets=[y,torch.nn.functional.interpolate(y.float(),size=(8,16,16),mode='nearest').long()]
            before=next(t.network.parameters()).detach().clone()
            out=t.train_step({'data':x,'target':targets})
            self.assertTrue(__import__('numpy').isfinite(out['loss']))
            self.assertFalse(torch.equal(before,next(t.network.parameters())))
            t.network.eval()
            with torch.no_grad():v=t.validation_step({'data':x,'target':targets})
            self.assertEqual(len(v['tp_hard']),15)

    def test_predictor_fp32_padding_tta_and_overlap(self):
        torch=self.torch
        class Identity(torch.nn.Module):
            def forward(self,x):return torch.cat([x,-x],dim=1)
        for cls in (self.m.AMOSPredictor,self.m.AMOSAMPPredictor):
            pred=cls(device=torch.device('cpu'),allow_tqdm=False)
            pred.network=Identity();pred.configuration_manager=SimpleNamespace(patch_size=[8,8,8])
            pred.label_manager=SimpleNamespace(num_segmentation_heads=2)
            pred.allowed_mirroring_axes=(0,1,2)
            for shape in [(1,5,6,7),(1,12,13,14)]:
                x=torch.randn(shape)
                out=pred.predict_sliding_window_return_logits(x)
                self.assertEqual(out.dtype,torch.float32)
                self.assertTrue(torch.allclose(out,torch.cat([x,-x]),atol=1e-5))

    def test_final_validation_uses_precision_predictor_and_restores(self):
        m=self.m
        def fake(t,save):
            module=__import__('importlib').import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer')
            return module.nnUNetPredictor
        with patch.object(m.nnUNetTrainer,'perform_actual_validation',fake):
            t=m.nnUNetTrainerAMOSAugFP32.__new__(m.nnUNetTrainerAMOSAugFP32)
            self.assertIs(t.perform_actual_validation(),m.AMOSPredictor)
            t=m.nnUNetTrainerAMOSAugAMP.__new__(m.nnUNetTrainerAMOSAugAMP)
            self.assertIs(t.perform_actual_validation(),m.AMOSAMPPredictor)


if __name__=='__main__':unittest.main()
