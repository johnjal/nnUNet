import ast
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

REPO=Path(__file__).resolve().parents[3]
SCRIPT=REPO/'experiments/acdc/analysis/reevaluate_augmentation_no_tta.py'
sys.path.insert(0,str(SCRIPT.parent))
spec=importlib.util.spec_from_file_location('reeval_test',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ReevaluationTests(unittest.TestCase):
    def fixture(self,root,theme):
        run=Path(root)/'runs'/theme
        trainer=m.TRAINERS[theme]
        model=run/'nnUNet_results/Dataset027_ACDC'/f'{trainer}__nnUNetPlans__3d_fullres'
        (model/'fold_0').mkdir(parents=True)
        config=dict(trainer=trainer,dataset_id=27,plans='nnUNetPlans',configuration='3d_fullres')
        (run/'config.snapshot.json').write_text(json.dumps({'configuration':config}))
        split={part:[f'patient{i:03d}_frame{frame:02d}' for i in ids for frame in (1,2)]
               for part,ids in [('train',range(1,81)),('val',range(81,101))]}
        (run/'splits_final.json').write_text(json.dumps([split]))
        (model/'dataset.json').write_text(json.dumps({'labels':{'background':0,'RV':1,'MLV':2,'LVC':3}}))
        for kind in ('best','final'):(model/'fold_0'/f'checkpoint_{kind}.safetensors').touch()
        return run,split

    def test_selects_both_checkpoints_and_rejects_split_mismatch(self):
        with TemporaryDirectory() as tmp:
            self.fixture(tmp,'none');run,split=self.fixture(tmp,'quality')
            c={'storage_root':tmp,'runs':{'none':'none','quality':'quality'},'checkpoints':['final','best']}
            records,cases=m.models_and_cases(c)
            self.assertEqual(len(records),4);self.assertEqual(len(cases),40)
            self.assertEqual({r['kind'] for r in records},{'final','best'})
            split['train'][0],split['val'][0]=split['val'][0],split['train'][0]
            (run/'splits_final.json').write_text(json.dumps([split]))
            with self.assertRaises(ValueError):m.models_and_cases(c)

    def test_inference_disables_mirroring_and_gpu_accumulation(self):
        tree=ast.parse(SCRIPT.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
               and n.func.id=='nnUNetPredictor']
        self.assertEqual(len(calls),1)
        kwargs={k.arg:ast.literal_eval(k.value) for k in calls[0].keywords if k.arg in
                ('use_mirroring','perform_everything_on_device')}
        self.assertEqual(kwargs,{'use_mirroring':False,'perform_everything_on_device':False})
        self.assertNotIn('nnUNetv2_train',SCRIPT.read_text())

    def test_clean_notebook_compiles(self):
        p=REPO/'experiments/acdc/notebooks/09_augmentation_reevaluate_no_tta.ipynb'
        for c in json.loads(p.read_text())['cells']:
            if c['cell_type']=='code':
                compile(''.join(c['source']),str(p),'exec')
                self.assertEqual(c['outputs'],[])

if __name__=='__main__':unittest.main()
