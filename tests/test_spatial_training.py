import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.spatial_split import overlapping_ids, validate_manifest
from src.utils.training_state import atomic_save, rng_state, restore_rng


class SplitTests(unittest.TestCase):
    def test_overlap_edge_and_containment(self):
        val=[dict(id='v',bounds=[0,0,10,10])]
        rows=[dict(id='edge',bounds=[10,0,20,10]),dict(id='inside',bounds=[1,1,2,2]),dict(id='cross',bounds=[9,9,20,20])]
        self.assertEqual(overlapping_ids(rows,val),{'inside','cross'})

    def test_manifest_rejects_overlap_and_test_contamination(self):
        rows=[dict(id='a',region='a',crs='EPSG:32617',split='train',bounds=[0,0,1,1]),
              dict(id='b',region='a',crs='EPSG:32617',split='validation',bounds=[2,0,3,1]),
              dict(id='c',region='c',crs='EPSG:32617',split='test',bounds=[0,0,1,1])]
        validate_manifest(dict(patches=rows))
        bad=copy.deepcopy(rows);bad[1]['bounds']=[0,0,1,1]
        with self.assertRaises(ValueError): validate_manifest(dict(patches=bad))
        bad=copy.deepcopy(rows);bad[2]['region']='a'
        with self.assertRaises(ValueError): validate_manifest(dict(patches=bad))

    def test_rng_roundtrip_and_atomic_save(self):
        state=rng_state()
        expected=(random.random(),np.random.rand(),torch.rand(3))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.pth';atomic_save(state,path)
            restore_rng(torch.load(path,weights_only=False))
            self.assertEqual(expected[0],random.random())
            self.assertEqual(expected[1],np.random.rand())
            torch.testing.assert_close(expected[2],torch.rand(3),rtol=0,atol=0)
            self.assertFalse(path.with_suffix('.pth.tmp').exists())

    def test_interrupted_training_matches_uninterrupted(self):
        from scripts import main
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, **kwargs): self.samples=[{'tile_id':str(i)} for i in range(4)]
            def __len__(self): return 4
            def __getitem__(self,i):
                x=torch.ones(1,4,4)*(i+1)/10 + random.random()/100
                return dict(lidar=x,s2=x,attrs=torch.ones(1),mask=torch.ones_like(x))
        class Model(torch.nn.Module):
            def __init__(self,**kwargs): super().__init__();self.weight=torch.nn.Parameter(torch.randn(()))
            def forward(self,noisy,s2,attrs,t): return noisy*self.weight
        import yaml
        cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'config.yaml').read_text())
        cfg['training'].update(epochs=2,num_workers=0,batch_size=2)
        cfg['system']['device']='cpu';cfg['logging']['wandb_name']='test'
        cfg['data'].update(s2_dirs=[],lidar_dirs=[])
        cfg['evaluation']['evaluate']=False
        with tempfile.TemporaryDirectory() as d, patch.object(main,'LidarS2Dataset',Dataset),patch.object(main,'ConditionalUNet',Model),patch.object(main.wandb,'init'),patch.object(main.wandb,'log'):
            cfg['logging']['save_dir']=str(Path(d)/'full');main.set_seed(42);main.train_model(copy.deepcopy(cfg))
            full=torch.load(Path(cfg['logging']['save_dir'])/'test_latest.pth',weights_only=False)
            cfg['logging']['save_dir']=str(Path(d)/'resumed');main.set_seed(42)
            def interrupt(state,path):
                atomic_save(state,path)
                if str(path).endswith('_latest.pth') and state['epoch']==1: raise InterruptedError()
            with patch.object(main,'atomic_save',interrupt),self.assertRaises(InterruptedError): main.train_model(copy.deepcopy(cfg))
            main.set_seed(999);main.train_model(copy.deepcopy(cfg),resume='auto')
            actual=torch.load(Path(cfg['logging']['save_dir'])/'test_latest.pth',weights_only=False)
            self.assertEqual(full['train_losses'],actual['train_losses'])
            self.assertEqual(full['best_val_loss'],actual['best_val_loss'])
            torch.testing.assert_close(full['model_state_dict']['weight'],actual['model_state_dict']['weight'],rtol=0,atol=0)

if __name__=='__main__': unittest.main()
