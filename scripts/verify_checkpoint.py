#!/usr/bin/env python3
"""Verify released weights and optionally validate copied inputs or run a bounded forward pass."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',type=Path,default=ROOT/'models/roughnet_cosine/roughnet_cosine_best.pth')
 p.add_argument('--registry',type=Path,default=ROOT/'metadata/checkpoints.json')
 p.add_argument('--manifest',type=Path)
 p.add_argument('--data-root',type=Path,default=ROOT/'input_data')
 p.add_argument('--verify-data',action='store_true');p.add_argument('--smoke-inference',action='store_true')
 p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
 a=p.parse_args()
 from src.utils.spatial_split import sha256_file,validate_manifest,portable_digest,verify_checkpoint_split
 checksum=sha256_file(a.checkpoint)
 records=json.loads(a.registry.read_text())['checkpoints'];record=next((r for r in records if r['sha256']==checksum),None)
 if record is None:p.error('Weights checksum is not in the checkpoint registry')
 report=dict(checkpoint=a.checkpoint.name,sha256=checksum,selected_epoch=record['epoch'],data_verified=False)
 if a.verify_data:
  manifest=json.loads((a.manifest or ROOT/record['manifest']).read_text())
  validate_manifest(manifest,data_root=a.data_root)
  if portable_digest(manifest)!=record['portable_split_digest']:p.error('Dataset identity differs from checkpoint registry')
  report['data_verified']=True
 if a.smoke_inference:
  import torch
  from src.model.unet import ConditionalUNet
  from src.data.dataset import LidarS2Dataset
  if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA is unavailable')
  torch.set_num_threads(1)
  model_type=record.get('model_type','deterministic_unet' if record.get('name')=='roughnet_unet_baseline' else 'diffusion')
  if model_type=='deterministic_unet':
   from src.utils.unet_baseline import load_checkpoint,predict
   model,cfg,saved=load_checkpoint(a.checkpoint,a.device);model.eval()
  elif model_type=='diffusion':
   saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=saved['config']
   k=cfg['training']['context_k']
   model=ConditionalUNet(in_channels=1,cond_channels=4*k,attr_dim=8*k,cond_k=k,**cfg['model']).to(a.device).eval()
   model.load_state_dict(saved['model_state_dict'],strict=True)
  else:p.error('Unknown checkpoint model_type')
  if a.verify_data:verify_checkpoint_split(cfg,manifest,a.checkpoint,a.registry)
  if saved['epoch']!=record['epoch']:p.error('Checkpoint epoch differs from registry')
  ds=LidarS2Dataset(str(ROOT/'input_data/sample_lidar_patch'),str(ROOT/'input_data/sample_s2_patch'),context_k=cfg['training']['context_k'],randomize_context=False,augment=False,split='test',target_s2_hw=(32,32))
  sample=ds[0]
  with torch.no_grad():
   s2,attrs=sample['s2'][None].to(a.device),sample['attrs'][None].to(a.device)
   prediction=predict(model,s2,attrs) if model_type=='deterministic_unet' else model(torch.zeros(1,1,32,32,device=a.device),s2,attrs,torch.tensor([999],device=a.device))
  if prediction.shape!=(1,1,32,32) or not torch.isfinite(prediction).all():p.error('Invalid smoke inference output')
  report['smoke_inference']=dict(device=a.device,shape=list(prediction.shape),model_type=model_type,timestep=0 if model_type=='deterministic_unet' else 999,scope='single forward pass; not full diffusion sampling')
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
