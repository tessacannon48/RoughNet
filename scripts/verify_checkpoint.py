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
 p.add_argument('--registry',type=Path,default=ROOT/'release/checkpoints.json')
 p.add_argument('--manifest',type=Path)
 p.add_argument('--data-root',type=Path,default=ROOT/'input_data')
 p.add_argument('--verify-data',action='store_true');p.add_argument('--smoke-inference',action='store_true')
 p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
 a=p.parse_args()
 from src.utils.spatial_split import sha256_file,validate_manifest,portable_digest,verify_checkpoint_split
 checksum=sha256_file(a.checkpoint)
 records=json.loads(a.registry.read_text())['checkpoints'];record=next((r for r in records if r['sha256']==checksum),None)
 if record is None:p.error('Weights checksum is not in the publication registry')
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
  saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=saved['config']
  if a.verify_data:verify_checkpoint_split(cfg,manifest,a.checkpoint,a.registry)
  if saved['epoch']!=record['epoch']:p.error('Checkpoint epoch differs from registry')
  model=ConditionalUNet(in_channels=1,cond_channels=24,attr_dim=48,cond_k=6,**cfg['model']).to(a.device).eval()
  model.load_state_dict(saved['model_state_dict'],strict=True)
  ds=LidarS2Dataset(str(ROOT/'input_data/sample_lidar_patch'),str(ROOT/'input_data/sample_s2_patch'),context_k=6,randomize_context=False,augment=False,split='test',target_s2_hw=(32,32))
  sample=ds[0]
  with torch.no_grad():
   prediction=model(torch.zeros(1,1,32,32,device=a.device),sample['s2'][None].to(a.device),sample['attrs'][None].to(a.device),torch.tensor([999],device=a.device))
  if prediction.shape!=(1,1,32,32) or not torch.isfinite(prediction).all():p.error('Invalid smoke inference output')
  report['smoke_inference']=dict(device=a.device,shape=list(prediction.shape),scope='single forward pass; not full diffusion sampling')
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
