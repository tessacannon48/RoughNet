"""Prepare conditioning controls only after checking the corrected training split."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts import conditioning_ablation as engine
from src.utils.spatial_split import validate_manifest,verify_checkpoint_split


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('reconstructions/roughnet_conditioning'))
    p.add_argument('--manifest',type=Path,default=Path('splits/original_clean_portable.json'))
    p.add_argument('--data-root',type=Path,default=engine.ROOT/'input_data')
    p.add_argument('--checkpoint-registry',type=Path,default=Path('release/checkpoints.json'))
    args=p.parse_args()
    split=json.loads(args.manifest.read_text())
    validate_manifest(split,engine.ROOT,data_root=args.data_root)
    checkpoint=engine.ROOT/'models/roughnet_cosine/roughnet_cosine_best.pth'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    verify_checkpoint_split(saved['config'],split,checkpoint,args.checkpoint_registry)
    del saved
    engine.prepare(argparse.Namespace(out=args.out,checkpoint=checkpoint,data_root=args.data_root,seeds=[42],max_patches=None))
    m=json.loads((args.out/'manifest.json').read_text())
    for region,spec in m['regions'].items():
        role='test' if region=='cambridge' else 'validation'
        expected={r['id'] for r in split['patches'] if r['region']==region and r['split']==role}
        if {r['tile_id'] for r in spec['records']}!=expected:raise ValueError('Conditioning cohort differs from corrected split')
    print('Corrected checkpoint and all validation/test conditioning cohorts verified.',flush=True)

if __name__=='__main__':main()
