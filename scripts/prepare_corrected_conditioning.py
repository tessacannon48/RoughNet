"""Prepare conditioning controls only after checking the corrected training split."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts import conditioning_ablation as engine
from src.utils.spatial_split import validate_manifest,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('reconstructions/conditioning_ablation_clean'))
    args=p.parse_args()
    split=json.loads((engine.ROOT/'splits/original_clean.json').read_text())
    validate_manifest(split,engine.ROOT)
    checkpoint=engine.ROOT/'models/original_clean/original_clean_best.pth'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['config']['data'].get('split_digest')!=digest(split):raise ValueError('Checkpoint differs from corrected split')
    del saved
    engine.prepare(argparse.Namespace(out=args.out,checkpoint=checkpoint,data_root=engine.ROOT/'input_data',seeds=[42],max_patches=None))
    m=json.loads((args.out/'manifest.json').read_text())
    for region,spec in m['regions'].items():
        role='test' if region=='cambridge' else 'validation'
        expected={r['id'] for r in split['patches'] if r['region']==region and r['split']==role}
        if {r['tile_id'] for r in spec['records']}!=expected:raise ValueError('Conditioning cohort differs from corrected split')
    print('Corrected checkpoint and all validation/test conditioning cohorts verified.',flush=True)

if __name__=='__main__':main()
