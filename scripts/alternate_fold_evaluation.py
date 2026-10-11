"""Fixed cosine/PLMS evaluation of the Pond Inlet + Cambridge Bay training fold.
Reuse the corrected evaluator's batching, metrics, archives and recovery checks.
Run from Dissertation/ and prepare on the machine that will perform inference.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts import noise_sampler_ablation as engine
from src.utils.spatial_split import validate_manifest,digest
from src.utils.conditioning_ablation import write_json,sha256_file

BASE_SOURCES=engine.sources

def sources():
    return {**BASE_SOURCES(), 'scripts/alternate_fold_evaluation.py':sha256_file(__file__)}


def prepare(args):
    if (args.out/'manifest.json').exists(): raise FileExistsError('Reuse existing study or choose another output directory')
    split=json.loads(args.manifest.read_text());validate_manifest(split,engine.ROOT)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=saved['config']
    if cfg['data'].get('split_digest')!=digest(split): raise ValueError('Checkpoint/split mismatch')
    if cfg['training']['noise_schedule']!='cosine' or cfg['training']['timesteps']!=1000: raise ValueError('Expected frozen cosine configuration')
    if split['test_region']!='tuk' or set(split['validation_zones'])!={'pondinlet','cambridge'}: raise ValueError('Wrong geographic fold')
    cohorts={}
    for region in ('pondinlet','cambridge','tuk'):
        role='test' if region=='tuk' else 'validation'
        rows=[r for r in split['patches'] if r['region']==region and r['split']==role]
        cohorts[region]=dict(split=role,records=rows,directories=split['directories'][region])
    write_json(args.out/'manifest.json',dict(version=1,checkpoints={'cosine':dict(path=str(args.checkpoint.resolve()),sha256=sha256_file(args.checkpoint),epoch=saved['epoch'],val_loss=saved['val_loss'])},
        configurations={'cosine':cfg},split_digest=digest(split),cohorts=cohorts,sources=sources(),seed=42,batch_size=8,max_patches=None,
        protocol='Alternate geographic fold; frozen cosine PLMS; all 1000 timesteps; paired initial noise seed 42'))
    print('Prepared alternate fold:',{r:len(c['records']) for r,c in cohorts.items()},flush=True)


def main():
    engine.SCHEDULES=('cosine',);engine.SAMPLERS=('plms',);engine.sources=sources
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','run','summarize'):
        p=sub.add_parser(name);p.add_argument('--out',type=Path,default=Path('reconstructions/tuk_heldout_clean_evaluation'))
        if name=='prepare':
            p.add_argument('--manifest',type=Path,default=Path('splits/tuk_heldout_clean.json'))
            p.add_argument('--checkpoint',type=Path,default=Path('models/tuk_heldout_clean/tuk_heldout_clean_best.pth'))
        else:p.add_argument('--split',choices=['validation','test'],default='test')
        if name=='run':p.add_argument('--device',default='cuda');p.add_argument('--max-batches',type=int)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='summarize':engine.summarize(args)
    else:
        args.schedules=['cosine'];args.samplers=['plms'];args.regions=None
        args.out.mkdir(parents=True,exist_ok=True)
        with (args.out/'run.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);engine.run(args)

if __name__=='__main__':main()
