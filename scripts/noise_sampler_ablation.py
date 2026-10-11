"""Prepare/run/resume/summarize the corrected 2 schedules x 3 samplers study.
Run from Dissertation/. Validation and test are explicit separate commands.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import default_collate
from src.data.dataset import LidarS2Dataset
from src.model.unet import ConditionalUNet
from src.diffusion.scheduler import LinearDiffusionScheduler, CosineDiffusionScheduler
from src.diffusion.sampling import p_sample_loop_ddpm, p_sample_loop_ddim, p_sample_loop_plms
from src.utils.conditioning_ablation import METRICS, stable_seed, paired_noise, patch_metrics, sha256_file, json_digest, write_json, save_predictions
from src.utils.spatial_split import validate_manifest, digest

SCHEDULES = ('linear', 'cosine')
SAMPLERS = ('ddpm', 'ddim', 'plms')


def sources():
    paths = [Path(__file__), ROOT/'scripts/evaluation.py', *ROOT.joinpath('src').rglob('*.py')]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def prepare(args):
    out = args.out.resolve()
    if (out/'manifest.json').exists():
        raise FileExistsError('Study already prepared; reuse it or choose another output directory')
    split = json.loads(args.manifest.read_text())
    validate_manifest(split, ROOT)
    configurations, checkpoints = {}, {}
    for schedule, path in [('linear', args.linear), ('cosine', args.cosine)]:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        cfg = saved['config']
        if cfg['training']['noise_schedule'] != schedule or cfg['training']['timesteps'] != 1000:
            raise ValueError('Checkpoint schedule/timestep mismatch')
        if cfg['data'].get('split_digest') != digest(split):
            raise ValueError('Checkpoint was not trained on the specified corrected split')
        configurations[schedule] = cfg
        checkpoints[schedule] = dict(path=str(path.resolve()), sha256=sha256_file(path), epoch=saved['epoch'], val_loss=saved['val_loss'])
    linear, cosine = configurations['linear'], configurations['cosine']
    if linear['model'] != cosine['model'] or linear['data'] != cosine['data']:
        raise ValueError('Models or training data differ between checkpoints')
    if {k:v for k,v in linear['training'].items() if k!='noise_schedule'} != {k:v for k,v in cosine['training'].items() if k!='noise_schedule'}:
        raise ValueError('Training settings differ beyond schedule')
    cohorts = {}
    for region in ('pondinlet', 'tuk', 'cambridge'):
        role = 'test' if region == 'cambridge' else 'validation'
        rows = [r for r in split['patches'] if r['region']==region and r['split']==role]
        if args.max_patches and len(rows)>args.max_patches:
            ix = sorted(np.random.default_rng(stable_seed('subset', region, 2026)).choice(len(rows),args.max_patches,replace=False))
            rows = [rows[i] for i in ix]
        if not rows:
            raise ValueError(f'Empty cohort: {region}')
        cohorts[region] = dict(split=role, records=rows, directories=split['directories'][region])
    write_json(out/'manifest.json', dict(version=1, checkpoints=checkpoints, configurations=configurations,
               split_digest=digest(split), cohorts=cohorts, sources=sources(), seed=args.seed,
               batch_size=args.batch_size, max_patches=args.max_patches,
               protocol='paper metrics; all 1000 timesteps; DDIM eta=0; PLMS order=4; paired initial noise; batch-seeded DDPM innovations'))
    print('Prepared', out, 'cohorts:', {r:len(c['records']) for r,c in cohorts.items()}, flush=True)
    print('Checkpoints:', checkpoints, flush=True)


def load_manifest(out):
    return json.loads((out/'manifest.json').read_text())


def shard_path(out, schedule, sampler, region, start):
    return out/'shards'/f'{schedule}_{sampler}_{region}_{start:06d}.json'


def valid_shard(path, manifest, schedule, sampler, region, ids, out):
    shard = json.loads(path.read_text())
    if shard['manifest_digest'] != json_digest(manifest) or shard['ids'] != ids:
        raise ValueError(f'Shard provenance differs: {path}')
    if len(shard['rows']) != len(ids) or [r['tile_id'] for r in shard['rows']] != ids:
        raise ValueError(f'Shard missing metrics: {path}')
    for row in shard['rows']:
        if (row['schedule'],row['sampler'],row['region']) != (schedule,sampler,region):
            raise ValueError(f'Wrong shard identity: {path}')
        if any(row[k] is None or not np.isfinite(row[k]) for k in METRICS):
            raise ValueError(f'Invalid metrics: {path}')
    if len(shard['archives']) != len(ids):
        raise ValueError('Missing archives')
    for a, pid in zip(shard['archives'], ids):
        expected = f'predictions/{schedule}/{sampler}/{region}/{pid}.npz'
        if a['path'] != expected or sha256_file(out/a['path']) != a['sha256']:
            raise ValueError(f'Archive changed or missing: {a}')
    return shard


def run(args):
    out = args.out.resolve(); m = load_manifest(out)
    if sources() != m['sources']:
        raise ValueError('Source code changed since prepare; make a new study')
    regions = [r for r,c in m['cohorts'].items() if c['split']==args.split]
    if args.regions:
        if set(args.regions)-set(regions): raise ValueError('Requested region belongs to another split')
        regions = args.regions
    for r in regions:
        for row in m['cohorts'][r]['records']:
            for f in row['files']:
                st=(ROOT/f['path']).stat()
                if (st.st_size, st.st_mtime_ns)!=(f['size'], f['mtime_ns']):
                    raise ValueError(f"Input changed: {f['path']}")
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
                 cudnn=torch.backends.cudnn.version(),device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                 cpu_threads=4,deterministic=True,tf32=False)
    if (out/'runtime.json').exists() and json.loads((out/'runtime.json').read_text())!=runtime:
        raise ValueError('Runtime differs from existing study')
    write_json(out/'runtime.json',runtime)
    completed=0
    for schedule in args.schedules:
        checkpoint=m['checkpoints'][schedule]
        if sha256_file(checkpoint['path'])!=checkpoint['sha256']: raise ValueError('Checkpoint changed')
        cfg=m['configurations'][schedule]
        saved=torch.load(checkpoint['path'],map_location='cpu',weights_only=False)
        model=ConditionalUNet(in_channels=1,cond_channels=24,attr_dim=48,cond_k=6,**cfg['model']).to(device).eval()
        model.load_state_dict(saved['model_state_dict'],strict=True);del saved
        scheduler=(LinearDiffusionScheduler if schedule=='linear' else CosineDiffusionScheduler)(timesteps=1000,device=device)
        for region in regions:
            cohort=m['cohorts'][region];ids=[r['id'] for r in cohort['records']]
            ds=LidarS2Dataset(str(ROOT/cohort['directories']['lidar']),str(ROOT/cohort['directories']['s2']),context_k=6,
                              randomize_context=False,augment=False,split='val',split_pids=ids)
            if [s['tile_id'] for s in ds.samples]!=ids: raise ValueError('Dataset omitted/reordered prepared IDs')
            for start in range(0,len(ids),m['batch_size']):
                target_ids=ids[start:start+m['batch_size']]
                pending=[]
                for sampler in args.samplers:
                    path=shard_path(out,schedule,sampler,region,start)
                    if path.exists(): valid_shard(path,m,schedule,sampler,region,target_ids,out)
                    else: pending.append(sampler)
                if not pending: continue
                target=default_collate([ds[i] for i in range(start,start+len(target_ids))])
                noise=paired_noise(region,target_ids,m['seed'],(1,256,256)).to(device)
                for sampler in pending:
                    path=shard_path(out,schedule,sampler,region,start)
                    before=time.monotonic()
                    with torch.no_grad():
                        kw=dict(initial_noise=noise)
                        if sampler=='ddpm':
                            kw['generator']=torch.Generator(device=device).manual_seed(stable_seed('ddpm_innovations',region,target_ids,m['seed']))
                        function={'ddpm':p_sample_loop_ddpm,'ddim':p_sample_loop_ddim,'plms':p_sample_loop_plms}[sampler]
                        pred=function(model,scheduler,noise.shape,target['s2'].to(device),target['attrs'].to(device),device,**kw).cpu()
                    rows,archives=[],[]
                    for i,pid in enumerate(target_ids):
                        values=patch_metrics(target['lidar'][i],pred[i],target['mask'][i]>0.5,cfg)
                        if any(v is None or not np.isfinite(v) for v in values.values()): raise ValueError('Non-finite metrics')
                        rows.append(dict(schedule=schedule,sampler=sampler,region=region,split=cohort['split'],tile_id=pid,**values))
                        rel=f'predictions/{schedule}/{sampler}/{region}/{pid}.npz'
                        save_predictions(out/rel,target['lidar'][i,0],target['mask'][i]>0.5,{'prediction':pred[i,0]},target['lidar_patch_mean'][i])
                        archives.append(dict(path=rel,sha256=sha256_file(out/rel)))
                    write_json(path,dict(manifest_digest=json_digest(m),ids=target_ids,rows=rows,archives=archives,elapsed_s=time.monotonic()-before))
                    print(f'{schedule}/{sampler}/{region}: {start+len(target_ids)}/{len(ids)} ({time.monotonic()-before:.1f}s)',flush=True)
                    completed+=1
                    if args.max_batches and completed>=args.max_batches: return
        del model


def summarize(args):
    out=args.out.resolve();m=load_manifest(out);rows=[]
    for schedule in SCHEDULES:
        for sampler in SAMPLERS:
            for region,c in m['cohorts'].items():
                if c['split']!=args.split: continue
                ids=[r['id'] for r in c['records']]
                for start in range(0,len(ids),m['batch_size']):
                    p=shard_path(out,schedule,sampler,region,start)
                    if not p.exists(): raise ValueError(f'Incomplete study: missing {p}')
                    rows.extend(valid_shard(p,m,schedule,sampler,region,ids[start:start+m['batch_size']],out)['rows'])
    dest=out/args.split;dest.mkdir(exist_ok=True)
    with (dest/'per_patch.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=sorted({r['region'] for r in rows})
    if args.split=='validation': groups+=['pooled_validation']
    summaries=[]
    for region in groups:
        for schedule in SCHEDULES:
            for sampler in SAMPLERS:
                cohort=[r for r in rows if r['schedule']==schedule and r['sampler']==sampler and (region=='pooled_validation' or r['region']==region)]
                row=dict(region=region,schedule=schedule,sampler=sampler,n=len(cohort))
                for metric in METRICS:
                    values=np.asarray([r[metric] for r in cohort]);row[metric+'_mean']=float(values.mean());row[metric+'_std']=float(values.std())
                summaries.append(row)
    with (dest/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    write_json(dest/'comparison.json',summaries)
    lines=['# Noise schedule × sampler comparison','',f'Existing paper metrics; split: {args.split}. Standard deviations use population convention. Pooled validation weights each patch equally.','']
    for region in groups:
        lines += [f'## {region}','', '| Schedule | Sampler | N | '+' | '.join(METRICS)+' |','|---|---|---:|'+ '|'.join(['---:']*len(METRICS))+'|']
        for row in summaries:
            if row['region']==region: lines.append(f"| {row['schedule']} | {row['sampler']} | {row['n']} | "+' | '.join(f"{row[k+'_mean']:.4f} ± {row[k+'_std']:.4f}" for k in METRICS)+' |')
        lines.append('')
    (dest/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('Wrote',dest/'comparison.csv','and comparison.md',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--out',type=Path,default=Path('reconstructions/noise_sampler_clean'))
    p.add_argument('--manifest',type=Path,default=Path('splits/original_clean.json'))
    p.add_argument('--linear',type=Path,default=Path('models/original_clean_linear/original_clean_linear_best.pth'))
    p.add_argument('--cosine',type=Path,default=Path('models/original_clean/original_clean_best.pth'))
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--max-patches',type=int)
    p=sub.add_parser('run');p.add_argument('--out',type=Path,default=Path('reconstructions/noise_sampler_clean'))
    p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--device',default='cuda')
    p.add_argument('--schedules',nargs='+',choices=SCHEDULES,default=list(SCHEDULES));p.add_argument('--samplers',nargs='+',choices=SAMPLERS,default=list(SAMPLERS))
    p.add_argument('--regions',nargs='+');p.add_argument('--max-batches',type=int)
    p=sub.add_parser('summarize');p.add_argument('--out',type=Path,default=Path('reconstructions/noise_sampler_clean'))
    p.add_argument('--split',choices=['validation','test'],default='validation')
    args=parser.parse_args()
    if args.command=='prepare' and (args.batch_size<1 or (args.max_patches is not None and args.max_patches<1)): parser.error('Counts must be positive')
    if args.command=='run':
        import fcntl
        args.out.mkdir(parents=True,exist_ok=True)
        with (args.out/'run.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)
    else: {'prepare':prepare,'summarize':summarize}[args.command](args)

if __name__=='__main__': main()
