#!/usr/bin/env python3
"""Run from the repository root. Audit paired raster bounds and prepare geographic folds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys
import rasterio
from rasterio.warp import transform_bounds
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.spatial_split import overlapping_ids, validate_manifest, sha256_file
REGIONS = ('pondinlet','tuk','cambridge')
FOLDS = {
    'original_clean': ({'pondinlet': 4, 'tuk': 13}, 'cambridge'),
    'tuk_heldout_clean': ({'pondinlet': 4, 'cambridge': 4}, 'tuk'),
}
FOOTPRINT_RULE = 'Union bounding box of LiDAR and all six S2 views in regional LiDAR CRS'


def inspect_patch(item):
    region, folder = item
    data_root = folder.parents[1]
    pid = folder.name.removeprefix('s2_patch_')
    lidar = data_root / f'lidar_patches_{region}/lidar_patch_{pid}.tif'
    paths = [lidar] + [folder/f't{i}.tif' for i in range(6)]
    with rasterio.open(lidar) as ds:
        crs = ds.crs
        if crs is None or not crs.is_projected or ds.shape!=(256,256) or ds.count!=2:
            raise ValueError(f'Unexpected LiDAR geometry: {lidar}')
    bounds=[]
    for path in paths:
        with rasterio.open(path) as ds:
            if ds.crs is None or ds.transform.b or ds.transform.d:
                raise ValueError(f'Missing CRS or rotated raster: {path}')
            if path!=lidar and (ds.count!=4 or ds.shape!=(26,26)):
                raise ValueError(f'Unexpected S2 shape: {path}')
            bounds.append(list(transform_bounds(ds.crs,crs,*ds.bounds,densify_pts=21)))
    attrs=json.loads((folder/'attrs.json').read_text())
    if len(attrs)!=6 or any(not isinstance(a,dict) for a in attrs):
        raise ValueError(f'Incomplete attributes: {folder}')
    files=[]
    for path in paths+[folder/'attrs.json',folder/'region.json']:
        st=path.stat(); files.append(dict(path=str(Path('input_data') / path.relative_to(data_root)),
                                        size=st.st_size,sha256=sha256_file(path)))
    return dict(region=region,id=pid,zone=json.loads((folder/'region.json').read_text())['region_id'],
                crs=str(crs),bounds=[min(b[0] for b in bounds),min(b[1] for b in bounds),max(b[2] for b in bounds),max(b[3] for b in bounds)],
                view_bounds=bounds,files=files)


def build_manifest(inventory, directories, name):
    zones, test = FOLDS[name]
    rows = copy.deepcopy(inventory)
    for row in rows:
        row['split'] = ('test' if row['region'] == test else
                        'validation' if row['zone'] == zones[row['region']] else 'train')
    for region in zones:
        train = [row for row in rows if row['region'] == region and row['split'] == 'train']
        validation = [row for row in rows if row['region'] == region and row['split'] == 'validation']
        excluded = overlapping_ids(train, validation)
        for row in train:
            if row['id'] in excluded:
                row['split'] = 'excluded'
    manifest = dict(version=1, name=name, validation_zones=zones, test_region=test,
                    footprint_rule=FOOTPRINT_RULE, directories=directories, patches=rows)
    return validate_manifest(manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--out',default='splits')
    p.add_argument('--data-root', type=Path, default=Path('input_data'))
    p.add_argument('--folds', nargs='+', choices=tuple(FOLDS),
                   default=['original_clean', 'tuk_heldout_clean'],
                   help='Prepare only these folds; defaults to the two existing configurations.')
    p.add_argument('--inventory-manifest', type=Path,
                   help='Reuse audited seven-footprint geometry after validating all saved file sizes/timestamps.')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    dirs={r:{k:f'input_data/{k}_patches_{r}' for k in ('lidar','s2')} for r in REGIONS}
    inventory=[]
    if args.inventory_manifest:
        source=json.loads(args.inventory_manifest.read_text())
        if (source.get('version') != 1 or source.get('footprint_rule') != FOOTPRINT_RULE
                or source.get('directories') != dirs
                or {row['region'] for row in source['patches']} != set(REGIONS)):
            raise ValueError('Inventory manifest does not describe the expected seven-footprint inputs')
        validate_manifest(source, Path.cwd(), data_root=args.data_root)
        inventory=copy.deepcopy(source['patches'])
        for row in inventory:
            for record in row['files']:
                relative = Path(record['path']).relative_to('input_data')
                record['sha256'] = sha256_file(args.data_root / relative)
                record.pop('mtime_ns', None)
        print(f'Reusing verified geometry for {len(inventory)} patches', flush=True)
    else:
        for region in REGIONS:
            folders=sorted((args.data_root / f's2_patches_{region}').glob('s2_patch_*'))
            if not folders: raise ValueError(f'No data for {region}; run from Dissertation/')
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for i,row in enumerate(pool.map(inspect_patch,[(region,f) for f in folders]),1):
                    inventory.append(row)
                    if i%500==0: print(f'{region}: {i}/{len(folders)} checked',flush=True)
    base=yaml.safe_load(Path('config.yaml').read_text())
    for name in args.folds:
        zones,test=FOLDS[name]
        m=build_manifest(inventory, dirs, name)
        m['version'] = 2
        validate_manifest(m)
        rows=m['patches']
        path=out/f'{name}_portable.json'
        if path.exists() and json.loads(path.read_text())!=m: raise ValueError(f'Refusing to replace different manifest: {path}')
        path.write_text(json.dumps(m,indent=2)+'\n')
        ids = {region: {split: [r['id'] for r in rows if r['region']==region and r['split']==split]
                        for split in ('train','validation','test','excluded')} for region in REGIONS}
        ids_filename = 'train_val_split_ids.json' if name == 'original_clean' else f'{name}_ids.json'
        (out/ids_filename).write_text(json.dumps(ids,indent=2)+'\n')
        cfg=copy.deepcopy(base)
        cfg['data'].update(train_locations=list(zones),validation_regions=list(zones.values()),split_manifest=str(path),evaluation_patch_ids=[],**{k+'_dirs':[dirs[r][k] for r in zones] for k in ('lidar','s2')})
        cfg['training'].update(batch_size=8,epochs=200,lr=1e-4,timesteps=1000,noise_schedule='cosine',num_workers=16,context_k=6,randomize_context=True,loss=dict(name='masked_hybrid_mse_loss',alpha=1.0))
        cfg['model'].update(unet_depth=4,base_channels=64,embed_dim=256,attention_variant='mid')
        run = {'original_clean':'roughnet_cosine', 'tuk_heldout_clean':'roughnet_cosine_tuk_heldout'}[name]
        cfg['logging'].update(run_name=run,wandb_name=run,save_dir=f'models/{run}',output_dir=f'reconstructions/{run}')
        cfg['evaluation'].update(evaluate=False,sampling_methods=['plms']);cfg['system'].update(device='cuda',debug=False)
        (out/f'{run}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        if name == 'original_clean':
            linear = copy.deepcopy(cfg)
            linear['training']['noise_schedule'] = 'linear'
            linear['logging'].update(run_name='roughnet_linear', wandb_name='roughnet_linear',
                                     save_dir='models/roughnet_linear', output_dir='reconstructions/roughnet_linear')
            (out/'roughnet_linear.yaml').write_text(yaml.safe_dump(linear, sort_keys=False))
        print(name,{s:sum(r['split']==s for r in rows) for s in ('train','validation','test','excluded')},flush=True)
if __name__=='__main__': main()
