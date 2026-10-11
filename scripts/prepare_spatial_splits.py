#!/usr/bin/env python3
"""Run from Dissertation/. Audit paired raster bounds and prepare both folds."""
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
from src.utils.spatial_split import overlapping_ids, validate_manifest
REGIONS = ('pondinlet','tuk','cambridge')


def inspect_patch(item):
    region, folder = item
    pid = folder.name.removeprefix('s2_patch_')
    lidar = Path(f'input_data/lidar_patches_{region}/lidar_patch_{pid}.tif')
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
        st=path.stat(); files.append(dict(path=str(path),size=st.st_size,mtime_ns=st.st_mtime_ns))
    return dict(region=region,id=pid,zone=json.loads((folder/'region.json').read_text())['region_id'],
                crs=str(crs),bounds=[min(b[0] for b in bounds),min(b[1] for b in bounds),max(b[2] for b in bounds),max(b[3] for b in bounds)],
                view_bounds=bounds,files=files)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--out',default='splits')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    inventory=[]
    for region in REGIONS:
        folders=sorted(Path(f'input_data/s2_patches_{region}').glob('s2_patch_*'))
        if not folders: raise ValueError(f'No data for {region}; run from Dissertation/')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i,row in enumerate(pool.map(inspect_patch,[(region,f) for f in folders]),1):
                inventory.append(row)
                if i%500==0: print(f'{region}: {i}/{len(folders)} checked',flush=True)
    base=yaml.safe_load(Path('config.yaml').read_text())
    dirs={r:{k:f'input_data/{k}_patches_{r}' for k in ('lidar','s2')} for r in REGIONS}
    for name,zones,test in [('original_clean',{'pondinlet':4,'tuk':13},'cambridge'),('tuk_heldout_clean',{'pondinlet':4,'cambridge':4},'tuk')]:
        rows=copy.deepcopy(inventory)
        for r in rows: r['split']='test' if r['region']==test else ('validation' if r['zone']==zones[r['region']] else 'train')
        for region in zones:
            train=[r for r in rows if r['region']==region and r['split']=='train']
            val=[r for r in rows if r['region']==region and r['split']=='validation']
            excluded=overlapping_ids(train,val)
            for r in train:
                if r['id'] in excluded: r['split']='excluded'
        m=dict(version=1,name=name,validation_zones=zones,test_region=test,footprint_rule='Union bounding box of LiDAR and all six S2 views in regional LiDAR CRS',directories=dirs,patches=rows)
        validate_manifest(m);path=out/f'{name}.json'
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
        cfg['logging'].update(run_name=name,wandb_name=name,save_dir=f'models/{name}',output_dir=f'reconstructions/{name}')
        cfg['evaluation'].update(evaluate=False,sampling_methods=['plms']);cfg['system'].update(device='cuda',debug=False)
        (out/f'{name}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        print(name,{s:sum(r['split']==s for r in rows) for s in ('train','validation','test','excluded')},flush=True)
if __name__=='__main__': main()
