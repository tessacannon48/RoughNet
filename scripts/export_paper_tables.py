#!/usr/bin/env python3
"""Export all manuscript tables from measured per-patch CSVs, without PyTorch."""
import argparse
from collections import defaultdict
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

METRICS = ('rmse_m','bias_m','sigma_error_pct','nae_deg','zncc','jsd','log_psd_rmse','nrmse_sigma')
PRECISION = dict(zip(METRICS,(3,4,2,3,3,4,3,3)))
CONFIGS = [(s,p) for s in ('linear','cosine') for p in ('ddpm','ddim','plms')]
REGIONS = ('pondinlet','tuk','cambridge')
INPUTS = {'validation':'noise_sampler_clean/validation/per_patch.csv',
          'test':'noise_sampler_clean/test/per_patch.csv',
          'conditioning':'conditioning_ablation_clean/per_patch.csv',
          'image_free':'image_free_baseline_clean/test/per_patch.csv',
          'alternate':'tuk_heldout_clean_evaluation/test/per_patch.csv'}


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_metrics(path, required=()):
    with Path(path).open(newline='') as stream:
        reader=csv.DictReader(stream)
        missing=set(('region','tile_id',*METRICS,*required))-set(reader.fieldnames or [])
        if missing: raise ValueError(f'{path}: missing columns {sorted(missing)}')
        rows=[]
        for raw in reader:
            row=dict(raw)
            if row.get('mode') == 'shuffle': row['mode'] = 'shuffled'
            if row['region'] not in REGIONS or not row['tile_id']: raise ValueError('Unknown region or empty patch ID')
            for key in METRICS:
                row[key]=float(row[key])
                if not math.isfinite(row[key]): raise ValueError(f'Non-finite {key}')
            rows.append(row)
    if not rows: raise ValueError(f'Empty input: {path}')
    return rows


def groups(rows, keys, expected):
    result=defaultdict(list)
    for row in rows: result[tuple(row[k] for k in keys)].append(row)
    if set(result)!=set(expected): raise ValueError(f'Unexpected or missing configurations: {set(result)^set(expected)}')
    return result


def validate_ids(rows, expected):
    ids=[r['tile_id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):
        raise ValueError('Duplicate, missing or unexpected patch IDs')


def mean(rows, **labels):
    return dict(**labels,n=len(rows),**{k:statistics.fmean(r[k] for r in rows) for k in METRICS})


def add_ranks(rows):
    ranks=[[] for _ in rows]
    for metric in METRICS:
        values=[abs(r[metric]) if metric=='bias_m' else -r[metric] if metric=='zncc' else r[metric] for r in rows]
        for i,v in enumerate(values):
            # Mean of the occupied one-based positions, including exact ties.
            ranks[i].append(1+sum(w<v for w in values)+(sum(w==v for w in values)-1)/2)
    return [dict(r,mean_rank=statistics.fmean(rank)) for r,rank in zip(rows,ranks)]


def rounded(value, places):
    number=Decimal(str(value)).quantize(Decimal(1).scaleb(-places),rounding=ROUND_HALF_UP)
    return f'{abs(number) if number==0 else number:.{places}f}'


def build_tables(results_root, split_ids, alternate_ids, unet_results=None, allow_incomplete=False,
                 expected_counts=(1122,168,2111,1676)):
    root=Path(results_root)
    primary=json.loads(Path(split_ids).read_text());alternate=json.loads(Path(alternate_ids).read_text())
    expected={r:primary[r]['test' if r=='cambridge' else 'validation'] for r in REGIONS}
    alternate_expected=alternate['tuk']['test']
    for ids in [*expected.values(),alternate_expected]:
        if len(ids)!=len(set(ids)) or any(not isinstance(x,str) for x in ids):raise ValueError('Split IDs must be unique strings')
    if tuple(map(len,[*expected.values(),alternate_expected]))!=expected_counts:raise ValueError('Wrong manuscript cohort counts')
    if not set(expected['tuk'])<=set(alternate_expected):raise ValueError('Shared Tuk IDs missing from alternate cohort')
    files={k:root/v for k,v in INPUTS.items()}
    rows={k:read_metrics(p, ('schedule','sampler','split') if k in ('validation','test','alternate') else ('mode','seed') if k=='conditioning' else ()) for k,p in files.items()}
    for key,split in [('validation','validation'),('test','test'),('alternate','test')]:
        if any(r['split']!=split for r in rows[key]):raise ValueError('Unexpected evaluation split')
    matrix=groups(rows['validation']+rows['test'],('region','schedule','sampler'),[(r,*c) for r in REGIONS for c in CONFIGS])
    summary={}
    for key,part in matrix.items():
        r,s,p=key;validate_ids(part,expected[r]);summary[key]=mean(part,region=r,method=f'{s} {p}',schedule=s,sampler=p)
    main1=[]
    for region in REGIONS[:2]:main1+=add_ranks([dict(summary[(region,*c)],panel=region) for c in CONFIGS])
    equal=[]
    for s,p in CONFIGS:
        a,b=[summary[(r,s,p)] for r in REGIONS[:2]]
        equal.append(dict(panel='equal_region',region='equal_region',method=f'{s} {p}',schedule=s,sampler=p,n=a['n']+b['n'],
                          **{k:(a[k]+b[k])/2 for k in METRICS}))
    main1+=add_ranks(equal)
    supp1=add_ranks([dict(summary[('cambridge',*c)],panel='cambridge') for c in CONFIGS])
    if any(r['seed']!='42' for r in rows['conditioning']):raise ValueError('Conditioning seed must be 42')
    conditioning=groups(rows['conditioning'],('region','mode'),[(r,m) for r in REGIONS for m in ('normal','shuffled','null')])
    main2=[]
    for region in REGIONS:
        for mode in ('normal','shuffled','null'):
            part=conditioning[(region,mode)];validate_ids(part,expected[region])
            main2.append(mean(part,panel=region,region=region,method=f'roughnet_{mode}',schedule='cosine',sampler='plms'))
        if region=='cambridge':
            part=rows['image_free']
            if any(r['region']!='cambridge' for r in part):raise ValueError('Image-free baseline must be Cambridge only')
            validate_ids(part,expected[region]);main2.append(mean(part,panel=region,region=region,method='image_free_baseline'))
    omitted=[]
    if unet_results:
        files['unet']=Path(unet_results);part=read_metrics(unet_results)
        if any(r['region']!='cambridge' or r.get('split','test')!='test' for r in part):raise ValueError('U-Net table rows must be Cambridge test')
        validate_ids(part,expected['cambridge']);main2.append(mean(part,panel='cambridge',region='cambridge',method='deterministic_unet'))
    elif allow_incomplete:omitted=['main_table_2: Cambridge Bay deterministic U-Net (pending colleague contribution)']
    else:raise ValueError('U-Net results required; --allow-incomplete explicitly omits its pending row')
    alt=groups(rows['alternate'],('region','schedule','sampler'),[('tuk','cosine','plms')])[('tuk','cosine','plms')]
    validate_ids(alt,alternate_expected)
    shared=[r for r in alt if r['tile_id'] in set(expected['tuk'])];validate_ids(shared,expected['tuk'])
    supp2=[mean(alt,panel='tuk',region='tuk',method='alternate',cohort='all_test'),
           mean(matrix[('tuk','cosine','plms')],panel='tuk',region='tuk',method='primary',cohort='shared_168'),
           mean(shared,panel='tuk',region='tuk',method='alternate',cohort='shared_168')]
    files.update(primary_ids=Path(split_ids),alternate_ids=Path(alternate_ids))
    meta=dict(schema=1,complete=not omitted,omitted_rows=omitted,
              inputs={k:dict(path=str(p.relative_to(root)) if p.is_relative_to(root) else p.name,sha256=checksum(p)) for k,p in files.items()},
              shared_tuk_ids=expected['tuk'],metrics=list(METRICS),rounding='decimal_half_up',
              ranking='unrounded values; absolute bias ascending; ZNCC descending; others ascending; average ties',
              aggregation='per-patch arithmetic mean; equal_region averages regional means equally')
    return dict(main_table_1=main1,main_table_2=main2,supplementary_table_1=supp1,supplementary_table_2=supp2),meta


def render(tables, meta):
    outputs={}
    for name,rows in tables.items():
        cols=list(dict.fromkeys(k for r in rows for k in r))
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=cols,lineterminator='\n');writer.writeheader();writer.writerows(rows)
        outputs[name+'.csv']=stream.getvalue()
        complete = meta['complete'] or name != 'main_table_2'
        outputs[name+'.json']=json.dumps(dict(rows=rows,metadata=meta,complete=complete),indent=2,sort_keys=True,allow_nan=False)+'\n'
        ranked='mean_rank' in cols
        headers=['Panel','Method','Cohort','N',*METRICS]+(['Mean rank'] if ranked else [])
        lines=['% Generated from unrounded measurements.', '% COMPLETE' if complete else '% INCOMPLETE: U-Net contribution pending.',
               '\\begin{tabular}{llll'+('r'*(8+int(ranked)))+'}', '\\hline',
               ' & '.join(h.replace('_',r'\_') for h in headers)+r' \\', '\\hline']
        for row in rows:
            cells=[row['panel'],row['method'],row.get('cohort',''),str(row['n'])]
            cells+=[rounded(row[k],PRECISION[k]) for k in METRICS]
            if ranked:cells.append(rounded(row['mean_rank'],2))
            lines.append(' & '.join(str(x).replace('_',r'\_') for x in cells)+r' \\')
        outputs[name+'.tex']='\n'.join([*lines,'\\hline','\\end{tabular}',''])
    outputs['manifest.json']=json.dumps(dict(meta,outputs={k:hashlib.sha256(v.encode()).hexdigest() for k,v in outputs.items()}),indent=2,sort_keys=True)+'\n'
    return outputs


def export(results_root, split_ids, alternate_ids, out, unet_results=None, allow_incomplete=False, overwrite=False):
    tables,meta=build_tables(results_root,split_ids,alternate_ids,unet_results,allow_incomplete)
    outputs=render(tables,meta);out=Path(out)
    for name,value in outputs.items():
        p=out/name
        if p.exists() and p.read_text()!=value and not overwrite:raise FileExistsError(f'Changed output {p}; choose another directory or --overwrite')
    out.mkdir(parents=True,exist_ok=True)
    for name,value in outputs.items():
        p=out/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(value);tmp.replace(p)
    return tables,meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,default=Path('reconstructions'))
    p.add_argument('--split-ids',type=Path,default=Path('splits/train_val_split_ids.json'))
    p.add_argument('--alternate-ids',type=Path,default=Path('splits/tuk_heldout_clean_ids.json'))
    p.add_argument('--out',type=Path,default=Path('results/paper'))
    p.add_argument('--unet-results',type=Path);p.add_argument('--allow-incomplete',action='store_true');p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    try:_,meta=export(**vars(args))
    except (ValueError,OSError,KeyError) as e:p.exit(2,str(e)+'\n')
    print('Exported all four tables; complete='+str(meta['complete']))

if __name__=='__main__':main()
