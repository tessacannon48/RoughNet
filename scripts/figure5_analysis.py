"""Prepare, resume and summarize the extended Figure 5 notebook analysis.

Default inputs are verified corrected cosine PLMS prediction archives: no GPU
inference, Colab, downloads or duplicate prediction TIFFs are needed.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import multiprocessing
from pathlib import Path
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import rasterio
import scipy
from scipy import stats
import yaml
from scripts.fig5_statistics import analyse_arrays, agreement, fit_variogram, weighted_kappa

REGIONS = {'pondinlet': 'Pond Inlet', 'tuk': 'Tuktoyaktuk', 'cambridge': 'Cambridge Bay'}


def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sources():
    return {name: sha(ROOT / 'scripts' / name) for name in ('figure5_analysis.py', 'fig5_statistics.py')}


def options_from_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    a = cfg['analysis']
    for name in ('min_valid', 'variogram_samples', 'fit_samples', 'histogram_bins'):
        if not isinstance(a[name], int) or a[name] < 2: raise ValueError(f'{name} must be an integer >=2')
    if not (0 < a['pixel_size_m'] and 0 < a['lag_width_m'] < a['max_lag_m'] <= a['max_range_m']):
        raise ValueError('Invalid pixel size/lag/range limits')
    if a['max_lag_m'] % a['lag_width_m'] > 1e-8: raise ValueError('Lag width must divide maximum lag')
    if a['max_range_m'] <= 50: raise ValueError('Maximum range must exceed 50 m initial fit guess')
    if a['histogram_range_m'][0] >= a['histogram_range_m'][1]: raise ValueError('Invalid histogram range')
    return cfg


def prepare(args):
    if (args.out / 'manifest.json').exists(): raise FileExistsError('Study exists; reuse it or select another --out')
    cfg = options_from_config(args.config)
    splits = json.loads(resolve(cfg['split_ids']).read_text())
    records, provenance = [], {}
    for set_name, source in cfg['sets'].items():
        if not set_name.replace('_', '').replace('-', '').isalnum(): raise ValueError('Unsafe set name')
        if source['type'] == 'cache':
            study = resolve(source['study']); m = json.loads((study / 'manifest.json').read_text())
            if m['configurations'][source['schedule']]['data']['pixel_size_m'] != cfg['analysis']['pixel_size_m']:
                raise ValueError('Analysis pixel size differs from cached study')
            provenance[set_name] = dict(manifest_sha256=sha(study / 'manifest.json'),
                                        checkpoint=m['checkpoints'][source['schedule']])
        elif source['type'] != 'geotiff': raise ValueError('Input type must be cache or geotiff')
        for region in args.regions:
            role = 'test' if region == 'cambridge' else 'validation'
            ids = splits[region][role]
            if args.patch_ids: ids = [pid for pid in ids if pid in args.patch_ids]
            if not ids: continue
            if source['type'] == 'cache':
                cohort = m['cohorts'][region]
                if cohort['split'] != role or [r['id'] for r in cohort['records']] != splits[region][role]:
                    raise ValueError(f'Cache cohort differs from corrected split: {region}')
                archives = {}
                for start in range(0, len(cohort['records']), m['batch_size']):
                    path = study / 'shards' / f"{source['schedule']}_{source['sampler']}_{region}_{start:06d}.json"
                    shard = json.loads(path.read_text())
                    expected = [r['id'] for r in cohort['records'][start:start + m['batch_size']]]
                    if shard['manifest_digest'] != digest(m) or shard['ids'] != expected or len(shard['archives']) != len(expected):
                        raise ValueError(f'Cache shard provenance/membership mismatch: {path}')
                    for pid, archive in zip(expected, shard['archives']):
                        rel = f"predictions/{source['schedule']}/{source['sampler']}/{region}/{pid}.npz"
                        if archive['path'] != rel: raise ValueError('Cache archive identity mismatch')
                        archives[pid] = archive
            for pid in ids:
                row = dict(set=set_name, region=region, pid=pid, split=role, type=source['type'])
                if source['type'] == 'cache':
                    row.update(path=str(study / archives[pid]['path']), sha256=archives[pid]['sha256'])
                else:
                    if source.get('prediction_domain') not in ('raw', 'residual'):
                        raise ValueError('GeoTIFF prediction_domain must be explicitly raw or residual')
                    gt = resolve(source['gt_dirs'][region]) / f'lidar_patch_{pid}.tif'
                    pred = resolve(source['pred_dirs'][region]) / f'pred_{pid}.tif'
                    with rasterio.open(gt) as ds:
                        t = ds.transform; pixel = cfg['analysis']['pixel_size_m']
                        if not np.allclose([abs(t.a), abs(t.e), t.b, t.d], [pixel, pixel, 0, 0]):
                            raise ValueError('GeoTIFF pixel geometry differs from analysis pixel size')
                    row.update(gt_path=str(gt), pred_path=str(pred), gt_sha256=sha(gt), pred_sha256=sha(pred),
                               prediction_domain=source['prediction_domain'])
                records.append(row)
    if not records: raise ValueError('No selected targets')
    if args.patch_ids and set(args.patch_ids) - {r['pid'] for r in records}: raise ValueError('Requested patch absent from selected cohorts')
    manifest = dict(version=1, options=cfg['analysis'], sources=sources(), records=records,
                    input_provenance=provenance, scope='subset' if args.patch_ids or set(args.regions) != set(REGIONS) else 'full',
                    notebook=dict(path='notebooks/Fig5_Extended_RoughNet (1).ipynb',
                                  sha256=sha(ROOT / 'notebooks/Fig5_Extended_RoughNet (1).ipynb')),
                    runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, rasterio=rasterio.__version__))
    write_json(args.out / 'manifest.json', manifest)
    print(f"Prepared {len(records)} set/target pairs; scope={manifest['scope']}", flush=True)


def load_pair(record):
    if record['type'] == 'cache':
        if sha(record['path']) != record['sha256']: raise ValueError('Cached prediction archive changed')
        with np.load(record['path'], allow_pickle=False) as z:
            return z['gt'].astype(float), z['prediction'].astype(float), z['mask'].astype(bool)
    if sha(record['gt_path']) != record['gt_sha256'] or sha(record['pred_path']) != record['pred_sha256']:
        raise ValueError('GeoTIFF inputs changed')
    with rasterio.open(record['gt_path']) as ds:
        raw = ds.read().astype(float); geometry = (ds.shape, ds.transform, ds.crs)
        mask = raw[1] > .5 if len(raw) > 1 else ds.read_masks(1).astype(bool) & np.isfinite(raw[0]) & (raw[0] > -100)
    with rasterio.open(record['pred_path']) as ds:
        if (ds.shape, ds.transform, ds.crs) != geometry: raise ValueError('GT/prediction grids differ')
        pred = ds.read(1).astype(float)
        if not ds.read_masks(1)[mask].all(): raise ValueError('Prediction has nodata on GT support')
    mean = raw[0][mask].mean()
    return raw[0] - mean, pred - mean if record['prediction_domain'] == 'raw' else pred, mask


def result_path(out, record):
    return out / 'patches' / record['set'] / record['region'] / f"{record['pid']}.json"


def load_result(out, record, manifest):
    path = result_path(out, record); value = json.loads(path.read_text())
    if value['manifest_digest'] != digest(manifest) or value['input'] != record: raise ValueError('Analysis provenance mismatch')
    if value['arrays_sha256'] != sha(path.with_suffix('.npz')): raise ValueError('Analysis arrays corrupted')
    return value


def work(task):
    out, record, manifest = task
    start = time.monotonic()
    gt, pred, mask = load_pair(record)
    row, details, arrays = analyse_arrays(gt, pred, mask, manifest['options'])
    path = result_path(out, record); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
    tmp.replace(path.with_suffix('.npz'))
    write_json(path, dict(manifest_digest=digest(manifest), input=record,
                          row=dict(set=record['set'], region=record['region'], pid=record['pid'], split=record['split'], **row),
                          details=details, arrays_sha256=sha(path.with_suffix('.npz')), seconds=time.monotonic() - start))
    return f"{record['set']}/{record['region']}/{record['pid']}: {time.monotonic() - start:.1f}s"


def load_manifest(out):
    m = json.loads((out / 'manifest.json').read_text())
    if m['sources'] != sources(): raise ValueError('Analysis source changed; prepare a separate output directory')
    runtime = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, rasterio=rasterio.__version__)
    if runtime != m['runtime']: raise ValueError('Analysis runtime changed')
    return m


def initialize_worker(out, manifest):
    global WORKER_OUT, WORKER_MANIFEST
    WORKER_OUT, WORKER_MANIFEST = out, manifest


def work_record(record):
    return work((WORKER_OUT, record, WORKER_MANIFEST))


def run(args):
    m = load_manifest(args.out); pending = []
    for record in m['records']:
        if result_path(args.out, record).exists():
            load_pair(record)  # Verify input bytes even for resumed records.
            load_result(args.out, record, m)
        else: pending.append(record)
    if args.max_patches: pending = pending[:args.max_patches]
    print(f'{len(pending)} patches pending in this invocation', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker, initargs=(args.out, m)) as pool:
        for message in pool.map(work_record, pending, chunksize=1): print(message, flush=True)


def summarize(args):
    m = load_manifest(args.out); results = []
    for r in m['records']:
        if not result_path(args.out, r).exists(): raise ValueError(f'Incomplete analysis: {r["region"]}/{r["pid"]}')
        results.append(load_result(args.out, r, m))
    rows = [r['row'] for r in results]; write_csv(args.out / 'per_patch.csv', rows)
    summaries = []
    a = m['options']; centers = np.arange(a['lag_width_m']/2, a['max_lag_m'], a['lag_width_m'])
    for set_name, region in sorted({(r['set'], r['region']) for r in rows}):
        selected = [r for r in results if r['row']['set'] == set_name and r['row']['region'] == region]
        group = [r['row'] for r in selected]
        arrays = []
        for r in selected:
            with np.load(result_path(args.out, r['input']).with_suffix('.npz')) as z: arrays.append(dict(z))
        summary = dict(set=set_name, region=region, split=group[0]['split'], n=len(group), scope=m['scope'])
        for key in ('gt', 'pred'):
            sigmas = np.asarray([r[f'sigma_{key}'] for r in group])
            summary[f'rms_patch_sigma_{key}_m'] = float(np.sqrt(np.mean(sigmas**2)))
            summary[f'median_patch_sigma_{key}_m'] = float(np.median(sigmas))
            sv = np.asarray([z[f'sv_{key}'] for z in arrays])
            count = np.isfinite(sv).sum(0)
            mean = np.divide(np.nansum(sv, axis=0), count, out=np.full(len(centers), np.nan), where=count > 0)
            fitted = fit_variogram(centers, mean, a['max_range_m'])
            summary[f'mean_patch_variogram_{key}_range_m'] = fitted['effective_range_m']
            summary[f'mean_patch_variogram_{key}_fit_status'] = fitted['status']
            summary[f'range_{key}_failed_n'] = sum(r[f'range_{key}_m'] is None for r in group)
            summary[f'range_{key}_beyond_lag_n'] = sum(r[f'range_{key}_beyond_lag'] for r in group)
            summary[f'range_{key}_upper_bound_n'] = sum(r[f'range_{key}_status'] == 'upper_bound' for r in group)
            for col in (f'range_{key}_m', f'acf_{key}_m', f'skew_{key}', f'kurtosis_excess_{key}'):
                values = [r[col] for r in group if r[col] is not None]
                summary[f'median_{col}'] = float(np.median(values)) if values else None
            for family in ('Normal', 'Skew', 'Gamma', 'Log-normal'):
                summary[f'best_{key}_{family}_n'] = sum(r[f'best_{key}'] == family for r in group)
            summary[f'family_{key}_failed_n'] = sum(r[f'best_{key}'] is None for r in group)
            summary[f'histogram_outside_fraction_{key}_mean'] = float(np.mean([r[f'histogram_outside_fraction_{key}'] for r in group]))
        for kind, left, right in [('sigma', 'sigma_gt', 'sigma_pred'), ('range', 'range_gt_m', 'range_pred_m')]:
            for name, value in agreement([r[left] for r in group], [r[right] for r in group]).items():
                summary[f'{kind}_{name}'] = value
        for metric in ('jsd_centered_1m', 'wasserstein_centered_m', 'rmse_residual_m', 'rmse_centered_m'):
            vals = [r[metric] for r in group]
            summary[f'{metric}_mean'] = float(np.mean(vals)); summary[f'{metric}_median'] = float(np.median(vals))
        ratios = [r['sigma_pred']/r['sigma_gt'] for r in group if r['sigma_gt'] > 0]
        summary['median_sigma_ratio'] = float(np.median(ratios)) if ratios else None
        summary['sigma_ratio_n'] = len(ratios)
        if a['exploratory_proxies']:
            for key in ('gt', 'pred'):
                summary[f'mean_ridge_fraction_{key}'] = float(np.mean([r[f'ridge_fraction_{key}'] for r in group]))
            x, y = [r['sigma_class_gt'] for r in group], [r['sigma_class_pred'] for r in group]
            summary['illustrative_sigma_class_accuracy'] = float(np.mean(np.asarray(x) == y))
            summary['illustrative_sigma_class_weighted_kappa'] = weighted_kappa(x, y, len(a['sigma_class_edges_m'])+1)
        # Pooled histograms omit out-of-range values; retain that limitation above.
        hg = sum(z['hist_gt'] for z in arrays).astype(float) + 1e-12
        hp = sum(z['hist_pred'] for z in arrays).astype(float) + 1e-12
        hg /= hg.sum(); hp /= hp.sum(); mid = .5 * (hg + hp)
        summary['pooled_hist_jsd_base2'] = float(.5 * (np.sum(hg*np.log2(hg/mid)) + np.sum(hp*np.log2(hp/mid))))
        summaries.append(summary)
    write_json(args.out / 'regional_summary.json', summaries); write_csv(args.out / 'regional_summary.csv', summaries)
    fields = ['region', 'n', 'rms_patch_sigma_gt_m', 'rms_patch_sigma_pred_m', 'median_sigma_ratio',
              'mean_patch_variogram_gt_range_m', 'mean_patch_variogram_pred_range_m',
              'sigma_ccc', 'range_spearman', 'range_gt_failed_n', 'range_pred_failed_n']
    lines = ['# Regional Figure 5 diagnostics', '',
             'RMS of patch σ, not a regional mosaic σ. Range fits use mean within-patch semivariograms, not cross-patch pairs.',
             f"Scope: {m['scope']}. No confidence intervals; overlapping patches are not independent.", '',
             '| Set | ' + ' | '.join(fields) + ' |', '|---|' + '|'.join(['---']*len(fields)) + '|']
    tex = ['% Centered diagnostics; scope=' + m['scope'], r'\begin{tabular}{llrrrrrr}',
           r'Set & Region & $n$ & $\sigma_{GT}$ & $\sigma_{pred}$ & Ratio & Range GT & Range pred \\', r'\hline']
    show = lambda v: f'{v:.4f}' if isinstance(v, float) else str(v) if v is not None else 'NA'
    for r in summaries:
        lines.append('| ' + r['set'] + ' | ' + ' | '.join(show(r[k]) for k in fields) + ' |')
        values = [r[k] for k in ['set', 'region', 'n', 'rms_patch_sigma_gt_m', 'rms_patch_sigma_pred_m',
                                 'median_sigma_ratio', 'mean_patch_variogram_gt_range_m', 'mean_patch_variogram_pred_range_m']]
        tex.append(' & '.join(show(v).replace('_', r'\_') for v in values) + r' \\')
    tex.append(r'\end{tabular}')
    (args.out / 'regional_summary.md').write_text('\n'.join(lines)+'\n')
    (args.out / 'regional_summary.tex').write_text('\n'.join(tex)+'\n')
    rep = []
    for r in rows:
        if r['pid'] != args.patch_id or r['region'] != args.patch_region: continue
        group = [v for v in rows if v['region'] == r['region'] and v['set'] == r['set']]
        for key in ('sigma_gt', 'sigma_pred', 'range_gt_m', 'range_pred_m', 'jsd_centered_1m', 'wasserstein_centered_m', 'rmse_residual_m'):
            values = [v[key] for v in group if v[key] is not None]
            if r[key] is not None:
                rep.append(dict(set=r['set'], region=r['region'], pid=r['pid'], metric=key, value=r[key],
                                regional_median=float(np.median(values)), percentile=float(stats.percentileofscore(values, r[key])), n=len(values)))
    if rep: write_csv(args.out / 'patch_representativeness.csv', rep)
    write_json(args.out / 'completion.json', dict(n=len(rows), scope=m['scope'], manifest_digest=digest(m),
                                                intervals='none; overlapping patches are not independent', summed_task_seconds=sum(r['seconds'] for r in results)))
    print(f'Summarized {len(rows)} pairs across {len(summaries)} set/region groups', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'summarize'):
        p = sub.add_parser(name); p.add_argument('--out', type=Path, default=ROOT / 'reconstructions/fig5_extended_clean')
        if name == 'prepare':
            p.add_argument('--config', type=Path, default=ROOT / 'configs/figure5.yaml')
            p.add_argument('--regions', nargs='+', choices=tuple(REGIONS), default=list(REGIONS))
            p.add_argument('--patch-ids', nargs='+')
        elif name == 'run':
            p.add_argument('--workers', type=int, choices=range(1, 33), default=2)
            p.add_argument('--max-patches', type=int)
        else:
            p.add_argument('--patch-id', default='12556'); p.add_argument('--patch-region', default='tuk', choices=tuple(REGIONS))
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'analysis.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare': prepare, 'run': run, 'summarize': summarize}[args.command](args)


if __name__ == '__main__': main()
