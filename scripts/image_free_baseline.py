"""Image-free empirical sampler of fully observed, corrected training patches.

No images, test-derived donor selection, fitted model or spatial alignment.
Run prepare/run/summarize from Dissertation/. All computation is on CPU.
"""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
import os
from pathlib import Path
import platform
import hashlib
import subprocess
import sys
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/roughnet-mpl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import rasterio
import scipy
import torch
from src.data.dataset import LidarS2Dataset
from src.utils.conditioning_ablation import METRICS, json_digest, patch_metrics, sha256_file, write_json
from src.utils.spatial_split import digest, validate_manifest


def sources():
    paths = [Path(__file__), ROOT / 'scripts/evaluation.py', *ROOT.joinpath('src').rglob('*.py')]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def read_lidar(path):
    """Reuse dataset demeaning without loading satellite images or attributes."""
    with rasterio.open(path) as src:
        raw = src.read().astype(np.float32)
    mask = raw[1] > 0.5 if len(raw) > 1 else np.ones(raw.shape[1:], dtype=bool)
    if raw.shape[1:] != (256, 256) or not mask.any() or not np.isfinite(raw[0][mask]).all():
        raise ValueError(f'Invalid LiDAR shape/valid elevations: {path}')
    # Invalid values do not contribute to the dataset's valid-pixel mean.
    data = torch.from_numpy(np.where(mask, raw[0], 0).copy())[None]
    mask = torch.from_numpy(mask)
    residual, mean = LidarS2Dataset._demean_lidar_patch(data, mask)
    return residual, mask, mean


def input_record(row, split):
    path = str(Path(split['directories'][row['region']]['lidar']) / f"lidar_patch_{row['id']}.tif")
    recorded = next(f for f in row['files'] if f['path'] == path)
    st = (ROOT / path).stat()
    if (st.st_size, st.st_mtime_ns) != (recorded['size'], recorded['mtime_ns']):
        raise ValueError(f'Input changed since split preparation: {path}')
    return dict(region=row['region'], tile_id=row['id'], path=path,
                size=st.st_size, mtime_ns=st.st_mtime_ns)


def assignments(targets, donors, seed):
    """Uniform replacement draws, independent of target values and batching."""
    if not donors:
        raise ValueError('No fully observed training donors')
    rng = np.random.Generator(np.random.PCG64(seed))
    return [dict(region=t['region'], tile_id=t['tile_id'], target_path=t['path'],
                 donor_region=donors[i]['region'], donor_tile_id=donors[i]['tile_id'],
                 donor_path=donors[i]['path'])
            for t, i in zip(targets, rng.integers(len(donors), size=len(targets)))]


def metric_signature(source):
    """Compare scorer bodies and their imported bindings, ignoring mosaic code."""
    tree = ast.parse(source)
    functions = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                 for name in ('_metric_params_from_cfg', '_compute_metrics_tensor')]
    used = {n.id for f in functions for n in ast.walk(f) if isinstance(n, ast.Name)}
    imports = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                binding = alias.asname or alias.name.split('.')[0]
                if binding in used:
                    imports[binding] = (getattr(node, 'module', None), alias.name)
    return dict(functions=[ast.dump(f, include_attributes=False) for f in functions], imports=imports)


def reference_parity(reference, current, out):
    parity = {}
    for name, checksum in reference['sources'].items():
        if name not in current or current[name] == checksum:
            continue
        if name != 'scripts/evaluation.py':
            raise ValueError(f'Scoring/data code differs from reference study: {name}')
        # The existing evaluation module gained cached mosaics after inference.
        # Recover its exact recorded version, then prove the scoring code agrees.
        revisions = subprocess.check_output(
            ['git', 'log', '--all', '--format=%H', '--', name], cwd=ROOT, text=True).splitlines()
        for revision in revisions:
            old = subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=ROOT)
            if hashlib.sha256(old).hexdigest() != checksum:
                continue
            if metric_signature(old) != metric_signature((ROOT / name).read_bytes()):
                raise ValueError('Reference evaluation metric functions/imports changed')
            snapshot = out / 'reference_evaluation.py'
            snapshot.write_bytes(old)
            parity = dict(reference_sha256=checksum, revision=revision,
                          current_sha256=current[name], functions_and_imports_identical=True)
            break
        else:
            raise ValueError('Exact reference evaluation source unavailable in local Git history')
    return parity


def prepare(args):
    if (args.out / 'manifest.json').exists():
        raise FileExistsError('Study already prepared; reuse it or choose a new output directory')
    split = json.loads(args.manifest.read_text())
    validate_manifest(split)
    if split['test_region'] != 'cambridge' or split['validation_zones'] != {'pondinlet': 4, 'tuk': 13}:
        raise ValueError('Expected corrected original geographic fold')
    reference = json.loads((args.reference_study / 'manifest.json').read_text())
    if reference['split_digest'] != digest(split) or reference['max_patches'] is not None:
        raise ValueError('Reference study must use the complete corrected split')
    current = sources()
    parity = reference_parity(reference, current, args.out)
    cfg = reference['configurations']['cosine']
    train = sorted((r for r in split['patches'] if r['split'] == 'train'), key=lambda r: (r['region'], r['id']))
    targets = sorted((r for r in split['patches'] if r['split'] == args.split), key=lambda r: (r['region'], r['id']))
    expected = {(region, r['id']) for region, c in reference['cohorts'].items()
                if c['split'] == args.split for r in c['records']}
    if {(r['region'], r['id']) for r in targets} != expected:
        raise ValueError('Target membership differs from reference study')
    train = [input_record(r, split) for r in train]
    targets = [input_record(r, split) for r in targets]

    def inspect(record):
        with rasterio.open(ROOT / record['path']) as src:
            raw = src.read().astype(np.float32)
        if raw.shape[1:] != (256, 256):
            raise ValueError(f"Unexpected donor dimensions: {record['path']}")
        mask = raw[1] > 0.5 if len(raw) > 1 else np.ones(raw.shape[1:], dtype=bool)
        reason = ('incomplete_mask' if not mask.all() else
                  'nonfinite_elevations' if not np.isfinite(raw[0]).all() else None)
        return dict(record, eligible=reason is None, exclusion_reason=reason, valid_pixels=int(mask.sum()))

    inventory = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, record in enumerate(pool.map(inspect, train), 1):
            inventory.append(record)
            if i % 1000 == 0:
                print(f'Audited {i}/{len(train)} training masks', flush=True)
    donors = [r for r in inventory if r['eligible']]
    pairs = assignments(targets, donors, args.seed)
    counts = {r: dict(training=sum(x['region'] == r for x in inventory),
                      eligible=sum(x['region'] == r for x in donors))
              for r in sorted({x['region'] for x in inventory})}
    reference_csv = args.reference_study / args.split / 'per_patch.csv'
    manifest = dict(version=1, split=args.split, split_digest=digest(split), seed=args.seed,
                    batch_size=args.batch_size, config=cfg, sources=current,
                    reference_metric_parity=parity,
                    donor_inventory=inventory, targets=targets, assignments=pairs, donor_counts=counts,
                    reference_csv=dict(path=str(reference_csv.resolve()), sha256=sha256_file(reference_csv)),
                    protocol='Uniform with replacement over fully valid corrected training LiDAR patches; '
                             'no rotation, scaling, filling, target matching or prediction re-demeaning; '
                             'donor valid-pixel demeaning; original target scoring mask; seed PCG64; '
                             'single draw per target; no confidence intervals')
    write_json(args.out / 'manifest.json', manifest)
    write_csv(args.out / 'assignments.csv', pairs)
    print(f'Prepared {len(pairs)} targets and {len(donors)}/{len(train)} eligible donors: {counts}', flush=True)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def load_study(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['sources'] != sources():
        raise ValueError('Source changed since prepare; use a new study directory')
    return manifest


def check_inputs(manifest):
    for record in manifest['donor_inventory'] + manifest['targets']:
        st = (ROOT / record['path']).stat()
        if (st.st_size, st.st_mtime_ns) != (record['size'], record['mtime_ns']):
            raise ValueError(f"Input changed: {record['path']}")


def valid_shard(path, manifest, pairs):
    shard = json.loads(path.read_text())
    if shard['manifest_digest'] != json_digest(manifest) or len(shard['rows']) != len(pairs):
        raise ValueError(f'Shard provenance/count mismatch: {path}')
    for row, pair in zip(shard['rows'], pairs):
        if any(row.get(k) != v for k, v in pair.items()):
            raise ValueError(f'Shard donor/target mismatch: {path}')
        if any(row.get(k) is None or not np.isfinite(row[k]) for k in METRICS):
            raise ValueError(f'Non-finite metrics: {path}')
    if shard['rows_digest'] != json_digest(shard['rows']):
        raise ValueError(f'Shard scores changed: {path}')
    return shard


def run(args):
    manifest = load_study(args.out)
    check_inputs(manifest)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                   rasterio=rasterio.__version__, scipy=scipy.__version__, device='cpu', threads=4,
                   platform=platform.platform(), deterministic=True)
    path = args.out / 'runtime.json'
    if path.exists() and json.loads(path.read_text()) != runtime:
        raise ValueError('Runtime differs from existing study')
    write_json(path, runtime)
    completed = 0
    pairs = manifest['assignments']
    for start in range(0, len(pairs), manifest['batch_size']):
        batch = pairs[start:start + manifest['batch_size']]
        path = args.out / 'shards' / f'{start:06d}.json'
        if path.exists():
            valid_shard(path, manifest, batch)
            continue
        before = time.monotonic()
        rows = []
        for pair in batch:
            gt, mask, _ = read_lidar(ROOT / pair['target_path'])
            pred, donor_mask, _ = read_lidar(ROOT / pair['donor_path'])
            if not donor_mask.all():
                raise ValueError('Prepared donor is no longer fully valid')
            values = patch_metrics(gt, pred, mask, manifest['config'])
            if any(v is None or not np.isfinite(v) for v in values.values()):
                raise ValueError(f'Undefined metric; no target silently excluded: {pair}')
            rows.append(dict(pair, valid_pixels=int(mask.sum()), **values))
        write_json(path, dict(manifest_digest=json_digest(manifest), rows=rows,
                              rows_digest=json_digest(rows), elapsed_s=time.monotonic() - before))
        print(f'{start + len(batch)}/{len(pairs)} targets ({time.monotonic() - before:.2f}s)', flush=True)
        completed += 1
        if args.max_batches is not None and completed >= args.max_batches:
            return


def summarize(args):
    manifest = load_study(args.out)
    rows = []
    pairs = manifest['assignments']
    timings = []
    for start in range(0, len(pairs), manifest['batch_size']):
        path = args.out / 'shards' / f'{start:06d}.json'
        if not path.exists():
            raise ValueError(f'Incomplete study: missing {path}')
        shard = valid_shard(path, manifest, pairs[start:start + manifest['batch_size']])
        rows.extend(shard['rows'])
        timings.append(shard['elapsed_s'])
    ref = manifest['reference_csv']
    if sha256_file(ref['path']) != ref['sha256']:
        raise ValueError('Reference scores changed')
    with open(ref['path']) as f:
        reference = [r for r in csv.DictReader(f) if r['schedule'] == 'cosine' and r['sampler'] == 'plms']
    keys = lambda records: [(r['region'], r['tile_id']) for r in records]
    if Counter(keys(reference)) != Counter(keys(rows)):
        raise ValueError('Reference scores have different/duplicate target membership')
    summaries = []
    for region in sorted({r['region'] for r in rows}):
        for method, records in [('empirical_training_patch', rows), ('roughnet_cosine_plms', reference)]:
            group = [r for r in records if r['region'] == region]
            summary = dict(region=region, method=method, n=len(group))
            for metric in METRICS:
                values = np.asarray([float(r[metric]) for r in group])
                if not np.isfinite(values).all():
                    raise ValueError('Reference has non-finite scores')
                summary[metric + '_mean'] = float(values.mean())
                summary[metric + '_std'] = float(values.std())
            summaries.append(summary)
    dest = args.out / manifest['split']
    write_csv(dest / 'per_patch.csv', rows)
    write_csv(dest / 'comparison.csv', summaries)
    write_json(dest / 'comparison.json', summaries)
    write_json(dest / 'completion.json', dict(n=len(rows), manifest_digest=json_digest(manifest),
                                             scored_seconds=sum(timings), donor_counts=manifest['donor_counts']))
    lines = ['# Image-free empirical baseline', '', manifest['protocol'], '',
             'Means and population SD across patches; single seed; no confidence intervals.', '',
             '| Region | Method | n | ' + ' | '.join(METRICS) + ' |',
             '|---|---|---:|' + '|'.join(['---:'] * len(METRICS)) + '|']
    for r in summaries:
        lines.append(f"| {r['region']} | {r['method']} | {r['n']} | " + ' | '.join(
            f"{r[k + '_mean']:.6f} ± {r[k + '_std']:.6f}" for k in METRICS) + ' |')
    (dest / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(f'Complete: {len(rows)} targets; tables in {dest}', flush=True)


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'run', 'summarize'):
        p = sub.add_parser(command)
        p.add_argument('--out', type=Path, default=Path('reconstructions/image_free_baseline_clean'))
        if command == 'prepare':
            p.add_argument('--manifest', type=Path, default=Path('splits/original_clean.json'))
            p.add_argument('--reference-study', type=Path, default=Path('reconstructions/noise_sampler_clean'))
            p.add_argument('--split', choices=('test', 'validation'), default='test')
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--batch-size', type=positive, default=8)
            p.add_argument('--workers', type=positive, default=8)
        elif command == 'run':
            p.add_argument('--max-batches', type=positive)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'study.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare': prepare, 'run': run, 'summarize': summarize}[args.command](args)


if __name__ == '__main__':
    main()
