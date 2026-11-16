"""Explicit splits using conservative bounds of LiDAR and all six S2 views."""
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(value):
    if value.get('version') == 2:
        return portable_digest(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def data_path(path, root=None, data_root=None):
    """Resolve recorded input_data-relative paths without trusting stored absolute paths."""
    path = Path(path)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Expected a relative dataset path: {path}')
    if data_root is not None:
        parts = path.parts[1:] if path.parts and path.parts[0] == 'input_data' else path.parts
        return Path(data_root).joinpath(*parts)
    return Path(root or Path.cwd()) / path


def portable_digest(manifest):
    """Scientific split/content identity, independent of timestamps and machine paths."""
    rows = []
    for row in sorted(manifest['patches'], key=lambda r: (r['region'], r['id'])):
        value = {k: row[k] for k in ('region', 'id', 'split', 'crs', 'bounds')}
        for key in ('zone', 'view_bounds'):
            if key in row:
                value[key] = row[key]
        value['files'] = []
        for f in sorted(row['files'], key=lambda f: f['path']):
            data_path(f['path'])
            if len(f.get('sha256', '')) != 64:
                raise ValueError('Portable inventories require SHA-256 for every input')
            value['files'].append({k: f[k] for k in ('path', 'size', 'sha256')})
        rows.append(value)
    identity = dict(schema='roughnet-dataset-v2', patches=rows,
                    validation_zones=manifest.get('validation_zones'),
                    test_region=manifest.get('test_region'),
                    footprint_rule=manifest.get('footprint_rule'))
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_checkpoint_split(config, manifest, checkpoint, registry=None):
    """Accept legacy weights only through a checksum-bound release compatibility record."""
    current = digest(manifest)
    if config['data'].get('split_digest') == current:
        return
    if manifest.get('version') != 2 or registry is None:
        raise ValueError('Checkpoint was not trained on the specified split')
    records = json.loads(Path(registry).read_text())['checkpoints']
    checksum = sha256_file(checkpoint)
    valid = any(r['sha256'] == checksum and r['portable_split_digest'] == current
                and r['legacy_split_digest'] == config['data'].get('split_digest')
                for r in records)
    if not valid:
        raise ValueError('No verified checkpoint/content/split compatibility record')


def overlapping_ids(train, validation):
    if not train or not validation:
        return set()
    b = np.asarray([r['bounds'] for r in validation])
    excluded = set()
    for row in train:
        l, bot, r, t = row['bounds']
        if ((b[:, 0] < r-1e-6) & (b[:, 2] > l+1e-6) & (b[:, 1] < t-1e-6) & (b[:, 3] > bot+1e-6)).any():
            excluded.add(row['id'])
    return excluded


def validate_manifest(manifest, root=None, data_root=None):
    rows = manifest['patches']
    for row in rows:
        b = np.asarray(row['bounds'])
        if b.shape != (4,) or not np.isfinite(b).all() or b[0] >= b[2] or b[1] >= b[3]:
            raise ValueError('Invalid footprint bounds')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Patch IDs must be globally unique')
    if {r['split'] for r in rows} - {'train','validation','test','excluded'}:
        raise ValueError('Unknown split')
    for region in {r['region'] for r in rows}:
        group = [r for r in rows if r['region'] == region]
        if len({r['crs'] for r in group}) != 1:
            raise ValueError('Mixed regional CRS')
        splits = {r['split'] for r in group}
        if 'test' in splits and splits != {'test'}:
            raise ValueError('Test region used for training/validation')
        if overlapping_ids([r for r in group if r['split']=='train'], [r for r in group if r['split']=='validation']):
            raise ValueError('Training footprints overlap validation')
    for split in ('train','validation','test'):
        if not any(r['split']==split for r in rows):
            raise ValueError(f'Empty {split} split')
    if manifest.get('version') == 2:
        portable_digest(manifest)
    if root is not None or data_root is not None:
        for row in rows:
            for f in row['files']:
                path = data_path(f['path'], root=root, data_root=data_root)
                st = path.stat()
                if manifest.get('version') == 2:
                    changed = st.st_size != f['size'] or sha256_file(path) != f['sha256']
                else:
                    changed = (st.st_size,st.st_mtime_ns) != (f['size'],f['mtime_ns'])
                if changed:
                    raise ValueError(f"Input changed: {f['path']}")
    return manifest


def load_training_split(path, config):
    data_root = config['data'].get('data_root')
    m = validate_manifest(json.loads(Path(path).read_text()), Path.cwd(), data_root=data_root)
    regions = {r['region'] for r in m['patches'] if r['split']=='train'}
    for kind in ('s2','lidar'):
        actual = {str(Path(p).resolve()) for p in config['data'][kind+'_dirs']}
        wanted = {str(data_path(m['directories'][r][kind], data_root=data_root).resolve()) for r in regions}
        if actual != wanted:
            raise ValueError(f'{kind} directories differ from manifest')
    config['data']['split_digest'] = digest(m)
    return tuple([r['id'] for r in m['patches'] if r['split']==s] for s in ('train','validation'))
