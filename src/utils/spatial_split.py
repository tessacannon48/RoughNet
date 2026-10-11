"""Explicit splits using conservative bounds of LiDAR and all six S2 views."""
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


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


def validate_manifest(manifest, root=None):
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
    if root is not None:
        for row in rows:
            for f in row['files']:
                st = (Path(root)/f['path']).stat()
                if (st.st_size,st.st_mtime_ns) != (f['size'],f['mtime_ns']):
                    raise ValueError(f"Input changed: {f['path']}")
    return manifest


def load_training_split(path, config):
    m = validate_manifest(json.loads(Path(path).read_text()), Path.cwd())
    regions = {r['region'] for r in m['patches'] if r['split']=='train'}
    for kind in ('s2','lidar'):
        actual = {str(Path(p).resolve()) for p in config['data'][kind+'_dirs']}
        wanted = {str(Path(m['directories'][r][kind]).resolve()) for r in regions}
        if actual != wanted:
            raise ValueError(f'{kind} directories differ from manifest')
    config['data']['split_digest'] = digest(m)
    return tuple([r['id'] for r in m['patches'] if r['split']==s] for s in ('train','validation'))
