"""Shared model and dataset handling for the deterministic RoughNet baseline."""
import copy
import json
from pathlib import Path

import torch

from src.data.dataset import LidarS2Dataset
from src.model.unet import ConditionalUNet
from src.utils.metrics import (
    masked_mse_loss, masked_mae_loss, masked_hybrid_mse_loss, masked_hybrid_mae_loss,
)
from src.utils.spatial_split import digest, sha256_file, validate_manifest

ROOT = Path(__file__).resolve().parents[2]
REGIONS = ('pondinlet', 'tuk', 'cambridge')
SPLITS = ('train', 'validation', 'test', 'excluded')
LOSSES = dict(masked_mse_loss=masked_mse_loss, masked_mae_loss=masked_mae_loss,
              masked_hybrid_mse_loss=masked_hybrid_mse_loss,
              masked_hybrid_mae_loss=masked_hybrid_mae_loss)


def resolve_path(value, root=ROOT):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def read_split_ids(path, expected_primary=True):
    """Read explicit regional IDs, rejecting overlap and changes to the paper split."""
    value = json.loads(Path(path).read_text())
    if set(value) != set(REGIONS):
        raise ValueError('Split IDs must contain all three paper regions')
    seen = set()
    for region in REGIONS:
        if set(value[region]) != set(SPLITS):
            raise ValueError('Each region requires train, validation, test and excluded IDs')
        for split in SPLITS:
            ids = value[region][split]
            if not isinstance(ids, list) or any(not isinstance(pid, str) or not pid for pid in ids):
                raise ValueError('Patch IDs must be nonempty strings, preserving leading zeros')
            for pid in ids:
                if pid in seen:
                    raise ValueError(f'Duplicate split assignment or nonunique patch ID: {pid}')
                seen.add(pid)
    if expected_primary:
        primary = json.loads((ROOT / 'splits/train_val_split_ids.json').read_text())
        if value != primary:
            raise ValueError('Split IDs differ from the exact primary manuscript split')
    return value


def prepare_config(cfg, *, data_root=None, split_ids=None, split_manifest=None,
                   root=ROOT, verify_data=False):
    """Relocate operational paths without inventing a checkpoint's training provenance.

    ``verify_data`` is used before new training. Evaluation checks the saved
    training identity separately before selecting a current evaluation cohort.
    """
    cfg = copy.deepcopy(cfg)
    data = cfg['data']
    locations = data.get('train_locations', ['pondinlet', 'tuk'])
    if locations != ['pondinlet', 'tuk']:
        raise ValueError('The manuscript baseline trains on Pond Inlet and Tuktoyaktuk only')
    data['train_locations'] = locations
    if data_root is None:
        data_root = data.get('data_root')
    if data_root is None:
        directories = [resolve_path(p, root) for k in ('lidar_dirs', 's2_dirs') for p in data.get(k, [])]
        parents = {p.parent for p in directories}
        if len(parents) > 1:
            raise ValueError('Supply --data-root for datasets stored in different locations')
        data_root = parents.pop() if parents else Path(root) / 'input_data'
    data_root = resolve_path(data_root, root)
    data['data_root'] = str(data_root)
    for kind in ('lidar', 's2'):
        wanted = [data_root / f'{kind}_patches_{r}' for r in locations]
        if data.get(kind + '_dirs'):
            # Relocation is explicit through data_root; region names still bind directories.
            names = [Path(p).name for p in data[kind + '_dirs']]
            if names != [p.name for p in wanted]:
                raise ValueError(f'{kind} directory names differ from the primary training regions')
        data[kind + '_dirs'] = list(map(str, wanted))
    id_path = resolve_path(split_ids or data.get('split_ids', 'splits/train_val_split_ids.json'), root)
    ids = read_split_ids(id_path)
    data['split_ids'] = str(id_path)
    data['split_ids_sha256'] = sha256_file(id_path)
    manifest_path = split_manifest or data.get('split_manifest')
    if manifest_path:
        manifest_path = resolve_path(manifest_path, root)
        data['split_manifest'] = str(manifest_path)
    if verify_data:
        if manifest_path is None:
            raise ValueError('New baseline training requires a portable spatial split manifest')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('version') != 2:
            raise ValueError('New baseline training requires a content-hashed portable split manifest')
        validate_manifest(manifest, root=root, data_root=data_root)
        for region in REGIONS:
            for split in SPLITS:
                actual = {r['id'] for r in manifest['patches'] if r['region'] == region and r['split'] == split}
                if actual != set(ids[region][split]):
                    raise ValueError('Portable manifest membership differs from the primary split IDs')
        previous = data.get('split_digest')
        current = digest(manifest)
        if previous and previous != current:
            raise ValueError('Configuration training split identity differs from the portable manifest')
        data['split_digest'] = current
    return cfg


def make_dataset(cfg, ids, split, *, region=None, augment=False, randomize_context=None):
    """Load exactly the requested primary IDs with consistent region directories."""
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Dataset split must be train, validation or test')
    if region is not None and region not in REGIONS:
        raise ValueError('Unknown region')
    if split == 'train' and region == 'cambridge':
        raise ValueError('Cambridge Bay cannot be used for baseline training')
    regions = [region] if region else (['cambridge'] if split == 'test' else ['pondinlet', 'tuk'])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Requested dataset IDs must be nonempty and unique')
    root = Path(cfg['data']['data_root'])
    tr = cfg['training']
    ds = LidarS2Dataset(
        lidar_dirs=[str(root / f'lidar_patches_{r}') for r in regions],
        s2_dirs=[str(root / f's2_patches_{r}') for r in regions],
        context_k=tr['context_k'],
        randomize_context=tr['randomize_context'] if randomize_context is None else randomize_context,
        augment=augment, debug=False, split_pids=ids,
        split='val' if split == 'validation' else split,
    )
    actual = [s['tile_id'] for s in ds.samples]
    if actual != list(ids):
        raise ValueError('Dataset dropped or reordered IDs from the requested split')
    for sample in ds.samples:
        lidar_region = Path(sample['lidar_path']).parent.name.removeprefix('lidar_patches_')
        s2_region = Path(sample['s2_group_dir']).parent.name.removeprefix('s2_patches_')
        if lidar_region not in regions or s2_region != lidar_region:
            raise ValueError('Dataset matched a patch to the wrong region')
    return ds


def build_model(cfg, device):
    k = cfg['training']['context_k']
    return ConditionalUNet(in_channels=1, cond_channels=4 * k, attr_dim=8 * k,
        cond_k=k, **cfg['model']).to(device)


def predict(model, s2, attrs):
    """The supplied deterministic baseline: zero LiDAR channel and timestep zero."""
    batch, _, height, width = s2.shape
    x = torch.zeros(batch, 1, height, width, device=s2.device)
    timestep = torch.zeros(batch, dtype=torch.long, device=s2.device)
    return model(x, s2, attrs, timestep)


def load_checkpoint(path, device='cpu'):
    """Load a trusted baseline checkpoint using its saved architecture/configuration."""
    saved = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(saved, dict) or 'config' not in saved or 'model_state_dict' not in saved:
        raise ValueError('A baseline checkpoint must include its saved configuration and model state')
    cfg = copy.deepcopy(saved['config'])
    name = cfg.get('baseline', {}).get('name')
    if name not in ('unet_regression', 'roughnet_unet_baseline', 'deterministic_unet'):
        raise ValueError('Checkpoint does not identify the deterministic U-Net baseline')
    if cfg.get('model_type', 'deterministic_unet') != 'deterministic_unet':
        raise ValueError('Checkpoint is not a deterministic U-Net')
    model = build_model(cfg, device)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    return model.eval(), cfg, saved
