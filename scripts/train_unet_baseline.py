#!/usr/bin/env python3
"""Train Weibin Chen's deterministic RoughNet U-Net baseline.

The model, zero LiDAR input, fixed timestep, loss, augmentation and Adam training
follow the supplied contribution. Path handling, explicit split checks, named
checkpoints and zero-worker epoch recovery follow this repository's conventions.
"""
import argparse
import copy
import csv
import fcntl
import json
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.utils.unet_baseline import (
    LOSSES, build_model, make_dataset, predict, prepare_config,
    read_split_ids, resolve_path,
)
from src.utils.spatial_split import sha256_file
from src.utils.training_state import atomic_save, restore_rng, resume_signature, rng_state

NAME = 'roughnet_unet_baseline'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/roughnet_unet_baseline.yaml')
    p.add_argument('--split-ids', type=Path, help='Exact primary regional JSON split IDs')
    p.add_argument('--manifest', type=Path, help='Portable spatial/content inventory for new training')
    p.add_argument('--data-root', type=Path, help='Directory containing lidar_patches_* and s2_patches_*')
    p.add_argument('--save-dir', '--out-dir', '--out_dir', dest='save_dir', type=Path,
                   help='Checkpoint directory; defaults to models/roughnet_unet_baseline')
    p.add_argument('--run-name', '--run_name', dest='run_name', help='Logging label; use --save-dir for a different run')
    for name, value_type in [('batch_size', int), ('epochs', int), ('lr', float),
                             ('num_workers', int), ('context_k', int), ('unet_depth', int),
                             ('base_channels', int), ('embed_dim', int)]:
        flags = ['--' + name.replace('_', '-')]
        if '_' in name:
            flags.append('--' + name)
        p.add_argument(*flags, dest=name, type=value_type)
    p.add_argument('--randomize-context', '--randomize_context', dest='randomize_context', action='store_true')
    p.add_argument('--attention-variant', '--attention_variant', dest='attention_variant',
                   choices=['default', 'none', 'mid', 'heavy', 'all'])
    p.add_argument('--loss-name', '--loss_name', dest='loss_name', choices=list(LOSSES))
    p.add_argument('--loss-alpha', '--loss_alpha', dest='loss_alpha', type=float)
    p.add_argument('--seed', type=int)
    p.add_argument('--device', help='PyTorch device, or auto to choose CUDA when available')
    p.add_argument('--resume', nargs='?', const='auto',
                   help='Resume latest completed epoch (only zero-worker recovery checkpoints), or supply its path')
    p.add_argument('--eval-only', '--eval_only', dest='eval_only', type=Path,
                   help='Use the saved checkpoint configuration for validation with evaluate_unet_baseline.py')
    p.add_argument('--legacy-splits', '--splits', dest='legacy_splits', type=Path,
                   help='Original CSV split proof for evaluation of a supplied legacy checkpoint; not a training input')
    p.add_argument('--wandb', action='store_true', help='Opt in to Weights & Biases logging')
    p.add_argument('--debug', action='store_true', help='Explicit 100 train / 100 validation patch experiment')
    return p.parse_args(argv)


def build_config(args):
    cfg = yaml.safe_load(args.config.read_text())
    tr, model = cfg['training'], cfg['model']
    for key in ('batch_size', 'epochs', 'lr', 'num_workers', 'context_k'):
        if getattr(args, key) is not None:
            tr[key] = getattr(args, key)
    for key in ('unet_depth', 'base_channels', 'embed_dim', 'attention_variant'):
        if getattr(args, key) is not None:
            model[key] = getattr(args, key)
    if args.randomize_context:
        tr['randomize_context'] = True
    if args.loss_name is not None:
        tr['loss']['name'] = args.loss_name
    if args.loss_alpha is not None:
        tr['loss']['alpha'] = args.loss_alpha
    if args.seed is not None:
        cfg['baseline']['seed'] = args.seed
    if args.device is not None:
        cfg['system']['device'] = args.device
    cfg['system']['debug'] = args.debug
    cfg['baseline']['wandb'] = args.wandb
    if args.save_dir is not None:
        cfg['logging']['save_dir'] = str(args.save_dir)
    elif args.debug:
        cfg['logging']['save_dir'] += '_debug'
    if args.run_name:
        cfg['logging']['run_name'] = args.run_name
    cfg = prepare_config(cfg, data_root=args.data_root, split_ids=args.split_ids,
                         split_manifest=args.manifest)
    cfg['logging']['save_dir'] = str(resolve_path(cfg['logging']['save_dir']))
    return cfg


def make_loader(dataset, cfg, validation=False):
    workers = cfg['training']['num_workers']
    if validation and workers:
        workers = max(1, workers // 2)
    return DataLoader(dataset, batch_size=cfg['training']['batch_size'],
                      shuffle=not validation, num_workers=workers,
                      pin_memory=cfg['system']['device'].startswith('cuda'),
                      persistent_workers=workers > 0)


def compute_loss(criterion, loss_name, pred, lidar, mask, alpha):
    if 'hybrid' in loss_name:
        loss, _, _ = criterion(pred, lidar, mask, alpha=alpha)
        return loss
    return criterion(pred, lidar, mask)


def source_identity():
    paths = ['scripts/train_unet_baseline.py', 'src/utils/unet_baseline.py',
             'src/data/dataset.py', 'src/data/processing.py', 'src/model/unet.py',
             'src/model/blocks.py', 'src/utils/metrics.py', 'src/utils/training_state.py']
    return {p: sha256_file(ROOT / p) for p in paths}


def training_signature(cfg):
    signature = resume_signature(cfg)
    signature['data'].pop('split_ids', None)
    return dict(configuration=signature, seed=cfg['baseline']['seed'],
                baseline=NAME, sources=source_identity(),
                runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__))


def train_model(config, resume=None):
    """Lock the selected run before validating inputs or writing any training output."""
    run_dir = resolve_path(config['logging']['save_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / 'train.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f'Baseline training is already running in {run_dir}') from error
        return _train_model(config, resume)


def _train_model(config, resume=None):
    cfg = prepare_config(config, verify_data=True)
    if cfg.get('model_type', 'deterministic_unet') != 'deterministic_unet':
        raise ValueError('Training configuration must describe the deterministic U-Net baseline')
    cfg['model_type'] = 'deterministic_unet'
    cfg.setdefault('baseline', {}).update(name=NAME, zero_lidar_input=True, fixed_timestep=0)
    cfg['baseline'].setdefault('seed', 42)
    tr = cfg['training']
    if tr['num_workers'] < 0 or tr['batch_size'] < 1 or tr['epochs'] < 1:
        raise ValueError('Workers must be nonnegative; batch size and epochs must be positive')
    if tr['lr'] <= 0 or tr['loss']['name'] not in LOSSES:
        raise ValueError('Invalid learning rate or loss')
    device_name = cfg['system'].get('device', 'auto')
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu')
    cfg['system']['device'] = device_name
    device = torch.device(device_name)
    torch.backends.cudnn.benchmark = True
    split_ids = read_split_ids(cfg['data']['split_ids'])
    train_ids = [pid for r in ('pondinlet', 'tuk') for pid in split_ids[r]['train']]
    validation_ids = [pid for r in ('pondinlet', 'tuk') for pid in split_ids[r]['validation']]
    if cfg['system'].get('debug'):
        train_ids, validation_ids = train_ids[:100], validation_ids[:100]
        cfg['baseline']['training_scope'] = 'debug_subset'
        cfg['baseline'].pop('training_split_ids_sha256', None)
    else:
        cfg['baseline']['training_scope'] = 'full_primary'
        cfg['baseline']['training_split_ids_sha256'] = cfg['data']['split_ids_sha256']
    cfg['baseline']['training_counts'] = dict(train=len(train_ids), validation=len(validation_ids))
    run_dir = resolve_path(cfg['logging']['save_dir'])
    cfg['logging']['save_dir'] = str(run_dir)
    latest_path, best_path = run_dir / f'{NAME}_latest.pth', run_dir / f'{NAME}_best.pth'
    resume_path = latest_path if resume == 'auto' and latest_path.exists() else (
        None if resume in (None, 'auto') else resolve_path(resume))
    existing = [run_dir / name for name in ('config.json', 'log.csv', 'split_ids.json')]
    if resume_path is None and (best_path.exists() or latest_path.exists() or any(p.exists() for p in existing)):
        raise FileExistsError('Run already has output; use explicit --resume or a new --save-dir')
    if resume_path is not None and resume_path.parent != run_dir and (
            best_path.exists() or latest_path.exists() or any(p.exists() for p in existing)):
        raise FileExistsError('An external recovery checkpoint requires a new empty --save-dir')
    if resume_path is not None and tr['num_workers'] > 0:
        raise ValueError('Exact recovery requires a checkpoint trained with --num-workers 0; persistent worker RNG states are not saved')
    set_seed(cfg['baseline']['seed'])
    train_dataset = make_dataset(cfg, train_ids, 'train', augment=True)
    val_dataset = make_dataset(cfg, validation_ids, 'validation', augment=False)
    train_loader = make_loader(train_dataset, cfg)
    val_loader = make_loader(val_dataset, cfg, validation=True)
    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tr['lr'])
    criterion = LOSSES[tr['loss']['name']]
    signature = training_signature(cfg)
    start_epoch, best_val, best_state, recovered_rng = 0, float('inf'), None, None
    train_losses, val_losses = [], []
    if resume_path is not None:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state.get('resume_signature') != signature or 'rng_state' not in state:
            raise ValueError('Checkpoint lacks matching recoverable training provenance; legacy weights remain usable for evaluation')
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        start_epoch, best_val = state['epoch'], state['best_val_loss']
        train_losses, val_losses = list(state['train_losses']), list(state['val_losses'])
        if len(train_losses) != start_epoch or len(val_losses) != start_epoch:
            raise ValueError('Recovery checkpoint loss history does not match its completed epoch')
        best_state = state.get('best_state')
        if best_state is None and state.get('val_loss') == best_val:
            best_state = copy.deepcopy(state)
        if start_epoch and best_state is None:
            raise ValueError('Recovery checkpoint does not contain its selected best state')
        recovered_rng = state['rng_state']
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / 'config.json').exists():
        (run_dir / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    if not (run_dir / 'split_ids.json').exists():
        (run_dir / 'split_ids.json').write_text(json.dumps(split_ids, indent=2) + '\n')
    log_path = run_dir / 'log.csv'
    logged = []
    if log_path.exists():
        with log_path.open(newline='') as stream:
            logged = list(csv.DictReader(stream))
    if resume_path is not None:
        previous = {int(r['epoch']): r for r in logged}
        retained = [dict(epoch=i + 1, train_loss=f'{a:.6f}', val_loss=f'{b:.6f}',
                         epoch_sec=previous.get(i + 1, {}).get('epoch_sec', ''))
                    for i, (a, b) in enumerate(zip(train_losses, val_losses))]
        with log_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'val_loss', 'epoch_sec'])
            writer.writeheader(); writer.writerows(retained)
    elif not log_path.exists():
        log_path.write_text('epoch,train_loss,val_loss,epoch_sec\n')
    if best_state is not None:
        atomic_save(best_state, best_path)
    wandb = None
    if cfg['baseline'].get('wandb'):
        import wandb
        wandb.init(project=cfg['logging']['wandb_project'], name=cfg['logging']['run_name'], config=cfg)
    if recovered_rng is not None:
        restore_rng(recovered_rng)
    print(f'ConditionalUNet params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | '
          f'train {len(train_dataset)} | validation {len(val_dataset)} | device {device}')
    if resume_path is None:
        atomic_save(dict(epoch=0, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                         config=cfg, best_val_loss=best_val, resume_signature=signature, rng_state=rng_state(),
                         train_losses=[], val_losses=[], best_state=None), latest_path)
    for epoch in range(start_epoch, tr['epochs']):
        started = time.perf_counter()
        model.train(); total = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{tr['epochs']}"):
            lidar = batch['lidar'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            prediction = predict(model, batch['s2'].to(device, non_blocking=True),
                                 batch['attrs'].to(device, non_blocking=True))
            loss = compute_loss(criterion, tr['loss']['name'], prediction, lidar, mask, tr['loss']['alpha'])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)
        model.eval(); total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                prediction = predict(model, batch['s2'].to(device), batch['attrs'].to(device))
                total += compute_loss(criterion, tr['loss']['name'], prediction,
                                      batch['lidar'].to(device), batch['mask'].to(device), tr['loss']['alpha']).item()
        val_loss = total / len(val_loader)
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise ValueError('Non-finite training/validation loss; refusing to save this epoch')
        train_losses.append(train_loss); val_losses.append(val_loss)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
        state = dict(epoch=epoch + 1, model_state_dict=model.state_dict(),
                     optimizer_state_dict=optimizer.state_dict(), val_loss=val_loss,
                     best_val_loss=best_val, config=cfg, resume_signature=signature,
                     rng_state=rng_state(), train_losses=train_losses, val_losses=val_losses)
        if improved:
            best_state = copy.deepcopy(state)
            atomic_save(best_state, best_path)
        atomic_save(dict(state, best_state=best_state), latest_path)
        duration = time.perf_counter() - started
        with log_path.open('a') as stream:
            stream.write(f'{epoch + 1},{train_loss:.6f},{val_loss:.6f},{duration:.1f}\n')
        if wandb:
            wandb.log({f"train_{tr['loss']['name']}": train_loss,
                       f"val_{tr['loss']['name']}": val_loss, 'epoch': epoch})
        print(f'Epoch {epoch + 1}: train {train_loss:.5f} | val {val_loss:.5f} | {duration:.0f}s')
    if wandb:
        wandb.finish()
    print(f'Best validation checkpoint: {best_path}')
    print(f'Evaluate measured paper metrics with scripts/evaluate_unet_baseline.py --checkpoint {best_path}')
    return dict(best=best_path, latest=latest_path)


def main(argv=None):
    args = parse_arguments(argv)
    if args.eval_only:
        from scripts.evaluate_unet_baseline import main as evaluate_main
        command = ['--checkpoint', str(resolve_path(args.eval_only)), '--split', 'validation']
        for flag, value in [('--data-root', args.data_root), ('--split-ids', args.split_ids),
                            ('--manifest', args.manifest), ('--legacy-splits', args.legacy_splits),
                            ('--device', args.device)]:
            if value is not None:
                command += [flag, str(value)]
        return evaluate_main(command)
    if args.legacy_splits:
        raise ValueError('Training uses canonical JSON --split-ids; --legacy-splits/--splits is only for legacy checkpoint evaluation')
    return train_model(build_config(args), resume=args.resume)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(str(error)) from error
