"""Atomic, epoch-boundary training recovery. Load only trusted local checkpoints."""
import os
from pathlib import Path
import random
import numpy as np
import torch


def atomic_save(state, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available():
            raise ValueError('CUDA checkpoint requires CUDA for reproducible recovery')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def resume_signature(config):
    # Operational output paths may move; scientific and loader configuration must match.
    return dict(model=config['model'], training=config['training'],
                split_digest=config['data'].get('split_digest'),
                data={k:v for k,v in config['data'].items() if k not in ('split_manifest','split_digest')},
                system=config['system'])
