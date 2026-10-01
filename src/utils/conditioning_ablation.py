"""Paired conditioning ablation utilities, separate from historical evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluation import _compute_metrics_tensor, _metric_params_from_cfg

MODES = ("normal", "shuffle", "null")
METRICS = ("rmse_m", "bias_m", "sigma_error_pct", "nae_deg", "zncc", "jsd", "log_psd_rmse", "nrmse_sigma")


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "little") % (2**63 - 1)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    """Atomic replacement; interrupted writes cannot masquerade as completed shards."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def disjoint(a, b):
    """Touching footprint edges are allowed, shared area is not."""
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def donor_permutation(records, seed, min_distance_m=512.0):
    """One-to-one within-region shuffle, excluding self and overlapping footprints.

    Repairs a random permutation by swaps. This is a constrained random control,
    not a claim of uniform sampling over all admissible permutations.
    """
    n = len(records)
    if n < 2:
        raise ValueError("At least two spatially separate patches are required")
    rng = np.random.default_rng(seed)

    def valid(i, j):
        a, b = records[i]["footprint"], records[j]["footprint"]
        ca = records[i].get("center", [(a[0]+a[2])/2, (a[1]+a[3])/2])
        cb = records[j].get("center", [(b[0]+b[2])/2, (b[1]+b[3])/2])
        distance = np.hypot(ca[0]-cb[0], ca[1]-cb[1])
        return i != j and disjoint(a, b) and distance >= min_distance_m

    for _ in range(100):
        perm = rng.permutation(n)
        for i in range(n):
            if valid(i, int(perm[i])):
                continue
            for j in rng.permutation(n):
                if i != j and valid(i, int(perm[j])) and valid(int(j), int(perm[i])):
                    perm[i], perm[j] = perm[j], perm[i]
                    break
        if all(valid(i, int(j)) for i, j in enumerate(perm)):
            return [int(j) for j in perm]
    raise ValueError("Cannot construct non-overlapping donor permutation; increase the patch pool")


def paired_noise(region, tile_ids, replicate, shape):
    """CPU-generated x_T is independent of batching, mode, and shuffle RNG."""
    return torch.stack([
        torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(
            stable_seed("initial_noise", region, tile_id, replicate)))
        for tile_id in tile_ids
    ])


def condition_inputs(mode, target, donor):
    if mode == "normal":
        return target["s2"], target["attrs"]
    if mode == "shuffle":
        return donor["s2"], donor["attrs"]
    if mode == "null":
        return torch.zeros_like(target["s2"]), torch.zeros_like(target["attrs"])
    raise ValueError(mode)


@torch.no_grad()
def patch_metrics(gt, pred, mask, config):
    """Use the paper's existing evaluator, including its spatial-mask conventions.

    Dataset targets are already demeaned. Set invalid GT pixels to NaN, matching
    evaluation.py's GeoTIFF reader, and leave predictions in residual space.
    In particular, do not independently demean predictions or redefine JSD/PSD.
    """
    gt, pred, mask = gt.squeeze(), pred.squeeze(), mask.squeeze().bool()
    if gt.ndim != 2 or pred.shape != gt.shape or mask.shape != gt.shape:
        raise ValueError("Expected matching 2D fields and validity mask")
    if not mask.any() or not torch.isfinite(gt[mask]).all() or not torch.isfinite(pred).all():
        raise ValueError("Empty support or non-finite target/prediction")
    gt = torch.where(mask, gt, torch.nan)
    values = _compute_metrics_tensor(gt, pred, mask, *_metric_params_from_cfg(config),
                                     use_absrel_epsilon=False)
    names = {
        "rmse_m": "rmse_phys_m", "bias_m": "bias_phys_m", "sigma_error_pct": "sigma_error_pct",
        "nae_deg": "normal_angle_error_deg", "zncc": "zncc", "jsd": "jsd",
        "log_psd_rmse": "psd_rmse", "nrmse_sigma": "nrmse_std",
    }
    return {name: float(values[key]) if np.isfinite(values[key]) else None for name, key in names.items()}


def save_predictions(path, gt, mask, predictions, patch_mean):
    """Atomic float32 archive; preserve the exact arrays used for scoring."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, gt=gt.numpy().astype(np.float32), mask=mask.numpy().astype(bool),
                            lidar_patch_mean=float(patch_mean),
                            **{mode: pred.numpy().astype(np.float32) for mode, pred in predictions.items()})
    tmp.replace(path)


def block_interval(values, centers, block_m, seed, repetitions=2000):
    """Spatial cluster bootstrap of a patch mean, conditional on averaged seeds.

    Resample blocks with replacement, retaining all patches and their original
    weights inside each selected block. Do not treat pixels or seeds as spatial
    replicates. Fewer than five occupied blocks is reported without an interval.
    """
    values, centers = np.asarray(values, float), np.asarray(centers, float)
    if block_m is None:
        values = values[np.isfinite(values)]
        return {"mean": float(values.mean()) if len(values) else None,
                "ci_low": None, "ci_high": None, "n_patches": len(values), "n_blocks": None}
    keep = np.isfinite(values)
    values, centers = values[keep], centers[keep]
    if len(values) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_patches": 0, "n_blocks": 0}
    grid = np.floor(centers / block_m).astype(np.int64)
    _, groups = np.unique(grid, axis=0, return_inverse=True)
    counts = np.bincount(groups)
    sums = np.bincount(groups, weights=values)
    n = len(counts)
    result = {"mean": float(values.mean()), "ci_low": None, "ci_high": None,
              "n_patches": len(values), "n_blocks": n}
    if n >= 5:
        rng = np.random.default_rng(seed)
        means = []
        for _ in range(repetitions):
            draw = rng.integers(n, size=n)
            means.append(sums[draw].sum() / counts[draw].sum())
        result["ci_low"], result["ci_high"] = map(float, np.quantile(means, [0.025, 0.975]))
    return result
