"""Reproducible final-checkpoint null/shuffle study. See docs/conditioning_ablation.md."""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import torch
from torch.utils.data import default_collate

from src.data.dataset import LidarS2Dataset
from src.diffusion.sampling import p_sample_loop_plms
from src.diffusion.scheduler import CosineDiffusionScheduler
from src.model.unet import ConditionalUNet
from src.utils.conditioning_ablation import (
    MODES, METRICS, stable_seed, sha256_file, json_digest, write_json,
    donor_permutation, paired_noise, condition_inputs, patch_metrics, block_interval, save_predictions,
)

REGIONS = {"pondinlet": 4, "tuk": 13, "cambridge": None}


def source_hashes():
    paths = [Path(__file__), ROOT / "scripts/evaluation.py", *ROOT.joinpath("src").rglob("*.py")]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def fingerprint(path):
    s = path.stat()
    return {"path": str(path), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def check_config(cfg):
    expected = {
        ("training", "context_k"): 6, ("training", "noise_schedule"): "cosine",
        ("training", "timesteps"): 1000, ("training", "lr"): 1e-4,
        ("training", "randomize_context"): True,
        ("model", "base_channels"): 64, ("model", "unet_depth"): 4,
        ("model", "attention_variant"): "mid",
    }
    errors = [f"{a}.{b}: expected {v}, got {cfg.get(a, {}).get(b)}"
              for (a, b), v in expected.items() if cfg.get(a, {}).get(b) != v]
    if cfg["training"].get("loss") != {"name": "masked_hybrid_mse_loss", "alpha": 1.0}:
        errors.append("Expected masked_hybrid_mse_loss with alpha=1.0")
    if sorted(cfg["data"].get("validation_regions", [])) != [4, 13]:
        errors.append("Expected held-out validation zones 4 and 13")
    if errors:
        raise ValueError("Checkpoint does not match the supplied final-model protocol:\n" + "\n".join(errors))


def prepare(args):
    out = args.out.resolve()
    if (out / "manifest.json").exists():
        raise FileExistsError("Manifest already exists; reuse it or choose a new output directory")
    checkpoint = args.checkpoint.resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    check_config(cfg)
    manifest = {
        "schema": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": saved.get("epoch"), "checkpoint_config": cfg,
        "sources": source_hashes(), "seeds": args.seeds,
        "sampler": "repository_plms_order4", "timesteps": 1000,
        "modes": list(MODES), "subset_max_patches": args.max_patches,
        "metric_protocol": "paper_evaluation_v1", "minimum_donor_distance_m": 512.0,
        "prediction_archive": "float32_npz", "data_root": str(args.data_root.resolve()),
        "regions": {},
    }
    del saved
    for region, zone in REGIONS.items():
        print(f"Preparing {region} (zone={zone})...", flush=True)
        s2dir = args.data_root.resolve() / f"s2_patches_{region}"
        ldir = args.data_root.resolve() / f"lidar_patches_{region}"
        candidates = []
        for folder in sorted(s2dir.glob("s2_patch_*")):
            pid = folder.name.removeprefix("s2_patch_")
            zone_file = folder / "region.json"
            rid = json.loads(zone_file.read_text())["region_id"]
            if zone is None or rid == zone:
                candidates.append((pid, folder, rid))
        eligible = len(candidates)
        if args.max_patches is not None and eligible > args.max_patches:
            rng = np.random.default_rng(stable_seed("subset", region, 2026))
            indices = sorted(rng.choice(eligible, args.max_patches, replace=False))
            candidates = [candidates[i] for i in indices]
        if len(candidates) < 2:
            raise ValueError(f"Insufficient patches in {s2dir}")
        records, reference_crs = [], None
        for pid, folder, rid in candidates:
            lidar = ldir / f"lidar_patch_{pid}.tif"
            files = [lidar, folder / "region.json", folder / "attrs.json", *[folder / f"t{i}.tif" for i in range(6)]]
            signatures = [fingerprint(p) for p in files]  # missing files are fatal
            attrs = json.loads((folder / "attrs.json").read_text())
            if len(attrs) != 6 or any(a is None for a in attrs):
                raise ValueError(f"Expected six complete metadata records: {folder}")
            with rasterio.open(lidar) as src:
                if src.shape != (256, 256) or not np.allclose(src.res, (1, 1)) or src.count != 2:
                    raise ValueError(f"Expected 256x256 two-band data/mask at 1 m: {lidar}")
                if not src.crs.is_projected or src.crs.linear_units != "metre":
                    raise ValueError(f"Expected metre-based projected CRS: {lidar}")
                crs = src.crs
                if reference_crs is None:
                    reference_crs = crs
                if crs != reference_crs:
                    raise ValueError("All patches within a region must share a CRS")
                bounds = list(src.bounds)
                # Read masks/targets once now to fail before a long inference run.
                data = src.read()
                valid = data[1] > 0.5
                if not valid.any() or not np.isfinite(data[0]).all():
                    raise ValueError(f"Invalid target values or empty mask: {lidar}")
            footprint = bounds.copy()
            for p in files[3:]:
                with rasterio.open(p) as src:
                    if src.count < 4 or src.crs is None:
                        raise ValueError(f"Missing bands or georeferencing: {p}")
                    b = transform_bounds(src.crs, crs, *src.bounds)
                footprint = [min(footprint[0], b[0]), min(footprint[1], b[1]),
                             max(footprint[2], b[2]), max(footprint[3], b[3])]
            records.append({"tile_id": pid, "zone_id": rid, "bounds": bounds,
                            "footprint": footprint, "center": [(bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2],
                            "valid_pixels": int(valid.sum()), "files": signatures})
        permutations = {str(seed): donor_permutation(records, stable_seed("donors", region, seed)) for seed in args.seeds}
        manifest["regions"][region] = {"zone": zone, "eligible_patches": eligible,
            "crs": str(reference_crs), "lidar_dir": str(ldir), "s2_dir": str(s2dir),
            "records": records, "donor_indices": permutations,
            "visual_tile_ids": [r["tile_id"] for r in records[:3]]}
        print(f"  {len(records)}/{eligible} patches, {len(permutations)} saved donor permutations", flush=True)
    write_json(out / "manifest.json", manifest)
    n = sum(len(s["records"]) for s in manifest["regions"].values())
    print(f"Planned: {n} targets x {len(args.seeds)} seeds x {len(MODES)} modes = {n*len(args.seeds)*len(MODES)} reconstructions.")
    print(f"Prepared {out / 'manifest.json'}; no inference performed.")


def load_manifest(out):
    path = out / "manifest.json"
    return json.loads(path.read_text())


def verify_inputs(manifest, regions):
    if source_hashes() != manifest["sources"]:
        raise ValueError("Source files changed since prepare; create a new study manifest")
    if sha256_file(manifest["checkpoint"]) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint content changed since prepare")
    for region in regions:
        for rec in manifest["regions"][region]["records"]:
            for signature in rec["files"]:
                if fingerprint(Path(signature["path"])) != signature:
                    raise ValueError(f"Data changed since prepare: {signature['path']}")


def run(args):
    out = args.out.resolve()
    manifest = load_manifest(out)
    regions = args.regions or list(REGIONS)
    seeds = args.seeds or manifest["seeds"]
    if any(s not in manifest["seeds"] for s in seeds):
        raise ValueError("Run seeds must be in the prepared manifest")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; full study requires a GPU host. CPU smoke tests use --device cpu.")
    verify_inputs(manifest, regions)
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    runtime = {"manifest_sha256": json_digest(manifest), "batch_size": args.batch_size,
               "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
               "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
               "deterministic_algorithms": True, "dtype": "float32", "cpu_threads": args.cpu_threads}
    runtime_path = out / "runtime.json"
    if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
        raise ValueError("Runtime/batch settings changed; use the original settings or a new study directory")
    write_json(runtime_path, runtime)
    cfg = manifest["checkpoint_config"]
    model = ConditionalUNet(in_channels=1, cond_channels=24, attr_dim=48, cond_k=6, **cfg["model"]).to(device)
    saved = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    del saved
    model.eval()
    scheduler = CosineDiffusionScheduler(timesteps=manifest["timesteps"], device=device)
    shards = out / "shards"
    shards.mkdir(exist_ok=True)
    completed_batches = 0
    for region in regions:
        spec = manifest["regions"][region]
        records = spec["records"]
        ids = [r["tile_id"] for r in records]
        dataset = LidarS2Dataset(spec["lidar_dir"], spec["s2_dir"], context_k=6,
                                randomize_context=False, augment=False, split="val", split_pids=ids)
        if [s["tile_id"] for s in dataset.samples] != ids:
            raise ValueError("Dataset silently omitted or reordered prepared patches")
        for seed in seeds:
            donors = spec["donor_indices"][str(seed)]
            for start in range(0, len(ids), args.batch_size):
                target_ids = ids[start:start+args.batch_size]
                path = shards / f"{region}_seed{seed}_{start:06d}.json"
                if path.exists():
                    shard = json.loads(path.read_text())
                    validate_shard(shard, manifest, runtime, region, seed, target_ids, out)
                    continue
                lock = path.with_suffix(".lock")
                # A crashed job leaves a lock: inspect/remove that lock before resuming.
                with lock.open("x") as f:
                    f.write(f"pid={os.getpid()}\n")
                try:
                    before = time.monotonic()
                    target = default_collate([dataset[i] for i in range(start, start+len(target_ids))])
                    donor = default_collate([dataset[donors[i]] for i in range(start, start+len(target_ids))])
                    for batch in (target, donor):
                        if any(not torch.isfinite(batch[k]).all() for k in ("s2", "attrs", "lidar", "mask")):
                            raise ValueError("Non-finite prepared input; refusing to start sampling")
                    noise = paired_noise(region, target_ids, seed, target["lidar"].shape[1:]).to(device)
                    predictions, rows, archives = {}, [], []
                    with torch.inference_mode():
                        for mode in MODES:
                            cond, attrs = condition_inputs(mode, target, donor)
                            pred = p_sample_loop_plms(model, scheduler, noise.shape, cond.to(device),
                                attrs.to(device), device, initial_noise=noise)
                            predictions[mode] = pred.cpu()
                        # Metrics on CPU for stable support handling; no ground-truth mean is added back.
                        for i, pid in enumerate(target_ids):
                            mask = target["mask"][i] > 0.5
                            for mode in MODES:
                                values = patch_metrics(target["lidar"][i], predictions[mode][i], mask, cfg)
                                rows.append({"region": region, "seed": seed, "tile_id": pid, "mode": mode,
                                    "donor_tile_id": ids[donors[start+i]] if mode == "shuffle" else None,
                                    "valid_pixels": int(mask.sum()), **values})
                            archive = Path("predictions") / region / f"seed{seed}" / f"{pid}.npz"
                            save_predictions(out / archive, target["lidar"][i, 0], mask,
                                             {m: p[i, 0] for m, p in predictions.items()},
                                             target["lidar_patch_mean"][i])
                            archives.append({"path": str(archive), "sha256": sha256_file(out / archive)})
                    elapsed = time.monotonic() - before
                    write_json(path, {"runtime_sha256": json_digest(runtime), "rows": rows, "elapsed_s": elapsed, "archives": archives})
                    print(f"{region} seed={seed}: {start+len(target_ids)}/{len(ids)} patches ({elapsed:.1f}s)", flush=True)
                finally:
                    lock.unlink()
                completed_batches += 1
                if args.max_batches is not None and completed_batches >= args.max_batches:
                    print("Stopped after requested batch limit; rerun without --max-batches to resume.", flush=True)
                    return


def validate_shard(shard, manifest, runtime, region, seed, ids, out=None):
    if shard["runtime_sha256"] != json_digest(runtime):
        raise ValueError("Shard provenance does not match this run")
    if out is not None:
        archives = shard.get("archives", [])
        expected_archives = {str(Path("predictions") / region / f"seed{seed}" / f"{pid}.npz") for pid in ids}
        if len(archives) != len(ids) or {a["path"] for a in archives} != expected_archives:
            raise ValueError("Missing prediction archives in completed shard")
        for archive in archives:
            path = out / archive["path"]
            if not path.exists() or sha256_file(path) != archive["sha256"]:
                raise ValueError(f"Missing or changed prediction archive: {path}")
    rows = shard["rows"]
    expected = {(region, seed, pid, mode) for pid in ids for mode in MODES}
    keys = [(r["region"], r["seed"], r["tile_id"], r["mode"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Incomplete/duplicate shard rows")
    spec = manifest["regions"][region]
    lookup = {r["tile_id"]: i for i, r in enumerate(spec["records"])}
    for row in rows:
        i = lookup[row["tile_id"]]
        donor_id = spec["records"][spec["donor_indices"][str(seed)][i]]["tile_id"]
        if row["donor_tile_id"] != (donor_id if row["mode"] == "shuffle" else None):
            raise ValueError("Donor mapping differs from the manifest")
        if any(k not in row for k in METRICS):
            raise ValueError("Missing metrics in shard")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(args):
    out = args.out.resolve()
    manifest = load_manifest(out)
    runtime = json.loads((out / "runtime.json").read_text())
    if runtime["manifest_sha256"] != json_digest(manifest):
        raise ValueError("Runtime manifest mismatch")
    rows = []
    expected_paths = set()
    for region, spec in manifest["regions"].items():
        ids = [r["tile_id"] for r in spec["records"]]
        for seed in manifest["seeds"]:
            for start in range(0, len(ids), runtime["batch_size"]):
                path = out / "shards" / f"{region}_seed{seed}_{start:06d}.json"
                expected_paths.add(path.name)
                if not path.exists():
                    raise ValueError(f"Study incomplete; missing {path.name}. Partial results cannot be a final table.")
                shard = json.loads(path.read_text())
                validate_shard(shard, manifest, runtime, region, seed, ids[start:start+runtime["batch_size"]], out)
                rows.extend(shard["rows"])
    if {p.name for p in (out / "shards").glob("*.json")} != expected_paths:
        raise ValueError("Unexpected shard files in output directory")
    write_csv(out / "per_patch.csv", rows)
    lookup = {(r["region"], r["seed"], r["tile_id"], r["mode"]): r for r in rows}
    summaries, deltas = [], []
    for region, spec in manifest["regions"].items():
        ids = [r["tile_id"] for r in spec["records"]]
        centers = [r["center"] for r in spec["records"]]
        for metric in METRICS:
            arrays = {mode: np.array([[lookup[region, seed, pid, mode][metric]
                         for pid in ids] for seed in manifest["seeds"]], dtype=float) for mode in MODES}
            # Require all conditions and repeats to be defined, to keep support paired.
            valid = np.all(np.isfinite(np.stack(list(arrays.values()))), axis=(0, 1))
            for mode, arr in arrays.items():
                patch_avg = np.where(valid, arr.mean(axis=0), np.nan)
                seed_means = arr[:, valid].mean(axis=1) if valid.any() else np.full(len(manifest["seeds"]), np.nan)
                for block_m in (args.block_sizes or [None]):
                    stat = block_interval(patch_avg, centers, block_m, stable_seed("bootstrap", region, metric, block_m), args.bootstrap)
                    summaries.append({"region": region, "mode": mode, "metric": metric, "block_m": block_m,
                        **stat, "excluded_patches": int((~valid).sum()), "n_seeds": len(manifest["seeds"]),
                        "seed_mean_sd": float(seed_means.std(ddof=1)) if len(seed_means)>1 and valid.any() else None})
                if mode != "normal":
                    diff = arr - arrays["normal"]
                    per_patch = np.where(valid, diff.mean(axis=0), np.nan)
                    seed_diffs = diff[:, valid].mean(axis=1) if valid.any() else np.full(len(manifest["seeds"]), np.nan)
                    for block_m in (args.block_sizes or [None]):
                        stat = block_interval(per_patch, centers, block_m, stable_seed("bootstrap", region, metric, block_m), args.bootstrap)
                        deltas.append({"region": region, "contrast": f"{mode}-normal", "metric": metric,
                            "block_m": block_m, **stat, "excluded_patches": int((~valid).sum()),
                            "seed_delta_sd": float(seed_diffs.std(ddof=1)) if len(seed_diffs)>1 and valid.any() else None,
                            "direction": "negative_is_worse" if metric == "zncc" else "signed_only" if metric == "bias_m" else "positive_is_worse"})
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "paired_deltas.csv", deltas)
    write_json(out / "summary.json", {"subset_only": manifest["subset_max_patches"] is not None,
        "manifest_sha256": json_digest(manifest), "bootstrap_repetitions": args.bootstrap if args.block_sizes else 0,
        "ci_scope": "optional spatial block bootstrap, conditional on chosen seeds" if args.block_sizes else "not requested",
        "summary": summaries, "paired_deltas": deltas})
    display_block = args.block_sizes[0] if args.block_sizes else None
    display = {(r["region"], r["mode"], r["metric"]): r for r in summaries if r["block_m"] == display_block}
    lines = ["% Generated; patch means and paired differences, using the existing paper metrics.",
             "% SUBSET / PIPELINE CHECK ONLY" if manifest["subset_max_patches"] else "% Full prepared study",
             r"\begin{tabular}{llrrrrrrrr}", r"\hline",
             r"Region & Condition & RMSE & Bias & $\sigma$ error & NAE & ZNCC & JSD & PSD & nRMSE$_\sigma$ \\", r"\hline"]
    for region in manifest["regions"]:
        for mode in MODES:
            vals = [display[region, mode, metric]["mean"] for metric in METRICS]
            lines.append(" & ".join([region, mode, *[f"{v:.4f}" if v is not None else "--" for v in vals]]) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    (out / "table.tex").write_text("\n".join(lines) + "\n")
    if args.plot:
        plot_deltas(out, deltas, manifest, display_block)
    print(f"Saved complete paired summaries and table to {out}")


def plot_deltas(out, deltas, manifest, block_m):
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = [("nrmse_sigma", "nRMSE / sigma"), ("sigma_error_pct", "Sigma error (percentage points)"),
               ("zncc", "ZNCC"), ("jsd", "JSD"), ("log_psd_rmse", "log-PSD RMSE")]
    regions = list(manifest["regions"])
    lookup = {(r["region"], r["contrast"], r["metric"]): r for r in deltas if r["block_m"] == block_m}
    fig, axes = plt.subplots(len(regions), len(columns), figsize=(16, 2.6*len(regions)), squeeze=False)
    for i, region in enumerate(regions):
        for j, (metric, title) in enumerate(columns):
            ax = axes[i, j]
            for y, contrast, color in [(1, "shuffle-normal", "#1764ab"), (0, "null-normal", "#bc4932")]:
                r = lookup[region, contrast, metric]
                if r["mean"] is None:
                    continue
                if r["ci_low"] is not None:
                    ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=color, linewidth=2)
                ax.plot(r["mean"], y, "o", color=color)
            ax.axvline(0, color="0.5", linestyle="--", linewidth=0.8)
            ax.set_yticks([0, 1], ["Null", "Shuffle"])
            ax.set_ylim(-0.6, 1.6)
            ax.set_xlabel("Ablation minus normal")
            ax.set_title(title + (" (lower is worse)" if metric == "zncc" else " (higher is worse)"), fontsize=9)
            if j == 0:
                ax.set_ylabel(region)
            ax.spines[["top", "right"]].set_visible(False)
    prefix = "SUBSET CHECK: " if manifest["subset_max_patches"] else ""
    interval_label = f"; 95% spatial intervals ({block_m} m blocks)" if block_m else ""
    fig.suptitle(prefix + "Paired conditioning effects" + interval_label)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"paired_deltas.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Lock checkpoint, patch cohort, and donor assignments")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=ROOT / "input_data")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--max-patches", type=positive, help="Subset for pipeline checks only")
    p = sub.add_parser("run", help="Run/resume full 1000-step paired PLMS inference")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=positive, default=8)
    p.add_argument("--cpu-threads", type=positive, default=4)
    p.add_argument("--max-batches", type=positive, help="Stop cleanly after this many new batches (benchmark/resume)")
    p.add_argument("--regions", nargs="+", choices=list(REGIONS))
    p.add_argument("--seeds", nargs="+", type=int)
    p = sub.add_parser("summarize", help="Validate completion and export region-specific tables")
    p.add_argument("--block-sizes", nargs="+", type=positive, help="Optional spatial bootstrap, e.g. 1024")
    p.add_argument("--plot", action="store_true", help="Optional paired-effect figure")
    p.add_argument("--bootstrap", type=positive, default=2000)
    for p in sub.choices.values():
        p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, "seeds", None) and len(args.seeds) != len(set(args.seeds)):
        parser.error("Seeds must be unique")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
