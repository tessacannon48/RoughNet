#!/usr/bin/env python3
"""Score the deterministic U-Net with the paper's shared patch metrics.

Unlike the supplied training script's original validation diagnostics, these
outputs use the same eight metric definitions and equal patch weighting as the
RoughNet comparisons. No diffusion sampler or prediction/ground-truth cache is
created. Load only checkpoints from a trusted source.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "roughnet-mpl"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.conditioning_ablation import METRICS, patch_metrics, write_json
from src.utils.spatial_split import digest, sha256_file, validate_manifest
from src.utils.unet_baseline import (
    load_checkpoint, make_dataset, predict, prepare_config, read_split_ids,
    resolve_path,
)


def validate_training_identity(config, split_ids, split_path, legacy_splits=None):
    """Verify recorded training membership without inventing old provenance.

    Integrated checkpoints retain the canonical JSON's SHA-256. The colleague's
    original checkpoint format instead retains the MD5 of a CSV; require that
    exact file and prove its train/validation membership against the canonical
    split. Test and excluded rows were not used by the trainer and are optional;
    any such rows that are present must still have their canonical roles.
    """
    baseline = config.get("baseline", {})
    expected = baseline.get("training_split_ids_sha256")
    current = sha256_file(split_path)
    if expected is not None:
        if expected != current:
            raise ValueError("Checkpoint training split differs from --split-ids")
        return {"method": "checkpoint_training_json_sha256", "sha256": current}
    if legacy_splits is None:
        raise ValueError(
            "Legacy checkpoint has no verified JSON training identity. Supply "
            "--legacy-splits with its original CSV split file; the saved MD5 and "
            "all canonical memberships must agree."
        )
    expected_md5 = baseline.get("splits_md5")
    if not expected_md5:
        raise ValueError("Legacy checkpoint does not record its training split CSV checksum")
    legacy_splits = Path(legacy_splits)
    if hashlib.md5(legacy_splits.read_bytes()).hexdigest() != expected_md5:
        raise ValueError("Original split CSV does not match checkpoint splits_md5")
    expected_rows = {}
    for region, partitions in split_ids.items():
        for role, ids in partitions.items():
            for pid in ids:
                if pid in expected_rows:
                    raise ValueError("Canonical patch IDs must be globally unique")
                expected_rows[pid] = (region, role)
    observed = {}
    with legacy_splits.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"tile_id", "split"} <= set(reader.fieldnames or []):
            raise ValueError("Legacy split CSV requires tile_id and split columns")
        for row in reader:
            pid = row["tile_id"]
            role = "validation" if row["split"] == "val" else row["split"]
            if pid in observed or pid not in expected_rows:
                raise ValueError("Duplicate or unexpected patch in original split CSV")
            region, expected_role = expected_rows[pid]
            if role != expected_role or (row.get("region") and row["region"] != region):
                raise ValueError("Original split CSV membership differs from the paper split")
            observed[pid] = (region, role)
    expected_training = {pid: value for pid, value in expected_rows.items()
                         if value[1] in ("train", "validation")}
    observed_training = {pid: value for pid, value in observed.items()
                         if value[1] in ("train", "validation")}
    if observed_training != expected_training:
        raise ValueError("Original split CSV omits canonical training or validation memberships")
    return {
        "method": "legacy_csv_checksum_and_canonical_memberships",
        "original_md5": expected_md5,
        "original_csv_sha256": sha256_file(legacy_splits),
        "canonical_json_sha256": current,
        "training_input_content_identity": "not recorded by the supplied legacy script",
    }


def verify_evaluation_inputs(config, split_ids, recorded_config):
    """Validate a supplied input inventory without retroactively binding old weights."""
    data = config["data"]
    manifest_path = data.get("split_manifest")
    recorded_digest = recorded_config["data"].get("split_digest")
    bound_training = bool(recorded_config.get("baseline", {}).get("training_split_ids_sha256"))
    if bound_training and not recorded_digest:
        raise ValueError("Integrated checkpoint is missing its verified training input identity")
    if manifest_path is None or not Path(manifest_path).is_file():
        if bound_training:
            raise ValueError("Provide --manifest with this checkpoint's portable input inventory")
        return {
            "status": "no_input_content_inventory_available",
            "training_content_binding": "not recorded by the supplied legacy script",
        }
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 2:
        raise ValueError("Evaluation requires a content-hashed portable input inventory")
    validate_manifest(manifest, root=ROOT, data_root=data["data_root"])
    for region, partitions in split_ids.items():
        for role, expected_ids in partitions.items():
            actual = {r["id"] for r in manifest["patches"]
                      if r["region"] == region and r["split"] == role}
            if actual != set(expected_ids):
                raise ValueError("Input inventory membership differs from the canonical split IDs")
    current = digest(manifest)
    if bound_training and recorded_digest != current:
        raise ValueError("Evaluation input contents differ from the checkpoint training identity")
    return {
        "status": "verified_portable_input_contents",
        "manifest_sha256": sha256_file(manifest_path),
        "portable_split_digest": current,
        "training_content_binding": "verified" if bound_training else "not recorded by the supplied legacy script",
    }


@torch.no_grad()
def score_dataset(model, dataset, config, device, *, region, split,
                  batch_size, num_workers):
    """Return unrounded paper metrics, one row per target patch."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    rows = []
    seen = set()
    for batch in tqdm(loader, desc=f"U-Net {split}/{region}"):
        prediction = predict(model, batch["s2"].to(device),
                             batch["attrs"].to(device)).cpu()
        for i, pid in enumerate(batch["tile_id"]):
            if not isinstance(pid, str) or pid in seen:
                raise ValueError("Dataset returned duplicate or non-string patch IDs")
            seen.add(pid)
            values = patch_metrics(batch["lidar"][i], prediction[i],
                                   batch["mask"][i] > 0.5, config)
            if any(values[k] is None or not np.isfinite(values[k]) for k in METRICS):
                raise ValueError(f"Non-finite paper metric for {region}/{pid}")
            rows.append(dict(region=region, tile_id=pid, split=split, **values))
    if len(rows) != len(dataset):
        raise ValueError("Evaluation did not score the complete requested dataset")
    return rows


def summarize(rows):
    """Equal patch weighting, separately by region; population standard deviation."""
    result = []
    for region in sorted({r["region"] for r in rows}):
        part = [r for r in rows if r["region"] == region]
        record = dict(region=region, method="deterministic_unet", n=len(part))
        for metric in METRICS:
            values = np.asarray([r[metric] for r in part], dtype=float)
            record[metric + "_mean"] = float(values.mean())
            record[metric + "_std"] = float(values.std())
        result.append(record)
    return result


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_fingerprints():
    paths = [Path(__file__), ROOT / "scripts/evaluation.py",
             *ROOT.joinpath("src").rglob("*.py")]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(set(paths))}


def evaluate(args):
    checkpoint = resolve_path(args.checkpoint)
    split_path = resolve_path(args.split_ids)
    out = resolve_path(args.out or f"reconstructions/roughnet_unet_baseline/{args.split}")
    if out.exists():
        raise FileExistsError("Evaluation output already exists; choose a new --out")
    checkpoint_sha256 = sha256_file(checkpoint)
    sources = source_fingerprints()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, original_config, saved = load_checkpoint(checkpoint, device=device)
    ids = read_split_ids(split_path)
    training_identity = validate_training_identity(
        original_config, ids, split_path,
        resolve_path(args.legacy_splits) if args.legacy_splits else None,
    )
    config = prepare_config(copy.deepcopy(original_config), data_root=args.data_root,
                            split_ids=split_path, split_manifest=args.manifest)
    input_identity = verify_evaluation_inputs(config, ids, original_config)
    regions = ("cambridge",) if args.split == "test" else ("pondinlet", "tuk")
    rows = []
    batch_size = int(config["training"]["batch_size"]) if args.batch_size is None else args.batch_size
    workers = args.num_workers if args.num_workers is not None else int(config["training"]["num_workers"])
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    for region in regions:
        dataset = make_dataset(config, ids[region][args.split], args.split, region=region, augment=False,
                               randomize_context=bool(config["training"].get("randomize_context", False)))
        part = score_dataset(model, dataset, config, device, region=region, split=args.split,
                             batch_size=batch_size, num_workers=workers)
        if [r["tile_id"] for r in part] != ids[region][args.split]:
            raise ValueError("Evaluation target IDs differ from the canonical cohort")
        rows.extend(part)
    comparisons = summarize(rows)
    if sha256_file(checkpoint) != checkpoint_sha256:
        raise ValueError("Checkpoint changed during evaluation; choose a fixed checkpoint")
    if source_fingerprints() != sources:
        raise ValueError("Evaluation source changed during scoring; rerun with fixed source")
    manifest = dict(
        schema=1, model_type="deterministic_unet", split=args.split,
        checkpoint=dict(path=str(checkpoint), sha256=checkpoint_sha256, epoch=saved["epoch"]),
        training_identity=training_identity,
        evaluation_inputs=input_identity,
        split_ids_sha256=sha256_file(split_path),
        cohorts={region: ids[region][args.split] for region in regions},
        configuration=original_config,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                     cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), device=str(device),
                     gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                     cpu_threads=4, deterministic=True, tf32=False),
        sources=sources,
        protocol="Zero LiDAR input; timestep 0; single deterministic forward pass; shared paper metrics; equal patch weighting",
    )
    out.mkdir(parents=True)
    write_csv(out / "per_patch.csv", rows)
    write_csv(out / "comparison.csv", comparisons)
    write_json(out / "comparison.json", comparisons)
    write_json(out / "manifest.json", manifest)
    print(f"Scored {len(rows)} patches; table-compatible metrics: {out / 'per_patch.csv'}")
    return rows, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("models/roughnet_unet_baseline/roughnet_unet_baseline_best.pth"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--split-ids", type=Path, default=Path("splits/train_val_split_ids.json"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path, help="Portable input-content inventory override")
    parser.add_argument("--legacy-splits", type=Path,
                        help="Original CSV whose saved MD5 proves a legacy checkpoint's training membership")
    parser.add_argument("--out", type=Path, help="New output directory; default: reconstructions/roughnet_unet_baseline/<split>")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args(argv)
    try:
        evaluate(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
