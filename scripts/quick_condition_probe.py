"""Quick Sentinel-2 conditioning probe.

Runs a small deterministic sampling comparison for three conditioning modes:
normal, shuffled, and null. The goal is to quickly check whether the
Sentinel-2 branch materially changes the generated output.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.processing import per_patch_percentile_scale_bandwise_shared
from src.model.unet import ConditionalUNet
from src.diffusion.scheduler import LinearDiffusionScheduler, CosineDiffusionScheduler
from src.diffusion.sampling import p_sample_loop_ddim, p_sample_loop_plms
from src.utils.recon_metrics import rmse as recon_rmse, bias as recon_bias, zncc as recon_zncc


def get_region_preset(region_name: str, model_type: str = "cosine"):
    key = region_name.strip().lower()
    model_type = model_type.strip().lower()
    root = REPO_ROOT

    if key == "pondinlet":
        zone_ids = [4]
    elif key == "tuk":
        zone_ids = [13]
    elif key == "cambridge":
        zone_ids = None
    else:
        raise ValueError(f"Unknown region: {region_name!r}")

    return {
        "region_key": key,
        "zone_ids": zone_ids,
        "ckpt_path": os.path.join(root, "models", f"{model_type}_k6_att_best.pth"),
        "s2_dir": os.path.join(root, "input_data", f"s2_patches_{key}"),
        "lidar_dir": os.path.join(root, "input_data", f"lidar_patches_{key}"),
    }


def load_checkpoint(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt["model_state_dict"], ckpt["config"], ckpt


def get_patch_ids_subset(s2_dir, zone_ids=None, max_tiles=None, seed=42):
    pids = sorted(
        os.path.basename(p).split("_")[-1]
        for p in glob.glob(os.path.join(s2_dir, "s2_patch_*"))
        if os.path.isdir(p)
    )

    if zone_ids is not None:
        zone_ids = set(zone_ids)
        filtered = []
        for pid in pids:
            rj = os.path.join(s2_dir, f"s2_patch_{pid}", "region.json")
            try:
                with open(rj, "r", encoding="utf-8") as f:
                    rid = json.load(f).get("region_id", None)
                if rid in zone_ids:
                    filtered.append(pid)
            except Exception:
                pass
        pids = filtered

    if max_tiles is not None and len(pids) > max_tiles:
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(pids), generator=gen).tolist()
        pids = [pids[i] for i in perm[:max_tiles]]

    return pids


def encode_angles_deg(az_deg, ze_deg):
    rad = math.pi / 180.0
    az = float(az_deg)
    ze = float(ze_deg)
    return torch.tensor([math.sin(az * rad), math.cos(az * rad), ze / 90.0], dtype=torch.float32)


def days_from_ref(date_val, ref_date):
    if date_val is None:
        return 0.0
    try:
        from datetime import date

        d = date.fromisoformat(str(date_val)[:10])
        return float((d - ref_date).days)
    except Exception:
        return 0.0


def resolve_region_ref_date(lidar_root_path):
    from datetime import date

    name = os.path.basename(os.path.normpath(lidar_root_path)).lower()
    if "cambridge" in name:
        return date(2024, 4, 18)
    if "tuq" in name:
        return date(2024, 4, 16)
    if "pondinlet" in name:
        return date(2024, 4, 26)
    return date(2024, 4, 26)


def parse_attrs_json(json_path, ref_date, max_s2=6):
    if not os.path.exists(json_path):
        return [torch.zeros(8) for _ in range(max_s2)]
    with open(json_path, "r", encoding="utf-8") as f:
        recs = json.load(f)
    feats = []
    for r in recs[:max_s2]:
        if r is None:
            feats.append(torch.zeros(8))
            continue
        cloud = torch.tensor(float(r.get("cloud_cover", 0.0)) / 100.0).clamp(0, 1)
        saz = encode_angles_deg(r.get("sun_azimuth_mean", 0.0), r.get("sun_zenith_mean", 0.0))
        vaz = encode_angles_deg(r.get("view_azimuth_mean", 0.0), r.get("view_zenith_mean", 0.0))
        age = torch.tensor(days_from_ref(r.get("acquisition_date"), ref_date=ref_date) / 30.0, dtype=torch.float32).view(1)
        feats.append(torch.cat([cloud.view(1), saz, vaz, age], dim=0))
    while len(feats) < max_s2:
        feats.append(torch.zeros(8))
    return feats


def demean_lidar_patch(data_1hw, mask_hw, eps=1e-6):
    m = (mask_hw.float() > 0.5).float()
    denom = m.sum().clamp_min(eps)
    patch_mean = (data_1hw[0] * m).sum() / denom
    data_demeaned = data_1hw - patch_mean.view(1, 1, 1)
    data_demeaned = data_demeaned * m.view(1, *m.shape)
    return data_demeaned, patch_mean


def load_batch(pids, lidar_root, s2_root, context_k, seed=42, target_s2_hw=(256, 256)):
    ref_date = resolve_region_ref_date(lidar_root)
    samples = []

    for pid in pids:
        lidar_path = os.path.join(lidar_root, f"lidar_patch_{pid}.tif")
        s2_group_dir = os.path.join(s2_root, f"s2_patch_{pid}")

        available_ids = [i for i in range(6) if os.path.exists(os.path.join(s2_group_dir, f"t{i}.tif"))]
        if len(available_ids) < context_k:
            continue
        chosen_ids = available_ids[:context_k]

        with rasterio.open(lidar_path) as src:
            lidar_full = torch.tensor(src.read().tolist(), dtype=torch.float32)
        if lidar_full.shape[0] == 1:
            data = lidar_full[0:1]
            mask = torch.ones_like(data[0])
        else:
            data = lidar_full[0:1]
            mask = lidar_full[1]
        data, patch_mean = demean_lidar_patch(data, mask)

        s2_list = []
        for i in chosen_ids:
            s2_path = os.path.join(s2_group_dir, f"t{i}.tif")
            with rasterio.open(s2_path) as src:
                arr = torch.tensor(src.read()[:4].tolist(), dtype=torch.float32)
            if arr.shape[-2:] != target_s2_hw:
                arr = torch.nn.functional.interpolate(arr.unsqueeze(0), size=target_s2_hw, mode="bilinear", align_corners=False).squeeze(0)
            s2_list.append(arr)
        s2 = torch.cat(s2_list, dim=0)

        attrs = torch.cat([parse_attrs_json(os.path.join(s2_group_dir, "attrs.json"), ref_date=ref_date, max_s2=6)[i] for i in chosen_ids], dim=0)
        s2 = per_patch_percentile_scale_bandwise_shared(s2, p_low=2.0, p_high=98.0, min_range=1e-3, clamp01=True)

        samples.append({
            "s2": s2.float(),
            "lidar": data.float(),
            "mask": mask.float(),
            "attrs": attrs.float(),
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "tile_id": pid,
            "lidar_patch_mean": patch_mean.float(),
        })

    if not samples:
        raise RuntimeError("No valid samples loaded for the probe.")

    batch = {}
    for key in samples[0].keys():
        if torch.is_tensor(samples[0][key]):
            batch[key] = torch.stack([s[key] for s in samples], dim=0)
        else:
            batch[key] = [s[key] for s in samples]
    return batch


def build_scheduler(config, device):
    timesteps = int(config["training"]["timesteps"])
    noise_schedule = config["training"].get("noise_schedule", "linear")
    if noise_schedule == "linear":
        return LinearDiffusionScheduler(timesteps=timesteps, device=device)
    return CosineDiffusionScheduler(timesteps=timesteps, device=device)


def make_model(config, device):
    context_k = int(config["training"]["context_k"])
    model = ConditionalUNet(
        in_channels=1,
        cond_channels=4 * context_k,
        attr_dim=8 * context_k,
        base_channels=int(config["model"]["base_channels"]),
        embed_dim=int(config["model"]["embed_dim"]),
        unet_depth=int(config["model"]["unet_depth"]),
        attention_variant=config["model"]["attention_variant"],
        cond_k=context_k,
    ).to(device)
    return model


def corrupt_condition(batch, mode: str):
    s2 = batch["s2"].clone()
    attrs = batch["attrs"].clone()

    if mode == "normal":
        return s2, attrs
    if mode == "null":
        return torch.zeros_like(s2), torch.zeros_like(attrs)
    if mode == "shuffle":
        if s2.size(0) < 2:
            return s2, attrs
        perm = torch.randperm(s2.size(0), device=s2.device)
        return s2[perm], attrs[perm]
    raise ValueError(f"Unknown conditioning mode: {mode}")


def sample_batch(model, scheduler, batch, device, mode: str, seed: int, inference_steps: int, sampler_name: str):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    s2, attrs = corrupt_condition(batch, mode)
    lidar = batch["lidar"].to(device)
    cond = s2.to(device)
    attrs = attrs.to(device)

    steps = max(1, min(int(inference_steps), int(scheduler.timesteps)))
    step_ids = np.linspace(0, int(scheduler.timesteps) - 1, num=steps, dtype=int).tolist()[::-1]

    if sampler_name == "plms":
        x = torch.randn(lidar.shape, device=device)
        prev_eps = []
        for i, t in enumerate(step_ids):
            t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            alpha_cumprod_t = scheduler.alpha_cumprod[t]
            alpha_cumprod_prev_t = scheduler.alpha_cumprod[step_ids[i + 1]] if i + 1 < len(step_ids) else torch.tensor(1.0, device=device)

            pred_x0 = model(x, cond, attrs, t_batch)
            pred_epsilon = (x - torch.sqrt(alpha_cumprod_t) * pred_x0) / torch.sqrt(1 - alpha_cumprod_t)

            prev_eps.append(pred_epsilon)
            if len(prev_eps) > 4:
                prev_eps = prev_eps[-4:]

            if len(prev_eps) == 1:
                eps = prev_eps[-1]
            elif len(prev_eps) == 2:
                eps = (3 / 2) * prev_eps[-1] - (1 / 2) * prev_eps[-2]
            elif len(prev_eps) == 3:
                eps = (23 / 12) * prev_eps[-1] - (16 / 12) * prev_eps[-2] + (5 / 12) * prev_eps[-3]
            else:
                eps = (55 / 24) * prev_eps[-1] - (59 / 24) * prev_eps[-2] + (37 / 24) * prev_eps[-3] - (9 / 24) * prev_eps[-4]

            x0_part = torch.sqrt(alpha_cumprod_prev_t) * pred_x0
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev_t) * eps
            x = x0_part + dir_xt
    elif sampler_name == "ddim":
        x = torch.randn(lidar.shape, device=device)
        for i, t in enumerate(step_ids):
            t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            pred_x0 = model(x, cond, attrs, t_batch)
            alpha_cumprod_t = scheduler.alpha_cumprod[t]
            alpha_cumprod_prev_t = scheduler.alpha_cumprod[step_ids[i + 1]] if i + 1 < len(step_ids) else torch.tensor(1.0, device=device)
            x0_part = torch.sqrt(alpha_cumprod_prev_t) * pred_x0
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev_t) * (x - torch.sqrt(alpha_cumprod_t) * pred_x0) / torch.sqrt(1 - alpha_cumprod_t)
            x = x0_part + dir_xt
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    return x.detach().cpu()


def plot_comparison(batch, outputs, out_path):
    gt = batch["lidar"].cpu().numpy()
    tile_ids = batch["tile_id"]
    modes = ["normal", "shuffle", "null"]

    rows = len(tile_ids)
    cols = 1 + len(modes)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)

    vmax = 0.5
    vmin = -0.5
    cmap = "RdBu_r"

    for i, tile_id in enumerate(tile_ids):
        axes[i, 0].imshow(gt[i, 0], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"GT\n{tile_id}")
        axes[i, 0].axis("off")

        for j, mode in enumerate(modes, start=1):
            pred = outputs[mode][i, 0].numpy()
            axes[i, j].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[i, j].set_title(mode)
            axes[i, j].axis("off")

    fig.suptitle("Quick Sentinel-2 conditioning probe", fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_tile_metrics(gt, pred, mask):
    mask_bool = mask.bool() if mask is not None else None
    return {
        "rmse": float(recon_rmse(gt, pred, mask=mask_bool).item()),
        "bias": float(recon_bias(gt, pred, mask=mask_bool).item()),
        "zncc": float(recon_zncc(gt, pred, mask=mask_bool).item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Quick Sentinel-2 conditioning ablation probe.")
    parser.add_argument("--region", type=str, default="pondinlet", choices=["pondinlet", "tuk", "cambridge"])
    parser.add_argument("--model", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--sampler", type=str, default="plms", choices=["plms", "ddim"], help="Sampler to use in the probe.")
    parser.add_argument("--max-tiles", type=int, default=2)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    preset = get_region_preset(args.region, args.model)
    ckpt_path = preset["ckpt_path"]
    s2_dir = preset["s2_dir"]
    lidar_dir = preset["lidar_dir"]
    base_out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "..", "figures", "quick_condition_probe")
    out_dir = os.path.abspath(base_out_dir)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device)
    state, cfg, _ = load_checkpoint(ckpt_path, device)
    cfg["system"] = cfg.get("system", {})
    cfg["system"]["device"] = str(device)

    scheduler = build_scheduler(cfg, device)
    model = make_model(cfg, device)
    model.load_state_dict(state)
    model.eval()

    zone_ids = preset["zone_ids"]
    pids = get_patch_ids_subset(s2_dir, zone_ids=zone_ids, max_tiles=args.max_tiles, seed=args.seed)
    if len(pids) < 2:
        raise RuntimeError("Need at least two tiles to test shuffled conditioning.")

    batch = load_batch(
        pids=pids,
        lidar_root=lidar_dir,
        s2_root=s2_dir,
        context_k=int(cfg["training"]["context_k"]),
        seed=args.seed,
    )

    outputs = {}
    with torch.no_grad():
        for mode in ["normal", "shuffle", "null"]:
            outputs[mode] = sample_batch(model, scheduler, batch, device, mode=mode, seed=args.seed, inference_steps=args.inference_steps, sampler_name=args.sampler)

    normal = outputs["normal"]
    shuffle = outputs["shuffle"]
    null = outputs["null"]
    mad_shuffle = torch.mean(torch.abs(normal - shuffle)).item()
    mad_null = torch.mean(torch.abs(normal - null)).item()

    tile_metrics = []
    for i, tile_id in enumerate(batch["tile_id"]):
        tile_metrics.append({
            "tile_id": tile_id,
            "normal": compute_tile_metrics(batch["lidar"][i], normal[i], batch["mask"][i]),
            "shuffle": compute_tile_metrics(batch["lidar"][i], shuffle[i], batch["mask"][i]),
            "null": compute_tile_metrics(batch["lidar"][i], null[i], batch["mask"][i]),
        })

    fig_path = os.path.join(out_dir, f"{args.region}_{args.model}_conditioning_probe.png")
    plot_comparison(batch, outputs, fig_path)

    summary_path = os.path.join(out_dir, f"{args.region}_{args.model}_conditioning_probe.txt")
    summary_json_path = os.path.join(out_dir, f"{args.region}_{args.model}_conditioning_probe.json")
    summary = {
        "checkpoint": ckpt_path,
        "region": args.region,
        "model": args.model,
        "sampler": args.sampler,
        "tiles": batch["tile_id"],
        "inference_steps": args.inference_steps,
        "mean_abs_diff_normal_vs_shuffle": mad_shuffle,
        "mean_abs_diff_normal_vs_null": mad_null,
        "tile_metrics": tile_metrics,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2))
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved figure: {fig_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved JSON: {summary_json_path}")
    print(f"Mean abs diff normal vs shuffle: {mad_shuffle:.6f}")
    print(f"Mean abs diff normal vs null: {mad_null:.6f}")


if __name__ == "__main__":
    main()