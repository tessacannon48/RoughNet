# evaluation.py

import os, sys, json, glob, time, argparse, csv
from math import ceil
import numpy as np
import torch
import torch.nn.functional as Ft
from torch.utils.data import DataLoader
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.enums import Resampling
from rasterio.warp import reproject
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.colors as mcolors
import warnings
import matplotlib.ticker as ticker
import argparse

# Add parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Project imports
from src.data.dataset import LidarS2Dataset
from src.model.unet import ConditionalUNet
from src.diffusion.scheduler import LinearDiffusionScheduler, CosineDiffusionScheduler
from src.diffusion.sampling import p_sample_loop_ddpm, p_sample_loop_ddim, p_sample_loop_plms
from src.utils.metrics import normalize_batch
from src.utils.recon_metrics import (
    rmse as rmse_recon,
    bias as bias_recon,
    sigma_error,
    corr_length_error,
    normal_angle_error,
    average_jsd_multiscale,
    log_psd_rmse,
)

warnings.filterwarnings("ignore")

# === REGION PRESETS ====================================================

def get_region_preset(region_name: str):
    """
    region_name:
        expected strings (case-insensitive):
        - 'pondinlet'
        - 'tuk'
        - 'cambridge'

    Returns a small config bundle:
        - region_key: lowercase, no whitespace (for dirs, filenames)
        - pretty_name: for plot titles only
        - zone_ids: [4] for Pond Inlet, [13] for Tuk, None for Cambridge
        - ckpt_path, s2_dir, lidar_dir, out_dir
    """
    key = region_name.strip().lower()
    if key not in ("pondinlet", "tuk", "cambridge"):
        raise ValueError(
            f"Unknown region name: {region_name!r}. "
            "Expected one of: 'pondinlet', 'tuk', 'cambridge'."
        )

    root = "/cs/student/projects2/aisd/2024/tcannon/dissertation/Dissertation"

    # Single checkpoint for now – change here if you ever have region-specific models
    ckpt_path = f"{root}/models/pondinlet_tuq_modelD_k6_att_best.pth"

    # Pretty names only for plot titles
    if key == "pondinlet":
        pretty_name = "Pond Inlet"
        zone_ids = [4]
    elif key == "tuk":
        pretty_name = "Tuktokaktuk"
        zone_ids = [13]
    else:  # key == "cambridge"
        pretty_name = "Cambridge Bay"
        zone_ids = None  

    # Validation vs test output dirs 
    if key in ("pondinlet", "tuk"):
        out_prefix = "final_val"
    else:
        out_prefix = "final_test"

    return {
        "region_key": key,         
        "pretty_name": pretty_name,            # for titles only
        "zone_ids": zone_ids,
        "ckpt_path": ckpt_path,
        "s2_dir":    f"{root}/input_data/s2_patches_{key}",
        "lidar_dir": f"{root}/input_data/lidar_patches_{key}",
        "out_dir":   f"{root}/figures/{out_prefix}_{key}",
    }



# === UTILS ====================================================================
def load_checkpoint(ckpt_path, device):
    assert isinstance(ckpt_path, (str, bytes, os.PathLike)), "ckpt_path must be a filepath string."
    ckpt_path = str(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", None)
    state = ckpt["model_state_dict"]
    return state, cfg, ckpt

def list_all_patch_ids(s2_dir):
    return sorted([
        os.path.basename(p).split('_')[-1]
        for p in glob.glob(os.path.join(s2_dir, "s2_patch_*"))
        if os.path.isdir(p)
    ])

def get_patch_ids_subset(s2_dir, zone_ids=None, max_tiles=None, seed=42, deterministic_order=True):
    pids = list_all_patch_ids(s2_dir)
    if zone_ids is not None:
        zone_ids = set(zone_ids)
        filtered = []
        for pid in pids:
            rj = os.path.join(s2_dir, f"s2_patch_{pid}", "region.json")
            try:
                with open(rj, "r") as f:
                    rid = json.load(f).get("region_id", None)
                if rid in zone_ids:
                    filtered.append(pid)
            except Exception:
                pass
        pids = filtered
    if (max_tiles is not None) and (len(pids) > max_tiles):
        if deterministic_order:
            pids = pids[:max_tiles]
        else:
            rng = np.random.default_rng(seed)
            pids = list(rng.choice(pids, size=max_tiles, replace=False))
    return pids

def find_lidar_patch(lidar_dir, tile_id):
    cands = glob.glob(os.path.join(lidar_dir, f"*{tile_id}*.tif"))
    if not cands:
        raise FileNotFoundError(f"No LiDAR patch found for tile_id={tile_id} in {lidar_dir}")
    ones = [c for c in cands if "1m" in os.path.basename(c)]
    return (ones[0] if ones else cands[0])

def write_tif_like(ref_tif, out_path, array_2d_float32):
    with rasterio.open(ref_tif) as ref:
        prof = ref.profile.copy()
    prof.update(
        dtype="float32", count=1, compress="deflate", predictor=3, tiled=True,
        blockxsize=min(256, prof["width"]), blockysize=min(256, prof["height"]),
        BIGTIFF="IF_SAFER"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(array_2d_float32.astype(np.float32), 1)

def mosaic_average_safe(tif_list, out_path, compress=None):
    assert len(tif_list) > 0, "No input tiles to mosaic."
    srcs = [rasterio.open(fp) for fp in tif_list]
    try:
        nodatas = [s.nodata for s in srcs]
        merge_kwargs = {}
        if all(nd == nodatas[0] for nd in nodatas) and (nodatas[0] is not None):
            merge_kwargs["nodata"] = nodatas[0]
        sum_arr, transform = rio_merge(srcs, method="sum", **merge_kwargs)
        cnt_arr, _         = rio_merge(srcs, method="count", **merge_kwargs)
        denom = np.maximum(cnt_arr.astype(np.float32), 1.0)
        avg2d = (sum_arr.astype(np.float32) / denom)[0]
        ref = srcs[0]
        prof = {
            "driver": "GTiff",
            "height": int(avg2d.shape[0]),
            "width":  int(avg2d.shape[1]),
            "count":  1,
            "dtype":  "float32",
            "crs":    ref.crs,
            "transform": transform,
            "tiled": False,
        }
        if merge_kwargs.get("nodata", None) is not None:
            prof["nodata"] = merge_kwargs["nodata"]
        if compress:
            prof["compress"] = compress
        try:
            from rasterio.shutil import delete as rio_delete
            if os.path.exists(out_path):
                rio_delete(out_path)
        except Exception:
            if os.path.exists(out_path):
                os.remove(out_path)
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(avg2d, 1)
    finally:
        for s in srcs:
            s.close()
    return out_path

# === VISUALIZATION =============================================================
def plot_2d_maps(gt_array, pred_array, diff_array, out_path):
    # Mask zero => NaN (common for empty cells)
    gt_array = np.where(gt_array == 0, np.nan, gt_array)
    pred_array = np.where(pred_array == 0, np.nan, pred_array)
    diff_array = np.where(diff_array == 0, np.nan, diff_array)
    # Cut the array area in half for better visualization
    h, w = gt_array.shape
    h_25th_percentile = h // 4
    h_75th_percentile = (3 * h) // 4
    gt_array = gt_array[h_25th_percentile:h_75th_percentile, :]
    pred_array = pred_array[h_25th_percentile:h_75th_percentile, :]
    diff_array = diff_array[h_25th_percentile:h_75th_percentile, :]

    # Rotate the arrays
    gt_array = np.rot90(gt_array)
    pred_array = np.rot90(pred_array)
    diff_array = np.rot90(diff_array)

    stack = np.stack([gt_array, pred_array], axis=0)
    vmin = float(np.nanpercentile(stack, 0.02))
    vmax = float(np.nanpercentile(stack, 99.98))
    
    # SymLog for residuals
    max_abs_error = np.nanmax(np.abs(diff_array))
    linthresh_val = 0.5  # meters in the linear region
    norm_error = mcolors.SymLogNorm(
        linthresh=linthresh_val, linscale=1.0,
        vmin=-max_abs_error, vmax=max_abs_error
    )

    fig, axes = plt.subplots(3, 1, figsize=(7, 5))

    im0 = axes[0].imshow(gt_array, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth"); axes[0].axis('off')

    im1 = axes[1].imshow(pred_array, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction"); axes[1].axis('off')

    im2 = axes[2].imshow(diff_array, cmap='seismic', norm=norm_error)
    axes[2].set_title("Error (Pred - GT)"); axes[2].axis('off')

    cbar_ax = fig.add_axes([0.05, 0.7, 0.92, 0.02])
    cbar0 = fig.colorbar(im0, cax=cbar_ax, orientation="horizontal")
    cbar0.set_label("m")
    cbar0.ax.xaxis.set_ticks_position('bottom')
    cbar0.ax.xaxis.set_label_position('bottom')

    cbar_ax = fig.add_axes([0.05, 0.41, 0.92, 0.02])
    cbar1 = fig.colorbar(im1, cax=cbar_ax, orientation="horizontal", shrink=0.1)
    cbar1.set_label("m")
    cbar1.ax.xaxis.set_ticks_position('bottom')
    cbar1.ax.xaxis.set_label_position('bottom')

    # Force raw decimals on symlog colorbar
    formatter = ticker.ScalarFormatter(useOffset=False, useMathText=False)
    formatter.set_scientific(False)
    formatter.set_powerlimits((0, 0))
    cbar_ax = fig.add_axes([0.05, 0.12, 0.92, 0.02])
    cbar2 = fig.colorbar(im2, cax=cbar_ax, orientation="horizontal", format=formatter)
    cbar2.set_label("m")
    cbar2.ax.xaxis.set_ticks_position('bottom')
    cbar2.ax.xaxis.set_label_position('bottom')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 2D composite to {out_path}")

# === METRIC HELPERS ===========================================================
def _metric_params_from_cfg(cfg):
    px = float(cfg.get("data", {}).get("pixel_size_m", 1.0))
    jsd_scales = tuple(cfg.get("evaluation", {}).get("jsd_scales_m", [1.0, 2.0, 5.0, 10.0]))
    jsd_bins   = int(cfg.get("evaluation", {}).get("jsd_bins", 256))
    use_sobel  = bool(cfg.get("evaluation", {}).get("nae_use_sobel", True))
    deg        = bool(cfg.get("evaluation", {}).get("nae_degrees", True))
    use_window = bool(cfg.get("evaluation", {}).get("psd_window", True))
    return px, jsd_scales, jsd_bins, use_sobel, deg, use_window

def _compute_metrics_tensor(gt_t, pr_t, mask_t, px, jsd_scales, jsd_bins, use_sobel, deg, use_window):
    # Print components of corr_length_error for debugging
    # Apply full mask
    full_mask = mask_t

    # Add threshold mask to avoid very small ground truth values
    min_valid_depth = 0.01  # meters
    gt_above_thresh = gt_t > min_valid_depth

    # Combine both masks
    valid_mask = full_mask & gt_above_thresh

    # Compute abs rel only on valid pixels
    abs_rel = torch.abs(gt_t - pr_t) / (gt_t + 1e-6)
    abs_rel = abs_rel[valid_mask]
    abs_rel_error = float(abs_rel.mean().item()) if abs_rel.numel() > 0 else float("nan")

    return {
        "rmse_phys_m": float(rmse_recon(gt_t, pr_t, mask=mask_t).item()),
        "bias_phys_m": float(bias_recon(gt_t, pr_t, mask=mask_t).item()),
        "sigma_error_pct": float(sigma_error(gt_t, pr_t, mask=mask_t).item()),
        "corr_length_error_pct": float(corr_length_error(gt_t, pr_t, mask=mask_t, pixel_size=px).item()),
        "normal_angle_error_deg": float(normal_angle_error(gt_t, pr_t, mask=mask_t, pixel_size=px, use_sobel=use_sobel, degrees=deg).item()),
        "jsd": float(average_jsd_multiscale(gt_t, pr_t, scales_m=jsd_scales, pixel_size=px, bins=jsd_bins, mask=mask_t).item()),
        "psd_rmse": float(log_psd_rmse(gt_t, pr_t, pixel_size=px, mask=mask_t, window=use_window).item()),
	    "abs_rel_error": abs_rel_error,
    }

def _valid_mask_from_arrays(gt_array, pred_array):
    return (~np.isnan(gt_array)) & (~np.isnan(pred_array)) & (gt_array != 0) & (pred_array != 0)

# === REGION-WIDE METRICS ======================================================
def compute_and_save_region_metrics(gt_array, pred_array, out_dir, cfg):
    stats_dir = os.path.join(out_dir, "reconstruction_statistics")
    os.makedirs(stats_dir, exist_ok=True)

    mask_np = _valid_mask_from_arrays(gt_array, pred_array)
    if not np.any(mask_np):
        raise ValueError("No valid pixels found for region metrics.")
    gt_t = torch.from_numpy(gt_array).float()
    pr_t = torch.from_numpy(pred_array).float()
    m_t  = torch.from_numpy(mask_np).bool()

    px, jsd_scales, jsd_bins, use_sobel, deg, use_window = _metric_params_from_cfg(cfg)
    metrics = _compute_metrics_tensor(gt_t, pr_t, m_t, px, jsd_scales, jsd_bins, use_sobel, deg, use_window)

    # masked stats
    gt_masked = torch.masked_select(gt_t, m_t)
    pr_masked = torch.masked_select(pr_t, m_t)

    region_stats = {
        "scope": "region_mosaic",
        "valid_pixel_count": int(m_t.sum().item()),
        **metrics,
        "gt_min_val": float(gt_masked.min().item()),
        "gt_max_val": float(gt_masked.max().item()),
        "gt_mean_val": float(gt_masked.mean().item()),
        "gt_std_val": float(gt_masked.std(unbiased=False).item()),
        "pred_min_val": float(pr_masked.min().item()),
        "pred_max_val": float(pr_masked.max().item()),
        "pred_mean_val": float(pr_masked.mean().item()),
        "pred_std_val": float(pr_masked.std(unbiased=False).item()),
        "pixel_size_m": px,
        "jsd_scales_m": list(jsd_scales),
        "jsd_bins": jsd_bins,
        "nae_use_sobel": use_sobel,
        "nae_degrees": deg,
        "psd_window": use_window,
    }

    out_json = os.path.join(stats_dir, "region_reconstruction_stats.json")
    with open(out_json, "w") as f:
        json.dump(region_stats, f, indent=4)
    print(f"Saved region reconstruction stats → {out_json}")

# === PER-PATCH METRICS ========================================================
def compute_and_save_patch_metrics(out_dir, cfg):
    """
    Reads per-patch predicted tiles in OUT_DIR/pred_tiles, matches each to its GT tile,
    computes reconstruction metrics per patch, and saves a CSV with one row per patch.
    Also prints macro- and weighted-averages.
    """
    stats_dir = os.path.join(out_dir, "reconstruction_statistics")
    os.makedirs(stats_dir, exist_ok=True)

    pred_tiles_dir = os.path.join(out_dir, "pred_tiles")
    if not os.path.isdir(pred_tiles_dir):
        raise FileNotFoundError(f"No pred_tiles directory found at {pred_tiles_dir}")

    pred_tifs = sorted(glob.glob(os.path.join(pred_tiles_dir, "pred_*.tif")))
    if len(pred_tifs) == 0:
        raise FileNotFoundError(f"No predicted tiles found in {pred_tiles_dir}")

    lidar_dir = cfg["data"].get("lidar_dir") or cfg["data"].get("lidar_dirs")
    if isinstance(lidar_dir, list):
        # choose first if multiple provided
        lidar_dir = lidar_dir[0]

    # metric parameters
    px, jsd_scales, jsd_bins, use_sobel, deg, use_window = _metric_params_from_cfg(cfg)

    rows = []
    for pred_fp in tqdm(pred_tifs, desc="Per-patch metrics"):
        # tile_id inferred from filename pred_{tileid}.tif
        basename = os.path.basename(pred_fp)
        tile_id = basename[len("pred_"):-4]

        gt_fp = find_lidar_patch(lidar_dir, tile_id)

        with rasterio.open(gt_fp) as g, rasterio.open(pred_fp) as p:
            gt = g.read(1).astype(np.float32)
            pr = p.read(1).astype(np.float32)

            # Align if needed (shouldn't be necessary, but safe)
            if (gt.shape != pr.shape) or (g.transform != p.transform):
                pr_aligned = np.zeros_like(gt, dtype=np.float32)
                reproject(
                    source=pr, destination=pr_aligned,
                    src_transform=p.transform, src_crs=p.crs,
                    dst_transform=g.transform, dst_crs=g.crs,
                    resampling=Resampling.bilinear
                )
                pr = pr_aligned

        mask_np = _valid_mask_from_arrays(gt, pr)
        if not np.any(mask_np):
            # still record the tile with NaNs so you can inspect failures
            row = {"tile_id": tile_id, "valid_pixel_count": 0}
            for k in ["rmse_phys_m","bias_phys_m","sigma_error_pct","corr_length_error_pct",
                      "normal_angle_error_deg","jsd","psd_rmse",
                      "gt_mean_val","gt_std_val","pred_mean_val","pred_std_val", "abs_rel_error",
                      "gt_min_val", "gt_max_val", "pred_min_val", "pred_max_val"]:
                row[k] = float("nan")
            rows.append(row)
            continue

        gt_t = torch.from_numpy(gt).float()
        pr_t = torch.from_numpy(pr).float()
        m_t  = torch.from_numpy(mask_np).bool()

        metrics = _compute_metrics_tensor(gt_t, pr_t, m_t, px, jsd_scales, jsd_bins, use_sobel, deg, use_window)

        gt_masked = torch.masked_select(gt_t, m_t)
        pr_masked = torch.masked_select(pr_t, m_t)

        row = {
            "tile_id": tile_id,
            "valid_pixel_count": int(m_t.sum().item()),
            **metrics,
            "gt_mean_val": float(gt_masked.mean().item()),
            "gt_std_val": float(gt_masked.std(unbiased=False).item()),
            "pred_mean_val": float(pr_masked.mean().item()),
            "pred_std_val": float(pr_masked.std(unbiased=False).item()),
            "gt_min_val": float(gt_masked.min().item()),
            "gt_max_val": float(gt_masked.max().item()),
            "pred_min_val": float(pr_masked.min().item()),
            "pred_max_val": float(pr_masked.max().item()),
        }
        rows.append(row)

    # Save CSV
    csv_path = os.path.join(stats_dir, "patch_reconstruction_stats.csv")
    fieldnames = [
        "tile_id", "valid_pixel_count",
        "rmse_phys_m", "bias_phys_m",
        "sigma_error_pct", "corr_length_error_pct",
        "normal_angle_error_deg", "jsd", "psd_rmse","abs_rel_error",
        "gt_mean_val", "gt_std_val", "pred_mean_val", "pred_std_val",
        "gt_min_val", "gt_max_val", "pred_min_val", "pred_max_val",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved per-patch reconstruction CSV → {csv_path}")

    # Also compute and print averages across patches
    def _nanmean(vals):
        arr = np.array(vals, dtype=np.float64)
        return float(np.nanmean(arr)) if arr.size else float("nan")
    def _weighted_mean(vals, weights):
        v = np.array(vals, dtype=np.float64)
        w = np.array(weights, dtype=np.float64)
        m = np.isfinite(v) & (w > 0)
        if not np.any(m):
            return float("nan")
        return float(np.average(v[m], weights=w[m]))

    weights = [r["valid_pixel_count"] for r in rows]
    metrics_keys = ["rmse_phys_m","bias_phys_m","sigma_error_pct","corr_length_error_pct",
                    "normal_angle_error_deg","jsd","psd_rmse","abs_rel_error"]

    macro_avgs = {k: _nanmean([r[k] for r in rows]) for k in metrics_keys}
    weighted_avgs = {k: _weighted_mean([r[k] for r in rows], weights) for k in metrics_keys}

    summary_json = {
        "scope": "per_patch_summary",
        "num_patches": len(rows),
        "macro_average": macro_avgs,
        "weighted_by_valid_pixel_count": weighted_avgs,
    }
    with open(os.path.join(stats_dir, "patch_reconstruction_summary.json"), "w") as f:
        json.dump(summary_json, f, indent=4)
    print("Saved per-patch summary JSON →", os.path.join(stats_dir, "patch_reconstruction_summary.json"))

def plot_region_pdfs(gt_array, pred_array, out_path,
                     bins=256, low_pct=0.5, high_pct=99.5,
                     smooth_sigma=None, logy=False, title=None):
    """
    Plot PDFs (density curves) of GT vs Prediction for the whole region.
    - x-axis is limited using percentiles of the *masked* combined data.
    - JSD between the two (discrete) distributions is computed and annotated.
    - Optionally smooth with a Gaussian kernel in 1D (if scipy is available).

    Args:
      gt_array, pred_array: 2D ndarrays (aligned mosaics, float)
      out_path: path to save the figure (PNG)
      bins: number of histogram bins
      low_pct, high_pct: percentiles for x-limits (e.g., 0.5/99.5)
      smooth_sigma: None or float (std dev in bins) for Gaussian smoothing
      logy: if True, use log-scale on Y axis
      title: optional figure title
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Valid mask: drop NaNs and zeros (your convention for empties)
    mask = (~np.isnan(gt_array)) & (~np.isnan(pred_array)) & (gt_array != 0) & (pred_array != 0)
    gt = gt_array[mask].astype(np.float64)
    pr = pred_array[mask].astype(np.float64)
    if gt.size == 0 or pr.size == 0:
        raise ValueError("No valid pixels to build PDFs.")

    # Robust x-limits from combined data
    combined = np.concatenate([gt, pr], axis=0)
    xmin = float(np.percentile(combined, low_pct))
    xmax = float(np.percentile(combined, high_pct))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
        # Fallback to overall min/max if percentiles are degenerate
        xmin = float(np.min(combined))
        xmax = float(np.max(combined))
    # Build common bins
    bin_edges = np.linspace(xmin, xmax, bins + 1, dtype=np.float64)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # Histograms → densities (integrate to 1 over [xmin, xmax])
    gt_hist, _ = np.histogram(gt, bins=bin_edges, density=True)
    pr_hist, _ = np.histogram(pr, bins=bin_edges, density=True)

    # Optional smoothing (if scipy present)
    if smooth_sigma is not None and smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter1d
            gt_hist = gaussian_filter1d(gt_hist, sigma=smooth_sigma, mode="nearest")
            pr_hist = gaussian_filter1d(pr_hist, sigma=smooth_sigma, mode="nearest")
        except Exception:
            # If SciPy isn't available, just continue without smoothing
            pass

    # --- JSD between discrete distributions on these bins ---
    # Convert densities to probability masses on each bin (~density * bin_width), then normalize
    eps = 1e-12
    p = gt_hist * bin_width
    q = pr_hist * bin_width
    p_sum = p.sum(); q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        # Degenerate case (all-zero densities inside range)
        jsd = float("nan")
    else:
        p = p / (p_sum + eps)
        q = q / (q_sum + eps)
        m = 0.5 * (p + q)
        # KL with safeguards
        def _kl(a, b):
            a_safe = np.clip(a, eps, 1.0)
            b_safe = np.clip(b, eps, 1.0)
            return float(np.sum(a_safe * np.log(a_safe / b_safe)))
        jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bin_centers, gt_hist, label=f"Ground truth (n={gt.size:,})", linewidth=2)
    ax.plot(bin_centers, pr_hist, label=f"Prediction (n={pr.size:,})", linewidth=2, linestyle="--")
    ax.set_xlim([xmin, xmax])
    ax.set_xlabel("LiDAR Residual (m)")
    ax.set_ylabel("Probability density")
    if title:
        ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.2)
    ax.legend()

    # Annotate JSD
    if np.isfinite(jsd):
        ax.text(0.02, 0.98, f"JSD = {jsd:.4f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=11,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved region PDF plot → {out_path}")

    # Also drop a tiny sidecar JSON with the settings + JSD
    meta = {
        "bins": bins,
        "low_pct": low_pct,
        "high_pct": high_pct,
        "smooth_sigma": smooth_sigma,
        "logy": logy,
        "xmin": xmin,
        "xmax": xmax,
        "valid_n_gt": int(gt.size),
        "valid_n_pred": int(pr.size),
        "jsd": jsd,
    }
    with open(out_path.replace(".png", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# 3D modeling

def subsample(arr, step):
    return arr[::step, ::step]

def plot_single_3d_surface(ax, lidar_array, title="3D LiDAR Surface Plot",
                           cmap='terrain', z_label='Elevation Deviations (m)',
                           vmin=None, vmax=None):
    # treat zeros as NaN for visualization
    lidar_array = np.where(lidar_array == 0, np.nan, lidar_array)
    h, w = lidar_array.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    ax.set_zlim(-1.5, 1.5)
    surf = ax.plot_surface(
        X, Y, lidar_array,
        cmap=cmap, alpha=0.9, edgecolor='none',
        rstride=1, cstride=1, vmin=vmin, vmax=vmax
    )
    # set x and y lims
    ax.set_xlim(0, 550)
    ax.set_ylim(0, 250)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.view_init(elev=15, azim=70)
    return surf

def plot_all_three_3d_surfaces(gt_array, pred_array, diff_array,
                               step=4, out_path=None, plot_title="Combined 3D Plots"):
    gt_s   = subsample(gt_array, step)
    pr_s   = subsample(pred_array, step)
    diff_s = subsample(diff_array, step)

    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(plot_title, fontsize=16)

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    s1 = plot_single_3d_surface(ax1, gt_s,  "Ground Truth",
                                cmap='terrain', z_label='LiDAR Residual (m)',
                                vmin=-0.1, vmax=1.1)
    fig.colorbar(s1, ax=ax1, shrink=0.5, aspect=10, label='m')
    ax1.set_ylabel('Pixel Y (1m)')
    ax1.set_xlabel('Pixel X (1m)')

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    s2 = plot_single_3d_surface(ax2, pr_s,  "Predicted",
                                cmap='terrain', z_label='LiDAR Residual (m)',
                                vmin=-0.1, vmax=1.1)
    fig.colorbar(s2, ax=ax2, shrink=0.5, aspect=10, label='m')
    ax2.set_ylabel('Pixel Y (1m)')
    ax2.set_xlabel('Pixel X (1m)')

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    s3 = plot_single_3d_surface(ax3, diff_s, "Error (Pred - GT)",
                                cmap='RdBu', z_label='Difference (m)',
                                vmin=-0.15, vmax=0.15)
    fig.colorbar(s3, ax=ax3, shrink=0.5, aspect=10, label='m')
    ax3.set_ylabel('Pixel Y (1m)')
    ax3.set_xlabel('Pixel X (1m)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f"Saved 3D plots to {out_path}")
    plt.close(fig)

# === PIPELINE =================================================================
@torch.no_grad()
def run_predictions_and_mosaics(ckpt_path, config_yaml, out_dir,
                                sampler_name="ddpm", batch_size=8, num_workers=4, device="cuda",
                                zone_ids=None, max_tiles=None, seed=42, deterministic_order=True):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load checkpoint and config
    state, cfg_from_ckpt, _ = load_checkpoint(ckpt_path, device)

    # Accept dict overrides, YAML path, or fall back to ckpt config
    if isinstance(config_yaml, dict):
        cfg = config_yaml
        print("Loaded config from provided dict overrides.")
    elif isinstance(config_yaml, str) and len(config_yaml) > 0:
        with open(config_yaml, "r") as f:
            cfg = yaml.safe_load(f)
        print("Loaded config from", config_yaml)
    else:
        cfg = cfg_from_ckpt
        print("Loaded config from checkpoint.")

    s2_dir        = cfg["data"]["s2_dir"]
    lidar_dir     = cfg["data"]["lidar_dir"]
    context_k     = cfg["training"]["context_k"]
    noise_sched   = cfg["training"]["noise_schedule"]
    timesteps     = cfg["training"]["timesteps"]
    base_channels = cfg["model"]["base_channels"]
    embed_dim     = cfg["model"]["embed_dim"]
    unet_depth    = cfg["model"]["unet_depth"]
    attention_variant = cfg["model"]["attention_variant"]

    print("\n=== Inference Config ===")
    print(f"S2 dir:    {s2_dir}")
    print(f"LiDAR dir: {lidar_dir}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Sampler: {sampler_name} | Timesteps: {timesteps} | Schedule: {noise_sched}")
    print(f"Model: UNet depth={unet_depth}, base={base_channels}, attn={attention_variant}, embed_dim={embed_dim}")
    print(f"Context k: {context_k}\n")

    scheduler = (
        LinearDiffusionScheduler(timesteps=timesteps, device=device)
        if noise_sched == "linear"
        else CosineDiffusionScheduler(timesteps=timesteps, device=device)
    )

    model = ConditionalUNet(
        in_channels=1,
        cond_channels=4 * context_k,
        attr_dim=8 * context_k,
        base_channels=base_channels,
        embed_dim=embed_dim,
        unet_depth=unet_depth,
        attention_variant=attention_variant,
        cond_k=context_k,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    subset_pids = get_patch_ids_subset(
        s2_dir=s2_dir, zone_ids=zone_ids, max_tiles=max_tiles,
        seed=seed, deterministic_order=deterministic_order
    )
    print(f"Using {len(subset_pids)} patch(es).")

    # Build dataset/loader
    dataset = LidarS2Dataset(
        lidar_dirs=lidar_dir if "lidar_dir" in cfg["data"] else cfg["data"].get("lidar_dirs", lidar_dir),
        s2_dirs=s2_dir if "s2_dir" in cfg["data"] else cfg["data"].get("s2_dirs", s2_dir),
        context_k=context_k,
        randomize_context=True,
        augment=False,
        debug=False,
        split_pids=subset_pids,
        split="val",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Choose sampler
    samplers = {
        "ddpm": lambda m, s, c, a, d: p_sample_loop_ddpm(m, scheduler, s, c, a, d),
        "ddim": lambda m, s, c, a, d: p_sample_loop_ddim(m, scheduler, s, c, a, d),
        "plms": lambda m, s, c, a, d: p_sample_loop_plms(m, scheduler, s, c, a, d),
    }
    if sampler_name not in samplers:
        raise ValueError(f"Unknown sampler: {sampler_name}")
    sampler = samplers[sampler_name]

    pred_tiles_dir = os.path.join(out_dir, "pred_tiles")
    os.makedirs(pred_tiles_dir, exist_ok=True)

    pred_tifs, gt_tifs = [], []
    start = time.perf_counter()

    processed = 0
    for batch in tqdm(loader, desc="Predicting tiles"):
        if (max_tiles is not None) and (processed >= max_tiles):
            break

        s2     = batch["s2"].to(device)
        attrs  = batch["attrs"].to(device)
        lidar  = batch["lidar"].to(device)
        tile_ids_batch = batch["tile_id"]

        pred = sampler(model, lidar.shape, s2, attrs, device).float().cpu().numpy()

        B = pred.shape[0]
        for i in range(B):
            if (max_tiles is not None) and (processed >= max_tiles):
                break
            tile_id = tile_ids_batch[i]
            gt_lidar_tif = find_lidar_patch(lidar_dir, tile_id)
            out_tif = os.path.join(pred_tiles_dir, f"pred_{tile_id}.tif")
            write_tif_like(gt_lidar_tif, out_tif, pred[i, 0])
            pred_tifs.append(out_tif)
            gt_tifs.append(gt_lidar_tif)
            processed += 1

    elapsed = time.perf_counter() - start
    print(f"\nFinished per-tile predictions for {len(pred_tifs)} tiles in {elapsed/60:.1f} min.")

    pred_mosaic_path = os.path.join(out_dir, "pred_mosaic.tif")
    print("Mosaicking predictions →", pred_mosaic_path)
    mosaic_average_safe(pred_tifs, pred_mosaic_path, compress="deflate")

    gt_mosaic_path = os.path.join(out_dir, "gt_mosaic.tif")
    print("Mosaicking ground truth →", gt_mosaic_path)
    mosaic_average_safe(sorted(list(set(gt_tifs))), gt_mosaic_path, compress="deflate")

    return pred_mosaic_path, gt_mosaic_path, cfg

def align_and_save_diff(pred_mosaic_path, gt_mosaic_path, out_dir):
    with rasterio.open(gt_mosaic_path) as g, rasterio.open(pred_mosaic_path) as p:
        gt_array = g.read(1).astype(np.float32)
        pred_array = p.read(1).astype(np.float32)
        if (gt_array.shape != pred_array.shape) or (g.transform != p.transform):
            print("Aligning prediction mosaic to ground truth grid...")
            pred_aligned = np.zeros_like(gt_array, dtype=np.float32)
            reproject(
                source=pred_array, destination=pred_aligned,
                src_transform=p.transform, src_crs=p.crs,
                dst_transform=g.transform, dst_crs=g.crs,
                resampling=Resampling.bilinear
            )
            pred_array = pred_aligned
        diff_array = pred_array - gt_array
        diff_path = os.path.join(out_dir, "diff_pred_minus_gt.tif")
        prof = g.profile.copy()
        prof.update(dtype="float32", count=1, compress="deflate")
        with rasterio.open(diff_path, "w", **prof) as dst:
            dst.write(diff_array.astype(np.float32), 1)
        print("Wrote diff raster →", diff_path)
    return gt_array, pred_array, diff_array, diff_path

# === MAIN =====================================================================
def main():

    parser = argparse.ArgumentParser(
        description="Evaluate RoughNet on a given region (val or test)."
    )
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        help="Region key (lowercase, no spaces): 'pondinlet', 'tuk', or 'cambridge'.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="plms",
        choices=["ddpm", "ddim", "plms"],
        help="Sampling method for diffusion.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for tile-wise prediction.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Optionally limit number of tiles for quick runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset selection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cuda' or 'cpu'. Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help=(
            "Skip prediction if pred_mosaic.tif and gt_mosaic.tif already exist "
            "in the region's out_dir. Reuses saved mosaics."
        ),
    )
    parser.add_argument(
        "--deterministic-order",
        action="store_true",
        help="Use deterministic tile ordering when downsampling with max-tiles.",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Resolve region preset (paths + zone_ids + ckpt_path)
    # -------------------------------------------------------------------------
    preset = get_region_preset(args.region)
    region_key      = preset["region_key"]      
    region_name     = preset["pretty_name"]     
    zone_ids        = preset["zone_ids"]
    CKPT_PATH       = preset["ckpt_path"]
    TEST_S2_DIR     = preset["s2_dir"]
    TEST_LIDAR_DIR  = preset["lidar_dir"]
    out_dir         = preset["out_dir"]

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Device + load config from checkpoint once
    # -------------------------------------------------------------------------
    device = args.device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    ckpt_tmp = torch.load(CKPT_PATH, map_location=device)
    config = ckpt_tmp["config"]

    # Override paths for eval data (all lowercase/no spaces)
    config["data"]["s2_dir"]    = TEST_S2_DIR
    config["data"]["lidar_dir"] = TEST_LIDAR_DIR
    if "logging" in config:
        config["logging"]["output_dir"] = out_dir
    if "system" in config:
        config["system"]["device"] = device

    # -------------------------------------------------------------------------
    # Prediction (or reuse existing mosaics)
    # -------------------------------------------------------------------------
    pred_mosaic_path = os.path.join(out_dir, "pred_mosaic.tif")
    gt_mosaic_path   = os.path.join(out_dir, "gt_mosaic.tif")

    reuse_ok = (
        args.skip_predict
        and os.path.exists(pred_mosaic_path)
        and os.path.exists(gt_mosaic_path)
    )

    if reuse_ok:
        print("Skipping prediction (reuse mode). Using existing mosaics:")
        print(f"  Pred mosaic: {pred_mosaic_path}")
        print(f"  GT mosaic:   {gt_mosaic_path}")
        cfg_used = config
    else:
        print("Running prediction + mosaicking...")
        pred_mosaic_path, gt_mosaic_path, cfg_used = run_predictions_and_mosaics(
            ckpt_path=CKPT_PATH,
            config_yaml=config,
            out_dir=out_dir,
            sampler_name=args.sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            zone_ids=zone_ids,
            max_tiles=args.max_tiles,
            seed=args.seed,
            deterministic_order=args.deterministic_order,
        )

    # -------------------------------------------------------------------------
    # Alignment + plots + metrics
    # -------------------------------------------------------------------------
    gt_array, pred_array, diff_array, diff_path = align_and_save_diff(
        pred_mosaic_path=pred_mosaic_path,
        gt_mosaic_path=gt_mosaic_path,
        out_dir=out_dir,
    )

    # Region PDF figure – filename uses lowercase key, title uses pretty name
    pdf_path = os.path.join(out_dir, f"{region_key}_pdfs.png")
    plot_region_pdfs(
        gt_array,
        pred_array,
        pdf_path,
        bins=256,
        low_pct=0.01,
        high_pct=99.99,
        smooth_sigma=1.5,
        logy=False,
        title=f"Region PDFs: GT vs Prediction ({region_name})",
    )

    # 2D composite
    two_d_path = os.path.join(out_dir, f"{region_key}_mosaic_2d.png")
    plot_2d_maps(gt_array, pred_array, diff_array, two_d_path)

    # 3D composite
    three_d_path = os.path.join(out_dir, f"{region_key}_mosaic_3d.png")
    plot_all_three_3d_surfaces(
        gt_array=gt_array,
        pred_array=pred_array,
        diff_array=diff_array,
        step=20,
        out_path=three_d_path,
        plot_title=f"Predicted 3D Surface for {region_name}",
    )

    # Region-wide metrics
    compute_and_save_region_metrics(gt_array, pred_array, out_dir, cfg_used)

    # Per-patch metrics
    compute_and_save_patch_metrics(out_dir, cfg_used)

    print("\nDone.")
    print("Outputs:")
    print(f"  Region:       {region_name} ({region_key})")
    print(f"  GT mosaic:    {gt_mosaic_path}")
    print(f"  Pred mosaic:  {pred_mosaic_path}")
    print(f"  Diff raster:  {diff_path}")
    print(f"  Region stats: {os.path.join(out_dir, 'reconstruction_statistics', 'region_reconstruction_stats.json')}")
    print(f"  Patch CSV:    {os.path.join(out_dir, 'reconstruction_statistics', 'patch_reconstruction_stats.csv')}")
    print(f"  Patch summary:{os.path.join(out_dir, 'reconstruction_statistics', 'patch_reconstruction_summary.json')}")


if __name__ == "__main__":
    main()
