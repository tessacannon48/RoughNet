#evaluation.py 

import os, sys, json, glob, time, argparse, csv
import numpy as np
import torch
from torch.utils.data import DataLoader
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import warnings
from rasterio.shutil import delete as rio_delete
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import LidarS2Dataset
from src.model.unet import ConditionalUNet
from src.diffusion.scheduler import LinearDiffusionScheduler, CosineDiffusionScheduler
from src.diffusion.sampling import p_sample_loop_ddpm, p_sample_loop_ddim, p_sample_loop_plms
from src.utils.recon_metrics import (
    rmse as rmse_recon,
    bias as bias_recon,
    sigma_error,
    normal_angle_error,
    average_jsd_multiscale,
    log_psd_rmse,
    zncc,
)

warnings.filterwarnings("ignore")

NODATA = -9999.0


def get_region_preset(region_name: str, model_type: str = "cosine"):
    key = region_name.strip().lower()
    if key not in ("pondinlet", "tuk", "cambridge"):
        raise ValueError(
            f"Unknown region name: {region_name!r}. "
            "Expected one of: 'pondinlet', 'tuk', 'cambridge'."
        )

    model_type = model_type.strip().lower()
    if model_type not in ("cosine", "linear"):
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            "Expected one of: 'cosine', 'linear'."
        )

    root = "/cs/student/projects2/aisd/2024/tcannon/dissertation/Dissertation"
    ckpt_path = f"{root}/models/{model_type}_k6_att_best.pth"

    if key == "pondinlet":
        pretty_name = "Pond Inlet"
        zone_ids = [4]
    elif key == "tuk":
        pretty_name = "Tuktokaktuk"
        zone_ids = [13]
    else:
        pretty_name = "Cambridge Bay"
        zone_ids = None

    if key in ("pondinlet", "tuk"):
        out_prefix = "final_val"
    else:
        out_prefix = "final_test"

    return {
        "region_key": key,
        "pretty_name": pretty_name,
        "zone_ids": zone_ids,
        "ckpt_path": ckpt_path,
        "s2_dir": f"{root}/input_data/s2_patches_{key}",
        "lidar_dir": f"{root}/input_data/lidar_patches_{key}",
        "out_dir": f"{root}/figures/{model_type}_model/{out_prefix}_{key}",
    }


def load_checkpoint(ckpt_path, device):
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
        dtype="float32",
        count=1,
        nodata=float(NODATA),
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=min(256, prof["width"]),
        blockysize=min(256, prof["height"]),
        BIGTIFF="IF_SAFER",
    )

    arr = array_2d_float32.astype(np.float32)
    arr = np.where(np.isfinite(arr), arr, NODATA).astype(np.float32)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr, 1)


def write_gt_singleband_from_patch(gt_patch_path, out_path):
    with rasterio.open(gt_patch_path) as src:
        arr = src.read().astype(np.float32)
        data = arr[0]
        if src.count >= 2:
            mask = arr[1] > 0.5
        else:
            mask = np.isfinite(data)

        data = np.where(mask & np.isfinite(data), data, NODATA).astype(np.float32)
        prof = src.profile.copy()
        prof.update(
            dtype="float32",
            count=1,
            nodata=float(NODATA),
            compress="deflate",
            predictor=3,
            tiled=False,
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        try:
            rio_delete(out_path)
        except Exception:
            os.remove(out_path)

    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(data, 1)
    return out_path


def _compute_distance_weights(shape, valid_mask=None):
    """Compute distance-to-edge weights for blending. Pixels near center get higher weights."""
    from scipy.ndimage import distance_transform_edt
    
    h, w = shape
    
    if valid_mask is not None:
        # Distance from invalid pixels (edges of valid region)
        weights = distance_transform_edt(valid_mask).astype(np.float32)
    else:
        # Distance from image edges
        edge_mask = np.ones((h, w), dtype=bool)
        edge_mask[0, :] = False
        edge_mask[-1, :] = False
        edge_mask[:, 0] = False
        edge_mask[:, -1] = False
        weights = distance_transform_edt(edge_mask).astype(np.float32)
    
    # Normalize to [0, 1] with minimum weight at edges
    if weights.max() > 0:
        weights = weights / weights.max()
    
    # Apply power to make center weighting stronger
    weights = weights ** 4
    
    # Ensure minimum weight for valid pixels
    weights = np.clip(weights, 0.001, 1.0)
    
    return weights


def build_averaged_mosaic(tif_list, out_path, compress=None, nodata=NODATA):
    """Build mosaic with distance-weighted blending at overlaps."""
    assert len(tif_list) > 0
    
    print("  Building blended mosaic with distance-weighted averaging...")
    
    # First pass: determine output bounds and resolution
    srcs = [rasterio.open(fp) for fp in tif_list]
    try:
        # Get combined bounds
        all_bounds = [s.bounds for s in srcs]
        dst_w = min(b.left for b in all_bounds)
        dst_s = min(b.bottom for b in all_bounds)
        dst_e = max(b.right for b in all_bounds)
        dst_n = max(b.top for b in all_bounds)
        
        # Use resolution from first source
        res_x = srcs[0].res[0]
        res_y = srcs[0].res[1]
        crs = srcs[0].crs
        
        # Compute output dimensions
        out_width = int(np.ceil((dst_e - dst_w) / res_x))
        out_height = int(np.ceil((dst_n - dst_s) / res_y))
        
        # Output transform
        from rasterio.transform import from_bounds
        out_transform = from_bounds(dst_w, dst_s, dst_e, dst_n, out_width, out_height)
        
        print(f"    Output size: {out_width} x {out_height}")
        
        # Initialize accumulators
        weighted_sum = np.zeros((out_height, out_width), dtype=np.float64)
        weight_sum = np.zeros((out_height, out_width), dtype=np.float64)
        
        # Process each tile
        for i, src in enumerate(tqdm(srcs, desc="    Blending tiles")):
            # Read data
            data = src.read(1).astype(np.float32)
            data = np.where(data == nodata, np.nan, data)

            #margin = 16  # discard 16px border
            #data[:margin, :] = np.nan
            #data[-margin:, :] = np.nan
            #data[:, :margin] = np.nan
            #data[:, -margin:] = np.nan
            
            # Create valid mask and compute weights
            valid = np.isfinite(data)
            if not np.any(valid):
                continue
            
            weights = _compute_distance_weights(data.shape, valid_mask=valid)
            weights = np.where(valid, weights, 0)
            
            # Find where this tile maps in output
            # Get tile bounds
            tile_bounds = src.bounds
            
            # Compute pixel coordinates in output
            col_off = int(np.round((tile_bounds.left - dst_w) / res_x))
            row_off = int(np.round((dst_n - tile_bounds.top) / res_y))
            
            # Handle edge cases
            src_row_start = max(0, -row_off)
            src_col_start = max(0, -col_off)
            dst_row_start = max(0, row_off)
            dst_col_start = max(0, col_off)
            
            src_row_end = min(data.shape[0], out_height - row_off)
            src_col_end = min(data.shape[1], out_width - col_off)
            dst_row_end = min(out_height, row_off + data.shape[0])
            dst_col_end = min(out_width, col_off + data.shape[1])
            
            # Extract regions
            src_data = data[src_row_start:src_row_end, src_col_start:src_col_end]
            src_weights = weights[src_row_start:src_row_end, src_col_start:src_col_end]
            
            # Replace NaN with 0 for accumulation (weights will be 0 there)
            src_data_clean = np.where(np.isfinite(src_data), src_data, 0)
            
            # Accumulate
            weighted_sum[dst_row_start:dst_row_end, dst_col_start:dst_col_end] += src_data_clean * src_weights
            weight_sum[dst_row_start:dst_row_end, dst_col_start:dst_col_end] += src_weights
        
        # Compute weighted average
        with np.errstate(divide='ignore', invalid='ignore'):
            result = weighted_sum / weight_sum
        
        # Set nodata where no data was accumulated
        result = np.where(weight_sum > 0, result, nodata).astype(np.float32)
        
        # Write output
        prof = {
            "driver": "GTiff",
            "height": out_height,
            "width": out_width,
            "count": 1,
            "dtype": "float32",
            "crs": crs,
            "transform": out_transform,
            "tiled": False,
            "nodata": float(nodata),
        }
        if compress:
            prof["compress"] = compress

        if os.path.exists(out_path):
            try:
                rio_delete(out_path)
            except Exception:
                os.remove(out_path)

        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(result, 1)
        
        print(f"    Blended mosaic saved to {out_path}")
        
    finally:
        for s in srcs:
            s.close()
    
    return out_path


def _apply_gt_cleanup(a):
    a = a.astype(np.float32)
    a = np.where(a == NODATA, np.nan, a)
    a = np.where(np.isfinite(a), a, np.nan)
    return a


def _apply_pred_cleanup(a):
    a = a.astype(np.float32)
    a = np.where(a == NODATA, np.nan, a)
    a = np.where(np.isfinite(a), a, np.nan)
    return a


def _compute_optimal_rotation(mask):
    """Compute optimal rotation angle to make the longest edge of the region horizontal."""
    from scipy.spatial import ConvexHull
    
    # Get coordinates of valid pixels
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return 0.0
    
    # Subsample for efficiency if too many points
    if len(xs) > 5000:
        idx = np.random.choice(len(xs), 5000, replace=False)
        xs, ys = xs[idx], ys[idx]
    
    points = np.column_stack([xs, ys])
    
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    except:
        # Fall back to all points if convex hull fails
        hull_points = points
    
    # Find minimum area bounding rectangle using rotating calipers approach
    # Test angles based on hull edges
    best_angle = 0.0
    best_aspect = 0.0  # width / height ratio
    
    n = len(hull_points)
    for i in range(n):
        # Get edge vector
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % n]
        edge = p2 - p1
        
        # Angle of this edge
        edge_angle = np.degrees(np.arctan2(edge[1], edge[0]))
        
        # Rotate all points by negative of this angle
        rad = np.radians(-edge_angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rotated = np.column_stack([
            hull_points[:, 0] * cos_a - hull_points[:, 1] * sin_a,
            hull_points[:, 0] * sin_a + hull_points[:, 1] * cos_a
        ])
        
        # Compute bounding box
        width = rotated[:, 0].max() - rotated[:, 0].min()
        height = rotated[:, 1].max() - rotated[:, 1].min()
        
        # We want width > height (longest side horizontal)
        aspect = width / max(height, 1e-6)
        
        if aspect > best_aspect:
            best_aspect = aspect
            best_angle = -edge_angle  # Negate to rotate data
    
    # Normalize angle to reasonable range
    while best_angle > 90:
        best_angle -= 180
    while best_angle < -90:
        best_angle += 180
    
    # If result would make it taller than wide, rotate 90 degrees
    if best_aspect < 1.0:
        best_angle += 90 if best_angle < 0 else -90
    
    return best_angle


def _crop_to_valid(arr, mask, padding=10):
    """Crop array to bounding box of valid pixels with padding."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return arr, (0, arr.shape[0], 0, arr.shape[1])
    
    y_min, y_max = max(0, ys.min() - padding), min(arr.shape[0], ys.max() + padding + 1)
    x_min, x_max = max(0, xs.min() - padding), min(arr.shape[1], xs.max() + padding + 1)
    
    return arr[y_min:y_max, x_min:x_max], (y_min, y_max, x_min, x_max)


def _rotate_and_crop(arr, angle, mask=None):
    """Rotate array and optionally crop to valid region."""
    from scipy.ndimage import rotate as ndimage_rotate
    
    if abs(angle) < 0.5:  # Skip rotation for very small angles
        rotated = arr
    else:
        # Rotate with NaN-safe handling
        arr_filled = np.where(np.isfinite(arr), arr, 0)
        rotated = ndimage_rotate(arr_filled, angle, reshape=True, order=1, mode='constant', cval=np.nan)
        
        # Also rotate the mask to track valid regions
        mask_rot = ndimage_rotate((~np.isnan(arr)).astype(float), angle, reshape=True, order=0, mode='constant', cval=0)
        rotated = np.where(mask_rot > 0.5, rotated, np.nan)
    
    return rotated


def plot_2d_maps(gt_array, pred_array, diff_array, out_path, auto_orient=True, manual_rotation=None, vmin_override=None, vmax_override=None, pct_clip=2.0, split_stack=False):
    gt_array = gt_array.astype(np.float32)
    pred_array = pred_array.astype(np.float32)
    diff_array = diff_array.astype(np.float32)
    
    # Create combined valid mask
    valid_mask = np.isfinite(gt_array) & np.isfinite(pred_array)
    
    if manual_rotation is not None:
        # Use manually specified rotation
        angle = manual_rotation
        print(f"  Manual rotation: {angle:.1f} degrees")
        
        # Rotate all arrays
        gt_rot = _rotate_and_crop(gt_array, angle)
        pred_rot = _rotate_and_crop(pred_array, angle)
        diff_rot = _rotate_and_crop(diff_array, angle)
        
        # Compute new valid mask and crop
        valid_rot = np.isfinite(gt_rot) & np.isfinite(pred_rot)
        gt_array, bbox = _crop_to_valid(gt_rot, valid_rot, padding=20)
        pred_array, _ = _crop_to_valid(pred_rot, valid_rot, padding=20)
        diff_array, _ = _crop_to_valid(diff_rot, valid_rot, padding=20)
    elif auto_orient:
        # Compute optimal rotation angle
        angle = _compute_optimal_rotation(valid_mask)
        print(f"  Auto-orientation: rotating by {angle:.1f} degrees")
        
        # Rotate all arrays
        gt_rot = _rotate_and_crop(gt_array, angle)
        pred_rot = _rotate_and_crop(pred_array, angle)
        diff_rot = _rotate_and_crop(diff_array, angle)
        
        # Compute new valid mask and crop
        valid_rot = np.isfinite(gt_rot) & np.isfinite(pred_rot)
        gt_array, bbox = _crop_to_valid(gt_rot, valid_rot, padding=20)
        pred_array, _ = _crop_to_valid(pred_rot, valid_rot, padding=20)
        diff_array, _ = _crop_to_valid(diff_rot, valid_rot, padding=20)
    else:
        # Just crop without rotation
        gt_array, bbox = _crop_to_valid(gt_array, valid_mask, padding=20)
        pred_array, _ = _crop_to_valid(pred_array, valid_mask, padding=20)
        diff_array, _ = _crop_to_valid(diff_array, valid_mask, padding=20)

    stack = np.stack([gt_array, pred_array], axis=0)
    
    # Color scale: use manual overrides if provided, otherwise percentile clipping
    if vmin_override is not None:
        vmin = float(vmin_override)
    else:
        vmin = float(np.nanpercentile(stack, pct_clip))
    
    if vmax_override is not None:
        vmax = float(vmax_override)
    else:
        vmax = float(np.nanpercentile(stack, 100 - pct_clip))
    
    print(f"  Color scale: vmin={vmin:.2f}, vmax={vmax:.2f} (pct_clip={pct_clip}%)")

    d = diff_array[np.isfinite(diff_array)]
    A = np.percentile(np.abs(d), 100 - pct_clip) if d.size else 1.0
    A = float(max(A, 1e-6))
    
    # Compute aspect ratio for proper scaling
    h, w = gt_array.shape
    aspect = w / h
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    if split_stack:
        # Split wide region in half horizontally, stack halves vertically
        mid = w // 2
        gt_left, gt_right = gt_array[:, :mid], gt_array[:, mid:]
        pred_left, pred_right = pred_array[:, :mid], pred_array[:, mid:]
        diff_left, diff_right = diff_array[:, :mid], diff_array[:, mid:]
        
        print(f"  Split-stack mode: splitting {w}px width into two {mid}px halves")
        
        # Figure sizing - 3 groups, each with 2 image rows + colorbar
        half_aspect = mid / h
        fig_width = min(12, max(6, half_aspect * 3))
        fig_height = fig_width / half_aspect * 6 + 5  # 6 image rows + colorbars + spacing
        
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # GridSpec: 11 rows (2 img + 1 cbar + spacer) × 3 groups, minus last spacer
        # Format: [img, img, cbar, space, img, img, cbar, space, img, img, cbar]
        gs = GridSpec(11, 1, figure=fig, 
                      height_ratios=[1, 1, 0.08, 0.65, 1, 1, 0.08, 0.65, 1, 1, 0.08],
                      hspace=0.02)
        
        groups = [
            ("Ground Truth", gt_left, gt_right, "terrain", vmin, vmax, "LiDAR Residual (m)", 0),
            ("Prediction", pred_left, pred_right, "terrain", vmin, vmax, "LiDAR Residual (m)", 4),
            ("Error (Pred - GT)", diff_left, diff_right, "seismic", -A, +A, "Error (m)", 8),
        ]
        
        for title, left_data, right_data, cmap, v0, v1, label, row_start in groups:
            ax_top = fig.add_subplot(gs[row_start])
            ax_bot = fig.add_subplot(gs[row_start + 1])
            cax = fig.add_subplot(gs[row_start + 2])
            
            # Title only on top image with padding
            ax_top.set_title(title, fontsize=12, fontweight='bold', pad=10)
            
            im_top = ax_top.imshow(left_data, cmap=cmap, vmin=v0, vmax=v1, aspect='equal')
            im_bot = ax_bot.imshow(right_data, cmap=cmap, vmin=v0, vmax=v1, aspect='equal')
            
            ax_top.axis("off")
            ax_bot.axis("off")
            
            # Shared colorbar below both halves
            cbar = fig.colorbar(im_top, cax=cax, orientation="horizontal")
            cbar.set_label(label, fontsize=10)
            cbar.ax.tick_params(labelsize=8)
    else:
        # Standard 3-row layout
        # Dynamically size figure based on data aspect ratio
        if aspect >= 1:  # Wider than tall
            fig_width = min(14, max(8, aspect * 4))
            fig_height = fig_width / aspect * 3 + 2  # 3 subplots + colorbar space
        else:  # Taller than wide
            fig_height = min(18, max(10, 12 / aspect))
            fig_width = fig_height * aspect / 3 + 1
        
        fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height))

        im0 = axes[0].imshow(gt_array, cmap="terrain", vmin=vmin, vmax=vmax, aspect='equal')
        axes[0].set_title("Ground Truth", fontsize=12, fontweight='bold')
        axes[0].axis("off")

        im1 = axes[1].imshow(pred_array, cmap="terrain", vmin=vmin, vmax=vmax, aspect='equal')
        axes[1].set_title("Prediction", fontsize=12, fontweight='bold')
        axes[1].axis("off")

        im2 = axes[2].imshow(diff_array, cmap="seismic", vmin=-A, vmax=+A, aspect='equal')
        axes[2].set_title("Error (Pred - GT)", fontsize=12, fontweight='bold')
        axes[2].axis("off")
        
        for ax, im, label in [(axes[0], im0, "LiDAR Residual (m)"), 
                               (axes[1], im1, "LiDAR Residual (m)"), 
                               (axes[2], im2, "Error (m)")]:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            cbar.set_label(label, fontsize=10)
            cbar.ax.tick_params(labelsize=8)

    if not split_stack:
        plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    print(f"Saved 2D composite to {out_path}")


def _metric_params_from_cfg(cfg):
    px = float(cfg.get("data", {}).get("pixel_size_m", 1.0))
    jsd_scales = tuple(cfg.get("evaluation", {}).get("jsd_scales_m", [1.0, 2.0, 5.0, 10.0]))
    jsd_bins = int(cfg.get("evaluation", {}).get("jsd_bins", 256))
    use_sobel = bool(cfg.get("evaluation", {}).get("nae_use_sobel", True))
    deg = bool(cfg.get("evaluation", {}).get("nae_degrees", True))
    use_window = bool(cfg.get("evaluation", {}).get("psd_window", True))
    return px, jsd_scales, jsd_bins, use_sobel, deg, use_window


def _compute_metrics_tensor(gt_t, pr_t, mask_t, px, jsd_scales, jsd_bins, use_sobel, deg, use_window):
    full_mask = mask_t & torch.isfinite(gt_t) & torch.isfinite(pr_t)

    diff = gt_t - pr_t
    denom = torch.abs(gt_t).clamp_min(1e-6)
    abs_rel = (torch.abs(diff) / denom)[full_mask]
    abs_rel_error = float(abs_rel.mean().item()) if abs_rel.numel() > 0 else float("nan")

    return {
        "rmse_phys_m": float(rmse_recon(gt_t, pr_t, mask=full_mask).item()),
        "bias_phys_m": float(bias_recon(gt_t, pr_t, mask=full_mask).item()),
        "sigma_error_pct": float(sigma_error(gt_t, pr_t, mask=full_mask).item()),
        "normal_angle_error_deg": float(normal_angle_error(gt_t, pr_t, mask=full_mask, pixel_size=px, use_sobel=use_sobel, degrees=deg).item()),
        "zncc": float(zncc(gt_t, pr_t, mask=full_mask).item()),
        "jsd": float(average_jsd_multiscale(gt_t, pr_t, scales_m=jsd_scales, pixel_size=px, bins=jsd_bins, mask=full_mask).item()),
        "psd_rmse": float(log_psd_rmse(gt_t, pr_t, pixel_size=px, mask=full_mask, window=use_window).item()),
        "abs_rel_error": abs_rel_error,
    }

def _valid_mask_from_arrays(gt_array, pred_array):
    return np.isfinite(gt_array) & np.isfinite(pred_array)


def compute_and_save_region_metrics(gt_array, pred_array, out_dir, cfg):
    stats_dir = os.path.join(out_dir, "reconstruction_statistics")
    os.makedirs(stats_dir, exist_ok=True)

    mask_np = _valid_mask_from_arrays(gt_array, pred_array)
    if not np.any(mask_np):
        raise ValueError("No valid pixels found for region metrics.")
    gt_t = torch.from_numpy(gt_array).float()
    pr_t = torch.from_numpy(pred_array).float()
    m_t = torch.from_numpy(mask_np).bool()

    px, jsd_scales, jsd_bins, use_sobel, deg, use_window = _metric_params_from_cfg(cfg)
    metrics = _compute_metrics_tensor(gt_t, pr_t, m_t, px, jsd_scales, jsd_bins, use_sobel, deg, use_window)

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


def compute_and_save_patch_metrics(out_dir, cfg):
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
        lidar_dir = lidar_dir[0]

    px, jsd_scales, jsd_bins, use_sobel, deg, use_window = _metric_params_from_cfg(cfg)

    rows = []
    for pred_fp in tqdm(pred_tifs, desc="Per-patch metrics"):
        basename = os.path.basename(pred_fp)
        tile_id = basename[len("pred_"):-4]

        gt_fp = find_lidar_patch(lidar_dir, tile_id)

        with rasterio.open(gt_fp) as g, rasterio.open(pred_fp) as p:
            g_arr = g.read().astype(np.float32)
            gt = g_arr[0]
            if g.count >= 2:
                gmask = g_arr[1] > 0.5
            else:
                gmask = np.isfinite(gt)

            gt = np.where(gmask & np.isfinite(gt), gt, np.nan).astype(np.float32)

            pr = p.read(1, masked=True).astype(np.float32).filled(np.nan)
            pr = np.where(pr == NODATA, np.nan, pr).astype(np.float32)

            gt = _apply_gt_cleanup(gt)
            pr = _apply_pred_cleanup(pr)

            if (gt.shape != pr.shape) or (g.transform != p.transform):
                pr_aligned = np.full_like(gt, NODATA, dtype=np.float32)
                reproject(
                    source=np.where(np.isnan(pr), NODATA, pr).astype(np.float32),
                    destination=pr_aligned,
                    src_transform=p.transform, src_crs=p.crs,
                    dst_transform=g.transform, dst_crs=g.crs,
                    resampling=Resampling.bilinear,
                    src_nodata=p.nodata if p.nodata is not None else NODATA,
                    dst_nodata=NODATA,
                )
                pr = np.where(pr_aligned == NODATA, np.nan, pr_aligned).astype(np.float32)
                pr = _apply_pred_cleanup(pr)

        mask_np = _valid_mask_from_arrays(gt, pr)
        if not np.any(mask_np):
            row = {"tile_id": tile_id, "valid_pixel_count": 0}
            for k in [
                "rmse_phys_m", "bias_phys_m", "sigma_error_pct",
                "normal_angle_error_deg", "zncc", "jsd", "psd_rmse",
                "gt_mean_val", "gt_std_val", "pred_mean_val", "pred_std_val", "abs_rel_error",
                "gt_min_val", "gt_max_val", "pred_min_val", "pred_max_val"
            ]:
                row[k] = float("nan")
            rows.append(row)
            continue

        gt_t = torch.from_numpy(gt).float()
        pr_t = torch.from_numpy(pr).float()
        m_t = torch.from_numpy(mask_np).bool()

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

    csv_path = os.path.join(stats_dir, "patch_reconstruction_stats.csv")
    fieldnames = [
        "tile_id", "valid_pixel_count",
        "rmse_phys_m", "bias_phys_m",
        "sigma_error_pct", "normal_angle_error_deg", "zncc", "jsd", "psd_rmse", "abs_rel_error",
        "gt_mean_val", "gt_std_val", "pred_mean_val", "pred_std_val",
        "gt_min_val", "gt_max_val", "pred_min_val", "pred_max_val",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved per-patch reconstruction CSV → {csv_path}")

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
    metrics_keys = ["rmse_phys_m", "bias_phys_m", "sigma_error_pct",
                    "normal_angle_error_deg", "zncc", "jsd", "psd_rmse", "abs_rel_error"]

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
    mask = np.isfinite(gt_array) & np.isfinite(pred_array)
    gt = gt_array[mask].astype(np.float64)
    pr = pred_array[mask].astype(np.float64)
    if gt.size == 0 or pr.size == 0:
        raise ValueError("No valid pixels to build PDFs.")

    combined = np.concatenate([gt, pr], axis=0)
    xmin = float(np.percentile(combined, low_pct))
    xmax = float(np.percentile(combined, high_pct))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
        xmin = float(np.min(combined))
        xmax = float(np.max(combined))

    bin_edges = np.linspace(xmin, xmax, bins + 1, dtype=np.float64)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    gt_hist, _ = np.histogram(gt, bins=bin_edges, density=True)
    pr_hist, _ = np.histogram(pr, bins=bin_edges, density=True)

    if smooth_sigma is not None and smooth_sigma > 0:
        gt_hist = gaussian_filter1d(gt_hist, sigma=smooth_sigma, mode="nearest")
        pr_hist = gaussian_filter1d(pr_hist, sigma=smooth_sigma, mode="nearest")

    eps = 1e-12
    p = gt_hist * bin_width
    q = pr_hist * bin_width
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        jsd = float("nan")
    else:
        p = p / (p_sum + eps)
        q = q / (q_sum + eps)
        m = 0.5 * (p + q)

        def _kl(a, b):
            a_safe = np.clip(a, eps, 1.0)
            b_safe = np.clip(b, eps, 1.0)
            return float(np.sum(a_safe * np.log(a_safe / b_safe)))

        jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bin_centers, gt_hist, label=f"Ground truth (n={gt.size:,})", linewidth=2)
    ax.plot(bin_centers, pr_hist, label=f"Prediction (n={pr.size:,})", linewidth=2, linestyle="--")
    ax.set_xlim([xmin, xmax])
    
    # Extend y-axis slightly to make room for JSD label
    ymin, ymax = ax.get_ylim()
    ax.set_ylim([ymin, ymax * 1.15])
    
    ax.set_xlabel("LiDAR Residual (m)")
    ax.set_ylabel("Probability density")
    if title:
        ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.2)
    ax.legend()

    if np.isfinite(jsd):
        ax.text(0.02, 0.98, f"JSD = {jsd:.4f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=11,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved region PDF plot → {out_path}")

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


def subsample(arr, step):
    return arr[::step, ::step]


def shared_vmin_vmax(gt, pr, pct_low=1, pct_high=99, symmetric=False):
    v = np.concatenate([gt[np.isfinite(gt)], pr[np.isfinite(pr)]])
    if v.size == 0:
        return (-1.0, 1.0)
    if symmetric:
        A = float(np.percentile(np.abs(v), pct_high))
        A = max(A, 1e-6)
        return (-A, +A)
    vmin = float(np.percentile(v, pct_low))
    vmax = float(np.percentile(v, pct_high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.nanmin(v))
        vmax = float(np.nanmax(v))
    return (vmin, vmax)


def plot_single_3d_surface(ax, lidar_array, title="3D",
                           cmap="terrain", z_label="LiDAR Residual (m)",
                           vmin=None, vmax=None, zmin=None, zmax=None):
    Z = lidar_array.astype(np.float32)
    mask = np.isfinite(Z)
    Zm = np.ma.array(Z, mask=~mask)

    h, w = Z.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    surf = ax.plot_surface(
        X, Y, Zm,
        cmap=cmap, alpha=0.9, edgecolor="none",
        rstride=1, cstride=1, vmin=vmin, vmax=vmax
    )

    ax.set_xlim(0, w - 1)
    ax.set_ylim(0, h - 1)

    if (zmin is not None) and (zmax is not None):
        ax.set_zlim(zmin, zmax)

    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.view_init(elev=15, azim=70)
    return surf


def plot_all_three_3d_surfaces(gt_array, pred_array, diff_array,
                               step=4, out_path=None, plot_title="Combined 3D Plots"):
    gt_s = subsample(gt_array, step)
    pr_s = subsample(pred_array, step)
    diff_s = subsample(diff_array, step)

    vmin_gp, vmax_gp = shared_vmin_vmax(gt_s, pr_s, pct_low=1, pct_high=99, symmetric=False)
    zmin_gp, zmax_gp = vmin_gp, vmax_gp

    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(plot_title, fontsize=16)

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    s1 = plot_single_3d_surface(ax1, gt_s, "Ground Truth",
                                cmap="terrain", z_label="LiDAR Residual (m)",
                                vmin=vmin_gp, vmax=vmax_gp, zmin=zmin_gp, zmax=zmax_gp)
    fig.colorbar(s1, ax=ax1, shrink=0.5, aspect=10, label="m")
    ax1.set_ylabel("Pixel Y (1m)")
    ax1.set_xlabel("Pixel X (1m)")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    s2 = plot_single_3d_surface(ax2, pr_s, "Predicted",
                                cmap="terrain", z_label="LiDAR Residual (m)",
                                vmin=vmin_gp, vmax=vmax_gp, zmin=zmin_gp, zmax=zmax_gp)
    fig.colorbar(s2, ax=ax2, shrink=0.5, aspect=10, label="m")
    ax2.set_ylabel("Pixel Y (1m)")
    ax2.set_xlabel("Pixel X (1m)")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    s3 = plot_single_3d_surface(ax3, diff_s, "Error (Pred - GT)",
                                cmap="RdBu", z_label="Difference (m)")
    fig.colorbar(s3, ax=ax3, shrink=0.5, aspect=10, label="m")
    ax3.set_ylabel("Pixel Y (1m)")
    ax3.set_xlabel("Pixel X (1m)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"Saved 3D plots to {out_path}")
    plt.close(fig)


@torch.no_grad()
def run_predictions_and_mosaics(ckpt_path, config_yaml, out_dir,
                                sampler_name="ddpm", batch_size=8, num_workers=4, device="cuda",
                                zone_ids=None, max_tiles=None, seed=42, deterministic_order=True):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    state, cfg_from_ckpt, _ = load_checkpoint(ckpt_path, device)

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

    s2_dir = cfg["data"]["s2_dir"]
    lidar_dir = cfg["data"]["lidar_dir"]
    context_k = cfg["training"]["context_k"]
    noise_sched = cfg["training"]["noise_schedule"]
    timesteps = cfg["training"]["timesteps"]
    base_channels = cfg["model"]["base_channels"]
    embed_dim = cfg["model"]["embed_dim"]
    unet_depth = cfg["model"]["unet_depth"]
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

    dataset = LidarS2Dataset(
        lidar_dirs=lidar_dir if "lidar_dir" in cfg["data"] else cfg["data"].get("lidar_dirs", lidar_dir),
        s2_dirs=s2_dir if "s2_dir" in cfg["data"] else cfg["data"].get("s2_dirs", s2_dir),
        context_k=context_k,
        randomize_context=False,
        augment=False,
        debug=False,
        split_pids=subset_pids,
        split="val",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    samplers = {
        "ddpm": lambda m, s, c, a, d: p_sample_loop_ddpm(m, scheduler, s, c, a, d),
        "ddim": lambda m, s, c, a, d: p_sample_loop_ddim(m, scheduler, s, c, a, d),
        "plms": lambda m, s, c, a, d: p_sample_loop_plms(m, scheduler, s, c, a, d),
    }
    if sampler_name not in samplers:
        raise ValueError(f"Unknown sampler: {sampler_name}")
    sampler = samplers[sampler_name]

    pred_tiles_dir = os.path.join(out_dir, "pred_tiles")
    gt_tiles_dir = os.path.join(out_dir, "gt_tiles")
    os.makedirs(pred_tiles_dir, exist_ok=True)
    os.makedirs(gt_tiles_dir, exist_ok=True)

    pred_tifs = []
    gt_tifs = []
    start = time.perf_counter()

    processed = 0
    for batch in tqdm(loader, desc="Predicting tiles"):
        if (max_tiles is not None) and (processed >= max_tiles):
            break

        s2 = batch["s2"].to(device)
        attrs = batch["attrs"].to(device)
        lidar = batch["lidar"].to(device)
        tile_ids_batch = batch["tile_id"]

        pred = sampler(model, lidar.shape, s2, attrs, device).float()

        pm = batch.get("lidar_patch_mean", None)
        pm = pm.to(device).view(-1, 1, 1, 1)
        pred = pred + pm

        pred = pred.cpu().numpy()

        B = pred.shape[0]
        for i in range(B):
            if (max_tiles is not None) and (processed >= max_tiles):
                break
            tile_id = tile_ids_batch[i]
            gt_lidar_tif = find_lidar_patch(lidar_dir, tile_id)

            out_pred_tif = os.path.join(pred_tiles_dir, f"pred_{tile_id}.tif")
            write_tif_like(gt_lidar_tif, out_pred_tif, pred[i, 0])
            pred_tifs.append(out_pred_tif)

            out_gt_tif = os.path.join(gt_tiles_dir, f"gt_{tile_id}.tif")
            write_gt_singleband_from_patch(gt_lidar_tif, out_gt_tif)
            gt_tifs.append(out_gt_tif)

            processed += 1

    elapsed = time.perf_counter() - start
    print(f"\nFinished per-tile predictions for {len(pred_tifs)} tiles in {elapsed/60:.1f} min.")

    pred_mosaic_path = os.path.join(out_dir, "pred_mosaic.tif")
    print("Mosaicking predictions →", pred_mosaic_path)
    build_averaged_mosaic(pred_tifs, pred_mosaic_path, compress="deflate")

    gt_mosaic_path = os.path.join(out_dir, "gt_mosaic.tif")
    print("Mosaicking ground truth →", gt_mosaic_path)
    build_averaged_mosaic(sorted(list(set(gt_tifs))), gt_mosaic_path, compress="deflate")

    return pred_mosaic_path, gt_mosaic_path, cfg


def align_and_save_diff(pred_mosaic_path, gt_mosaic_path, out_dir):
    with rasterio.open(gt_mosaic_path) as g, rasterio.open(pred_mosaic_path) as p:
        gt_array = g.read(1, masked=True).astype(np.float32).filled(np.nan)
        pred_array = p.read(1, masked=True).astype(np.float32).filled(np.nan)

        gt_array = np.where(gt_array == NODATA, np.nan, gt_array)
        pred_array = np.where(pred_array == NODATA, np.nan, pred_array)

        gt_array = _apply_gt_cleanup(gt_array)
        pred_array = _apply_pred_cleanup(pred_array)

        if (gt_array.shape != pred_array.shape) or (g.transform != p.transform):
            pred_aligned = np.full_like(gt_array, NODATA, dtype=np.float32)
            reproject(
                source=np.where(np.isnan(pred_array), NODATA, pred_array).astype(np.float32),
                destination=pred_aligned,
                src_transform=p.transform, src_crs=p.crs,
                dst_transform=g.transform, dst_crs=g.crs,
                resampling=Resampling.bilinear,
                src_nodata=p.nodata if p.nodata is not None else NODATA,
                dst_nodata=NODATA,
            )
            pred_array = np.where(pred_aligned == NODATA, np.nan, pred_aligned).astype(np.float32)
            pred_array = _apply_pred_cleanup(pred_array)

        diff_array = pred_array - gt_array
        diff_path = os.path.join(out_dir, "diff_pred_minus_gt.tif")

        diff_to_write = np.where(np.isfinite(diff_array), diff_array, NODATA).astype(np.float32)

        prof = g.profile.copy()
        prof.update(dtype="float32", count=1, compress="deflate", nodata=float(NODATA))

        with rasterio.open(diff_path, "w", **prof) as dst:
            dst.write(diff_to_write, 1)

        print("Wrote diff raster →", diff_path)
    return gt_array, pred_array, diff_array, diff_path


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
        "--samplers",
        type=str,
        default=None,
        help=(
            "Comma-separated list of samplers to run (overrides --sampler). "
            "Example: 'ddpm,ddim,plms'."
        ),
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
        "--model",
        type=str,
        default="cosine",
        choices=["cosine", "linear"],
        help="Model type to use: 'cosine' or 'linear'. Defaults to 'cosine'.",
    )
    parser.add_argument(
        "--deterministic-order",
        action="store_true",
        help="Use deterministic tile ordering when downsampling with max-tiles.",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip reconstruction statistics calculation (only regenerate plots).",
    )
    parser.add_argument(
        "--rotation",
        type=float,
        default=None,
        help="Manual rotation angle in degrees for 2D plots. Use interactive_rotation.py to find optimal value.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Manual minimum value for elevation colorbar. If not set, uses percentile clipping.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Manual maximum value for elevation colorbar. If not set, uses percentile clipping.",
    )
    parser.add_argument(
        "--pct-clip",
        type=float,
        default=2.0,
        help="Percentile for color scale clipping (default: 2.0, meaning clip at 2nd and 98th percentile).",
    )
    parser.add_argument(
        "--rebuild-mosaic",
        action="store_true", 
        help="Rebuild mosaics from existing pred_tiles/ even when using --skip-predict.",
    )
    parser.add_argument(
        "--split-stack",
        action="store_true",
        help="Split wide regions in half horizontally and stack vertically for better visibility (useful for Tuk).",
    )

    args = parser.parse_args()

    preset = get_region_preset(args.region, args.model)
    region_key = preset["region_key"]
    region_name = preset["pretty_name"]
    zone_ids = preset["zone_ids"]
    CKPT_PATH = preset["ckpt_path"]
    TEST_S2_DIR = preset["s2_dir"]
    TEST_LIDAR_DIR = preset["lidar_dir"]
    base_out_dir = preset["out_dir"]

    device = args.device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    ckpt_tmp = torch.load(CKPT_PATH, map_location=device)
    config = ckpt_tmp["config"]

    config["data"]["s2_dir"] = TEST_S2_DIR
    config["data"]["lidar_dir"] = TEST_LIDAR_DIR
    if "system" in config:
        config["system"]["device"] = device

    if args.samplers:
        samplers = [s.strip().lower() for s in args.samplers.split(",") if s.strip()]
    else:
        samplers = [args.sampler]

    invalid = [s for s in samplers if s not in ("ddpm", "ddim", "plms")]
    if invalid:
        raise ValueError(f"Unknown sampler(s): {invalid}. Choose from ddpm, ddim, plms.")

    for sampler_name in samplers:
        out_dir = f"{base_out_dir}_{sampler_name}"
        os.makedirs(out_dir, exist_ok=True)
        if "logging" in config:
            config["logging"]["output_dir"] = out_dir

        pred_mosaic_path = os.path.join(out_dir, "pred_mosaic.tif")
        gt_mosaic_path = os.path.join(out_dir, "gt_mosaic.tif")

        reuse_ok = (
            args.skip_predict
            and os.path.exists(pred_mosaic_path)
            and os.path.exists(gt_mosaic_path)
            and not args.rebuild_mosaic  # Force rebuild if requested
        )

        if reuse_ok:
            print("Skipping prediction (reuse mode). Using existing mosaics:")
            print(f"  Pred mosaic: {pred_mosaic_path}")
            print(f"  GT mosaic:   {gt_mosaic_path}")
            cfg_used = config
        elif args.skip_predict and args.rebuild_mosaic:
            # Rebuild mosaics from existing tiles without re-predicting
            print("Rebuilding mosaics from existing tiles (--rebuild-mosaic)...")
            pred_tiles_dir = os.path.join(out_dir, "pred_tiles")
            gt_tiles_dir = os.path.join(out_dir, "gt_tiles")
            
            if not os.path.isdir(pred_tiles_dir):
                raise FileNotFoundError(f"No pred_tiles directory found at {pred_tiles_dir}")
            
            pred_tifs = sorted(glob.glob(os.path.join(pred_tiles_dir, "pred_*.tif")))
            gt_tifs = sorted(glob.glob(os.path.join(gt_tiles_dir, "gt_*.tif")))
            
            if not pred_tifs:
                raise FileNotFoundError(f"No predicted tiles found in {pred_tiles_dir}")
            
            print(f"  Found {len(pred_tifs)} predicted tiles, {len(gt_tifs)} GT tiles")
            
            print("  Mosaicking predictions →", pred_mosaic_path)
            build_averaged_mosaic(pred_tifs, pred_mosaic_path, compress="deflate")
            
            print("  Mosaicking ground truth →", gt_mosaic_path)
            build_averaged_mosaic(gt_tifs, gt_mosaic_path, compress="deflate")
            
            cfg_used = config
        else:
            print(f"Running prediction + mosaicking (sampler={sampler_name})...")
            pred_mosaic_path, gt_mosaic_path, cfg_used = run_predictions_and_mosaics(
                ckpt_path=CKPT_PATH,
                config_yaml=config,
                out_dir=out_dir,
                sampler_name=sampler_name,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                zone_ids=zone_ids,
                max_tiles=args.max_tiles,
                seed=args.seed,
                deterministic_order=args.deterministic_order,
            )

        gt_array, pred_array, diff_array, diff_path = align_and_save_diff(
            pred_mosaic_path=pred_mosaic_path,
            gt_mosaic_path=gt_mosaic_path,
            out_dir=out_dir,
        )

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

        two_d_path = os.path.join(out_dir, f"{region_key}_mosaic_2d.png")
        plot_2d_maps(
            gt_array, pred_array, diff_array, two_d_path,
            manual_rotation=args.rotation,
            vmin_override=args.vmin,
            vmax_override=args.vmax,
            pct_clip=args.pct_clip,
            split_stack=args.split_stack,
        )

        three_d_path = os.path.join(out_dir, f"{region_key}_mosaic_3d.png")
        plot_all_three_3d_surfaces(
            gt_array=gt_array,
            pred_array=pred_array,
            diff_array=diff_array,
            step=20,
            out_path=three_d_path,
            plot_title=f"Predicted 3D Surface for {region_name}",
        )

        if not args.skip_stats:
            compute_and_save_region_metrics(gt_array, pred_array, out_dir, cfg_used)
            compute_and_save_patch_metrics(out_dir, cfg_used)
        else:
            print("Skipping reconstruction statistics (--skip-stats).")

        print("\nDone.")
        print("Outputs:")
        print(f"  Region:       {region_name} ({region_key})")
        print(f"  Sampler:      {sampler_name}")
        print(f"  GT mosaic:    {gt_mosaic_path}")
        print(f"  Pred mosaic:  {pred_mosaic_path}")
        print(f"  Diff raster:  {diff_path}")
        print(f"  Region stats: {os.path.join(out_dir, 'reconstruction_statistics', 'region_reconstruction_stats.json')}")
        print(f"  Patch CSV:    {os.path.join(out_dir, 'reconstruction_statistics', 'patch_reconstruction_stats.csv')}")
        print(f"  Patch summary:{os.path.join(out_dir, 'reconstruction_statistics', 'patch_reconstruction_summary.json')}")


if __name__ == "__main__":
    main()