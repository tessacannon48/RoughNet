#!/usr/bin/env python3
"""Rotation preview tool to find optimal orientation for 2D plots."""

import os
import sys
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - works without display
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as ndimage_rotate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NODATA = -9999.0


def load_mosaic(region, model="cosine", sampler="plms"):
    """Load the GT mosaic for a given region."""
    root = PROJECT_ROOT
    prefix = "final_val" if region in ("pondinlet", "tuk") else "final_test"
    gt_path = f"{root}/figures/{model}_model/{prefix}_{region}_{sampler}/gt_mosaic.tif"
    
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT mosaic not found: {gt_path}")
    
    with rasterio.open(gt_path) as src:
        arr = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        arr = np.where(arr == NODATA, np.nan, arr)
    
    return arr


def rotate_array(arr, angle):
    """Rotate array by given angle in degrees."""
    if abs(angle) < 0.1:
        return arr
    
    arr_filled = np.where(np.isfinite(arr), arr, 0)
    rotated = ndimage_rotate(arr_filled, angle, reshape=True, order=1, mode='constant', cval=np.nan)
    
    # Rotate mask too
    mask_rot = ndimage_rotate((~np.isnan(arr)).astype(float), angle, reshape=True, order=0, mode='constant', cval=0)
    rotated = np.where(mask_rot > 0.5, rotated, np.nan)
    
    return rotated


def crop_to_valid(arr, padding=20):
    """Crop to bounding box of valid pixels."""
    mask = np.isfinite(arr)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return arr
    
    y_min = max(0, ys.min() - padding)
    y_max = min(arr.shape[0], ys.max() + padding + 1)
    x_min = max(0, xs.min() - padding)
    x_max = min(arr.shape[1], xs.max() + padding + 1)
    
    return arr[y_min:y_max, x_min:x_max]


def generate_rotation_grid(region, model="cosine", sampler="plms", angles=None):
    """Generate a grid of rotation previews."""
    print(f"Loading {region} mosaic...")
    original = load_mosaic(region, model, sampler)
    
    # Subsample for faster processing
    step = max(1, min(original.shape) // 400)
    data = original[::step, ::step]
    print(f"Using {step}x subsampled data ({data.shape})")
    
    if angles is None:
        # Default: test every 15 degrees from -90 to 90
        angles = list(range(-90, 91, 15))
    
    n_angles = len(angles)
    cols = 4
    rows = (n_angles + cols - 1) // cols
    
    vmin, vmax = np.nanpercentile(data, [1, 99])
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()
    
    for i, angle in enumerate(angles):
        print(f"  Generating rotation {angle}°...")
        rotated = rotate_array(data, angle)
        cropped = crop_to_valid(rotated)
        
        h, w = cropped.shape
        aspect = w / h
        
        axes[i].imshow(cropped, cmap='terrain', vmin=vmin, vmax=vmax, aspect='equal')
        axes[i].set_title(f"Rotation: {angle}°\n(aspect {aspect:.2f})", fontsize=11, fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_angles, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Rotation Preview Grid - {region.title()} ({model}/{sampler})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_path = f"{PROJECT_ROOT}/figures/rotation_preview_{region}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nSaved rotation preview to: {out_path}")
    print(f"\nOnce you find the best angle, run:")
    print(f"  python evaluation.py --region {region} --sampler {sampler} --model {model} --skip-predict --skip-stats --rotation <ANGLE>")
    
    return out_path


def generate_fine_grid(region, model, sampler, center_angle, step=2, span=20):
    """Generate fine-grained rotation previews around a center angle."""
    angles = [center_angle + i for i in range(-span, span + 1, step)]
    return generate_rotation_grid(region, model, sampler, angles)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate rotation preview grid for finding optimal plot orientation")
    parser.add_argument("--region", type=str, required=True, choices=["pondinlet", "tuk", "cambridge"])
    parser.add_argument("--model", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--sampler", type=str, default="plms", choices=["ddpm", "ddim", "plms"])
    parser.add_argument("--fine", type=float, default=None, 
                        help="Center angle for fine-grained search (generates ±20° around this value in 2° steps)")
    parser.add_argument("--angles", type=str, default=None,
                        help="Comma-separated list of specific angles to test, e.g. '45,50,55,60'")
    
    args = parser.parse_args()
    
    if args.angles:
        angles = [float(a.strip()) for a in args.angles.split(",")]
        generate_rotation_grid(args.region, args.model, args.sampler, angles)
    elif args.fine is not None:
        generate_fine_grid(args.region, args.model, args.sampler, args.fine)
    else:
        generate_rotation_grid(args.region, args.model, args.sampler)
