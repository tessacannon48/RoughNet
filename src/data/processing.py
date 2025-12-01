import os, glob, torch, rasterio
import numpy as np
from tqdm import tqdm

# =============================================================================
# DATA PREPOCESSING METHODS
# =============================================================================

def compute_s2_mean_std_multi(
    s2_root_list,
    num_times=6,
    num_bands=4,
    filenames=None,
    patch_group_dirs=None
):
    """
    Compute dataset-level mean/std for multi-temporal Sentinel-2 sets across multiple roots.

    Args:
        s2_root_list (str or list): One or more root directories containing S2 patch groups.
        num_times (int): Number of S2 timestamps per patch.
        num_bands (int): Number of S2 bands (e.g., 4 for RGB+NIR).
        filenames (list): List of filenames for each timestamp (e.g., ["t0.tif", "t1.tif", ...]).
        patch_group_dirs (list): Optional specific list of S2 group directories to compute stats on.
                                 If None, all "s2_patch_*" directories from all roots are used.
    """

    # Ensure input list consistency
    if isinstance(s2_root_list, str):
        s2_root_list = [s2_root_list]

    if filenames is None:
        filenames = [f"t{i}.tif" for i in range(num_times)]

    # === Collect all S2 patch directories ===
    if patch_group_dirs is None:
        group_dirs = []
        for root in s2_root_list:
            dirs = sorted([d for d in glob.glob(os.path.join(root, "s2_patch_*"))
                           if os.path.isdir(d)])
            group_dirs.extend(dirs)
    else:
        group_dirs = patch_group_dirs

    print(f"Computing S2 stats across {len(group_dirs)} patch directories from {len(s2_root_list)} roots.")
    for root in s2_root_list:
        print(f"  → {root}")

    # === Initialize accumulators ===
    C = num_times * num_bands
    sums   = torch.zeros(C, dtype=torch.float64)
    sums2  = torch.zeros(C, dtype=torch.float64)
    counts = torch.zeros(C, dtype=torch.float64)

    # === Compute incremental stats ===
    for gdir in tqdm(group_dirs, desc=f"Computing S2 stats ({num_times}x{num_bands})"):
        for ti, fname in enumerate(filenames):
            fp = os.path.join(gdir, fname)
            if not os.path.exists(fp):
                continue
            try:
                with rasterio.open(fp) as src:
                    arr = src.read()[:num_bands].astype(np.float32)
            except Exception as e:
                print(f"Warning: failed to read {fp}: {e}")
                continue

            arr = torch.from_numpy(arr).reshape(num_bands, -1)
            finite = torch.isfinite(arr)
            safe = torch.where(finite, arr, torch.zeros_like(arr))

            idx0, idx1 = ti * num_bands, (ti + 1) * num_bands
            sums[idx0:idx1]  += safe.sum(dim=1, dtype=torch.float64)
            sums2[idx0:idx1] += (safe ** 2).sum(dim=1, dtype=torch.float64)
            counts[idx0:idx1] += finite.sum(dim=1, dtype=torch.float64)

    # === Finalize statistics ===
    counts = torch.clamp(counts, min=1.0)
    mean = (sums / counts).to(torch.float32)
    var = (sums2 / counts) - (mean.to(torch.float64) ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-12)).to(torch.float32)

    print(f"Completed computation: mean/std for {C} channels ({num_times}×{num_bands}).")
    return mean, std

@torch.no_grad()
def per_patch_percentile_scale_bandwise_shared(s2: torch.Tensor,
                                               p_low: float = 2.0,
                                               p_high: float = 98.0,
                                               min_range: float = 1e-3,
                                               clamp01: bool = True):
    """
    s2: (4*T, H, W) with channels ordered [R,G,B,NIR] per time step.
    For each spectral band (R,G,B,NIR), pool all time slices in THIS PATCH,
    compute robust percentiles [p_low, p_high], and apply the SAME linear map
    to every time slice of that band. Returns scaled tensor (same shape).
    """
    assert s2.dim() == 3 and s2.shape[0] % 4 == 0, "Expected (4*T, H, W)"
    C, H, W = s2.shape
    T = C // 4
    out = s2.clone()

    # work in-place on CPU/GPU tensors
    for b in range(4):  # 0=R, 1=G, 2=B, 3=NIR
        band_stack = out[b::4]                 # shape: (T, H, W)
        x = band_stack.reshape(-1)             # all times pooled

        finite = torch.isfinite(x)
        if not finite.any():
            band_stack.zero_()
            continue

        xf = x[finite]
        qlow = torch.quantile(xf, torch.tensor(p_low/100.0, device=xf.device))
        qhigh = torch.quantile(xf, torch.tensor(p_high/100.0, device=xf.device))

        # ensure usable dynamic range
        if (qhigh - qlow) < min_range:
            med = torch.quantile(xf, torch.tensor(0.5, device=xf.device))
            qlow = med - min_range/2
            qhigh = med + min_range/2

        # clip then scale to [0,1]
        band_stack.clamp_(min=qlow.item(), max=qhigh.item())
        band_stack.sub_(qlow).div_(qhigh - qlow + 1e-6)

        if clamp01:
            band_stack.clamp_(0.0, 1.0)

        out[b::4] = band_stack
    return out