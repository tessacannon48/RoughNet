# recon_metrics.py
# Utilities for evaluating residual topography reconstructions

from __future__ import annotations
import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _ensure_float(x: torch.Tensor) -> torch.Tensor:
    return x if torch.is_floating_point(x) else x.float()

def _broadcast_mask_like(a: torch.Tensor, b: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if b is None:
        return None
    # expand mask to match a's shape across batch/channel dims if needed
    if b.shape == a.shape:
        return b
    # allow HxW mask to be broadcast to NxCxHxW or NxHxW
    while b.dim() < a.dim():
        b = b.unsqueeze(0)
    return b.expand_as(a)

def _valid_mask(gt: torch.Tensor, pred: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    # start with finite mask
    base = torch.isfinite(gt) & torch.isfinite(pred)
    if mask is not None:
        mask = _broadcast_mask_like(gt, mask)
        base = base & mask.bool()
    return base


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: Optional[Iterable[int]]=None, keepdim: bool=False) -> torch.Tensor:
    if dim is None:
        dim = tuple(range(x.dim()))
    elif isinstance(dim, int):
        dim = (dim,)
    mask = mask.to(x.dtype)
    x = x * mask
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1e-12)
    return x.sum(dim=dim, keepdim=keepdim) / denom


def _to_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts:
      HxW
      NxHxW
      NxCxHxW
    Returns:
      (N*, H, W) where N* is max(1, N*C) for convenient batch iteration.
    """
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    if x.dim() == 4:
        N, C, H, W = x.shape
        return x.reshape(N * C, H, W)
    raise ValueError(f"Unsupported tensor shape {x.shape}")


def _apply_hann2d(x: torch.Tensor) -> torch.Tensor:
    """Apply separable Hann window per 2D map to reduce spectral leakage."""
    H, W = x.shape
    device = x.device
    wy = torch.hann_window(H, periodic=True, device=device, dtype=x.dtype)
    wx = torch.hann_window(W, periodic=True, device=device, dtype=x.dtype)
    w2d = wy[:, None] * wx[None, :]
    return x * w2d


def _gaussian_kernel2d(sigma_px: float, device, dtype) -> torch.Tensor:
    # kernel size ~ 6*sigma rounded to odd
    k = max(3, int(round(6 * sigma_px)))
    if k % 2 == 0:
        k += 1
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="xy")
    ker = torch.exp(-(xx**2 + yy**2) / (2 * sigma_px**2 + 1e-12))
    ker = ker / ker.sum()
    return ker


def _blur2d(x: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Gaussian blur each 2D map with sigma in *pixels*."""
    if sigma_px <= 0:
        return x
    ker = _gaussian_kernel2d(sigma_px, x.device, x.dtype)
    ker = ker.unsqueeze(0).unsqueeze(0)  # 1x1xkxk
    B, H, W = x.shape
    x = x.unsqueeze(1)  # Bx1xHxW
    pad = ker.shape[-1] // 2
    x = F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), ker)
    return x.squeeze(1)


def _radial_bins(shape: Tuple[int, int], pixel_size: float, device, dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct radial frequency grid for isotropic PSD:
    Returns:
      k_mag: HxW tensor of radial wavenumber [cycles per meter]
      k_edges: bin edges (L,) in cycles/m
      k_centers: (L-1,) bin centers
    """
    H, W = shape
    # spatial frequency axes in cycles per meter
    ky = torch.fft.fftfreq(H, d=pixel_size, device=device)
    kx = torch.fft.fftfreq(W, d=pixel_size, device=device)
    KX, KY = torch.meshgrid(kx, ky, indexing="xy")
    k_mag = torch.sqrt(KX**2 + KY**2).to(dtype)
    # bin edges from 0 to Nyquist
    k_max = float(torch.max(k_mag))
    nbins = max(16, int(math.sqrt(H * W)))  # heuristic
    k_edges = torch.linspace(0.0, k_max, steps=nbins + 1, device=device, dtype=dtype)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
    return k_mag, k_edges, k_centers


def _radial_profile(values: torch.Tensor, radii: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """
    Bin-average 'values' over 'radii' using provided bin edges.
    Returns length = len(edges)-1 tensor.
    """
    # digitize
    idx = torch.bucketize(radii.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < edges.numel() - 1)
    idx = idx[valid]
    v = values.ravel()[valid]
    nbins = edges.numel() - 1
    num = torch.zeros(nbins, device=values.device, dtype=values.dtype)
    den = torch.zeros(nbins, device=values.device, dtype=values.dtype)
    num.scatter_add_(0, idx, v)
    den.scatter_add_(0, idx, torch.ones_like(v))
    den = den.clamp_min(1e-12)
    return num / den


def _acf2d(x: torch.Tensor) -> torch.Tensor:
    """
    Zero-mean autocorrelation via Wiener–Khinchin:
      acf = ifft2( |fft2(x)|^2 ), normalized so acf[0,0] = variance * Npix
    Returns real tensor same shape as x.
    """
    X = torch.fft.fft2(x)
    S = (X * torch.conj(X))
    acf = torch.fft.ifft2(S).real
    return acf


def _find_lengthscale_from_acf(acf: torch.Tensor, pixel_size: float) -> float:
    """
    Compute isotropic correlation length as the radius where acf drops to 1/e of acf(0).
    Uses radial profile through center and linear interpolation between bins.
    """
    H, W = acf.shape
    device, dtype = acf.device, acf.dtype
    # shift so center at (0,0) for radial binning convenience
    acf0 = torch.fft.fftshift(acf)
    peak = acf0[H // 2, W // 2].abs().clamp_min(1e-12)
    prof_len = min(H, W)
    # Build radius map in pixels
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    rr = torch.sqrt((yy - H // 2).float()**2 + (xx - W // 2).float()**2)
    # Bin edges: 0..min(H,W)/2 pixels
    rmax = 0.5 * min(H, W)
    bins = torch.linspace(0, rmax, steps=int(rmax) + 1, device=device, dtype=dtype)
    prof = _radial_profile(acf0, rr, bins)
    prof = prof / peak  # normalize so C(0)=1
    target = math.exp(-1.0)
    # locate first index where prof < 1/e
    below = torch.nonzero(prof < target, as_tuple=False)
    if below.numel() == 0:
        return float("nan")
    i = int(below[0, 0].item())
    if i == 0:
        ell_px = bins[1].item()
    else:
        x0, x1 = bins[i - 1].item(), bins[i].item()
        y0, y1 = prof[i - 1].item(), prof[i].item()
        # linear interpolation to y=1/e
        if abs(y1 - y0) < 1e-12:
            ell_px = x1
        else:
            t = (target - y0) / (y1 - y0)
            ell_px = x0 + t * (x1 - x0)
    return float(ell_px * pixel_size)


def _common_bins(a: torch.Tensor, b: torch.Tensor, bins: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    amin = _nanmin(a)
    amax = _nanmax(a)
    bmin = _nanmin(b)
    bmax = _nanmax(b)
    lo = torch.minimum(amin, bmin)
    hi = torch.maximum(amax, bmax)
    if not torch.isfinite(lo):
        lo = torch.tensor(0.0, device=a.device, dtype=a.dtype)
    if not torch.isfinite(hi):
        hi = torch.tensor(1.0, device=a.device, dtype=a.dtype)
    if (hi - lo) < 1e-12:
        hi = lo + 1.0
    edges = torch.linspace(lo.item(), hi.item(), steps=bins + 1, device=a.device, dtype=a.dtype)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _hist_pdf(x: torch.Tensor, edges: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    # x: (B,H,W) values
    if mask is not None:
        mask = _broadcast_mask_like(x, mask).reshape(x.shape)
        x = torch.where(mask, x, torch.nan)
    # flatten ignoring NaNs
    x = x.reshape(-1)
    valid = torch.isfinite(x)
    x = x[valid]
    if x.numel() == 0:
        return torch.full((edges.numel() - 1,), 0.0, device=edges.device, dtype=edges.dtype)
    hist = torch.histc(x, bins=edges.numel() - 1, min=edges[0].item(), max=edges[-1].item())
    pdf = hist / hist.sum().clamp_min(1.0)
    return pdf


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12, base2: bool = True) -> torch.Tensor:
    """
    Jensen–Shannon divergence between discrete PDFs p and q.
    Returns scalar in [0,1] when log base = 2.
    """
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    m = 0.5 * (p + q)
    log = torch.log2 if base2 else torch.log
    dkl_pm = (p * (log(p) - log(m))).sum()
    dkl_qm = (q * (log(q) - log(m))).sum()
    return 0.5 * (dkl_pm + dkl_qm)

def agg(values, weights=None, reduce="mean"):
    """
    values: list[float] per-tile
    weights: list[float] per-tile (e.g., valid pixel counts) or None
    reduce: "mean" | "median" | "weighted"
    """
    import numpy as np
    v = np.array(values, dtype=float)
    if reduce == "median":
        return float(np.nanmedian(v))
    if reduce == "weighted":
        if weights is None:
            return float(np.nanmean(v))
        w = np.array(weights, dtype=float)
        w = np.where(np.isfinite(v), w, 0.0)
        return float(np.nansum(v * w) / max(np.nansum(w), 1e-12))
    return float(np.nanmean(v))

def _nanmin(x: torch.Tensor) -> torch.Tensor:
    v = x[torch.isfinite(x)]
    if v.numel() == 0:
        return torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
    return v.min()

def _nanmax(x: torch.Tensor) -> torch.Tensor:
    v = x[torch.isfinite(x)]
    if v.numel() == 0:
        return torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
    return v.max()

# =============================================================================
# RECONSTRUCTION METRICS
# =============================================================================

@torch.no_grad()
def rmse(gt: torch.Tensor, pred: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Root Mean Square Error (meters).
    Accepts shapes: HxW, NxHxW, NxCxHxW. Returns scalar tensor (mean over batch/maps).
    """
    gt = _ensure_float(gt)
    pred = _ensure_float(pred)
    err = pred - gt
    m = _valid_mask(gt, pred, mask)
    if err.dim() == 2:
        mse = (err[m] ** 2).mean()
        return torch.sqrt(mse)
    # batch-aware mean
    mse = (err**2)[m].mean()
    return torch.sqrt(mse)


@torch.no_grad()
def bias(gt: torch.Tensor, pred: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mean signed error (meters): mean(pred - gt).
    """
    gt = _ensure_float(gt)
    pred = _ensure_float(pred)
    diff = pred - gt
    m = _valid_mask(gt, pred, mask)
    if diff.dim() == 2:
        return diff[m].mean()
    return diff[m].mean()


@torch.no_grad()
def sigma_error(gt: torch.Tensor, pred: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Relative RMS height (std dev) error as a percentage:
      |(std_pred - std_true)/std_true| * 100
    """
    gt = _ensure_float(gt)
    pred = _ensure_float(pred)
    m = _valid_mask(gt, pred, mask)

    def masked_std(x, msk):
        vals = x[msk]
        if vals.numel() == 0:
            return torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
        mu = vals.mean()
        return vals.sub(mu).pow(2).mean().sqrt()

    s_true = masked_std(gt, m)
    s_pred = masked_std(pred, m)
    rel = (s_pred - s_true).abs() / s_true.clamp_min(1e-12)
    return rel * 100.0


@torch.no_grad()
def corr_length_error(
    gt: torch.Tensor,
    pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    pixel_size: float = 1.0,
) -> torch.Tensor:
    """
    Correlation length relative error (%), using isotropic ACF via FFT (Wiener–Khinchin).
    Length scale ℓ is the radius where normalized ACF falls to 1/e.
    Returns |(ℓ_pred - ℓ_true)/ℓ_true| * 100.
    """
    gt = _ensure_float(gt)
    pred = _ensure_float(pred)

    # Reduce to 2D maps (batch over first dim)
    GT = _to_2d(gt)
    PR = _to_2d(pred)
    if mask is not None:
        mask = _broadcast_mask_like(GT, mask)

    ells_true = []
    ells_pred = []
    for i in range(GT.shape[0]):
        g = GT[i]
        p = PR[i]
        if mask is not None:
            m = mask[i].bool()
            g = torch.where(m, g, torch.nan)
            p = torch.where(m, p, torch.nan)
        # fill NaNs with local mean to avoid FFT NaNs (approximate; mask already affects stats)
        def _fillnan(z):
            if torch.isnan(z).any():
                mean = torch.nanmean(z)
                z = torch.where(torch.isfinite(z), z, mean)
            return z - torch.nanmean(z)

        g = _fillnan(g)
        p = _fillnan(p)
        # autocorrelation
        acf_g = _acf2d(g)
        acf_p = _acf2d(p)
        # get ℓ in meters
        ell_g = _find_lengthscale_from_acf(acf_g, pixel_size)
        ell_p = _find_lengthscale_from_acf(acf_p, pixel_size)
        ells_true.append(ell_g)
        ells_pred.append(ell_p)

    ell_true = torch.tensor(ells_true, device=gt.device, dtype=gt.dtype)
    ell_pred = torch.tensor(ells_pred, device=gt.device, dtype=gt.dtype)
    rel = (ell_pred - ell_true).abs() / ell_true.clamp_min(1e-12)
    return torch.nanmean(rel) * 100.0


@torch.no_grad()
def normal_angle_error(
    gt: torch.Tensor,
    pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    pixel_size: float = 1.0,
    use_sobel: bool = True,
    degrees: bool = True,
) -> torch.Tensor:
    """
    Normal Angle Error (NAE): mean arccos( n_pred · n_true ), averaging over valid pixels.
    Normals from gradients: n = (-dz/dx, -dz/dy, 1), with spacing = pixel_size (meters).
    """
    gt = _ensure_float(gt)
    pred = _ensure_float(pred)
    device = gt.device
    dtype = gt.dtype

    def gradients(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if use_sobel:
            # 3x3 Sobel kernels
            kx = torch.tensor([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], device=device, dtype=dtype) / 8.0
            ky = torch.tensor([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]], device=device, dtype=dtype) / 8.0
            kx = kx.view(1, 1, 3, 3)
            ky = ky.view(1, 1, 3, 3)
            z4d = z.unsqueeze(0).unsqueeze(0)  # 1x1xHxW
            gx = F.conv2d(F.pad(z4d, (1, 1, 1, 1), mode="reflect"), kx).squeeze()
            gy = F.conv2d(F.pad(z4d, (1, 1, 1, 1), mode="reflect"), ky).squeeze()
        else:
            # simple central differences
            gy = (F.pad(z, (0, 0, 1, 0))[:-1, :] - F.pad(z, (0, 0, 0, 1))[1:, :]) * 0.5
            gx = (F.pad(z, (1, 0, 0, 0))[:, :-1] - F.pad(z, (0, 1, 0, 0))[:, 1:]) * 0.5
        # convert to slope per meter
        return gx / pixel_size, gy / pixel_size

    GT = _to_2d(gt)
    PR = _to_2d(pred)
    if mask is not None:
        mask = _broadcast_mask_like(GT, mask)

    angles = []
    for i in range(GT.shape[0]):
        g = GT[i]
        p = PR[i]
        if mask is not None:
            m = mask[i].bool()
        else:
            m = torch.isfinite(g) & torch.isfinite(p)

        # compute normals
        gx, gy = gradients(g)
        px, py = gradients(p)
        # n = (-dz/dx, -dz/dy, 1)
        ng = torch.stack((-gx, -gy, torch.ones_like(g)), dim=0)  # 3xHxW
        np_ = torch.stack((-px, -py, torch.ones_like(p)), dim=0)
        # normalize
        ng = ng / (torch.linalg.norm(ng, dim=0, keepdim=False).clamp_min(1e-12))
        np_ = np_ / (torch.linalg.norm(np_, dim=0, keepdim=False).clamp_min(1e-12))
        # dot -> acos
        dot = (ng * np_).sum(dim=0).clamp(-1.0, 1.0)
        ang = torch.acos(dot)
        if degrees:
            ang = ang * (180.0 / math.pi)
        ang = ang[m]
        if ang.numel() > 0:
            angles.append(ang.mean())
    if len(angles) == 0:
        return torch.tensor(float("nan"), device=device, dtype=dtype)
    return torch.stack(angles).mean()


@torch.no_grad()
def average_jsd_multiscale(
    gt: torch.Tensor,
    pred: torch.Tensor,
    scales_m: Iterable[float] = (1.0, 2.0, 5.0, 10.0),
    pixel_size: float = 1.0,
    bins: int = 256,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Average Jensen–Shannon Divergence of residual PDFs across smoothing scales (in meters).
    Steps:
      1) Smooth gt and pred with Gaussian (σ = scale / sqrt(8*ln2) if you want FWHM, but here we treat 'scale' as σ).
         We assume 'scale' passed is σ (meters). If you prefer FWHM, pass scale / 2.355.
      2) Build shared histogram bins per scale across gt & pred.
      3) Compute JSD (base-2) per scale, then average.
    Returns scalar in [0,1].
    """
    gt = _ensure_float(gt)
    pred = _ensure_float(pred)
    GT = _to_2d(gt)
    PR = _to_2d(pred)
    if mask is not None:
        mask = _broadcast_mask_like(GT, mask)

    jsds = []
    for s in scales_m:
        sigma_px = max(0.0, float(s) / float(pixel_size))
        g_blur = _blur2d(GT, sigma_px)
        p_blur = _blur2d(PR, sigma_px)

        # PDFs over all maps combined for stability
        g_all = g_blur
        p_all = p_blur
        if mask is not None:
            m_all = mask
        else:
            m_all = torch.isfinite(g_all) & torch.isfinite(p_all)

        # shared bins
        edges, _ = _common_bins(g_all[m_all], p_all[m_all], bins=bins)
        pg = _hist_pdf(g_all, edges, mask=m_all)
        pp = _hist_pdf(p_all, edges, mask=m_all)
        jsd = _js_divergence(pg, pp, base2=True)
        jsds.append(jsd)

    if len(jsds) == 0:
        return torch.tensor(float("nan"), device=gt.device, dtype=gt.dtype)
    return torch.stack(jsds).mean()


@torch.no_grad()
def log_psd_rmse(
    gt: torch.Tensor,
    pred: torch.Tensor,
    pixel_size: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    window: bool = True,
) -> torch.Tensor:
    """
    log-PSD RMSE over isotropic radial spectra.
      PSD P(k) = |F{z}|^2 / N^2  (we use normalized FFT amplitude; constant factors cancel in log-space)
    Steps:
      - Optional Hann window to reduce leakage
      - 2D FFT -> power
      - Radially average to P(k)
      - Compare log10 spectra on shared k bins
    Returns scalar RMSE in log10 space.
    """
    gt = _ensure_float(gt)
    pred = _ensure_float(pred)

    Zt = _to_2d(gt)
    Zp = _to_2d(pred)
    if mask is not None:
        mask = _broadcast_mask_like(Zt, mask)

    logs_true = []
    logs_pred = []
    # We’ll accumulate spectra then align on common k grid via interpolation
    k_ref = None
    for i in range(Zt.shape[0]):
        a = Zt[i]
        b = Zp[i]
        # basic NaN handling (replace with mean)
        def _fillnan(z):
            if torch.isnan(z).any():
                mean = torch.nanmean(z)
                z = torch.where(torch.isfinite(z), z, mean)
            return z - torch.nanmean(z)
        a = _fillnan(a)
        b = _fillnan(b)

        if window:
            a = _apply_hann2d(a)
            b = _apply_hann2d(b)

        Ha, Wa = a.shape
        device, dtype = a.device, a.dtype

        # FFT & power
        Fa = torch.fft.fft2(a)
        Fb = torch.fft.fft2(b)
        Pa = (Fa.real**2 + Fa.imag**2) / (Ha * Wa)
        Pb = (Fb.real**2 + Fb.imag**2) / (Ha * Wa)

        k_mag, k_edges, k_centers = _radial_bins((Ha, Wa), pixel_size, device, dtype)
        sp_a = _radial_profile(torch.fft.fftshift(Pa), torch.fft.fftshift(k_mag), k_edges)
        sp_b = _radial_profile(torch.fft.fftshift(Pb), torch.fft.fftshift(k_mag), k_edges)

        # keep positive-k bins (exclude DC to avoid log10 issues)
        valid = sp_a > 0
        valid &= sp_b > 0
        valid &= k_centers > 0
        if valid.any():
            if k_ref is None:
                k_ref = k_centers[valid]
            # Interpolate onto k_ref if shapes differ
            def _interp(x_from, k_from, k_to):
                # simple linear interpolation in log-log domain for stability
                x_from = torch.clamp(x_from, min=1e-30)
                logx = torch.log10(x_from)
                logk = torch.log10(torch.clamp(k_from, min=1e-12))
                logk_to = torch.log10(torch.clamp(k_to, min=1e-12))
                # torch interp1d replacement
                order = torch.argsort(logk)
                logk = logk[order]
                logx = logx[order]
                # clamp outside range
                logk_to = torch.clamp(logk_to, min=logk.min(), max=logk.max())
                # manual linear interpolation
                idx = torch.searchsorted(logk, logk_to)
                idx = torch.clamp(idx, 1, logk.numel() - 1)
                x0 = logk[idx - 1]; x1 = logk[idx]
                y0 = logx[idx - 1]; y1 = logx[idx]
                t = (logk_to - x0) / (x1 - x0 + 1e-12)
                return y0 + t * (y1 - y0)  # returns log10 spectrum

            logPa = _interp(sp_a[valid], k_centers[valid], k_ref)
            logPb = _interp(sp_b[valid], k_centers[valid], k_ref)
            logs_true.append(logPa)
            logs_pred.append(logPb)

    if len(logs_true) == 0:
        return torch.tensor(float("nan"), device=gt.device, dtype=gt.dtype)

    LT = torch.stack(logs_true)  # B x K
    LP = torch.stack(logs_pred)  # B x K
    rmse = torch.sqrt(torch.mean((LP - LT) ** 2))
    return rmse
