"""Figure 5 statistics adapted from Fig5_Extended_RoughNet (1).ipynb.

These centered distribution/variogram diagnostics do not replace the paper's
frozen eight-metric evaluator. No plotting or inference occurs on import.
"""
import warnings
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, jensenshannon

FAMILIES = ('Normal', 'Skew', 'Gamma', 'Log-normal')


def exp_variogram(h, nugget, sill, eff_range):
    return nugget + sill * (1 - np.exp(-3 * h / eff_range))


def fit_variogram(centers, empirical, max_range):
    ok = np.isfinite(empirical)
    failed = dict(params=None, effective_range_m=None, status='insufficient_bins', beyond_lag=False)
    if ok.sum() < 4:
        return failed
    values = np.asarray(empirical)[ok]
    if np.max(values) <= 0:
        return dict(failed, status='constant_field')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', category=RuntimeWarning)
            params, _ = curve_fit(exp_variogram, np.asarray(centers)[ok], values,
                                  p0=[max(0, values.min()), max(np.ptp(values), 1e-12), 50],
                                  bounds=([0, 0, 1e-6], [np.inf, np.inf, max_range]), maxfev=10000)
        if not np.isfinite(params).all():
            raise ValueError('Non-finite variogram parameters')
        status = 'upper_bound' if params[2] >= max_range * .999 else 'ok'
        lag_limit = centers[-1] + .5 * (centers[1] - centers[0])
        return dict(params=params.tolist(), effective_range_m=float(params[2]), status=status,
                    beyond_lag=bool(params[2] > lag_limit))
    except (ValueError, RuntimeError, RuntimeWarning, FloatingPointError) as exc:
        return dict(failed, status='fit_failed', error=str(exc))


def compute_variogram(image, mask, options):
    y, x = np.where(mask)
    idx = np.random.RandomState(options['seed']).choice(
        len(y), min(options['variogram_samples'], len(y)), replace=False)
    distances = pdist(np.column_stack((x[idx], y[idx])) * options['pixel_size_m'])
    squared = .5 * pdist(image[mask][idx, None], metric='sqeuclidean')
    edges = np.arange(0, options['max_lag_m'] + options['lag_width_m'] * .5, options['lag_width_m'])
    bins = np.digitize(distances, edges) - 1
    valid = (bins >= 0) & (bins < len(edges) - 1)
    counts = np.bincount(bins[valid], minlength=len(edges) - 1)
    sums = np.bincount(bins[valid], weights=squared[valid], minlength=len(edges) - 1)
    empirical = np.divide(sums, counts, out=np.full(len(counts), np.nan), where=counts > 0)
    centers = (edges[:-1] + edges[1:]) / 2
    return empirical, counts, fit_variogram(centers, empirical, options['max_range_m'])


def acf_efold_length(z, mask, pixel_size_m=1., max_lag=128):
    z = np.where(mask, z - z[mask].mean(), 0.)
    shape = tuple(2 * v for v in z.shape)
    num = np.fft.irfft2(np.abs(np.fft.rfft2(z, s=shape)) ** 2, s=shape)
    den = np.fft.irfft2(np.abs(np.fft.rfft2(mask.astype(float), s=shape)) ** 2, s=shape)
    acf = np.divide(num, den, out=np.full(shape, np.nan), where=den > .05 * den[0, 0])
    if not np.isfinite(acf[0, 0]) or acf[0, 0] <= 0:
        return None
    acf /= acf[0, 0]
    yy = np.fft.fftfreq(shape[0]) * shape[0]
    xx = np.fft.fftfreq(shape[1]) * shape[1]
    radial = np.hypot(yy[:, None], xx[None, :]).round().astype(int)
    sel = (radial <= max_lag) & np.isfinite(acf)
    counts = np.bincount(radial[sel], minlength=max_lag + 1)
    prof = np.divide(np.bincount(radial[sel], weights=acf[sel], minlength=max_lag + 1),
                     counts, out=np.full(max_lag + 1, np.nan), where=counts > 0)
    below = np.where(np.isfinite(prof) & (prof < 1 / np.e))[0]
    if not len(below):
        return None
    j = below[0]
    if j == 0 or not np.isfinite(prof[j - 1]):
        return float(j * pixel_size_m)
    return float((j - 1 + (prof[j - 1] - 1/np.e) / (prof[j - 1] - prof[j])) * pixel_size_m)


def family_pdf(name, params, x, shift):
    family = {'Normal': stats.norm, 'Skew': stats.skewnorm, 'Gamma': stats.gamma, 'Log-normal': stats.lognorm}[name]
    return family.pdf(x + shift if name in ('Gamma', 'Log-normal') else x, *params)


def fit_families(data, n_fit=5000, seed=42):
    shift = float(abs(np.min(data)) + .01)
    fits = {}
    if np.std(data) <= 1e-12:
        return fits, None, shift
    rng = np.random.RandomState(seed)
    sample = data[rng.choice(len(data), n_fit, replace=False)] if len(data) > n_fit else data
    hist, edges = np.histogram(data, bins=100)
    centers = (edges[:-1] + edges[1:]) / 2
    empirical = hist / hist.sum()
    for name, family in zip(FAMILIES, (stats.norm, stats.skewnorm, stats.gamma, stats.lognorm)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                params = family.fit(sample + shift if name in ('Gamma', 'Log-normal') else sample)
                values = family_pdf(name, params, centers, shift) + 1e-10
                distance = float(jensenshannon(empirical, values / values.sum()))
            if not np.isfinite(distance) or not np.isfinite(params).all():
                raise ValueError('Non-finite distribution fit')
            fits[name] = dict(params=list(map(float, params)), js_distance_ln=distance)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            fits[name] = dict(params=None, js_distance_ln=None, error=str(exc))
    valid = {k: v for k, v in fits.items() if v['js_distance_ln'] is not None}
    best = min(valid, key=lambda k: valid[k]['js_distance_ln']) if valid else None
    return fits, best, shift


def jsd_base2(a, b, bins=256):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    edges = np.linspace(lo, hi if hi > lo else lo + 1, bins + 1)
    p = np.histogram(a, edges)[0].astype(float) + 1e-12
    q = np.histogram(b, edges)[0].astype(float) + 1e-12
    p /= p.sum(); q /= q.sum(); mid = .5 * (p + q)
    return float(.5 * (np.sum(p * np.log2(p / mid)) + np.sum(q * np.log2(q / mid))))


def analyse_arrays(gt, pred, mask, options):
    if gt.shape != pred.shape or gt.shape != mask.shape or gt.ndim != 2:
        raise ValueError('GT, prediction and mask must be matching 2D arrays')
    mask = mask.astype(bool).copy()
    if not np.isfinite(gt[mask]).all() or not np.isfinite(pred).all():
        raise ValueError('Non-finite input; do not silently reduce valid scoring support')
    original_n = int(mask.sum())
    if options['mad_k'] is not None and original_n:
        med = np.median(gt[mask]); mad = np.median(abs(gt[mask] - med))
        if mad > 0:
            mask &= abs(gt - med) <= options['mad_k'] * mad
    if mask.sum() < options['min_valid']:
        raise ValueError(f'Only {mask.sum()} valid pixels; minimum is {options["min_valid"]}')
    raw_g, raw_p = gt[mask], pred[mask]
    g, p = raw_g - raw_g.mean(), raw_p - raw_p.mean()
    fits_g, best_g, shift_g = fit_families(g, options['fit_samples'], options['seed'])
    fits_p, best_p, shift_p = fit_families(p, options['fit_samples'], options['seed'])
    sv_g, counts, fit_g = compute_variogram(gt, mask, options)
    sv_p, _, fit_p = compute_variogram(pred, mask, options)
    row = dict(n_valid_original=original_n, n_valid=int(mask.sum()),
               mean_gt=float(raw_g.mean()), mean_pred=float(raw_p.mean()),
               sigma_gt=float(g.std()), sigma_pred=float(p.std()),
               rmse_residual_m=float(np.sqrt(np.mean((raw_g - raw_p)**2))),
               rmse_centered_m=float(np.sqrt(np.mean((g - p)**2))),
               jsd_centered_1m=jsd_base2(g, p),
               best_gt=best_g, best_pred=best_p)
    for key, data, fit, image in [('gt', g, fit_g, gt), ('pred', p, fit_p, pred)]:
        row.update({f'range_{key}_m': fit['effective_range_m'], f'range_{key}_status': fit['status'],
                    f'range_{key}_beyond_lag': fit['beyond_lag'],
                    f'acf_{key}_m': acf_efold_length(image, mask, options['pixel_size_m'])})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            row[f'skew_{key}'] = float(stats.skew(data))
            row[f'kurtosis_excess_{key}'] = float(stats.kurtosis(data))
        if options['exploratory_proxies']:
            row[f'ridge_fraction_{key}'] = float((data > options['ridge_height_m']).mean())
            row[f'sigma_class_{key}'] = int(np.digitize(data.std(), options['sigma_class_edges_m']))
    # Preserve the notebook's bounded Wasserstein subsample for large fields.
    idx = np.random.RandomState(1).choice(len(g), min(20000, len(g)), replace=False)
    row['wasserstein_centered_m'] = float(stats.wasserstein_distance(g[idx], p[idx]))
    edges = np.linspace(*options['histogram_range_m'], options['histogram_bins'] + 1)
    details = dict(fits_gt=fits_g, fits_pred=fits_p, shift_gt=shift_g, shift_pred=shift_p,
                   variogram_gt=fit_g, variogram_pred=fit_p)
    arrays = dict(sv_gt=sv_g, sv_pred=sv_p, pair_counts=counts,
                  hist_gt=np.histogram(g, edges)[0], hist_pred=np.histogram(p, edges)[0])
    row['histogram_outside_fraction_gt'] = float(1 - arrays['hist_gt'].sum() / len(g))
    row['histogram_outside_fraction_pred'] = float(1 - arrays['hist_pred'].sum() / len(p))
    return row, details, arrays


def agreement(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 2:
        return dict(n=len(pairs), ccc=None, spearman=None)
    a, b = np.asarray(pairs).T
    denominator = a.var() + b.var() + (a.mean() - b.mean())**2
    rho = stats.spearmanr(a, b).statistic if a.std() > 0 and b.std() > 0 else np.nan
    return dict(n=len(pairs), ccc=float(2*np.mean((a-a.mean())*(b-b.mean()))/denominator) if denominator else None,
                spearman=float(rho) if np.isfinite(rho) else None)


def weighted_kappa(x, y, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=float)
    for a, b in zip(x, y): matrix[a, b] += 1
    if not matrix.sum(): return None
    expected = np.outer(matrix.sum(1), matrix.sum(0)) / matrix.sum()
    weights = abs(np.arange(n_classes)[:, None] - np.arange(n_classes)[None, :])
    denominator = np.sum(weights * expected)
    return float(1 - np.sum(weights * matrix)/denominator) if denominator else None
