"""Render Figure 5 and its regional extension from saved analysis.

Edit configs/figure5.yaml style settings and rerun this script without
recomputing distribution fits/variograms. Does not run diffusion inference.
"""
import argparse
import json
import os
import re
from pathlib import Path
import sys
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/roughnet-mpl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np
from scipy import stats
import yaml
from scripts.figure5_analysis import REGIONS, load_manifest, load_result, load_pair, result_path, sha, write_json
from scripts.fig5_statistics import FAMILIES, exp_variogram, family_pdf, agreement, weighted_kappa


def apply_style(style):
    plt.rcParams.update({'font.family': style['font_family'], 'font.size': style['font_size'],
                         'axes.titlesize': style['title_size'], 'legend.fontsize': style['legend_size'],
                         'savefig.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})


def save(fig, name, args, manifest, style, panel_data=None):
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for ext in style['formats']:
        if ext not in ('png', 'pdf', 'svg'): raise ValueError('Formats must be png, pdf or svg')
        path = args.figure_dir / f'{name}.{ext}'
        fig.savefig(path, dpi=style['dpi'], bbox_inches='tight')
        outputs[path.name] = sha(path)
    write_json(args.figure_dir / f'{name}.json', dict(analysis_manifest_sha256=sha(args.out/'manifest.json'),
               plotting_source_sha256=sha(__file__), scope=manifest['scope'], style=style, files=outputs, panel_data=panel_data,
               interpretation='Illustrative centered shape diagnostics; not a replacement for frozen paper metrics'))
    plt.close(fig)


def fmt(value, digits=3):
    return f'{value:.{digits}f}' if value is not None else 'unavailable'


def roughness_confusion(rows, edges):
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or not len(edges) or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0) or edges[0] <= 0:
        raise ValueError('Class edges must be finite, positive and strictly increasing')
    x = np.asarray([r['sigma_gt'] for r in rows]); y = np.asarray([r['sigma_pred'] for r in rows])
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x < 0) or np.any(y < 0):
        raise ValueError('Invalid roughness values')
    gt, pred = np.digitize(x, edges), np.digitize(y, edges)
    n = len(edges)+1; matrix = np.zeros((n,n),int)
    np.add.at(matrix,(gt,pred),1)
    denom = matrix.sum(1,keepdims=True)
    normalized = np.divide(matrix,denom,out=np.zeros_like(matrix,dtype=float),where=denom>0)
    return dict(counts=matrix.tolist(), row_fractions=normalized.tolist(), n=len(rows),
                accuracy=float(np.mean(gt == pred)) if len(rows) else None,
                weighted_kappa=weighted_kappa(gt,pred,n))


def patch_figure(args, manifest, style, set_name):
    region, pid = style['patch_region'], str(style['patch_id'])
    matches = [r for r in manifest['records'] if (r['set'], r['region'], r['pid']) == (set_name, region, pid)]
    if not matches: raise ValueError(f'Patch {region}/{pid} absent from this prepared analysis')
    record = matches[0]; result = load_result(args.out, record, manifest)
    row, details = result['row'], result['details']
    gt, pred, mask = load_pair(record)
    if manifest['options']['mad_k'] is not None:
        med = np.median(gt[mask]); mad = np.median(abs(gt[mask] - med))
        if mad > 0: mask &= abs(gt-med) <= manifest['options']['mad_k'] * mad
    g, p = gt[mask] - gt[mask].mean(), pred[mask] - pred[mask].mean()
    fig = plt.figure(figsize=style['patch_figsize'])
    gs = fig.add_gridspec(3, 2, height_ratios=style['patch_height_ratios'], hspace=style['hspace'], wspace=style['wspace'])
    limit = style['elevation_limit_m']
    norm = (SymLogNorm(linthresh=style['elevation_linthresh_m'], vmin=-limit, vmax=limit)
            if style['elevation_norm'] == 'symlog' else Normalize(-limit, limit))
    for i, (image, title) in enumerate([(gt, 'LiDAR'), (pred, 'RoughNet prediction' if set_name == 'corrected' else set_name)]):
        ax = fig.add_subplot(gs[0, i]); centered = image - image[mask].mean()
        im = ax.imshow(np.rot90(np.ma.masked_where(~mask, centered), style['rotate_quarters']),
                       cmap=style['colormap'], norm=norm, interpolation='nearest')
        ax.set_axis_off()
        if style['show_titles']: ax.set_title(style.get('patch_map_titles', [title, title])[i])
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label='Centered residual elevation (m)')
    x = np.linspace(min(g.min(), p.min()), max(g.max(), p.max()), 500)
    for i, (data, key, color, label) in enumerate([(g, 'gt', style['gt_color'], 'LiDAR'), (p, 'pred', style['prediction_color'], 'Prediction')]):
        ax = fig.add_subplot(gs[1, i]); fits = details[f'fits_{key}']; best = row[f'best_{key}']
        ax.hist(data, bins=style['histogram_bins'], density=True, histtype='step', color=color, lw=style['line_width'], label='Empirical density')
        for family, fitted in fits.items():
            if fitted['params'] is None: continue
            ax.plot(x, family_pdf(family, fitted['params'], x, details[f'shift_{key}']),
                    color=style['family_colors'][family],
                    lw=style['best_fit_line_width'] if family == best else style['line_width'],
                    ls='-' if family == best else '--', label=family)
        ax.set(xlabel='Centered residual elevation (m)', ylabel='Density (m⁻¹)', yscale=style['density_yscale'], ylim=style['density_ylim'])
        if style['show_titles']: ax.set_title(style.get('patch_density_titles', ['LiDAR distribution', 'Prediction distribution'])[i])
        if style['show_legend']: ax.legend()
        ax.grid(alpha=style['grid_alpha'])
    ax = fig.add_subplot(gs[2, 0]); options = manifest['options']
    centers = np.arange(options['lag_width_m']/2, options['max_lag_m'], options['lag_width_m'])
    h = np.linspace(0, options['max_lag_m'], 200)
    with np.load(result_path(args.out, record).with_suffix('.npz')) as z:
        for key, label, color in [('gt', 'LiDAR', style['gt_color']), ('pred', 'Prediction', style['prediction_color'])]:
            ax.scatter(centers, z[f'sv_{key}'], s=style['scatter_size'], color=color, alpha=.5)
            fit = details[f'variogram_{key}']
            if fit['params'] is not None:
                ax.plot(h, exp_variogram(h, *fit['params']), color=color, lw=style['line_width'],
                        ls='-' if key == 'gt' else '--',
                        label=f"{label}: {fmt(fit['effective_range_m'], 1)} m ({fit['status']})")
        ax.set(xlabel='Lag distance (m)', ylabel='Semivariance (m²)')
        if style['show_titles']: ax.set_title(style.get('variogram_title', 'Empirical points and exponential fits'))
        if style['show_legend']: ax.legend(fontsize=style['legend_size'])
        ax.grid(alpha=style['grid_alpha'])
    ax = fig.add_subplot(gs[2, 1]); ax.axis('off')
    text = [f"PATCH {pid} ({REGIONS[region]}, model: {set_name})", '━' * 40, '',
            f"► INPUT       σ {row['sigma_gt']:.4f} m | range {fmt(row['range_gt_m'],1)} m | best {row['best_gt']}",
            f"► PREDICTION  σ {row['sigma_pred']:.4f} m | range {fmt(row['range_pred_m'],1)} m | best {row['best_pred']}",
            f"► GT–Pred JS divergence (base 2): {row['jsd_centered_1m']:.4f}", '',
            '► FIT JS DISTANCE      Input  |  Pred']
    for family in FAMILIES:
        a = details['fits_gt'].get(family, {}).get('js_distance_ln')
        b = details['fits_pred'].get(family, {}).get('js_distance_ln')
        text.append(f'  • {family:<11}:  {fmt(a,4)} | {fmt(b,4)}')
    ax.text(.02, .5, '\n'.join(text), transform=ax.transAxes, va='center',
            family=style.get('metrics_box_font_family','monospace'),
            fontsize=style.get('metrics_box_font_size',10),
            bbox=dict(boxstyle=f"round,pad={style.get('metrics_box_padding',1.0)}",
                      facecolor=style.get('metrics_box_facecolor','whitesmoke'),
                      edgecolor=style.get('metrics_box_edgecolor','gray')))
    save(fig, f'figure5_patch{pid}_{set_name}', args, manifest, style)


def extended_figure(args, manifest, style, set_name):
    results = [load_result(args.out, r, manifest) for r in manifest['records'] if r['set'] == set_name]
    if not results: raise ValueError(f'Set not found: {set_name}')
    panel_f = style.get('extended_panel_f','roughness_classes')
    if panel_f not in ('roughness_classes','acf'): raise ValueError('Unknown extended_panel_f')
    panel_data = None
    regions = [r for r in REGIONS if any(x['row']['region'] == r for x in results)]
    roughness_fig, roughness_axes = plt.subplots(1, 3, figsize=style.get('extended_roughness_figsize', [18, 5.5]),
                                                gridspec_kw={'width_ratios':style.get('roughness_width_ratios',[1.15,1.15,1])})
    structure_fig, structure_axes = plt.subplots(1, 3, figsize=style.get('extended_structure_figsize', [18, 5.5]))
    # Keep notebook panel indices internally; each output gets independent a/b/c labels.
    axes = [roughness_axes[0], structure_axes[0], roughness_axes[1],
            structure_axes[1], structure_axes[2], roughness_axes[2]]
    groups = {r: [x for x in results if x['row']['region'] == r] for r in regions}
    maximum_sigma = max(x['row'][k] for x in results for k in ('sigma_gt', 'sigma_pred')) * 100
    for region in regions:
        selected = groups[region]; rows = [r['row'] for r in selected]
        color, marker = style['region_colors'][region], style['region_markers'][region]
        scatter = dict(s=style['scatter_size'], alpha=style['scatter_alpha'], color=color, marker=marker, linewidths=0)
        x = [r['sigma_gt']*100 for r in rows]; y = [r['sigma_pred']*100 for r in rows]
        sigma_agreement = agreement(x,y)
        axes[0].scatter(x, y, label=f"{REGIONS[region]} (n={len(rows)}, CCC={fmt(sigma_agreement['ccc'],2)}, ρ={fmt(sigma_agreement['spearman'],2)})", **scatter)
        valid = [r for r in rows if r['range_gt_m'] is not None and r['range_pred_m'] is not None]
        range_x, range_y = [r['range_gt_m'] for r in valid], [r['range_pred_m'] for r in valid]
        axes[1].scatter(range_x, range_y, label=f"{REGIONS[region]} (ρ={fmt(agreement(range_x,range_y)['spearman'],2)})", **scatter)
        edges = np.linspace(*manifest['options']['histogram_range_m'], manifest['options']['histogram_bins']+1)
        centers = (edges[:-1] + edges[1:])/2
        pooled = {k: np.zeros(len(centers)) for k in ('gt', 'pred')}
        for item in selected:
            with np.load(result_path(args.out, item['input']).with_suffix('.npz')) as z:
                for k in pooled: pooled[k] += z[f'hist_{k}']
        for k, hist in pooled.items():
            density = hist / (hist.sum() * np.diff(edges)) if hist.sum() else np.full_like(hist, np.nan)
            axes[2].plot(centers, np.where(density > 0, density, np.nan), color=color,
                         lw=style['line_width'], ls='-' if k == 'gt' else '--', label=f"{REGIONS[region]} {'LiDAR' if k == 'gt' else 'prediction'}")
        for k, family in enumerate(FAMILIES):
            j = regions.index(region); yy = k + (j-(len(regions)-1)/2)*.22
            fractions = [sum(r[f'best_{side}'] == family for r in rows)/len(rows) for side in ('gt', 'pred')]
            axes[4].plot(fractions, [yy, yy], color=color, lw=style['line_width'])
            axes[4].scatter([fractions[0]], [yy], facecolors='white', edgecolors=color, marker=marker)
            axes[4].scatter([fractions[1]], [yy], color=color, marker=marker, label=REGIONS[region] if k == 0 else None)
        acf_rows = [r for r in rows if r['acf_gt_m'] is not None and r['acf_pred_m'] is not None]
        if panel_f == 'acf':
            axes[5].scatter([r['acf_gt_m'] for r in acf_rows], [r['acf_pred_m'] for r in acf_rows], label=REGIONS[region], **scatter)
    positive = [x['row'][k]*100 for x in results for k in ('sigma_gt', 'sigma_pred') if x['row'][k] > 0]
    lo, hi = style.get('sigma_limits_cm') or (max(1e-4, min(positive)*.8) if positive else .01, max(1, maximum_sigma*1.1))
    axes[0].plot([lo, hi], [lo, hi], 'k--', lw=1)
    axes[0].set(xscale='log', yscale='log', xlim=(lo,hi), ylim=(lo,hi), xlabel='LiDAR σ (cm)', ylabel='Predicted σ (cm)')
    lag, limit = manifest['options']['max_lag_m'], manifest['options']['max_range_m']
    axes[1].axvspan(lag, limit+10, color='#ddd', alpha=.5, lw=0)
    axes[1].axhspan(lag, limit+10, color='#ddd', alpha=.5, lw=0)
    axes[1].plot([0,limit], [0,limit], 'k--', lw=1)
    axes[1].set(xlim=(0,limit+5), ylim=(0,limit+5), xlabel='LiDAR effective range (m)', ylabel='Predicted effective range (m)')
    axes[1].text(.98,.02,'Shading: beyond observed lag',transform=axes[1].transAxes,ha='right',fontsize=style['legend_size'])
    axes[2].set(xlim=style['pooled_xlim'], ylim=style['pooled_ylim'], yscale='log', xlabel='Centered residual elevation (m)', ylabel='Pooled density (m⁻¹)')
    data = [[r['row']['jsd_centered_1m'] for r in groups[region]] for region in regions]
    boxes = axes[3].boxplot(data, widths=.5, showfliers=False, patch_artist=True, medianprops=dict(color='black'))
    for i, region in enumerate(regions):
        boxes['boxes'][i].set(facecolor=style['region_colors'][region], alpha=.35)
        axes[3].scatter(np.random.default_rng(0).normal(i+1,.06,len(data[i])), data[i], s=4, alpha=.25, linewidths=0, color=style['region_colors'][region])
        example = [r['row'] for r in groups[region] if r['row']['pid'] == str(style['patch_id']) and region == style['patch_region']]
        if example:
            value = example[0]['jsd_centered_1m']
            percentile = stats.percentileofscore(data[i], value)
            axes[3].scatter(i+1,value,marker='*',color='black',s=260,zorder=5,
                            label=f"Fig. 5 patch {style['patch_id']}\n({percentile:.0f}th pct.)")
    axes[3].set_xticks(range(1,len(regions)+1), [REGIONS[r] for r in regions])
    axes[3].set(yscale=style.get('jsd_yscale','log'), ylabel='Centered 1 m JS divergence (base 2)')
    if style['show_legend'] and axes[3].get_legend_handles_labels()[0]:
        axes[3].legend(loc=style.get('patch_legend_location','upper right'),
                       fontsize=style['legend_size'],frameon=False)
    if style.get('jsd_ylim') is not None: axes[3].set_ylim(style['jsd_ylim'])
    axes[4].scatter([], [], facecolors='white', edgecolors='#555', s=50, label='open = LiDAR')
    axes[4].scatter([], [], color='#555', s=50, label='filled = prediction')
    axes[4].set_yticks(range(len(FAMILIES)), FAMILIES)
    axes[4].set(xlim=(-.02,1.02), xlabel='Fraction of patches (open: LiDAR; filled: prediction)')
    if panel_f == 'roughness_classes':
        edges = style.get('sigma_class_edges_m',manifest['options']['sigma_class_edges_m'])
        panel_data = dict(sigma_class_edges_m=edges, pooled=roughness_confusion([r['row'] for r in results],edges),
                          regions={region:roughness_confusion([r['row'] for r in groups[region]],edges) for region in regions})
        cm = np.asarray(panel_data['pooled']['counts']); normalized = np.asarray(panel_data['pooled']['row_fractions']); n=len(cm)
        axes[5].imshow(normalized,cmap=style.get('confusion_colormap','Blues'),vmin=0,vmax=1,aspect='auto')
        for i in range(n):
            for j in range(n):
                label=f'{normalized[i,j]:.0%}\n(n={cm[i,j]})' if cm[i].sum() else '—\n(n=0)'
                axes[5].text(j,i,label,ha='center',va='center',color='white' if normalized[i,j]>.6 else 'black',fontsize=style['legend_size'])
        labels=style.get('sigma_class_labels') or [f'Class {i+1}' for i in range(n)]
        if len(labels)!=n: raise ValueError('Class labels must match number of bins')
        axes[5].set_xticks(range(n),labels,rotation=25,ha='right');axes[5].set_yticks(range(n),labels)
        axes[5].set(xlabel='Predicted roughness class',ylabel='LiDAR roughness class')
    else:
        maximum = manifest['options']['pixel_size_m']*128
        axes[5].plot([0,maximum],[0,maximum],'k--',lw=1)
        axes[5].set(xlabel='LiDAR ACF e-folding length (m)',ylabel='Predicted ACF e-folding length (m)')
    titles = style.get('extended_titles') or ['(a) Patch roughness amplitude', '(b) Variogram effective range', '(c) Pooled regional densities',
              '(d) Distribution mismatch', '(e) Best candidate family', '(f) Roughness-class agreement']
    if panel_f=='acf': titles = [*titles[:5], '(f) ACF spatial scale']
    letters = {0:'a', 2:'b', 5:'c', 1:'a', 3:'b', 4:'c'}
    for i, ax in enumerate(axes):
        if style['show_titles']: ax.set_title(f"({letters[i]}) " + re.sub(r'^\([a-f]\)\s*', '', titles[i]), loc='left')
        ax.grid(alpha=style['grid_alpha'])
        if i==5 and panel_f=='roughness_classes': ax.grid(False)
        if style['show_legend'] and i in (0,1,2,4):
            ax.legend(fontsize=style['legend_size'], frameon=False, markerscale=2 if i in (0,1) else 1,
                      loc='upper left' if i in (0,1) else 'best')
    for fig in (roughness_fig, structure_fig):
        fig.tight_layout(rect=(0,0,1,.97), w_pad=style.get('extended_wpad',2.5))
    roughness_fig.align_xlabels(roughness_axes)
    save(roughness_fig, f'figure5_extended_roughness_{set_name}', args, manifest, style,
         panel_data=dict(original_panels=['a','c','f'], confusion=panel_data))
    save(structure_fig, f'figure5_extended_structure_{set_name}', args, manifest, style,
         panel_data=dict(original_panels=['b','d','e'], range_pairs={r:sum(x['row']['range_gt_m'] is not None and x['row']['range_pred_m'] is not None for x in groups[r]) for r in regions}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'reconstructions/fig5_extended_clean')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/figure5.yaml')
    parser.add_argument('--figure-dir',type=Path,default=ROOT/'figures/final_figures/figure5')
    parser.add_argument('--kind',choices=('patch','extended','all'),default='all')
    parser.add_argument('--set',dest='set_name')
    args=parser.parse_args(); style=yaml.safe_load(args.config.read_text())['style']; manifest=load_manifest(args.out)
    set_name=args.set_name or style['main_set']
    with plt.rc_context():
        apply_style(style)
        if args.kind in ('patch','all'): patch_figure(args,manifest,style,set_name)
        if args.kind in ('extended','all'): extended_figure(args,manifest,style,set_name)
    print(f'Figures and provenance written to {args.figure_dir}')


if __name__ == '__main__': main()
