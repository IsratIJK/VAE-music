"""
evaluation.py
-------------
All evaluation, analysis, and reporting functions.
"""

import os
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F

from vae import DEVICE, SEED, LATENT_DIM, AUDIO_FEAT_DIM
from clustering import (MODEL_LABELS, COLORS_M, Z_KEYS_ALL,
                        compute_metrics, _nan_metrics)
from dataset import OUTPUT_DIR

warnings.filterwarnings('ignore')

ALGOS = ['KMeans', 'Agglomerative_Ward', 'Agglomerative_Complete', 'DBSCAN']


# Metrics Heatmap Dashboard (6 Metrics)

def build_metrics_df(all_results):
    """Build the full metrics DataFrame from all_results. Returns df_all."""
    rows_list = []
    for ds_key, res in all_results.items():
        if res is None: continue
        for zkey, zlab in MODEL_LABELS.items():
            if zkey not in res['cl']: continue
            for algo in ALGOS:
                if algo not in res['cl'][zkey]: continue
                m = res['cl'][zkey][algo]['metrics']
                rows_list.append({
                    'Dataset': ds_key, 'Features': zlab, 'Algorithm': algo,
                    'Silhouette': round(m['sil'],    4) if not np.isnan(m['sil'])    else np.nan,
                    'Davies-Bouldin': round(m['db'],     4) if not np.isnan(m['db'])     else np.nan,
                    'Calinski-H': round(m['ch'],     1) if not np.isnan(m['ch'])     else np.nan,
                    'ARI': round(m['ari'],    4) if not np.isnan(m['ari'])    else np.nan,
                    'NMI': round(m['nmi'],    4) if not np.isnan(m['nmi'])    else np.nan,
                    'Purity': round(m['purity'], 4) if not np.isnan(m['purity']) else np.nan,
                })
    return pd.DataFrame(rows_list)


def print_metrics_table(df_all, out_dir=OUTPUT_DIR):
    pd.set_option('display.max_rows', 200); pd.set_option('display.width', 200)
    print('=' * 110)
    print('FULL METRICS TABLE  |  Sil  DB  CH  ARI  NMI  Purity')
    print('=' * 110)
    print(df_all.to_string(index=False))
    df_all.to_csv(f'{out_dir}/full_metrics.csv', index=False)
    print('\nSaved: full_metrics.csv')


def plot_metrics_heatmap(df_all, out_dir=OUTPUT_DIR):
    feat_order  = list(MODEL_LABELS.values())
    metrics_cfg = [
        ('Silhouette', 'higher is better', 'Blues'),
        ('Davies-Bouldin', 'lower is better',  'Reds_r'),
        ('Calinski-H', 'higher is better', 'Greens'),
        ('ARI', 'higher is better', 'Purples'),
        ('NMI', 'higher is better', 'Oranges'),
        ('Purity', 'higher is better', 'YlOrBr'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(28, 16), squeeze=False)
    fig.suptitle('Clustering Quality Heatmap - All 6 Metrics\nRows=Dataset+Algorithm | Cols=Feature Space',
                 fontsize=14, fontweight='bold')
    for ax, (metric, note, cmap) in zip(axes.flat, metrics_cfg):
        if metric not in df_all.columns: ax.set_visible(False); continue
        pivot = df_all.pivot_table(index=['Dataset', 'Algorithm'], columns='Features',
                                   values=metric, aggfunc='mean')
        cols = [c for c in feat_order if c in pivot.columns]
        sns.heatmap(pivot[cols].astype(float), ax=ax, annot=True, fmt='.3f',
                    cmap=cmap, linewidths=0.4, linecolor='white', cbar_kws={'shrink': 0.8})
        ax.set_title(f'{metric}  ({note})', fontweight='bold', fontsize=11)
        ax.set_xlabel('Feature Space')
        ax.tick_params(axis='x', rotation=20, labelsize=8)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/metrics_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show(); print('Saved: metrics_heatmap.png')


# Best Metrics Bar Charts

def plot_best_metrics_bar(all_results, out_dir=OUTPUT_DIR):
    feat_cols = list(MODEL_LABELS.values())
    datasets = [k for k, v in all_results.items() if v is not None]
    x = np.arange(len(datasets))
    w = 0.07

    summary = []
    for ds_key in datasets:
        res = all_results[ds_key]
        for zkey, zlab in MODEL_LABELS.items():
            if zkey not in res['cl']: continue
            def _b(m, fn):
                v = [res['cl'][zkey][a]['metrics'][m]
                     for a in ['KMeans', 'Agglomerative_Ward']
                     if a in res['cl'][zkey] and not np.isnan(res['cl'][zkey][a]['metrics'][m])]
                return fn(v) if v else np.nan
            summary.append({'Dataset': ds_key, 'Features': zlab,
                             'Best Sil': _b('sil', max), 'Best NMI': _b('nmi', max),
                             'Best ARI': _b('ari', max), 'Best Purity': _b('purity', max)})
    df_sum = pd.DataFrame(summary)

    fig, axes = plt.subplots(1, 4, figsize=(32, 7))
    fig.suptitle('Best Score per Feature Space (K-Means & Agglom-Ward)', fontsize=13, fontweight='bold')
    for ax, (metric, ylabel) in zip(axes, [
        ('Best Sil', 'Silhouette (higher is better)'),
        ('Best NMI', 'NMI (higher is better)'),
        ('Best ARI', 'ARI (higher is better)'),
        ('Best Purity', 'Purity (higher is better)'),
    ]):
        for fi, feat in enumerate(feat_cols):
            subset = df_sum[df_sum.Features == feat]
            vals = [subset[subset.Dataset == d][metric].values[0]
                      if len(subset[subset.Dataset == d]) > 0 else np.nan
                      for d in datasets]
            offset = fi * w - (len(feat_cols) / 2) * w
            bars = ax.bar(x + offset, vals, w, label=feat,
                            color=COLORS_M.get(feat, '#888888'), alpha=0.88)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                            f'{v:.3f}', ha='center', va='bottom', fontsize=5, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=11)
        ax.set_ylabel(ylabel); ax.set_title(ylabel, fontweight='bold')
        ax.legend(fontsize=7, ncol=2); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/best_metrics_bar.png', dpi=150, bbox_inches='tight')
    plt.show(); print('Saved: best_metrics_bar.png')


# VAE vs PCA Baseline

def plot_vae_vs_baseline(all_results, out_dir=OUTPUT_DIR):
    vae_keys_delta = [(k, v) for k, v in MODEL_LABELS.items() if k not in ('pca', 'raw')]
    valid = [(k, v) for k, v in all_results.items() if v is not None]
    datasets = [k for k, _ in valid]
    x = np.arange(len(datasets))
    w = 0.07

    fig, axes = plt.subplots(1, 2, figsize=(26, 6), squeeze=False)
    fig.suptitle('VAE vs PCA Baseline - delta Silhouette & delta NMI (K-Means)\n'
                 'Positive = better than PCA | Negative = worse',
                 fontsize=13, fontweight='bold')

    for ax, metric in zip(axes[0], ['sil', 'nmi']):
        ylabel = f'delta {metric.upper()} (model - PCA)'
        for fi, (zkey, zlabel) in enumerate(vae_keys_delta):
            deltas = []
            for _, res in valid:
                vae_m = res['cl'].get(zkey, {}).get('KMeans', {}).get('metrics', {})
                pca_m = res['cl'].get('pca', {}).get('KMeans', {}).get('metrics', {})
                v_val = vae_m.get(metric, np.nan)
                p_val = pca_m.get(metric, np.nan)
                deltas.append(v_val - p_val if not (np.isnan(v_val) or np.isnan(p_val)) else np.nan)
            offset = fi * w - (len(vae_keys_delta) / 2) * w
            bars = ax.bar(x + offset, deltas, w, label=zlabel,
                            color=COLORS_M.get(zlabel, '#888888'), alpha=0.85)
            for bar, v in zip(bars, deltas):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + (0.002 if v >= 0 else -0.012),
                            f'{v:+.3f}', ha='center', va='bottom', fontsize=5, fontweight='bold')
        ax.axhline(0, color='black', linewidth=1, linestyle='--', label='PCA baseline')
        ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=11)
        ax.set_ylabel(ylabel); ax.set_title(ylabel, fontweight='bold')
        ax.legend(fontsize=7, ncol=2); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/vae_vs_baseline.png', dpi=150, bbox_inches='tight')
    plt.show(); print('Saved: vae_vs_baseline.png')


# Beta-VAE Disentanglement Analysis

def plot_disentanglement(all_results, out_dir=OUTPUT_DIR):
    for key, res in all_results.items():
        if res is None: continue
        Z_mlp = res['Z']['mlp']
        Z_beta = res['Z']['beta']
        Z_cvae = res['Z']['cvae']
        y_true = res['y_true']
        n_cl = res['n_class']
        PAL = plt.colormaps['tab20'].resampled(n_cl)
        n_show = min(5, LATENT_DIM)
        best_beta = res.get('best_beta', 4.0)

        models_to_show = [
            (Z_mlp, 'MLP-VAE (baseline)'),
            (Z_beta, f'Beta-VAE (beta={best_beta:.1f}, best)'),
            (Z_cvae, 'CVAE (zero-condition)'),
        ]

        fig, axes = plt.subplots(len(models_to_show), n_show,
                                 figsize=(n_show * 3.5, 4 * len(models_to_show)),
                                 squeeze=False)
        fig.suptitle(f'Latent Dimension Distributions - Disentanglement Analysis\n{res["name"]}',
                     fontsize=13, fontweight='bold')

        for row, (Z, title) in enumerate(models_to_show):
            for di in range(n_show):
                ax = axes[row, di]
                for gi in range(n_cl):
                    vals = Z[y_true == gi, di]
                    if len(vals) > 0:
                        base_color = PAL(gi)
                        ax.hist(vals, bins=20, alpha=0.75, color=base_color,
                                density=True, edgecolor='black', linewidth=0.3,
                                label=res['le'].classes_[gi] if di == 0 else None)
                ax.set_title(f'{title}\ndim {di}' if di == 0 else f'dim {di}', fontsize=8)
                ax.set_yticks([]); ax.grid(alpha=0.2)
            axes[row, 0].legend(fontsize=6, loc='upper right', title='genre', title_fontsize=6)

        plt.tight_layout()
        fname = f'{out_dir}/disentangle_{key.lower()}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'Saved: {fname}')


# Beta-VAE Latent Traversal

def plot_latent_traversal(all_results, out_dir=OUTPUT_DIR):
    _traversal_key = next(
        (k for k, v in all_results.items()
         if v is not None and any(v['models'].get(m) is not None
                                   for m in ['mlp', 'beta', 'cvae'])), None)

    if _traversal_key is None:
        print('No valid model found -- skipping latent traversal.')
        return

    res = all_results[_traversal_key]
    n_dims = 5
    n_steps = 7
    z_range = np.linspace(-3, 3, n_steps)

    model_configs = {
        'mlp': 'MLP-VAE (baseline)',
        'beta': f'Beta-VAE (beta={res.get("best_beta", "?"):.1f}, best)',
        'cvae': 'CVAE (zero-condition)',
    }

    for model_key, model_label in model_configs.items():
        model_obj = res['models'].get(model_key)
        if model_obj is None:
            print(f'{model_label} not found -- skipping.')
            continue

        model_obj = model_obj.eval()
        Z_mean = torch.FloatTensor(res['Z'][model_key]).mean(0)

        fig, axes = plt.subplots(n_dims, n_steps,
                                 figsize=(2 * n_steps, 2 * n_dims),
                                 squeeze=False)

        for di in range(n_dims):
            for ti, val in enumerate(z_range):
                z = Z_mean.clone()
                z[di] = val

                with torch.no_grad():
                    if model_key == 'cvae':
                        cond_dim = model_obj.cond_dim
                        cond = torch.zeros(1, cond_dim).to(DEVICE)
                        recon = model_obj.decode(
                                       z.unsqueeze(0).to(DEVICE), cond
                                   ).cpu().numpy().flatten()
                    else:
                        recon = model_obj.decode(
                                    z.unsqueeze(0).to(DEVICE)
                                ).cpu().numpy().flatten()

                ax = axes[di, ti]
                ax.plot(recon, lw=0.8, color='#1565C0')
                ax.set_xticks([]); ax.set_yticks([])
                if ti == 0: ax.set_ylabel(f'dim {di}', fontsize=7)
                if di == 0: ax.set_title(f'z={val:+.1f}', fontsize=7)

        plt.suptitle(f'{model_label} Latent Traversal - {_traversal_key}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()

        safe_key  = model_key.replace('/', '_')
        save_path = f'{out_dir}/latent_traversal_{safe_key}.png'
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.show()
        print(f'Saved: {save_path}  (dataset={_traversal_key}, model={model_label})')


# Reconstruction Examples

def plot_reconstruction_examples(all_results, out_dir=OUTPUT_DIR):
    rng = np.random.default_rng(SEED)
    for key, res in all_results.items():
        if res is None: continue
        X_sc = res['X_sc']
        models = res['models']
        n_show = 5
        idx = rng.choice(len(X_sc), min(n_show, len(X_sc)), replace=False)
        fig, axes = plt.subplots(len(idx), 3, figsize=(18, len(idx) * 2.8), squeeze=False)
        fig.suptitle(f'Reconstruction Examples - {res["name"]}', fontsize=13, fontweight='bold')
        for row, i in enumerate(idx):
            x_orig = torch.FloatTensor(X_sc[i:i+1]).to(DEVICE)
            genre  = res['y_labels'][i]
            for col, (mkey, model) in enumerate([
                ('MLP-VAE', models['mlp']),
                ('Beta-VAE', models['beta']),
                ('CVAE', models['cvae']),
            ]):
                model.eval()
                with torch.no_grad():
                    if mkey == 'CVAE':
                        n_cl = res['n_class']
                        c = torch.zeros(1, n_cl, device=DEVICE)
                        c[0, res['le'].transform([genre])[0]] = 1.0
                        recon, _, _, _ = model(x_orig, c)
                    else:
                        recon, _, _, _ = model(x_orig)
                orig_np = x_orig.cpu().numpy().flatten()
                recon_np = recon.cpu().numpy().flatten()
                show_dim = min(AUDIO_FEAT_DIM, len(orig_np))
                ax = axes[row, col]
                ax.plot(orig_np[:show_dim],  color='#1565C0', lw=1.5,
                        label='Original' if row == 0 else None)
                ax.plot(recon_np[:show_dim], color='#FF5722', lw=1.5, linestyle='--',
                        label='Recon' if row == 0 else None)
                mse = float(np.mean((orig_np[:show_dim] - recon_np[:show_dim]) ** 2))
                ax.set_title(f'{mkey} | {genre[:12]} | MSE={mse:.4f}', fontsize=7)
                ax.set_yticks([]); ax.grid(alpha=0.2)
                if row == 0: ax.legend(fontsize=7)
        plt.tight_layout()
        fname = f'{out_dir}/reconstruction_{key.lower()}.png'
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.show(); print(f'Saved: {fname}')


# Quantitative Analysis & Interpretation

def print_quantitative_analysis(all_results):
    print('=' * 80)
    print(' ANALYSIS: All VAE Variants vs PCA Baseline (K-Means, Silhouette + NMI)')
    print('=' * 80)
    for ds_key, res in all_results.items():
        if res is None: continue
        print(f'\n  Dataset: {ds_key}')
        print(f' {"" :-<76}')
        pca_sil = res['cl']['pca']['KMeans']['metrics']['sil']
        pca_nmi = res['cl']['pca']['KMeans']['metrics']['nmi']
        for zkey, zlab in MODEL_LABELS.items():
            if zkey in ('pca', 'raw') or zkey not in res['cl']: continue
            m = res['cl'][zkey]['KMeans']['metrics']
            vae_sil = m['sil']; vae_nmi = m['nmi']
            if np.isnan(vae_sil) or np.isnan(pca_sil):
                print(f'  {ds_key:<12} | {zlab:<20} | Sil: NaN -- skipped'); continue
            d_sil = vae_sil - pca_sil; d_nmi = vae_nmi - pca_nmi
            pct = d_sil / abs(pca_sil) * 100 if pca_sil != 0 else 0
            v = ('BETTER' if d_sil > 0.005 else 'WORSE' if d_sil < -0.005 else 'SIMILAR')
            print(f'  {ds_key:<12} | {zlab:<20} | '
                  f'Sil: {vae_sil:.4f} vs PCA {pca_sil:.4f} '
                  f'delta={d_sil:+.4f}({pct:+.1f}%)  NMI delta={d_nmi:+.4f}  {v}')

    print()
    print('INTERPRETATION')
    print('-' * 70)
    print('BETTER - Non-linear encoder captures manifold structure PCA cannot.')
    print('When: high-dim data, complex genre boundaries, entangled audio features.')
    print()
    print('WORSE - Small dataset (VAE overfits), very low-dim data, beta too high.')
    print()
    print('Conv2D-VAE : captures local time-frequency correlations (delta-stacked MFCC).')
    print('HybridConvVAE: end-to-end Conv2D + lyric fusion -- strongest when real lyrics available.')
    print('Beta-VAE : disentangled latent -> individual dims align with audio factors.')
    print('CVAE : genre-aware latent -- useful for conditional generation.')
    print('MultiModalVAE: joint audio+lyric encoder, best when lyrics are informative.')
    print()
    print('NMI: symmetric, corrects for cluster size. ARI: corrects for random labelling.')
    print('Purity: fraction of samples in majority-genre clusters -- easy to interpret.')


# Head-to-Head Paradigm Comparison

METRICS_INFO = [
    ('sil', 'Silhouette',True),
    ('db', 'Davies-Bouldin', False),
    ('ch', 'Calinski-H', True),
    ('ari', 'ARI', True),
    ('nmi', 'NMI', True),
    ('purity', 'Purity', True),
]

PARADIGMS = [
    ('vae_best', 'Best-VAE', '#1565C0'),
    ('pca', 'PCA + K-Means', '#B71C1C'),
    ('ae', 'Autoencoder + K-Means', '#FF8F00'),
    ('raw', 'Direct Spectral', '#2E7D32'),
]

METRIC_LABELS_H2H = {
    'sil': 'Silhouette (up)',
    'db': 'Davies-Bouldin (dn)',
    'ch': 'Calinski-H (up)',
    'ari': 'ARI (up)',
    'nmi': 'NMI (up)',
    'purity': 'Purity (up)',
}


def _get_kmeans_metrics(res, zkey):
    try:
        return res['cl'][zkey]['KMeans']['metrics']
    except (KeyError, TypeError):
        return None


def _best_vae_key(res):
    VAE_KEYS = ['mlp', 'conv', 'hyb_conv', 'hyb_mlp', 'beta', 'cvae', 'conv1d', 'mm']
    best_key, best_sil = None, -np.inf
    for k in VAE_KEYS:
        m = _get_kmeans_metrics(res, k)
        if m is not None and not np.isnan(m.get('sil', np.nan)):
            if m['sil'] > best_sil:
                best_sil, best_key = m['sil'], k
    return best_key, best_sil


def paradigm_comparison(all_results, out_dir=OUTPUT_DIR):
    """
    Head-to-head comparison: Best-VAE vs PCA+KMeans vs AE+KMeans vs Direct Spectral.
    Produces bar chart, radar chart, ranked summary table, and CSV.
    """
    metric_col_order = [METRIC_LABELS_H2H[mk] for mk, _, _ in METRICS_INFO]
    higher_map = {METRIC_LABELS_H2H[mk]: hb for mk, _, hb in METRICS_INFO}

    # collect results
    comparison_rows = []
    vae_best_info = {}

    print('=' * 90)
    print('  PARADIGM COMPARISON  |  K-Means  |  All 6 Metrics')
    print('=' * 90)

    for ds_key, res in all_results.items():
        if res is None:
            continue
        best_vae_k, _ = _best_vae_key(res)
        vae_best_info[ds_key] = (best_vae_k, MODEL_LABELS.get(best_vae_k, str(best_vae_k)))

        for p_key, p_label, _ in PARADIGMS:
            if p_key == 'vae_best':
                m = _get_kmeans_metrics(res, best_vae_k) if best_vae_k else None
            else:
                m = _get_kmeans_metrics(res, p_key)

            row = {'Dataset': ds_key, 'Paradigm': p_label}
            for mk, mlabel, _ in METRICS_INFO:
                row[METRIC_LABELS_H2H[mk]] = m.get(mk, np.nan) if m else np.nan
            comparison_rows.append(row)

    df_cmp = pd.DataFrame(comparison_rows)
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 180)
    print(df_cmp.to_string(index=False))
    print()
    print(' Best VAE variant selected per dataset:')
    for ds_key, (bk, bl) in vae_best_info.items():
        print(f'    {ds_key:<12} -> {bl}')

    # bar chart
    datasets = [k for k, v in all_results.items() if v is not None]
    n_ds = len(datasets)
    n_para = len(PARADIGMS)
    w = 0.18
    x = np.arange(n_ds)

    fig, axes = plt.subplots(2, 3, figsize=(28, 14), squeeze=False)
    fig.suptitle(
        'Head-to-Head Clustering Paradigm Comparison\n'
        'Blue=Best-VAE  |  Red=PCA+KMeans  |  Orange=AE+KMeans  |  Green=Direct Spectral',
        fontsize=14, fontweight='bold'
    )

    for ax, col in zip(axes.flat, metric_col_order):
        for fi, (p_key, p_label, color) in enumerate(PARADIGMS):
            vals = []
            for ds_key in datasets:
                row = df_cmp[(df_cmp['Dataset'] == ds_key) & (df_cmp['Paradigm'] == p_label)]
                vals.append(float(row[col].values[0]) if len(row) > 0 else np.nan)
            offset = (fi - n_para / 2 + 0.5) * w
            bars = ax.bar(x + offset, vals, w, label=p_label, color=color, alpha=0.85,
                            edgecolor='white', linewidth=0.5)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + abs(bar.get_height()) * 0.015,
                            f'{v:.3f}', ha='center', va='bottom', fontsize=6.5,
                            fontweight='bold', rotation=45)
        arrow = 'higher is better' if higher_map[col] else 'lower is better'
        ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=11)
        ax.set_ylabel(col, fontsize=10)
        ax.set_title(f'{col}  ({arrow})', fontweight='bold', fontsize=11)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/paradigm_comparison_bar.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: paradigm_comparison_bar.png')

    # ranked summary table
    print()
    print('=' * 90)
    print(' RANKED SUMMARY (rank 1 = best per metric per dataset)')
    print('=' * 90)

    rank_rows = []
    for ds_key in datasets:
        sub = df_cmp[df_cmp['Dataset'] == ds_key].copy()
        for col, hb in higher_map.items():
            vals = sub[col].values.astype(float)
            order = np.argsort(vals * (-1 if hb else 1))
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, n_para + 1)
            for i, (_, p_label, _) in enumerate(PARADIGMS):
                rank_rows.append({
                    'Dataset': ds_key, 'Paradigm': p_label,
                    'Metric': col, 'Value': vals[i], 'Rank': ranks[i],
                })

    df_rank = pd.DataFrame(rank_rows)
    avg_rank = (df_rank.groupby(['Dataset', 'Paradigm'])['Rank']
                       .mean().reset_index()
                       .rename(columns={'Rank': 'Avg Rank (lower=better)'}))
    avg_rank['Avg Rank (lower=better)'] = avg_rank['Avg Rank (lower=better)'].round(2)
    pivot_rank = avg_rank.pivot(index='Dataset', columns='Paradigm',
                                values='Avg Rank (lower=better)')
    print(pivot_rank.to_string())
    print()
    print('  Winner (lowest avg rank) per dataset:')
    for ds_key in datasets:
        row = pivot_rank.loc[ds_key]
        winner = row.idxmin()
        print(f' {ds_key:<12} -> {winner}  (avg rank {row.min():.2f})')

    # radar chart
    try:
        radar_metric_cols = [METRIC_LABELS_H2H[mk]
                             for mk in ['sil', 'ari', 'nmi', 'purity', 'ch', 'db']]
        N = len(radar_metric_cols)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig_r, ax_r = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))
        fig_r.suptitle(
            'Paradigm Radar - Mean Score Across All Datasets\n'
            '(all metrics normalised to [0,1]; Davies-Bouldin inverted)',
            fontsize=12, fontweight='bold'
        )

        radar_data = {}
        for _, p_label, _ in PARADIGMS:
            sub = df_cmp[df_cmp['Paradigm'] == p_label]
            scores = [float(sub[col].mean()) for col in radar_metric_cols]
            radar_data[p_label] = scores

        all_sc = np.array(list(radar_data.values()))
        col_min = all_sc.min(axis=0)
        col_max = all_sc.max(axis=0)
        col_rng = np.where(col_max - col_min < 1e-9, 1.0, col_max - col_min)
        db_idx = radar_metric_cols.index(METRIC_LABELS_H2H['db'])

        for (_, p_label, color), raw_sc in zip(PARADIGMS, all_sc):
            norm_sc = (raw_sc - col_min) / col_rng
            norm_sc[db_idx] = 1.0 - norm_sc[db_idx]   # invert DB
            values = norm_sc.tolist() + [norm_sc[0]]
            ax_r.plot(angles, values, 'o-', linewidth=2, color=color, label=p_label)
            ax_r.fill(angles, values, alpha=0.12, color=color)

        radar_tick_labels = ['Silhouette', 'ARI', 'NMI', 'Purity', 'Calinski-H', 'DB (inv)']
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(radar_tick_labels, size=10)
        ax_r.set_ylim(0, 1)
        ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_r.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], size=7, color='grey')
        ax_r.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
        ax_r.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/paradigm_radar.png', dpi=150, bbox_inches='tight')
        plt.show()
        print('Saved: paradigm_radar.png')
    except Exception as e_radar:
        print(f'  [Radar chart skipped: {e_radar}]')

    # export CSV
    df_cmp.to_csv(f'{out_dir}/paradigm_comparison.csv', index=False)
    print('Saved: paradigm_comparison.csv')

    print()
    print('INTERPRETATION')
    print('-' * 70)
    print('Blue  Best-VAE : non-linear latent + KL regularisation -> smooth,')
    print(' clusterable space. Wins on complex genre boundaries.')
    print('Red   PCA+K-Means : fast, interpretable linear baseline.')
    print('Orange AE+K-Means : non-linear compression but NO KL.')
    print('Green  Direct Spect : K-Means on raw 65-dim features.')


# Final Summary Report

def print_final_report(all_results):
    SEP = '=' * 80
    print(SEP)
    print(' ADVANCED MULTI-MODAL VAE CLUSTERING - FINAL REPORT  (Combined Task)')
    print(' 3 Datasets: GTZAN, BanglaGITI, BMGCD')
    print(' 11 Feature Spaces x 4 Algorithms x 6 Metrics | No Synthetic Audio Data')
    print(SEP)

    def _f(v):
        return f'{v:.4f}' if isinstance(v, float) and not np.isnan(v) else '  N/A '

    for ds_key, res in all_results.items():
        if res is None: continue
        print(f'\n  {res["name"]}  |  {len(res["y_labels"])} samples  {res["n_class"]} genres')
        print(f' {"Features":<20} {"Algorithm":<24} {"Sil":>7} {"DB(dn)":>7} {"CH":>9} {"ARI":>7} {"NMI":>7} {"Purity":>7}')
        print('  ' + '-' * 87)
        for zkey, zlab in MODEL_LABELS.items():
            if zkey not in res['cl']: continue
            for algo in ['KMeans', 'Agglomerative_Ward', 'Agglomerative_Complete', 'DBSCAN']:
                if algo not in res['cl'][zkey]: continue
                m = res['cl'][zkey][algo]['metrics']
                print(f'  {zlab:<20} {algo:<24} {_f(m["sil"]):>7} {_f(m["db"]):>7} '
                      f'{_f(m["ch"]):>9} {_f(m["ari"]):>7} {_f(m["nmi"]):>7} {_f(m["purity"]):>7}')

    print()
    print(SEP)
    print(' LYRICS COVERAGE SUMMARY')
    print(SEP)
    for ds_key, res in all_results.items():
        if res is None: continue
        has_real = res.get('has_real_lyrics', np.array([]))
        n_real = int(np.sum(has_real))
        n_total = len(res['y_labels'])
        pct = 100 * n_real / max(n_total, 1)
        print(f' {ds_key:<12}: {n_real:>4}/{n_total} real lyrics ({pct:.1f}%) '
              f'| {n_total - n_real} neutral fallback')
    print()
    print(' Note: GTZAN lyrics are 100% genre-seed based (numeric filenames = no title).')
    print(' BanglaGITI/BMGCD: real lyrics scraped where available (gaanesuno.com).')
    print(' MultiModalVAE uses joint audio+lyric encoder -- strongest with real lyrics.')


# Download All Results

def download_results(out_dir=OUTPUT_DIR):
    zip_path = os.path.join(out_dir, 'vae_combined_results')
    shutil.make_archive(zip_path, 'zip', out_dir)
    try:
        from google.colab import files
        files.download(zip_path + '.zip')
        print('Download started!')
    except ImportError:
        print(f'Zip saved: {zip_path}.zip')
        print('(google.colab not available — running outside Colab)')


print('evaluation.py loaded')
print('Functions: build_metrics_df | plot_metrics_heatmap | plot_best_metrics_bar')
print('plot_vae_vs_baseline | plot_disentanglement | plot_latent_traversal')
print('plot_reconstruction_examples | paradigm_comparison | print_final_report | download_results')
