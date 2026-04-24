"""
clustering.py
-------------
Multi-algorithm clustering engine (KMeans, Agglomerative, DBSCAN),
elbow analysis, full experiment pipeline across all 9 models + PCA + Raw,
and all visualisation functions (t-SNE, UMAP, genre/language plots,
DBSCAN analysis, cluster composition, language separation, training curves).
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, adjusted_rand_score,
    normalized_mutual_info_score,
)
import umap

from vae import (
    DEVICE, SEED, LATENT_DIM, BATCH_SIZE, EPOCHS, LR, BETA, BETA_VAE_B,
    BETA_VALUES, AUDIO_FEAT_DIM, LYRIC_DIM, FUSION_DIM, KMEANS_NINIT,
    N_MFCC_ROWS, TIME_FRAMES, MFCC_2D_DIM,
    MLPVAE, BetaVAE, CVAE, ConvVAE, Autoencoder, MultiModalVAE,
    Conv2DVAE, HybridConvVAE,
    normalize_for_conv2d, align_for_conv2d,
    train_model, extract_latent, vae_loss_fn,
)
from dataset import make_multimodal, make_genre_onehot, OUTPUT_DIR

warnings.filterwarnings('ignore')

# Model display labels & colors
MODEL_LABELS = {
    'mlp':      'MLP-VAE',
    'conv':     'Conv2D-VAE',
    'hyb_conv': 'Hybrid-Conv-VAE',
    'hyb_mlp':  'Hybrid-MLP-VAE',
    'beta':     'Beta-VAE',
    'cvae':     'CVAE',
    'conv1d':   'Conv1D-VAE',
    'ae':       'Autoencoder',
    'mm':       'MultiModalVAE',
    'pca':      'PCA',
    'raw':      'Raw-Spectral',
}

COLORS_M = {
    'MLP-VAE':         '#1565C0',
    'Conv2D-VAE':      '#6A1B9A',
    'Hybrid-Conv-VAE': '#2E7D32',
    'Hybrid-MLP-VAE':  '#E65100',
    'Beta-VAE':        '#AD1457',
    'CVAE':            '#00838F',
    'Conv1D-VAE':      '#558B2F',
    'Autoencoder':     '#FF8F00',
    'MultiModalVAE':   '#00695C',
    'PCA':             '#B71C1C',
    'Raw-Spectral':    '#546E7A',
}

LANG_MK  = {'English': 'o', 'Bangla': '^'}
LANG_COL = {'English': '#1565C0', 'Bangla': '#C62828'}

# Subsets for plot filtering
Z_KEYS_ALL = ['mlp', 'conv', 'hyb_conv', 'hyb_mlp',
              'beta', 'cvae', 'conv1d', 'ae', 'mm', 'pca', 'raw']
SKIP_VIS   = {'pca'}   # PCA already low-dim — use first 2 components directly


# Multi-Algorithm Clustering Engine

def _fmt(v):
    """Format metric float for console."""
    return f'{v:+.3f}' if (isinstance(v, float) and not np.isnan(v)) else '   NaN'


def cluster_purity(y_true_masked, cluster_labels_masked):
    """Compute cluster purity on already-masked arrays (noise excluded)."""
    if len(y_true_masked) == 0:
        return np.nan
    yt = LabelEncoder().fit_transform(y_true_masked)
    cl = cluster_labels_masked
    total = sum(np.bincount(yt[cl == k]).max() for k in np.unique(cl))
    return total / len(yt)


def compute_metrics(Z, y_true, cluster_labels):
    """
    Compute all 6 metrics on noise-free subset.
    Returns dict: sil, db, ch, nmi, ari, purity

    NB2 technique (exact):
    1. Guard degenerate Z (NaN / Inf / collapsed std < 1e-6) BEFORE anything.
    2. Minimum-sample check: len(Zm) >= 10  (NB2 threshold, not n_cl+1).
    3. NMI uses average_method='arithmetic'  (NB2 exact call).
    """
    nan = np.nan

    # Guard: invalid / degenerate Z
    if Z is None or len(Z) == 0:
        return dict(sil=nan, db=nan, ch=nan, nmi=nan, ari=nan, purity=nan)
    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        return dict(sil=nan, db=nan, ch=nan, nmi=nan, ari=nan, purity=nan)
    if np.std(Z) < 1e-6:
        return dict(sil=nan, db=nan, ch=nan, nmi=nan, ari=nan, purity=nan)

    mask = cluster_labels != -1
    Zm   = Z[mask]
    ym   = np.asarray(y_true)[mask]
    cm   = cluster_labels[mask]
    n_cl = len(set(cm))

    # Guard: too few samples or clusters
    if n_cl < 2 or len(Zm) < 10:
        return dict(sil=nan, db=nan, ch=nan, nmi=nan, ari=nan, purity=nan)

    return dict(
        sil    = silhouette_score(Zm, cm),
        db     = davies_bouldin_score(Zm, cm),
        ch     = calinski_harabasz_score(Zm, cm),
        nmi    = normalized_mutual_info_score(ym, cm, average_method='arithmetic'),
        ari    = adjusted_rand_score(ym, cm),
        purity = cluster_purity(ym, cm),
    )


def _nan_metrics():
    """Return an all-NaN metrics dict (used when Z is degenerate)."""
    n = np.nan
    return dict(sil=n, db=n, ch=n, nmi=n, ari=n, purity=n)


def run_clustering(Z, y_true, n_class, tag=''):
    """
    Run KMeans, Agglomerative (Ward + Complete), DBSCAN.
    Returns dict: algo -> {labels, metrics, [n_found, noise_pct, eps]}

    NB2 technique (exact):
    Degenerate Z guard at the TOP — NaN/Inf or collapsed std < 1e-6
    -> returns NaN for all metrics immediately.
    """
    K       = n_class
    results = {}

    # NB2 upfront guard
    _degenerate = (
        Z is None or len(Z) == 0
        or np.any(np.isnan(Z)) or np.any(np.isinf(Z))
        or float(np.std(Z)) < 1e-6
    )
    if _degenerate:
        reason = ('NaN/Inf in Z' if (Z is not None and (np.any(np.isnan(Z)) or np.any(np.isinf(Z))))
                  else 'latent space collapsed (std<1e-6)')
        if tag:
            print(f'  [{tag}] SKIP clustering — {reason}')
        _dummy_labels = np.zeros(len(Z) if Z is not None else 0, dtype=int)
        _dummy_db = dict(labels=_dummy_labels, eps=0.0, noise_pct=100.0,
                         n_found=0, metrics=_nan_metrics())
        _dummy    = dict(labels=_dummy_labels, metrics=_nan_metrics())
        return dict(KMeans=_dummy, Agglomerative_Ward=_dummy,
                    Agglomerative_Complete=_dummy, DBSCAN=_dummy_db)

    # K-Means
    km = KMeans(n_clusters=K, n_init=KMEANS_NINIT, random_state=42).fit(Z)
    results['KMeans'] = {'labels': km.labels_, 'metrics': compute_metrics(Z, y_true, km.labels_)}

    # Agglomerative Ward
    agg_w = AgglomerativeClustering(n_clusters=K, linkage='ward').fit(Z)
    results['Agglomerative_Ward'] = {
        'labels': agg_w.labels_,
        'metrics': compute_metrics(Z, y_true, agg_w.labels_),
    }

    # Agglomerative Complete
    agg_c = AgglomerativeClustering(n_clusters=K, linkage='complete').fit(Z)
    results['Agglomerative_Complete'] = {
        'labels': agg_c.labels_,
        'metrics': compute_metrics(Z, y_true, agg_c.labels_),
    }

    # DBSCAN — eps auto-tuned via percentile sweep on L2-normalised Z
    Z_norm    = normalize(Z, norm='l2')
    min_samp  = max(3, len(Z) // (K * 10))
    nbrs      = NearestNeighbors(n_neighbors=min_samp).fit(Z_norm)
    dists, _  = nbrs.kneighbors(Z_norm)
    kth_dists = np.sort(dists[:, -1])

    best_labels, best_eps, best_n = None, None, -1
    for pct in range(5, 96, 5):
        eps_try   = float(np.percentile(kth_dists, pct))
        l_try     = DBSCAN(eps=eps_try, min_samples=min_samp).fit_predict(Z_norm)
        n_try     = len(set(l_try)) - (1 if -1 in l_try else 0)
        noise_try = (l_try == -1).mean()
        if n_try >= 2 and noise_try < 0.30:
            if best_n == -1 or abs(n_try - K) < abs(best_n - K):
                best_labels, best_eps, best_n = l_try, eps_try, n_try

    if best_labels is None:   # fallback
        best_eps    = float(np.percentile(kth_dists, 50))
        best_labels = DBSCAN(eps=best_eps, min_samples=min_samp).fit_predict(Z_norm)
        best_n      = len(set(best_labels)) - (1 if -1 in best_labels else 0)

    noise_pct = float((best_labels == -1).mean() * 100)
    mask_db   = best_labels != -1
    db_metrics = (
        compute_metrics(Z_norm[mask_db], y_true[mask_db], best_labels[mask_db])
        if best_n >= 2 and mask_db.sum() > 1
        else _nan_metrics()
    )

    results['DBSCAN'] = {
        'labels':    best_labels,
        'eps':       best_eps,
        'noise_pct': noise_pct,
        'n_found':   best_n,
        'metrics':   db_metrics,
    }

    if tag:
        print(f'  [{tag}]')
        for algo, r in results.items():
            m = r['metrics']
            extra = (f'  clusters={r["n_found"]}  noise={r["noise_pct"]:.0f}%'
                     if algo == 'DBSCAN' else '')
            print(f'    {algo:<25} '
                  f'Sil={_fmt(m["sil"])}  DB={_fmt(m["db"])}  '
                  f'NMI={_fmt(m["nmi"])}  ARI={_fmt(m["ari"])}  '
                  f'Pur={_fmt(m["purity"])}{extra}')
    return results


# Elbow Analysis Helper

def elbow_analysis(Z, k_range=range(2, 16)):
    """Inertia, silhouette, CH across k values. Returns optimal_k via silhouette argmax.

    NB2 technique: same degenerate-Z guards used in run_clustering / compute_metrics.
    """
    k_range = [k for k in k_range if k < len(Z)]
    if len(k_range) < 2:
        print('  elbow_analysis: too few samples.')
        return {}
    if Z is None or len(Z) == 0:
        print('  elbow_analysis: empty Z.')
        return {}
    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        print('  elbow_analysis: Z contains NaN/Inf — skipping.')
        return {}
    if np.std(Z) < 1e-6:
        print('  elbow_analysis: latent space collapsed (std<1e-6) — skipping.')
        return {}
    inertias, sils, chs, dbs = [], [], [], []
    for k in k_range:
        km     = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Z)
        n_uniq = len(set(km.labels_))
        inertias.append(km.inertia_)
        if n_uniq < 2:
            sils.append(float('nan')); chs.append(float('nan')); dbs.append(float('nan'))
        else:
            sils.append(silhouette_score(Z, km.labels_))
            chs.append(calinski_harabasz_score(Z, km.labels_))
            dbs.append(davies_bouldin_score(Z, km.labels_))
    sils_arr  = np.array(sils, dtype=float)
    valid_idx = np.where(~np.isnan(sils_arr))[0]
    optimal_k = k_range[int(valid_idx[np.argmax(sils_arr[valid_idx])])] if len(valid_idx) > 0 else k_range[0]
    return dict(k_range=list(k_range), inertias=inertias,
                sil_scores=sils, ch_scores=chs, db_scores=dbs, optimal_k=optimal_k)


# Full Experiment Pipeline

def full_pipeline(X_raw, y_labels, lang_labels, dataset_name,
                  file_paths=None, X_raw_2d=None, scaler=None):
    """
    Full pipeline: 9 models + PCA baseline + raw spectral clustering.

    X_raw    : (N, 65)   — 65-dim audio features
    X_raw_2d : (N, 7680) — delta-stacked MFCC for Conv2DVAE / HybridConvVAE
    scaler   : fitted StandardScaler (pass scaler_all for cross-dataset consistency)
    """
    if len(X_raw) == 0:
        print(f'  SKIP {dataset_name} — empty dataset.')
        return None

    SEP = '=' * 70
    print(f'\n{SEP}')
    print(f'  DATASET : {dataset_name}')
    print(f'  Samples={len(X_raw)} | Features={X_raw.shape[1]} | '
          f'Genres={len(np.unique(y_labels))}')
    print(SEP)

    # Scaling
    if scaler is not None:
        X_sc = scaler.transform(X_raw).astype(np.float32)
    else:
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_raw).astype(np.float32)

    le      = LabelEncoder()
    y_true  = le.fit_transform(y_labels)
    n_class = len(le.classes_)
    pca_dim = min(LATENT_DIM, X_sc.shape[1], X_sc.shape[0] - 1)

    # Records for lyrics
    records = [
        {'file':     file_paths[i] if file_paths is not None else None,
         'genre':    str(y_labels[i]),
         'language': str(lang_labels[i])}
        for i in range(len(X_raw))
    ]

    # Hybrid features (audio + lyrics)
    print('  Building multi-modal features (audio + real lyrics)...')
    X_hybrid, has_real, X_lyric_l2 = make_multimodal(X_raw, records)
    has_real     = np.array(has_real, dtype=bool)
    X_hybrid_sc  = X_hybrid.astype(np.float32)   # already L2 normalized
    X_multimodal = np.hstack([X_sc, X_lyric_l2]).astype(np.float32)
    C_oh = make_genre_onehot(y_labels, le)

    # Conv2D prep
    if X_raw_2d is not None and len(X_raw_2d) > 0:
        X_conv2d    = normalize_for_conv2d(X_raw_2d)
        X_conv2d    = align_for_conv2d(X_conv2d)
        _has_conv2d = True
    else:
        X_conv2d    = None
        _has_conv2d = False
        print('  [Conv2DVAE] X_raw_2d not provided — skipping Conv2D branch')

    # MLP-VAE
    print('  MLP-VAE ...')
    m_mlp, mlp_hist, mlp_loss = train_model(
        X_sc, MLPVAE(X_sc.shape[1], LATENT_DIM).to(DEVICE), model_type='vae', beta=1.0)
    Z_mlp = extract_latent(m_mlp, X_sc, model_type='vae')

    # Conv2D-VAE
    if _has_conv2d:
        print(f'  Conv2DVAE ({N_MFCC_ROWS}x{TIME_FRAMES}) ...')
        m_conv, conv_hist, conv_loss = train_model(
            X_conv2d, Conv2DVAE().to(DEVICE), model_type='vae', beta=1.0)
        Z_conv = extract_latent(m_conv, X_conv2d, model_type='vae')
    else:
        print('  Conv2DVAE fallback -> MLP-VAE on 65-dim ...')
        m_conv, conv_hist, conv_loss = train_model(
            X_sc, MLPVAE(X_sc.shape[1], LATENT_DIM).to(DEVICE), model_type='vae', beta=1.0)
        Z_conv = extract_latent(m_conv, X_sc, model_type='vae')

    # HybridConvVAE
    if _has_conv2d:
        print('  HybridConvVAE (end-to-end Conv+Lyric) ...')
        X_hybrid_conv = np.hstack([X_conv2d, X_lyric_l2]).astype(np.float32)
        m_hyb_conv, hyb_conv_hist, hyb_conv_loss = train_model(
            X_hybrid_conv, HybridConvVAE().to(DEVICE),
            model_type='hybrid_conv', beta=1.0)
        Z_hyb_conv = extract_latent(m_hyb_conv, X_hybrid_conv, model_type='hybrid_conv')
    else:
        print('  HybridConvVAE fallback -> Hybrid-MLP-VAE ...')
        m_hyb_conv, hyb_conv_hist, hyb_conv_loss = train_model(
            X_hybrid_sc, MLPVAE(X_hybrid_sc.shape[1], LATENT_DIM).to(DEVICE),
            model_type='vae', beta=1.0)
        Z_hyb_conv = extract_latent(m_hyb_conv, X_hybrid_sc, model_type='vae')

    # Hybrid-MLP-VAE
    print('  Hybrid-MLP-VAE ...')
    m_hyb_mlp, hyb_mlp_hist, hyb_mlp_loss = train_model(
        X_hybrid_sc, MLPVAE(X_hybrid_sc.shape[1], LATENT_DIM).to(DEVICE),
        model_type='vae', beta=1.0)
    Z_hyb_mlp = extract_latent(m_hyb_mlp, X_hybrid_sc, model_type='vae')

    # Beta-VAE sweep
    print(f'  Beta-VAE sweep {BETA_VALUES} ...')
    beta_sweep      = {}
    best_beta_sil   = -np.inf
    best_beta_val   = BETA_VAE_B
    best_beta_Z     = None; best_beta_model = None
    best_beta_hist  = None; best_beta_loss  = float('inf')
    for beta_val in BETA_VALUES:
        m_b, h_b, l_b = train_model(
            X_sc, BetaVAE(X_sc.shape[1], LATENT_DIM).to(DEVICE),
            model_type='vae', beta=beta_val, verbose=False)
        Z_b   = extract_latent(m_b, X_sc, model_type='vae')
        km_b  = KMeans(n_clusters=n_class, n_init=20, random_state=42).fit(Z_b)
        n_uniq = len(set(km_b.labels_))
        m_met  = compute_metrics(Z_b, y_true, km_b.labels_) if n_uniq >= 2 else \
                 dict(sil=float('nan'), db=float('nan'), ch=float('nan'),
                      nmi=float('nan'), ari=float('nan'), purity=float('nan'))
        beta_sweep[beta_val] = dict(metrics=m_met, Z=Z_b, model=m_b, hist=h_b)
        sil = m_met['sil']
        if not np.isnan(sil) and sil > best_beta_sil:
            best_beta_sil   = sil;   best_beta_val   = beta_val
            best_beta_Z     = Z_b;   best_beta_model = m_b
            best_beta_hist  = h_b;   best_beta_loss  = l_b
    print(f'    Best beta={best_beta_val}  Sil={best_beta_sil:.4f}')
    m_beta    = best_beta_model; Z_beta    = best_beta_Z
    beta_hist = best_beta_hist;  beta_loss = best_beta_loss

    # CVAE
    print('  CVAE ...')
    m_cvae, cvae_hist, cvae_loss = train_model(
        X_sc, CVAE(X_sc.shape[1], n_class, LATENT_DIM).to(DEVICE),
        y_onehot=C_oh, model_type='cvae', beta=1.0)
    Z_cvae = extract_latent(m_cvae, X_sc, model_type='cvae')

    # Conv1D-VAE
    print('  Conv1D-VAE ...')
    m_conv1d, conv1d_hist, conv1d_loss = train_model(
        X_sc, ConvVAE(X_sc.shape[1], LATENT_DIM).to(DEVICE),
        model_type='vae', beta=1.0)
    Z_conv1d = extract_latent(m_conv1d, X_sc, model_type='vae')

    # Autoencoder
    print('  Autoencoder ...')
    m_ae, ae_hist, ae_loss = train_model(
        X_sc, Autoencoder(X_sc.shape[1], LATENT_DIM).to(DEVICE), model_type='ae')
    Z_ae = extract_latent(m_ae, X_sc, model_type='ae')

    # MultiModalVAE
    print('  MultiModalVAE ...')
    m_mm, mm_hist, mm_loss = train_model(
        X_multimodal,
        MultiModalVAE(AUDIO_FEAT_DIM, LYRIC_DIM, FUSION_DIM, LATENT_DIM).to(DEVICE),
        model_type='multimodal', audio_dim=AUDIO_FEAT_DIM, beta=1.0)
    Z_mm = extract_latent(m_mm, X_multimodal,
                          model_type='multimodal', audio_dim=AUDIO_FEAT_DIM)

    # PCA baseline
    print('  PCA baseline ...')
    Z_pca = PCA(n_components=pca_dim, random_state=42).fit_transform(X_sc)

    # Raw spectral
    print('  Direct feature clustering ...')
    cl_raw = run_clustering(X_sc, y_true, n_class, 'Raw-Spectral')

    # Elbow analysis
    print('  Elbow analysis ...')
    elbow = elbow_analysis(Z_mlp, k_range=range(2, min(22, n_class + 5)))

    # Clustering on all latent spaces
    print('\n  --- Clustering ---')
    cl_mlp      = run_clustering(Z_mlp,      y_true, n_class, 'MLP-VAE')
    cl_conv     = run_clustering(Z_conv,     y_true, n_class, 'Conv2D-VAE')
    cl_hyb_conv = run_clustering(Z_hyb_conv, y_true, n_class, 'Hybrid-Conv-VAE')
    cl_hyb_mlp  = run_clustering(Z_hyb_mlp,  y_true, n_class, 'Hybrid-MLP-VAE')
    cl_beta     = run_clustering(Z_beta,     y_true, n_class, 'Beta-VAE')
    cl_cvae     = run_clustering(Z_cvae,     y_true, n_class, 'CVAE')
    cl_conv1d   = run_clustering(Z_conv1d,   y_true, n_class, 'Conv1D-VAE')
    cl_ae       = run_clustering(Z_ae,       y_true, n_class, 'Autoencoder')
    cl_mm       = run_clustering(Z_mm,       y_true, n_class, 'MultiModalVAE')
    cl_pca      = run_clustering(Z_pca,      y_true, n_class, 'PCA')

    return dict(
        name=dataset_name, X_sc=X_sc, y_true=y_true,
        y_labels=y_labels, lang_labels=lang_labels,
        le=le, n_class=n_class, elbow=elbow,
        has_real_lyrics=has_real, scaler=scaler,
        best_beta=best_beta_val, beta_sweep=beta_sweep,
        Z=dict(mlp=Z_mlp, conv=Z_conv,
               hyb_conv=Z_hyb_conv, hyb_mlp=Z_hyb_mlp,
               beta=Z_beta, cvae=Z_cvae,
               conv1d=Z_conv1d, ae=Z_ae,
               mm=Z_mm, pca=Z_pca, raw=X_sc),
        cl=dict(mlp=cl_mlp, conv=cl_conv,
                hyb_conv=cl_hyb_conv, hyb_mlp=cl_hyb_mlp,
                beta=cl_beta, cvae=cl_cvae,
                conv1d=cl_conv1d, ae=cl_ae,
                mm=cl_mm, pca=cl_pca, raw=cl_raw),
        hist=dict(mlp=mlp_hist, conv=conv_hist,
                  hyb_conv=hyb_conv_hist, hyb_mlp=hyb_mlp_hist,
                  beta=beta_hist, cvae=cvae_hist,
                  conv1d=conv1d_hist, ae=ae_hist, mm=mm_hist),
        loss=dict(mlp=mlp_loss, conv=conv_loss,
                  hyb_conv=hyb_conv_loss, hyb_mlp=hyb_mlp_loss,
                  beta=beta_loss, cvae=cvae_loss,
                  conv1d=conv1d_loss, ae=ae_loss, mm=mm_loss),
        models=dict(mlp=m_mlp, conv=m_conv,
                    hyb_conv=m_hyb_conv, hyb_mlp=m_hyb_mlp,
                    beta=m_beta, cvae=m_cvae,
                    conv1d=m_conv1d, ae=m_ae, mm=m_mm),
    )


# Genre Distribution Overview

def plot_genre_distribution(all_results, out_dir=OUTPUT_DIR):
    valid = [(k, v) for k, v in all_results.items() if v is not None]
    fig, axes = plt.subplots(1, len(valid), figsize=(8 * len(valid), 5))
    if len(valid) == 1:
        axes = [axes]
    fig.suptitle('Genre Distribution per Dataset', fontsize=14, fontweight='bold')
    for ax, (key, res) in zip(axes, valid):
        unique, counts = np.unique(res['y_labels'], return_counts=True)
        idx = np.argsort(counts)[::-1]
        bars = ax.bar(range(len(unique)), counts[idx],
                      color=plt.cm.tab20(np.linspace(0, 1, len(unique))))
        ax.set_xticks(range(len(unique)))
        ax.set_xticklabels(unique[idx], rotation=40, ha='right', fontsize=8)
        ax.set_title(res['name'], fontweight='bold'); ax.set_ylabel('Count'); ax.grid(axis='y', alpha=0.3)
        for bar, c in zip(bars, counts[idx]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(c), ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/genre_distribution.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: genre_distribution.png')


# t-SNE + UMAP Dimensionality Reduction

def compute_projections(all_results):
    """Compute t-SNE and UMAP for every latent space in all_results. Returns updated all_results."""
    print('Computing t-SNE + UMAP (~5-10 min)...')
    for key, res in all_results.items():
        if res is None: continue
        print(f'  Dataset: {key}'); res['vis'] = {}
        for zkey, Z in res['Z'].items():
            if zkey in SKIP_VIS:
                res['vis'][zkey] = {'tsne': Z[:, :2], 'umap': Z[:, :2]}
                print(f'    {zkey}... passthrough (2D already)')
                continue
            print(f'    {zkey}...', end=' ', flush=True)
            perp = min(40, max(5, Z.shape[0] // 3))
            n_nb = min(30, Z.shape[0] - 1)
            res['vis'][zkey] = {
                'tsne': TSNE(n_components=2, perplexity=perp, n_iter=1000,
                             random_state=42).fit_transform(Z),
                'umap': umap.UMAP(n_components=2, n_neighbors=n_nb,
                                  min_dist=0.1, random_state=42).fit_transform(Z),
            }
            print(f'done (N={Z.shape[0]}, perp={perp}, n_nb={n_nb})')
    print('All reductions done.')
    return all_results


# Latent Space Plots — All Models (UMAP + t-SNE)

def plot_latent_umap(all_results, out_dir=OUTPUT_DIR):
    """Plot UMAP latent space by genre and language for all models."""
    for key, res in all_results.items():
        if res is None: continue
        n_class  = res['n_class']
        PAL      = plt.colormaps['tab20'].resampled(n_class)
        n_models = len(Z_KEYS_ALL)

        fig, axes = plt.subplots(n_models, 2, figsize=(18, n_models * 3.5), squeeze=False)
        fig.suptitle(f'Latent Space (UMAP) — All Models — {res["name"]}',
                     fontsize=14, fontweight='bold')

        for row, zkey in enumerate(Z_KEYS_ALL):
            if zkey not in res['vis']: continue
            Z2 = res['vis'][zkey]['umap']
            cl = res['cl'][zkey]
            ml = MODEL_LABELS[zkey]

            # Left: genre colour
            ax = axes[row, 0]
            for gi in range(n_class):
                m = res['y_true'] == gi
                if m.any():
                    ax.scatter(Z2[m, 0], Z2[m, 1], c=[PAL(gi)], s=20, alpha=0.9, linewidths=0)
            sil = cl['KMeans']['metrics']['sil']
            nmi = cl['KMeans']['metrics']['nmi']
            ax.set_title(
                f'{ml} | Genre | Sil={sil:.3f}  NMI={nmi:.3f}'
                if not np.isnan(sil) else f'{ml} | Genre',
                fontsize=9, fontweight='bold'
            )
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.15)

            # Right: language separation
            ax = axes[row, 1]
            for lang, color, mk in [('English', '#0D47A1', 'o'), ('Bangla', '#B71C1C', '^')]:
                lm = res['lang_labels'] == lang
                if lm.any():
                    ax.scatter(Z2[lm, 0], Z2[lm, 1], c=color, marker=mk,
                               s=40 if lang == 'Bangla' else 20, alpha=0.9, label=lang,
                               linewidths=0, edgecolors='black' if lang == 'Bangla' else 'none')
            ax.set_title(f'{ml} | Language', fontsize=9, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.15)
            if row == 0:
                ax.legend(fontsize=8)

        plt.tight_layout()
        fname = f'{out_dir}/latent_all_{key.lower()}.png'
        plt.savefig(fname, dpi=110, bbox_inches='tight')
        plt.show()
        print(f'Saved: {fname}')


def plot_latent_tsne(all_results, out_dir=OUTPUT_DIR):
    """Plot t-SNE latent space by genre and language for all models."""
    for key, res in all_results.items():
        if res is None: continue
        n_class  = res['n_class']
        PAL      = plt.colormaps['tab20'].resampled(n_class)
        n_models = len(Z_KEYS_ALL)

        fig, axes = plt.subplots(n_models, 2, figsize=(18, n_models * 3.5), squeeze=False)
        fig.suptitle(f'Latent Space (t-SNE) — All Models — {res["name"]}',
                     fontsize=14, fontweight='bold')

        for row, zkey in enumerate(Z_KEYS_ALL):
            if zkey not in res['vis']: continue
            Z2 = res['vis'][zkey]['tsne']
            cl = res['cl'][zkey]
            ml = MODEL_LABELS[zkey]

            # Left: genre
            ax = axes[row, 0]
            for gi in range(n_class):
                m = res['y_true'] == gi
                if m.any():
                    ax.scatter(Z2[m, 0], Z2[m, 1], c=[PAL(gi)], s=18, alpha=0.9, linewidths=0)
            sil = cl['KMeans']['metrics']['sil']
            nmi = cl['KMeans']['metrics']['nmi']
            ax.set_title(
                f'{ml} | Genre | Sil={sil:.3f}  NMI={nmi:.3f}'
                if not np.isnan(sil) else f'{ml} | Genre',
                fontsize=9, fontweight='bold'
            )
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.15)

            # Right: language
            ax = axes[row, 1]
            for lang, color, mk in [('English', '#0D47A1', 'o'), ('Bangla', '#B71C1C', '^')]:
                lm = res['lang_labels'] == lang
                if lm.any():
                    ax.scatter(Z2[lm, 0], Z2[lm, 1], c=color, marker=mk,
                               s=40 if lang == 'Bangla' else 18, alpha=0.9,
                               linewidths=0, label=lang,
                               edgecolors='black' if lang == 'Bangla' else 'none')
            ax.set_title(f'{ml} | Language', fontsize=9, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.15)
            if row == 0:
                ax.legend(fontsize=8)

        plt.tight_layout()
        fname = f'{out_dir}/latent_tsne_{key.lower()}.png'
        plt.savefig(fname, dpi=110, bbox_inches='tight')
        plt.show()
        print(f'Saved: {fname}')


# Elbow Method Plots

def plot_elbow(all_results, out_dir=OUTPUT_DIR):
    valid_elbow = [(k, v) for k, v in all_results.items() if v is not None and v.get('elbow')]
    if not valid_elbow:
        print('No valid elbow results.')
        return
    fig, axes = plt.subplots(len(valid_elbow), 3,
                             figsize=(18, len(valid_elbow) * 5), squeeze=False)
    fig.suptitle('Elbow Method  (red = true genres | green = suggested K)',
                 fontsize=14, fontweight='bold')
    for row, (key, res) in enumerate(valid_elbow):
        elbow = res['elbow']
        ks    = elbow['k_range']
        opt_k = elbow.get('optimal_k')
        for col, (vals, ylabel, title) in enumerate([
            (elbow['inertias'],   'Inertia',          'Inertia (lower is better)'),
            (elbow['sil_scores'], 'Silhouette Score',  'Silhouette (higher is better)'),
            (elbow['ch_scores'],  'Calinski-Harabasz', 'CH Index (higher is better)'),
        ]):
            ax = axes[row, col]
            ax.plot(ks, vals, 'o-', color='#1565C0', linewidth=2, markersize=5)
            ax.axvline(res['n_class'], color='red', linestyle='--', alpha=0.7,
                       label=f'K={res["n_class"]} (true)')
            if opt_k is not None:
                ax.axvline(opt_k, color='green', linestyle=':', alpha=0.7,
                           label=f'K={opt_k} (suggested)')
            ax.set_xlabel('K'); ax.set_ylabel(ylabel)
            ax.set_title(f'{key} — {title}', fontweight='bold')
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/elbow_plots.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('Saved: elbow_plots.png')


# DBSCAN Cluster Analysis

def plot_dbscan(all_results, out_dir=OUTPUT_DIR):
    valid = [(k, v) for k, v in all_results.items() if v is not None]
    fig, axes = plt.subplots(
        len(valid), len(Z_KEYS_ALL),
        figsize=(7 * len(Z_KEYS_ALL), 5 * len(valid)),
        squeeze=False
    )
    fig.suptitle('DBSCAN Results (grey = noise points)', fontsize=14, fontweight='bold')
    for row, (key, res) in enumerate(valid):
        for col, zkey in enumerate(Z_KEYS_ALL):
            ax   = axes[row, col]
            Z2   = res['vis'][zkey]['umap']
            db   = res['cl'][zkey]['DBSCAN']
            lbls = db['labels']
            noise = lbls == -1
            ax.scatter(Z2[noise, 0], Z2[noise, 1], c='lightgrey', s=16, alpha=0.6, linewidths=0)
            if (~noise).any():
                ax.scatter(Z2[~noise, 0], Z2[~noise, 1], c=lbls[~noise],
                           cmap='tab10', s=22, alpha=0.9, linewidths=0)
            sil = db['metrics']['sil']
            t = (f'{key} | {MODEL_LABELS[zkey]}\n'
                 f'{db["n_found"]} clusters  noise={db["noise_pct"]:.0f}%')
            if not np.isnan(sil):
                t += f'  Sil={sil:.3f}'
            ax.set_title(t, fontsize=7, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/dbscan_analysis.png', dpi=120, bbox_inches='tight')
    plt.show()
    print('Saved: dbscan_analysis.png')


# Cluster Composition Heatmap

def plot_cluster_composition(all_results, out_dir=OUTPUT_DIR):
    ROW1 = ['mlp', 'conv', 'hyb_conv', 'hyb_mlp', 'beta', 'cvae']
    ROW2 = ['conv1d', 'ae', 'mm', 'pca', 'raw']
    for key, res in all_results.items():
        if res is None: continue
        fig, axes = plt.subplots(2, max(len(ROW1), len(ROW2)),
                                 figsize=(7 * max(len(ROW1), len(ROW2)), 14), squeeze=False)
        fig.suptitle(f'Cluster Composition — {res["name"]}\n(genre % within each K-Means cluster)',
                     fontsize=13, fontweight='bold')
        for row_idx, row_keys in enumerate([ROW1, ROW2]):
            for col_idx, zkey in enumerate(row_keys):
                ax     = axes[row_idx, col_idx]
                labels = res['cl'][zkey]['KMeans']['labels']
                df_tmp = pd.DataFrame({'cluster': labels, 'genre': res['y_labels']})
                ct     = pd.crosstab(df_tmp['cluster'], df_tmp['genre'])
                ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
                sns.heatmap(ct_pct, ax=ax, annot=True, fmt='.0f', cmap='YlOrRd',
                            linewidths=0.3, cbar_kws={'label': '%', 'shrink': 0.8},
                            annot_kws={'size': 6})
                ax.set_title(MODEL_LABELS[zkey], fontweight='bold', fontsize=10)
                ax.tick_params(axis='x', rotation=45, labelsize=7)
            for col_idx in range(len(row_keys), max(len(ROW1), len(ROW2))):
                axes[row_idx, col_idx].set_visible(False)
        plt.tight_layout()
        fname = f'{out_dir}/cluster_composition_{key.lower()}.png'
        plt.savefig(fname, dpi=130, bbox_inches='tight')
        plt.show()
        print(f'Saved: {fname}')


# English vs Bangla Language Separation

def plot_language_separation(all_results, out_dir=OUTPUT_DIR):
    ROW1 = ['mlp', 'conv', 'hyb_conv', 'hyb_mlp', 'beta', 'cvae']
    ROW2 = ['conv1d', 'ae', 'mm', 'pca', 'raw']
    valid  = [(k, v) for k, v in all_results.items() if v is not None]
    n_cols = max(len(ROW1), len(ROW2))
    fig, axes = plt.subplots(len(valid) * 2, n_cols,
                             figsize=(8 * n_cols, 6 * len(valid) * 2),
                             squeeze=False)
    for ds_idx, (key, res) in enumerate(valid):
        for row_offset, row_keys in enumerate([ROW1, ROW2]):
            row = ds_idx * 2 + row_offset
            for col, zkey in enumerate(row_keys):
                ax   = axes[row, col]
                Z2   = res['vis'][zkey]['umap']
                lang = res['lang_labels']
                for lng in ['English', 'Bangla']:
                    mask = lang == lng
                    if mask.any():
                        ax.scatter(Z2[mask, 0], Z2[mask, 1],
                                   c=LANG_COL[lng], marker=LANG_MK[lng],
                                   s=130, alpha=1, label=lng, linewidths=0)
                ax.set_title(f'{key} | {MODEL_LABELS[zkey]}', fontsize=11, fontweight='bold')
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.2)
                if ds_idx == 0 and row_offset == 0 and col == 0:
                    ax.legend(fontsize=10)
            for col in range(len(row_keys), n_cols):
                axes[row, col].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/language_separation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: language_separation.png')


# Training Loss Curves

def plot_training_curves(all_results, out_dir=OUTPUT_DIR):
    MODEL_COLORS_TRAIN = {
        'mlp':      '#1565C0', 'conv':     '#6A1B9A',
        'hyb_conv': '#2E7D32', 'hyb_mlp':  '#E65100',
        'beta':     '#AD1457', 'cvae':     '#00838F',
        'conv1d':   '#558B2F', 'ae':       '#FF8F00',
        'mm':       '#00695C',
    }
    valid  = [(k, v) for k, v in all_results.items() if v is not None]
    n_cols = len(valid)
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 5), squeeze=False)
    fig.suptitle('VAE Training Loss Curves per Dataset', fontsize=13, fontweight='bold')
    for ax, (key, res) in zip(axes[0], valid):
        for mkey, color in MODEL_COLORS_TRAIN.items():
            hist = res['hist'].get(mkey)
            loss = res['loss'].get(mkey)
            if hist is None or loss is None: continue
            train_losses = [h[0] for h in hist]
            ax.plot(train_losses, color=color, linewidth=2,
                    label=f'{MODEL_LABELS.get(mkey, mkey)} ({loss:.4f})')
        ax.set_title(key, fontweight='bold'); ax.set_xlabel('Epoch'); ax.set_ylabel('Train Loss')
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: training_curves.png')


print('clustering.py loaded')
print('   Clustering: KMeans | Agglom-Ward | Agglom-Complete | DBSCAN')
print('   Metrics: Silhouette | DB | CH | NMI | ARI | Purity (6 total)')
print('   Pipeline: full_pipeline() | compute_projections()')
