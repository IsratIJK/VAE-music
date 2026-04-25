"""
main.py
-------
Top-level runner — mirrors the original notebook step-by-step.
Upload all 5 .py files to /content/ on Google Colab, then run:

    !python main.py

Install dependencies first (Step 1):
    !pip install -q umap-learn scikit-learn matplotlib seaborn tqdm requests
    !pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
    !pip install -q yt-dlp librosa soundfile audioread
    !pip install -q lyricsgenius beautifulsoup4
    !pip install -q gdown kaggle
    !apt-get install -q -y ffmpeg
"""

import os
import random
import warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')

# ── Step 2: Reproducibility ───────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Module imports ─────────────────────────────────────────────────────────────
from vae import (
    DEVICE, LATENT_DIM, HIDDEN_DIMS, CONV_CHANNELS, EPOCHS,
    EARLY_STOP_PATIENCE, LR, BETA, BETA_VAE_B, BETA_VALUES,
    BATCH_SIZE, LYRIC_DIM, FUSION_DIM, KMEANS_NINIT,
    N_MFCC, TIME_FRAMES, N_MFCC_ROWS, MFCC_2D_DIM, AUDIO_FEAT_DIM,
    normalize_for_conv2d, align_for_conv2d,
    MLPVAE, BetaVAE, CVAE, ConvVAE, Autoencoder,
    MultiModalVAE, Conv2DVAE, HybridConvVAE,
    train_model, extract_latent, vae_loss_fn,
)
from dataset import (
    OUTPUT_DIR, download_gtzan, build_all_datasets,
    make_multimodal, make_genre_onehot,
)
from clustering import (
    MODEL_LABELS, COLORS_M, Z_KEYS_ALL,
    run_clustering, elbow_analysis, compute_metrics,
    full_pipeline, compute_projections,
    plot_genre_distribution, plot_latent_umap, plot_latent_tsne,
    plot_elbow, plot_dbscan, plot_cluster_composition,
    plot_language_separation, plot_training_curves,
)
from evaluation import (
    build_metrics_df, print_metrics_table, plot_metrics_heatmap,
    plot_best_metrics_bar, plot_vae_vs_baseline,
    plot_disentanglement, plot_latent_traversal,
    plot_reconstruction_examples, paradigm_comparison,
    print_quantitative_analysis, print_final_report, download_results,
)

print(f'✅ Config ready | Device: {DEVICE}')
print(f'   N_MFCC_ROWS={N_MFCC_ROWS}  TIME_FRAMES={TIME_FRAMES}  MFCC_2D_DIM={MFCC_2D_DIM}')
print(f'   AUDIO_FEAT_DIM={AUDIO_FEAT_DIM}  LYRIC_DIM={LYRIC_DIM}  LATENT_DIM={LATENT_DIM}')


# ════════════════════════════════════════════════════════════════════════════
#  Steps 11 + 12 — Download datasets & build feature arrays
# ════════════════════════════════════════════════════════════════════════════

(X_gtzan, X_gtzan_2d, y_gtzan, lang_gtzan, paths_gtzan, records_gtzan,
 X_bg,    X_bg_2d,    y_bg,    lang_bg,    paths_bg,    records_bg,
 X_bm,    X_bm_2d,    y_bm,    lang_bm,    paths_bm,    records_bm,
 scaler_all) = build_all_datasets()


# ════════════════════════════════════════════════════════════════════════════
#  Step 13 — Full Experiment Pipeline (all datasets)
# ════════════════════════════════════════════════════════════════════════════

all_results = {}

all_results['GTZAN'] = full_pipeline(
    X_gtzan, y_gtzan, lang_gtzan, 'GTZAN (English Multi-Genre)',
    file_paths=paths_gtzan, X_raw_2d=X_gtzan_2d, scaler=scaler_all)

if len(X_bg) > 0:
    all_results['BanglaGITI'] = full_pipeline(
        X_bg, y_bg, lang_bg, 'BanglaGITI (Bengali Songs)',
        file_paths=paths_bg, X_raw_2d=X_bg_2d, scaler=scaler_all)

if len(X_bm) > 0:
    all_results['BMGCD'] = full_pipeline(
        X_bm, y_bm, lang_bm, 'BMGCD (Bangla Music Genre)',
        file_paths=paths_bm, X_raw_2d=X_bm_2d, scaler=scaler_all)

print('\n✅ All experiments complete!')


# ════════════════════════════════════════════════════════════════════════════
#  Step 14 — Genre Distribution Overview
# ════════════════════════════════════════════════════════════════════════════

plot_genre_distribution(all_results, out_dir=OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════════════════
#  Step 15 — t-SNE + UMAP Dimensionality Reduction
# ════════════════════════════════════════════════════════════════════════════

all_results = compute_projections(all_results)


# ════════════════════════════════════════════════════════════════════════════
#  Step 16 — Latent Space Plots — All Models (UMAP + t-SNE)
# ════════════════════════════════════════════════════════════════════════════

plot_latent_umap(all_results, out_dir=OUTPUT_DIR)
plot_latent_tsne(all_results, out_dir=OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════════════════
#  Step 17 — Elbow Method Plots
# ════════════════════════════════════════════════════════════════════════════

plot_elbow(all_results, out_dir=OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════════════════
#  Step 18 — DBSCAN Cluster Analysis
# ════════════════════════════════════════════════════════════════════════════

plot_dbscan(all_results, out_dir=OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════════════════
#  Step 19 — Cluster Composition Heatmap
# ════════════════════════════════════════════════════════════════════════════

plot_cluster_composition(all_results, out_dir=OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════════════════
#  Step 20 — English vs Bangla Language Separation
# ════════════════════════════════════════════════════════════════════════════

plot_language_separation(all_results, out_dir=OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════════════════
#  Step 21 — Metrics Heatmap Dashboard
# ════════════════════════════════════════════════════════════════════════════

df_all = build_metrics_df(all_results)
print_metrics_table(df_all, out_dir=OUTPUT_DIR)
plot_metrics_heatmap(df_all, out_dir=OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════════════════
#  Step 22 — Best Metrics Bar Charts
# ════════════════════════════════════════════════════════════════════════════

plot_best_metrics_bar(all_results, out_dir=OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════════════════
#  Step 23 — Training Loss Curves
# ════════════════════════════════════════════════════════════════════════════

plot_training_curves(all_results, out_dir=OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════════════════
#  Step 24 — VAE vs PCA Baseline — Δ Analysis
# ════════════════════════════════════════════════════════════════════════════

plot_vae_vs_baseline(all_results, out_dir=OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════════════════
#  Step 25 — Beta-VAE Disentanglement Analysis
# ════════════════════════════════════════════════════════════════════════════

plot_disentanglement(all_results, out_dir=OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════════════════
#  Step 26 — Beta-VAE Latent Traversal
# ════════════════════════════════════════════════════════════════════════════

plot_latent_traversal(all_results, out_dir=OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════════════════
#  Step 27 — Reconstruction Examples
# ════════════════════════════════════════════════════════════════════════════

plot_reconstruction_examples(all_results, out_dir=OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════════════════
#  Step 28 — Quantitative Analysis & Interpretation
# ════════════════════════════════════════════════════════════════════════════

print_quantitative_analysis(all_results)


# ════════════════════════════════════════════════════════════════════════════
#  Step 28b — Head-to-Head Paradigm Comparison
# ════════════════════════════════════════════════════════════════════════════

paradigm_comparison(all_results, out_dir=OUTPUT_DIR)


# ════════════════════════════════════════════════════════════════════════════
#  Step 29 — Final Summary Report
# ════════════════════════════════════════════════════════════════════════════

print_final_report(all_results)


# ════════════════════════════════════════════════════════════════════════════
#  Step 30 — Download All Results
# ════════════════════════════════════════════════════════════════════════════

download_results(out_dir=OUTPUT_DIR)
