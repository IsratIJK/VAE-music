"""
main.py
-------
Runs the Medium Task VAE multi-modal music clustering pipeline locally.
Uses data/music_dataset/ instead of downloading from GTZAN / Kaggle / yt-dlp.

Expected dataset layout
-----------------------
    data/music_dataset/
    ├── english/          # one sub-folder per genre
    │   ├── blues/
    │   │   ├── track1.wav
    │   │   └── ...
    │   ├── jazz/
    │   └── ...
    └── bangla/           # 3 songs per genre is fine
        ├── Baul/
        ├── Folk/
        └── ...

Run from any directory:
    python "src/Medium Task/main.py"

Install dependencies:
    pip install torch torchvision
    pip install librosa soundfile audioread umap-learn
    pip install scikit-learn matplotlib seaborn tqdm
    pip install lyricsgenius beautifulsoup4 requests

Small-dataset options (see config section below):
    SMALL_DATASET          = True   # lowers metric thresholds for tiny splits
    ALLOW_SINGLE_SONG_CLUSTER = False  # set True to let DBSCAN use min_samples=1
"""

import os
import sys
import random
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

MUSIC_DATASET_DIR = os.path.join(_REPO_ROOT, 'music_dataset')
GTZAN_DIR         = os.path.join(MUSIC_DATASET_DIR, 'gtzan')        # English
BANGLAGITI_DIR    = os.path.join(MUSIC_DATASET_DIR, 'banglagiti')   # Bangla
BMGCD_DIR         = os.path.join(MUSIC_DATASET_DIR, 'bmgcd')        # Bangla
OUT_DIR           = os.path.join(_HERE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Small-dataset config ──────────────────────────────────────────────────────
# Set SMALL_DATASET = True whenever any split has fewer than ~30 songs.
# With 3 songs per genre (Bangla), this MUST be True.
SMALL_DATASET             = True   # patches metric threshold & early stopping
ALLOW_SINGLE_SONG_CLUSTER = False  # set True to allow DBSCAN min_samples=1

# Minimum audio files required per genre folder to include that genre.
# Set to 1 to include genres with even a single track.
MIN_PER_GENRE = 1

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Module path ───────────────────────────────────────────────────────────────
# Ensure Python finds dataset/vae/clustering/evaluation from any working dir.
sys.path.insert(0, _HERE)

# ── Import modules & apply small-dataset patches ──────────────────────────────
import vae as _vae_mod

if SMALL_DATASET:
    # Disable early stopping: val-set of 1-2 samples is too noisy to guide it.
    # EARLY_STOP_PATIENCE is read at runtime from vae's namespace, so this works.
    _vae_mod.EARLY_STOP_PATIENCE = 999

import clustering as _cl_mod

if SMALL_DATASET:
    # Require ≥ 3 noise-free samples (instead of 10) to compute clustering metrics.
    _cl_mod.SMALL_DATASET_MIN_SAMPLES = 3

_cl_mod.ALLOW_SINGLE_SONG_CLUSTER = ALLOW_SINGLE_SONG_CLUSTER

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
    OUTPUT_DIR,
    collect_audio_from_dir,
    extract_audio_features, extract_mfcc_2d,
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

print(f'Config ready | Device: {DEVICE}')
print(f'SMALL_DATASET={SMALL_DATASET}  '
      f'ALLOW_SINGLE_SONG_CLUSTER={ALLOW_SINGLE_SONG_CLUSTER}  '
      f'MIN_PER_GENRE={MIN_PER_GENRE}')
print(f'N_MFCC_ROWS={N_MFCC_ROWS}  TIME_FRAMES={TIME_FRAMES}  '
      f'MFCC_2D_DIM={MFCC_2D_DIM}')
print(f'AUDIO_FEAT_DIM={AUDIO_FEAT_DIM}  LYRIC_DIM={LYRIC_DIM}  '
      f'LATENT_DIM={LATENT_DIM}')


# ── Local dataset loader ──────────────────────────────────────────────────────

def load_local_dataset(root_dir, language, min_per_genre=1):
    """
    Walk root_dir where each immediate sub-folder is a genre label.
    Returns (X, X_2d, y, lang, paths, records) or empty arrays if root is missing.

    Works with as few as 1 file per genre.
    """
    from collections import Counter
    from tqdm import tqdm

    if not os.path.isdir(root_dir):
        print(f'  SKIP {language}: directory not found — {root_dir}')
        return (
            np.zeros((0, AUDIO_FEAT_DIM), dtype=np.float32),
            np.zeros((0, MFCC_2D_DIM),   dtype=np.float32),
            np.array([]), np.array([]), [], [],
        )

    recs = collect_audio_from_dir(root_dir, min_per_genre=min_per_genre)
    if not recs:
        print(f'  SKIP {language}: no audio files in {root_dir}')
        return (
            np.zeros((0, AUDIO_FEAT_DIM), dtype=np.float32),
            np.zeros((0, MFCC_2D_DIM),   dtype=np.float32),
            np.array([]), np.array([]), [], [],
        )

    X, X_2d, y, lang, paths = [], [], [], [], []
    for fpath, genre in tqdm(recs, desc=f'  librosa [{language}]', leave=False):
        feat    = extract_audio_features(fpath)
        feat_2d = extract_mfcc_2d(fpath)
        if feat is not None and feat_2d is not None:
            X.append(feat);    X_2d.append(feat_2d)
            y.append(genre);   lang.append(language)
            paths.append(fpath)

    if len(X) == 0:
        print(f'  WARNING: 0 features extracted from {root_dir}')
        return (
            np.zeros((0, AUDIO_FEAT_DIM), dtype=np.float32),
            np.zeros((0, MFCC_2D_DIM),   dtype=np.float32),
            np.array([]), np.array([]), [], [],
        )

    X    = np.array(X,    dtype=np.float32)
    X_2d = np.array(X_2d, dtype=np.float32)
    y    = np.array(y)
    lang = np.array(lang)
    counts = dict(Counter(y.tolist()))
    print(f'  {language}: {X.shape[0]} samples | '
          f'Genres ({len(counts)}): {counts}')

    records = [
        {'file': p, 'genre': g, 'language': lbl}
        for p, g, lbl in zip(paths, y.tolist(), lang.tolist())
    ]
    return X, X_2d, y, lang, paths, records


# ── Step 1: Load datasets ─────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('STEP 1 — Loading local music_dataset')
print(f'  GTZAN      : {GTZAN_DIR}')
print(f'  BanglaGITI : {BANGLAGITI_DIR}')
print(f'  BMGCD      : {BMGCD_DIR}')
print('=' * 60)

X_gt, X_gt_2d, y_gt, lang_gt, paths_gt, rec_gt = load_local_dataset(
    GTZAN_DIR, 'English', min_per_genre=MIN_PER_GENRE)

X_bg, X_bg_2d, y_bg, lang_bg, paths_bg, rec_bg = load_local_dataset(
    BANGLAGITI_DIR, 'Bangla', min_per_genre=MIN_PER_GENRE)

X_bm, X_bm_2d, y_bm, lang_bm, paths_bm, rec_bm = load_local_dataset(
    BMGCD_DIR, 'Bangla', min_per_genre=MIN_PER_GENRE)

if len(X_gt) == 0 and len(X_bg) == 0 and len(X_bm) == 0:
    raise RuntimeError(
        'No audio files found.\n'
        'Place your audio files in:\n'
        f'  {GTZAN_DIR}/<genre>/*.wav      (English)\n'
        f'  {BANGLAGITI_DIR}/<genre>/*.wav (Bangla)\n'
        f'  {BMGCD_DIR}/<genre>/*.wav      (Bangla)\n'
        'Each genre must be a sub-folder; .wav and .mp3 are accepted.'
    )

# Fit a shared scaler on all available audio
from sklearn.preprocessing import StandardScaler
_parts = [X for X in [X_gt, X_bg, X_bm] if len(X) > 0]
X_all  = np.vstack(_parts).astype(np.float32)
scaler_all = StandardScaler().fit(X_all)
print(f'\nscaler_all fitted on {X_all.shape[0]} combined samples')


# ── Step 2: Full experiment pipeline ─────────────────────────────────────────
print('\n' + '=' * 60)
print('STEP 2 — Training all models (9 VAE variants + PCA + Raw)')
print('=' * 60)

all_results = {}

if len(X_gt) > 0:
    all_results['GTZAN'] = full_pipeline(
        X_gt, y_gt, lang_gt, 'GTZAN (English Multi-Genre)',
        file_paths=paths_gt, X_raw_2d=X_gt_2d, scaler=scaler_all)

if len(X_bg) > 0:
    all_results['BanglaGITI'] = full_pipeline(
        X_bg, y_bg, lang_bg, 'BanglaGITI (Bengali Songs)',
        file_paths=paths_bg, X_raw_2d=X_bg_2d, scaler=scaler_all)

if len(X_bm) > 0:
    all_results['BMGCD'] = full_pipeline(
        X_bm, y_bm, lang_bm, 'BMGCD (Bangla Music Genre)',
        file_paths=paths_bm, X_raw_2d=X_bm_2d, scaler=scaler_all)

if not all_results:
    raise RuntimeError('All datasets were empty — pipeline aborted.')

print('\nAll model experiments complete!')


# ── Step 3: Visualisation ────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('STEP 3 — Visualisation')
print('=' * 60)

plot_genre_distribution(all_results, out_dir=OUT_DIR)

print('Computing t-SNE + UMAP projections ...')
all_results = compute_projections(all_results)

plot_latent_umap(all_results, out_dir=OUT_DIR)
plot_latent_tsne(all_results, out_dir=OUT_DIR)
plot_elbow(all_results, out_dir=OUT_DIR)
plot_dbscan(all_results, out_dir=OUT_DIR)
plot_cluster_composition(all_results, out_dir=OUT_DIR)
plot_language_separation(all_results, out_dir=OUT_DIR)
plot_training_curves(all_results, out_dir=OUT_DIR)


# ── Step 4: Evaluation ───────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('STEP 4 — Evaluation & Reports')
print('=' * 60)

df_all = build_metrics_df(all_results)
print_metrics_table(df_all, out_dir=OUT_DIR)
plot_metrics_heatmap(df_all, out_dir=OUT_DIR)
plot_best_metrics_bar(all_results, out_dir=OUT_DIR)
plot_vae_vs_baseline(all_results, out_dir=OUT_DIR)
plot_disentanglement(all_results, out_dir=OUT_DIR)
plot_latent_traversal(all_results, out_dir=OUT_DIR)
plot_reconstruction_examples(all_results, out_dir=OUT_DIR)
print_quantitative_analysis(all_results)
paradigm_comparison(all_results, out_dir=OUT_DIR)
print_final_report(all_results)

download_results(out_dir=OUT_DIR)

print('\nAll outputs saved to:', OUT_DIR)
