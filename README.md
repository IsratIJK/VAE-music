# VAE-Music: Unsupervised Music Clustering via Variational Autoencoders

> GitHub: https://github.com/IsratIJK/VAE-music



## Introduction

This repository presents a comprehensive study on **Variational Autoencoders (VAEs)** for unsupervised representation learning and clustering of audio data. The project investigates how different VAE architectures and latent space enhancement techniques affect clustering performance across datasets of varying scale and complexity.

We systematically explore multiple VAE variants and evaluate their ability to learn meaningful latent representations, which are then used for clustering and analysis.

![Unified Pipeline](README%20image%20files/unified_pipeline.png)



## Repository Structure

```
VAE-music/
├── notebooks/
│   ├── easy_task.ipynb          # Basic MLP-VAE on small hybrid EN+BN dataset
│   ├── medium_task.ipynb        # Multi-architecture VAE on GTZAN + Bangla datasets
│   └── hard_task.ipynb          # Full evaluation pipeline with paradigm comparison
│
├── src/
│   ├── Easy Task/
│   │   ├── vae.py               # MLP-VAE encoder/decoder + training loop
│   │   ├── dataset.py           # 102-dim audio feature extraction + yt-dlp download
│   │   ├── clustering.py        # K-Means, PCA baseline, elbow method
│   │   └── evaluation.py        # Silhouette, Calinski-Harabasz, t-SNE/UMAP plots
│   │
│   ├── Medium Task/
│   │   ├── vae.py               # All 8 architectures + unified training engine
│   │   ├── dataset.py           # 65-dim + 7680-dim MFCC, real lyrics pipeline
│   │   ├── clustering.py        # KMeans + Agglomerative + DBSCAN, full pipeline
│   │   └── evaluation.py        # 6-metric evaluation, heatmaps, paradigm comparison
│   │
│   ├── Hard Task/
│   │   ├── vae.py               # Same architectures as Medium Task
│   │   ├── dataset.py           # Same pipeline as Medium Task
│   │   ├── clustering.py        # Same pipeline as Medium Task
│   │   ├── evaluation.py        # Disentanglement, latent traversal, reconstruction
│   │   └── main.py              # Top-level runner for all 30 steps
│   │
│   └── data/
│       ├── gtzan.py             # GTZAN dataset loader
│       ├── features.py          # Shared feature extraction utilities
│       ├── fma.py               # FMA dataset loader
│       ├── lmd.py               # LMD dataset loader
│       └── bangla.py            # Bangla dataset loader
│
├── scripts/
│   ├── run_easy.py              # CLI entry point for Easy Task
│   ├── run_medium.py            # CLI entry point for Medium Task
│   └── run_hard.py              # CLI entry point for Hard Task
│
├── config/
│   └── config.py                # Centralised hyperparameters
│
├── data/
│   ├── audio/                   # Audio files (git-ignored)
│   └── lyrics/                  # Lyrics cache (git-ignored)
│
├── results/                     # Output plots, CSVs, model checkpoints
└── requirements.txt
```



## Project Tasks

This project is divided into three progressive levels of difficulty. Each stage builds upon the previous to systematically develop and evaluate VAE-based representation learning for music data.

---

## Easy Task

**Goal:** Establish a proof-of-concept VAE pipeline on a small hybrid English + Bangla dataset.

### Data
- ~92 English tracks across 7 genres (Rock, Pop, Jazz, Classical, HipHop, Blues, Country), downloaded via yt-dlp (30s clips)
- Bangla tracks downloaded via yt-dlp (limited availability in runtime)
- Audio-only mode (no lyrics in this task)

### Feature Extraction
- **102-dim** flat audio vector per track:
  - MFCCs ×40 | MFCC-Δ ×13 | Chroma STFT ×12 | Mel-spectrogram stats ×8
  - Tonnetz ×6 | Spectral features ×9 | Rhythm ×3 | ZCR + RMS ×2 | Chroma CQT ×9

### VAE Architecture
- `Encoder`: Linear → BatchNorm → LeakyReLU stacked (256 → 128) → μ, log σ²
- `Decoder`: Reversed stack → reconstruction
- **Latent dim:** 16 | **β:** 1.0 | **Epochs:** 100 | **Optimizer:** AdamW + CosineAnnealingLR

### Clustering
- K-Means on VAE latent space (K = number of genres)
- Baseline: PCA + K-Means (16 components, 76.2% variance explained)
- Optimal K selected via elbow method (inertia sweep over K ∈ [2, 15])

### Visualization
- t-SNE and UMAP of latent space coloured by genre and language

### Evaluation Metrics
- Silhouette Score
- Calinski–Harabasz Index

### Running the Easy Task

#### Prerequisites

```bash
pip install torch librosa umap-learn scikit-learn pandas matplotlib tqdm yt-dlp
brew install ffmpeg       # macOS
# sudo apt install ffmpeg # Linux
```

#### Option A: Download the dataset in person (via yt-dlp)

The pipeline can automatically download ~20 WAV clips per genre from YouTube.

```bash
cd "src/Easy Task"
python main.py
```

> **If you get `HTTP Error 403: Forbidden`**, YouTube is blocking the automated download.
> Fix it by:
> 1. Updating yt-dlp: `pip install -U yt-dlp`
> 2. Making sure you are **logged into YouTube in Chrome**, then re-running — the script
>    passes `--cookies-from-browser chrome` so yt-dlp uses your session cookies to bypass the block.
>    If you use a different browser, edit the two `--cookies-from-browser` lines in
>    `src/Easy Task/dataset.py` (one in `download_genre_yt`, one in `download_bangla_genre`)
>    and change `chrome` to `firefox` or `safari`.

#### Option B: Use a pre-downloaded dataset (recommended)

If you already have audio files organised as:

```
data/VAE_Music_Dataset/
├── english/
│   ├── Rock/        *.wav / *.mp3
│   ├── Pop/
│   ├── Jazz/
│   ├── Classical/
│   ├── HipHop/
│   ├── Blues/
│   └── Country/
└── bangla/
    ├── Baul/
    ├── Folk/
    ├── Rabindra/
    ├── ModernPop/
    └── Classical/
```

The `main.py` already points to this location. Just run:

```bash
cd "src/Easy Task"
python main.py
```

No downloads will be triggered. The script goes straight to feature extraction using whatever audio files are present. Empty genre folders are skipped automatically.

#### Outputs

All plots, the model checkpoint, and CSVs are saved to `src/Easy Task/outputs/`:

| File | Description |
|---|---|
| `dataset_distribution.png` | Track count by language and genre |
| `training_curves.png` | VAE total / reconstruction / KL loss over epochs |
| `elbow_method.png` | Inertia vs K. Use this to pick the best K |
| `tsne_visualization.png` | t-SNE of latent space coloured by genre and language |
| `umap_visualization.png` | UMAP of latent space coloured by genre and language |
| `metrics_comparison.png` | VAE vs PCA baseline (Silhouette + Calinski-Harabasz) |
| `cluster_composition.png` | Language breakdown per cluster |
| `vae_music_model.pt` | Saved model checkpoint |
| `cluster_assignments.csv` | Per-track cluster labels |
| `metrics_table.csv` | Numeric evaluation results |

---

## Medium Task

**Goal:** Scale up to multiple VAE architectures and three real-world datasets with hybrid audio + lyrics features.

### Datasets
| Dataset | Language | Genres | Source |
|---|---|---|---|
| GTZAN | English | 10 (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) | HTTP mirror / Kaggle |
| BanglaGITI | Bangla | 5 (Baul, Folk, Rabindra, ModernPop, Classical) | Kaggle / yt-dlp fallback |
| BMGCD | Bangla | 5 (Adhunik, Baul, Classical, Folk, Rabindra) | Kaggle / yt-dlp fallback |

### Feature Extraction
- **65-dim flat audio vector** per track (MFCC mean+std ×40, Chroma ×12, Spectral ×5, Tempo ×1, Contrast ×7)
- **7680-dim delta-stacked MFCC spectrogram** (20 MFCC + Δ + Δ² stacked, 128 time frames) for Conv2D models
- **Lyrics** fetched via Genius API (English) and gaanesuno.com scraper (Bangla), embedded as TF-IDF + LSA (128-dim)
- Multi-modal input = L2-norm(audio) ‖ L2-norm(lyric embeddings)

### VAE Architectures (9 models)

| Model | Input | Description |
|---|---|---|
| MLP-VAE | 65-dim | 3-layer MLP encoder/decoder (256→128→64) |
| Conv2D-VAE | 7680-dim | Conv2D on delta-stacked MFCC (60×128) |
| Hybrid-Conv-VAE | 7680+128-dim | Conv2D spectrogram + lyric fusion |
| Hybrid-MLP-VAE | 65+128-dim | MLP on L2(audio) ‖ L2(lyrics) |
| Beta-VAE | 65-dim | Deeper MLP (512→256→128→64), β swept over [0.5, 1, 2, 4, 8, 16] |
| CVAE | 65+genre-dim | Genre one-hot concatenated to encoder and decoder input |
| Conv1D-VAE | 65-dim | 1D convolution treating feature vector as a signal |
| Autoencoder | 65-dim | Deterministic baseline (no KL term) |
| MultiModalVAE | 65+128-dim | Dedicated audio + lyric projection heads, reconstructs audio only |

Plus **PCA baseline** and **Raw-Spectral** (K-Means directly on 65-dim features).

### Clustering Algorithms
- **K-Means** (n_init=20)
- **Agglomerative: Ward linkage**
- **Agglomerative: Complete linkage**
- **DBSCAN** with automatic ε tuning via k-NN percentile sweep (minimises |found clusters − K| while keeping noise < 30%)

### Evaluation Metrics (6)
- Silhouette Score ↑
- Davies–Bouldin Index ↓
- Calinski–Harabasz Index ↑
- Adjusted Rand Index (ARI) ↑
- Normalized Mutual Information (NMI) ↑
- Cluster Purity ↑

### Visualizations
- UMAP + t-SNE for all 11 feature spaces (genre and language colouring)
- Elbow plots (inertia, silhouette, CH vs K)
- DBSCAN cluster analysis (noise points highlighted)
- Cluster composition heatmaps (genre % per cluster)
- English vs Bangla language separation plots
- Training loss curves per model per dataset

### Running the Medium Task

#### Prerequisites

```bash
pip install torch torchvision
pip install librosa soundfile audioread
pip install umap-learn scikit-learn matplotlib seaborn
pip install tqdm lyricsgenius beautifulsoup4 requests
brew install ffmpeg       # macOS
# sudo apt install ffmpeg # Linux
```

#### Dataset layout

Place audio files under `music_dataset/` at the repo root, organised as:

```
music_dataset/
├── gtzan/                    # English — one sub-folder per genre
│   ├── blues/
│   │   ├── blues.00001.wav
│   │   └── ...
│   ├── classical/
│   ├── country/
│   ├── disco/
│   ├── hiphop/
│   ├── jazz/
│   ├── metal/
│   ├── pop/
│   ├── reggae/
│   └── rock/
├── banglagiti/               # Bangla — 3 .wav files per genre is fine
│   ├── Baul/
│   ├── Classical/
│   ├── Folk/
│   ├── Modern_Pop/
│   └── Rabindra_Sangeet/
└── bmgcd/                    # Bangla — 3 .wav files per genre is fine
    ├── Adhunik/
    ├── Baul/
    ├── Classical/
    ├── Folk/
    └── Rabindra/
```

`.wav` and `.mp3` are both accepted. Genres with fewer files than `MIN_PER_GENRE` (default `1`) are skipped automatically.

#### Run

```bash
cd "src/Medium Task"
python main.py
```

#### Small-dataset mode

The Bangla datasets ship with only **3 songs per genre**, which is too small for the default thresholds. Two flags at the top of `main.py` handle this automatically:

```python
SMALL_DATASET             = True   # lowers metric minimum from 10 → 3 samples
                                   # disables early stopping (noisy 2-sample val set)
ALLOW_SINGLE_SONG_CLUSTER = False  # set True to allow DBSCAN min_samples = 1
```

`SMALL_DATASET = True` does two things under the hood:
- Patches `vae.EARLY_STOP_PATIENCE = 999`. Itprevents early stopping from misfiring on the tiny 2-sample validation split that results from a 90/10 split of ~15 tracks.
- Patches `clustering.SMALL_DATASET_MIN_SAMPLES = 3`. It allows clustering metrics to be computed when as few as 3 noise-free samples are present (instead of the default 10).

If you want DBSCAN to assign every point to a cluster (no noise points), set `ALLOW_SINGLE_SONG_CLUSTER = True`. This changes the DBSCAN `min_samples` parameter from `max(3, N/(K×10))` to `1`.

#### Lyrics fetching

The pipeline automatically fetches lyrics to build multi-modal features:
- **English (GTZAN) :** GTZAN files have numeric names (`blues.00001.wav`), so lyrics lookup is skipped; all tracks receive a neutral fallback embedding.
- **Bangla (BanglaGITI / BMGCD) :** The script scrapes [gaanesuno.com](https://gaanesuno.com) using the track filename as a search query. If a file is named by YouTube video ID (e.g. `u9UpVidGgik.wav`), the lookup will fail and a neutral fallback is used. This is handled gracefully with no crash.

To use the Genius API for English lyrics, set the `GENIUS_TOKEN` environment variable:
```bash
export GENIUS_TOKEN=your_token_here
```

#### Outputs

All plots, CSVs, and the zip archive are saved to `src/Medium Task/outputs/`:

| File | Description |
|---|---|
| `genre_distribution.png` | Track count per genre for each dataset |
| `latent_all_<dataset>.png` | UMAP of all 11 feature spaces (genre + language) |
| `latent_tsne_<dataset>.png` | t-SNE of all 11 feature spaces |
| `elbow_plots.png` | Inertia / Silhouette / CH vs K per dataset |
| `dbscan_analysis.png` | DBSCAN clusters with noise points highlighted |
| `cluster_composition_<dataset>.png` | Genre % heatmap per KMeans cluster |
| `language_separation.png` | English vs Bangla separation in UMAP space |
| `training_curves.png` | Training loss per model per dataset |
| `metrics_heatmap.png` | 6-metric heatmap: all feature spaces × all algorithms |
| `best_metrics_bar.png` | Best Silhouette / NMI / ARI / Purity per feature space |
| `vae_vs_baseline.png` | ΔVAE − PCA for Silhouette and NMI |
| `disentangle_<dataset>.png` | Per-dimension latent histograms (MLP / Beta / CVAE) |
| `latent_traversal_<model>.png` | Latent traversal across 5 dims × 7 steps |
| `reconstruction_<dataset>.png` | Original vs reconstructed feature vectors |
| `paradigm_comparison_bar.png` | Best-VAE vs PCA vs AE vs Direct Spectral (6 metrics) |
| `paradigm_radar.png` | Normalised radar chart across all paradigms |
| `full_metrics.csv` | 11 feature spaces × 4 algorithms × 6 metrics |
| `paradigm_comparison.csv` | Head-to-head paradigm results |
| `vae_combined_results.zip` | All of the above zipped for easy download |

---

## Hard Task

**Goal:** Full evaluation and interpretability pipeline. It extends the Medium Task with in-depth comparative analysis, disentanglement study, and paradigm comparison.

### Additional Analyses

**Disentanglement analysis**
Compares latent dimension distributions across MLP-VAE, Beta-VAE (best β), and CVAE. Plots per-dimension histograms per genre class; reports variance entropy (lower = more axis-aligned, more disentangled).

**Latent traversal**
Traverses 5 latent dimensions across z ∈ [−3, +3] (7 steps each) for MLP-VAE, Beta-VAE, and CVAE. Shows reconstructed audio feature profiles to expose which dimensions control which audio attributes.

**Reconstruction examples**
Side-by-side original vs. reconstructed feature vectors (65-dim) for MLP-VAE, Beta-VAE, and CVAE on 5 randomly sampled tracks per dataset.

**Head-to-head paradigm comparison**
Compares four approaches on all 6 metrics:

| Paradigm | Description |
|---|---|
| Best-VAE | Best-performing VAE variant per dataset (selected by Silhouette) |
| PCA + K-Means | Linear dimensionality reduction baseline |
| Autoencoder + K-Means | Non-linear compression without KL regularisation |
| Direct Spectral | K-Means on raw 65-dim audio features |

Outputs: bar charts, ranked summary table, normalised radar chart.

**Full metrics table**
11 feature spaces × 4 algorithms × 6 metrics per dataset, exported to `full_metrics.csv`.

### Running the Hard Task

The Hard Task uses the **exact same setup** as the Medium Task. Same dataset layout, prerequisites, small-dataset mode, and lyrics-fetching behaviour. See [Running the Medium Task](#running-the-medium-task) for all of that.

The only difference is the entry point and output directory:

```bash
python "src/Hard Task/main.py"
```

Outputs are saved to `src/Hard Task/outputs/`. Same file set as the Medium Task, with the addition of:

| File | Description |
|---|---|
| `disentangle_<dataset>.png` | Per-dimension latent histograms for MLP-VAE, Beta-VAE, CVAE |
| `latent_traversal_mlp.png` | MLP-VAE latent traversal (5 dims × 7 steps) |
| `latent_traversal_beta.png` | Beta-VAE latent traversal |
| `latent_traversal_cvae.png` | CVAE latent traversal |
| `reconstruction_<dataset>.png` | Original vs reconstructed features (5 tracks × 3 models) |
| `paradigm_comparison_bar.png` | Best-VAE vs PCA vs AE vs Direct Spectral (6 metrics) |
| `paradigm_radar.png` | Normalised radar chart across all paradigms |
| `paradigm_comparison.csv` | Head-to-head paradigm results |

---

## VAE Variants

The following Variational Autoencoder architectures are implemented and evaluated:

- **MLP-VAE :** 3-layer MLP encoder/decoder
- **Beta-VAE :** deeper MLP with β-sweep [0.5, 1, 2, 4, 8, 16] for disentangled representations
- **CVAE :** Conditional VAE; genre one-hot label concatenated to encoder and decoder inputs
- **Conv1D-VAE :** Treats the 65-dim feature vector as a 1-D signal; Conv1d encoder + ConvTranspose1d decoder
- **Conv2D-VAE :** Full 2D convolutional VAE on delta-stacked MFCC spectrograms (60 × 128)
- **Hybrid-Conv-VAE :** End-to-end Conv2D encoder fused with lyric projection head
- **Hybrid-MLP-VAE :** MLP-VAE on L2-normalised audio ‖ lyric concatenation
- **MultiModalVAE :** Dedicated audio + lyric projection branches; reconstructs audio only
- **Autoencoder :** Deterministic baseline (no KL divergence)



## Clustering Methods

Clustering is performed on learned latent representations using multiple algorithms:

- **K-Means** (n_init=20, seed=42)
- **Agglomerative Clustering (Ward linkage)**
- **Agglomerative Clustering (Complete linkage)**
- **DBSCAN** ε auto-tuned via k-NN distance percentile sweep on L2-normalised latent space

These methods evaluate how different latent structures behave under varying assumptions of density, hierarchy, and cluster geometry.



## Evaluation Metrics

Model performance is evaluated using six standard clustering metrics:

- **Silhouette Score** ↑ : intra-cluster cohesion vs inter-cluster separation
- **Davies–Bouldin Index** ↓ : average similarity of each cluster to its most similar neighbour
- **Calinski–Harabasz Index** ↑ : ratio of between-cluster to within-cluster dispersion
- **Normalized Mutual Information (NMI)** ↑ : symmetric, corrects for cluster size
- **Adjusted Rand Index (ARI)** ↑ : corrects for chance label agreement
- **Cluster Purity** ↑ : fraction of samples that belong to the majority genre in their cluster



## Key Findings

- **Conv2D-VAE** captures local time-frequency correlations in delta-stacked MFCCs that flat MLP-VAE misses, improving clustering on larger datasets.
- **Hybrid-Conv-VAE** (Conv2D + lyrics) is the strongest multi-modal variant when real lyrics are available.
- **Beta-VAE** produces more axis-aligned latent dimensions (lower variance entropy), aiding interpretability. Best β varies by dataset.
- **CVAE** produces genre-aware latent structure useful for conditional generation; clustering uses unconditional encoding (zero condition vector).
- **MultiModalVAE** improves NMI and Purity when lyrics are informative; falls back gracefully on neutral text.
- In small or low-diversity datasets, simple baselines (PCA + K-Means) can match or outperform deep generative models due to limited training data.
- DBSCAN and Agglomerative clustering capture different structural properties compared to K-Means; Ward linkage generally performs best among hierarchical methods.
- Lyrics coverage is limited for GTZAN (numeric filenames prevent title lookup) but adds signal for BanglaGITI and BMGCD via gaanesuno.com scraping.



## Datasets

The experiments in this project are conducted on multiple music datasets:

- **GTZAN Dataset**
  Standard benchmark for music genre classification. 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock), 1000 tracks total. Downloaded via HTTP mirrors or Kaggle.

- **BanglaGITI Dataset**
  Bangla music genre dataset from Kaggle (`priyanjanasarkar/banglagiti`). Falls back to yt-dlp download (Rabindra Sangeet, Baul, Folk, Modern Pop, Classical) if Kaggle is unavailable.

- **BMGCD (Bangla Music Genre Classification Dataset)**
  Bangla genre dataset from Kaggle (`mdimranhassan/bangla-music-genre-classification`). Falls back to yt-dlp (Adhunik, Baul, Classical, Folk, Rabindra) if Kaggle is unavailable.

- **YouTube-based Hybrid Dataset (Easy Task)**
  Small-scale dataset of English (7 genres × 20 tracks) and Bangla songs downloaded via yt-dlp, used in the Easy Task experiments.

In addition, multimodal experiments utilise:
- MFCC and spectrogram-based audio features
- Lyrics embeddings (Genius API for English; gaanesuno.com for Bangla)
- Genre and language metadata (where available)

## A Quick Information about the Models

### MLP-VAE
- **Encoder**: Linear → BatchNorm → LeakyReLU stacked (256→128→64) → μ, log σ²
- **Decoder**: Linear stack reversed → reconstruction
- **Loss**: MSE reconstruction + β × KL divergence (β=1)

### Beta-VAE
Deeper MLP encoder (512→256→128→64) with tighter log-variance clamping (±4) for stable high-β training. β swept over [0.5, 1, 2, 4, 8, 16]; best β selected by silhouette score.

### CVAE (Conditional VAE)
Genre one-hot label concatenated to encoder input and decoder input.
Clustering uses unconditional encoding (zero condition vector).

### Conv1D-VAE
Feature vector treated as a 1-D signal. Conv1d encoder (1→32→64→128, stride-2) with ConvTranspose1d decoder.

### Conv2D-VAE
Input: delta-stacked MFCC spectrogram (60 × 128). Conv2d encoder (1→32→64→128, stride-2, kernel 3×3) → μ/log σ² → ConvTranspose2d decoder with AdaptiveAvgPool2d output alignment.

### Hybrid-Conv-VAE
Conv2D encoder on the spectrogram branch fused with a lyric projection head (Linear → LayerNorm → LeakyReLU). Fusion layer combines both branches before producing μ/log σ². Input: flattened (7680 + 128 dims).

### Hybrid-MLP-VAE
Standard MLP-VAE operating on L2-normalised audio ‖ L2-normalised lyric embedding (65 + 128 = 193 dims).

### Multi-Modal VAE
Dedicated `audio_proj` and `lyric_proj` heads (Linear → LayerNorm → ReLU) project both modalities to `fusion_dim=256` each. Concatenated → shared MLP encoder → μ/log σ². Reconstructs audio only. Full β-VAE objective.

### Autoencoder (Deterministic Baseline)
Identical MLP topology to MLP-VAE but no reparameterisation or KL term. Trained with MSE reconstruction loss only.



## Configuration

All hyperparameters are centralised in `config/config.py`. Key settings:

```python
# Core VAE
LATENT_DIM         = 32      # VAE latent space dimension
EPOCHS             = 100     # Training epochs
LR                 = 1e-3    # AdamW learning rate
BETA_DEFAULT       = 1.0     # β for standard VAE
BETA_VAE_B         = 4.0     # β for Beta-VAE

# Bangla data
N_BANGLA_PER_GENRE = 30      # Bangla tracks per genre (yt-dlp)

# Advanced extensions
BETA_VALUES        = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]   # β-sweep
SWEEP_EPOCHS       = 60      # Reduced epochs for β-sweep
N_INTERP           = 12      # SLERP interpolation steps

```

## Reproducibility

All random seeds are fixed:
```python
NUMPY_SEED = 42
TORCH_SEED = 42
```

The device is selected automatically (`DEVICE_STR = "auto"`): CUDA if available, otherwise CPU.







