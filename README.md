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
- **Agglomerative — Ward linkage**
- **Agglomerative — Complete linkage**
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

---

## Hard Task

**Goal:** Full evaluation and interpretability pipeline — extends the Medium Task with in-depth comparative analysis, disentanglement study, and paradigm comparison.

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



## VAE Variants

The following Variational Autoencoder architectures are implemented and evaluated:

- **MLP-VAE** — 3-layer MLP encoder/decoder
- **Beta-VAE** — deeper MLP with β-sweep [0.5, 1, 2, 4, 8, 16] for disentangled representations
- **CVAE** — Conditional VAE; genre one-hot label concatenated to encoder and decoder inputs
- **Conv1D-VAE** — Treats the 65-dim feature vector as a 1-D signal; Conv1d encoder + ConvTranspose1d decoder
- **Conv2D-VAE** — Full 2D convolutional VAE on delta-stacked MFCC spectrograms (60 × 128)
- **Hybrid-Conv-VAE** — End-to-end Conv2D encoder fused with lyric projection head
- **Hybrid-MLP-VAE** — MLP-VAE on L2-normalised audio ‖ lyric concatenation
- **MultiModalVAE** — Dedicated audio + lyric projection branches; reconstructs audio only
- **Autoencoder** — Deterministic baseline (no KL divergence)



## Clustering Methods

Clustering is performed on learned latent representations using multiple algorithms:

- **K-Means** (n_init=20, seed=42)
- **Agglomerative Clustering — Ward linkage**
- **Agglomerative Clustering — Complete linkage**
- **DBSCAN** — ε auto-tuned via k-NN distance percentile sweep on L2-normalised latent space

These methods evaluate how different latent structures behave under varying assumptions of density, hierarchy, and cluster geometry.



## Evaluation Metrics

Model performance is evaluated using six standard clustering metrics:

- **Silhouette Score** ↑ — intra-cluster cohesion vs inter-cluster separation
- **Davies–Bouldin Index** ↓ — average similarity of each cluster to its most similar neighbour
- **Calinski–Harabasz Index** ↑ — ratio of between-cluster to within-cluster dispersion
- **Normalized Mutual Information (NMI)** ↑ — symmetric, corrects for cluster size
- **Adjusted Rand Index (ARI)** ↑ — corrects for chance label agreement
- **Cluster Purity** ↑ — fraction of samples that belong to the majority genre in their cluster



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

<!-- ## Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv            # Linux/Mac
python -m venv .venv             # Windows

source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# For GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For Bangla audio download you also need `ffmpeg`:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg
# macOS
brew install ffmpeg -->
```
<!-- 
## Usage -->

<!-- ### Command-line scripts

```bash
# Easy Task (MLP-VAE, FMA + simulated Bangla)
python scripts/run_easy.py

# Medium Task (3 VAE variants × 3 datasets × 3 algorithms)
python scripts/run_medium.py

# Hard Task — full pipeline including all 7 advanced extensions
python scripts/run_hard.py

# Hard Task — skip advanced extensions for faster testing
python scripts/run_hard.py --no-extensions

# Common flags (all scripts)
--epochs 100          # Reduce training time for testing
--no-download        # Use cached data only (skip HTTP downloads)
--latent-dim 64      # Change latent dimension
```

All outputs are written to `results/easy/`, `results/medium/`, or `results/hard/` respectively. -->

<!-- ### Jupyter notebooks

```bash
jupyter notebook notebooks/easy_task.ipynb
jupyter notebook notebooks/medium_task.ipynb
jupyter notebook notebooks/hard_task.ipynb
``` -->

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
