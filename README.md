# VAE-Music: Unsupervised Music Clustering via Variational Autoencoders

> GitHub: https://github.com/IsratIJK/VAE-music

An unsupervised learning pipeline that uses Variational Autoencoders (VAEs) to extract latent representations from music audio features and performs clustering to discover genre structure across a hybrid English + Bangla dataset.

## Project Overview

This project implements a full pipeline:
- Extract latent representations from audio features using multiple VAE variants
- Cluster the latent space with K-Means, Agglomerative, and DBSCAN
- Compare against PCA + K-Means and Spectral baselines using six evaluation metrics
- Analyse disentanglement, multi-modal fusion, cross-language structure, and domain adaptation

## Repository Structure

```
VAE-music/
├── config/
│   └── config.py                  # All hyperparameters, paths, and constants
├── src/
│   ├── models/
│   │   ├── base_vae.py            # Shared MLP builder + β-VAE loss
│   │   ├── mlp_vae.py             # MLP-VAE and Beta-VAE
│   │   ├── conv_vae.py            # Conv1D-VAE
│   │   ├── cvae.py                # Conditional VAE (CVAE)
│   │   ├── autoencoder.py         # Deterministic Autoencoder baseline
│   │   ├── gmvae.py               # Gaussian Mixture VAE (K-component prior)
│   │   ├── contrastive_vae.py     # InfoNCE + β-VAE with projection head
│   │   └── dann_vae.py            # Domain Adversarial VAE (gradient reversal)
│   ├── data/
│   │   ├── features.py            # Librosa audio feature extraction (57-dim)
│   │   ├── bangla.py              # yt-dlp downloader + Bangla feature extractor
│   │   ├── fma.py                 # FMA metadata downloader and loader
│   │   ├── lmd.py                 # LMD MIDI dataset loader
│   │   └── gtzan.py               # GTZAN CSV loader
│   ├── features/
│   │   └── hybrid.py              # TF-IDF lyrics + audio fusion
│   ├── clustering/
│   │   └── engine.py              # K-Means / Agglomerative / DBSCAN + 6 metrics + MIG
│   ├── training/
│   │   └── trainer.py             # Unified training loop + latent extraction
│   └── visualization/
│       └── plots.py               # All plotting functions (20+ figures)
├── scripts/
│   ├── run_easy.py                # Easy Task end-to-end script
│   ├── run_medium.py              # Medium Task end-to-end script
│   └── run_hard.py                # Hard Task end-to-end script (all extensions)
├── notebooks/
│   ├── easy_task.ipynb            # Interactive Easy Task notebook
│   ├── medium_task.ipynb          # Interactive Medium Task notebook
│   └── hard_task.ipynb            # Interactive Hard Task notebook
├── data/
│   ├── audio/                     # Downloaded audio files (gitignored)
│   └── lyrics/                    # Lyrics data (gitignored)
├── results/
│   ├── easy/                      # Easy Task plots and CSVs (committed)
│   ├── medium/                    # Medium Task plots and CSVs (committed)
│   └── hard/                      # Hard Task plots and CSVs (committed)
└── requirements.txt
```

## Tasks

### Easy Task
- MLP-VAE on FMA Small (8 000 English tracks) + 100 simulated Bangla tracks
- K-Means clustering on 32-dim latent space
- Baseline: PCA + K-Means
- Metrics: Silhouette Score, Calinski-Harabasz Index
- Visualisations: t-SNE, UMAP, elbow method, cluster composition

### Medium Task
- Three VAE architectures: MLP-VAE, Conv1D-VAE, Hybrid-VAE (audio + TF-IDF lyrics)
- Three datasets: FMA, LMD (MIDI), GTZAN — each augmented with real Bangla audio via yt-dlp
- Three clustering algorithms: K-Means, Agglomerative, DBSCAN
- Six metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz, NMI, ARI, Cluster Purity

### Hard Task
Core pipeline:
- Six VAE model variants: MLP-VAE, Beta-VAE (β=4), CVAE, Conv1D-VAE, Autoencoder, Multi-Modal VAE
- Two non-learned baselines: PCA + K-Means, Spectral embedding + K-Means
- Three datasets (FMA, LMD, GTZAN) each augmented with real Bangla audio
- Six evaluation metrics including Cluster Purity
- Full visualisation suite: t-SNE/UMAP per model, reconstruction examples, heatmaps, training curves

Advanced Extensions (unique research contributions):

| # | Extension | Description |
|---|-----------|-------------|
| Ext-1 | **GMVAE** | Gaussian Mixture VAE with K-component learned prior; soft component assignment |
| Ext-2 | **β-Sensitivity** | Beta-VAE sweep over β ∈ {0.5, 1, 2, 4, 8, 16}; reconstruction vs. clustering tradeoff |
| Ext-3 | **MIG** | Mutual Information Gap disentanglement metric (Chen et al. 2018) |
| Ext-4 | **SLERP Interpolation** | Spherical linear interpolation between genre centroids in latent space |
| Ext-5 | **Zero-Shot Transfer** | Cross-dataset transfer (FMA↔LMD↔GTZAN) with PCA alignment |
| Ext-A | **Contrastive VAE** | InfoNCE (NT-Xent / SimCLR) + β-VAE with projection head |
| Ext-B | **DANN-VAE** | Domain Adversarial VAE with gradient reversal layer (Ganin 2016) |

Final output: mega comparison heatmap of all 11 models × 4 metrics × 3 datasets + quantitative report CSV.

## Datasets

| Dataset | Tracks | Genres | Source |
|---------|--------|--------|--------|
| FMA Small | 8 000 | 8 | Pre-extracted librosa features (~342 MB) |
| LMD | ~9 000 | 6 | MIDI files (clean_midi, ~57 MB) |
| GTZAN | 1 000 | 10 | Pre-extracted CSV from GitHub |
| Bangla | ~150 | 5 | Real YouTube audio via yt-dlp |

## Installation

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
brew install ffmpeg
```

## Usage

### Command-line scripts

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
--epochs 50          # Reduce training time for testing
--no-download        # Use cached data only (skip HTTP downloads)
--latent-dim 16      # Change latent dimension
```

All outputs are written to `results/easy/`, `results/medium/`, or `results/hard/` respectively.

### Jupyter notebooks

```bash
jupyter notebook notebooks/easy_task.ipynb
jupyter notebook notebooks/medium_task.ipynb
jupyter notebook notebooks/hard_task.ipynb
```

## Model Architectures

### MLP-VAE
- **Encoder**: Linear → BatchNorm → LeakyReLU stacked (256→128→64) → μ, log σ²
- **Decoder**: Linear stack reversed → reconstruction
- **Loss**: MSE reconstruction + β × KL divergence (β=1)

### Beta-VAE
Identical to MLP-VAE with β=4.0 to encourage disentangled representations.

### CVAE (Conditional VAE)
Genre one-hot label concatenated to encoder input and decoder input.
Clustering uses unconditional encoding (zero condition vector).

### Conv1D-VAE
Feature vector treated as a 1-D signal. Conv1d encoder (1→32→64→128, stride-2) with ConvTranspose1d decoder.

### Multi-Modal VAE
Input = L2_norm(audio) ‖ L2_norm(TF-IDF lyrics PCA-32) ‖ genre one-hot. Full β-VAE objective.

### GMVAE (Gaussian Mixture VAE)
Replaces the standard N(0, I) prior with a K-component Gaussian mixture.
A learned `qy_net` produces soft component assignments; per-component (μ_k, σ_k) are trained end-to-end.
KL is approximated as an upper bound: KL(q(z|x) ‖ Σ_k q(y=k) p(z|y=k)).

### Contrastive VAE
Combines β-VAE ELBO with InfoNCE (NT-Xent / SimCLR) loss.
A projection head maps z → ℝ^64 (L2-normalised) for contrastive loss only; clustering uses z directly.
Same-genre pairs are treated as positives; cross-genre pairs within a batch as negatives.

### DANN-VAE (Domain Adversarial)
Shared encoder across FMA / LMD / GTZAN with a domain classifier head.
Gradient reversal layer (Ganin et al. 2016) flips the domain gradient sign during backprop,
forcing the encoder to learn domain-invariant representations.
λ is progressively ramped from 0 → 1 over training epochs.

## Evaluation Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| Silhouette Score | ↑ | Mean inter/intra cluster distance ratio |
| Davies-Bouldin | ↓ | Average cluster separation / compactness |
| Calinski-Harabasz | ↑ | Between-cluster / within-cluster dispersion |
| NMI | ↑ | Normalised mutual information with true labels |
| ARI | ↑ | Adjusted Rand Index (corrects for chance) |
| Cluster Purity | ↑ | Fraction of majority-class samples per cluster |
| MIG | ↑ | Mutual Information Gap — disentanglement quality |

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
LAMBDA_VALUES      = [0.1, 0.5, 1.0]                    # InfoNCE weight sweep
CONTRASTIVE_TEMPERATURE = 0.07                           # InfoNCE temperature τ
DANN_COMMON_DIM    = 32      # PCA-aligned dimension for DANN
DANN_DOMAIN_WEIGHT = 0.5     # λ_d weight on domain adversarial loss
```

## Reproducibility

All random seeds are fixed:
```python
NUMPY_SEED = 42
TORCH_SEED = 42
```

The device is selected automatically (`DEVICE_STR = "auto"`): CUDA if available, otherwise CPU.
