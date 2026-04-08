# VAE-Music: Unsupervised Music Clustering via Variational Autoencoders

> GitHub: https://github.com/IsratIJK/VAE-music

An unsupervised learning pipeline that uses Variational Autoencoders (VAEs) to extract latent representations from music audio features and performs clustering to discover genre structure across a hybrid English + Bangla dataset.

## Project Overview

This project implements a full pipeline:
- Extract latent representations from audio features using VAE variants
- Cluster the latent space with K-Means, Agglomerative, and DBSCAN
- Compare against PCA + K-Means baselines using multiple evaluation metrics
- Analyse disentanglement, multi-modal fusion, and cross-language structure

## Repository Structure

```
VAE-music/
├── config/
│   └── config.py              # All hyperparameters, paths, and constants
├── src/
│   ├── models/
│   │   ├── base_vae.py        # Shared MLP builder + β-VAE loss
│   │   ├── mlp_vae.py         # MLP-VAE and Beta-VAE
│   │   ├── conv_vae.py        # Conv1D-VAE
│   │   ├── cvae.py            # Conditional VAE (CVAE)
│   │   └── autoencoder.py     # Deterministic Autoencoder baseline
│   ├── data/
│   │   ├── features.py        # Librosa audio feature extraction (57-dim)
│   │   ├── bangla.py          # yt-dlp downloader + Bangla feature extractor
│   │   ├── fma.py             # FMA metadata downloader and loader
│   │   ├── lmd.py             # LMD MIDI dataset loader
│   │   └── gtzan.py           # GTZAN CSV loader
│   ├── features/
│   │   └── hybrid.py          # TF-IDF lyrics + audio fusion
│   ├── clustering/
│   │   └── engine.py          # K-Means / Agglomerative / DBSCAN + 6 metrics
│   ├── training/
│   │   └── trainer.py         # Unified training loop + latent extraction
│   └── visualization/
│       └── plots.py           # All plotting functions
├── scripts/
│   ├── run_easy.py            # Easy Task end-to-end script
│   ├── run_medium.py          # Medium Task end-to-end script
│   └── run_hard.py            # Hard Task end-to-end script
├── notebooks/
│   ├── easy_task.ipynb        # Interactive Easy Task notebook
│   ├── medium_task.ipynb      # Interactive Medium Task notebook
│   └── hard_task.ipynb        # Interactive Hard Task notebook
├── data/
│   ├── audio/                 # Downloaded audio files (gitignored)
│   └── lyrics/                # Lyrics data (gitignored)
├── results/                   # Generated plots and CSVs (gitignored)
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
- Three datasets: FMA, LMD (MIDI), GTZAN — each with real Bangla audio via yt-dlp
- Three clustering algorithms: K-Means, Agglomerative, DBSCAN
- Five metrics: Silhouette, Davies-Bouldin, Calinski-H, ARI, NMI

### Hard Task
- Six model variants: MLP-VAE, Beta-VAE (β=4), CVAE, Conv1D-VAE, Autoencoder, Multi-Modal VAE
- Eight latent methods including PCA and Spectral baselines
- Six metrics including Cluster Purity
- Beta-VAE disentanglement analysis
- Reconstruction visualisation from VAE latent space

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
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

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

# Hard Task (6 VAE variants + 2 baselines × 3 datasets × 3 algorithms)
python scripts/run_hard.py

# Common flags
--epochs 50          # Reduce training time for testing
--no-download        # Use cached data only (skip HTTP downloads)
--latent-dim 16      # Change latent dimension
```

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
- **Loss**: MSE reconstruction + β × KL divergence

### Beta-VAE
Identical to MLP-VAE with β=4.0 to encourage disentangled representations.

### CVAE
Genre one-hot label concatenated to encoder input and decoder input.
Clustering uses unconditional encoding (zero condition vector).

### Conv1D-VAE
Feature vector treated as 1-D signal. Conv1d encoder (1→32→64→128, stride-2) with ConvTranspose1d decoder.

### Multi-Modal VAE
Input = L2_norm(audio) ‖ L2_norm(TF-IDF lyrics PCA-32) ‖ genre one-hot

## Evaluation Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| Silhouette Score | ↑ | Mean inter/intra cluster distance ratio |
| Davies-Bouldin | ↓ | Average cluster separation / compactness |
| Calinski-Harabasz | ↑ | Between-cluster / within-cluster dispersion |
| NMI | ↑ | Normalised mutual information with true labels |
| ARI | ↑ | Adjusted Rand Index (corrects for chance) |
| Cluster Purity | ↑ | Fraction of majority-class samples per cluster |

## Configuration

All hyperparameters are in `config/config.py`. Key settings:

```python
LATENT_DIM         = 32      # VAE latent space dimension
EPOCHS             = 100     # Training epochs
LR                 = 1e-3    # AdamW learning rate
BETA_VAE_B         = 4.0     # Beta-VAE KL weight
N_BANGLA_PER_GENRE = 30      # Bangla tracks per genre (yt-dlp)
```

## Reproducibility

All random seeds are fixed via `config/config.py`:
```python
NUMPY_SEED = 42
TORCH_SEED = 42
```