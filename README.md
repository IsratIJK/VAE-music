# VAE-Music: Unsupervised Music Clustering via Variational Autoencoders

> GitHub: https://github.com/IsratIJK/VAE-music



## Introduction

This repository presents a comprehensive study on **Variational Autoencoders (VAEs)** for unsupervised representation learning and clustering of audio data. The project investigates how different VAE architectures and latent space enhancement techniques affect clustering performance across datasets of varying scale and complexity.

We systematically explore multiple VAE variants and evaluate their ability to learn meaningful latent representations, which are then used for clustering and analysis.



## VAE Variants

The following Variational Autoencoder architectures are implemented and evaluated:

- MLP-VAE  
- Convolutional VAE (Conv-VAE)  
- Hybrid VAE (MLP + CNN)  
- **β-VAE** (strong baseline)  
- Conditional VAE (CVAE)  
- Multimodal VAE (MM-VAE)  



## Clustering Methods

Clustering is performed on learned latent representations using multiple algorithms:

- **K-Means**  
- **DBSCAN (Density-Based Clustering)**  
- **Agglomerative Clustering**  

These methods are used to evaluate how different latent structures behave under varying assumptions of density, hierarchy, and cluster geometry.



## Advanced Techniques

To further enhance latent space quality and structure, the following techniques are explored:

 **Diffusion-based latent refinement (DDIM)**  
  Improves latent smoothness through iterative denoising in latent space.

 **Ensemble latent fusion**  
  Integrates latent representations from multiple models to form a more robust unified embedding.

 **Spectral graph refinement**  
  Uses graph-based spectral structure to improve latent separability and clustering consistency.



## Evaluation Metrics

Model performance is evaluated using standard clustering metrics:

- Silhouette Score  
- Normalized Mutual Information (NMI)  
- Adjusted Rand Index (ARI)  
- Purity  
- Davies–Bouldin Index  



## Key Findings

- Representation quality is more important than model complexity, especially in low-data regimes.  
- Simple baselines (e.g., PCA in small datasets) can sometimes outperform deep generative models.  
- **β-VAE consistently provides the most stable baseline performance.**  
- Diffusion-based refinement improves latent smoothness but may degrade clustering separability.  
- Spectral and ensemble methods improve structure but increase computational complexity.  
- DBSCAN and Agglomerative clustering capture different structural properties compared to K-Means.



## Objective

This repository provides a **modular and extensible framework** for:

- Representation learning using VAEs  
- Latent space analysis and visualization  
- Multi-algorithm clustering evaluation  
- Advanced latent refinement techniques  

It aims to support reproducible research in **audio representation learning and unsupervised clustering**.




## Repository Structure

## Tasks

## Project Tasks

This project is divided into three progressive levels of difficulty: Easy, Medium, and Hard. Each stage builds upon the previous one to systematically develop and evaluate VAE-based representation learning for music data.

---

## 🟢 Easy Task

- Implement a basic **Variational Autoencoder (VAE)** for feature extraction from music data.  
- Use a small hybrid language music dataset (English + Bangla songs).  
- Perform clustering using **K-Means** on latent features.  
- Visualize clusters using **t-SNE** or **UMAP**.  

### Baseline Comparison
- Compare VAE-based features with:
  - PCA + K-Means baseline  

### Evaluation Metrics
- Silhouette Score  
- Calinski–Harabasz Index  



## 🟡 Medium Task

- Enhance the VAE with a **convolutional architecture** for processing spectrograms or MFCC features.  
- Introduce **hybrid feature representation** combining:
  - Audio features  
  - Lyrics embeddings  

### Clustering Methods
- K-Means  
- Agglomerative Clustering  
- DBSCAN  

### Evaluation Metrics
- Silhouette Score  
- Davies–Bouldin Index  
- Adjusted Rand Index (ARI) *(if partial labels are available)*  

### Analysis Goal
- Compare clustering quality across methods  
- Analyze why VAE-based representations perform better or worse than baseline approaches  



## 🔴 Hard Task

- Implement advanced VAE variants:
  - **Conditional VAE (CVAE)**  
  - **β-VAE (disentangled representation learning)**  

- Perform **multi-modal clustering** using:
  - Audio features  
  - Lyrics embeddings  
  - Genre information  

### Evaluation Metrics
- Silhouette Score  
- Normalized Mutual Information (NMI)  
- Adjusted Rand Index (ARI)  
- Cluster Purity  

### Visualization Requirements
- Latent space visualization (t-SNE / UMAP)  
- Cluster distribution across:
  - Languages  
  - Genres  
- Reconstruction examples from latent space  

### Comparative Analysis
Compare VAE-based approaches with:

- PCA + K-Means  
- Autoencoder + K-Means  
- Direct spectral feature clustering  








## Datasets

The experiments in this project are conducted on multiple music datasets:

- **GTZAN Dataset**  
  A standard benchmark dataset for music genre classification and clustering tasks.

- **BMGCD (Bangla Music Genre Classification Dataset)**  
  A dataset specifically designed for Bangla music genre classification.

- **BanglaGITI Dataset**  
  A Bangla music dataset used for genre-based and linguistic variation analysis in music understanding tasks.

- **YouTube-based Hybrid Dataset**  
  A small-scale dataset containing both English and Bangla songs, used for multimodal and low-resource experiments.

In addition, multimodal experiments utilize:
- MFCC and spectrogram-based audio features  
- Lyrics embeddings  
- Genre and language metadata (where available)

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
