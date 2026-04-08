"""
Easy Task: Basic VAE for music feature extraction and clustering.

Pipeline
--------
1. Download FMA Small metadata (pre-extracted librosa features, ~342 MB).
2. Simulate Bangla music features (genre-realistic Gaussian distributions).
3. Merge English + Bangla -> 519-dim hybrid dataset.
4. Train a standard MLP-VAE.
5. Cluster latent space with K-Means.
6. Compare against PCA + K-Means baseline.
7. Visualise with t-SNE and UMAP.
8. Report Silhouette Score and Calinski-Harabasz Index.

Usage
-----
    python scripts/run_easy.py [--epochs 100] [--latent-dim 32] [--no-download]

All outputs are saved to results/easy/ (relative to project root).
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so src/ and config/ are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

import umap as umap_lib

from config.config import (
    BANGLA_QUERIES, BETA_DEFAULT, EPOCHS, HIDDEN_DIMS,
    KMEANS_NINIT, LATENT_DIM, LR, BATCH_SIZE, NUMPY_SEED, TORCH_SEED,
    FMA_METADATA_URL,
)
from src.data.fma import download_fma_metadata, load_fma
from src.models.mlp_vae import MLPVAE
from src.training.trainer import extract_latent, train_model
from src.visualization.plots import (
    plot_cluster_composition,
    plot_dataset_distribution,
    plot_elbow,
    plot_language_separation,
    plot_metrics_comparison,
    plot_training_curves,
    plot_tsne_umap,
)

warnings.filterwarnings("ignore")
np.random.seed(NUMPY_SEED)
torch.manual_seed(TORCH_SEED)


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAE Easy Task")
    p.add_argument("--epochs", type=int, default=EPOCHS,
                   help="Training epochs (default: %(default)s)")
    p.add_argument("--latent-dim", type=int, default=LATENT_DIM,
                   help="VAE latent dimension (default: %(default)s)")
    p.add_argument("--no-download", action="store_true",
                   help="Skip FMA metadata download (use cached data only)")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Bangla feature simulation (Easy Task only)
# -----------------------------------------------------------------------------

def simulate_bangla_features(en_mean: np.ndarray, en_std: np.ndarray, 
                             n_per_genre: int = 20) -> \
                                  tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate genre-realistic Bangla music feature vectors.

    Samples from multivariate Gaussians anchored on the English FMA
    distribution with per-genre mean shifts and std scales that reflect
    known acoustic properties of each Bangla genre.

    Parameters
    ----------
    en_mean: Feature-wise mean of English (FMA) data.
    en_std: Feature-wise std of English (FMA) data.
    n_per_genre: Samples to generate per genre.

    Returns
    -------
    (X_bangla, y_bangla, lang_bangla)
    """
    N_FEAT = len(en_mean)
    rng = np.random.default_rng(NUMPY_SEED)

    bangla_profiles = {
        "Baul": {
            "mean_shift": rng.uniform(-0.6, -0.3, N_FEAT),
            "std_scale": rng.uniform(0.5, 0.8, N_FEAT),
        },
        "Folk": {
            "mean_shift": rng.uniform(-0.2, 0.2, N_FEAT),
            "std_scale": rng.uniform(0.7, 1.0, N_FEAT),
        },
        "Rabindra": {
            "mean_shift": rng.uniform(-0.8, -0.4, N_FEAT),
            "std_scale": rng.uniform(0.4, 0.6, N_FEAT),
        },
        "ModernPop": {
            "mean_shift": rng.uniform(0.3, 0.7, N_FEAT),
            "std_scale": rng.uniform(0.9, 1.3, N_FEAT),
        },
        "Classical": {
            "mean_shift": rng.uniform(-1.0, -0.6, N_FEAT),
            "std_scale": rng.uniform(0.3, 0.5, N_FEAT),
        }
    }

    X_list, y_list, lang_list = [], [], []
    for genre, profile in bangla_profiles.items():
        shifted_mean = en_mean + profile["mean_shift"] * en_std
        genre_std = en_std * profile["std_scale"]
        samples = rng.normal(
            loc=shifted_mean, scale=genre_std,
            size=(n_per_genre, N_FEAT)
        ).astype(np.float32)
        X_list.append(samples)
        y_list.extend([genre] * n_per_genre)
        lang_list.extend(["Bangla"] * n_per_genre)

    return (
        np.vstack(X_list),
        np.array(y_list),
        np.array(lang_list)
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("  VAE EASY TASK")
    print(f"  Device: {device} | Epochs: {args.epochs} | Latent: {args.latent_dim}")
    print(f"{'='*60}\n")

    # -- Directories --------------------------------------------------------
    fma_dir = PROJECT_ROOT / "data" / "fma" / "fma_metadata"
    results_dir = PROJECT_ROOT / "results" / "easy"
    results_dir.mkdir(parents=True, exist_ok=True)

    # -- Step 1: FMA data ------------------------------------------------------
    if not args.no_download:
        download_fma_metadata(fma_dir, FMA_METADATA_URL)

    X_en, y_en = load_fma(fma_dir)
    lang_en = np.array(["English"] * len(X_en))
    X_en = np.nan_to_num(X_en, nan=0.0, posinf=0.0, neginf=0.0)

    # -- Step 2: Simulated Bangla features ----------------------------------------
    print("\n[Step 2] Generating simulated Bangla features …")
    en_mean = X_en.mean(axis=0)
    en_std = X_en.std(axis=0) + 1e-8
    X_bn, y_bn, lang_bn = simulate_bangla_features(en_mean, en_std)
    print(f"  Bangla: {X_bn.shape} | Genres: {np.unique(y_bn).tolist()}")

    # -- Step 3: Merge --------------------------------------------------------
    X_raw = np.vstack([X_en, X_bn])
    y_labels = np.concatenate([y_en, y_bn])
    lang_labels = np.concatenate([lang_en, lang_bn])
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    le_genre = LabelEncoder()
    y_genre = le_genre.fit_transform(y_labels)
    le_lang = LabelEncoder()
    y_lang = le_lang.fit_transform(lang_labels)
    K = len(le_genre.classes_)

    print(f"\n  Merged: {X_scaled.shape} | Genres={K} | K={K}")

    # Distribution plot
    plot_dataset_distribution(
        y_labels, lang_labels,
        title="Easy Task — Dataset Distribution",
        save_path=results_dir / "dataset_distribution.png"
    )

    # -- Step 4: Train VAE ---------------------------------------------------
    print("\n[Step 4] Training MLP-VAE …")
    model = MLPVAE(X_scaled.shape[1], args.latent_dim, h=HIDDEN_DIMS)
    model, history, best_loss = train_model(
        X_scaled, model,
        model_type="vae", beta=BETA_DEFAULT,
        epochs=args.epochs, lr=LR, batch_size=BATCH_SIZE,
        device=device, verbose=True
    )
    print(f"  Best loss: {best_loss:.4f}")

    plot_training_curves(
        histories={"MLP-VAE": history},
        losses={"MLP-VAE": best_loss},
        model_colors={"MLP-VAE": "#1565C0"},
        title="Easy Task — Training Curves",
        save_path=results_dir / "training_curves.png"
    )

    # -- Step 5: Extract latent codes -------------------------------------------
    print("\n[Step 5] Extracting latent representations …")
    Z = extract_latent(model, X_scaled, device=device)
    np.save(results_dir / "latent_Z.npy", Z)
    print(f"  Latent shape: {Z.shape}")

    # -- Step 6: Elbow method -------------------------------------------------
    from src.clustering.engine import elbow_analysis
    print("\n[Step 6] Elbow analysis …")
    elbow = elbow_analysis(Z, k_range=range(2, 18))
    plot_elbow(elbow, true_k=K,
               title="Easy Task - Optimal K (Elbow Method)",
               save_path=results_dir / "elbow_method.png")

    # -- Step 7: Clustering --------------------------------------------------
    print("\n[Step 7] K-Means clustering …")
    km_vae = KMeans(n_clusters=K, n_init=KMEANS_NINIT, random_state=42)
    labels_vae = km_vae.fit_predict(Z)

    pca = PCA(n_components=args.latent_dim, random_state=42)
    Z_pca = pca.fit_transform(X_scaled)
    km_pca = KMeans(n_clusters=K, n_init=KMEANS_NINIT, random_state=42)
    labels_pca = km_pca.fit_predict(Z_pca)

    sil_vae = silhouette_score(Z, labels_vae)
    sil_pca = silhouette_score(Z_pca, labels_pca)
    ch_vae = calinski_harabasz_score(Z, labels_vae)
    ch_pca = calinski_harabasz_score(Z_pca, labels_pca)

    # -- Step 8: t-SNE + UMAP visualisations ------------------------------------
    print("\n[Step 8] t-SNE …")
    Z_tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42).fit_transform(Z)
    plot_tsne_umap(
        Z_tsne, y_genre, y_lang, list(le_genre.classes_), list(le_lang.classes_),
        projection="t-SNE",
        title="Easy Task — t-SNE Latent Space",
        save_path=results_dir / "tsne_visualization.png"
    )

    print("  UMAP …")
    Z_umap = umap_lib.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42).fit_transform(Z)
    plot_tsne_umap(
        Z_umap, y_genre, y_lang, list(le_genre.classes_), list(le_lang.classes_),
        projection="UMAP",
        title="Easy Task — UMAP Latent Space",
        save_path=results_dir / "umap_visualization.png",
    )

    # -- Step 9: Metrics + report -------------------------------------------
    plot_metrics_comparison(
        metrics_dict={
            "VAE + KMeans": {"Silhouette": sil_vae, "Calinski-H": ch_vae},
            "PCA + KMeans": {"Silhouette": sil_pca, "Calinski-H": ch_pca}
        },
        metric_keys=["Silhouette", "Calinski-H"],
        metric_labels=["Silhouette Score (↑)", "Calinski-Harabasz (↑)"],
        title="Easy Task — VAE vs PCA Baseline",
        save_path=results_dir / "metrics_comparison.png"
    )

    plot_cluster_composition(
        labels_vae, y_labels,
        model_name="VAE + KMeans",
        title="Easy Task — Cluster Composition",
        save_path=results_dir / "cluster_composition.png"
    )

    # Language separation in UMAP
    from config.config import LANG_COLORS, LANG_MARKERS
    plot_language_separation(
        Z_umap, lang_labels, LANG_COLORS, LANG_MARKERS,
        title="Easy Task — Language Separation (UMAP)",
        save_path=results_dir / "language_separation.png"
    )

    # Save cluster CSV + metrics table
    df_clusters = pd.DataFrame({
        "genre": y_labels, "language": lang_labels,
        "cluster_vae": labels_vae, "cluster_pca": labels_pca
    })
    df_clusters.to_csv(results_dir / "cluster_assignments.csv", index=False)

    df_metrics = pd.DataFrame({
        "Method": ["VAE + KMeans", "PCA + KMeans"],
        "Silhouette Score": [round(sil_vae, 4), round(sil_pca, 4)],
        "Calinski-Harabasz": [round(ch_vae, 2), round(ch_pca, 2)]
    })
    df_metrics.to_csv(results_dir / "metrics_table.csv", index=False)

    # Save model checkpoint
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "input_dim": X_scaled.shape[1],
                "hidden_dims": HIDDEN_DIMS,
                "latent_dim": args.latent_dim,
                "beta": BETA_DEFAULT
            }
        },
        results_dir / "vae_easy_model.pt"
    )

    # -- Final report ------------------------------------------------------
    sep = "=" * 60
    winner_sil = "VAE" if sil_vae > sil_pca else "PCA"
    winner_ch  = "VAE" if ch_vae > ch_pca  else "PCA"
    print(f"\n{sep}")
    print("  EASY TASK — FINAL REPORT")
    print(sep)
    print(f"  Dataset   : {len(y_labels)} tracks ({(lang_labels=='English').sum()} EN + {(lang_labels=='Bangla').sum()} BN)")
    print(f"  Features  : {X_scaled.shape[1]} dims | Genres: {K}")
    print(f"  Latent dim: {args.latent_dim} | Best loss: {best_loss:.4f}")
    print()
    print(f"  {'Method':<20} {'Silhouette':>12} {'Calinski-H':>12}")
    print(f"  {'-'*46}")
    print(f"  {'VAE + KMeans':<20} {sil_vae:>12.4f} {ch_vae:>12.1f}")
    print(f"  {'PCA + KMeans':<20} {sil_pca:>12.4f} {ch_pca:>12.1f}")
    print()
    print(f"  Winner (Silhouette) : {winner_sil}")
    print(f"  Winner (CH Index)   : {winner_ch}")
    print(f"\n  Results saved to: {results_dir}")
    print(sep)


if __name__ == "__main__":
    main()
