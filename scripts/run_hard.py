"""
Hard Task: Advanced multi-modal VAE music clustering.

Pipeline
--------
1. Download FMA metadata, LMD (MIDI), GTZAN CSV.
2. Download real Bangla audio via yt-dlp + extract librosa features.
3. For each dataset train 6 model variants:
   - MLP-VAE        (β = 1)
   - Beta-VAE       (β = 4)
   - CVAE           (genre-conditioned)
   - Conv1D-VAE
   - Autoencoder    (deterministic baseline)
   - Multi-modal VAE (audio + TF-IDF lyrics + genre one-hot)
4. Two non-learned baselines:
   - PCA + KMeans
   - Spectral (raw scaled features + KMeans)
5. Run KMeans / Agglomerative / DBSCAN on every latent space.
6. Compute 6 metrics: Silhouette, Davies-Bouldin, CH, NMI, ARI, Purity.
7. Produce all visualisations:
   - Latent space per model (t-SNE + UMAP)
   - Cluster distribution (language stacked bar per cluster)
   - Beta-VAE disentanglement histograms
   - Reconstruction examples
   - Language separation
   - Full metrics heatmap + bar charts
   - Training loss curves
   - Quantitative analysis report

Usage
-----
    python scripts/run_hard.py [--epochs 100] [--no-download]

All outputs saved to results/hard/ (relative to project root).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings

import numpy as np
import pandas as pd
import torch
import umap as umap_lib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config.config import (
    BANGLA_DIR, BANGLA_QUERIES, BATCH_SIZE, BETA_VAE_B,
    EPOCHS, FMA_METADATA_URL, GENRE_VOCAB, GTZAN_URLS,
    HIDDEN_DIMS, KMEANS_NINIT, LANG_COLORS, LANG_MARKERS,
    LATENT_DIM, LMD_URL, LR, LYRIC_DIM, MODEL_COLORS,
    N_BANGLA_PER_GENRE, NUMPY_SEED, TORCH_SEED,
)
from src.clustering.engine import run_clustering
from src.data.bangla import get_bangla
from src.data.fma import download_fma_metadata, load_fma
from src.data.gtzan import download_gtzan_csv, load_gtzan
from src.data.lmd import download_lmd, load_lmd
from src.features.hybrid import make_genre_onehot, make_multimodal
from src.models.autoencoder import Autoencoder
from src.models.conv_vae import ConvVAE
from src.models.cvae import CVAE
from src.models.mlp_vae import BetaVAE, MLPVAE
from src.training.trainer import extract_latent, train_model
from src.visualization.plots import (
    plot_cluster_composition,
    plot_disentanglement,
    plot_language_separation,
    plot_metrics_heatmap,
    plot_reconstruction,
    plot_training_curves,
    plot_tsne_umap
)

warnings.filterwarnings("ignore")
np.random.seed(NUMPY_SEED)
torch.manual_seed(TORCH_SEED)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAE Hard Task")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    p.add_argument("--beta", type=float, default=BETA_VAE_B,
                   help="β value for Beta-VAE (default: %(default)s)")
    p.add_argument("--no-download", action="store_true")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Full hard-task pipeline per dataset
# -----------------------------------------------------------------------------

def full_pipeline(X_raw: np.ndarray, y_labels: np.ndarray, lang_labels: np.ndarray, 
                  dataset_name: str, latent_dim: int, epochs: int, beta_vae_b: float, 
                  device: torch.device, results_dir: Path) -> dict:
    """Train 6 VAE variants + 2 baselines, cluster, visualise for one dataset."""
    SEP = "=" * 65
    ds_key = dataset_name.split()[0]
    print(f"\n{SEP}")
    print(f"  DATASET: {dataset_name}")
    print(f"  Samples={len(X_raw)} | Features={X_raw.shape[1]} | Genres={len(np.unique(y_labels))}")
    print(SEP)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_raw)
    le = LabelEncoder()
    y_true = le.fit_transform(y_labels)
    n_class = len(le.classes_)
    pca_dim = min(latent_dim, X_sc.shape[1])

    # Multi-modal features and CVAE condition matrix
    print("  Building multi-modal features …")
    X_mm = make_multimodal(X_sc, y_labels, le, GENRE_VOCAB, lyric_dim=LYRIC_DIM)
    X_mm_sc = StandardScaler().fit_transform(X_mm)
    C_oh = make_genre_onehot(y_labels, le)

    # -- Train 6 model variants ----------------------------------------------------
    print("  [1/6] MLP-VAE …")
    m1 = MLPVAE(X_sc.shape[1], latent_dim, h=HIDDEN_DIMS).to(device)
    m1, h1, l1 = train_model(X_sc, m1, model_type="vae", beta=1.0, epochs=epochs, lr=LR, batch_size=BATCH_SIZE, device=device)
    Z1 = extract_latent(m1, X_sc, device=device)

    print(f"  [2/6] Beta-VAE (β={beta_vae_b}) …")
    m2 = BetaVAE(X_sc.shape[1], latent_dim, h=HIDDEN_DIMS).to(device)
    m2, h2, l2 = train_model(X_sc, m2, model_type="beta_vae", beta=beta_vae_b, epochs=epochs, lr=LR, batch_size=BATCH_SIZE, device=device)
    Z2 = extract_latent(m2, X_sc, device=device)

    print("  [3/6] CVAE …")
    m3 = CVAE(X_sc.shape[1], n_class, latent_dim, h=HIDDEN_DIMS).to(device)
    m3, h3, l3 = train_model(X_sc, m3, y_onehot=C_oh, model_type="cvae", beta=1.0, epochs=epochs, lr=LR, batch_size=BATCH_SIZE, device=device)
    Z3 = extract_latent(m3, X_sc, device=device)

    print("  [4/6] Conv1D-VAE …")
    m4 = ConvVAE(X_sc.shape[1], latent_dim).to(device)
    m4, h4, l4 = train_model(X_sc, m4, model_type="vae", beta=1.0, epochs=epochs, lr=LR, batch_size=BATCH_SIZE, device=device)
    Z4 = extract_latent(m4, X_sc, device=device)

    print("  [5/6] Autoencoder …")
    m5 = Autoencoder(X_sc.shape[1], latent_dim, h=HIDDEN_DIMS).to(device)
    m5, h5, l5 = train_model(X_sc, m5, model_type="ae", epochs=epochs, lr=LR, batch_size=BATCH_SIZE, device=device)
    Z5 = extract_latent(m5, X_sc, device=device)

    print("  [6/6] Multi-modal VAE …")
    m6 = MLPVAE(X_mm_sc.shape[1], latent_dim, h=HIDDEN_DIMS).to(device)
    m6, h6, l6 = train_model(X_mm_sc, m6, model_type="vae", beta=1.0, epochs=epochs, lr=LR, batch_size=BATCH_SIZE, device=device)
    Z6 = extract_latent(m6, X_mm_sc, device=device)

    # Baselines
    Z_pca  = PCA(n_components=pca_dim, random_state=42).fit_transform(X_sc)
    Z_spec = X_sc.copy()  # raw scaled features

    # -- Clustering ----------------------------------------------------
    print("  Clustering all latent spaces …")
    cl = {
        "MLP-VAE": run_clustering(Z1, y_true, n_class, "MLP-VAE", kmeans_ninit=KMEANS_NINIT),
        "Beta-VAE": run_clustering(Z2, y_true, n_class, "Beta-VAE", kmeans_ninit=KMEANS_NINIT),
        "CVAE": run_clustering(Z3, y_true, n_class, "CVAE", kmeans_ninit=KMEANS_NINIT),
        "Conv-VAE": run_clustering(Z4, y_true, n_class, "Conv-VAE", kmeans_ninit=KMEANS_NINIT),
        "AE": run_clustering(Z5, y_true, n_class, "Autoencoder",kmeans_ninit=KMEANS_NINIT),
        "Multimodal": run_clustering(Z6, y_true, n_class, "Multimodal", kmeans_ninit=KMEANS_NINIT),
        "PCA": run_clustering(Z_pca, y_true, n_class, "PCA", kmeans_ninit=KMEANS_NINIT),
        "Spectral": run_clustering(Z_spec,y_true, n_class, "Spectral", kmeans_ninit=KMEANS_NINIT)
    }

    # -- Dimensionality reduction ----------------------------------------------------
    print("  t-SNE + UMAP …")
    le_lang = LabelEncoder().fit(lang_labels)
    y_lang = le_lang.transform(lang_labels)
    Z_dict = dict(mlp=Z1, beta=Z2, cvae=Z3, conv=Z4, ae=Z5, mm=Z6, pca=Z_pca)
    vis: dict[str, dict] = {}

    for zkey, Z in Z_dict.items():
        z_tsne = TSNE(n_components=2, perplexity=40, random_state=42).fit_transform(Z)
        z_umap = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(Z)
        vis[zkey] = { "tsne": z_tsne, "umap": z_umap }

    # -- Visualisations ----------------------------------------------------
    z_label_map = {
        "mlp": "MLP-VAE", "beta": "Beta-VAE", "cvae": "CVAE",
        "conv": "Conv-VAE", "ae": "Autoencoder", "mm": "Multimodal", "pca": "PCA"
    }

    # Latent space plots (MLP-VAE as representative)
    plot_tsne_umap(
        vis["mlp"]["umap"], y_true, y_lang,
        list(le.classes_), list(le_lang.classes_),
        projection="UMAP",
        title=f"{dataset_name} — MLP-VAE UMAP",
        save_path=results_dir / f"latent_umap_{ds_key.lower()}_mlp.png"
    )

    # Disentanglement
    plot_disentanglement(
        Z1, Z2, y_true, list(le.classes_), beta_val=beta_vae_b,
        title=f"Disentanglement — {dataset_name}",
        save_path=results_dir / f"disentangle_{ds_key.lower()}.png"
    )

    # Reconstruction
    plot_reconstruction(
        X_sc, models={"mlp": m1, "beta": m2, "cvae": m3},
        y_labels=y_labels, le=le, device=device,
        title=f"Reconstruction — {dataset_name}",
        save_path=results_dir / f"reconstruction_{ds_key.lower()}.png"
    )

    # Language separation
    plot_language_separation(
        vis["mlp"]["umap"], lang_labels, LANG_COLORS, LANG_MARKERS,
        title=f"{dataset_name} — Language Separation (MLP-VAE UMAP)",
        save_path=results_dir / f"lang_separation_{ds_key.lower()}.png"
    )

    # Cluster distribution (MLP-VAE KMeans)
    plot_cluster_composition(
        cl["MLP-VAE"]["KMeans"]["labels"], y_labels,
        model_name=f"MLP-VAE KMeans ({dataset_name})",
        save_path=results_dir / f"cluster_dist_{ds_key.lower()}.png"
    )

    # Training curves
    plot_training_curves(
        histories={"MLP-VAE": h1, "Beta-VAE": h2, "CVAE": h3,
                   "Conv-VAE": h4, "AE": h5, "Multimodal": h6},
        losses={"MLP-VAE": l1, "Beta-VAE": l2, "CVAE": l3,
                "Conv-VAE": l4, "AE": l5, "Multimodal": l6},
        model_colors=MODEL_COLORS,
        title=f"Training Curves — {dataset_name}",
        save_path=results_dir / f"training_curves_{ds_key.lower()}.png"
    )

    return dict(
        name=dataset_name, X_sc=X_sc, y_true=y_true,
        y_labels=y_labels, lang_labels=lang_labels,
        le=le, n_class=n_class, vis=vis, cl=cl,
        Z=dict(mlp=Z1, beta=Z2, cvae=Z3, conv=Z4, ae=Z5, mm=Z6, pca=Z_pca, spec=Z_spec),
        hist=dict(mlp=h1, beta=h2, cvae=h3, conv=h4, ae=h5, mm=h6),
        loss=dict(mlp=l1, beta=l2, cvae=l3, conv=l4, ae=l5, mm=l6),
        models=dict(mlp=m1, beta=m2, cvae=m3, conv=m4, ae=m5, mm=m6)
    )


# ----------------------------------------------------
# Main
# ----------------------------------------------------

def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print("  VAE HARD TASK")
    print(f"  Device: {device} | Epochs: {args.epochs} | β={args.beta}")
    print(f"{'='*65}\n")

    fma_dir = PROJECT_ROOT / "data" / "fma" / "fma_metadata"
    lmd_dir = PROJECT_ROOT / "data" / "lmd"
    gtzan_csv = PROJECT_ROOT / "data" / "gtzan" / "features_30_sec.csv"
    results_dir = PROJECT_ROOT / "results" / "hard"
    results_dir.mkdir(parents=True, exist_ok=True)

    # -- Downloads ----------------------------------------------------
    if not args.no_download:
        download_fma_metadata(fma_dir, FMA_METADATA_URL)
        download_lmd(lmd_dir, LMD_URL)
        download_gtzan_csv(gtzan_csv, GTZAN_URLS)

    # -- Load + append Bangla ----------------------------------------------------
    X_fma, y_fma = load_fma(fma_dir)
    X_lmd, y_lmd = load_lmd(lmd_dir)
    X_gh, y_gh = load_gtzan(gtzan_csv)

    datasets_raw: list[tuple] = []
    for X_en, y_en, ds_name in [
        (X_fma, y_fma, "FMA (Free Music Archive)"),
        (X_lmd, y_lmd, "LMD (Lakh MIDI Dataset)"),
        (X_gh, y_gh, "GTZAN via GitHub"),
    ]:
        lang_en = np.array(["English"] * len(X_en))
        bX, by, bl = get_bangla(
            X_en.shape[1], BANGLA_QUERIES, BANGLA_DIR, N_BANGLA_PER_GENRE
        )
        if len(bX) > 0:
            X_m = np.vstack([X_en, bX])
            y_m = np.concatenate([y_en, by])
            l_m = np.concatenate([lang_en, bl])
        else:
            X_m, y_m, l_m = X_en, y_en, lang_en
        datasets_raw.append((X_m, y_m, l_m, ds_name))

    # -- Experiments ----------------------------------------------------
    ALL: dict[str, dict] = {}
    for X_raw, y_labels, lang_labels, ds_name in datasets_raw:
        ds_key = ds_name.split()[0]
        ALL[ds_key] = full_pipeline(
            X_raw, y_labels, lang_labels, ds_name,
            latent_dim=args.latent_dim, epochs=args.epochs,
            beta_vae_b=args.beta, device=device, results_dir=results_dir
        )

    # -- Aggregate metrics  ----------------------------------------------------
    MODEL_KEYS = ["MLP-VAE", "Beta-VAE", "CVAE", "Conv-VAE",
                  "AE", "Multimodal", "PCA", "Spectral"]
    rows = []
    for ds_key, res in ALL.items():
        for mkey in MODEL_KEYS:
            for algo in ["KMeans", "Agglomerative", "DBSCAN"]:
                r = res["cl"][mkey][algo]
                rows.append({
                    "Dataset": ds_key,
                    "Model": mkey,
                    "Algorithm": algo,
                    "Silhouette": round(r["sil"], 4) if not np.isnan(r["sil"]) else np.nan,
                    "Davies-Bouldin": round(r["db"], 4) if not np.isnan(r["db"]) else np.nan,
                    "Calinski-H": round(r["ch"], 1) if not np.isnan(r["ch"]) else np.nan,
                    "NMI": round(r["nmi"], 4) if not np.isnan(r["nmi"]) else np.nan,
                    "ARI": round(r["ari"], 4) if not np.isnan(r["ari"]) else np.nan,
                    "Purity": round(r["purity"], 4) if not np.isnan(r["purity"]) else np.nan
                })

    df_all = pd.DataFrame(rows)
    df_all.to_csv(results_dir / "full_metrics.csv", index=False)
    print(f"\n  Saved: {results_dir / 'full_metrics.csv'}")

    plot_metrics_heatmap(df_all, save_path=results_dir / "metrics_heatmap.png")

    # Save cluster CSV per dataset
    for ds_key, res in ALL.items():
        pd.DataFrame({
            "genre": res["y_labels"],
            "language": res["lang_labels"],
            "kmeans_mlp": res["cl"]["MLP-VAE"]["KMeans"]["labels"],
            "kmeans_beta": res["cl"]["Beta-VAE"]["KMeans"]["labels"],
            "kmeans_cvae": res["cl"]["CVAE"]["KMeans"]["labels"],
            "kmeans_pca": res["cl"]["PCA"]["KMeans"]["labels"]
        }).to_csv(results_dir / f"clusters_{ds_key.lower()}.csv", index=False)

    # -- Quantitative analysis ----------------------------------------------------
    sep = "=" * 72
    print(f"\n{sep}")
    print("  HARD TASK - FINAL REPORT")
    print("  Models: MLP-VAE · Beta-VAE · CVAE · Conv-VAE · AE · Multimodal · PCA · Spectral")
    print(f"  Algorithms: KMeans · Agglomerative · DBSCAN")
    print(f"  Metrics: Silhouette · Davies-Bouldin · CH · NMI · ARI · Purity")
    print(sep)

    for ds_key, res in ALL.items():
        print(f"\n  Dataset: {res['name']}")
        best_sil = max(
            (res["cl"][m]["KMeans"]["sil"], m)
            for m in MODEL_KEYS
            if not np.isnan(res["cl"][m]["KMeans"]["sil"])
        )
        print(f"    Best Silhouette → {best_sil[1]} = {best_sil[0]:.4f}")

    print(f"\n  All results saved to: {results_dir}")
    print(sep)


if __name__ == "__main__":
    main()
