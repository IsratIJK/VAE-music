"""
Medium Task: Enhanced comparative VAE music clustering.

Pipeline
--------
1. Download FMA metadata, LMD (MIDI), GTZAN CSV.
2. Download real Bangla audio via yt-dlp + extract librosa features.
3. Append Bangla to each English dataset.
4. For each dataset:
   a. Train MLP-VAE, Conv1D-VAE, Hybrid MLP-VAE (audio + TF-IDF lyrics).
   b. PCA baseline.
   c. Elbow analysis.
   d. Run KMeans / Agglomerative / DBSCAN on every latent space.
   e. Compute Silhouette, Davies-Bouldin, Calinski-H, ARI, NMI.
5. Produce all visualisations:
   - t-SNE + UMAP per model per dataset
   - Elbow plots
   - DBSCAN cluster analysis
   - Cluster composition heatmaps
   - Language separation plots
   - Full metrics table + heatmap
   - VAE vs PCA delta bar chart

Usage
-----
    python scripts/run_medium.py [--epochs 100] [--no-download]

All outputs saved to results/medium/ (relative to project root).
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
    BANGLA_DIR, BANGLA_QUERIES, BATCH_SIZE, BETA_DEFAULT,
    EPOCHS, FMA_METADATA_URL, GENRE_VOCAB, GTZAN_URLS,
    HIDDEN_DIMS, KMEANS_NINIT, LANG_COLORS, LANG_MARKERS,
    LATENT_DIM, LMD_URL, LR, LYRIC_DIM, MODEL_COLORS,
    N_BANGLA_PER_GENRE, NUMPY_SEED, TORCH_SEED,
)
from src.clustering.engine import elbow_analysis, run_clustering
from src.data.bangla import get_bangla
from src.data.fma import download_fma_metadata, load_fma
from src.data.gtzan import download_gtzan_csv, load_gtzan
from src.data.lmd import download_lmd, load_lmd
from src.features.hybrid import make_hybrid
from src.models.conv_vae import ConvVAE
from src.models.mlp_vae import MLPVAE
from src.training.trainer import extract_latent, train_model
from src.visualization.plots import (
    plot_cluster_composition,
    plot_elbow,
    plot_language_separation,
    plot_metrics_heatmap,
    plot_training_curves,
    plot_tsne_umap,
)

warnings.filterwarnings("ignore")
np.random.seed(NUMPY_SEED)
torch.manual_seed(TORCH_SEED)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAE Medium Task")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    p.add_argument("--no-download", action="store_true",
                   help="Skip all dataset downloads")
    return p.parse_args()


# ------------------------------------------------------------------------------
# Full experiment pipeline for one dataset
# ------------------------------------------------------------------------------

def full_pipeline(X_raw: np.ndarray, y_labels: np.ndarray, lang_labels: np.ndarray, 
                  dataset_name: str, latent_dim: int, epochs: int, 
                  device: torch.device, results_dir: Path) -> dict:
    """Train VAE variants, cluster, visualise for one dataset.

    Returns a result dict compatible with the metrics aggregation below.
    """
    SEP = "=" * 65
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
    ds_key = dataset_name.split()[0]

    # Hybrid features (audio + TF-IDF lyrics)
    print("  Building hybrid features …")
    X_hyb_sc = StandardScaler().fit_transform(
        make_hybrid(X_sc, y_labels, GENRE_VOCAB, lyric_dim=LYRIC_DIM)
    )

    # Train all models
    print("  [1/4] MLP-VAE …")
    mlp_model = MLPVAE(X_sc.shape[1], latent_dim, h=HIDDEN_DIMS)
    mlp_model, mlp_hist, mlp_loss = train_model(
        X_sc, mlp_model, model_type="vae", epochs=epochs,
        lr=LR, batch_size=BATCH_SIZE, device=device
    )
    Z_mlp = extract_latent(mlp_model, X_sc, device=device)

    print("  [2/4] Conv1D-VAE …")
    conv_model = ConvVAE(X_sc.shape[1], latent_dim)
    conv_model, conv_hist, conv_loss = train_model(
        X_sc, conv_model, model_type="vae", epochs=epochs,
        lr=LR, batch_size=BATCH_SIZE, device=device
    )
    Z_conv = extract_latent(conv_model, X_sc, device=device)

    print("  [3/4] Hybrid MLP-VAE …")
    hyb_model = MLPVAE(X_hyb_sc.shape[1], latent_dim, h=HIDDEN_DIMS)
    hyb_model, hyb_hist, hyb_loss = train_model(
        X_hyb_sc, hyb_model, model_type="vae", epochs=epochs,
        lr=LR, batch_size=BATCH_SIZE, device=device
    )
    Z_hyb = extract_latent(hyb_model, X_hyb_sc, device=device)

    print("  [4/4] PCA baseline …")
    Z_pca = PCA(n_components=pca_dim, random_state=42).fit_transform(X_sc)

    # Elbow analysis
    print("  Elbow analysis …")
    elbow = elbow_analysis(Z_mlp, k_range=range(2, min(22, n_class + 5)))
    plot_elbow(
        elbow, true_k=n_class,
        title=f"Elbow — {dataset_name}",
        save_path=results_dir / f"elbow_{ds_key.lower()}.png",
    )

    # Clustering
    print("  Clustering …")
    cl = {
        "mlp": run_clustering(Z_mlp, y_true, n_class, "MLP-VAE", kmeans_ninit=KMEANS_NINIT),
        "conv": run_clustering(Z_conv, y_true, n_class, "Conv-VAE", kmeans_ninit=KMEANS_NINIT),
        "hybrid": run_clustering(Z_hyb, y_true, n_class, "Hybrid-VAE", kmeans_ninit=KMEANS_NINIT),
        "pca": run_clustering(Z_pca, y_true, n_class, "PCA", kmeans_ninit=KMEANS_NINIT)
    }

    # t-SNE + UMAP
    print("  t-SNE + UMAP …")
    le_lang = LabelEncoder().fit(lang_labels)
    y_lang = le_lang.transform(lang_labels)
    vis: dict[str, dict] = {}
    for zkey, Z in [("mlp", Z_mlp), ("conv", Z_conv), ("hybrid", Z_hyb), ("pca", Z_pca)]:
        z_tsne = TSNE(n_components=2, perplexity=40, random_state=42).fit_transform(Z)
        z_umap = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(Z)
        vis[zkey] = {"tsne": z_tsne, "umap": z_umap}

        lname = {"mlp": "MLP-VAE", "conv": "Conv-VAE", "hybrid": "Hybrid-VAE", "pca": "PCA"}[zkey]
        plot_tsne_umap(
            z_tsne, y_true, y_lang,
            list(le.classes_), list(le_lang.classes_),
            projection="t-SNE",
            title=f"{dataset_name} — {lname} t-SNE",
            save_path=results_dir / f"tsne_{ds_key.lower()}_{zkey}.png"
        )

    # Cluster composition heatmap (KMeans, MLP-VAE)
    plot_cluster_composition(
        cl["mlp"]["KMeans"]["labels"], y_labels,
        model_name=f"MLP-VAE ({dataset_name})",
        save_path=results_dir / f"cluster_composition_{ds_key.lower()}.png"
    )

    # Language separation (MLP-VAE UMAP)
    plot_language_separation(
        vis["mlp"]["umap"], lang_labels, LANG_COLORS, LANG_MARKERS,
        title=f"{dataset_name} — Language Separation (MLP-VAE UMAP)",
        save_path=results_dir / f"lang_separation_{ds_key.lower()}.png"
    )

    # Training curves
    plot_training_curves(
        histories={"MLP-VAE": mlp_hist, "Conv-VAE": conv_hist, "Hybrid-VAE": hyb_hist},
        losses={"MLP-VAE": mlp_loss, "Conv-VAE": conv_loss, "Hybrid-VAE": hyb_loss},
        model_colors={
            "MLP-VAE": MODEL_COLORS["MLP-VAE"],
            "Conv-VAE": MODEL_COLORS["Conv-VAE"],
            "Hybrid-VAE": MODEL_COLORS["Multimodal"]
        },
        title=f"Training Curves — {dataset_name}",
        save_path=results_dir / f"training_curves_{ds_key.lower()}.png"
    )

    return dict(
        name=dataset_name, X_sc=X_sc, y_true=y_true,
        y_labels=y_labels, lang_labels=lang_labels,
        le=le, n_class=n_class, elbow=elbow, vis=vis,
        Z=dict(mlp=Z_mlp, conv=Z_conv, hybrid=Z_hyb, pca=Z_pca),
        cl=cl,
        hist=dict(mlp=mlp_hist, conv=conv_hist, hybrid=hyb_hist),
        loss=dict(mlp=mlp_loss, conv=conv_loss, hybrid=hyb_loss)
    )


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print("  VAE MEDIUM TASK")
    print(f"  Device: {device} | Epochs: {args.epochs}")
    print(f"{'='*65}\n")

    fma_dir = PROJECT_ROOT / "data" / "fma" / "fma_metadata"
    lmd_dir = PROJECT_ROOT / "data" / "lmd"
    gtzan_csv = PROJECT_ROOT / "data" / "gtzan" / "features_30_sec.csv"
    results_dir = PROJECT_ROOT / "results" / "medium"
    results_dir.mkdir(parents=True, exist_ok=True)

    # -- Downloads ----------------------------------------------
    if not args.no_download:
        download_fma_metadata(fma_dir, FMA_METADATA_URL)
        download_lmd(lmd_dir, LMD_URL)
        download_gtzan_csv(gtzan_csv, GTZAN_URLS)

    # -- Load datasets --------------------------------------------------
    X_fma, y_fma = load_fma(fma_dir)
    X_lmd, y_lmd = load_lmd(lmd_dir)
    X_gh, y_gh = load_gtzan(gtzan_csv)

    # -- Append real Bangla (cached per feature dim) ------------------------
    for X, y, lang_arr, name in [
        (X_fma, y_fma, None, "FMA"),
        (X_lmd, y_lmd, None, "LMD"),
        (X_gh, y_gh, None, "GTZAN"),
    ]:
        pass  # handled below with mutable lists

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
            X_merged = np.vstack([X_en, bX])
            y_merged = np.concatenate([y_en, by])
            lang_merged = np.concatenate([lang_en, bl])
        else:
            X_merged, y_merged, lang_merged = X_en, y_en, lang_en
        datasets_raw.append((X_merged, y_merged, lang_merged, ds_name))

    # -- Run experiments ----------------------------------------------
    all_results: dict[str, dict] = {}
    for X_raw, y_labels, lang_labels, ds_name in datasets_raw:
        ds_key = ds_name.split()[0]
        res = full_pipeline(
            X_raw, y_labels, lang_labels, ds_name,
            latent_dim=args.latent_dim, epochs=args.epochs,
            device=device, results_dir=results_dir,
        )
        all_results[ds_key] = res

    # -- Aggregate metrics table ------------------------------------
    rows = []
    label_map = {"mlp": "MLP-VAE", "conv": "Conv-VAE", "hybrid": "Hybrid-VAE", "pca": "PCA"}
    for ds_key, res in all_results.items():
        for zkey, zlab in label_map.items():
            for algo in ["KMeans", "Agglomerative", "DBSCAN"]:
                r = res["cl"][zkey][algo]
                rows.append({
                    "Dataset": ds_key,
                    "Features": zlab,
                    "Algorithm": algo,
                    "Silhouette": round(r["sil"], 4) if not np.isnan(r["sil"]) else np.nan,
                    "Davies-Bouldin": round(r["db"], 4) if not np.isnan(r["db"]) else np.nan,
                    "Calinski-H": round(r["ch"], 1) if not np.isnan(r["ch"]) else np.nan,
                    "ARI": round(r["ari"], 4) if not np.isnan(r["ari"]) else np.nan,
                    "NMI": round(r["nmi"], 4) if not np.isnan(r["nmi"]) else np.nan
                })

    df_all = pd.DataFrame(rows)
    df_all.to_csv(results_dir / "full_metrics.csv", index=False)
    print(f"\n  Saved: {results_dir / 'full_metrics.csv'}")

    plot_metrics_heatmap(
        df_all.rename(columns={"Features": "Model"}),
        save_path=results_dir / "metrics_heatmap.png"
    )

    # -- Final summary -------------------------------------------------
    sep = "=" * 65
    print(f"\n{sep}")
    print("  MEDIUM TASK — FINAL REPORT")
    print(f"  3 Datasets × 4 Feature Spaces × 3 Algorithms × 5 Metrics")
    print(sep)
    for ds_key, res in all_results.items():
        print(f"\n  {res['name']}  |  {len(res['y_labels'])} samples  {res['n_class']} genres")
    print(f"\n  All results saved to: {results_dir}")
    print(sep)


if __name__ == "__main__":
    main()
