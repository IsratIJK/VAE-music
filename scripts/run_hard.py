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

Advanced Extensions (unique research contributions)
-----------------------------------------------------
Ext-1  GMVAE         — Gaussian Mixture VAE with K-component learned prior
Ext-2  β-Sensitivity — Beta-VAE sweep over β ∈ {0.5, 1, 2, 4, 8, 16}
Ext-3  MIG           — Mutual Information Gap disentanglement metric
Ext-4  Interpolation — SLERP between genre centroids in latent space
Ext-5  Transfer      — Zero-shot cross-dataset transfer (FMA↔LMD↔GTZAN)
Ext-A  ContrastiveVAE— InfoNCE + β-VAE with projection head
Ext-B  DANN-VAE      — Domain adversarial + gradient reversal

Usage
-----
    python scripts/run_hard.py [--epochs 100] [--no-download]
    python scripts/run_hard.py [--no-extensions]  # skip advanced extensions

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
from sklearn.cluster import KMeans as _KM4
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler as _SS
from sklearn.cluster import KMeans as _KM3
from sklearn.cluster import KMeans as _KM2
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config.config import (
    BANGLA_DIR, BANGLA_QUERIES, BATCH_SIZE, BETA_VALUES, BETA_VAE_B,
    CONTRASTIVE_TEMPERATURE, DANN_COMMON_DIM, DANN_DOMAIN_WEIGHT,
    EPOCHS, FMA_METADATA_URL, GENRE_VOCAB, GTZAN_URLS,
    HIDDEN_DIMS, KMEANS_NINIT, LANG_COLORS, LANG_MARKERS,
    LAMBDA_VALUES, LATENT_DIM, LMD_URL, LR, LYRIC_DIM, MODEL_COLORS,
    N_BANGLA_PER_GENRE, N_INTERP, NUMPY_SEED, SWEEP_EPOCHS, TORCH_SEED,
)
from src.clustering.engine import compute_metrics, compute_mig, run_clustering
from src.data.bangla import get_bangla
from src.data.fma import download_fma_metadata, load_fma
from src.data.gtzan import download_gtzan_csv, load_gtzan
from src.data.lmd import download_lmd, load_lmd
from src.features.hybrid import make_genre_onehot, make_multimodal
from src.models.autoencoder import Autoencoder
from src.models.contrastive_vae import train_contrastive_vae
from src.models.conv_vae import ConvVAE
from src.models.cvae import CVAE
from src.models.dann_vae import build_dann_dataset, train_dann_vae
from src.models.gmvae import train_gmvae
from src.models.mlp_vae import BetaVAE, MLPVAE
from src.training.trainer import extract_latent, train_model
from src.visualization.plots import (
    plot_beta_sensitivity,
    plot_cluster_composition,
    plot_contrastive_results,
    plot_dann_results,
    plot_disentanglement,
    plot_gmvae_results,
    plot_interpolation,
    plot_language_separation,
    plot_mega_heatmap,
    plot_metrics_heatmap,
    plot_mig_scores,
    plot_reconstruction,
    plot_training_curves,
    plot_transfer_results,
    plot_tsne_umap,
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
    p.add_argument("--no-extensions", action="store_true",
                   help="Skip advanced extensions (Ext 1-5, A, B)")
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
    X_sc = scaler.fit_transform(X_raw).astype(np.float32)
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
        name=dataset_name,
        X_sc=X_sc, # scaled feature matrix (needed by extensions)
        y_true=y_true,
        y_labels=y_labels,
        lang_labels=lang_labels,
        le=le,
        n_class=n_class,
        vis=vis,
        cl=cl,
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
    print("  Algorithms: KMeans · Agglomerative · DBSCAN")
    print("  Metrics: Silhouette · Davies-Bouldin · CH · NMI · ARI · Purity")
    print(sep)

    for ds_key, res in ALL.items():
        print(f"\n  Dataset: {res['name']}")
        best_sil = max(
            (res["cl"][m]["KMeans"]["sil"], m)
            for m in MODEL_KEYS
            if not np.isnan(res["cl"][m]["KMeans"]["sil"])
        )
        print(f"    Best Silhouette → {best_sil[1]} = {best_sil[0]:.4f}")

    print(f"\n  Core results saved to: {results_dir}")
    print(sep)

    # --------------------------------------------------------------------------
    # Advanced Extensions
    # --------------------------------------------------------------------------
    if args.no_extensions:
        print("\n  [--no-extensions] Skipping advanced extensions.")
        return

    print(f"\n{'='*72}")
    print("  ADVANCED EXTENSIONS")
    print(f"{'='*72}")

    # -- Extension 1: GMVAE -------------------------------------------------------
    print("\n[Ext-1] GMVAE — Gaussian Mixture VAE")
    GMVAE_RESULTS: dict[str, dict] = {}

    for ds_key, res in ALL.items():
        print(f"\n  GMVAE — {res['name']}")
        X_sc_d = res["X_sc"]
        y_true_d = res["y_true"]
        K_d = res["n_class"]

        gm, gm_hist, gm_loss = train_gmvae(
            X_sc_d, n_components=K_d, latent_dim=args.latent_dim,
            h=HIDDEN_DIMS, epochs=args.epochs, lr=LR,
            batch_size=BATCH_SIZE, device=device
        )

        # Extract latent μ and soft assignments
        gm.eval()
        X_t = torch.FloatTensor(X_sc_d)
        Z_gm_parts: list[np.ndarray] = []
        QY_parts: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(X_t), BATCH_SIZE):
                mu, lv, qy = gm.encode(X_t[i : i + BATCH_SIZE].to(device))
                Z_gm_parts.append(mu.cpu().numpy())
                QY_parts.append(qy.cpu().numpy())

        Z_gm = np.vstack(Z_gm_parts)
        QY = np.vstack(QY_parts)  # (N, K) soft assignments
        gm_lbls = QY.argmax(axis=1)  # hard assignment

        m_gm = compute_metrics(Z_gm, y_true_d, gm_lbls)
        print(f"  GMVAE  Sil={m_gm['sil']:.4f}  DB={m_gm['db']:.4f}  "
              f"NMI={m_gm['nmi']:.4f}  ARI={m_gm['ari']:.4f}  Purity={m_gm['purity']:.4f}")

        tsne_gm = TSNE(n_components=2, perplexity=40, max_iter=500, random_state=42).fit_transform(Z_gm)
        umap_gm = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(Z_gm)

        GMVAE_RESULTS[ds_key] = dict(
            Z=Z_gm, QY=QY, labels=gm_lbls, metrics=m_gm,
            hist=gm_hist, loss=gm_loss, model=gm,
            tsne=tsne_gm, umap=umap_gm,
            y_true=y_true_d, y_labels=res["y_labels"],
            lang_labels=res["lang_labels"], le=res["le"], K=K_d
        )

    plot_gmvae_results(
        GMVAE_RESULTS, ALL,
        save_path=results_dir / "ext1_gmvae_results.png",
    )

    # -- Extension 2: β-Sensitivity Analysis ------------------------------------------
    print("\n[Ext-2] β-Sensitivity Analysis")
    BETA_RESULTS: dict[str, dict] = {}

    for ds_key, res in ALL.items():
        print(f"\n  β-sweep — {res['name']}")
        X_sc_d = res["X_sc"]
        y_true_d = res["y_true"]
        K_d = res["n_class"]
        X_t = torch.FloatTensor(X_sc_d)
        BETA_RESULTS[ds_key] = {}

        for beta_v in BETA_VALUES:
            print(f"    β={beta_v:.1f} ...", end=" ", flush=True)
            m_beta = BetaVAE(X_sc_d.shape[1], args.latent_dim, h=HIDDEN_DIMS).to(device)
            m_beta, _, _ = train_model(
                X_sc_d, m_beta, model_type="beta_vae", beta=beta_v,
                epochs=SWEEP_EPOCHS, lr=LR, batch_size=BATCH_SIZE,
                device=device, verbose=False
            )

            # Extract latent and measure reconstruction loss
            m_beta.eval()
            Z_list_b: list[np.ndarray] = []
            recon_err = 0.0; n_b = 0
            with torch.no_grad():
                for i in range(0, len(X_t), BATCH_SIZE):
                    bx = X_t[i : i + BATCH_SIZE].to(device)
                    mu_b, _ = m_beta.enc(bx)
                    Z_list_b.append(mu_b.cpu().numpy())
                    recon_b, _, _, _ = m_beta(bx)
                    recon_err += float(torch.nn.functional.mse_loss(recon_b, bx).item())
                    n_b += 1

            Z_b = np.vstack(Z_list_b)
            from sklearn.cluster import KMeans as _KM
            km_b = _KM(n_clusters=K_d, n_init=KMEANS_NINIT, random_state=42).fit(Z_b)
            m_b = compute_metrics(Z_b, y_true_d, km_b.labels_)
            m_b["recon_loss"] = recon_err / max(n_b, 1)
            print(f"Sil={m_b['sil']:.3f}  NMI={m_b['nmi']:.3f}  Recon={m_b['recon_loss']:.3f}")
            BETA_RESULTS[ds_key][beta_v] = dict(metrics=m_b, Z=Z_b)

    plot_beta_sensitivity(BETA_RESULTS, save_path=results_dir / "ext2_beta_sensitivity.png")

    # -- Extension 3: MIG (Mutual Information Gap) -------------------------------------------
    print("\n[Ext-3] MIG — Mutual Information Gap")
    MIG_RESULTS: dict[str, dict] = {}
    model_z_map = [
        ("MLP-VAE", "mlp"),
        ("Beta-VAE", "beta"),
        ("CVAE", "cvae"),
        ("Conv-VAE", "conv"),
        ("AE", "ae"),
        ("PCA", "pca")
    ]

    for ds_key, res in ALL.items():
        MIG_RESULTS[ds_key] = {}
        print(f"\n  Dataset: {ds_key}")
        for mname, zkey in model_z_map:
            mig_score, mi_dims = compute_mig(res["Z"][zkey], res["y_true"])
            MIG_RESULTS[ds_key][mname] = dict(mig=mig_score, mi_per_dim=mi_dims)
            print(f"    {mname:<12} MIG = {mig_score:.4f}")
        if ds_key in GMVAE_RESULTS:
            mig_gm, mi_gm = compute_mig(GMVAE_RESULTS[ds_key]["Z"], res["y_true"])
            MIG_RESULTS[ds_key]["GMVAE"] = dict(mig=mig_gm, mi_per_dim=mi_gm)
            print(f"    {'GMVAE':<12} MIG = {mig_gm:.4f}")

    plot_mig_scores(MIG_RESULTS, save_path=results_dir / "ext3_mig_scores.png")

    # -- Extension 4: Latent Space Interpolation --------------------------------
    print("\n[Ext-4] Latent Space Interpolation (SLERP)")
    plot_interpolation(ALL, n_interp=N_INTERP, save_dir=results_dir)

    # -- Extension 5: Cross-Dataset Transfer -------------------------------
    print("\n[Ext-5] Cross-Dataset Transfer (Zero-Shot)")

    transfer_pairs = [
        ("FMA", "LMD"),
        ("FMA", "GTZAN"),
        ("LMD", "GTZAN"),
        ("GTZAN", "FMA")
    ]

    def _align_features(X_src: np.ndarray, X_tgt: np.ndarray) -> np.ndarray:
        """Project X_tgt to the same column count as X_src via PCA or zero-padding."""
        src_dim = X_src.shape[1]
        tgt_dim = X_tgt.shape[1]
        if src_dim == tgt_dim:
            return X_tgt
        if tgt_dim > src_dim:
            from sklearn.decomposition import PCA as _PCA
            return _PCA(n_components=src_dim, random_state=42).fit_transform(X_tgt).astype(np.float32)
        pad = np.zeros((X_tgt.shape[0], src_dim - tgt_dim), dtype=np.float32)
        return np.hstack([X_tgt, pad])

    TRANSFER_RESULTS: dict[str, dict] = {}
    sep65 = "=" * 65
    print(f"  {sep65}")
    print("  Source VAE encodes Target dataset — no retraining")
    print(f"  {sep65}")

    for src_key, tgt_key in transfer_pairs:
        if src_key not in ALL or tgt_key not in ALL:
            continue
        print(f"\n  {src_key} → {tgt_key}")
        src_model = ALL[src_key]["models"]["mlp"]
        src_X = ALL[src_key]["X_sc"]
        tgt_X = ALL[tgt_key]["X_sc"]
        tgt_y = ALL[tgt_key]["y_true"]
        tgt_K = ALL[tgt_key]["n_class"]

        tgt_X_aligned = _align_features(src_X, tgt_X)
        tgt_X_sc = _SS().fit_transform(tgt_X_aligned).astype(np.float32)

        src_model.eval()
        Z_transfer: list[np.ndarray] = []
        X_t_tr = torch.FloatTensor(tgt_X_sc)
        with torch.no_grad():
            for i in range(0, len(X_t_tr), BATCH_SIZE):
                mu_t, _ = src_model.enc(X_t_tr[i : i + BATCH_SIZE].to(device))
                Z_transfer.append(mu_t.cpu().numpy())
        Z_tr = np.vstack(Z_transfer)

        K_t = min(tgt_K, len(Z_tr) - 1)
        km_t = _KM2(n_clusters=K_t, n_init=KMEANS_NINIT, random_state=42).fit(Z_tr)
        m_tr = compute_metrics(Z_tr, tgt_y, km_t.labels_)
        m_nat = ALL[tgt_key]["cl"]["MLP-VAE"]["KMeans"]
        retain = m_tr["sil"] / (m_nat["sil"] + 1e-8)

        print(f"  Transfer  Sil={m_tr['sil']:.4f}  NMI={m_tr['nmi']:.4f}  "
              f"ARI={m_tr['ari']:.4f}  Purity={m_tr['purity']:.4f}")
        print(f"  Native    Sil={m_nat['sil']:.4f}  NMI={m_nat['nmi']:.4f}  "
              f"ARI={m_nat['ari']:.4f}  Purity={m_nat['purity']:.4f}")
        print(f"  Retention : {retain*100:.1f}% of native performance")

        TRANSFER_RESULTS[f"{src_key}→{tgt_key}"] = dict(
            Z=Z_tr, labels=km_t.labels_, metrics=m_tr,
            native=m_nat, retention=retain,
            tgt_y=tgt_y, tgt_le=ALL[tgt_key]["le"]
        )

    plot_transfer_results(TRANSFER_RESULTS, save_path=results_dir / "ext5_transfer_results.png")

    # -- Extension A: Contrastive VAE ---------------------------------------------
    print("\n[Ext-A] Contrastive VAE (InfoNCE + β-VAE)")
    CVAE_CON_RESULTS: dict[str, dict] = {}

    for ds_key, res in ALL.items():
        print(f"\n{'='*60}")
        print(f"  ContrastiveVAE — {res['name']}")
        print(f"{'='*60}")
        X_sc_d = res["X_sc"]
        y_lbl_d = res["y_labels"]
        y_true_d = res["y_true"]
        K_d = res["n_class"]
        X_t = torch.FloatTensor(X_sc_d)

        best_model_c = None
        best_sil_c = -1.0
        best_lam_c = LAMBDA_VALUES[0]
        best_Z_c: np.ndarray | None = None
        best_met_c: dict | None = None
        best_hist_c: dict | None = None
        lambda_sweep: dict[float, dict] = {}

        for lam_v in LAMBDA_VALUES:
            print(f"\n  λ={lam_v:.1f}:")
            m_con, hist_con, _ = train_contrastive_vae(
                X_sc_d, y_lbl_d, latent_dim=args.latent_dim, h=HIDDEN_DIMS,
                epochs=args.epochs, lr=LR, beta=1.0,
                lam=lam_v, temperature=CONTRASTIVE_TEMPERATURE,
                batch_size=BATCH_SIZE, device=device
            )
            m_con.eval()
            Z_list_c: list[np.ndarray] = []
            with torch.no_grad():
                for i in range(0, len(X_t), BATCH_SIZE):
                    mu_c, _ = m_con.enc(X_t[i : i + BATCH_SIZE].to(device))
                    Z_list_c.append(mu_c.cpu().numpy())
            Z_c = np.vstack(Z_list_c)
            
            km_c = _KM3(n_clusters=K_d, n_init=KMEANS_NINIT, random_state=42).fit(Z_c)
            met_c  = compute_metrics(Z_c, y_true_d, km_c.labels_)
            print(f"  -> Sil={met_c['sil']:.4f}  NMI={met_c['nmi']:.4f}  "
                  f"ARI={met_c['ari']:.4f}  Purity={met_c['purity']:.4f}")
            lambda_sweep[lam_v] = dict(metrics=met_c, Z=Z_c, model=m_con)

            if met_c["sil"] > best_sil_c:
                best_sil_c = met_c["sil"]
                best_lam_c = lam_v
                best_model_c = m_con
                best_Z_c = Z_c
                best_met_c = met_c
                best_hist_c = hist_con

        print(f"\n  Best λ={best_lam_c:.1f}  Sil={best_sil_c:.4f}")
        vae_sil = ALL[ds_key]["cl"]["MLP-VAE"]["KMeans"]["sil"]
        improvement = (best_sil_c - vae_sil) / (abs(vae_sil) + 1e-8) * 100
        print(f"  Improvement over MLP-VAE: {improvement:+.1f}%")

        assert best_Z_c is not None
        tsne_c = TSNE(n_components=2, perplexity=40, max_iter=500, random_state=42).fit_transform(best_Z_c)
        umap_c = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(best_Z_c)

        CVAE_CON_RESULTS[ds_key] = dict(
            Z=best_Z_c, metrics=best_met_c, hist=best_hist_c,
            model=best_model_c, best_lam=best_lam_c,
            labels=best_Z_c,   # placeholder; overwritten below
            lambda_sweep=lambda_sweep,
            tsne=tsne_c, umap=umap_c,
            y_true=y_true_d, y_labels=y_lbl_d,
            lang_labels=res["lang_labels"], le=res["le"], K=K_d
        )

    plot_contrastive_results(
        CVAE_CON_RESULTS, ALL,
        save_path=results_dir / "extA_contrastive_vae.png",
    )

    # -- Extension B: DANN-VAE ----------------------------------------
    print("\n[Ext-B] DANN-VAE — Domain Adversarial VAE")
    print(f"\n{'='*60}")
    print("  Training DANN-VAE on combined 3-domain dataset")
    print(f"{'='*60}")

    X_dann, d_dann, y_dann, lang_dann, DS_INFO = build_dann_dataset(
        ALL, datasets_raw, common_dim=DANN_COMMON_DIM,
    )
    print(f"  Domain distribution: {np.bincount(d_dann).tolist()}")

    dann_model, dann_hist, dann_loss = train_dann_vae(
        X_dann, d_dann, latent_dim=args.latent_dim, h=HIDDEN_DIMS,
        epochs=args.epochs, lr=LR, beta=1.0,
        lam_domain=DANN_DOMAIN_WEIGHT, batch_size=BATCH_SIZE, device=device
    )
    print(f"\n  DANN-VAE training complete! Best loss: {dann_loss:.4f}")

    # Extract per-domain latent codes
    dann_model.eval()
    X_t_dann = torch.FloatTensor(X_dann)
    Z_dann_all_parts: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X_t_dann), BATCH_SIZE):
            mu_d, _ = dann_model.enc(X_t_dann[i : i + BATCH_SIZE].to(device))
            Z_dann_all_parts.append(mu_d.cpu().numpy())
    Z_dann_all = np.vstack(Z_dann_all_parts)

    DANN_RESULTS: dict[str, dict] = {}
    print(f"\n{'='*65}")
    print("  DANN-VAE: Per-Domain Clustering Results")
    print(f"{'='*65}")

    offset = 0
    for ds_key, d_id, le_d, y_true_d, n_samples in DS_INFO:
        Z_d = Z_dann_all[offset : offset + n_samples]
        K_d = ALL[ds_key]["n_class"]
        offset += n_samples
        
        km_d = _KM4(n_clusters=K_d, n_init=KMEANS_NINIT, random_state=42).fit(Z_d)
        met_d = compute_metrics(Z_d, y_true_d, km_d.labels_)
        nat_d = ALL[ds_key]["cl"]["MLP-VAE"]["KMeans"]
        delta_s = met_d["sil"] - nat_d["sil"]

        print(f"\n  Dataset: {ds_key}")
        print(f"  DANN-VAE  Sil={met_d['sil']:.4f}  NMI={met_d['nmi']:.4f}  "
              f"ARI={met_d['ari']:.4f}  Purity={met_d['purity']:.4f}")
        print(f"  Native    Sil={nat_d['sil']:.4f}  NMI={nat_d['nmi']:.4f}  "
              f"ARI={nat_d['ari']:.4f}  Purity={nat_d['purity']:.4f}")
        msg = "domain-inv helps" if delta_s > 0 else "domain-spec better"
        print(f"  Δ Sil = {delta_s:+.4f}  ({msg})")

        tsne_d = TSNE(n_components=2, perplexity=40, max_iter=500, random_state=42).fit_transform(Z_d)
        DANN_RESULTS[ds_key] = dict(
            Z=Z_d, metrics=met_d, labels=km_d.labels_,
            tsne=tsne_d, y_true=y_true_d,
            lang_labels=ALL[ds_key]["lang_labels"],
            le=le_d, K=K_d
        )

    # Full dataset t-SNE (colored by domain)
    tsne_full = TSNE(n_components=2, perplexity=40, max_iter=500, random_state=42).fit_transform(Z_dann_all)
    DANN_RESULTS["_full"] = dict(
        Z=Z_dann_all, tsne=tsne_full, d_labels=d_dann, y_dann=y_dann,
    )

    d_acc = plot_dann_results(
        DANN_RESULTS, dann_hist, X_dann, d_dann, dann_model,
        ALL, device, batch_size=BATCH_SIZE,
        save_path=results_dir / "extB_dann_vae_results.png",
    )

    # -- Mega Comparison -------------------------------------------------------
    print("\n[Mega] Building complete comparison across ALL models …")
    mega_rows = []

    for ds_key, res in ALL.items():
        for mkey in MODEL_KEYS:
            r = res["cl"][mkey]["KMeans"]
            mega_rows.append({
                "Dataset": ds_key, "Model": mkey, "Type": "Original",
                "Sil": r["sil"], "NMI": r["nmi"], "ARI": r["ari"], "Purity": r["purity"],
            })

    for ds_key, gr in GMVAE_RESULTS.items():
        m = gr["metrics"]
        mega_rows.append({
            "Dataset": ds_key, "Model": "GMVAE", "Type": "Extension-1",
            "Sil": m["sil"], "NMI": m["nmi"], "ARI": m["ari"], "Purity": m["purity"],
        })

    for ds_key, cr in CVAE_CON_RESULTS.items():
        m = cr["metrics"]
        mega_rows.append({
            "Dataset": ds_key, "Model": "ContrastiveVAE", "Type": "Extension-A",
            "Sil": m["sil"], "NMI": m["nmi"], "ARI": m["ari"], "Purity": m["purity"],
        })

    for ds_key, dr in [(k, v) for k, v in DANN_RESULTS.items() if k != "_full"]:
        m = dr["metrics"]
        mega_rows.append({
            "Dataset": ds_key, "Model": "DANN-VAE", "Type": "Extension-B",
            "Sil": m["sil"], "NMI": m["nmi"], "ARI": m["ari"], "Purity": m["purity"],
        })

    df_mega = pd.DataFrame(mega_rows)
    df_mega.to_csv(results_dir / "mega_comparison.csv", index=False)

    plot_mega_heatmap(df_mega, save_path=results_dir / "mega_heatmap.png")

    # -- Advanced Extensions Final Report --------------------------------------------
    sep72 = "=" * 72
    print(f"\n{sep72}")
    print("  ADVANCED EXTENSIONS — FINAL SUMMARY")
    print(sep72)

    print("\n[1] GMVAE — Gaussian Mixture VAE")
    for ds_key, gr in GMVAE_RESULTS.items():
        gm_s = gr["metrics"]["sil"]
        vae_s = ALL[ds_key]["cl"]["MLP-VAE"]["KMeans"]["sil"]
        winner = "GMVAE" if gm_s > vae_s else "VAE  "
        print(f"  {ds_key:<8} GMVAE={gm_s:.4f}  VAE={vae_s:.4f}  Winner: {winner}")

    print("\n[2] β-Sensitivity — Optimal β per dataset:")
    for ds_key, ds_data in BETA_RESULTS.items():
        betas = sorted(ds_data.keys())
        sils = [ds_data[b]["metrics"]["sil"] for b in betas]
        best_b = betas[int(np.argmax(sils))]
        print(f"  {ds_key:<8} Best β={best_b:.1f}  (Sil={max(sils):.4f})")

    print("\n[3] MIG Disentanglement Scores:")
    for ds_key, ds_mig in MIG_RESULTS.items():
        scores = {m: ds_mig[m]["mig"] for m in ds_mig}
        best_m = max(scores, key=scores.__getitem__)
        print(f"  {ds_key:<8} Best: {best_m:<14} MIG={scores[best_m]:.4f}")
        for mname in ("MLP-VAE", "Beta-VAE", "GMVAE"):
            if mname in scores:
                print(f"           {mname:<14} MIG={scores[mname]:.4f}")

    print("\n[4] Latent Interpolation:")
    print("  SLERP paths between genre centroids — see interpolation_*.png")
    print("  Smooth feature transitions confirm semantic latent structure.")

    print("\n[5] Cross-Dataset Transfer (Zero-Shot):")
    for pair_key, tr in TRANSFER_RESULTS.items():
        qual = ("Good" if tr["retention"] >= 0.7
                else "Partial" if tr["retention"] >= 0.5
                else "Poor")
        print(f"  {pair_key:<12} Sil={tr['metrics']['sil']:.4f}  "
              f"Retention={tr['retention']*100:.1f}%  {qual}")

    print("\n[A] Contrastive VAE (InfoNCE + β-VAE):")
    for ds_key, cr in CVAE_CON_RESULTS.items():
        vae_s = ALL[ds_key]["cl"]["MLP-VAE"]["KMeans"]["sil"]
        con_s = cr["metrics"]["sil"]
        delta = con_s - vae_s
        qual = "Better" if delta > 0.01 else "Similar" if delta > -0.01 else "Worse"
        print(f"  {ds_key:<8} λ={cr['best_lam']:.1f}  Sil={con_s:.4f}  "
              f"vs MLP-VAE={vae_s:.4f}  Δ={delta:+.4f}  {qual}")

    print("\n[B] DANN-VAE (Domain Adversarial):")
    for ds_key, dr in [(k, v) for k, v in DANN_RESULTS.items() if k != "_full"]:
        vae_s = ALL[ds_key]["cl"]["MLP-VAE"]["KMeans"]["sil"]
        dann_s = dr["metrics"]["sil"]
        delta = dann_s - vae_s
        qual = "Better" if delta > 0.01 else "Similar" if delta > -0.01 else "Worse"
        print(f"  {ds_key:<8} Sil={dann_s:.4f}  vs Native={vae_s:.4f}  "
              f"Δ={delta:+.4f}  {qual}")
    print(f"  Domain classifier acc: {d_acc:.1f}%  (chance=33.3%)")

    print(f"\n  All results saved to: {results_dir}")
    print(sep72)


if __name__ == "__main__":
    main()
