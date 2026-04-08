"""
Visualization functions for the VAE-music project.

All functions accept a 'save_path' argument.  When provided the figure is
saved to that path before display.

All paths are relative so no user directory information is embedded in the output.

Functions
---------
plot_dataset_distribution   : Bar charts of language and genre counts.
plot_training_curves        : Loss curves per model.
plot_elbow                  : Inertia / silhouette / CH vs K.
plot_tsne_umap              : t-SNE and UMAP scatter plots.
plot_cluster_composition    : Heatmap of genre % per cluster.
plot_language_separation    : English vs Bangla in UMAP space.
plot_metrics_comparison     : Bar chart comparing VAE vs PCA metrics.
plot_metrics_heatmap        : Heatmap of all metrics x all models.
plot_disentanglement        : Latent dimension histograms MLP-VAE vs Beta-VAE.
plot_reconstruction         : Original vs reconstructed feature vectors.
plot_gmvae_results          : GMVAE learned component vs true genre (t-SNE/UMAP).
plot_beta_sensitivity       : β-sweep metric curves for Beta-VAE.
plot_mig_scores             : MIG disentanglement bar chart and per-dim MI.
plot_interpolation          : SLERP feature-evolution heatmap and t-SNE path.
plot_transfer_results       : Cross-dataset transfer vs native performance bars.
plot_contrastive_results    : ContrastiveVAE latent space and λ-sensitivity.
plot_dann_results           : DANN-VAE domain alignment visualisation.
plot_mega_heatmap           : Full model x metric x dataset mega heatmap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def _save(fig: plt.Figure, save_path: Path | str | None) -> None:
    """Save figure if save_path is given, then show."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)


# ------------------------------------------------------------------------------
# Dataset overview
# ------------------------------------------------------------------------------

def plot_dataset_distribution(y_labels: np.ndarray, lang_labels: np.ndarray, 
                              title: str = "Dataset Distribution", 
                              save_path: Path | str | None = None) -> None:
    """Plot language distribution and genre-by-language bar charts."""
    df = pd.DataFrame({"genre": y_labels, "language": lang_labels})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    lang_counts = df["language"].value_counts()
    axes[0].bar(
        lang_counts.index, lang_counts.values,
        color=["#4e8ef7", "#f7914e"], edgecolor="white", linewidth=1.5
    )
    axes[0].set_title("Language Distribution", fontweight="bold")
    axes[0].set_ylabel("Track Count")
    for i, v in enumerate(lang_counts.values):
        axes[0].text(i, v + 10, str(v), ha="center", fontweight="bold")

    genre_counts = df.groupby(["genre", "language"]).size().unstack(fill_value=0)
    genre_counts.plot(
        kind="bar", ax=axes[1], color=["#f7914e", "#4e8ef7"], edgecolor="white"
    )
    axes[1].set_title("Genre Distribution by Language", fontweight="bold")
    axes[1].set_ylabel("Track Count")
    axes[1].tick_params(axis="x", rotation=40)
    axes[1].legend(title="Language")

    plt.tight_layout()
    _save(fig, save_path)


# -----------------------------------------------------------------------------
# Training curves
# -----------------------------------------------------------------------------

def plot_training_curves(histories: dict[str, list[float]], losses: dict[str, float], 
                         model_colors: dict[str, str], 
                         title: str = "VAE Training Loss Curves", 
                         save_path: Path | str | None = None) -> None:
    """Plot training loss curves for multiple models.

    Parameters
    ----------
    histories: {model_label: [per-epoch loss]}.
    losses: {model_label: best_loss_value}.
    model_colors : {model_label: hex color string}.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, history in histories.items():
        color = model_colors.get(label, "#555")
        best = losses.get(label, history[-1] if history else 0)
        ax.plot(history, color=color, linewidth=2, label=f"{label} ({best:.4f})")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)


# ----------------------------------------------------------------------------
# Elbow method
# ----------------------------------------------------------------------------

def plot_elbow(elbow: dict[str, list], true_k: int, title: str = "Elbow Method", 
               save_path: Path | str | None = None) -> None:
    """Plot inertia, silhouette, and CH index vs K with a vertical line at true K."""
    ks = elbow["k_range"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, (vals, ylabel, subtitle) in zip(
        axes,
        [
            (elbow["inertias"], "Inertia", "Inertia ↓"),
            (elbow["sil_scores"], "Silhouette Score", "Silhouette ↑"),
            (elbow["ch_scores"], "Calinski-Harabasz", "CH Index ↑"),
        ],
    ):
        ax.plot(ks, vals, "o-", color="#1565C0", linewidth=2, markersize=5)
        ax.axvline(
            true_k, color="red", linestyle="--", alpha=0.7,
            label=f"K={true_k} (true genres)"
        )
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)


# ----------------------------------------------------------------------------
# Latent space: t-SNE / UMAP
# ----------------------------------------------------------------------------

def plot_tsne_umap(Z_2d: np.ndarray, y_genre: np.ndarray, y_lang: np.ndarray, 
                   genre_classes: list[str], lang_classes: list[str], 
                   projection: str = "t-SNE", title: str = "", 
                   save_path: Path | str | None = None) -> None:
    """Two-panel scatter: left coloured by genre, right coloured by language.

    Parameters
    ----------
    Z_2d: 2-D reduced latent array (N, 2).
    y_genre: Integer genre labels (N,).
    y_lang: Integer language labels (N,).
    genre_classes: List of genre name strings.
    lang_classes: List of language name strings.
    projection: Label for axis titles (e.g. "t-SNE", "UMAP").
    """
    palette = plt.cm.tab20.colors
    lang_colors = ["#f7914e", "#4e8ef7"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    # Left: by genre
    for gi, genre in enumerate(genre_classes):
        mask = y_genre == gi
        axes[0].scatter(
            Z_2d[mask, 0], Z_2d[mask, 1],
            c=[palette[gi % len(palette)]], label=genre, alpha=0.5, s=8,
        )
    axes[0].set_title(f"{projection}: Latent Space (by Genre)", fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[0].set_xlabel(f"{projection} 1")
    axes[0].set_ylabel(f"{projection} 2")

    # Right: by language
    for lid, lang in enumerate(lang_classes):
        mask = y_lang == lid
        axes[1].scatter(
            Z_2d[mask, 0], Z_2d[mask, 1],
            c=lang_colors[lid % len(lang_colors)], label=lang, alpha=0.5, s=8,
        )
    axes[1].set_title(f"{projection}: Latent Space (by Language)", fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].set_xlabel(f"{projection} 1")
    axes[1].set_ylabel(f"{projection} 2")

    plt.tight_layout()
    _save(fig, save_path)


# -----------------------------------------------------------------------------
# Cluster composition heatmap
# -----------------------------------------------------------------------------

def plot_cluster_composition(cluster_labels: np.ndarray, y_labels: np.ndarray, 
                             model_name: str = "VAE", 
                             title: str = "Cluster Composition", 
                             save_path: Path | str | None = None) -> None:
    """Heatmap showing genre % within each K-Means cluster."""
    df = pd.DataFrame({"cluster": cluster_labels, "genre": y_labels})
    ct = pd.crosstab(df["cluster"], df["genre"])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(max(8, ct_pct.shape[1]), 6))
    sns.heatmap(
        ct_pct, ax=ax, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.3, cbar_kws={"label": "%", "shrink": 0.8},
        annot_kws={"size": 7}
    )
    ax.set_title(f"{title} — {model_name}\n(genre % within each cluster)", fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    plt.tight_layout()
    _save(fig, save_path)


# ----------------------------------------------------------------------------
# Language separation
# -----------------------------------------------------------------------------

def plot_language_separation(Z_umap: np.ndarray, lang_labels: np.ndarray, 
                             lang_colors: dict[str, str], 
                             lang_markers: dict[str, str], 
                             title: str = "English vs Bangla - UMAP Latent Space", 
                             save_path: Path | str | None = None) -> None:
    """Scatter plot of English vs Bangla in 2-D UMAP space."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for lang, color in lang_colors.items():
        mask = lang_labels == lang
        ax.scatter(
            Z_umap[mask, 0], Z_umap[mask, 1],
            c=color, marker=lang_markers.get(lang, "o"),
            s=10, alpha=0.55, label=lang, linewidths=0
        )
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=10, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.15)
    plt.tight_layout()
    _save(fig, save_path)


# ----------------------------------------------------------------------------
# Metric comparison bar chart
# -----------------------------------------------------------------------------

def plot_metrics_comparison(metrics_dict: dict[str, dict[str, float]], 
                            metric_keys: list[str], metric_labels: list[str], 
                            title: str = "Model Comparison", 
                            save_path: Path | str | None = None) -> None:
    """Bar chart comparing metric values across methods.

    Parameters
    ----------
    metrics_dict: {method_name: {metric_key: value}}.
    metric_keys: Keys to plot from each method dict.
    metric_labels: Display labels for each metric.
    """
    methods = list(metrics_dict.keys())
    n_metrics = len(metric_keys)
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    bar_colors = plt.cm.tab10.colors
    for ax, mk, ml in zip(axes, metric_keys, metric_labels):
        vals = [metrics_dict[m].get(mk, float("nan")) for m in methods]
        bars = ax.bar(x, vals, color=bar_colors[: len(methods)], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
        ax.set_title(ml, fontweight="bold")
        ax.set_ylabel(ml)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(v) * 0.02,
                    f"{v:.4f}", ha="center", fontsize=8, fontweight="bold"
                )
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)


# ----------------------------------------------------------------------------
# Full metrics heatmap
# -----------------------------------------------------------------------------

def plot_metrics_heatmap(df_metrics: pd.DataFrame, 
                         save_path: Path | str | None = None) -> None:
    """Heatmap dashboard of all 6 metrics x models x datasets.

    Parameters
    ----------
    df_metrics: DataFrame with columns:
                Dataset, Model / Features, Algorithm, Silhouette,
                Davies-Bouldin, Calinski-H, NMI, ARI, Purity.
    """
    metric_cfg = [
        ("Silhouette", "Blues", "↑ better"),
        ("NMI", "Greens", "↑ better"),
        ("ARI", "Purples", "↑ better"),
        ("Purity", "Oranges", "↑ better"),
        ("Davies-Bouldin", "Reds_r", "↓ better"),
        ("Calinski-H", "YlGn", "↑ better")
    ]
    model_col = "Model" if "Model" in df_metrics.columns else "Features"

    df_km = df_metrics[df_metrics["Algorithm"] == "KMeans"].copy()
    n_panels = len(metric_cfg)
    fig, axes = plt.subplots(2, 3, figsize=(26, 14))
    fig.suptitle(
        "KMeans Clustering Quality Heatmap\nRows=Dataset | Cols=Model",
        fontsize=14, fontweight="bold"
    )

    for ax, (metric, cmap, note) in zip(axes.flat, metric_cfg):
        if metric not in df_km.columns:
            ax.set_visible(False)
            continue
        pivot = df_km.pivot_table(
            index="Dataset", columns=model_col,
            values=metric, aggfunc="mean"
        )
        sns.heatmap(
            pivot.astype(float), ax=ax, annot=True, fmt=".3f",
            cmap=cmap, linewidths=0.4, linecolor="white",
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title(f"{metric}  ({note})", fontweight="bold", fontsize=11)
        ax.set_xlabel(model_col)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    _save(fig, save_path)


# ----------------------------------------------------------------------------
# Beta-VAE disentanglement
# -----------------------------------------------------------------------------

def plot_disentanglement(Z_mlp: np.ndarray, Z_beta: np.ndarray, y_true: np.ndarray, 
                         genre_classes: list[str], beta_val: float = 4.0, 
                         n_show: int = 8, title: str = "Latent Dimension Distributions", 
                         save_path: Path | str | None = None) -> None:
    """Histogram of first *n_show* latent dimensions for MLP-VAE vs Beta-VAE."""
    n_show = min(n_show, Z_mlp.shape[1])
    n_class = len(genre_classes)
    palette = plt.cm.tab20.colors

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 3.5, 7))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for di in range(n_show):
        for row, (Z, label) in enumerate(
            [(Z_mlp, "MLP-VAE"), (Z_beta, f"Beta-VAE (β={beta_val:.0f})")]
        ):
            ax = axes[row, di]
            for gi in range(n_class):
                vals = Z[y_true == gi, di]
                if len(vals) > 0:
                    ax.hist(
                        vals, bins=20, alpha=0.55, color=palette[gi % len(palette)],
                        density=True,
                        label=genre_classes[gi] if di == 0 else None
                    )
            ax.set_title(
                f"dim {di}\n{label}" if di == 0 else f"dim {di}", fontsize=8
            )
            ax.set_yticks([])
            ax.grid(alpha=0.2)

    if n_show > 0:
        axes[0, 0].legend(fontsize=6, loc="upper right", title="genre", title_fontsize=6)
    plt.tight_layout()
    _save(fig, save_path)


# ----------------------------------------------------------------------------
# Reconstruction examples
# ----------------------------------------------------------------------------

def plot_reconstruction(X_sc: np.ndarray, models: dict[str, Any], y_labels: np.ndarray, 
                        le: Any, n_show: int = 6, n_dims_show: int = 60, 
                        title: str = "Reconstruction Examples", 
                        save_path: Path | str | None = None, device: Any = None) -> None:
    """Plot original vs reconstructed feature vectors for random samples.

    Parameters
    ----------
    X_sc: Scaled feature matrix.
    models: {model_key: trained_model}. Keys: "mlp", "beta", "cvae".
    y_labels: Genre string labels (N,).
    le: Fitted LabelEncoder.
    n_show: Number of random samples to display.
    n_dims_show: Number of feature dimensions to plot per sample.
    device: Torch device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_items = [
        ("mlp",  models.get("mlp"), "MLP-VAE"),
        ("beta", models.get("beta"), "Beta-VAE"),
        ("cvae", models.get("cvae"), "CVAE")
    ]
    model_items = [(k, m, l) for k, m, l in model_items if m is not None]
    n_cols = len(model_items)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_sc), n_show, replace=False)

    fig, axes = plt.subplots(n_show, n_cols, figsize=(6 * n_cols, n_show * 2.5))
    if n_show == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    n_class = len(le.classes_)
    x_axis = np.arange(min(n_dims_show, X_sc.shape[1]))

    for row, i in enumerate(idx):
        x_orig = torch.FloatTensor(X_sc[i : i + 1]).to(device)
        genre  = y_labels[i]

        for col, (mkey, model, mlabel) in enumerate(model_items):
            model.eval()
            with torch.no_grad():
                if mkey == "cvae":
                    c = torch.zeros(1, n_class, device=device)
                    g_id = le.transform([genre])[0]
                    c[0, g_id] = 1.0
                    recon, _, _, _ = model(x_orig, c)
                elif mkey == "ae":
                    recon, _ = model(x_orig)
                else:
                    recon, _, _, _ = model(x_orig)

            orig_np = x_orig.cpu().numpy().flatten()
            recon_np = recon.cpu().numpy().flatten()
            dim_end = len(x_axis)

            ax = axes[row][col] if n_show > 1 else axes[col]
            ax.plot(x_axis, orig_np[:dim_end],  color="#1565C0", lw=1.5,
                    label="Original" if row == 0 else None)
            ax.plot(x_axis, recon_np[:dim_end], color="#FF5722", lw=1.5,
                    linestyle="--", label="Recon" if row == 0 else None)
            mse = float(np.mean((orig_np - recon_np) ** 2))
            ax.set_title(f"{mlabel} | {str(genre)[:12]} | MSE={mse:.4f}", fontsize=7)
            ax.set_yticks([])
            ax.grid(alpha=0.2)
            if row == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    _save(fig, save_path)


# ----------------------------------------------------------------------------
# Extension visualisations
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# GMVAE - Gaussian Mixture components vs true genres
# ----------------------------------------------------------------------------

def plot_gmvae_results(gmvae_results: dict[str, dict], all_results: dict[str, dict], 
                       save_path: Path | str | None = None) -> None:
    """Four-panel grid per dataset: t-SNE/UMAP x true genre/GMVAE component.

    Parameters
    ----------
    gmvae_results: {ds_key: {tsne, umap, labels, metrics, y_true, K, model}}.
    all_results: Main experiment results dict with MLP-VAE/PCA metrics for comparison.
    save_path: Output file path.
    """
    n_ds = len(gmvae_results)
    fig, axes = plt.subplots(n_ds, 4, figsize=(26, n_ds * 6))
    if n_ds == 1:
        axes = [axes]
    fig.suptitle("GMVAE: Learned Mixture Components vs True Genres",
                 fontsize=14, fontweight="bold")

    for row, (ds_key, gr) in enumerate(gmvae_results.items()):
        pal = plt.cm.get_cmap("tab20", gr["K"])

        for col, (Z2, title) in enumerate([
            (gr["tsne"], "t-SNE | True Genre"),
            (gr["tsne"], "t-SNE | GMVAE Component"),
            (gr["umap"], "UMAP | True Genre"),
            (gr["umap"], "UMAP | GMVAE Component")
        ]):
            ax = axes[row][col]
            if col in (0, 2):   # colour by true genre
                for gi in range(gr["K"]):
                    m = gr["y_true"] == gi
                    if m.any():
                        ax.scatter(Z2[m, 0], Z2[m, 1], c=[pal(gi)],
                                   s=8, alpha=0.65, linewidths=0)
            else:               # colour by GMVAE component
                ax.scatter(Z2[:, 0], Z2[:, 1], c=gr["labels"],
                           cmap="tab20", s=8, alpha=0.65, linewidths=0)
                # Annotate component centroids
                for k in range(gr["K"]):
                    mask = gr["labels"] == k
                    if mask.any():
                        cx = Z2[mask, 0].mean()
                        cy = Z2[mask, 1].mean()
                        ax.text(cx, cy, str(k), fontsize=7, fontweight="bold",
                                ha="center", va="center",
                                bbox=dict(boxstyle="round,pad=0.2",
                                          facecolor="white", alpha=0.7,
                                          edgecolor="grey"))

            subtitle = f"{ds_key} | {title}"
            if col == 1:
                m_ = gr["metrics"]
                subtitle += f"\nSil={m_['sil']:.3f} NMI={m_['nmi']:.3f} ARI={m_['ari']:.3f}"
            ax.set_title(subtitle, fontsize=8, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.15)

    plt.tight_layout()
    _save(fig, save_path)

    # Comparison table
    print(f"\n{'='*70}")
    print("  GMVAE vs MLP-VAE vs PCA — KMeans Clustering Quality")
    print(f"{'='*70}")
    print(f"  {'Dataset':<8} {'Model':<14} {'Sil':>8} {'DB':>8} {'NMI':>8} {'ARI':>8} {'Purity':>8}")
    print("  " + "-" * 64)
    for ds_key, gr in gmvae_results.items():
        for mname, m in [
            ("GMVAE", gr["metrics"]),
            ("MLP-VAE", all_results[ds_key]["cl"]["MLP-VAE"]["KMeans"]),
            ("PCA", all_results[ds_key]["cl"]["PCA"]["KMeans"])
        ]:
            print(f"  {ds_key:<8} {mname:<14} "
                  f"{m['sil']:>8.4f} {m['db']:>8.4f} "
                  f"{m['nmi']:>8.4f} {m['ari']:>8.4f} {m['purity']:>8.4f}")
        print()


# ----------------------------------------------------------------------------
# β-sensitivity analysis
# ----------------------------------------------------------------------------

def plot_beta_sensitivity(beta_results: dict[str, dict], 
                          save_path: Path | str | None = None) -> None:
    """Six-panel plot of metric vs β for each dataset.

    Parameters
    ----------
    beta_results: {ds_key: {beta_val: {metrics: {…}, Z: …}}}.
    save_path: Output file path.
    """
    ds_colors = {"FMA": "#1565C0", "LMD": "#2E7D32", "GTZAN": "#C62828"}
    metric_cfg = [
        ("recon_loss", "Reconstruction Loss (↓)", False),
        ("sil", "Silhouette Score (↑)", True),
        ("nmi", "NMI (↑)", True),
        ("ari", "ARI (↑)", True),
        ("purity", "Cluster Purity (↑)", True),
        ("db", "Davies-Bouldin (↓)", False)
    ]

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle(
        "β-VAE Sensitivity Analysis\n"
        "Reconstruction vs Disentanglement vs Clustering Quality",
        fontsize=14, fontweight="bold"
    )

    for ax, (metric, ylabel, ascending) in zip(axes.flat, metric_cfg):
        for ds_key, color in ds_colors.items():
            if ds_key not in beta_results:
                continue
            betas = sorted(beta_results[ds_key].keys())
            vals = [beta_results[ds_key][b]["metrics"].get(metric, float("nan")) for b in betas]
            ax.plot(betas, vals, "o-", color=color, lw=2.5, markersize=7, label=ds_key)
            best_i = int(np.argmin(vals) if not ascending else np.argmax(vals))
            ax.axvline(betas[best_i], color=color, lw=0.8, linestyle=":", alpha=0.7)

        ax.set_xlabel("β value (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold", fontsize=11)
        ax.set_xscale("log")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.axvspan(1.0, 4.0, alpha=0.07, color="green")

    plt.tight_layout()
    _save(fig, save_path)

    print("\nOptimal β per dataset (maximises Silhouette):")
    for ds_key, ds_data in beta_results.items():
        betas = sorted(ds_data.keys())
        sils = [ds_data[b]["metrics"]["sil"] for b in betas]
        best_b = betas[int(np.argmax(sils))]
        print(f"  {ds_key:<8} → β={best_b:.1f}  (Sil={max(sils):.4f})")


# ----------------------------------------------------------------------------
# MIG scores
# ----------------------------------------------------------------------------

def plot_mig_scores(mig_results: dict[str, dict], 
                    save_path: Path | str | None = None) -> None:
    """Horizontal bar chart of MIG score per model per dataset,
    plus per-dimension MI heatmap for Beta-VAE.

    Parameters
    ----------
    mig_results: {ds_key: {model_name: {mig: float, mi_per_dim: ndarray}}}.
    save_path: Output file path.
    """
    mig_colors = {
        "MLP-VAE": "#1565C0", "Beta-VAE": "#6A1B9A", "CVAE": "#00838F",
        "Conv-VAE": "#2E7D32", "AE": "#E65100", "PCA": "#B71C1C", "GMVAE": "#FF8F00"
    }
    model_order = ["MLP-VAE", "Beta-VAE", "CVAE", "Conv-VAE", "AE", "PCA", "GMVAE"]

    n_ds = len(mig_results)
    fig, axes = plt.subplots(1, n_ds, figsize=(8 * n_ds, 7))
    if n_ds == 1:
        axes = [axes]
    fig.suptitle(
        "MIG (Mutual Information Gap) — Disentanglement Score per Model\n"
        "Higher = More Disentangled Latent Space",
        fontsize=14, fontweight="bold"
    )

    for ax, (ds_key, res_mig) in zip(axes, mig_results.items()):
        models = [m for m in model_order if m in res_mig]
        migs = [res_mig[m]["mig"] for m in models]
        colors = [mig_colors.get(m, "#555") for m in models]

        bars = ax.barh(models, migs, color=colors, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, migs):
            ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=10, fontweight="bold")

        ax.set_xlim(0, (max(migs) if migs else 0.1) * 1.3 + 0.01)
        ax.set_title(ds_key, fontweight="bold", fontsize=13)
        ax.set_xlabel("MIG Score (↑)")
        ax.axvline(0.1, color="grey", lw=1, linestyle="--", alpha=0.5)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)

    # Per-dim MI heatmap for Beta-VAE
    fig2, axes2 = plt.subplots(1, n_ds, figsize=(8 * n_ds, 5))
    if n_ds == 1:
        axes2 = [axes2]
    fig2.suptitle(
        "Mutual Information per Latent Dimension — Beta-VAE\n"
        "Ideally one spike per genre factor",
        fontsize=13, fontweight="bold"
    )

    for ax, (ds_key, res_mig) in zip(axes2, mig_results.items()):
        if "Beta-VAE" not in res_mig:
            continue
        mi_dims = res_mig["Beta-VAE"]["mi_per_dim"]
        n_show = min(20, len(mi_dims))
        top_idx = np.argsort(mi_dims)[::-1][:n_show]
        top_mi = mi_dims[top_idx]

        ax.bar(range(n_show), top_mi, color="#6A1B9A", alpha=0.8)
        ax.set_xticks(range(n_show))
        ax.set_xticklabels([f"z{i}" for i in top_idx], rotation=45, fontsize=8)
        ax.set_title(f"{ds_key} - Beta-VAE MI per dim", fontweight="bold")
        ax.set_ylabel("MI with genre label")
        ax.grid(axis="y", alpha=0.3)
        if len(top_mi) > 1:
            ax.text(0, top_mi[0] * 0.95,
                    f"z{top_idx[0]}\nMIG gap={top_mi[0]-top_mi[1]:.3f}",
                    ha="center", va="top", fontsize=9,
                    color="white", fontweight="bold")

    plt.tight_layout()
    mig_dim_path = None
    if save_path is not None:
        p = Path(save_path)
        mig_dim_path = p.parent / (p.stem + "_per_dim" + p.suffix)
    _save(fig2, mig_dim_path)


# ----------------------------------------------------------------------------
# Latent space interpolation (SLERP)
# ----------------------------------------------------------------------------

def plot_interpolation(all_results: dict[str, dict], n_interp: int = 12, 
                       save_dir: Path | str | None = None) -> None:
    """SLERP feature-evolution heatmap and trajectory in t-SNE space.

    Parameters
    ----------
    all_results: Main experiment results dict; each value must have
                  keys: Z['mlp'], le, y_true, name, vis['mlp']['tsne'].
    n_interp: Number of interpolation steps.
    save_dir: Directory to save per-dataset PNGs.
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    def _slerp(z1: np.ndarray, z2: np.ndarray, t: float) -> np.ndarray:
        z1_n = z1 / (np.linalg.norm(z1) + 1e-8)
        z2_n = z2 / (np.linalg.norm(z2) + 1e-8)
        dot  = float(np.clip(np.dot(z1_n, z2_n), -1.0, 1.0))
        theta = np.arccos(dot)
        if abs(theta) < 1e-6:
            return (1 - t) * z1 + t * z2
        return (np.sin((1 - t) * theta) / np.sin(theta)) * z1 + \
               (np.sin(t * theta) / np.sin(theta)) * z2

    for ds_key, res in all_results.items():
        Z_mlp = res["Z"]["mlp"]
        le = res["le"]
        y_true = res["y_true"]
        classes = list(le.classes_)
        n_cl = len(classes)
        step = max(1, n_cl // 4)
        pairs = [(classes[i], classes[i + step])
                   for i in range(0, min(4 * step, n_cl - step), step)]
        if not pairs:
            pairs = [(classes[0], classes[-1])]

        n_pairs = len(pairs)
        fig, axes = plt.subplots(n_pairs, 1, figsize=(20, n_pairs * 3.5))
        if n_pairs == 1:
            axes = [axes]
        fig.suptitle(
            f"Latent Space Interpolation (SLERP) — {res['name']}\n"
            "MLP-VAE: Feature Evolution between Genre Centroids",
            fontsize=13, fontweight="bold"
        )

        for ax, (ga, gb) in zip(axes, pairs):
            if ga not in classes or gb not in classes:
                ax.text(0.5, 0.5, "Genre pair not found",
                        ha="center", va="center", transform=ax.transAxes)
                continue
            ia = le.transform([ga])[0]
            ib = le.transform([gb])[0]
            z_a = Z_mlp[y_true == ia].mean(axis=0)
            z_b = Z_mlp[y_true == ib].mean(axis=0)

            t_vals = np.linspace(0, 1, n_interp)
            z_path = np.array([_slerp(z_a, z_b, t) for t in t_vals])
            recon_np = res["models"]["mlp"].decode(
                torch.FloatTensor(z_path).to(
                    next(res["models"]["mlp"].parameters()).device
                )
            ).detach().cpu().numpy()

            show_dim = min(40, recon_np.shape[1])
            img = ax.imshow(recon_np[:, :show_dim].T, aspect="auto",
                            cmap="RdYlBu_r", interpolation="bilinear")
            plt.colorbar(img, ax=ax, fraction=0.02, pad=0.01, label="Feature value (std)")
            ax.axvline(0, color="#1565C0", lw=2.5, alpha=0.8)
            ax.axvline(n_interp - 1, color="#C62828", lw=2.5, alpha=0.8)
            ax.set_title(f"{ga}  →→→  {gb}", fontweight="bold", fontsize=11)
            ax.set_xlabel(f"Interpolation step (0={ga}, {n_interp-1}={gb})")
            ax.set_ylabel("Feature dim")
            mid = n_interp // 2
            ax.axvline(mid, color="grey", lw=1, linestyle="--", alpha=0.6)
            ax.text(mid, show_dim * 0.05, "mid", ha="center", fontsize=8, color="grey")

        plt.tight_layout()
        sp = (save_dir / f"interpolation_{ds_key.lower()}.png") if save_dir else None
        _save(fig, sp)

        # 2-D trajectory in t-SNE space
        tsne2 = res["vis"]["mlp"]["tsne"]
        n_show = min(3, len(pairs))
        step2 = max(1, n_cl // 3)
        pairs2 = [(classes[i], classes[i + step2])
                  for i in range(0, min(n_show * step2, n_cl - step2), step2)]

        fig2, ax2 = plt.subplots(figsize=(12, 9))
        pal = plt.cm.get_cmap("tab20", n_cl)
        for gi in range(n_cl):
            m = y_true == gi
            ax2.scatter(tsne2[m, 0], tsne2[m, 1], c=[pal(gi)],
                        s=8, alpha=0.35, linewidths=0)

        for ga, gb in pairs2:
            if ga not in classes or gb not in classes:
                continue
            ia = le.transform([ga])[0]
            ib = le.transform([gb])[0]
            ca = tsne2[y_true == ia].mean(axis=0)
            cb = tsne2[y_true == ib].mean(axis=0)
            ax2.annotate("", xy=cb, xytext=ca,
                         arrowprops=dict(arrowstyle="->", color="black", lw=2))
            ax2.scatter(*ca, c="black", s=80, zorder=5, marker="*")
            ax2.scatter(*cb, c="black", s=80, zorder=5, marker="*")
            ax2.text(ca[0], ca[1] + 1, ga[:8], fontsize=8, ha="center", fontweight="bold")
            ax2.text(cb[0], cb[1] + 1, gb[:8], fontsize=8, ha="center", fontweight="bold")

        ax2.set_title(f"Interpolation Paths in t-SNE Space — {res['name']}",
                      fontweight="bold", fontsize=12)
        ax2.set_xticks([]); ax2.set_yticks([]); ax2.grid(alpha=0.15)
        sp2 = (save_dir / f"interp_path_{ds_key.lower()}.png") if save_dir else None
        _save(fig2, sp2)


# ----------------------------------------------------------------------------
# Cross-dataset transfer
# ----------------------------------------------------------------------------

def plot_transfer_results(transfer_results: dict[str, dict], 
                          save_path: Path | str | None = None) -> None:
    """Grouped bar chart of transfer vs native metrics plus retention bars.

    Parameters
    ----------
    transfer_results: {pair_key: {metrics, native, retention, …}}.
    save_path: Output file path.
    """
    pairs = list(transfer_results.keys())
    metrics_ = ["sil", "nmi", "ari", "purity"]
    m_labels = ["Silhouette", "NMI", "ARI", "Purity"]
    x = np.arange(len(pairs))
    w = 0.35

    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    fig.suptitle(
        "Cross-Dataset Transfer vs Native Performance\n"
        "VAE trained on Source — evaluated on Target (zero-shot)",
        fontsize=14, fontweight="bold"
    )

    for ax, (metric, mlabel) in zip(axes, zip(metrics_, m_labels)):
        nat_vals = [transfer_results[p]["native"].get(metric, 0.0) for p in pairs]
        tra_vals = [transfer_results[p]["metrics"].get(metric, 0.0) for p in pairs]

        b1 = ax.bar(x - w / 2, nat_vals, w, label="Native (target-trained)",
                    color="#1565C0", alpha=0.85)
        b2 = ax.bar(x + w / 2, tra_vals, w, label="Transfer (zero-shot)",
                    color="#FF8F00", alpha=0.85)

        for bar, v in zip(list(b1) + list(b2), nat_vals + tra_vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(pairs, rotation=25, fontsize=9)
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)

    # Retention bar chart
    retentions = [transfer_results[p]["retention"] * 100 for p in pairs]
    colors_ret = ["#2E7D32" if r >= 70 else "#F57F17" if r >= 50 else "#C62828"
                   for r in retentions]

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    bars = ax2.bar(pairs, retentions, color=colors_ret, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, retentions):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax2.axhline(70, color="green",  lw=1.5, linestyle="--", alpha=0.7, label="Good (70%)")
    ax2.axhline(50, color="orange", lw=1.5, linestyle="--", alpha=0.7, label="Acceptable (50%)")
    ax2.set_ylabel("Silhouette Retention (%)")
    ax2.set_ylim(0, 120)
    ax2.set_title("Zero-Shot Transfer Retention\n% of native Silhouette preserved",
                  fontweight="bold", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    ret_path = None
    if save_path is not None:
        p = Path(save_path)
        ret_path = p.parent / (p.stem + "_retention" + p.suffix)
    _save(fig2, ret_path)


# ----------------------------------------------------------------------------
# Contrastive VAE visualisations
# ----------------------------------------------------------------------------

def plot_contrastive_results(cvae_con_results: dict[str, dict], 
                             all_results: dict[str, dict], 
                             save_path: Path | str | None = None) -> None:
    """Three figures: latent comparison, λ-sensitivity, training curve decomposition.

    Parameters
    ----------
    cvae_con_results: {ds_key: {tsne, umap, metrics, hist, best_lam, lambda_sweep, y_true, K}}.
    all_results: Main experiment results for MLP-VAE comparison.
    save_path: Base output path; suffixes _lambda and _curves are appended.
    """
    n_ds = len(cvae_con_results)

    # Figure 1: latent space comparison
    fig, axes = plt.subplots(n_ds, 4, figsize=(26, n_ds * 6))
    if n_ds == 1:
        axes = [axes]
    fig.suptitle("MLP-VAE vs Contrastive VAE — Latent Space Quality",
                 fontsize=14, fontweight="bold")

    for row, (ds_key, cr) in enumerate(cvae_con_results.items()):
        pal = plt.cm.get_cmap("tab20", cr["K"])
        mlp_m = all_results[ds_key]["cl"]["MLP-VAE"]["KMeans"]

        for col, (Z2, title, sil_, nmi_) in enumerate([
            (all_results[ds_key]["vis"]["mlp"]["tsne"], "MLP-VAE t-SNE",
             mlp_m["sil"], mlp_m["nmi"]),
            (all_results[ds_key]["vis"]["mlp"]["umap"], "MLP-VAE UMAP",
             mlp_m["sil"], mlp_m["nmi"]),
            (cr["tsne"], "ContrastiveVAE t-SNE", cr["metrics"]["sil"], cr["metrics"]["nmi"]),
            (cr["umap"], "ContrastiveVAE UMAP",  cr["metrics"]["sil"], cr["metrics"]["nmi"]),
        ]):
            ax = axes[row][col]
            for gi in range(cr["K"]):
                m = cr["y_true"] == gi
                if m.any():
                    ax.scatter(Z2[m, 0], Z2[m, 1], c=[pal(gi)],
                               s=8, alpha=0.65, linewidths=0)
            ax.set_title(f"{ds_key}\n{title}  Sil={sil_:.3f} NMI={nmi_:.3f}",
                         fontsize=8, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.15)

    plt.tight_layout()
    _save(fig, save_path)

    # Figure 2: λ-sensitivity
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
    fig2.suptitle("Contrastive VAE — λ (InfoNCE weight) Sensitivity",
                  fontsize=13, fontweight="bold")

    for ax, (metric, ylabel) in zip(axes2, [("sil", "Silhouette (↑)"),
                                             ("nmi", "NMI (↑)"),
                                             ("ari", "ARI (↑)")]):
        for ds_key, cr in cvae_con_results.items():
            lams = sorted(cr["lambda_sweep"].keys())
            vals = [cr["lambda_sweep"][l]["metrics"][metric] for l in lams]
            ax.plot(lams, vals, "o-", lw=2.5, markersize=8, label=ds_key)
            ax.scatter([cr["best_lam"]], [cr["lambda_sweep"][cr["best_lam"]]["metrics"][metric]],
                       s=120, zorder=5, marker="*", edgecolors="black", linewidths=1)
        ax.set_xlabel("λ (InfoNCE weight)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    lam_path = None
    if save_path is not None:
        p = Path(save_path)
        lam_path = p.parent / (p.stem + "_lambda" + p.suffix)
    _save(fig2, lam_path)

    # Figure 3: training curve decomposition
    fig3, axes3 = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5))
    if n_ds == 1:
        axes3 = [axes3]
    fig3.suptitle("Contrastive VAE — Training Loss Decomposition",
                  fontsize=13, fontweight="bold")

    for ax, (ds_key, cr) in zip(axes3, cvae_con_results.items()):
        h = cr["hist"]
        ep = range(1, len(h["total"]) + 1)
        ax.plot(ep, h["total"], color="black", lw=2.5, label="Total")
        ax.plot(ep, h["elbo"], color="#1565C0", lw=2, label="ELBO", linestyle="--")
        ax.plot(ep, h["infonce"], color="#C62828", lw=2, label="InfoNCE", linestyle=":")
        ax.set_title(ds_key, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    curves_path = None
    if save_path is not None:
        p = Path(save_path)
        curves_path = p.parent / (p.stem + "_curves" + p.suffix)
    _save(fig3, curves_path)


# ----------------------------------------------------------------------------
# DANN-VAE visualisations
# ----------------------------------------------------------------------------

def plot_dann_results(dann_results: dict[str, dict], dann_hist: dict[str, list[float]], 
                      X_dann: np.ndarray, d_dann: np.ndarray, dann_model: Any, 
                      all_results: dict[str, dict], device: Any, batch_size: int = 256, 
                      save_path: Path | str | None = None) -> float:
    """Domain alignment visualisation: full dataset colored by domain/genre,
    domain confusion bar, training curves, and per-domain cluster plots.

    Parameters
    ----------
    dann_results: {ds_key: {Z, metrics, labels, tsne, y_true, K}} + '_full' key.
    dann_hist: Training history with keys 'total', 'elbo', 'domain'.
    X_dann: Combined aligned feature matrix.
    d_dann: Domain label array.
    dann_model: Trained DANN-VAE model.
    all_results: Main experiment results for native comparison.
    device: Torch device.
    batch_size: Inference batch size.
    save_path: Output file path.

    Returns
    -------
    domain_classifier_accuracy (%) — how well the domain clf can identify datasets
    (lower = more domain-invariant).
    """
    # Compute domain classifier accuracy on latent Z
    dann_model.eval()
    X_t_full = torch.FloatTensor(X_dann)
    d_preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X_t_full), batch_size):
            bx = X_t_full[i : i + batch_size].to(device)
            mu, _ = dann_model.enc(bx)
            logits = dann_model.domain_clf(mu)
            d_preds.append(logits.argmax(dim=1).cpu().numpy())
    d_preds_all = np.concatenate(d_preds)
    d_acc = float((d_preds_all == d_dann).mean() * 100)
    chance = 100.0 / 3

    full = dann_results.get("_full", {})
    tsne_f = full.get("tsne", np.zeros((1, 2)))
    domain_data = [(k, v) for k, v in dann_results.items() if k != "_full"]

    DS_COLORS_D = {0: "#1565C0", 1: "#2E7D32", 2: "#C62828"}
    DS_NAMES_D  = {0: "FMA", 1: "LMD", 2: "GTZAN"}

    fig = plt.figure(figsize=(28, 20))
    fig.suptitle(
        "DANN-VAE: Domain-Invariant Latent Space\n"
        "Top: All 3 datasets in shared space | Bottom: Per-domain clusters",
        fontsize=14, fontweight="bold",
    )
    outer = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

    # Row 0, Col 0: all domains by dataset colour
    ax = fig.add_subplot(outer[0, 0])
    for d_id, color in DS_COLORS_D.items():
        m = full.get("d_labels", d_dann) == d_id
        ax.scatter(tsne_f[m, 0], tsne_f[m, 1], c=color, s=6,
                   alpha=0.5, label=DS_NAMES_D[d_id], linewidths=0)
    ax.set_title("All Domains - Colored by Dataset\n(mixed = domain-invariant)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.15)

    # Row 0, Col 1: same but all at once
    ax = fig.add_subplot(outer[0, 1])
    colors_per_pt = [DS_COLORS_D[int(d)] for d in (full.get("d_labels", d_dann))]
    ax.scatter(tsne_f[:, 0], tsne_f[:, 1], c=colors_per_pt, s=5, alpha=0.4, linewidths=0)
    ax.set_title("Domain Overlap Density\n(DANN forces mixture)",
                 fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.15)

    # Row 0, Col 2: domain classifier accuracy bar
    ax = fig.add_subplot(outer[0, 2])
    bar_color = "#1565C0" if d_acc < 50 else "#C62828"
    bars = ax.bar(["Chance\n(33.3%)", "DANN-VAE\ndomain acc"],
                  [chance, d_acc], color=["grey", bar_color], width=0.5, alpha=0.85)
    ax.axhline(chance, color="grey", lw=1.5, linestyle="--")
    for bar, v in zip(bars, [chance, d_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Domain Classification Accuracy (%)")
    ax.set_title("Domain Confusion\n(lower = more domain-invariant)",
                 fontsize=9, fontweight="bold")
    note = "Domain-invariant!" if d_acc < 50 else "Partial alignment"
    ax.text(0.5, 0.05, note, ha="center", transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            color="green" if d_acc < 50 else "orange")
    ax.grid(axis="y", alpha=0.3)

    # Row 0, Col 3: training curves
    ax = fig.add_subplot(outer[0, 3])
    ep = range(1, len(dann_hist["total"]) + 1)
    ax.plot(ep, dann_hist["total"], color="black", lw=2.5, label="Total")
    ax.plot(ep, dann_hist["elbo"], color="#1565C0", lw=2, label="ELBO", linestyle="--")
    ax.plot(ep, dann_hist["domain"], color="#C62828", lw=2, label="Domain", linestyle=":")
    ax.set_title("DANN-VAE Training Curves", fontsize=9, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 1: per-domain cluster scatter
    for col, (ds_key, dr) in enumerate(domain_data[:4]):
        pal = plt.cm.get_cmap("tab20", dr["K"])
        ax = fig.add_subplot(outer[1, col])
        for gi in range(dr["K"]):
            m = dr["y_true"] == gi
            if m.any():
                ax.scatter(dr["tsne"][m, 0], dr["tsne"][m, 1],
                           c=[pal(gi)], s=8, alpha=0.65, linewidths=0)
        m_ = dr["metrics"]
        ax.set_title(f"{ds_key} (DANN-VAE)\nSil={m_['sil']:.3f} NMI={m_['nmi']:.3f}",
                     fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.15)

    plt.tight_layout()
    _save(fig, save_path)

    print(f"\n  Domain classifier accuracy on latent Z: {d_acc:.1f}%")
    print(f"  Chance level: {chance:.1f}%")
    if d_acc < 50:
        print("  DANN successfully reduced domain information in Z")
    else:
        print("  Domain still partially encoded — try higher lam_domain")

    return d_acc


# ----------------------------------------------------------------------------
# Mega comparison heatmap
# ----------------------------------------------------------------------------

def plot_mega_heatmap(df_mega: pd.DataFrame, save_path: Path | str | None = None) -> None:
    """Three-panel heatmap (one per dataset) comparing all models x 4 metrics.

    Draws separator lines between Baselines, VAE Variants, and Novel Extensions.

    Parameters
    ----------
    df_mega: DataFrame with columns Dataset, Model, Type, Sil, NMI, ARI, Purity.
    save_path: Output file path.
    """
    model_order = [
        "PCA", "Spectral",
        "MLP-VAE", "Beta-VAE", "CVAE", "Conv-VAE", "AE", "Multimodal",
        "GMVAE", "ContrastiveVAE", "DANN-VAE"
    ]
    metric_order = ["Sil", "NMI", "ARI", "Purity"]
    datasets = sorted(df_mega["Dataset"].unique().tolist())

    fig, axes = plt.subplots(1, len(datasets), figsize=(13 * len(datasets), 9))
    if len(datasets) == 1:
        axes = [axes]
    fig.suptitle(
        "All Models - KMeans Clustering Quality Heatmap\n"
        "Baselines | VAE Variants | Novel Extensions",
        fontsize=15, fontweight="bold",
    )

    for ax, ds_key in zip(axes, datasets):
        sub = df_mega[df_mega["Dataset"] == ds_key]
        rows_ = []
        for mkey in model_order:
            row_ = sub[sub["Model"] == mkey]
            if len(row_) > 0:
                rows_.append({
                    "Model": mkey,
                    "Sil": float(row_["Sil"].values[0]),
                    "NMI": float(row_["NMI"].values[0]),
                    "ARI": float(row_["ARI"].values[0]),
                    "Purity": float(row_["Purity"].values[0]),
                })

        if not rows_:
            continue

        df_h = pd.DataFrame(rows_).set_index("Model")[metric_order]
        n_baseline = sum(1 for m in df_h.index if m in ("PCA", "Spectral"))
        n_vae = sum(1 for m in df_h.index
                         if m in ("MLP-VAE", "Beta-VAE", "CVAE", "Conv-VAE", "AE", "Multimodal"))

        sns.heatmap(
            df_h.astype(float), ax=ax, annot=True, fmt=".3f",
            cmap="YlOrRd", linewidths=0.5, linecolor="white",
            vmin=0, vmax=1, cbar_kws={"shrink": 0.7}
        )
        ax.set_title(ds_key, fontweight="bold", fontsize=14)
        ax.set_xlabel("Metric")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

        # Group separator lines
        ax.axhline(n_baseline, color="blue", lw=2.5, linestyle="--", alpha=0.7)
        ax.axhline(n_baseline + n_vae, color="purple", lw=2.5, linestyle="--", alpha=0.7)

        # Group labels
        ax.text(-0.15, n_baseline / 2,
                "Baselines", ha="right", va="center", fontsize=9,
                color="blue", fontweight="bold", transform=ax.transData)
        ax.text(-0.15, n_baseline + n_vae / 2,
                "VAE Variants", ha="right", va="center", fontsize=9,
                color="purple", fontweight="bold", transform=ax.transData)
        if n_baseline + n_vae < len(df_h):
            ax.text(-0.15, n_baseline + n_vae + 1.5,
                    "Novel Extensions", ha="right", va="center", fontsize=9,
                    color="darkgreen", fontweight="bold", transform=ax.transData)

    plt.tight_layout()
    _save(fig, save_path)
