"""
Visualization functions for the VAE-music project.

All functions accept a 'save_path' argument.  When provided the figure is
saved to that path before display. 

All paths are relative so no user directory information is embedded in the output.

Functions
---------
plot_dataset_distribution: Bar charts of language and genre counts.
plot_training_curves: Loss curves per model.
plot_elbow: Inertia / silhouette / CH vs K.
plot_tsne_umap: t-SNE and UMAP scatter plots.
plot_cluster_composition: Heatmap of genre % per cluster.
plot_language_separation: English vs Bangla in UMAP space.
plot_metrics_comparison: Bar chart comparing VAE vs PCA metrics.
plot_metrics_heatmap: Heatmap of all metrics x all models.
plot_disentanglement: Latent dimension histograms MLP-VAE vs Beta-VAE.
plot_reconstruction: Original vs reconstructed feature vectors.
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
