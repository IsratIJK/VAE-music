"""
Multi-algorithm clustering engine with comprehensive evaluation metrics.

Algorithms
----------
KMeans: Fast partitional clustering (spherical clusters).
AgglomerativeClustering (Ward): Hierarchical clustering.
DBSCAN: Density-based clustering with auto-tuned ε.

Metrics (all 6 returned per algorithm)
---------------------------------------
Silhouette Score - (range -1 to +1; higher = better-separated clusters)
Davies-Bouldin Index - (lower = better)
Calinski-Harabasz - (higher = better-defined clusters)
NMI - (normalised mutual information with true labels)
ARI - (adjusted rand index, corrects for chance)
Cluster Purity - (fraction of majority-class samples per cluster)

Functions
---------
compute_metrics: Compute all 6 metrics for a given label assignment.
cluster_purity: Compute cluster purity score.
run_clustering: Run all three algorithms on a latent space Z.
elbow_analysis: Inertia + silhouette + CH over a range of K values.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


def cluster_purity(y_true: np.ndarray, cluster_labels: np.ndarray) -> float:
    """Compute cluster purity.

    For each cluster the majority true-label count is summed and divided
    by the total number of non-noise points.

    Parameters
    ----------
    y_true : Integer true class labels (N,).
    cluster_labels : Predicted cluster labels (N,); -1 = noise (DBSCAN).

    Returns
    -------
    Purity score in [0, 1], or NaN if no valid clusters.
    """
    mask = cluster_labels != -1
    yt = y_true[mask]
    cl = cluster_labels[mask]

    if len(yt) == 0:
        return float("nan")

    total = 0
    for k in np.unique(cl):
        km = cl == k
        if km.sum() == 0:
            continue
        total += int(np.bincount(yt[km]).max())

    return total / len(yt)


def compute_metrics(
    Z: np.ndarray,
    y_true: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict[str, float]:
    """Compute all 6 clustering quality metrics.

    Noise points (label = -1 from DBSCAN) are excluded before computing
    metrics that require cluster membership.

    Parameters
    ----------
    Z: Latent feature matrix (N, latent_dim).
    y_true: Integer true class labels (N,).
    cluster_labels: Predicted cluster labels (N,).

    Returns
    -------
    Dict with keys: sil, db, ch, nmi, ari, purity.
    All NaN if fewer than 2 clusters or too few samples.
    """
    nan = float("nan")
    mask = cluster_labels != -1
    Zm = Z[mask]
    ym = y_true[mask]
    cm = cluster_labels[mask]
    n_cl = len(set(cm))

    if n_cl < 2 or Zm.shape[0] < n_cl + 1:
        return dict(sil=nan, db=nan, ch=nan, nmi=nan, ari=nan, purity=nan)

    return dict(
        sil = float(silhouette_score(Zm, cm)),
        db = float(davies_bouldin_score(Zm, cm)),
        ch = float(calinski_harabasz_score(Zm, cm)),
        nmi = float(normalized_mutual_info_score(ym, cm, average_method="arithmetic")),
        ari = float(adjusted_rand_score(ym, cm)),
        purity = cluster_purity(y_true, cluster_labels),
    )


def run_clustering(
    Z: np.ndarray,
    y_true: np.ndarray,
    n_class: int,
    tag: str = "",
    kmeans_ninit: int = 20,
    verbose: bool = True,
) -> dict[str, dict]:
    """Run KMeans, Agglomerative, and DBSCAN on latent matrix Z.

    DBSCAN is auto-tuned via the 90th-percentile of the 5-NN distances
    (a standard heuristic from Ester et al. 1996).

    Parameters
    ----------
    Z: Latent representation matrix (N, latent_dim).
    y_true: Integer true class labels (N,).
    n_class: Number of clusters K (= number of genres).
    tag: Optional label printed before results.
    kmeans_ninit: Number of KMeans random restarts.
    verbose: Print per-algorithm metric summary.

    Returns
    -------
    Nested dict: {algorithm_name: {labels, sil, db, ch, nmi, ari, purity, …}}
    """
    K = n_class
    results: dict[str, dict] = {}

    # -- K-Means --------------------------------------------------------------
    km = KMeans(n_clusters=K, n_init=kmeans_ninit, random_state=42).fit(Z)
    results["KMeans"] = {
        "labels": km.labels_,
        **compute_metrics(Z, y_true, km.labels_),
    }

    # -- Agglomerative (Ward linkage) ----------------------------------
    agg = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Z)
    results["Agglomerative"] = {
        "labels": agg.labels_,
        **compute_metrics(Z, y_true, agg.labels_),
    }

    # -- DBSCAN (auto DBSCAN via k-distance heuristic) ----------------------
    nn_ = NearestNeighbors(n_neighbors=5).fit(Z)
    dists, _ = nn_.kneighbors(Z)
    eps = float(np.percentile(dists[:, -1], 90))
    min_s = max(3, len(Z) // (K * 10))
    db_res = DBSCAN(eps=eps, min_samples=min_s).fit(Z)
    l_db = db_res.labels_
    results["DBSCAN"] = {
        "labels": l_db,
        "eps": eps,
        "n_found": len(set(l_db)) - (1 if -1 in l_db else 0),
        "noise_pct": float((l_db == -1).mean() * 100),
        **compute_metrics(Z, y_true, l_db),
    }

    # -- Optional console summary -----------------------------------------------
    if verbose:
        if tag:
            print(f"  [{tag}]")
        for algo, r in results.items():
            def _fmt(v: float) -> str:
                return f"{v:+.3f}" if not (isinstance(v, float) and np.isnan(v)) else "  NaN"
            extra = ""
            if algo == "DBSCAN":
                extra = (
                    f"  eps={r['eps']:.3f} noise={r['noise_pct']:.0f}%"
                    f" found={r['n_found']}"
                )
            print(
                f"    {algo:<15} Sil={_fmt(r['sil'])} DB={_fmt(r['db'])}"
                f" NMI={_fmt(r['nmi'])} ARI={_fmt(r['ari'])}"
                f" Pur={_fmt(r['purity'])}{extra}"
            )

    return results


def elbow_analysis(
    Z: np.ndarray,
    k_range: range = range(2, 16),
    kmeans_ninit: int = 10,
) -> dict[str, list]:
    """Compute inertia, silhouette, and CH index over a range of K.

    Used to select the optimal number of clusters visually (elbow method).

    Parameters
    ----------
    Z: Latent representation matrix (N, latent_dim).
    k_range: Range of K values to evaluate.
    kmeans_ninit: KMeans n_init (lower for speed).

    Returns
    -------
    Dict with keys: k_range, inertias, sil_scores, ch_scores.
    """
    inertias: list[float] = []
    sil_scores: list[float] = []
    ch_scores: list[float] = []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=kmeans_ninit, random_state=42).fit(Z)
        inertias.append(float(km.inertia_))
        sil_scores.append(float(silhouette_score(Z, km.labels_)))
        ch_scores.append(float(calinski_harabasz_score(Z, km.labels_)))

    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "sil_scores": sil_scores,
        "ch_scores": ch_scores,
    }
