"""
evaluation.py
-------------
Clustering metric computation, comparison bar chart, final report,
model + artifact saving, and Colab download trigger.
"""

import os
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, calinski_harabasz_score

warnings.filterwarnings('ignore')


# Evaluation Metrics — VAE vs PCA Baseline

def compute_metrics(Z, labels_vae, Z_pca, labels_pca):
    """
    Compute Silhouette Score and Calinski-Harabasz Index
    for both VAE+KMeans and PCA+KMeans.

    Returns
    -------
    results : pd.DataFrame  — formatted metric table
    metrics : dict          — raw float values for the report
    """
    sil_vae = silhouette_score(Z, labels_vae)
    sil_pca = silhouette_score(Z_pca, labels_pca)
    ch_vae = calinski_harabasz_score(Z, labels_vae)
    ch_pca = calinski_harabasz_score(Z_pca, labels_pca)

    results = pd.DataFrame({
        'Method': ['VAE + KMeans', 'PCA + KMeans'],
        'Silhouette Score': [round(sil_vae, 4), round(sil_pca, 4)],
        'Calinski-Harabasz': [round(ch_vae, 2), round(ch_pca, 2)],
    })

    print('EVALUATION RESULTS')
    print(results.to_string(index=False))
    print(f'Silhouette winner: {"VAE" if sil_vae > sil_pca else "PCA"}')
    print(f'CH Index winner: {"VAE" if ch_vae > ch_pca else "PCA"}')

    metrics = {
        'sil_vae': sil_vae, 'sil_pca': sil_pca,
        'ch_vae': ch_vae, 'ch_pca': ch_pca,
    }
    return results, metrics


def plot_metrics_comparison(metrics, out_dir='/content'):
    """
    Bar chart comparing Silhouette Score and Calinski-Harabasz Index
    between VAE+KMeans and PCA+KMeans.
    """
    os.makedirs(out_dir, exist_ok=True)

    sil_vae = metrics['sil_vae']
    sil_pca = metrics['sil_pca']
    ch_vae = metrics['ch_vae']
    ch_pca = metrics['ch_pca']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    methods = ['VAE + KMeans', 'PCA + KMeans']
    bar_colors = ['#4e8ef7', '#f7914e']

    axes[0].bar(methods, [sil_vae, sil_pca],
                color=bar_colors, edgecolor='white', linewidth=1.5)
    axes[0].set_title('Silhouette Score\n(higher = better)', fontweight='bold')
    axes[0].set_ylabel('Score')
    for i, v in enumerate([sil_vae, sil_pca]):
        axes[0].text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')

    axes[1].bar(methods, [ch_vae, ch_pca],
                color=bar_colors, edgecolor='white', linewidth=1.5)
    axes[1].set_title('Calinski-Harabasz Index\n(higher = better)', fontweight='bold')
    axes[1].set_ylabel('Score')
    for i, v in enumerate([ch_vae, ch_pca]):
        axes[1].text(i, v + 5, f'{v:.1f}', ha='center', fontweight='bold')

    plt.suptitle('VAE vs PCA Baseline Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: metrics_comparison.png')


# Final Report

def print_final_report(df, X_scaled, le_genre, metrics,
                        INPUT_DIM, HIDDEN_DIMS, LATENT_DIM, K, best_loss):
    """
    Print a formatted ASCII final report to stdout.

    Parameters
    ----------
    df         : pd.DataFrame — full dataset with language column
    X_scaled   : np.ndarray   — scaled feature matrix (for shape)
    le_genre   : LabelEncoder
    metrics    : dict         — from compute_metrics()
    INPUT_DIM  : int
    HIDDEN_DIMS: list
    LATENT_DIM : int
    K          : int          — number of clusters used
    best_loss  : float
    """
    sil_vae = metrics['sil_vae']
    sil_pca = metrics['sil_pca']
    ch_vae = metrics['ch_vae']
    ch_pca = metrics['ch_pca']

    print('VAE Music Clustering — FINAL REPORT (Hybrid Dataset)')
    print()
    print('DATASET')
    print(f'  English  : {(df["language"]=="English").sum()}  (GTZAN genres)')
    print(f'  Bangla   : {(df["language"]=="Bangla").sum()}  (Baul/Folk/Rabindra/Pop/Classical)')
    print(f'  Total    : {len(df)}')
    print(f'  Features : {X_scaled.shape[1]}  (MFCC+Chroma+Spectral+ZCR+RMS)')
    print(f'  Genres   : {len(le_genre.classes_)}')
    print()
    print('VAE ARCHITECTURE')
    print(f'  Input    : {INPUT_DIM}')
    print(f'  Hidden   : {HIDDEN_DIMS}')
    print(f'  Latent   : {LATENT_DIM}')
    print(f'  Best loss: {best_loss:.4f}')
    print()
    print(f'EVALUATION (K = {K})')
    print(f'  {"":20s}  {"VAE+KMeans":>12}  {"PCA+KMeans":>12}  Winner')
    print(f'  {"Silhouette Score":20s}  {sil_vae:>12.4f}  {sil_pca:>12.4f}  {"VAE" if sil_vae > sil_pca else "PCA"}')
    print(f'  {"Calinski-Harabasz":20s}  {ch_vae:>12.1f}  {ch_pca:>12.1f}  {"VAE" if ch_vae > ch_pca else "PCA"}')
    print()
    print('OUTPUT FILES SAVED')
    for fname in [
        'dataset_distribution.png',
        'training_curves.png',
        'elbow_method.png',
        'tsne_visualization.png',
        'umap_visualization.png',
        'metrics_comparison.png',
        'cluster_composition.png',
    ]:
        print(f'  /content/{fname}')


# Save & Download All Results

def save_all_results(df, Z, results, scaler,
                     best_state, INPUT_DIM, HIDDEN_DIMS, LATENT_DIM, BETA,
                     out_dir='/content'):
    """
    Save model checkpoint, latent codes, CSVs, and zip everything.
    Also triggers Colab download of the zip.

    Parameters
    ----------
    df         : pd.DataFrame — must have 'cluster_vae' and 'cluster_pca' columns
    Z          : np.ndarray   — latent representations
    results    : pd.DataFrame — metrics table from compute_metrics()
    scaler     : StandardScaler — fitted scaler
    best_state : dict         — VAE state_dict (CPU tensors)
    out_dir    : str          — base /content directory
    """
    import torch

    os.makedirs(out_dir, exist_ok=True)

    # save model checkpoint (includes scaler params)
    torch.save({
        'model_state': best_state,
        'config': {
            'input_dim': INPUT_DIM,
            'hidden_dims': HIDDEN_DIMS,
            'latent_dim': LATENT_DIM,
            'beta': BETA,
        },
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
    }, f'{out_dir}/vae_music_model.pt')

    # save latent codes + cluster assignments + metrics
    np.save(f'{out_dir}/latent_Z.npy', Z)
    df[['file', 'genre', 'language', 'cluster_vae', 'cluster_pca']].to_csv(
        f'{out_dir}/cluster_assignments.csv', index=False)
    results.to_csv(f'{out_dir}/metrics_table.csv', index=False)

    # zip all output artifacts
    ZIP_DIR = f'{out_dir}/vae_hybrid_outputs'
    os.makedirs(ZIP_DIR, exist_ok=True)

    artifacts = [
        f'{out_dir}/vae_music_model.pt',
        f'{out_dir}/latent_Z.npy',
        f'{out_dir}/cluster_assignments.csv',
        f'{out_dir}/metrics_table.csv',
        f'{out_dir}/dataset_distribution.png',
        f'{out_dir}/training_curves.png',
        f'{out_dir}/elbow_method.png',
        f'{out_dir}/tsne_visualization.png',
        f'{out_dir}/umap_visualization.png',
        f'{out_dir}/metrics_comparison.png',
        f'{out_dir}/cluster_composition.png',
    ]
    for f in artifacts:
        if os.path.exists(f):
            shutil.copy(f, ZIP_DIR)

    shutil.make_archive(f'{out_dir}/vae_hybrid_results', 'zip', ZIP_DIR)

    # trigger Colab download
    try:
        from google.colab import files
        files.download(f'{out_dir}/vae_hybrid_results.zip')
        print('Download started!')
    except ImportError:
        print(f'Zip saved: {out_dir}/vae_hybrid_results.zip')
        print('(google.colab not available — running outside Colab)')
