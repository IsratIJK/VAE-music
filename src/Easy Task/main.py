"""
main.py
-------
Orchestrates the full VAE music clustering pipeline:
  dataset -> VAE training -> clustering -> evaluation
"""

import os

# Config 
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_ROOT = os.path.join(_repo_root, 'data', 'VAE_Music_Dataset')
ENGLISH_DIR = os.path.join(DATA_ROOT, 'english')
BANGLA_DIR = os.path.join(DATA_ROOT, 'bangla')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

HIDDEN_DIMS = [256, 128]
LATENT_DIM = 16
BATCH_SIZE = 16   # lowered: only ~82 english tracks available
EPOCHS = 100
LR = 1e-3
BETA = 1.0
K = 6    # adjusted for smaller dataset

os.makedirs(OUT_DIR, exist_ok=True)

# Extract features (skip download — using existing dataset)
from dataset import build_dataset, plot_dataset_distribution

print('=' * 60)
print('STEP 1 — Dataset')
print(f' English: {ENGLISH_DIR}')
print(f' Bangla: {BANGLA_DIR}')
print('=' * 60)

df, X_scaled, y_genre, y_lang, le_genre, le_lang, scaler = build_dataset(
    english_dir=ENGLISH_DIR,
    bangla_dir=BANGLA_DIR,
)

plot_dataset_distribution(df, out_dir=OUT_DIR)

INPUT_DIM = X_scaled.shape[1]

# Train VAE 
from vae import train_vae, plot_training_curves, extract_latent

print('\n' + '=' * 60)
print('STEP 2 — VAE Training')
print('=' * 60)

vae, history, best_loss = train_vae(
    X_scaled,
    out_dir=OUT_DIR,
    hidden_dims=HIDDEN_DIMS,
    latent_dim=LATENT_DIM,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    lr=LR,
    beta=BETA,
)

plot_training_curves(history, out_dir=OUT_DIR)

Z = extract_latent(vae, X_scaled, LATENT_DIM)

# Clustering 
from clustering import (
    plot_elbow, cluster_vae, cluster_pca_baseline,
    plot_tsne, plot_umap, plot_cluster_composition,
)

print('\n' + '=' * 60)
print('STEP 3 — Clustering')
print('=' * 60)

plot_elbow(Z, out_dir=OUT_DIR)

km_vae, labels_vae = cluster_vae(Z, K)
pca, Z_pca, km_pca, labels_pca = cluster_pca_baseline(X_scaled, K, LATENT_DIM)

df['cluster_vae'] = labels_vae
df['cluster_pca'] = labels_pca

Z_tsne = plot_tsne(Z, y_genre, y_lang, le_genre, le_lang, out_dir=OUT_DIR)
Z_umap = plot_umap(Z, y_genre, y_lang, le_genre, le_lang, out_dir=OUT_DIR)
plot_cluster_composition(df, out_dir=OUT_DIR)

# Evaluation 
from evaluation import (
    compute_metrics, plot_metrics_comparison,
    print_final_report, save_all_results,
)

print('\n' + '=' * 60)
print('STEP 4 — Evaluation')
print('=' * 60)

results, metrics = compute_metrics(Z, labels_vae, Z_pca, labels_pca)
plot_metrics_comparison(metrics, out_dir=OUT_DIR)
print_final_report(
    df, X_scaled, le_genre, metrics,
    INPUT_DIM, HIDDEN_DIMS, LATENT_DIM, K, best_loss,
)

best_state = {k: v.cpu() for k, v in vae.state_dict().items()}
save_all_results(
    df, Z, results, scaler,
    best_state, INPUT_DIM, HIDDEN_DIMS, LATENT_DIM, BETA,
    out_dir=OUT_DIR,
)

print('\nAll outputs saved to:', OUT_DIR)
