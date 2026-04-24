import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

warnings.filterwarnings('ignore')


def plot_elbow(Z, k_range=range(2, 16), out_dir='/content'):
    os.makedirs(out_dir, exist_ok=True)
    inertias = []

    for k in tqdm(k_range, desc='Elbow sweep'):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(Z)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), inertias, 'o-', color='#4e8ef7', linewidth=2, markersize=6)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method - Optimal K Selection', fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/elbow_method.png', dpi=150, bbox_inches='tight')
    plt.show()

    return inertias


def cluster_vae(Z, K):
    km_vae = KMeans(n_clusters=K, n_init=20, random_state=42)
    labels_vae = km_vae.fit_predict(Z)
    print(f'VAE + KMeans clustering done (K={K})')
    print(f'Cluster sizes: {np.bincount(labels_vae)}')
    return km_vae, labels_vae


def cluster_pca_baseline(X_scaled, K, latent_dim=16):
    pca = PCA(n_components=latent_dim, random_state=42)
    Z_pca = pca.fit_transform(X_scaled)

    km_pca = KMeans(n_clusters=K, n_init=20, random_state=42)
    labels_pca = km_pca.fit_predict(Z_pca)

    print(f'PCA + KMeans baseline done')
    print(f'PCA explained variance ({latent_dim} components): '
          f'{pca.explained_variance_ratio_.sum():.2%}')
    return pca, Z_pca, km_pca, labels_pca


def plot_tsne(Z, y_genre, y_lang, le_genre, le_lang, out_dir='/content'):
    os.makedirs(out_dir, exist_ok=True)

    print('Running t-SNE...')
    tsne = TSNE(n_components=2, perplexity=min(30, len(Z) // 4),
                n_iter=1000, random_state=42)
    Z_tsne = tsne.fit_transform(Z)
    print('t-SNE done.')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    palette = plt.cm.tab20.colors
    colors_lang = {0: '#f7914e', 1: '#4e8ef7'}
    lang_names = le_lang.classes_

    for i, genre in enumerate(le_genre.classes_):
        mask = y_genre == i
        axes[0].scatter(Z_tsne[mask, 0], Z_tsne[mask, 1],
                        c=[palette[i % len(palette)]], label=genre, alpha=0.7, s=18)
    axes[0].set_title('t-SNE: VAE Latent Space (by Genre)', fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=7, ncol=2)
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')

    for lid in [0, 1]:
        mask = y_lang == lid
        axes[1].scatter(Z_tsne[mask, 0], Z_tsne[mask, 1],
                        c=colors_lang[lid], label=lang_names[lid], alpha=0.7, s=18)
    axes[1].set_title('t-SNE: VAE Latent Space (by Language)', fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig(f'{out_dir}/tsne_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: tsne_visualization.png')

    return Z_tsne


def plot_umap(Z, y_genre, y_lang, le_genre, le_lang, out_dir='/content'):
    os.makedirs(out_dir, exist_ok=True)

    print('Running UMAP...')
    reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(Z) - 1),
                        min_dist=0.1, random_state=42)
    Z_umap = reducer.fit_transform(Z)
    print('UMAP done.')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    palette = plt.cm.tab20.colors
    colors_lang = {0: '#f7914e', 1: '#4e8ef7'}
    lang_names = le_lang.classes_

    for i, genre in enumerate(le_genre.classes_):
        mask = y_genre == i
        axes[0].scatter(Z_umap[mask, 0], Z_umap[mask, 1],
                        c=[palette[i % len(palette)]], label=genre, alpha=0.7, s=18)
    axes[0].set_title('UMAP: VAE Latent Space (by Genre)', fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=7, ncol=2)
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')

    for lid in [0, 1]:
        mask = y_lang == lid
        axes[1].scatter(Z_umap[mask, 0], Z_umap[mask, 1],
                        c=colors_lang[lid], label=lang_names[lid], alpha=0.7, s=18)
    axes[1].set_title('UMAP: VAE Latent Space (by Language)', fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')

    plt.tight_layout()
    plt.savefig(f'{out_dir}/umap_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: umap_visualization.png')

    return Z_umap


def plot_cluster_composition(df, out_dir='/content'):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, col, title in zip(
        axes,
        ['cluster_vae', 'cluster_pca'],
        ['VAE + KMeans Cluster Composition', 'PCA + KMeans Cluster Composition']
    ):
        comp = df.groupby([col, 'language']).size().unstack(fill_value=0)
        comp.plot(kind='bar', ax=ax, color=['#f7914e', '#4e8ef7'],
                  edgecolor='white', linewidth=1.2)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Track Count')
        ax.legend(title='Language')
        ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/cluster_composition.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: cluster_composition.png')
