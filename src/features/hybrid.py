"""
Hybrid feature engineering: audio + TF-IDF lyrics + optional genre one-hot.

Strategy
--------
1. Build a synthetic lyrics corpus by sampling from per-genre keyword
   vocabularies (simulates the text modality without requiring real lyrics).
2. Embed lyrics with TF-IDF -> PCA (32 dims).
3. Fuse audio and lyrics via L2-normalisation then concatenation.
4. Optionally append a genre one-hot vector (used by CVAE / multi-modal VAE).

In a production system, replace ''make_tfidf_embedding'' with a real
SentenceTransformer or word2vec embedding.

Functions
---------
make_tfidf_embedding: Build TF-IDF PCA lyrics embedding from genre labels.
make_genre_onehot: Build genre one-hot matrix.
make_hybrid: Fuse audio + lyrics (L2-normalised concat).
make_multimodal: Fuse audio + lyrics + genre one-hot.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, normalize


def make_tfidf_embedding(y_labels: np.ndarray, genre_vocab: dict[str, str],
    n_components: int = 32, n_words: int = 40) -> np.ndarray:
    """Generate a TF-IDF -> PCA lyrics embedding for each sample.

    For each sample a synthetic document is created by randomly sampling
    words from that genre's keyword vocabulary. 
    
    TF-IDF over the corpus is then reduced to *n_components* dimensions 
    via PCA.

    Parameters
    ----------
    y_labels: String genre labels (N,).
    genre_vocab: Mapping of genre -> space-separated keyword string.
    n_components: Target PCA dimensionality (padded with zeros if data
                   has fewer components than requested).
    n_words: Number of words to sample per document.

    Returns
    -------
    Float32 array of shape (N, n_components).
    """
    fallback = "music rhythm melody beat sound"
    docs: list[str] = []

    for lbl in y_labels:
        vocab = genre_vocab.get(str(lbl), fallback)
        words = vocab.split()
        # Use a deterministic seed per label for reproducibility
        seed = abs(hash(str(lbl))) % (2 ** 31)
        rng = np.random.default_rng(seed)
        docs.append(" ".join(rng.choice(words, size=n_words, replace=True)))

    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(docs).toarray().astype(np.float32)

    pca_d = min(n_components, X_tfidf.shape[1], X_tfidf.shape[0] - 1)
    X_pca = PCA(n_components=pca_d, random_state=42).fit_transform(X_tfidf)

    # Zero-pad if PCA produced fewer components than requested
    if X_pca.shape[1] < n_components:
        padding = np.zeros(
            (X_pca.shape[0], n_components - X_pca.shape[1]), dtype=np.float32
        )
        X_pca = np.hstack([X_pca, padding])

    return X_pca.astype(np.float32)


def make_genre_onehot(y_labels: np.ndarray, le: LabelEncoder) -> np.ndarray:
    """Build a one-hot genre matrix for CVAE conditioning.

    Parameters
    ----------
    y_labels: String genre labels (N,).
    le: Fitted LabelEncoder.

    Returns
    -------
    Float32 array of shape (N, n_classes).
    """
    indices = le.transform(y_labels)
    return np.eye(len(le.classes_), dtype=np.float32)[indices]


def make_hybrid(X_audio: np.ndarray, y_labels: np.ndarray, genre_vocab: dict[str, str],
    lyric_dim: int = 32) -> np.ndarray:
    """Fuse audio and lyrics embeddings (L2-normalised concatenation).

    Both modalities are L2-normalised before concatenation so that neither
    dominates the other despite differing scales.

    Parameters
    ----------
    X_audio: Scaled audio feature matrix (N, audio_dim).
    y_labels: Genre string labels (N,).
    genre_vocab: Genre -> keyword vocabulary.
    lyric_dim: TF-IDF PCA dimension.

    Returns
    -------
    Float32 array of shape (N, audio_dim + lyric_dim).
    """
    X_lyric = make_tfidf_embedding(y_labels, genre_vocab, n_components=lyric_dim)
    return np.hstack(
        [normalize(X_audio, norm="l2"), normalize(X_lyric, norm="l2")]
    ).astype(np.float32)


def make_multimodal(X_audio: np.ndarray, y_labels: np.ndarray, le: LabelEncoder,
    genre_vocab: dict[str, str], lyric_dim: int = 32) -> np.ndarray:
    """Fuse audio + lyrics + genre one-hot for multi-modal VAE input.

    Concatenation: L2_norm(audio) ‖ L2_norm(lyrics) ‖ one_hot(genre)

    Parameters
    ----------
    X_audio: Scaled audio feature matrix (N, audio_dim).
    y_labels: Genre string labels (N,).
    le: Fitted LabelEncoder (for one-hot encoding).
    genre_vocab: Genre -> keyword vocabulary.
    lyric_dim: TF-IDF PCA dimension.

    Returns
    -------
    Float32 array of shape (N, audio_dim + lyric_dim + n_classes).
    """
    X_lyric = make_tfidf_embedding(y_labels, genre_vocab, n_components=lyric_dim)
    X_genre = make_genre_onehot(y_labels, le)
    return np.hstack(
        [normalize(X_audio, norm="l2"), normalize(X_lyric, norm="l2"), X_genre]
    ).astype(np.float32)
