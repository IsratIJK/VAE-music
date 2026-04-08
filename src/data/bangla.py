"""
Real Bangla music dataset builder.

Downloads Bangla songs from YouTube via yt-dlp for five genres
(Rabindra Sangeet, Baul, Folk, Modern Pop, Classical), extracts
57-dim librosa audio features, and returns aligned NumPy arrays.

A local cache dictionary avoids re-downloading when multiple
dataset pipelines request the same feature dimension.

Functions
---------
download_bangla_genre: Download audio for a single Bangla genre.
build_bangla_dataset: Download all genres and extract features.
get_bangla: Cached wrapper - call this from dataset loaders.
"""

from __future__ import annotations

import glob
import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .features import extract_audio_features, FEATURE_DIM


# Module-level cache: {feat_dim -> (X, y, lang)}
_bangla_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def download_bangla_genre(genre: str, query: str, out_dir: Path, n: int = 30) -> list[str]:
    """Download up to *n* YouTube audio files for a Bangla genre using yt-dlp.

    Audio is saved as WAV files under ''out_dir / genre /''.  If that
    directory already contains ≥ n files the download is skipped.

    Parameters
    ----------
    genre: Genre label (used as sub-directory name).
    query: YouTube search query string.
    out_dir: Root directory for Bangla audio files.
    n: Maximum number of tracks to download.

    Returns
    -------
    List of file paths that were found / downloaded.
    """
    genre_dir = out_dir / genre
    genre_dir.mkdir(parents=True, exist_ok=True)

    existing = glob.glob(str(genre_dir / "*.wav")) + glob.glob(
        str(genre_dir / "*.mp3")
    )
    if len(existing) >= n:
        print(f"  [Bangla] {genre}: {len(existing)} files cached, skipping download.")
        return existing[:n]

    print(f"  [Bangla] {genre}: downloading up to {n} tracks via yt-dlp …")
    cmd = [
        "yt-dlp",
        f"ytsearch{n}:{query}",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "5",
        "--no-playlist",
        "--output", str(genre_dir / "%(id)s.%(ext)s"),
        "--quiet",
        "--no-warnings",
        "--ignore-errors",
        "--socket-timeout", "30",
        "--retries", "2"
    ]
    try:
        subprocess.run(cmd, timeout=300, check=False)
    except subprocess.TimeoutExpired:
        print(f"    [Bangla] Timeout for {genre} - using partial downloads.")

    files = glob.glob(str(genre_dir / "*.wav")) + glob.glob(
        str(genre_dir / "*.mp3")
    )
    print(f"    -> {len(files)} files for {genre}")
    return files


def build_bangla_dataset(bangla_queries: dict[str, str], out_dir: Path, 
                         n_per_genre: int = 30, target_feat_dim: int = FEATURE_DIM) -> \
                            tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download real Bangla audio and extract librosa features.

    Parameters
    ----------
    bangla_queries: Mapping of {genre_name: youtube_search_query}.
    out_dir: Directory to save downloaded audio files.
    n_per_genre: Tracks to download per genre.
    target_feat_dim: Output feature dimensionality (pad/truncate to match).

    Returns
    -------
    X: Float32 array (N, target_feat_dim).
    y: String array (N,) of genre labels.
    lang: String array (N,) - all values are "Bangla".
    """
    out_dir = Path(out_dir)
    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    lang_list: list[str] = []

    for genre, query in bangla_queries.items():
        files = download_bangla_genre(genre, query, out_dir, n=n_per_genre)
        genre_feats: list[np.ndarray] = []

        for fpath in tqdm(files, desc=f"  librosa {genre}", leave=False):
            feat = extract_audio_features(fpath)
            if feat is not None:
                genre_feats.append(feat)

        print(f"  [{genre}] → {len(genre_feats)} valid feature vectors extracted")

        for feat in genre_feats:
            # Align feature dimensionality via padding or truncation
            if len(feat) < target_feat_dim:
                feat = np.pad(feat, (0, target_feat_dim - len(feat)))
            elif len(feat) > target_feat_dim:
                feat = feat[:target_feat_dim]
            X_list.append(feat.astype(np.float32))
            y_list.append(genre)
            lang_list.append("Bangla")

    if len(X_list) == 0:
        print("  [Bangla] No audio could be extracted — returning empty arrays.")
        empty = np.zeros((0, target_feat_dim), dtype=np.float32)
        return empty, np.array([]), np.array([])

    X = np.array(X_list, dtype=np.float32)
    print(f"\n  [Bangla] Dataset built: {X.shape} | Genres: {list(bangla_queries)}")
    return X, np.array(y_list), np.array(lang_list)


def get_bangla(feat_dim: int, bangla_queries: dict[str, str], out_dir: Path, 
               n_per_genre: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cached Bangla arrays, downloading only on the first call.

    This prevents redundant downloads when the same Bangla data is appended
    to multiple datasets (FMA, LMD, GTZAN) that may have different feature
    dimensions.  
    
    The cache key is the target feature dimension.

    Parameters
    ----------
    feat_dim: Target feature dimensionality (for alignment).
    bangla_queries: See ''build_bangla_dataset''.
    out_dir: Root audio directory.
    n_per_genre: Tracks per genre.

    Returns
    -------
    (X, y, lang) arrays aligned to feat_dim.
    """
    global _bangla_cache
    if feat_dim not in _bangla_cache:
        _bangla_cache[feat_dim] = build_bangla_dataset(
            bangla_queries=bangla_queries,
            out_dir=out_dir,
            n_per_genre=n_per_genre,
            target_feat_dim=feat_dim
        )
    return _bangla_cache[feat_dim]
