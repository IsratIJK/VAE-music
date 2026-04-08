"""
FMA (Free Music Archive) dataset loader.

Downloads the fma_metadata.zip archive (~342 MB) which contains
pre-extracted librosa features for 8 000 tracks from FMA Small
across 8 genres - no audio download required.

Functions
---------
download_fma_metadata: Download + extract the metadata archive.
load_fma: Load feature matrix and genre labels.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def download_fma_metadata(data_dir: Path, url: str) -> None:
    """Download and extract fma_metadata.zip into 'data_dir'.

    Only features.csv, tracks.csv, and genres.csv are extracted.
    The zip is deleted afterwards to save disk space.

    Parameters
    ----------
    data_dir: Destination directory (created if it does not exist).
    url: Download URL for fma_metadata.zip.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if (data_dir / "features.csv").exists():
        print(f"  [FMA] Metadata already present at {data_dir}")
        return

    zip_path = data_dir.parent / "fma_metadata.zip"
    print(f"  [FMA] Downloading fma_metadata.zip (~342 MB) from {url} …")

    response = requests.get(url, stream=True, timeout=300)
    total = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc="FMA"
    ) as bar:
        for chunk in response.iter_content(65536):
            fh.write(chunk)
            bar.update(len(chunk))

    print("  [FMA] Extracting CSV files …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(("features.csv", "tracks.csv", "genres.csv")):
                print(f"    -> {name}")
                zf.extract(name, data_dir.parent)
                extracted = data_dir.parent / "fma_metadata" / name.split("/")[-1]
                dest = data_dir / name.split("/")[-1]
                if extracted.exists() and not dest.exists():
                    extracted.rename(dest)

    zip_path.unlink(missing_ok=True)
    print("  [FMA] Done.")


def load_fma(data_dir: Path, max_samples: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
    """Load FMA Small feature matrix and genre labels.

    Reads features.csv (multi-level column index) and tracks.csv,
    filters to the FMA Small subset, aligns indices, and drops NaNs.

    Parameters
    ----------
    data_dir: Directory containing features.csv and tracks.csv.
    max_samples: Cap on number of returned samples (random subset if exceeded).

    Returns
    -------
    X: Float32 feature array (N, feat_dim).
    y: String array (N,) of top-level genre labels.
    """
    data_dir = Path(data_dir)

    features = pd.read_csv(
        data_dir / "features.csv", index_col=0, header=[0, 1, 2]
    )
    tracks = pd.read_csv(
        data_dir / "tracks.csv", index_col=0, header=[0, 1]
    )

    # Top-level genre for each track (drop missing)
    genre_series = tracks["track"]["genre_top"].dropna()
    genre_series = genre_series[genre_series.str.strip() != ""]

    # Select the feature groups available in FMA (multi-level column index)
    def _safe_get(df: pd.DataFrame, key: tuple) -> pd.DataFrame:
        try:
            part = df[key]
            return part if isinstance(part, pd.DataFrame) else part.to_frame()
        except KeyError:
            return pd.DataFrame()

    parts = [
        p for p in [
            _safe_get(features, ("mfcc", "mean")),
            _safe_get(features, ("mfcc", "std")),
            _safe_get(features, ("chroma_cens", "mean")),
            _safe_get(features, ("spectral_centroid", "mean")),
            _safe_get(features, ("spectral_bandwidth", "mean")),
            _safe_get(features, ("spectral_rolloff", "mean")),
            _safe_get(features, ("zcr", "mean")),
            _safe_get(features, ("rmse", "mean")),
        ]
        if not p.empty
    ]

    feat_df = pd.concat(parts, axis=1)
    feat_df.columns = [f"f{i}" for i in range(feat_df.shape[1])]

    common_idx = feat_df.index.intersection(genre_series.index)
    X = feat_df.loc[common_idx].values.astype(np.float32)
    y = genre_series.loc[common_idx].values

    # Drop rows with NaN features
    valid = ~np.isnan(X).any(axis=1)
    X, y = X[valid], y[valid]

    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    print(f"  [FMA] Loaded: {X.shape} | Genres: {np.unique(y).tolist()}")
    return X, y
