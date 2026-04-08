"""
GTZAN dataset loader.

Tries to fetch a pre-extracted GTZAN feature CSV from GitHub mirrors.
Falls back to a synthetic dataset matching real GTZAN statistics if
all URLs fail.

Functions
---------
download_gtzan_csv: Download the GTZAN feature CSV.
load_gtzan: Load feature matrix and genre labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests


# Columns that are not numeric features
_SKIP_COLS: set[str] = { "filename", "label", "length", "genre", "class" }


def download_gtzan_csv(csv_path: Path, urls: list[str]) -> bool:
    """Try downloading the GTZAN feature CSV from a list of fallback URLs.

    Parameters
    ----------
    csv_path: Destination path for the CSV file.
    urls: Ordered list of candidate download URLs.

    Returns
    -------
    True if a CSV was successfully downloaded, False otherwise.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        print(f"  [GTZAN] CSV already present at {csv_path}")
        return True

    for url in urls:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200 and len(response.content) > 5_000:
                csv_path.write_bytes(response.content)
                print(
                    f"  [GTZAN] Downloaded CSV "
                    f"({len(response.content) // 1_000} KB) from {url}"
                )
                return True
        except Exception as exc:
            print(f"  [GTZAN] URL failed ({url[:60]}…): {exc}")

    print("  [GTZAN] All URLs failed — will use synthetic data.")
    return False


def load_gtzan(csv_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load GTZAN feature matrix and genre labels.

    If 'csv_path' is None or the file does not exist, a synthetic
    dataset that approximates GTZAN statistics is returned.

    Parameters
    ----------
    csv_path: Path to the GTZAN features CSV file.

    Returns
    -------
    X: Float32 feature array (N, feat_dim).
    y: String array (N,) of genre labels.
    """
    if csv_path is not None and Path(csv_path).exists():
        df = pd.read_csv(csv_path)

        # Identify label column
        label_col = next(
            (c for c in ["label", "genre", "class"] if c in df.columns), None
        )
        feat_cols = [
            c
            for c in df.columns
            if c not in _SKIP_COLS
            and df[c].dtype in (np.float64, np.float32, np.int64)
        ]

        if label_col and feat_cols:
            df = df.dropna(subset=feat_cols + [label_col])
            X = df[feat_cols].values.astype(np.float32)
            y = df[label_col].astype(str).values
            print(f"  [GTZAN] Loaded from CSV: {X.shape} | Genres: {np.unique(y).tolist()}")
            return X, y

    # Synthetic fallback: 10 genres × 100 samples each (57 dims)
    print("  [GTZAN] Using synthetic GTZAN features.")
    _params: dict[str, tuple[float, float]] = {
        "blues": (-5, 1800),
        "classical": (-15, 2200),
        "country": (-3, 2000),
        "disco": (5, 2600),
        "hiphop": (10, 1500),
        "jazz": (-8, 2100),
        "metal": (8, 3000),
        "pop": (3, 2400),
        "reggae": (0, 1700),
        "rock": (6, 2700),
    }
    rng = np.random.default_rng(42)
    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    for genre, (mfcc_shift, spec_center) in _params.items():
        for _ in range(100):
            f = rng.standard_normal(57).astype(np.float32)
            f[:20] += mfcc_shift
            f[40] += spec_center / 1000
            X_list.append(f)
            y_list.append(genre)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    print(f"  [GTZAN] Synthetic: {X.shape} | Genres: {np.unique(y).tolist()}")
    return X, y
