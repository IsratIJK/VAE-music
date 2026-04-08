"""
Audio feature extraction using librosa.

Extracts a fixed 57-dimensional feature vector per audio file:
  - MFCC mean (20) + std (20)
  - Chroma CENS mean (12)
  - Spectral centroid mean (1)
  - Spectral bandwidth mean (1)
  - Spectral rolloff mean (1)
  - Zero-crossing rate mean (1)
  - RMS energy mean (1)
  Total: 57 features

Functions
---------
extract_audio_features: Extract features from a single audio file path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# librosa is an optional dependency — imported lazily so the rest of the
# project can be imported without it (e.g. for unit tests or PCA-only runs).
try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

FEATURE_DIM = 57  # total number of extracted features


def extract_audio_features(fpath: str | Path, sr: int = 22050, duration: float = 30.0, 
                           n_mfcc: int = 20) -> np.ndarray | None:
    """Extract a 57-dim audio feature vector from an audio file.

    Parameters
    ----------
    fpath: Path to the audio file (wav, mp3, etc.).
    sr: Target sample rate (default 22 050 Hz).
    duration: Maximum clip length in seconds (default 30 s - same as GTZAN).
    n_mfcc: Number of MFCC coefficients (default 20).

    Returns
    -------
    Float32 array of shape (57,), or None if loading / extraction fails
    or the clip is shorter than 3 seconds.
    """
    if not _LIBROSA_AVAILABLE:
        raise ImportError(
            "librosa is required for audio feature extraction. "
            "Install it with: pip install librosa"
        )

    try:
        y, _ = librosa.load(str(fpath), sr=sr, duration=duration, mono=True)

        # Skip very short clips (< 3 s)
        if len(y) < sr * 3:
            return None

        # MFCC: mean (n_mfcc,) + std (n_mfcc,)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = mfcc.mean(axis=1) # (n_mfcc,)
        mfcc_std = mfcc.std(axis=1) # (n_mfcc,)

        # Chroma CENS: mean (12,)
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)

        # Spectral features: each (1,)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        rms = librosa.feature.rms(y=y).mean()

        feat = np.concatenate([
            mfcc_mean,  # 20
            mfcc_std,   # 20
            chroma_mean, # 12
            [spec_centroid, spec_bandwidth, spec_rolloff,  #  3
             zcr, rms]  #  2 -> total 57
        ]).astype(np.float32)

        return feat

    except Exception:
        # Return None for any loading / processing error so callers can skip
        return None
