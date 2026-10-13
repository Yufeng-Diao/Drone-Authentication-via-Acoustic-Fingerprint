"""Base MFCC adapter around the original, attributed numerical core."""

import numpy as np
from sklearn.preprocessing import normalize

from toolbox.mfcc_base import mfcc
from .config import Settings


def extract_segment(samples: np.ndarray, settings: Settings) -> np.ndarray:
    """Return (frames, n_filters - 1) float64 features for one continuous segment.

    Pre-emphasis, framing and padding never borrow samples from another split.
    Delta features and fractional selection are deliberately outside P1.
    """
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("Expected a nonempty, finite, one-dimensional audio segment")
    frame_s = settings.frame_ms / 1000
    base = mfcc(samples, samplerate=44100, winlen=frame_s, winstep=frame_s / 2,
                numcep=settings.n_filters, nfilt=settings.n_filters, lowfreq=0,
                highfreq=8000, preemph=0.97, ceplifter=22, appendEnergy=True,
                winfunc=np.hamming)
    features = normalize(base[:, 1:], norm="l2", axis=1)
    if features.shape[1] != settings.n_filters - 1 or not np.isfinite(features).all():
        raise ValueError("MFCC extraction returned invalid dimensions or nonfinite values")
    return features
