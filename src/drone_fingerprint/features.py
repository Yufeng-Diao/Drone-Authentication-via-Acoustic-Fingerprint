"""Segment-local MFCCs, true derivatives, coefficient selection and final L2."""

import math
import numpy as np
from sklearn.preprocessing import normalize

from toolbox.mfcc_base import mfcc, delta
from toolbox.mfcc_sigproc import round_half_up
from .config import Settings


def feature_metadata(settings: Settings) -> dict:
    """Model identity includes extracted and retained dimensions, not filenames."""
    count = (settings.n_filters - 1) * settings.portion // 3
    return {"schema_version": 2, "sample_rate": 44100, "frame_ms": settings.frame_ms,
            "hop_fraction": "1/2", "window": "hamming", "preemphasis": 0.97,
            "fft": "smallest power of two >= frame samples", "band_hz": [0, 8000],
            "frame_peak_normalization": "max absolute", "dct": "II orthonormal", "lifter": 22,
            "append_energy": True, "extracted_coefficients": settings.n_filters,
            "selected_coefficients_per_block": count, "portion_thirds": settings.portion,
            "blocks": settings.blocks, "columns": count * settings.blocks,
            "delta": "recursive N=1; edge replication; continuous segment only",
            "selection": "drop coefficient 0 of each block, then half-open prefix",
            "normalization": "L2 after concatenation; zero rows stay zero",
            "framing": "ROUND_HALF_UP; ceil final frame; local zero padding"}


def select_features(base: np.ndarray, settings: Settings) -> np.ndarray:
    """Select from one continuous segment's unnormalized base MFCC matrix."""
    if (base.ndim != 2 or not len(base) or base.shape[1] != settings.n_filters
            or not np.isfinite(base).all()):
        raise ValueError("Expected finite base MFCC rows with n_filters columns")
    blocks = [base]
    for _ in range(settings.blocks - 1):
        blocks.append(delta(blocks[-1], 1))
    count = feature_metadata(settings)["selected_coefficients_per_block"]
    features = normalize(np.concatenate([b[:, 1:1 + count] for b in blocks], axis=1), norm="l2", axis=1)
    if not np.isfinite(features).all():
        raise ValueError("MFCC selection returned nonfinite values")
    return features


def extract_base(samples: np.ndarray, settings: Settings) -> np.ndarray:
    """Return raw MFCCs for a single continuous segment.

    Pre-emphasis, framing and padding never borrow samples from another split.
    """
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("Expected a nonempty, finite, one-dimensional audio segment")
    return _base_mfcc(samples.astype(np.float64, copy=False), settings, 0.97)


def _base_mfcc(samples, settings, preemphasis):
    frame_s = settings.frame_ms / 1000
    return mfcc(samples, samplerate=44100, winlen=frame_s, winstep=frame_s / 2,
                numcep=settings.n_filters, nfilt=settings.n_filters, lowfreq=0,
                highfreq=8000, preemph=preemphasis, ceplifter=22, appendEnergy=True,
                winfunc=np.hamming)


def extract_segment(samples: np.ndarray, settings: Settings) -> np.ndarray:
    """Return float64 (frames, selected columns) features; never join segments here."""
    return select_features(extract_base(samples, settings), settings)


def frame_geometry(samples: int, settings: Settings) -> tuple[int, int, int]:
    """Return frame length, hop and padded frame count using the P2 convention."""
    frame_s = settings.frame_ms / 1000
    length = round_half_up(frame_s * 44100)
    hop = round_half_up(frame_s / 2 * 44100)
    count = 1 if samples <= length else 1 + math.ceil((samples - length) / hop)
    return length, hop, count


def iter_base_frames(reader, start: int, end: int, settings: Settings, batch_frames: int = 32):
    """Yield raw (batch frames, filters) matrices without resetting pre-emphasis.

    Reader.read uses absolute sample intervals. Only the final frame of the
    continuous segment is padded; overlap and one preceding sample are reread.
    """
    if type(batch_frames) is not int or batch_frames < 1:
        raise ValueError('batch_frames must be a positive integer')
    length, hop, count = frame_geometry(end - start, settings)
    for first in range(0, count, batch_frames):
        frames = min(batch_frames, count - first)
        offset = start + first * hop
        stop = min(end, offset + (frames - 1) * hop + length)
        previous = offset > start
        samples = reader.read(offset - int(previous), stop)
        emphasized = np.r_[samples[0], samples[1:] - 0.97 * samples[:-1]]
        if previous:
            emphasized = emphasized[1:]
        base = _base_mfcc(emphasized, settings, 0)
        if base.shape != (frames, settings.n_filters) or not np.isfinite(base).all():
            raise ValueError('Bounded extraction returned invalid MFCC frames')
        yield base


def iter_selected_frames(base, settings: Settings, batch_frames: int = 256):
    """Apply derivatives with a two-frame halo, then select and normalize rows."""
    for start in range(0, len(base), batch_frames):
        end = min(len(base), start + batch_frames)
        left, right = max(0, start - 2), min(len(base), end + 2)
        selected = select_features(base[left:right], settings)
        yield selected[start - left:end - left]
