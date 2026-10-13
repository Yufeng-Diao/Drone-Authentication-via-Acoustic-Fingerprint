"""Reproducible floating-point AWGN on a complete recording timeline."""

import hashlib
import numpy as np


def add_noise(samples: np.ndarray, recording_sha256: str, seed: int, snr_quarter_db: int) -> tuple[np.ndarray, dict]:
    """Use full-recording power; callers apply noisy samples only to test intervals."""
    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1 or not signal.size or not np.isfinite(signal).all():
        raise ValueError("AWGN requires nonempty finite one-dimensional samples")
    if type(snr_quarter_db) is not int or not -32 <= snr_quarter_db <= 60:
        raise ValueError("AWGN SNR must be an integer quarter-dB from -32 through 60")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("AWGN master seed must be an unsigned 32-bit integer")
    if len(recording_sha256) != 64 or any(c not in "0123456789abcdef" for c in recording_sha256):
        raise ValueError("AWGN requires a lowercase recording SHA-256")
    with np.errstate(over="ignore"):
        power = float(np.mean(signal ** 2))
    if not np.isfinite(power) or power <= 0:
        raise ValueError("AWGN SNR is undefined for zero or nonfinite signal power")
    local_seed = int(hashlib.sha256(f"{seed}:{recording_sha256}:{snr_quarter_db}".encode()).hexdigest()[:16], 16)
    rng = np.random.Generator(np.random.PCG64(local_seed))
    noise = rng.standard_normal(len(signal)) * np.sqrt(power / 10 ** (snr_quarter_db / 40))
    noisy = signal + noise
    if not np.isfinite(noisy).all():
        raise ValueError("AWGN addition produced nonfinite samples")
    return noisy, {"generator": "PCG64", "seed": local_seed, "requested_snr_db": snr_quarter_db / 4,
                   "signal_power": power, "measured_full_recording_snr_db": float(10 * np.log10(power / np.mean(noise ** 2)))}
