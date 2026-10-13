"""Read-only WAV loading and paired, interval-based data selection."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import wave

import numpy as np


def sha256(path: Path) -> str:
    """Hash a file in bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_recordings(manifest_path: Path, subset_path: Path, identities: tuple[int, ...]) -> list[dict]:
    """Select complete distance pairs and reject stale or ambiguous manifests."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or subset["schema_version"] != 1:
        raise ValueError("Unsupported dataset manifest schema")
    if subset["recordings_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Subset references a different recordings manifest")
    records = {r["path"]: r for r in manifest["recordings"]}
    paths = subset["drone_recordings"]
    if len(records) != len(manifest["recordings"]) or len(set(paths)) != len(paths):
        raise ValueError("Duplicate recording paths in manifest or subset")
    if not set(paths) <= records.keys():
        raise ValueError("Subset contains an unknown recording")
    selected = [records[p] for p in paths if records[p]["identity"] in identities]
    if {r["identity"] for r in selected} != set(identities):
        raise ValueError("Requested identity is missing from the subset")
    pairs = defaultdict(list)
    for record in selected:
        if not record["legacy_date_selected"]:
            raise ValueError("Selected recording is outside the frozen paper date selection")
        pairs[record["pair_key"]].append(record)
    for pair in pairs.values():
        if (len(pair) != 2 or {r["distance_m"] for r in pair} != {1, 5}
                or len({(r["identity"], r["frames"]) for r in pair}) != 1):
            raise ValueError("Selected distance pair is incomplete or has inconsistent metadata")
    return selected


def read_recording(audio_root: Path, record: dict) -> np.ndarray:
    """Validate a mono 44.1 kHz PCM16 WAV and return float64 samples, shape (N,)."""
    relative = PurePosixPath(record["path"])
    root = audio_root.resolve()
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or "\\" in record["path"] or not path.is_relative_to(root):
        raise ValueError("Recording path escapes the audio root")
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"Recording checksum or size mismatch: {record['path']}")
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype(), audio.getnframes()) != (1, 2, 44100, "NONE", record["frames"]):
            raise ValueError(f"Expected mono PCM16 at 44100 Hz with the indexed frame count: {record['path']}")
        payload = audio.readframes(audio.getnframes())
    if len(payload) != record["frames"] * 2:
        raise ValueError(f"Truncated WAV payload: {record['path']}")
    return np.frombuffer(payload, dtype="<i2").astype(np.float64)


def split_intervals(frames: int) -> dict[str, list[tuple[int, int]]]:
    """Half-open sample intervals, split before all signal processing."""
    if frames < 100:
        raise ValueError("Recording must contain at least 100 samples")
    boundaries = [frames * p // 100 for p in (0, 20, 35, 65, 80, 100)]
    return {"train": [(boundaries[i], boundaries[i + 1]) for i in (0, 2, 4)],
            "test": [(boundaries[i], boundaries[i + 1]) for i in (1, 3)]}
