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


def select_room_noise(manifest_path: Path, subset_path: Path) -> dict:
    """Select the explicitly declared training-only interval (full or subset)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    if subset["recordings_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Subset references a different recordings manifest")
    matches = [r for r in manifest["recordings"] if r["path"] == subset["room_noise_recording"]]
    if len(matches) != 1 or matches[0]["identity"] is not None:
        raise ValueError("Expected one room-noise recording without a drone identity")
    record = dict(matches[0])
    interval = subset["room_noise_interval_samples"]
    if (not isinstance(interval, list) or len(interval) != 2 or any(type(i) is not int for i in interval)
            or not 0 <= interval[0] < interval[1] <= record["frames"]):
        raise ValueError("Invalid room-noise training interval")
    record["training_interval"] = interval
    return record


def labelled_intervals(record: dict, partition: dict | None) -> list[tuple[str, int, int, int]]:
    """Return (split, start, end, classifier label), keeping source identity separate."""
    identity = record["identity"]
    if identity is None:
        if partition is None:
            raise ValueError("Room noise is only valid in open-set training")
        start, end = record["training_interval"]
        if not 0 <= start < end <= record["frames"]:
            raise ValueError("Room-noise interval exceeds recording bounds")
        return [("train", start, end, 0)]
    intervals = split_intervals(record["frames"])
    label = identity
    if partition is not None:
        if identity in partition["background"]:
            intervals = {"train": intervals["train"]}
            label = 0
        elif identity in partition["attack"]:
            intervals = {"test": intervals["test"]}
            label = 0
        elif identity not in partition["registered"]:
            raise ValueError("Recording identity is absent from partition")
    return [(split, start, end, label) for split, ranges in intervals.items() for start, end in ranges]
