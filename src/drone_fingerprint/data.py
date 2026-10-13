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
    path = recording_path(audio_root, record)
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


def recording_path(audio_root: Path, record: dict) -> Path:
    """Resolve a manifest path without allowing escapes or symlink redirection."""
    relative = PurePosixPath(record['path'])
    root = audio_root.resolve()
    path = (root / relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or '\\' in record['path'] or not path.is_relative_to(root):
        raise ValueError('Recording path escapes the audio root')
    return path


def file_signature(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def inspect_recording(audio_root: Path, record: dict) -> dict:
    """Hash and measure PCM power in one bounded pass; retain the data offset."""
    path = recording_path(audio_root, record)
    before = file_signature(path)
    checksum, square_sum = hashlib.sha256(), 0
    with path.open('rb') as stream:
        with wave.open(stream, 'rb') as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype(), audio.getnframes()) != (1, 2, 44100, 'NONE', record['frames']):
                raise ValueError(f"Expected mono PCM16 at 44100 Hz with the indexed frame count: {record['path']}")
            offset = stream.tell()
            stream.seek(0)
            checksum.update(stream.read(offset))
            remaining = record['frames'] * 2
            while remaining:
                payload = stream.read(min(remaining, 2**20))
                if not payload or len(payload) % 2:
                    raise ValueError(f"Truncated WAV payload: {record['path']}")
                checksum.update(payload)
                values = np.frombuffer(payload, dtype='<i2').astype(np.int64)
                square_sum += int(np.sum(values * values, dtype=np.int64))
                remaining -= len(payload)
            while chunk := stream.read(2**20):
                checksum.update(chunk)
    if before != file_signature(path):
        raise ValueError(f"Recording changed during verification: {record['path']}")
    if before[2] != record['bytes'] or checksum.hexdigest() != record['sha256']:
        raise ValueError(f"Recording checksum or size mismatch: {record['path']}")
    if record['frames'] <= 0:
        raise ValueError('Recording must contain PCM samples')
    return {'signature': before, 'sha256': record['sha256'], 'frames': record['frames'],
            'data_offset': offset, 'signal_power': square_sum / record['frames']}


class PCMReader:
    """Read bounded float64 intervals from an already verified PCM16 recording."""

    def __init__(self, audio_root: Path, record: dict, info: dict):
        self.path = recording_path(audio_root, record)
        self.info = info
        if file_signature(self.path) != info['signature']:
            raise ValueError('Recording changed since index validation')
        self.stream = self.path.open('rb', buffering=0)
        self.buffer_start = 0
        self.buffer = np.empty(0)

    def read(self, start: int, end: int) -> np.ndarray:
        if not 0 <= start < end <= self.info['frames']:
            raise ValueError('PCM read interval exceeds recording bounds')
        previous_end = self.buffer_start + len(self.buffer)
        overlap = self.buffer[start-self.buffer_start:end-self.buffer_start] if self.buffer_start <= start < previous_end else np.empty(0)
        read_start = start + len(overlap)
        self.stream.seek(self.info['data_offset'] + read_start * 2)
        payload = self.stream.read((end - read_start) * 2)
        if len(payload) != (end - read_start) * 2:
            raise ValueError('Truncated PCM interval')
        fresh = np.frombuffer(payload, dtype='<i2').astype(np.float64)
        self.buffer = np.concatenate([overlap, fresh]) if len(overlap) else fresh
        self.buffer_start = start
        return self.buffer

    def close(self):
        self.stream.close()
        if file_signature(self.path) != self.info['signature']:
            raise ValueError('Recording changed during extraction')
