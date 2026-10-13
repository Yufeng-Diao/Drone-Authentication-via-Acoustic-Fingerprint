"""Deterministic WAV fixtures for installation checks, never paper evidence."""

import json
from pathlib import Path
import wave

import numpy as np

from .data import sha256


def create_fixture(root: Path) -> tuple[Path, Path]:
    """Write four 20-second tonal recordings and matching manifest/subset JSON."""
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(2022)
    t = np.arange(20 * 44100) / 44100
    records = []
    for identity, frequency in ((1, 350), (2, 1200)):
        for distance in (1, 5):
            samples = 9000 * (np.sin(2*np.pi*frequency*t) + 0.2*np.sin(2*np.pi*2.3*frequency*t))
            samples += rng.normal(0, 200, size=len(t))
            path = root / f"synthetic_d{identity}_{distance}m.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
                audio.writeframes(samples.astype("<i2").tobytes())
            records.append({"path": path.name, "identity": identity, "distance_m": distance,
                            "pair_key": f"synthetic_d{identity}", "frames": len(t), "bytes": path.stat().st_size,
                            "sha256": sha256(path), "legacy_date_selected": True})
    manifest, subset = root / "recordings.json", root / "subset.json"
    manifest.write_text(json.dumps({"schema_version": 1, "scope": "Synthetic fixture only; not public recordings",
                                    "recordings": records}, indent=2) + "\n", encoding="utf-8")
    subset.write_text(json.dumps({"schema_version": 1, "name": "synthetic-installation-check",
                                  "recordings_manifest_sha256": sha256(manifest),
                                  "drone_recordings": [r["path"] for r in records]}, indent=2) + "\n", encoding="utf-8")
    return manifest, subset
