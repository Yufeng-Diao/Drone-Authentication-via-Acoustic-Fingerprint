"""Explicit configuration for the P1 base-MFCC closed-set workflow."""

from dataclasses import dataclass
import json
from pathlib import Path

MODELS = ("QDA", "LDA", "LSVM", "RBF-SVM", "KNN", "DT", "RF", "GNB")


@dataclass(frozen=True)
class Settings:
    """Feature dimensions and labels; P1 supports base MFCCs without noise."""

    identities: tuple[int, ...] = (1, 2)
    frame_ms: int = 1000
    n_filters: int = 50
    model: str = "GNB"
    seed: int = 2022

    def __post_init__(self):
        if (len(self.identities) < 2 or len(set(self.identities)) != len(self.identities)
                or any(type(i) is not int or not 1 <= i <= 24 for i in self.identities)):
            raise ValueError("identities must contain at least two distinct integers from 1 through 24")
        if type(self.frame_ms) is not int or not 20 <= self.frame_ms <= 2520:
            raise ValueError("frame_ms must be an integer from 20 through 2520")
        if type(self.n_filters) is not int or not 26 <= self.n_filters <= 271:
            raise ValueError("n_filters must be an integer from 26 through 271")
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {', '.join(MODELS)}")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer from 0 through 2**32 - 1")


def load_config(path: Path) -> tuple[Settings, Path, Path]:
    """Load strict JSON; manifest paths are relative to the configuration file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    keys = {"schema_version", "recordings_manifest", "subset_manifest", "identities", "frame_ms", "n_filters", "model", "seed"}
    if set(raw) != keys or raw["schema_version"] != 1:
        raise ValueError(f"Configuration must use schema_version 1 and exactly these keys: {sorted(keys)}")
    settings = Settings(tuple(raw["identities"]), raw["frame_ms"], raw["n_filters"], raw["model"], raw["seed"])
    return settings, (path.parent / raw["recordings_manifest"]).resolve(), (path.parent / raw["subset_manifest"]).resolve()
