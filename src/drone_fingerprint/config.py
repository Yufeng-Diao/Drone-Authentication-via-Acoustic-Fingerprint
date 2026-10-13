"""Strict experiment settings with explicit feature and partition semantics."""

from dataclasses import dataclass
import json
from pathlib import Path

MODELS = ("QDA", "LDA", "LSVM", "RBF-SVM", "KNN", "DT", "RF", "GNB")


@dataclass(frozen=True)
class Settings:
    """A fraction uses `portion / 3` of coefficients after dropping coefficient 0."""

    identities: tuple[int, ...] = (1, 2)
    frame_ms: int = 1000
    n_filters: int = 50
    model: str = "GNB"
    seed: int = 2022
    blocks: int = 1
    portion: int = 3
    test_snr_quarter_db: int | None = None
    partition: dict | None = None

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
        for name in ("blocks", "portion"):
            if type(getattr(self, name)) is not int or getattr(self, name) not in (1, 2, 3):
                raise ValueError(f"{name} must be 1, 2 or 3")
        if self.test_snr_quarter_db is not None and (type(self.test_snr_quarter_db) is not int
                                                   or not -32 <= self.test_snr_quarter_db <= 60):
            raise ValueError("test_snr_quarter_db must be an integer from -32 through 60, or null")
        if self.partition is not None:
            p = self.partition
            if set(p) != {"trial", "registered", "background", "attack"}:
                raise ValueError("Partition must specify trial, registered, background and attack")
            groups = [p[k] for k in ("registered", "background", "attack")]
            if (type(p["trial"]) is not int or not 1 <= p["trial"] <= 10
                    or any(not isinstance(g, list) or len(g) != 8 for g in groups)
                    or any(type(i) is not int for g in groups for i in g)
                    or sorted(i for g in groups for i in g) != list(range(1, 25))
                    or set(self.identities) != set(range(1, 25))):
                raise ValueError("Open-set partition requires disjoint 8/8/8 roles covering identities 1..24")


def load_config(path: Path) -> tuple[Settings, Path, Path]:
    """Load strict JSON; manifest paths are relative to the configuration file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    keys = {"schema_version", "recordings_manifest", "subset_manifest", "identities", "frame_ms", "n_filters", "model", "seed"}
    optional = {"blocks", "portion", "test_snr_quarter_db", "trial", "partitions_manifest"}
    if not keys <= raw.keys() or raw.keys() - keys - optional or raw["schema_version"] != 1:
        raise ValueError(f"Configuration requires exactly these keys: {sorted(keys)}; optional: {sorted(optional)}")
    if ("trial" in raw) != ("partitions_manifest" in raw):
        raise ValueError("trial and partitions_manifest must be supplied together")
    partition = None
    if "trial" in raw:
        presets = json.loads((path.parent / raw["partitions_manifest"]).read_text(encoding="utf-8"))
        matches = [p for p in presets["partitions"] if p["trial"] == raw["trial"]]
        if presets["schema_version"] != 1 or type(raw["trial"]) is not int or len(matches) != 1:
            raise ValueError("Unsupported or ambiguous attack partition preset")
        partition = matches[0]
    settings = Settings(tuple(raw["identities"]), raw["frame_ms"], raw["n_filters"], raw["model"], raw["seed"],
                        **{k: raw[k] for k in ("blocks", "portion", "test_snr_quarter_db") if k in raw},
                        partition=partition)
    return settings, (path.parent / raw["recordings_manifest"]).resolve(), (path.parent / raw["subset_manifest"]).resolve()
