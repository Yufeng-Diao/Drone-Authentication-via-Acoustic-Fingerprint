"""Task identities and checked loading of completed, trusted local artifacts."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import joblib
import toolbox.mfcc_base

from .data import sha256
from .features import feature_metadata


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def task_identity(records, settings, provenance, environment, save_features) -> str:
    """Bind the input content, resolved settings, implementation and actual runtime."""
    files = sorted(Path(__file__).parent.glob("*.py"))
    core = Path(toolbox.mfcc_base.__file__).parent
    files += [core / "mfcc_base.py", core / "mfcc_sigproc.py"]
    return digest({"schema_version": 2, "records": records, "settings": asdict(settings),
                   "features": feature_metadata(settings), "provenance": provenance, "environment": environment,
                   "source": {p.name: sha256(p) for p in files}, "save_features": save_features})


def completed_result(output: Path, task_id: str | None = None) -> dict:
    """A matching success marker and intact artifacts are required for reuse."""
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    if (result.get("schema_version") != 2 or result.get("status") != "passed"
            or (output / "failure.json").exists() or (task_id is not None and result.get("task_id") != task_id)):
        raise ValueError("Completed task metadata does not match this run; use a new output directory")
    names = set(result["artifacts"])
    reference = result.get('model_reference')
    required = {'predictions.npz'} if reference else {'model.joblib', 'predictions.npz'}
    if not required <= names or names - {"model.joblib", "predictions.npz", "features.npz"}:
        raise ValueError("Completed result has missing or unknown artifacts")
    if reference:
        source = output / reference['path']
        metadata = json.loads((source / 'result.json').read_text(encoding='utf-8'))
        if metadata.get('model_reference') is not None:
            raise ValueError('Model dependency must be a training result, not another evaluation')
        dependency = completed_result(source, reference['task_id'])
        if dependency['artifacts']['model.joblib']['sha256'] != reference['sha256'] or 'model.joblib' in names:
            raise ValueError('Model dependency checksum mismatch or duplicate model artifact')
    for name, metadata in result["artifacts"].items():
        path = output / name
        if path.is_symlink() or path.stat().st_size != metadata["bytes"] or sha256(path) != metadata["sha256"]:
            raise ValueError(f"Artifact checksum or size mismatch: {name}")
    return result


def load_model(output: Path, settings):
    """Load only trusted joblib files; checksum validation is not a security sandbox."""
    result = completed_result(output)
    if result.get('model_reference') is not None:
        raise ValueError('Load the original training result instead of an evaluation')
    if result["feature_metadata"] != feature_metadata(settings):
        raise ValueError("Model feature metadata mismatch, even if selected dimensions agree")
    expected, recorded = asdict(settings), dict(result["configuration"])
    expected["identities"] = list(expected["identities"])
    expected.pop("test_snr_quarter_db")
    recorded.pop("test_snr_quarter_db")
    if expected != recorded:
        raise ValueError("Model training configuration or label partition mismatch")
    model = joblib.load(output / "model.joblib")
    if model.n_features_in_ != result["feature_metadata"]["columns"] or model.classes_.tolist() != sorted(result["labels"]):
        raise ValueError("Loaded model dimensions or labels disagree with completion metadata")
    return model
