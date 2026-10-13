"""A small, traceable WAV -> MFCC -> classifier -> results workflow."""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import threading
import time
import warnings

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .config import Settings
from .data import read_recording, sha256, split_intervals
from .features import extract_segment
from .models import make_model


def create_output(path: Path, audio_root: Path | None = None) -> Path:
    """Create a fresh external run directory; never overwrite an existing run."""
    path = path.resolve()
    if any((parent / ".git").exists() or (parent / "pyproject.toml").exists() for parent in (path, *path.parents)):
        raise ValueError("Output must be outside every Git checkout or Python source project")
    if audio_root is not None and path.is_relative_to(audio_root.resolve()):
        raise ValueError("Output must not be inside the input audio directory")
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment() -> dict:
    """Record versions actually imported by this run, not claimed support."""
    return {"python": sys.version, "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("drone-acoustic-fingerprint", "numpy", "scipy", "scikit-learn", "matplotlib", "joblib")}}


@contextmanager
def timed_phase(name: str, timings: dict, interval_s: float = 10):
    """Emit elapsed-time heartbeats, with no fit timeout or invented percentage."""
    start, cpu_start = time.perf_counter(), time.process_time()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(interval_s):
            print(f"{name}: still running; elapsed={time.perf_counter()-start:.1f}s; process CPU={time.process_time()-cpu_start:.1f}s", flush=True)

    print(f"{name}: started", flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    status = "failed"
    try:
        yield
        status = "finished"
    finally:
        stop.set()
        thread.join()
        elapsed = time.perf_counter() - start
        timings[name] = timings.get(name, 0) + elapsed
        print(f"{name}: {status}; elapsed={elapsed:.3f}s", flush=True)


def run_closed_set(records: list[dict], audio_root: Path, settings: Settings,
                   output: Path, provenance: dict) -> dict:
    """Run one clean, base-MFCC model; persist predictions and a trusted model.

    One recording is loaded at a time. Only the compact feature rows accumulate
    in memory; no intermediate WAV or feature files are written in P1.
    """
    output = create_output(output, audio_root)
    started = datetime.now(timezone.utc).isoformat()
    timings, captured_warnings = {}, []
    try:
        features, labels = {"train": [], "test": []}, {"train": [], "test": []}
        segment_index = []
        for index, record in enumerate(records, 1):
            print(f"Recording {index}/{len(records)}: {record['path']}", flush=True)
            with timed_phase("loading", timings):
                samples = read_recording(audio_root, record)
            with timed_phase("extraction", timings):
                for split, intervals in split_intervals(len(samples)).items():
                    for start, end in intervals:
                        block = extract_segment(samples[start:end], settings)
                        features[split].append(block)
                        labels[split].append(np.full(len(block), record["identity"], dtype=np.int64))
                        segment_index.append({"recording": record["path"], "identity": record["identity"],
                                              "split": split, "start_sample": start, "end_sample": end,
                                              "feature_rows": len(block)})
            del samples
        x_train, x_test = (np.concatenate(features[s]) for s in ("train", "test"))
        y_train, y_test = (np.concatenate(labels[s]) for s in ("train", "test"))
        model = make_model(settings.model, settings.seed)
        print(f"Model {settings.model}: {len(x_train)} training rows, {x_train.shape[1]} features; no wall-time limit", flush=True)
        observed = []
        try:
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                with timed_phase("training", timings):
                    model.fit(x_train, y_train)
                with timed_phase("evaluation", timings):
                    prediction = model.predict(x_test)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_test, prediction, labels=settings.identities, average="macro", zero_division=0)
                    metrics = {"accuracy": float(accuracy_score(y_test, prediction)),
                               "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1)}
        finally:
            captured_warnings = [{"category": w.category.__name__, "message": str(w.message)} for w in observed]
            for warning in captured_warnings:
                print(f"WARNING {warning['category']}: {warning['message']}", file=sys.stderr, flush=True)
        with timed_phase("serialization", timings):
            joblib.dump(model, output / "model.joblib")
            np.savez(output / "predictions.npz", truth=y_test, prediction=prediction)
        result = {"schema_version": 1, "status": "passed", "scope": "P1 closed-set smoke run; not paper reproduction",
                  "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                  "configuration": asdict(settings), "provenance": provenance, "environment": environment(),
                  "model_parameters": model.get_params(deep=False), "warnings": captured_warnings,
                  "timings_s": timings, "metrics": metrics, "labels": list(settings.identities),
                  "confusion_matrix": confusion_matrix(y_test, prediction, labels=settings.identities).tolist(),
                  "training_rows": len(y_train), "test_rows": len(y_test), "feature_columns": x_train.shape[1],
                  "recordings": [{k: r[k] for k in ("path", "sha256", "frames", "identity", "distance_m")} for r in records],
                  "segments": segment_index,
                  "artifacts": {name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
                                for name in ("model.joblib", "predictions.npz")}}
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Completed: {output / 'result.json'}; smoke accuracy={metrics['accuracy']:.4f}", flush=True)
        return result
    except Exception as error:
        failure = {"status": "failed", "started_at": started, "error": str(error), "timings_s": timings,
                   "configuration": asdict(settings), "provenance": provenance, "warnings": captured_warnings}
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise
