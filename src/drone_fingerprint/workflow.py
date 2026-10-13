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
from sklearn.metrics import confusion_matrix

from .config import Settings
from .data import read_recording, sha256, labelled_intervals
from .features import extract_segment, feature_metadata
from .noise import add_noise
from .metrics import classification_metrics, open_set_metrics
from .artifacts import task_identity, completed_result
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


def run_experiment(records: list[dict], audio_root: Path, settings: Settings,
                   output: Path, provenance: dict, *, resume: bool = False, save_features: bool = False) -> dict:
    """Run one closed/open-set task with clean training and optional noisy testing.

    Audio is read per recording; feature rows accumulate in memory. Optional
    feature snapshots preserve the corrected P2 baseline for P3 comparisons.
    """
    runtime = environment()
    task_id = task_identity(records, settings, provenance, runtime, save_features)
    if resume and output.exists():
        result = completed_result(output, task_id)
        # Check raw inputs too: a stale manifest alone must not authorize reuse.
        for record in records:
            read_recording(audio_root, record)
        print(f"Reused completed task {task_id}; no fitting or artifact writes", flush=True)
        return result
    output = create_output(output, audio_root)
    started = datetime.now(timezone.utc).isoformat()
    timings, captured_warnings = {}, []
    try:
        if settings.partition is not None and sum(r["identity"] is None for r in records) != 1:
            raise ValueError("Open-set training requires exactly one declared room-noise interval")
        features, labels = {"train": [], "test": []}, {"train": [], "test": []}
        segment_index, noise_records, original_test_identity = [], [], []
        result_labels = list(settings.identities) if settings.partition is None else [0] + settings.partition["registered"]
        for index, record in enumerate(records, 1):
            print(f"Recording {index}/{len(records)}: {record['path']}", flush=True)
            with timed_phase("loading", timings):
                samples = read_recording(audio_root, record)
            intervals = labelled_intervals(record, settings.partition)
            noisy = None
            if settings.test_snr_quarter_db is not None and any(i[0] == "test" for i in intervals):
                with timed_phase("noise", timings):
                    noisy, info = add_noise(samples, record["sha256"], settings.seed, settings.test_snr_quarter_db)
                    noise_records.append({"recording": record["path"], **info})
            with timed_phase("extraction", timings):
                for split, start, end, label in intervals:
                    signal = noisy if split == "test" and noisy is not None else samples
                    block = extract_segment(signal[start:end], settings)
                    features[split].append(block)
                    labels[split].append(np.full(len(block), label, dtype=np.int64))
                    if split == "test":
                        original_test_identity.append(np.full(len(block), record["identity"], dtype=np.int64))
                    segment_index.append({"recording": record["path"], "identity": record["identity"],
                                          "classifier_label": label, "split": split, "start_sample": start,
                                          "end_sample": end, "feature_rows": len(block)})
            del samples, noisy, signal
        x_train, x_test = (np.concatenate(features[s]) for s in ("train", "test"))
        y_train, y_test = (np.concatenate(labels[s]) for s in ("train", "test"))
        original_identity = np.concatenate(original_test_identity)
        if set(y_train) != set(result_labels) or set(y_test) != set(result_labels):
            raise ValueError("Train/test labels do not cover the declared classes")
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
                    metrics = classification_metrics(y_test, prediction, result_labels)
                    if settings.partition is not None:
                        metrics.update(open_set_metrics(original_identity, prediction, settings.partition))
        finally:
            captured_warnings = [{"category": w.category.__name__, "message": str(w.message)} for w in observed]
            for warning in captured_warnings:
                print(f"WARNING {warning['category']}: {warning['message']}", file=sys.stderr, flush=True)
        with timed_phase("serialization", timings):
            joblib.dump(model, output / "model.joblib")
            np.savez(output / "predictions.npz", truth=y_test, prediction=prediction, original_identity=original_identity)
            if save_features:
                np.savez(output / "features.npz", train=x_train, test=x_test, train_labels=y_train, test_labels=y_test)
        result = {"schema_version": 2, "status": "passed", "task_id": task_id,
                  "scope": "Corrected validation workflow; not full paper reproduction",
                  "feature_metadata": feature_metadata(settings), "noise": noise_records,
                  "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                  "configuration": asdict(settings), "provenance": provenance, "environment": runtime,
                  "model_parameters": model.get_params(deep=False), "warnings": captured_warnings,
                  "timings_s": timings, "metrics": metrics, "labels": result_labels,
                  "confusion_matrix": confusion_matrix(y_test, prediction, labels=result_labels).tolist(),
                  "training_rows": len(y_train), "test_rows": len(y_test), "feature_columns": x_train.shape[1],
                  "recordings": [{k: r[k] for k in ("path", "sha256", "frames", "identity", "distance_m")} for r in records],
                  "segments": segment_index,
                  "artifacts": {name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
                                for name in (["model.joblib", "predictions.npz"] + (["features.npz"] if save_features else []))}}
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Completed: {output / 'result.json'}; smoke accuracy={metrics['accuracy']:.4f}", flush=True)
        return result
    except Exception as error:
        failure = {"status": "failed", "task_id": task_id, "started_at": started, "error": str(error), "timings_s": timings,
                   "configuration": asdict(settings), "provenance": provenance, "warnings": captured_warnings}
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise
