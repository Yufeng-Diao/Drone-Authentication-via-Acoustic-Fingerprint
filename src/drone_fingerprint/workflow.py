"""A small, traceable WAV -> MFCC -> classifier -> results workflow."""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import os
import sys
import threading
import time
import warnings

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from .config import Settings
from .data import sha256
from .features import feature_metadata
from .metrics import classification_metrics, open_set_metrics
from .artifacts import task_identity, completed_result, load_model
from .models import make_model
from .pipeline import prepare_features, cache_lock, recording_index
from .storage import create_output, external_path, atomic_json


def environment() -> dict:
    """Record versions actually imported by this run, not claimed support."""
    return {"python": sys.version, "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("drone-acoustic-fingerprint", "numpy", "scipy", "scikit-learn", "matplotlib", "joblib")}}


@contextmanager
def timed_phase(name: str, timings: dict, interval_s: float = 10, *, status_path: Path | None = None):
    """Emit elapsed-time heartbeats, with no fit timeout or invented percentage."""
    start, cpu_start = time.perf_counter(), time.process_time()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(interval_s):
            # Linux counters diagnose activity, not estimator convergence or SSD wear.
            rss = next(line.strip() for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
            io = dict(line.split(':', 1) for line in Path('/proc/self/io').read_text().splitlines())
            print(f"{datetime.now(timezone.utc).isoformat()} {name}: still running; pid={os.getpid()}; "
                  f"elapsed={time.perf_counter()-start:.1f}s; process CPU={time.process_time()-cpu_start:.1f}s; "
                  f"{rss}; read_bytes={io['read_bytes'].strip()}; write_bytes={io['write_bytes'].strip()}", flush=True)

    print(f"{name}: started", flush=True)
    if status_path is not None:
        atomic_json(status_path, {'phase': name, 'status': 'running', 'pid': os.getpid(),
                                  'at': datetime.now(timezone.utc).isoformat()})
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
        if status_path is not None:
            atomic_json(status_path, {'phase': name, 'status': status, 'pid': os.getpid(),
                                      'at': datetime.now(timezone.utc).isoformat()})
        print(f"{name}: {status}; elapsed={elapsed:.3f}s", flush=True)


@threadpool_limits.wrap(limits=1)
def run_experiment(records: list[dict], audio_root: Path, settings: Settings,
                   output: Path, provenance: dict, *, resume: bool = False, save_features: bool = False,
                   cache_root: Path | None = None, verify_inputs: bool = False, batch_frames: int = 32,
                   trained_from: Path | None = None) -> dict:
    """Run one closed/open-set task with clean training and optional noisy testing.

    Bounded audio/frame batches populate shared float64 feature caches. Estimators
    consume read-only mmaps; individual estimators may still allocate copies.
    """
    cache_root = external_path(cache_root or output.parent / 'feature-cache', audio_root)
    if output.resolve().is_relative_to(cache_root) or cache_root.is_relative_to(output.resolve()):
        raise ValueError('Cache and run output directories must be separate')
    runtime = environment()
    model_reference = None
    if trained_from is not None:
        dependency = completed_result(trained_from)
        record_fields = ('path', 'sha256', 'frames', 'identity', 'distance_m')
        if (dependency.get('model_reference') is not None
                or dependency['configuration']['test_snr_quarter_db'] is not None
                or dependency['environment'] != runtime
                or dependency['recordings'] != [{k: r[k] for k in record_fields} for r in records]
                or any(dependency['provenance'].get(k) != provenance.get(k) for k in
                       ('recordings_manifest_sha256', 'subset_manifest_sha256'))):
            raise ValueError('Clean model dependency has different data, runtime or training provenance')
        model_reference = {'path': str(trained_from.resolve()), 'task_id': dependency['task_id'],
                           'sha256': dependency['artifacts']['model.joblib']['sha256']}
        provenance = {**provenance, 'model_dependency': {k: model_reference[k] for k in ('task_id', 'sha256')}}
    task_id = task_identity(records, settings, provenance, runtime, save_features)
    if resume and output.exists():
        result = completed_result(output, task_id)
        # Check raw inputs too: a stale manifest alone must not authorize reuse.
        with cache_lock(cache_root, audio_root) as root:
            recording_index(records, audio_root, root, verify_inputs)
        print(f"Reused completed task {task_id}; no fitting or artifact writes", flush=True)
        return result
    output = create_output(output, audio_root)
    started = datetime.now(timezone.utc).isoformat()
    timings, captured_warnings = {}, []
    try:
        if settings.partition is not None and sum(r["identity"] is None for r in records) != 1:
            raise ValueError("Open-set training requires exactly one declared room-noise interval")
        result_labels = list(settings.identities) if settings.partition is None else [0] + settings.partition['registered']
        with timed_phase('feature preparation', timings, status_path=output / 'progress.json'):
            arrays, labels, original_identity, segment_index, noise_records, cache_stats = prepare_features(
                records, audio_root, settings, cache_root, verify_inputs=verify_inputs, batch_frames=batch_frames)
        x_train, x_test = arrays['train'], arrays['test']
        y_train, y_test = labels['train'], labels['test']
        if set(y_train) != set(result_labels) or set(y_test) != set(result_labels):
            raise ValueError("Train/test labels do not cover the declared classes")
        with timed_phase('model initialization', timings, status_path=output / 'progress.json'):
            model = make_model(settings.model, settings.seed) if trained_from is None else load_model(trained_from, settings)
        print(f"Model {settings.model}: {len(x_train)} training rows, {len(x_test)} test rows, "
              f"{x_train.shape[1]} features; no wall-time limit", flush=True)
        observed = []
        try:
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                if trained_from is None:
                    with timed_phase("training", timings, status_path=output / 'progress.json'):
                        model.fit(x_train, y_train)
                else:
                    print(f"Reusing clean model from {trained_from}; evaluation only", flush=True)
                with timed_phase("evaluation", timings, status_path=output / 'progress.json'):
                    prediction = model.predict(x_test)
                    metrics = classification_metrics(y_test, prediction, result_labels)
                    if settings.partition is not None:
                        metrics.update(open_set_metrics(original_identity, prediction, settings.partition))
        finally:
            captured_warnings = [{"category": w.category.__name__, "message": str(w.message)} for w in observed]
            for warning in captured_warnings:
                print(f"WARNING {warning['category']}: {warning['message']}", file=sys.stderr, flush=True)
        with timed_phase("serialization", timings, status_path=output / 'progress.json'):
            if trained_from is None:
                joblib.dump(model, output / "model.joblib")
            np.savez(output / "predictions.npz", truth=y_test, prediction=prediction, original_identity=original_identity)
            if save_features:
                np.savez(output / "features.npz", train=x_train, test=x_test, train_labels=y_train, test_labels=y_test)
        result = {"schema_version": 2, "status": "passed", "task_id": task_id,
                  "scope": "Corrected protocol run; input manifests define validation or full dataset coverage",
                  "feature_metadata": feature_metadata(settings), "noise": noise_records,
                  "cache": {"root": str(cache_root), **cache_stats}, "thread_limit": 1,
                  "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                  "configuration": asdict(settings), "provenance": provenance, "environment": runtime,
                  "model_reference": model_reference,
                  "model_parameters": model.get_params(deep=False), "warnings": captured_warnings,
                  "timings_s": timings, "metrics": metrics, "labels": result_labels,
                  "confusion_matrix": confusion_matrix(y_test, prediction, labels=result_labels).tolist(),
                  "training_rows": len(y_train), "test_rows": len(y_test), "feature_columns": x_train.shape[1],
                  "recordings": [{k: r[k] for k in ("path", "sha256", "frames", "identity", "distance_m")} for r in records],
                  "segments": segment_index,
                  "artifacts": {name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
                                for name in (["predictions.npz"] + (["model.joblib"] if trained_from is None else [])
                                             + (["features.npz"] if save_features else []))}}
        atomic_json(output / "result.json", result)
        print(f"Completed: {output / 'result.json'}; accuracy={metrics['accuracy']:.4f}", flush=True)
        return result
    except BaseException as error:
        failure = {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   "task_id": task_id, "started_at": started, "error": repr(error), "timings_s": timings,
                   "configuration": asdict(settings), "provenance": provenance, "warnings": captured_warnings}
        atomic_json(output / "failure.json", failure)
        raise
