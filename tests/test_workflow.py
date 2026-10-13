"""Observable checks for the installed P1 workflow and its input boundaries."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import time
import wave
import warnings

import joblib
import numpy as np
import pytest

from drone_fingerprint.config import Settings, load_config, MODELS
from drone_fingerprint.data import read_recording, select_recordings, sha256, split_intervals
from drone_fingerprint.features import extract_segment
from drone_fingerprint.models import make_model
from drone_fingerprint.synthetic import create_fixture
from drone_fingerprint.workflow import create_output, run_experiment, timed_phase


@pytest.fixture
def dataset(tmp_path):
    manifest, subset = create_fixture(tmp_path / "audio")
    records = select_recordings(manifest, subset, (1, 2))
    return manifest, subset, records


@pytest.mark.parametrize("kwargs", [{"identities": (1, 1)}, {"frame_ms": 0}, {"frame_ms": True},
                                   {"n_filters": 500}, {"seed": -1}, {"model": "unknown"}])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_unknown_configuration_option_is_not_ignored(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"schema_version": 1, "unexpected": true}')
    with pytest.raises(ValueError, match="exactly these keys"):
        load_config(config)


def test_subset_hash_and_complete_pairs_required(dataset):
    manifest, subset, records = dataset
    raw = json.loads(subset.read_text())
    raw["drone_recordings"].pop()
    subset.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="pair"):
        select_recordings(manifest, subset, (1, 2))
    raw["recordings_manifest_sha256"] = "0" * 64
    subset.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="different"):
        select_recordings(manifest, subset, (1, 2))


def test_wave_format_and_content_are_checked(dataset):
    manifest, _, records = dataset
    record = records[0]
    path = manifest.parent / record["path"]
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\xff")
    with pytest.raises(ValueError, match="checksum"):
        read_recording(manifest.parent, record)
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x00" * record["frames"])
    record.update(bytes=path.stat().st_size, sha256=sha256(path))
    with pytest.raises(ValueError, match="44100"):
        read_recording(manifest.parent, record)


def test_recording_path_cannot_escape_dataset(dataset):
    manifest, _, records = dataset
    records[0]["path"] = "../outside.wav"
    with pytest.raises(ValueError, match="escapes"):
        read_recording(manifest.parent, records[0])


def test_split_matches_frozen_sample_intervals():
    assert split_intervals(1000) == {"train": [(0, 200), (350, 650), (800, 1000)],
                                     "test": [(200, 350), (650, 800)]}


def test_silence_short_segments_and_nonfinite_audio():
    features = extract_segment(np.zeros(100), Settings())
    assert features.shape == (1, 49) and np.isfinite(features).all()
    norm = np.linalg.norm(features[0])
    assert np.isclose(norm, 0) or np.isclose(norm, 1)
    with pytest.raises(ValueError, match="finite"):
        extract_segment(np.array([np.nan]), Settings())


def test_external_outputs_cannot_overwrite_inputs_or_previous_runs(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="Git checkout"):
        create_output(repo / "accidental-output")
    audio = tmp_path / "audio"
    audio.mkdir()
    with pytest.raises(ValueError, match="input audio"):
        create_output(audio / "run", audio)
    with pytest.raises(FileExistsError):
        create_output(tmp_path)
    source = tmp_path / "unpacked-source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    with pytest.raises(ValueError, match="source project"):
        create_output(source / "run")


def test_heartbeat_during_silent_work():
    buffer, timings = io.StringIO(), {}
    with redirect_stdout(buffer):
        with timed_phase("silent operation", timings, interval_s=0.01):
            time.sleep(0.08)
    assert "still running" in buffer.getvalue()
    assert "finished" in buffer.getvalue() and timings["silent operation"] >= 0.08


def test_paper_model_constructors_preserve_migration_decisions():
    assert len([make_model(name) for name in MODELS]) == 8
    assert make_model("LSVM").get_params()["dual"] is True
    assert make_model("RF").get_params()["max_features"] == "sqrt"
    assert make_model("DT").get_params()["random_state"] == 1
    assert make_model("RBF-SVM").get_params()["max_iter"] == 1000


def test_failure_never_creates_a_success_result(tmp_path, dataset):
    manifest, _, records = dataset
    (manifest.parent / records[0]["path"]).unlink()
    output = tmp_path / "failed-run"
    with pytest.raises(FileNotFoundError):
        run_experiment(records, manifest.parent, Settings(), output, {"dataset": "synthetic"})
    assert not (output / "result.json").exists()
    assert json.loads((output / "failure.json").read_text())["status"] == "failed"


def test_wav_to_model_predictions_are_repeatable(tmp_path, dataset):
    manifest, _, records = dataset
    first = run_experiment(records, manifest.parent, Settings(), tmp_path / "first", {"dataset": "synthetic"})
    second = run_experiment(records, manifest.parent, Settings(), tmp_path / "second", {"dataset": "synthetic"})
    assert first["training_rows"] == 100 and first["test_rows"] == 40
    assert first["feature_columns"] == 49
    assert first["metrics"]["accuracy"] >= 0.95  # Distinct synthetic tones, not a real-data target.
    a = np.load(tmp_path / "first/predictions.npz", allow_pickle=False)
    b = np.load(tmp_path / "second/predictions.npz", allow_pickle=False)
    np.testing.assert_array_equal(a["prediction"], b["prediction"])
    assert first["confusion_matrix"] == second["confusion_matrix"]
    model = joblib.load(tmp_path / "first/model.joblib")
    samples = read_recording(manifest.parent, records[0])
    held_out = extract_segment(samples[4*44100:7*44100], Settings())
    np.testing.assert_array_equal(model.predict(held_out), a["prediction"][:5])


def test_failed_fit_preserves_its_warnings(tmp_path, dataset, monkeypatch, capsys):
    from drone_fingerprint import workflow

    class FailingEstimator:
        def fit(self, features, labels):
            warnings.warn("Fixture warning before failed fit", UserWarning)
            raise ValueError("Fixture fit failure")

    monkeypatch.setattr(workflow, "make_model", lambda *args: FailingEstimator())
    manifest, _, records = dataset
    output = tmp_path / "failed-fit"
    with pytest.raises(ValueError, match="Fixture fit failure"):
        run_experiment(records, manifest.parent, Settings(), output, {"dataset": "synthetic"})
    failure = json.loads((output / "failure.json").read_text())
    assert failure["warnings"][0]["message"] == "Fixture warning before failed fit"
    assert "Fixture warning before failed fit" in capsys.readouterr().err
    assert not (output / "result.json").exists()
