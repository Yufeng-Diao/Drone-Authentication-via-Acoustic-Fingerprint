"""Independent numerical references and contracts for the corrected workflow."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from drone_fingerprint.artifacts import load_model
from drone_fingerprint.config import Settings, load_config
from drone_fingerprint.data import labelled_intervals, select_room_noise, select_recordings, sha256
from drone_fingerprint.features import extract_base, extract_segment, select_features, feature_metadata
from drone_fingerprint.metrics import classification_metrics, open_set_metrics
from drone_fingerprint.noise import add_noise
from drone_fingerprint.synthetic import create_fixture
from drone_fingerprint.workflow import run_experiment
from toolbox.mfcc_base import delta

ROOT = Path(__file__).resolve().parents[1]


def test_awgn_power_snr_repeatability_and_rng_isolation():
    x = np.tile(np.array([-1000, 1000], dtype=np.int16), 500_000)
    assert np.mean(np.abs(x) ** 2) == 16960  # Historical overflow, before int64 summation.
    np.random.seed(9)
    before = np.random.get_state()
    for quarter_db in (-32, 0, 60):
        noisy, info = add_noise(x, 'a' * 64, 2022, quarter_db)
        assert info['signal_power'] == 1_000_000
        measured = 10 * np.log10(1_000_000 / np.mean((noisy - x) ** 2))
        assert abs(measured - quarter_db / 4) < 0.05
        assert info['measured_full_recording_snr_db'] == pytest.approx(measured)
        np.testing.assert_array_equal(noisy, add_noise(x, 'a' * 64, 2022, quarter_db)[0])
        assert not np.array_equal(noisy, add_noise(x, 'b' * 64, 2022, quarter_db)[0])
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(x[:2], [-1000, 1000])


@pytest.mark.parametrize('samples', [np.zeros(10), np.array([np.nan]), np.array([np.inf]),
                                    np.array([]), np.ones((2, 2)), np.array([1e308])])
def test_invalid_awgn_signal(samples):
    with pytest.raises(ValueError):
        add_noise(samples, 'a' * 64, 2022, 0)


@pytest.mark.parametrize('kwargs', [{'blocks': 0}, {'portion': 4}, {'portion': True},
                                    {'test_snr_quarter_db': 0.5}, {'test_snr_quarter_db': 61},
                                    {'partition': {'trial': 1}}])
def test_invalid_numerical_settings(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_delta_is_recursive_with_replicated_edges():
    base = np.repeat(np.arange(7.)[:, None], 26, axis=1)
    first = np.array([0.5, 1, 1, 1, 1, 1, 0.5])
    second = np.array([0.25, 0.25, 0, 0, 0, -0.25, -0.25])
    np.testing.assert_array_equal(delta(base, 1)[:, 0], first)
    np.testing.assert_array_equal(delta(delta(base, 1), 1)[:, 0], second)
    assert delta(base, 2)[3, 0] == 1  # A wider first derivative, not acceleration.
    expected = np.column_stack([np.repeat(v[:, None], 25, axis=1) for v in (np.arange(7.), first, second)])
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(select_features(base, Settings(n_filters=26, blocks=3)), expected, atol=1e-15)


def test_deltas_never_cross_discontinuous_segments():
    segments = [np.ones((6, 26)), np.full((6, 26), 100.)]
    settings = Settings(n_filters=26, blocks=3)
    corrected = np.concatenate([select_features(b, settings) for b in segments])
    assert np.count_nonzero(corrected[:, 25:]) == 0
    historical = delta(np.concatenate(segments), 1)
    assert historical[5, 0] == historical[6, 0] == 49.5


@pytest.mark.parametrize('filters', [26, 50, 201, 271])
@pytest.mark.parametrize('portion', [1, 2, 3])
@pytest.mark.parametrize('blocks', [1, 2, 3])
def test_actual_extraction_dimensions_and_final_norm(filters, portion, blocks):
    settings = Settings(n_filters=filters, portion=portion, blocks=blocks, frame_ms=100)
    t = np.arange(11025) / 44100
    features = extract_segment(1000 * np.sin(2 * np.pi * 450 * t), settings)
    count = ((filters - 1) * portion // 3) * blocks
    assert features.shape == (4, count)
    assert feature_metadata(settings)['columns'] == count
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-12, rtol=0)
    assert np.count_nonzero(select_features(np.zeros((4, filters)), settings)) == 0


def test_base_mfcc_against_explicit_single_frame_reference():
    """Independent matrix DCT and explicit mel triangles on one zero-padded frame."""
    signal = np.array([0., 100., -200., 300., -400.])
    settings = Settings(frame_ms=20, n_filters=26)
    frame = np.zeros(882)
    frame[:5] = np.r_[signal[0], signal[1:] - 0.97 * signal[:-1]]
    frame *= np.hamming(882)
    frame /= np.max(np.abs(frame))
    power = abs(np.fft.rfft(frame, 1024)) ** 2 / 1024
    mel_edges = np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 28)
    bins = np.floor(1025 * (700 * (10 ** (mel_edges / 2595) - 1)) / 44100).astype(int)
    energies = []
    for left, center, right in zip(bins[:-2], bins[1:-1], bins[2:]):
        triangle = np.zeros(len(power))
        triangle[left:center] = np.arange(center - left) / (center - left)
        triangle[center:right] = np.arange(right - center, 0, -1) / (right - center)
        energies.append(max(np.dot(power, triangle), np.finfo(float).eps))
    dct_matrix = np.cos(np.pi * np.arange(26)[:, None] * (np.arange(26) + 0.5) / 26) * np.sqrt(2 / 26)
    dct_matrix[0] /= np.sqrt(2)
    reference = (dct_matrix @ np.log(energies)) * (1 + 11 * np.sin(np.pi * np.arange(26) / 22))
    reference[0] = np.log(power.sum())
    np.testing.assert_allclose(extract_base(signal, settings)[0], reference, atol=1e-10, rtol=1e-12)


def test_standard_macro_f1_is_mean_of_per_class_f1():
    m = classification_metrics([1, 1, 2], [1, 1, 1], [1, 2])
    assert m == pytest.approx(dict(accuracy=2/3, macro_precision=1/3, macro_recall=0.5, macro_f1=0.4))
    with pytest.raises(ValueError):
        classification_metrics([1], [0], [1, 2])


def test_attack_identity_macro_differs_from_collapsed_pooled_rate():
    p = json.loads((ROOT / 'config/paper/attack_partitions.json').read_text())['partitions'][0]
    original = np.r_[np.arange(1, 9), 17, np.repeat(np.arange(18, 25), 9)]
    prediction = np.r_[np.arange(1, 9), 0, np.ones(63, dtype=int)]
    m = open_set_metrics(original, prediction, p)
    assert m['registered_macro_recall'] == 1
    assert m['attack_macro_rejection'] == 1/8
    assert m['attack_pooled_rejection'] == 1/64
    with pytest.raises(ValueError, match='every registered/attack'):
        open_set_metrics(np.r_[original, 9], np.r_[prediction, 0], p)


def test_frozen_partition_and_training_only_room_noise():
    settings, manifest, subset = load_config(ROOT / 'config/validation/open_set_trial_01.json')
    frozen = json.loads((ROOT / 'config/paper/attack_partitions.json').read_text())['partitions'][0]
    assert settings.partition == frozen
    noise = select_room_noise(manifest, subset)
    assert labelled_intervals(noise, frozen) == [('train', 0, 60 * 44100, 0)]
    full_noise = dict(noise, training_interval=[0, noise['frames']])
    assert labelled_intervals(full_noise, frozen) == [('train', 0, noise['frames'], 0)]
    for identity in range(1, 25):
        rows = labelled_intervals({'identity': identity, 'frames': 1000}, frozen)
        assert {r[0] for r in rows} == ({'train', 'test'} if identity <= 8 else {'train'} if identity <= 16 else {'test'})
        assert {r[3] for r in rows} == {identity if identity <= 8 else 0}


def test_noise_does_not_change_training_and_completed_reuse_is_checked(tmp_path, monkeypatch):
    manifest, subset = create_fixture(tmp_path / 'audio')
    records = select_recordings(manifest, subset, (1, 2))
    settings = Settings(blocks=3, portion=1)
    provenance = {'dataset': 'test fixture'}
    clean_dir, noisy_dir = tmp_path / 'clean', tmp_path / 'noisy'
    clean = run_experiment(records, manifest.parent, settings, clean_dir, provenance, save_features=True)
    noisy_settings = replace(settings, test_snr_quarter_db=0)
    noisy = run_experiment(records, manifest.parent, noisy_settings, noisy_dir, provenance, save_features=True)
    with np.load(clean_dir / 'features.npz') as a, np.load(noisy_dir / 'features.npz') as b:
        np.testing.assert_array_equal(a['train'], b['train'])
        assert not np.array_equal(a['test'], b['test'])
    assert clean['task_id'] != noisy['task_id']
    model = load_model(clean_dir, noisy_settings)  # A noise evaluation may reuse clean training.
    assert model.n_features_in_ == 48
    with pytest.raises(ValueError, match='feature metadata'):
        load_model(clean_dir, replace(settings, n_filters=51))  # Same 48 columns, different extraction.
    with pytest.raises(ValueError, match='training configuration'):
        load_model(clean_dir, replace(settings, seed=1))
    before = {p.name: (sha256(p), p.stat().st_mtime_ns) for p in clean_dir.iterdir()}
    monkeypatch.setattr('drone_fingerprint.workflow.make_model', lambda *args: pytest.fail('Resume must not fit'))
    reused = run_experiment(records, manifest.parent, settings, clean_dir, provenance, resume=True, save_features=True)
    assert reused['task_id'] == clean['task_id']
    assert before == {p.name: (sha256(p), p.stat().st_mtime_ns) for p in clean_dir.iterdir()}
    with pytest.raises(ValueError, match='does not match'):
        run_experiment(records, manifest.parent, replace(settings, seed=1), clean_dir, provenance, resume=True, save_features=True)
    (clean_dir / 'model.joblib').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        load_model(clean_dir, settings)
    with pytest.raises(ValueError, match='checksum'):
        run_experiment(records, manifest.parent, settings, clean_dir, provenance, resume=True, save_features=True)
    (noisy_dir / 'result.json').write_text('{corrupt')
    with pytest.raises(ValueError):
        load_model(noisy_dir, noisy_settings)


def test_cli_bad_configuration_has_nonzero_actionable_exit(tmp_path):
    config = tmp_path / 'bad.json'
    config.write_text('{"schema_version": 1}')
    result = subprocess.run([sys.executable, '-I', '-B', '-m', 'drone_fingerprint.cli', 'run',
                             '--config', str(config), '--audio-root', str(tmp_path),
                             '--output', str(tmp_path / 'output')], capture_output=True, text=True)
    assert result.returncode == 1 and 'ERROR: Configuration requires' in result.stderr
    assert not (tmp_path / 'output').exists()


def test_completed_marker_and_artifacts_are_mandatory(tmp_path):
    from drone_fingerprint.artifacts import completed_result
    with pytest.raises(FileNotFoundError):
        completed_result(tmp_path)
    result = {'schema_version': 2, 'status': 'passed', 'artifacts': {}}
    (tmp_path / 'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='missing'):
        completed_result(tmp_path)
    (tmp_path / 'failure.json').write_text('{}')
    with pytest.raises(ValueError, match='does not match'):
        completed_result(tmp_path)


def test_open_set_training_requires_room_noise(tmp_path):
    settings, _, _ = load_config(ROOT / 'config/validation/open_set_trial_01.json')
    with pytest.raises(ValueError, match='room-noise'):
        run_experiment([], tmp_path / 'audio', settings, tmp_path / 'missing-noise', {})


def test_short_padded_frames_and_fraction_selection_reference():
    # Selection must discard column 0 before taking exactly eight of the remaining 25.
    base = np.arange(26., dtype=float)[None, :]
    expected = np.arange(1., 9.)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(select_features(base, Settings(n_filters=26, portion=1))[0], expected)
    for length, expected_frames in [(1, 1), (44100, 1), (44101, 2), (66150, 2), (66151, 3)]:
        assert extract_segment(np.ones(length), Settings()).shape == (expected_frames, 49)


def test_background_label_order_changes_knn_vote_ties():
    """A background label is scientifically observable when eight votes tie."""
    from drone_fingerprint.models import make_model

    features = np.arange(8, dtype=float).reshape(-1, 1)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    query = np.array([[3.5]])
    background_first = make_model('KNN').fit(features, labels).predict(query)
    background_last = make_model('KNN').fit(features, np.where(labels == 0, 25, labels)).predict(query)
    background_last = np.where(background_last == 25, 0, background_last)
    assert background_first.tolist() == [0]
    assert background_last.tolist() == [1]
