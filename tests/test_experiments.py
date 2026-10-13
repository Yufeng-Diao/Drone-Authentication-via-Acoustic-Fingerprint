"""Paper coverage, model reuse, honest failures and interrupted task restart."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from drone_fingerprint.artifacts import completed_result
from drone_fingerprint.config import MODELS, Settings
from drone_fingerprint.data import select_recordings, sha256
from drone_fingerprint.experiments import load_plan, paper_tasks, run_plan, sweep_lock
from drone_fingerprint.synthetic import create_fixture
from drone_fingerprint.workflow import run_experiment

PAPER = Path(__file__).resolve().parents[1] / 'config/paper'


def test_complete_paper_grid_and_dependencies():
    protocol = json.loads((PAPER / 'protocol.json').read_text())
    partitions = json.loads((PAPER / 'attack_partitions.json').read_text())
    tasks = paper_tasks(protocol, partitions, {e['id']: {} for e in protocol['experiments']}, list(MODELS))
    assert len(tasks) == 4096 + 3600 + 744
    assert len({t['id'] for t in tasks}) == len(tasks)
    seen = set()
    for task in tasks:
        if task['dependency']:
            assert task['dependency'] in seen
        seen.add(task['id'])
    for name, expected in [('figure2', 408), ('figure3', 3600), ('table2', 8), ('tables4_5', 80), ('figure4', 3600), ('figure5', 744)]:
        assert sum(t['preset'] == name for t in tasks) == expected
    assert {t['settings']['partition']['trial'] for t in tasks if t['preset'] == 'tables4_5'} == set(range(1, 11))


@pytest.mark.parametrize('presets,models', [({}, ['GNB']), ({'unknown': {}}, ['GNB']),
    ({'figure2': {'frame_ms': [1000]}}, ['GNB']), ({'figure3': {'blocks': [True]}}, ['GNB']),
    ({'figure5': {'snr_quarter_db': [61]}}, ['GNB']), ({'table2': {}}, ['GNB', 'GNB']),
    ({'table2': {'typo': [1]}}, ['GNB'])])
def test_invalid_grid_rejected(presets, models):
    with pytest.raises(ValueError):
        paper_tasks(json.loads((PAPER / 'protocol.json').read_text()),
                    json.loads((PAPER / 'attack_partitions.json').read_text()), presets, models)


def test_full_subset_and_resource_estimate():
    plan = load_plan(PAPER / 'full_sweep.json', presets=['table2', 'tables4_5'], models=['GNB'])
    assert len(plan['record_sets']['DS2']) == 240
    assert len(plan['record_sets']['open']) == 241
    noise = next(r for r in plan['record_sets']['open'] if r['identity'] is None)
    assert noise['training_interval'] == [0, noise['frames']]
    assert plan['counts'] == {'fits': 11, 'evaluations': 0, 'tasks': 11}
    for task in plan['tasks']:
        r = task['resources']
        assert r['base_and_selected_bytes'] == (r['train_rows'] + r['test_rows']) * 401 * 8


@pytest.fixture
def small_plan(tmp_path):
    manifest, subset = create_fixture(tmp_path / 'audio')
    protocol = json.loads((PAPER / 'protocol.json').read_text())
    protocol['datasets']['DS2'] = [1, 2]
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(json.dumps(protocol))
    config = tmp_path / 'sweep-config.json'
    config.write_text(json.dumps({'schema_version': 1, 'protocol': str(protocol_path),
        'recordings_manifest': str(manifest), 'subset_manifest': str(subset),
        'partitions_manifest': str(PAPER / 'attack_partitions.json'), 'models': ['GNB', 'DT'],
        'presets': {'table2': {}, 'figure5': {'snr_quarter_db': [0]}}}))
    return load_plan(config), manifest.parent


def test_interrupt_resume_matches_fresh_and_never_refits_completed(small_plan, tmp_path, monkeypatch):
    from drone_fingerprint import workflow
    plan, audio = small_plan
    output, cache, logs = [tmp_path / k for k in ('resumed', 'cache', 'logs')]
    original = workflow.make_model
    fits = []

    def interrupt_second(name, seed):
        fits.append(name)
        if name == 'DT':
            raise KeyboardInterrupt()
        return original(name, seed)

    monkeypatch.setattr(workflow, 'make_model', interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        run_plan(plan, audio, output, cache, logs)
    state = json.loads((output / 'sweep.json').read_text())
    assert state['status'] == 'interrupted'
    first = output / state['tasks'][plan['tasks'][0]['id']]['attempts'][0]['directory']
    snapshot = {p.name: (sha256(p), p.stat().st_mtime_ns) for p in first.iterdir()}
    failed = state['tasks'][plan['tasks'][1]['id']]['attempts'][0]
    assert not (output / failed['directory'] / 'result.json').exists()
    assert json.loads((output / failed['directory'] / 'failure.json').read_text())['status'] == 'interrupted'
    monkeypatch.setattr(workflow, 'make_model', original)
    resumed = run_plan(plan, audio, output, cache, logs, resume=True)
    assert resumed['status'] == 'completed' and len(resumed['tasks']) == 4
    assert {p.name: (sha256(p), p.stat().st_mtime_ns) for p in first.iterdir()} == snapshot
    assert len(resumed['tasks'][plan['tasks'][1]['id']]['attempts']) == 2
    fresh = run_plan(plan, audio, tmp_path / 'fresh', cache, tmp_path / 'fresh-logs')
    for task in plan['tasks']:
        a = output / resumed['tasks'][task['id']]['attempts'][-1]['directory']
        b = tmp_path / 'fresh' / fresh['tasks'][task['id']]['attempts'][-1]['directory']
        assert completed_result(a)['metrics'] == completed_result(b)['metrics']
        with np.load(a / 'predictions.npz') as left, np.load(b / 'predictions.npz') as right:
            for key in left.files:
                np.testing.assert_array_equal(left[key], right[key])
        if task['kind'] == 'evaluate':
            assert not (a / 'model.joblib').exists()
            assert 'training' not in completed_result(a)['timings_s']
    assert run_plan(plan, audio, output, cache, logs, resume=True)['tasks'] == resumed['tasks']
    (first / 'model.joblib').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum|size'):
        run_plan(plan, audio, output, cache, logs, resume=True)
    assert len(json.loads((output / 'sweep.json').read_text())['tasks'][plan['tasks'][0]['id']]['attempts']) == 1


def test_changed_selection_or_source_cannot_resume(small_plan, tmp_path):
    plan, audio = small_plan
    run_plan(plan, audio, tmp_path / 'out', tmp_path / 'cache', tmp_path / 'logs')
    plan['tasks'] = plan['tasks'][:1]
    with pytest.raises(ValueError, match='changed'):
        run_plan(plan, audio, tmp_path / 'out', tmp_path / 'cache', tmp_path / 'logs', resume=True)


def test_live_lock_rejects_duplicate_and_releases(tmp_path):
    with sweep_lock(tmp_path):
        with pytest.raises(ValueError, match='live runner'):
            with sweep_lock(tmp_path):
                pytest.fail('Duplicate runner acquired the lock')
    with sweep_lock(tmp_path):
        pass


def test_evaluation_uses_clean_model_without_fit_and_checks_dependency(tmp_path, monkeypatch):
    from drone_fingerprint import workflow
    manifest, subset = create_fixture(tmp_path / 'audio')
    records = select_recordings(manifest, subset, (1, 2))
    settings = Settings()
    clean = tmp_path / 'clean'
    run_experiment(records, manifest.parent, settings, clean, {})

    def no_new_model(*args):
        pytest.fail('Evaluation must not initialize a fresh estimator')

    monkeypatch.setattr(workflow, 'make_model', no_new_model)
    noisy = replace(settings, test_snr_quarter_db=0)
    result = run_experiment(records, manifest.parent, noisy, tmp_path / 'noisy', {}, trained_from=clean)
    assert result['model_reference']['sha256'] == sha256(clean / 'model.joblib')
    with pytest.raises(ValueError, match='provenance'):
        run_experiment(records, manifest.parent, noisy, tmp_path / 'wrong-data',
                       {'subset_manifest_sha256': 'changed'}, trained_from=clean)
    with pytest.raises(ValueError, match='configuration'):
        run_experiment(records, manifest.parent, replace(noisy, seed=1), tmp_path / 'wrong-seed', {}, trained_from=clean)
    (clean / 'model.joblib').write_bytes(b'broken')
    with pytest.raises(ValueError, match='checksum|size'):
        completed_result(tmp_path / 'noisy')


def test_published_result_is_recovered_after_ledger_write_crash(small_plan, tmp_path, monkeypatch):
    from drone_fingerprint import workflow
    plan, audio = small_plan
    plan['tasks'] = plan['tasks'][:1]
    output = tmp_path / 'crashed'
    state = run_plan(plan, audio, output, tmp_path / 'cache', tmp_path / 'logs')
    entry = next(iter(state['tasks'].values()))
    entry['status'] = entry['attempts'][0]['status'] = state['status'] = 'running'
    (output / 'sweep.json').write_text(json.dumps(state))
    monkeypatch.setattr(workflow, 'make_model', lambda *args: pytest.fail('Published result must be reused'))
    restored = run_plan(plan, audio, output, tmp_path / 'cache', tmp_path / 'logs', resume=True)
    assert restored['status'] == 'completed'
    assert len(next(iter(restored['tasks'].values()))['attempts']) == 1


def test_missing_input_is_failed_and_recovery_preserves_failed_attempt(small_plan, tmp_path):
    plan, audio = small_plan
    plan['tasks'] = plan['tasks'][:1]
    output = tmp_path / 'failed'
    wav = audio / plan['record_sets']['DS2'][0]['path']
    moved = wav.with_suffix('.hidden')
    wav.rename(moved)
    try:
        with pytest.raises(FileNotFoundError):
            run_plan(plan, audio, output, tmp_path / 'cache', tmp_path / 'logs')
    finally:
        moved.rename(wav)
    state = json.loads((output / 'sweep.json').read_text())
    entry = next(iter(state['tasks'].values()))
    assert state['status'] == entry['status'] == 'failed'
    assert not (output / entry['attempts'][0]['directory'] / 'result.json').exists()
    restored = run_plan(plan, audio, output, tmp_path / 'cache', tmp_path / 'logs', resume=True)
    attempts = next(iter(restored['tasks'].values()))['attempts']
    assert [a['status'] for a in attempts] == ['failed', 'completed']


def test_paper_qda_matches_unbiased_covariance_density_and_warns_at_low_scale():
    from drone_fingerprint.qda import PaperQDA
    rng = np.random.default_rng(2022)
    train = np.vstack([rng.normal(size=(30, 4)) * 1e-5 + mean for mean in (0, 0.001)])
    labels = np.repeat([1, 2], 30)
    test = train[::7]
    with pytest.warns(UserWarning, match='collinear'):
        model = PaperQDA().fit(train, labels)
    oracle = []
    for label in (1, 2):
        group = train[labels == label]
        covariance = np.cov(group, rowvar=False, ddof=1)
        delta = test - group.mean(axis=0)
        oracle.append(-0.5 * (np.einsum('ij,jk,ik->i', delta, np.linalg.inv(covariance), delta)
                              + np.linalg.slogdet(covariance)[1]) + np.log(0.5))
    np.testing.assert_allclose(model._decision_function(test), np.asarray(oracle).T, rtol=1e-12, atol=1e-8)
    np.testing.assert_array_equal(model.predict(test), np.argmax(oracle, axis=0) + 1)


def test_paper_qda_rejects_degenerate_data_and_unsupported_solver():
    from drone_fingerprint.qda import PaperQDA
    with pytest.warns(UserWarning), pytest.raises(ValueError, match='zero or nonfinite'):
        PaperQDA().fit(np.zeros((6, 2)), [1, 1, 1, 2, 2, 2])
    with pytest.raises(ValueError, match='historical SVD'):
        PaperQDA(solver='eigen').fit(np.ones((6, 2)), [1, 1, 1, 2, 2, 2])
