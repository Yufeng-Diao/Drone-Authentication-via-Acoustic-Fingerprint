"""Expand paper grids and execute sequential, restartable task attempts.

The planner is plain data. The runner uses the same single-task workflow as the
CLI; a completed result is the only authority for reuse, never a log message.
"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
from itertools import product
import json
import os
from pathlib import Path
import sys
import subprocess
import uuid

from .artifacts import digest, task_identity
from .config import MODELS, Settings
from .data import labelled_intervals, select_recordings, select_room_noise, sha256
from .features import frame_geometry
from .storage import atomic_json, external_path
from .workflow import environment, run_experiment


def paper_tasks(protocol: dict, partitions: dict, presets: dict, models: list[str]) -> list[dict]:
    """Expand selected values, including clean-model dependencies before evaluations."""
    experiments = {e['id']: e for e in protocol['experiments']}
    if not presets or not set(presets) <= experiments.keys():
        raise ValueError('Select one or more known paper presets')
    if not models or len(set(models)) != len(models) or not set(models) <= set(MODELS):
        raise ValueError('Select distinct paper model names')
    if protocol['schema_version'] != 1 or partitions['schema_version'] != 1:
        raise ValueError('Unsupported paper protocol or partition schema')
    roles = {p['trial']: p for p in partitions['partitions']}
    if len(roles) != 10 or set(roles) != set(range(1, 11)):
        raise ValueError('Expected ten unique frozen attack partitions')
    tasks = {}

    def add(preset, settings, dependency=None):
        key = preset + '-' + digest(asdict(settings))[:16]
        tasks.setdefault(key, {'id': key, 'preset': preset, 'kind': 'evaluate' if dependency else 'fit',
                               'settings': asdict(settings), 'dependency': dependency})
        return key

    for name, filters in presets.items():
        experiment = experiments[name]
        base_name = experiment.get('reuse_models_from', name)
        base = experiments[base_name]
        values = {'frame_ms': base['frame_ms'], 'n_filters': base['n_filters'], 'blocks': base['blocks'],
                  'portion': [3 * a // b for a, b in base['fractions']],
                  'trial': base.get('partition_trials', [None]),
                  'snr_quarter_db': experiment.get('snr_quarter_db', [None])}
        if not isinstance(filters, dict) or not set(filters) <= values.keys():
            raise ValueError(f'Unknown sweep selector for {name}')
        for key, selected in filters.items():
            if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
                    or any(type(v) is not int or v not in values[key] for v in selected)):
                raise ValueError(f'{name}: {key} must select distinct values from the frozen grid')
            values[key] = selected
        identities = protocol['datasets']['DS1' if base['dataset'] == 'DS1' else 'DS2']
        for frame, filters_count, blocks, portion, trial, snr, model in product(*values.values(), models):
            settings = Settings(tuple(identities), frame, filters_count, model, 2022, blocks, portion,
                                partition=roles[trial] if trial is not None else None)
            dependency = add(base_name, settings)
            if base_name != name:
                add(name, replace(settings, test_snr_quarter_db=snr), dependency)
    return list(tasks.values())


def load_plan(path: Path, *, presets=None, models=None) -> dict:
    """Resolve a strict sweep file and estimate feature sizes without reading WAVs."""
    raw = json.loads(path.read_text(encoding='utf-8'))
    keys = {'schema_version', 'protocol', 'recordings_manifest', 'subset_manifest', 'partitions_manifest', 'presets', 'models'}
    if set(raw) != keys or raw['schema_version'] != 1:
        raise ValueError(f'Sweep configuration requires exactly {sorted(keys)}')
    paths = {key: (path.parent / raw[key]).resolve() for key in
             ('protocol', 'recordings_manifest', 'subset_manifest', 'partitions_manifest')}
    protocol = json.loads(paths['protocol'].read_text(encoding='utf-8'))
    partitions = json.loads(paths['partitions_manifest'].read_text(encoding='utf-8'))
    selected = raw['presets'] if presets is None else {name: raw['presets'].get(name, {}) for name in presets}
    tasks = paper_tasks(protocol, partitions, selected, raw['models'] if models is None else models)
    record_sets = {}
    for task in tasks:
        settings = Settings(**{**task['settings'], 'identities': tuple(task['settings']['identities'])})
        group = 'open' if settings.partition else ('DS1' if len(settings.identities) == 8 else 'DS2')
        if group not in record_sets:
            records = select_recordings(paths['recordings_manifest'], paths['subset_manifest'], settings.identities)
            if settings.partition:
                records.append(select_room_noise(paths['recordings_manifest'], paths['subset_manifest']))
            record_sets[group] = records
        task['record_group'] = group
        rows = {'train': 0, 'test': 0}
        for record in record_sets[group]:
            for split, start, end, _ in labelled_intervals(record, settings.partition):
                rows[split] += frame_geometry(end - start, settings)[2]
        columns = ((settings.n_filters - 1) * settings.portion // 3) * settings.blocks
        task['resources'] = {**{k + '_rows': v for k, v in rows.items()}, 'feature_columns': columns,
                             'base_and_selected_bytes': sum(rows.values()) * (settings.n_filters + columns) * 8}
    return {'schema_version': 1, 'tasks': tasks, 'record_sets': record_sets,
            'inputs': {key: {'path': str(value), 'sha256': sha256(value)} for key, value in paths.items()},
            'counts': {'fits': sum(t['kind'] == 'fit' for t in tasks),
                       'evaluations': sum(t['kind'] == 'evaluate' for t in tasks), 'tasks': len(tasks)},
            'resource_note': 'Array bytes per task before cache sharing. Estimators allocate additional memory; '
                             'model sizes and wall time are not predicted. Execution is sequential with one numerical thread.'}


def now():
    return datetime.now(timezone.utc).isoformat()


def source_checkout() -> dict:
    """Record checkout provenance when running from source; task IDs also hash code."""
    root = Path(__file__).resolve().parents[2]
    if not (root / '.git').exists():
        return {'kind': 'installed distribution; implementation content is bound by plan_id'}
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=root, check=True, capture_output=True, text=True).stdout
    return {'kind': 'checkout', 'revision': revision, 'dirty': bool(dirty)}


@contextmanager
def sweep_lock(output: Path):
    """Fail promptly on duplicate runners; a killed process releases its Linux lock."""
    with (output / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('This sweep already has a live runner; do not start a duplicate fit') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Transcript:
    """Mirror Python output to the terminal and one external attempt log."""

    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        return self.log.write(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def run_plan(plan: dict, audio_root: Path, output: Path, cache_root: Path, log_root: Path, *, resume=False) -> dict:
    """Keep immutable attempts, restart unfinished fits, and publish unique completions."""
    output, cache_root, log_root = [external_path(p, audio_root) for p in (output, cache_root, log_root)]
    roots = [output, cache_root, log_root, audio_root.resolve()]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i + 1:]):
        raise ValueError('Audio, output, cache and log directories must be separate')
    identity = task_identity([], Settings(), {**plan, 'inputs': {k: v['sha256'] for k, v in plan['inputs'].items()}}, environment(), False)
    output.mkdir(parents=True, exist_ok=resume)
    log_root.mkdir(parents=True, exist_ok=True)
    with sweep_lock(output):
        state_path = output / 'sweep.json'
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding='utf-8'))
            if state['plan_id'] != identity:
                raise ValueError('Sweep source, runtime, inputs or selection changed; use a new output directory')
        else:
            # A crash before plan publication cannot authorize adopting arbitrary output.
            if any(p.name != '.lock' for p in output.iterdir()):
                raise ValueError('Output contains files without a sweep completion ledger')
            state = {'schema_version': 1, 'plan_id': identity, 'started_at': now(), 'status': 'pending',
                     'source_checkout': source_checkout(),
                     'tasks': {t['id']: {'status': 'pending', 'attempts': []} for t in plan['tasks']}}
            atomic_json(output / 'plan.json', plan)
        state.update(status='running', pid=os.getpid())
        atomic_json(state_path, state)
        for number, task in enumerate(plan['tasks'], 1):
            entry = state['tasks'][task['id']]
            settings = Settings(**{**task['settings'], 'identities': tuple(task['settings']['identities'])})
            records = plan['record_sets'][task['record_group']]
            provenance = {'experiment': task['preset'], 'config_id': task['id'], 'dataset': 'Explicit manifest selection',
                          **{k + '_sha256': v['sha256'] for k, v in plan['inputs'].items()}}
            dependency = None
            if task['dependency']:
                dependency = output / state['tasks'][task['dependency']]['attempts'][-1]['directory']
            previous = entry['attempts'][-1] if entry['attempts'] else None
            print(f"{now()} Task {number}/{len(plan['tasks'])}: {task['preset']} {settings.model}; "
                  f"frame={settings.frame_ms}ms filters={settings.n_filters} blocks={settings.blocks} "
                  f"portion={settings.portion}/3 trial={settings.partition['trial'] if settings.partition else '-'} "
                  f"SNR={settings.test_snr_quarter_db}; completed={sum(e['status'] == 'completed' for e in state['tasks'].values())}", flush=True)
            attempt = None
            try:
                if previous and (entry['status'] == 'completed' or (output / previous['directory'] / 'result.json').exists()):
                    result = run_experiment(records, audio_root, settings, output / previous['directory'], provenance,
                                            resume=True, cache_root=cache_root, trained_from=dependency)
                    previous['status'] = entry['status'] = 'completed'
                    entry['task_id'] = result['task_id']
                    atomic_json(state_path, state)
                    continue
                if previous and previous['status'] == 'running':
                    previous.update(status='interrupted', detected_at=now(), reason='Previous process ended without completion')
                directory = f"tasks/{task['id']}/attempt-{len(entry['attempts']) + 1:03d}"
                log = log_root / f"{task['id']}-attempt-{len(entry['attempts']) + 1:03d}-{uuid.uuid4().hex[:8]}.log"
                attempt = {'directory': directory, 'log': str(log), 'status': 'running', 'started_at': now()}
                entry['attempts'].append(attempt)
                entry['status'] = 'running'
                atomic_json(state_path, state)
                with log.open('x', encoding='utf-8') as stream:
                    with redirect_stdout(Transcript(sys.stdout, stream)), redirect_stderr(Transcript(sys.stderr, stream)):
                        print(f"{now()} {task['id']}: {task['kind']} attempt started", flush=True)
                        result = run_experiment(records, audio_root, settings, output / directory, provenance,
                                                cache_root=cache_root, trained_from=dependency)
                        print(f"{now()} {task['id']}: attempt completed", flush=True)
                attempt.update(status='completed', finished_at=now())
                entry.update(status='completed', task_id=result['task_id'])
                entry.pop('error', None)
                atomic_json(state_path, state)
            except BaseException as error:
                status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
                if attempt:
                    attempt.update(status=status, finished_at=now(), error=repr(error))
                entry.update(status=status, error=repr(error))
                state.update(status=status, finished_at=now())
                atomic_json(state_path, state)
                raise
        state.update(status='completed', finished_at=now())
        atomic_json(state_path, state)
        return state
