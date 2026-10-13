"""Read verified sweep completions and arrange paper data without training."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from .artifacts import completed_result
from .config import MODELS, Settings
from .data import sha256
from .experiments import paper_tasks
from .features import feature_metadata
from .metrics import classification_metrics, open_set_metrics

PRESETS = ('figure2', 'figure3', 'figure4', 'table2', 'figure5', 'tables4_5')


def collect_results(runs: list[Path], paper_dir: Path) -> dict:
    """Use only each task's latest completed attempt; never scan old attempt folders.

    Inputs, runtime and seed must agree. Identical repeated predictions are
    deduplicated; conflicting repeats require the researcher to choose a run.
    Models are checksummed, never deserialized. Audio and feature caches are not read.
    """
    protocol = json.loads((paper_dir / 'protocol.json').read_text())
    partitions = json.loads((paper_dir / 'attack_partitions.json').read_text())
    expected = {t['id']: t for t in paper_tasks(protocol, partitions, {p: {} for p in PRESETS}, list(MODELS))}
    hashes = {key: sha256(paper_dir / name) for key, name in
              (('protocol', 'protocol.json'), ('recordings_manifest', 'recordings.json'),
               ('partitions_manifest', 'attack_partitions.json'))}
    results, sources, skipped, duplicate_count = {}, {}, [], 0
    selection = None
    for root in runs:
        root = root.resolve()
        state = json.loads((root / 'sweep.json').read_text())
        plan = json.loads((root / 'plan.json').read_text())
        if state['schema_version'] != 1 or plan['schema_version'] != 1:
            raise ValueError('Unsupported sweep schema')
        tasks = {t['id']: t for t in plan['tasks']}
        if len(tasks) != len(plan['tasks']) or set(tasks) != set(state['tasks']):
            raise ValueError('Sweep task list and status ledger disagree')
        if any(plan['inputs'][k]['sha256'] != value for k, value in hashes.items()):
            raise ValueError('Sweep input manifest or paper protocol differs from the plotting specification')
        for key, task in tasks.items():
            if key not in expected or any(task[k] != json.loads(json.dumps(expected[key][k])) for k in
                                          ('preset', 'kind', 'settings', 'dependency')):
                raise ValueError(f'Task is outside its declared paper grid: {key}')
            entry = state['tasks'][key]
            if entry['status'] != 'completed':
                skipped.append({'run': str(root), 'id': key, 'status': entry['status']})
                continue
            if not entry['attempts'] or entry['attempts'][-1]['status'] != 'completed':
                raise ValueError('Completed task has no completed current attempt')
            directory = (root / entry['attempts'][-1]['directory']).resolve()
            if not directory.is_relative_to(root):
                raise ValueError('Attempt directory escapes its sweep')
            result = completed_result(directory, entry['task_id'])
            if result['configuration'] != task['settings']:
                raise ValueError('Result configuration disagrees with its planned task')
            settings = Settings(**{**result['configuration'], 'identities': tuple(result['configuration']['identities'])})
            if result['feature_metadata'] != feature_metadata(settings):
                raise ValueError('Result feature convention differs from the plotting specification')
            if result['provenance'].get('experiment') != task['preset'] or result['provenance'].get('config_id') != key:
                raise ValueError('Result experiment identity disagrees with the task ledger')
            if any(result['provenance'].get(k + '_sha256') != v['sha256'] for k, v in plan['inputs'].items()):
                raise ValueError('Result input provenance disagrees with its sweep')
            identity = {'inputs': {k: v['sha256'] for k, v in plan['inputs'].items()},
                        'environment': result['environment'], 'seed': settings.seed}
            if selection is not None and identity != selection:
                raise ValueError('Cannot combine different datasets, runtimes, seeds or protocols in one figure set')
            selection = identity
            if (result.get('model_reference') is not None) != (task['kind'] == 'evaluate'):
                raise ValueError('Evaluation task must reference its clean trained model')
            with np.load(directory / 'predictions.npz', allow_pickle=False) as arrays:
                truth, prediction, original = (arrays[k] for k in ('truth', 'prediction', 'original_identity'))
                metrics = classification_metrics(truth, prediction, result['labels'])
                if settings.partition:
                    metrics.update(open_set_metrics(original, prediction, settings.partition))
                if (len(truth) != result['test_rows'] or original.shape != truth.shape
                        or confusion_matrix(truth, prediction, labels=result['labels']).tolist() != result['confusion_matrix']):
                    raise ValueError('Prediction counts or confusion matrix disagree with saved results')
            if metrics != result['metrics']:
                raise ValueError('Saved metrics disagree with the prediction arrays')
            source = {'run': str(root), 'attempt': str(directory), 'task_id': result['task_id'],
                      'result_sha256': sha256(directory / 'result.json'), 'plan_id': state['plan_id'],
                      'source_checkout': state.get('source_checkout')}
            if key in results:
                previous = results[key]
                if (previous['metrics'] != metrics or previous['artifacts']['predictions.npz'] != result['artifacts']['predictions.npz']
                        or previous['segments'] != result['segments'] or previous['model_parameters'] != result['model_parameters']):
                    raise ValueError(f'Conflicting repetitions for {key}; select one sweep explicitly')
                duplicate_count += 1
            else:
                results[key] = result
            sources.setdefault(key, []).append(source)
    if not results:
        raise ValueError('No verified completed paper tasks were found')
    full_data = selection['inputs']['subset_manifest'] == sha256(paper_dir / 'full_subset.json')
    return {'results': results, 'sources': sources, 'skipped': skipped, 'duplicates': duplicate_count,
            'selection': selection, 'full_dataset': full_data, 'protocol': protocol, 'partitions': partitions,
            'expected_counts': {p: sum(t['preset'] == p for t in expected.values()) for p in PRESETS}}


def figure_series(bundle: dict, preset: str) -> list[dict]:
    """One complete x grid per panel/model; missing y values remain null (NaN in plots)."""
    experiments = {e['id']: e for e in bundle['protocol']['experiments']}
    base = experiments[experiments[preset].get('reuse_models_from', preset)]
    axis = 'frame_ms' if preset == 'figure2' else ('snr_quarter_db' if preset == 'figure5' else 'n_filters')
    values = experiments[preset]['snr_quarter_db'] if preset == 'figure5' else base[axis]
    panels = [(b, p) for b in (1, 2, 3) for p in (1, 2, 3)] if preset in ('figure3', 'figure4') else [(1, 3)]
    series = []
    for blocks, portion in panels:
        for model in MODELS:
            observed = {}
            for result in bundle['results'].values():
                s = result['configuration']
                if (result['provenance']['experiment'] == preset and s['model'] == model
                        and s['blocks'] == blocks and s['portion'] == portion):
                    x = s['test_snr_quarter_db'] if axis == 'snr_quarter_db' else s[axis]
                    observed[x] = result['metrics']['accuracy'] * 100
            series.append({'preset': preset, 'model': model, 'blocks': blocks, 'portion': portion,
                           'x_unit': 'dB' if preset == 'figure5' else ('ms' if preset == 'figure2' else 'filters'),
                           'y_unit': 'percent', 'x': [x / 4 if preset == 'figure5' else x for x in values],
                           'y': [observed.get(x) for x in values]})
    return series


def paper_tables(bundle: dict) -> dict:
    """Export the paper tables; a ten-trial mean requires all ten observations."""
    results = list(bundle['results'].values())
    table2 = []
    for model in MODELS:
        match = next((r for r in results if r['provenance']['experiment'] == 'table2' and r['configuration']['model'] == model), None)
        table2.append({'model': model, **{key: match['metrics'][key] * 100 if match else None
                                       for key in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')}})
    tables = {'table2': {'columns': list(table2[0]), 'rows': table2},
              'table3': {'columns': ['trial', 'registered', 'background', 'attack'], 'rows': bundle['partitions']['partitions']}}
    for name, metric in (('table4', 'registered_macro_recall'), ('table5', 'attack_macro_rejection'),
                         ('table5_pooled_comparison', 'attack_pooled_rejection')):
        rows = []
        for trial in range(1, 11):
            row = {'trial': trial}
            for model in MODELS:
                result = next((r for r in results if r['provenance']['experiment'] == 'tables4_5'
                               and r['configuration']['model'] == model and r['configuration']['partition']['trial'] == trial), None)
                row[model] = result['metrics'][metric] * 100 if result else None
            rows.append(row)
        average = {'trial': 'Mean (10 trials)'}
        coverage = {'trial': 'Trials present'}
        for model in MODELS:
            values = [r[model] for r in rows if r[model] is not None]
            average[model] = float(np.mean(values)) if len(values) == 10 else None
            coverage[model] = f'{len(values)}/10'
        tables[name] = {'metric': metric, 'columns': ['trial', *MODELS], 'rows': rows + [average, coverage]}
    return tables
