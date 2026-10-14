"""Verify P6 completions and gallery, inventory resources, and compare printed tables."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

from drone_fingerprint.data import sha256
from drone_fingerprint.storage import external_path, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--audio-root', type=Path, required=True)
    args = parser.parse_args()
    evidence = external_path(args.evidence)
    def read(path):
        return json.loads(path.read_text())
    session = read(evidence / 'session.json')
    run, cache = Path(session['run']), Path(session['cache'])
    plan, state = read(run / 'plan.json'), read(run / 'sweep.json')
    gallery = evidence / 'gallery'
    manifest = read(gallery / 'manifest.json')
    assert state['status'] == 'completed' and len(state['tasks']) == 8440
    assert manifest['status'] == 'passed' and manifest['full_dataset'] and manifest['full_grid']
    assert manifest['verified_results'] == 8440 and not manifest['skipped'] and not manifest['duplicate_results_merged']
    assert set(manifest['sources']) == set(state['tasks']) == {task['id'] for task in plan['tasks']}
    execution = read(evidence / 'execution-result.json')
    assert execution['training_exit_code'] == execution['plot_exit_code'] == 0
    warnings, times, cache_events = Counter(), Counter(), Counter()
    artifacts, result_signatures = {}, {}
    for task in plan['tasks']:
        entry = state['tasks'][task['id']]
        assert entry['status'] == 'completed' and len(entry['attempts']) == 1
        attempt = entry['attempts'][0]
        assert attempt['status'] == 'completed'
        directory = run / attempt['directory']
        result_path = directory / 'result.json'
        result = read(result_path)
        assert result['task_id'] == entry['task_id'] and result['configuration'] == task['settings']
        assert sha256(result_path) == manifest['sources'][task['id']][0]['result_sha256']
        result_signatures[str(result_path.relative_to(run))] = {'sha256': sha256(result_path), 'mtime_ns': result_path.stat().st_mtime_ns}
        for name, record in result['artifacts'].items():
            path = directory / name
            assert path.stat().st_size == record['bytes'] and sha256(path) == record['sha256'], str(path)
            artifacts[str(path.relative_to(run))] = record
        if task['kind'] == 'evaluate':
            reference = result['model_reference']
            source = (directory / reference['path']).resolve()
            assert source == (run / state['tasks'][task['dependency']]['attempts'][0]['directory']).resolve()
            assert reference['sha256'] == read(source / 'result.json')['artifacts']['model.joblib']['sha256']
        assert result['thread_limit'] == 1 and not (directory / 'failure.json').exists()
        for warning in result['warnings']:
            warnings[(task['preset'], task['settings']['model'], warning['category'], warning['message'])] += 1
        for phase, seconds in result['timings_s'].items():
            times[(task['preset'], task['settings']['model'], phase)] += seconds
        for key in ('base_hits', 'base_misses', 'selected_hits', 'selected_misses', 'hashed_recordings'):
            cache_events[key] += result['cache'][key]
    print('Verified 8,440 result records and their saved model/prediction hashes.', flush=True)
    for name, record in manifest['artifacts'].items():
        path = gallery / name
        assert path.stat().st_size == record['bytes'] and sha256(path) == record['sha256']
    series = read(gallery / 'plot-data.json')['series']
    assert sum(len(row['y']) for row in series) == 8352
    assert all(value is not None for row in series for value in row['y'])
    cache_arrays = []
    for directory in cache.iterdir():
        if not directory.is_dir():
            continue
        assert not directory.name.startswith('.building-')
        metadata = read(directory / 'metadata.json')
        path = directory / 'features.npy'
        assert metadata['status'] == 'complete' and path.stat().st_size == metadata['bytes']
        assert sha256(path) == metadata['sha256']
        cache_arrays.append({'path': str(path.relative_to(cache)), 'bytes': metadata['bytes'], 'sha256': metadata['sha256']})
    records = read(Path(session['frozen_config']) / 'paper/recordings.json')['recordings']
    for record in records:
        path = args.audio_root / record['path']
        assert path.stat().st_size == record['bytes'] and sha256(path) == record['sha256']
        assert not path.stat().st_mode & 0o222
    print(f'Verified {len(cache_arrays)} cache arrays and {len(records)} read-only source WAVs.', flush=True)
    def inventory(root):
        groups = defaultdict(lambda: {'files': 0, 'bytes': 0, 'allocated_bytes': 0})
        for path in root.rglob('*'):
            if path.is_file():
                stat = path.stat()
                item = groups[path.suffix or '(no suffix)']
                item['files'] += 1; item['bytes'] += stat.st_size; item['allocated_bytes'] += stat.st_blocks * 512
        return dict(groups)
    published = read(evidence / 'published-tables.json')['tables']
    tables = read(gallery / 'tables.json')['tables']
    comparisons = []
    for row in tables['table2']['rows']:
        for metric, printed_key in [('accuracy', 'accuracy'), ('macro_precision', 'precision'), ('macro_recall', 'recall'), ('macro_f1', 'f1')]:
            printed = published['table2']['rows'][row['model']][printed_key]
            comparisons.append({'table': 'table2', 'model': row['model'], 'metric': metric, 'trial': None,
                                'published_percent': printed, 'rerun_percent': row[metric], 'difference_pp': row[metric] - printed})
    for table in ('table4', 'table5', 'table5_pooled_comparison'):
        source = 'table5' if table.startswith('table5') else table
        for printed, row in zip(published[source]['rows'], tables[table]['rows'][:11], strict=True):
            assert str(row['trial']) == printed['trial'] or row['trial'] == 'Mean (10 trials)' and printed['trial'] == 'Average'
            for model in published['table2']['rows']:
                value = row[model]
                assert value is not None
                comparisons.append({'table': table, 'model': model, 'metric': tables[table]['metric'], 'trial': printed['trial'],
                                    'published_percent': printed[model], 'rerun_percent': value, 'difference_pp': value - printed[model]})
    assert len(comparisons) == 296
    events = [json.loads(line) for line in (evidence / 'execution.jsonl').read_text().splitlines()]
    begin = next(row for row in events if row['event'] == 'trainer_start')
    end = next(row for row in events if row['event'] == 'trainer_exit')
    summary = {'status': 'passed', 'tasks': 8440, 'fits': 4096, 'evaluations': 4344, 'attempts': 8440,
               'missing_figure_points': 0, 'verified_figure_points': 8352, 'verified_wavs': len(records),
               'verified_cache_arrays': len(cache_arrays), 'wall_seconds': (datetime.fromisoformat(end['at']) - datetime.fromisoformat(begin['at'])).total_seconds(),
               'trainer_exit': end, 'cache_events': dict(cache_events), 'run_inventory': inventory(run),
               'cache_inventory': inventory(cache), 'gallery_inventory': inventory(gallery),
               'warnings': [{'preset': k[0], 'model': k[1], 'category': k[2], 'message': k[3], 'count': v} for k, v in warnings.items()],
               'phase_seconds': [{'preset': k[0], 'model': k[1], 'phase': k[2], 'seconds': v} for k, v in times.items()]}
    atomic_json(evidence / 'full-validation.json', summary)
    atomic_json(evidence / 'paper-comparison.json', {'units': 'percentage points; rerun minus printed percent', 'rows': comparisons})
    atomic_json(evidence / 'verified-inputs.json', {'results': result_signatures, 'artifacts': artifacts, 'cache_arrays': cache_arrays})
    print(json.dumps({key: summary[key] for key in ('status', 'tasks', 'wall_seconds', 'verified_cache_arrays', 'verified_wavs')}, indent=2))


if __name__ == '__main__':
    main()
