"""Isolate background-class ordering on saved full-data open-set features.

This diagnostic changes only training label order, maps predictions back to the
frozen public labels and writes separate evidence. It never replaces sweep results.
"""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from drone_fingerprint.artifacts import completed_result
from drone_fingerprint.config import Settings
from drone_fingerprint.metrics import open_set_metrics
from drone_fingerprint.models import make_model
from drone_fingerprint.pipeline import prepare_features
from drone_fingerprint.storage import create_output, atomic_json
from drone_fingerprint.workflow import timed_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--audio-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = create_output(args.output, args.audio_root)
    plan = json.loads((args.run / 'plan.json').read_text())
    state = json.loads((args.run / 'sweep.json').read_text())
    assert state['status'] == 'completed'
    rows = []
    with threadpool_limits(limits=1):
        for task in plan['tasks']:
            if task['preset'] != 'tables4_5' or task['settings']['model'] not in ('RBF-SVM', 'KNN'):
                continue
            settings = Settings(**{**task['settings'], 'identities': tuple(task['settings']['identities'])})
            directory = args.run / state['tasks'][task['id']]['attempts'][-1]['directory']
            original = completed_result(directory, state['tasks'][task['id']]['task_id'])
            with np.load(directory / 'predictions.npz', allow_pickle=False) as saved:
                saved_prediction = saved['prediction'].copy()
            arrays, labels, identities, segments, _, _ = prepare_features(
                plan['record_sets']['open'], args.audio_root, settings, args.cache_root)
            assert original['segments'] == segments
            assert set(labels['train']) == {0, *settings.partition['registered']}
            trial = settings.partition['trial']
            # Trial 1 repeats the unchanged procedure before the controlled relabeling.
            variants = ('unchanged', 'background_last') if trial == 1 else ('background_last',)
            for variant in variants:
                train_labels = labels['train'] if variant == 'unchanged' else np.where(labels['train'] == 0, 25, labels['train'])
                timings = {}
                model = make_model(settings.model, settings.seed)
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter('always')
                    with timed_phase(f'{settings.model} trial {trial} {variant} fit', timings):
                        model.fit(arrays['train'], train_labels)
                    with timed_phase(f'{settings.model} trial {trial} {variant} predict', timings):
                        prediction = model.predict(arrays['test'])
                if variant == 'background_last':
                    prediction = np.where(prediction == 25, 0, prediction)
                else:
                    np.testing.assert_array_equal(prediction, saved_prediction)
                metrics = open_set_metrics(identities, prediction, settings.partition)
                filename = f'trial-{trial:02d}-{settings.model}-{variant}.npz'
                np.savez(output / filename, prediction=prediction, original_identity=identities)
                rows.append({'task': task['id'], 'trial': trial, 'model': settings.model, 'variant': variant,
                             'changed_predictions': int(np.count_nonzero(prediction != saved_prediction)),
                             'metrics': metrics, 'baseline_metrics': original['metrics'], 'timings_s': timings,
                             'warnings': [{'category': w.category.__name__, 'message': str(w.message)} for w in observed],
                             'predictions': filename})
                atomic_json(output / 'diagnostic.json', {'status': 'running', 'rows': rows})
                print(json.dumps({k: rows[-1][k] for k in ('trial', 'model', 'variant', 'changed_predictions')}), flush=True)
    assert len(rows) == 22
    atomic_json(output / 'diagnostic.json', {'status': 'passed', 'scope': 'Controlled class-order diagnostic, not a replacement protocol or recovered historical environment.',
                'changed_factor': 'Background training label 0 -> 25; predictions mapped back to 0. Feature values, row order, seed and model parameters unchanged.',
                'legacy_reference': 'Historical Trainer_pkl.label_proc assigns background label 24 after drone labels 0..23; modern public labels are 1..24 with background 0.',
                'rows': rows})


if __name__ == '__main__':
    main()
