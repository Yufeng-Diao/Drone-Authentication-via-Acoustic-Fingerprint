"""Validate external P4 acceptance records and render a headless review report."""

import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from drone_fingerprint.artifacts import completed_result
from drone_fingerprint.config import MODELS
from drone_fingerprint.storage import external_path

PRESETS = ('figure2', 'figure3', 'figure4', 'table2', 'figure5', 'tables4_5')


def read_sweep(root):
    state = json.loads((root / 'sweep.json').read_text())
    plan = json.loads((root / 'plan.json').read_text())
    assert state['status'] == 'completed'
    results = {}
    for task in plan['tasks']:
        entry = state['tasks'][task['id']]
        assert entry['status'] == 'completed'
        attempt = root / entry['attempts'][-1]['directory']
        result = completed_result(attempt, entry['task_id'])
        assert result['configuration'] == task['settings']
        assert (result.get('model_reference') is not None) == (task['kind'] == 'evaluate')
        if task['kind'] == 'evaluate':
            assert 'training' not in result['timings_s'] and not (attempt / 'model.joblib').exists()
        results[task['id']] = (task, result, attempt)
    return state, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    args = parser.parse_args()
    evidence, runs = external_path(args.evidence), external_path(args.runs)
    previous = json.loads((evidence / 'manifest.json').read_text()) if (evidence / 'manifest.json').exists() else {}
    planned = json.loads((evidence / 'full-plan.json').read_text())
    assert planned['counts'] == {'fits': 4096, 'evaluations': 4344, 'tasks': 8440}
    tests = {}
    for version in ('314', '313'):
        tree = ET.parse(evidence / f'tests-{version}.xml')
        assert not tree.findall('.//failure') and not tree.findall('.//error')
        tests[version] = len(tree.findall('.//testcase'))
    sweeps = {name: read_sweep(runs / name) for name in
              ('eight-model-final', 'eight-model-py313', 'representative', 'slow-fit', 'recovery', 'fresh-comparison')}
    measured = {}
    for name in ('eight-model-final', 'representative', 'slow-fit'):
        measured.update(sweeps[name][1])
    smoke = {result['configuration']['model']: result for _, result, _ in sweeps['eight-model-final'][1].values()}
    assert set(smoke) == set(MODELS)
    for key, (_, first, first_path) in sweeps['eight-model-final'][1].items():
        _, second, second_path = sweeps['eight-model-py313'][1][key]
        assert first['metrics'] == second['metrics']
        with np.load(first_path / 'predictions.npz', allow_pickle=False) as a, np.load(second_path / 'predictions.npz', allow_pickle=False) as b:
            for field in a.files:
                np.testing.assert_array_equal(a[field], b[field])
    open_trials = [result for task, result, _ in measured.values() if task['preset'] == 'tables4_5']
    assert {r['configuration']['partition']['trial'] for r in open_trials} == set(range(1, 11))
    checks = json.loads((evidence / 'recovery-checks.json').read_text())
    assert checks['interrupt_exit_code'] == 130 and checks['completed_attempt_files_unchanged']
    assert not checks['repeat_resume_added_attempts']
    comparisons = []
    for key, (_, left, left_path) in sweeps['recovery'][1].items():
        _, right, right_path = sweeps['fresh-comparison'][1][key]
        assert left['metrics'] == right['metrics'] and left['segments'] == right['segments']
        with np.load(left_path / 'predictions.npz', allow_pickle=False) as a, np.load(right_path / 'predictions.npz', allow_pickle=False) as b:
            for field in a.files:
                np.testing.assert_array_equal(a[field], b[field])
        comparisons.append({'model': left['configuration']['model'], 'predictions_labels_intervals_metrics_exact': True})
    qda = json.loads((evidence / 'qda-replay.json').read_text())
    assert qda['max_absolute_score_difference'] == 0 and qda['predictions_exact']
    failed = json.loads((runs / 'eight-model-smoke/sweep.json').read_text())
    assert failed['status'] == 'failed' and not list((runs / 'eight-model-smoke').rglob('result.json'))
    commands = [json.loads(line) for line in (evidence / 'commands.jsonl').read_text().splitlines()]
    exits = {r['name']: r['exit_code'] for r in commands if 'exit_code' in r}
    for name in ('duplicate-live-run', 'changed-plan-rejected', 'original-qda-plan-source-rejected', 'missing-input-failure'):
        assert exits[name] == 1
    missing = json.loads((runs / 'missing-input/sweep.json').read_text())
    assert missing['status'] == 'failed' and not list((runs / 'missing-input').rglob('result.json'))
    integrity = json.loads((evidence / 'integrity.json').read_text())
    assert integrity['status'] == 'passed' and integrity['raw_wavs_read_only'] == 257
    records = []
    for task, result, directory in sweeps['slow-fit'][1].values():
        entry = sweeps['slow-fit'][0]['tasks'][task['id']]
        text = Path(entry['attempts'][-1]['log']).read_text()
        for match in re.finditer(r'(\S+) (training|evaluation): still running; pid=(\d+); elapsed=([\d.]+)s; process CPU=([\d.]+)s; VmRSS:\s+(\d+) kB', text):
            timestamp, phase, pid, elapsed, cpu, rss = match.groups()
            records.append({'model': result['configuration']['model'], 'phase': phase, 'timestamp': timestamp,
                            'pid': int(pid), 'elapsed_s': float(elapsed), 'cpu_s': float(cpu), 'rss_mib': int(rss) / 1024})
    assert any(r['phase'] == 'training' and r['elapsed_s'] >= 10 for r in records)
    (evidence / 'heartbeats.json').write_text(json.dumps(records, indent=2) + '\n')
    figures = evidence / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        fig.savefig(figures / (name + '.svg'), bbox_inches='tight')
        fig.savefig(figures / (name + '.png'), dpi=160, bbox_inches='tight')
        plt.close(fig)

    grid = np.zeros((len(MODELS), len(PRESETS)), dtype=int)
    for task, result, _ in measured.values():
        grid[MODELS.index(result['configuration']['model']), PRESETS.index(task['preset'])] += 1
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    ax.imshow(grid > 0, cmap=matplotlib.colors.ListedColormap(['#e9eef2', '#238d81']), vmin=0, vmax=1, aspect='auto')
    ax.set(xticks=range(6), xticklabels=PRESETS, yticks=range(8), yticklabels=MODELS,
           title='Completed validation tasks by model and preset')
    for i in range(8):
        for j in range(6):
            ax.text(j, i, str(grid[i, j]) if grid[i, j] else 'not run', ha='center', va='center', color='white' if grid[i, j] else '#53616c')
    save(fig, 'completion-grid')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for model, phase in sorted({(r['model'], r['phase']) for r in records}):
        group = [r for r in records if (r['model'], r['phase']) == (model, phase)]
        for ax, key, label in zip(axes, ('cpu_s', 'rss_mib'), ('Process CPU during phase (s)', 'Resident memory (MiB)')):
            ax.plot([r['elapsed_s'] for r in group], [r[key] for r in group], marker='o', label=f'{model} {phase}')
            ax.set(xlabel='Elapsed time within phase (s)', ylabel=label)
    axes[0].legend(fontsize=8)
    fig.suptitle('Actual slow-fit heartbeats: activity, not a convergence estimate')
    save(fig, 'heartbeats')

    state = sweeps['recovery'][0]
    start = datetime.fromisoformat(state['started_at'])
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    ticks, labels, index = [], [], 0
    for task, _, _ in sweeps['recovery'][1].values():
        for number, attempt in enumerate(state['tasks'][task['id']]['attempts'], 1):
            a = (datetime.fromisoformat(attempt['started_at']) - start).total_seconds()
            b = (datetime.fromisoformat(attempt['finished_at']) - start).total_seconds()
            color = '#bd6b38' if attempt['status'] != 'completed' else '#238d81'
            ax.barh(index, b-a, left=a, color=color)
            ticks.append(index); labels.append(f"{task['settings']['model']} attempt {number}: {attempt['status']}")
            index += 1
    ax.set(yticks=ticks, yticklabels=labels, xlabel='Seconds since sweep start', title='Actual interrupted and resumed attempt timeline')
    save(fig, 'recovery-timeline')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    names = [r['model'] for r in comparisons]
    for offset, (name, color) in enumerate((('recovery', '#238d81'), ('fresh-comparison', '#879bad'))):
        scores = [result['metrics']['accuracy'] for _, result, _ in sweeps[name][1].values()]
        axes[0].bar(np.arange(len(names)) + offset * 0.35, scores, width=0.35, label=name, color=color)
    axes[0].set(xticks=np.arange(len(names)) + 0.175, xticklabels=names, ylabel='Validation accuracy', ylim=(0, 1))
    axes[0].legend()
    axes[1].axis('off')
    axes[1].text(0.05, 0.8, 'Exact equality verified', fontsize=16, weight='bold')
    axes[1].text(0.05, 0.6, 'Truth and prediction arrays\nOriginal identities and sample intervals\nAll saved metrics\n\nCompleted attempt files retain hashes and mtimes\nRepeated resume adds zero attempts', va='top', linespacing=1.6)
    save(fig, 'recovery-equivalence')

    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    ax.bar(MODELS, [smoke[m]['timings_s']['training'] for m in MODELS], color='#367eab')
    ax.set(ylabel='Fit duration (s)', title='Eight-model validation fits: 8,110 rows × 200 features')
    save(fig, 'eight-model-fits')

    def link(path, label):
        return f'<a href="{escape(os.path.relpath(path, evidence))}">{escape(label)}</a>'

    rows = ''.join(f"<tr><td>{m}</td><td>{smoke[m]['timings_s']['training']:.3f}</td><td>{smoke[m]['metrics']['accuracy']:.4f}</td><td>{len(smoke[m]['warnings'])}</td></tr>" for m in MODELS)
    logs = ''.join('<li>' + link(evidence / r['log'], f"{r['name']} (exit {r['exit_code']})") + '</li>' for r in commands if 'exit_code' in r and 'log' in r)
    manifest = {**previous, 'phase': 'P4', 'status': 'passed', 'generated_at': datetime.now(timezone.utc).isoformat(),
                'baseline_commit': 'b76f73859ec43adfbb3de3139de15a66063031a4', 'tests': tests,
                'planned': planned['counts'], 'unique_validation_tasks': int(grid.sum()),
                'eight_models': list(smoke), 'ten_open_trials': True, 'heartbeat_count': len(records),
                'eight_model_predictions_and_metrics_equal_across_python': True,
                'recovery': {**checks, 'comparisons': comparisons}, 'qda_replay': qda,
                'limits': ['No full 4096-fit run', 'No estimator checkpointing', 'No remote CI or publication',
                           'Single observations, not a runtime benchmark', 'Other output directories are independent runs']}
    (evidence / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P4 · Experiment execution and recovery</title><style>body{{font:16px/1.6 system-ui,sans-serif;color:#24323c;background:#f3f6f8;margin:0}}main{{max-width:1100px;margin:auto;padding:40px 24px}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}h1,h2{{line-height:1.25}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #dce3e7;text-align:left}}a{{color:#165d91}}.badge{{color:#126457;font-weight:700}}code{{background:#edf1f4;padding:2px 5px}}li{{margin:6px 0}}</style><main>
<p class="badge">P4 ACCEPTANCE PASSED · Ubuntu · external evidence</p><h1>Paper tasks that show their work and recover safely</h1>
<p>All eight models completed real-data validation. Six presets are executable; ten fixed open-set trials were exercised. The full plan contains 4,096 fits and 4,344 evaluation-only tasks, but it was only dry-run.</p>
<section><h2>Coverage you can inspect</h2><p>{int(grid.sum())} distinct validation configurations completed across the main validation runs. Numbers are completed tasks, not full-paper coverage. Empty cells mean not run. Input coverage is the fixed 48-WAV subset, with 60 seconds of room noise for open-set trials.</p><img src="figures/completion-grid.svg" alt="Completed validation task counts for each of eight models and six paper presets"><p>{link(evidence/'full-plan.json','Full dry-run plan')} · {link(evidence/'manifest.json','Machine-readable acceptance')}</p></section>
<section><h2>Real slow-training visibility</h2><p>Heartbeats below come from actual 20 ms SVM fits and predictions: 136,324 training rows, 58,412 test rows, 49 features. They contain no synthetic waiting. CPU and memory show activity; they do not establish convergence or an ETA. SVM iteration limits remain 1,000 and warnings remain visible.</p><img src="figures/heartbeats.svg" alt="Recorded elapsed time versus process CPU and memory for actual slow fits"><p>{link(evidence/'logs/slow-fit.log','Complete slow-fit transcript')} · {link(evidence/'heartbeats.json','Parsed timestamped heartbeats')}</p></section>
<section><h2>Interrupted work resumes without duplicate results</h2><p>A deliberate SIGINT was sent after the second task entered real RBF-SVM training. The CLI exited 130. Native estimator code delayed delivery until control returned to Python; heartbeats continued during that interval. The completed GNB task remained byte-for-byte unchanged, including modification times; the SVM restarted in attempt 002. A concurrent runner was rejected with exit 1. A second resume added no attempts.</p><img src="figures/recovery-timeline.svg" alt="Measured start and finish times of successful, interrupted and restarted attempts"><img src="figures/recovery-equivalence.svg" alt="Exact result comparison between resumed and uninterrupted runs"><p>{link(evidence/'interrupted-state.json','State at interruption')} · {link(evidence/'recovery-checks.json','Recovery checks')} · {link(runs/'recovery/sweep.json','Final attempt ledger')}</p></section>
<section><h2>Eight real-data model fits</h2><p>These subset accuracies demonstrate execution and are not reproduced paper values. Times are single observations, not benchmark claims.</p><img src="figures/eight-model-fits.svg" alt="Measured fit durations for all eight models"><table><tr><th>Model</th><th>Fit seconds</th><th>Subset accuracy</th><th>Warnings</th></tr>{rows}</table></section>
<section><h2>Compatibility issue found and resolved</h2><p>The first modern QDA fit failed its covariance rank check. A constructor-only audit had missed changed fit behavior and covariance normalization. The explicit PaperQDA compatibility fit retains historical SVD scaling by n−1 and warning-only rank checks, with no added regularization. Invalid zero scales and nonfinite scores fail safely.</p><p>The upstream 1.0.2 fit method was replayed on the same 8,110 × 200 real feature matrix. Scores, scales and predictions match exactly (maximum score error 0). This is a method replay in the modern environment, not a recreated legacy environment.</p><p>{link(evidence/'logs/eight-model-smoke.log','Original QDA failure')} · {link(evidence/'qda-replay.json','Numerical replay record')} · {link(evidence/'model-audit.json','Updated constructor audit')}</p></section>
<section><h2>Checks and limits</h2><p>Python 3.14: {tests['314']} tests passed. Python 3.13: {tests['313']} tests passed. All eight real-data model predictions and metrics also match exactly across the two interpreters; 3.13 used a non-editable installed wheel. Tests cover full grid counts, numerical correctness, cache integrity, model references, failed attempts, concurrent-run rejection and restart. Changed source/selection and missing inputs return nonzero exits.</p><p>Full training, paper-ready figures, remote CI and publication remain later work. There are no estimator-level checkpoints. Separate output directories intentionally represent separate runs. Model references currently use absolute paths. A model's completed state does not imply convergence.</p><p>Development records also retain an initially over-strict timestamp assertion and a corrected validation-script cache filename; neither was hidden by removing failed logs.</p></section>
<section><h2>Provenance and next phase</h2><p>Source commit: <code>{escape(previous.get('source_commit', 'pending local phase commit'))}</code>. Validation ran on the modified worktree based on <code>b76f738</code>; task identities bind the actual Python source content, manifest hashes, runtime and resolved settings. The original public and private checkouts remain clean. All 257 WAV files retain their P3 inventory signatures and read-only permissions.</p><p>{link(evidence/'integrity.json','Hardware, runtime, lock and package integrity')} · {link(evidence/'acceptance.json','Acceptance criteria')} · {link(evidence/'source.patch','Phase source patch')}</p><p>P5 will generate paper figures and tables from the current completed attempt of each task. Failed attempts must remain excluded and missing grid cells must remain visibly missing. The full reproduction run remains P6.</p></section>
<section><h2>Commands and raw logs</h2><p>{link(evidence/'commands.jsonl','Command ledger with arguments, times and exits')}</p><ul>{logs}</ul></section></main></html>'''
    (evidence / 'report.html').write_text(html, encoding='utf-8')
    print(json.dumps({'status': 'passed', 'tests': tests, 'validation_tasks': int(grid.sum()), 'heartbeats': len(records), 'report': str(evidence / 'report.html')}, indent=2))


if __name__ == '__main__':
    main()
