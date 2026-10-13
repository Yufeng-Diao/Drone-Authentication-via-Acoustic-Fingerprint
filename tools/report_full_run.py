"""Render an external P6 status snapshot; live updates use drone-auth monitor."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from drone_fingerprint.storage import external_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    evidence = external_path(args.evidence)

    def read(name):
        return json.loads((evidence / name).read_text())

    session, preflight = read('session.json'), read('preflight.json')
    run = Path(session['run'])
    plan = json.loads((run / 'plan.json').read_text())
    state = json.loads((run / 'sweep.json').read_text())
    assert preflight['status'] == 'passed'
    assert plan['counts'] == {'fits': 4096, 'evaluations': 4344, 'tasks': 8440}
    assert set(state['tasks']) == {task['id'] for task in plan['tasks']}
    tests = ET.parse(evidence / 'tests.xml')
    assert len(tests.findall('.//testcase')) == 154
    assert not tests.findall('.//failure') and not tests.findall('.//error')
    assert read('browser-proof.json')['status'] == 'passed'
    models = ['QDA', 'LDA', 'LSVM', 'RBF-SVM', 'KNN', 'DT', 'RF', 'GNB']
    presets = ['figure2', 'figure3', 'figure4', 'table2', 'figure5', 'tables4_5']
    completed = np.zeros((8, 6), dtype=int)
    expected = np.zeros_like(completed)
    for task in plan['tasks']:
        row, col = models.index(task['settings']['model']), presets.index(task['preset'])
        expected[row, col] += 1
        completed[row, col] += state['tasks'][task['id']]['status'] == 'completed'
    counts = Counter(entry['status'] for entry in state['tasks'].values())
    fig, ax = plt.subplots(figsize=(10, 4.8), layout='constrained')
    ax.imshow(completed / expected, vmin=0, vmax=1, cmap='Greens', aspect='auto')
    ax.set(xticks=range(6), xticklabels=presets, yticks=range(8), yticklabels=models,
           title='Completed / planned tasks — status snapshot, not runtime progress')
    for row in range(8):
        for col in range(6):
            ax.text(col, row, f'{completed[row, col]}/{expected[row, col]}', ha='center', va='center')
    fig.savefig(evidence / 'coverage.png', dpi=150)
    fig.savefig(evidence / 'coverage.svg')
    plt.close(fig)
    samples = [json.loads(line) for line in (evidence / 'resources.jsonl').read_text().splitlines()]
    observed = [row for row in samples if row['process']]
    assert observed
    start = datetime.fromisoformat(samples[0]['at'])
    minutes = [(datetime.fromisoformat(row['at']) - start).total_seconds() / 60 for row in observed]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
    axes[0].plot(minutes, [row['process']['rss_bytes'] / 2**20 for row in observed], marker='.')
    axes[0].set(xlabel='Minutes since first sample', ylabel='Runner RSS (MiB)', title='Observed memory, not final peak')
    axes[1].plot(minutes, [row['process']['cpu_seconds'] for row in observed], marker='.')
    axes[1].set(xlabel='Minutes since first sample', ylabel='Cumulative CPU seconds', title='Actual runner activity')
    fig.savefig(evidence / 'resources.png', dpi=150)
    fig.savefig(evidence / 'resources.svg')
    plt.close(fig)
    timestamp = datetime.now(timezone.utc).isoformat()
    summary = {'observed_at': timestamp, 'ledger_status': state['status'], 'counts': dict(counts),
               'p6_complete': False, 'training_revision': session['commit'], 'resource_samples': len(samples),
               'remaining_gates': ['Full training and verified complete gallery', 'Paper comparison and reproducibility review',
                                   'Final release review; remote CI and publication when authorized']}
    (evidence / 'report-status.json').write_text(json.dumps(summary, indent=2) + '\n')
    (evidence / 'report.html').write_text(f'''<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P6 full experiment status</title><style>
body{{font:16px/1.65 system-ui,sans-serif;color:#203538;max-width:1100px;margin:36px auto;padding:0 24px;background:#f3f6f3}}
img{{max-width:100%;background:white;border:1px solid #dce5e2;border-radius:10px;margin:12px 0}}
a{{color:#117864}}code{{overflow-wrap:anywhere}}.note{{padding:18px;background:#fff1dc;border-left:4px solid #b57d21}}
</style></head><body><p>P6 · FULL PUBLIC-DATA RUN · IN PROGRESS</p><h1>Complete paper experiment execution</h1>
<p class="note">Snapshot at {timestamp}. P6 is not complete. This offline report does not update itself;
use the live monitor on forwarded port 8765 for current progress. No full reproduction claim.</p>
<p>Ledger status: <strong>{escape(state['status'])}</strong>. Completed: <strong>{counts['completed']:,} / 8,440</strong>.
Running: {counts['running']}; failed: {counts['failed']}; interrupted: {counts['interrupted']}; pending: {counts['pending']}.</p>
<img src="coverage.png" alt="Completed versus planned coverage for eight classifiers and six experiment presets">
<h2>Execution and resources</h2><p>4,096 fits and 4,344 evaluations, sequentially with one numerical thread.
Full selection: 240 drone recordings and the complete room-noise interval for open-set training.
The non-editable Python 3.14.8 installation matches revision <code>{session['commit']}</code>.
All 154 regression tests pass in that environment. Independent browser and training processes run detached.
Closing the browser or SSH connection leaves training running.</p>
<p>The feature payload estimate is {preflight['estimated_cache_data_bytes']/2**30:.2f} GiB across
{preflight['unique_base_arrays']} base arrays and {preflight['unique_selected_arrays']} selected arrays.
Initial free disk was {preflight['disk_free_bytes']/2**30:.2f} GiB. Models, predictions, temporary files and
metadata require additional space; this estimate is not a peak estimator-memory guarantee.</p>
<img src="resources.png" alt="Observed runner memory and cumulative CPU time from external resource samples">
<p>Resource samples are recorded once per minute. Terminal phase heartbeats appear every ten seconds;
the webpage polls every three seconds. Process I/O counters do not measure physical SSD wear.
No wall-time cutoff or silent model-parameter change is applied. Final plotting runs automatically only if the full trainer exits successfully.</p>
<h2>Actual browser evidence</h2><p>This capture comes from the full-data run, not the earlier validation subset.</p>
<a href="live-full-plan.png"><img src="live-full-plan.png" alt="Live browser monitoring of the full 8440-task run"></a>
<h2>Scientific comparison prepared</h2><p>The printed Tables 2, 4 and 5 are transcribed from the author's PDF,
including all ten trials and printed averages. Rounding consistency was checked. New measurements remain
separate from published values; Table 5 macro and pooled rejection must be distinguished. Original figure
CSV files are unavailable, so no pointwise reference curves are invented.</p>
<ul><li><a href="preflight.json">Source/runtime/data and resource preflight</a></li>
<li><a href="full-plan.json">Complete resolved plan</a> · <a href="published-tables.json">Published table reference</a></li>
<li><a href="tests.xml">154-test installed-environment result</a> · <a href="browser-proof.json">Browser checks</a></li>
<li><a href="resources.jsonl">Resource timeline</a> · <a href="execution.jsonl">Trainer execution record</a></li>
<li><a href="commands.jsonl">Acceptance commands</a> · <a href="logs/trainer.log">Trainer transcript</a></li>
<li><a href="../../../Drone-Authentication-refactor/docs/RELEASE_STATUS.md">Release criteria</a> ·
<a href="../../../Drone-Authentication-refactor/docs/LIVE_MONITOR.md">Monitoring and SSH instructions</a></li></ul>
<h2>Remaining gates</h2><p>Full-grid completion, verified final figures/tables, measured artifact storage,
scientific differences, reproducibility checks and final release review remain open. No push or remote CI
has occurred. Failures must be inspected and retained before an explicit checked resume; they are not skipped.
The native monitor remains available for the author. The temporary screenshot browser was stopped.</p>
</body></html>''')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
