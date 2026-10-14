"""Render the final P6 evidence report from verified full results and diagnostics."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from drone_fingerprint.storage import external_path, atomic_json

MODELS = ['QDA', 'LDA', 'LSVM', 'RBF-SVM', 'KNN', 'DT', 'RF', 'GNB']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    evidence = external_path(args.evidence)
    def read(name):
        return json.loads((evidence / name).read_text())
    checks = read('full-validation.json')
    replay = read('reproducibility.json')
    diagnostic = read('label-order-diagnostic/diagnostic.json')
    review = read('release-review.json')
    for item in (checks, replay, diagnostic, review):
        assert item['status'] == 'passed'
    for version in ('314', '313'):
        tree = ET.parse(evidence / f'final-tests{version}.xml')
        assert len(tree.findall('.//testcase')) == 155
        assert not tree.findall('.//failure') and not tree.findall('.//error')
    assert read('browser-closure.json')['status'] == 'passed'
    assert checks['tasks'] == 8440 and checks['missing_figure_points'] == 0
    assert len(replay['rows']) == 8 and all(row['predictions_equal'] and row['model_file_hash_equal'] for row in replay['rows'])
    assert len(diagnostic['rows']) == 22
    assert all(row['changed_predictions'] == 0 for row in diagnostic['rows'] if row['variant'] == 'unchanged')
    comparisons = read('paper-comparison.json')['rows']
    tables = read('gallery/tables.json')['tables']
    published = read('published-tables.json')['tables']
    manifest = read('gallery/manifest.json')
    assert manifest['full_dataset'] and manifest['full_grid'] and manifest['verified_results'] == 8440
    figures = evidence / 'closure-figures'
    figures.mkdir(exist_ok=True)
    def save(fig, name):
        fig.savefig(figures / (name + '.png'), dpi=160)
        fig.savefig(figures / (name + '.svg'))
        plt.close(fig)
    matrix = np.array([[next(row['difference_pp'] for row in comparisons if row['table'] == 'table2'
                            and row['model'] == model and row['metric'] == metric)
                        for metric in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')] for model in MODELS])
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    limit = np.max(np.abs(matrix))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    ax.set(xticks=range(4), xticklabels=['Accuracy', 'Macro precision', 'Macro recall', 'Macro F1'],
           yticks=range(8), yticklabels=MODELS, title='Table 2: corrected rerun minus published value (percentage points)')
    for i in range(8):
        for j in range(4):
            ax.text(j, i, f'{matrix[i,j]:+.3f}', ha='center', va='center',
                    color='white' if abs(matrix[i,j]) > limit*.6 else 'black')
    fig.colorbar(im, ax=ax, label='Percentage points')
    save(fig, 'table2-differences')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    x = np.arange(8)
    for ax, name, title in zip(axes, ('table4', 'table5'), ('Registered macro recall', 'Attack rejection')):
        pub = [published[name]['rows'][-1][m] for m in MODELS]
        new = [tables[name]['rows'][-2][m] for m in MODELS]
        ax.bar(x-.18, pub, .36, label='Published mean')
        ax.bar(x+.18, new, .36, label='Corrected macro mean')
        if name == 'table5':
            ax.plot(x, [tables['table5_pooled_comparison']['rows'][-2][m] for m in MODELS], 'k_', markersize=12, label='Corrected pooled mean')
        ax.set(xticks=x, xticklabels=MODELS, ylim=(0, 105), ylabel='Percent', title=title)
        ax.tick_params(axis='x', labelrotation=30)
        ax.legend(fontsize=8)
    save(fig, 'open-set-comparison')
    means = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    for ax, model in zip(axes, ('RBF-SVM', 'KNN')):
        rows = [row for row in diagnostic['rows'] if row['model'] == model and row['variant'] == 'background_last']
        assert len(rows) == 10 and {row['trial'] for row in rows} == set(range(1, 11))
        metrics = ('registered_macro_recall', 'attack_macro_rejection')
        means[model] = {key: float(np.mean([row['metrics'][key] for row in rows])) * 100
                        for key in (*metrics, 'attack_pooled_rejection')}
        original = [tables[name]['rows'][-2][model] for name in ('table4', 'table5')]
        printed = [published[name]['rows'][-1][model] for name in ('table4', 'table5')]
        axis = np.arange(2)
        ax.bar(axis-.25, original, .25, label='Frozen background-first')
        ax.bar(axis, [means[model][k] for k in metrics], .25, label='Diagnostic background-last')
        ax.bar(axis+.25, printed, .25, label='Published')
        ax.set(xticks=axis, xticklabels=['Registered recall', 'Attack rejection'], ylim=(0, 105), ylabel='Percent', title=model)
        ax.legend(fontsize=8)
    save(fig, 'label-order')
    atomic_json(evidence / 'label-order-summary.json', {'status': 'passed', 'ten_trial_diagnostic_means_percent': means})
    samples = [json.loads(line) for line in (evidence / 'resources.jsonl').read_text().splitlines()]
    samples = [row for row in samples if row['process']]
    start = datetime.fromisoformat(samples[0]['at'])
    hours = [(datetime.fromisoformat(row['at']) - start).total_seconds()/3600 for row in samples]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(hours, [row['process']['rss_bytes']/2**30 for row in samples])
    axes[0].axhline(checks['trainer_exit']['max_rss_kib']/2**20, linestyle='--', color='tab:orange', label='Kernel high-water mark')
    axes[0].set(xlabel='Hours since first resource sample', ylabel='GiB', title='Runner resident memory'); axes[0].legend()
    for field, label in [('read_bytes', 'OS-accounted reads'), ('write_bytes', 'OS-accounted writes')]:
        axes[1].plot(hours, [row['process'].get(field, 0)/2**30 for row in samples], label=label)
    axes[1].set(xlabel='Hours since first resource sample', ylabel='GiB, cumulative', title='Process storage I/O — not SSD wear'); axes[1].legend()
    save(fig, 'resources')
    durations = Counter()
    for row in checks['phase_seconds']:
        durations[row['model']] += row['seconds']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].barh(MODELS, [durations[m]/3600 for m in MODELS]); axes[0].set(xlabel='Sum of recorded phase hours', title='Actual full-run time by classifier')
    sizes = {'Models': checks['run_inventory']['.joblib']['bytes'], 'Predictions': checks['run_inventory']['.npz']['bytes'],
             'Run metadata': checks['run_inventory']['.json']['bytes'], 'Feature cache': sum(v['bytes'] for v in checks['cache_inventory'].values())}
    axes[1].barh(list(sizes), [v/2**30 for v in sizes.values()]); axes[1].set(xlabel='GiB retained', title='Storage by purpose')
    save(fig, 'costs')
    table_rows = ''.join(f"<tr><td>{row['model']}</td><td>{row['published_percent']:.2f}</td><td>{row['rerun_percent']:.4f}</td><td>{row['difference_pp']:+.4f}</td></tr>"
                         for row in comparisons if row['table']=='table2' and row['metric']=='accuracy')
    coverage = ''.join(f'<tr><td>{name}</td><td>{value["completed"]:,} / {value["expected"]:,}</td></tr>' for name, value in manifest['coverage'].items())
    warning_count = Counter()
    for row in checks['warnings']:
        warning_count[row['category']] += row['count']
    def figure(name, caption):
        return f'<figure><a href="closure-figures/{name}.svg"><img src="closure-figures/{name}.png" alt="{escape(caption)}"></a><figcaption>{escape(caption)}</figcaption></figure>'
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P6 full-run acceptance and scientific differences</title><style>
body{{font:16px/1.65 system-ui,sans-serif;color:#203538;background:#f3f6f3;max-width:1150px;margin:40px auto;padding:0 24px}}
a{{color:#117864}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:38px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.card,figure{{padding:20px;background:white;border:1px solid #dce5e2;border-radius:10px;margin:18px 0}}.card strong{{font-size:26px;display:block}}
img{{width:100%;height:auto}}figcaption{{color:#607477;font-size:14px}}.note{{border-left:4px solid #b57d21;background:#fff1dc;padding:18px}}
table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;border-bottom:1px solid #dce5e2;padding:8px}}code{{overflow-wrap:anywhere}}
@media(max-width:700px){{.cards{{grid-template-columns:1fr}}h1{{font-size:28px}}}}
</style></head><body><p>P6 · LOCAL ACCEPTANCE COMPLETE · PUBLICATION NOT PERFORMED</p>
<h1>Full results, reproducibility and differences from the paper</h1>
<p class="note">All tasks in the frozen corrected protocol completed and passed verification. This is not an exact
recovery of every historical score. The largest open-set difference is sensitive to background-class ordering;
the controlled experiment below preserves and explains that result instead of replacing the full-run outputs.</p>
<div class="cards"><div class="card"><strong>8,440 / 8,440</strong>4,096 fits + 4,344 evaluations; no failed or retried tasks</div>
<div class="card"><strong>{checks['wall_seconds']/3600:.2f} hours</strong>Observed full training and evaluation wall time</div>
<div class="card"><strong>{checks['trainer_exit']['max_rss_kib']/2**20:.2f} GiB</strong>Kernel-reported runner peak resident memory</div></div>
<p><a href="gallery/index.html">Open complete paper figure and table gallery</a> ·
<a href="paper-comparison.json">All 296 numerical comparisons</a> · <a href="PR_DRAFT.md">Prepared PR description</a></p>
<h2>Complete coverage and verified files</h2><table><tr><th>Experiment</th><th>Completed / expected</th></tr>{coverage}</table>
<p>All 8,352 figure points are present. The gallery exports six figure groups in PNG/SVG/PDF and five tables in
Markdown/LaTeX, with JSON plot data and metrics. The export recomputed metrics from saved predictions. Closure
verification independently checked result provenance and every model/prediction checksum, gallery checksums,
1,934 cache-array checksums and all 257 original read-only WAV SHA-256 values. There are no new WAV files.</p>
<p>Models are saved once per clean fit; 4,344 evaluations reference existing models. Full-run artifacts occupy
{sum(v['bytes'] for v in checks['run_inventory'].values())/2**30:.2f} GiB in {sum(v['files'] for v in checks['run_inventory'].values()):,} files;
cache occupies {sum(v['bytes'] for v in checks['cache_inventory'].values())/2**30:.2f} GiB in {sum(v['files'] for v in checks['cache_inventory'].values()):,} files.
These are per-configuration arrays/results, not per-frame audio or feature files.</p>
<h2>Closed-set Table 2</h2><table><tr><th>Classifier</th><th>Published accuracy (%)</th><th>Rerun (%)</th><th>Difference (pp)</th></tr>{table_rows}</table>
{figure('table2-differences', 'All four Table 2 metrics: maximum absolute difference is '+f'{limit:.4f}'+' percentage points. Printed values have two decimals.')}
<p>QDA, LDA and LSVM match the printed four metrics at two-decimal precision. RF, DT and RBF-SVM differ slightly.
Seeds, input order, documented numerical corrections and library migration differ from the historical execution;
this comparison does not isolate those factors for each estimator.</p>
<h2>Open-set Tables 4 and 5</h2>
{figure('open-set-comparison', 'Ten-trial published and corrected means. Table 5 macro and pooled rejection remain separate; aggregation alone does not explain the RBF-SVM gap.')}
<p>QDA registered recall is {tables['table4']['rows'][-2]['QDA']:.4f}% (published 94.19%).
RBF-SVM registered recall is {tables['table4']['rows'][-2]['RBF-SVM']:.4f}% (published 13.05%);
its macro attack rejection is {tables['table5']['rows'][-2]['RBF-SVM']:.4f}% (published 98.10%).</p>
<h2>Controlled diagnosis: background-class ordering</h2>
<p>The legacy PKL trainer maps drones to 0–23 and background to 24, placing background last. The frozen modern
protocol uses drone identities 1–24 and background 0, placing background first. The diagnostic changes only
background training label 0 to 25 and maps predictions back to 0. Features, row order, seed, classifier settings
and evaluation metrics are identical. RBF-SVM and KNN were tested on all ten full-data partitions.
An unchanged trial-1 refit for each model exactly reproduces its saved predictions.</p>
{figure('label-order', 'Background-first is the primary frozen run. Background-last is separate diagnostic evidence, not a replacement result or an old-environment replay.')}
<p>With background last, RBF-SVM mean registered recall is {means['RBF-SVM']['registered_macro_recall']:.4f}% and
macro attack rejection is {means['RBF-SVM']['attack_macro_rejection']:.4f}%; KNN means are
{means['KNN']['registered_macro_recall']:.4f}% and {means['KNN']['attack_macro_rejection']:.4f}%.
The experiment establishes label-order sensitivity under these configured procedures. It does not prove that
class ordering explains every remaining historical difference. The capped SVM solver still emits convergence
warnings; a completed fit is not proof of convergence. No iteration limit or frozen label convention was changed.</p>
<h2>Reproducibility and figure review</h2>
<p>All eight full-data Table 2 classifiers were fitted again in the same locked runtime. Predictions, metrics,
confusion matrices, scientific settings and model-file hashes match exactly. This is a bounded reproducibility
check, not a second execution of all 8,440 tasks or a cross-machine guarantee. Both supported Python versions passed 155 tests, including a KNN tie-order fixture. Earlier Docker
acceptance remains applicable because runtime source and lock are unchanged.</p>
<p>All six generated figure groups were visually reviewed for labels, units, legends and coverage. Figure 2 retains
the longer-frame trend and Figure 5 the broad SNR trend. Figure 3/4 derivative panels materially differ, especially
KNN: the modern definitions use true segment-local derivatives and final concatenated normalization. P2 documents
those corrections; this visual review is not a controlled attribution of each full-grid difference. Original curve
CSV files were not retained, so no numerical pointwise paper-curve error is claimed.</p>
<h2>Measured resources and retained warnings</h2>
{figure('resources', 'Minute-level measurements during the original full run. Kernel peak RSS is recorded separately; no physical SSD lifetime inference is made.')}
{figure('costs', 'Observed phase time and retained storage. Diagnostic/refit costs are separate from the original full-run measurements.')}
<p>The run retained {warning_count['ConvergenceWarning']} convergence warnings and {warning_count['FutureWarning']}
SVC parameter-deprecation warnings. OS-accounted output blocks total {checks['trainer_exit']['output_blocks']*512/2**30:.2f} GiB.
This exceeds retained artifact size because writing metadata/state and publishing outputs adds traffic. It is not
physical NAND writes, a controlled I/O benchmark or evidence about the earlier SSD failure. The full ledger is
atomically rewritten at state transitions; a future larger sweep may benefit from a journal, but no unmeasured
storage redesign was introduced during release validation.</p>
<h2>Completed browser view</h2><a href="completed-full-plan.png"><img src="completed-full-plan.png" alt="Browser confirms 8440 completed tasks with no running process"></a>
<h2>Release review and scope</h2>
<p>The maintained path is plain configuration → planner → workflow → bounded features → estimator → saved results.
Historical wrappers remain explicitly labelled reference code; licences and attribution are retained. Runtime code,
model definitions and the dependency lock match the validated training revision. Release changes add evidence tools
and scientific documentation. No generated data, model, log, local PDF or private source is included in Git.</p>
<p>The localhost monitor remains available to inspect completed status. This report and final gallery work offline.
Remote CI, push and publication were not performed. Ubuntu is the supported platform. The within-recording split
does not establish unseen-recording or unseen-room generalization. LaTeX tabular fragments were exported, not compiled.</p>
<ul><li><a href="full-validation.json">Integrity, task timings, warning and storage inventory</a></li>
<li><a href="reproducibility.json">Exact full-data eight-classifier repeat</a></li>
<li><a href="label-order-diagnostic/diagnostic.json">All controlled diagnostic runs and warnings</a></li>
<li><a href="release-review.json">Source, documentation and release review</a> · <a href="source-manifest.json">Source revisions</a></li>
<li><a href="commands.jsonl">Commands and exits</a> · <a href="execution.jsonl">Original execution and kernel usage</a></li>
<li><a href="resources.jsonl">Raw resource samples</a> · <a href="final-tests314.xml">155 Python 3.14 tests</a> · <a href="final-tests313.xml">155 Python 3.13 tests</a></li>
<li><a href="launch-report.html">Preserved launch-time report</a> · <a href="published-tables.json">Printed reference values</a></li></ul>
<footer>Generated {datetime.now(timezone.utc).isoformat()}. Full run uses revision 62f6d2e. Local completion includes
scientific differences and limitations; publication is a separate action.</footer></body></html>'''
    for target in re.findall(r'(?:href|src)="([^"]+)"', html):
        assert (evidence / target).is_file(), target
    (evidence / 'report.html').write_text(html)
    atomic_json(evidence / 'report-status.json', {'p6_local_acceptance_complete': True, 'tasks': 8440,
                'exact_historical_reproduction': False, 'publication_performed': False,
                'label_order_diagnostic_means_percent': means, 'table2_max_difference_pp': float(limit)})
    print(f'Final P6 report: {evidence / "report.html"}')


if __name__ == '__main__':
    main()
