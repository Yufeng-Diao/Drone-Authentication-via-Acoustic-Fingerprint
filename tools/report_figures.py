"""Validate the external P5 gallery and tutorial records, then render its report."""

import argparse
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

from drone_fingerprint.config import MODELS
from drone_fingerprint.data import sha256
from drone_fingerprint.results import PRESETS
from drone_fingerprint.storage import external_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    args = parser.parse_args()
    evidence, runs = external_path(args.evidence), external_path(args.runs)
    gallery = evidence / 'gallery'
    manifest = json.loads((gallery / 'manifest.json').read_text())
    metrics = json.loads((gallery / 'metrics.json').read_text())['results']
    series = json.loads((gallery / 'plot-data.json').read_text())['series']
    tables = json.loads((gallery / 'tables.json').read_text())['tables']
    assert manifest['status'] == 'passed' and not manifest['full_dataset'] and not manifest['full_grid']
    assert manifest['verified_results'] == len(metrics) == 62
    assert manifest['duplicate_results_merged'] == 1 and len(series) == 160
    for name, record in manifest['artifacts'].items():
        assert (gallery / name).stat().st_size == record['bytes']
        assert sha256(gallery / name) == record['sha256']
    for filename in ('results.py', 'plotting.py'):
        assert sha256(Path(__file__).resolve().parents[1] / 'src/drone_fingerprint' / filename) == manifest['plot_source_sha256'][filename]
    grid = np.zeros((len(MODELS), len(PRESETS)), dtype=int)
    for result in metrics:
        grid[MODELS.index(result['model']), PRESETS.index(result['experiment'])] += 1
    assert grid.sum() == 62 and grid[:, PRESETS.index('table2')].tolist() == [1] * 8
    for preset in ('figure3', 'figure4'):
        panel = [s for s in series if s['preset'] == preset and s['model'] == 'GNB']
        assert [(s['blocks'], s['portion']) for s in panel] == [(b, p) for b in (1, 2, 3) for p in (1, 2, 3)]
        assert all(sum(v is not None for v in s['y']) == 2 for s in panel)
    assert sum(v is not None for s in series for v in s['y']) == 44
    assert tables['table4']['rows'][-1]['GNB'] == '10/10'
    assert tables['table5']['rows'][-2]['GNB'] is not None
    assert all(tables['table5']['rows'][-2][m] is None for m in MODELS if m != 'GNB')
    integrity = json.loads((evidence / 'integrity.json').read_text())
    plot_integrity = json.loads((evidence / 'plot-input-integrity.json').read_text())
    assert integrity['raw_wavs'] == 257 and integrity['raw_signatures_unchanged']
    assert plot_integrity['sha256_size_mtime_unchanged']
    tests = {}
    for version in ('314', '313'):
        tree = ET.parse(evidence / f'tests-py{version}.xml')
        assert not tree.findall('.//failure') and not tree.findall('.//error') and not tree.findall('.//skipped')
        tests[version] = len(tree.findall('.//testcase'))
        assert tests[version] == 143
    commands = [json.loads(line) for line in (evidence / 'commands.jsonl').read_text().splitlines()]
    finished = [c for c in commands if c['event'] == 'finished']
    exits = {c['name']: c['exit_code'] for c in finished}
    required = ('setup-py314', 'tests-py314', 'gallery-validation-fits', 'final-gallery',
                'tutorial-synthetic', 'tutorial-public', 'example-features', 'example-plot',
                'tutorial-full-plan', 'setup-py313', 'tests-py313', 'wheel-synthetic-py313',
                'integrity-final', 'build', 'package-check')
    assert all(exits[name] == 0 for name in required)
    assert json.loads((evidence / 'full-plan.json').read_text())['counts'] == {'fits': 4096, 'evaluations': 4344, 'tasks': 8440}
    sweep = json.loads((runs / 'gallery-validation/sweep.json').read_text())
    assert sweep['status'] == 'completed' and len(sweep['tasks']) == 53
    comparison = json.loads((runs / 'example-features/comparison.json').read_text())
    assert [row['columns'] for row in comparison] == [49, 98, 147]
    container = json.loads((evidence / 'container-status.json').read_text())
    assert container['status'] == 'pending_user_deferred'
    for html in (gallery / 'index.html',):
        for target in re.findall(r'(?:href|src)="([^"]+)"', html.read_text()):
            assert (html.parent / target).is_file(), target
    figures = evidence / 'figures'
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    ax.imshow(grid, cmap='Blues', vmin=0, vmax=18)
    ax.set(xticks=range(6), xticklabels=PRESETS, yticks=range(8), yticklabels=MODELS,
           title='P5 gallery: verified validation configurations\nCounts are subset observations, not full-paper coverage')
    for row in range(8):
        for col in range(6):
            ax.text(col, row, str(grid[row, col]) if grid[row, col] else '—', ha='center', va='center',
                    color='white' if grid[row, col] > 9 else '#17354a')
    for suffix in ('png', 'svg'):
        fig.savefig(figures / f'coverage.{suffix}', dpi=150)
    plt.close(fig)
    previous = json.loads((evidence / 'manifest.json').read_text()) if (evidence / 'manifest.json').exists() else {}
    acceptance = {**previous, 'phase': 'P5', 'status': 'passed_with_deferred_container_smoke',
                  'generated_at': datetime.now(timezone.utc).isoformat(), 'tests': tests,
                  'verified_gallery_results': 62, 'new_gallery_tasks': {'fits': 32, 'evaluations': 21},
                  'gallery_exports': len(manifest['artifacts']), 'plotted_points': 44,
                  'missing_plot_points': sum(v is None for s in series for v in s['y']),
                  'plot_inputs_unchanged': plot_integrity, 'container': container,
                  'required_commands_passed': list(required), 'coverage': manifest['coverage'],
                  'limits': ['Container execution explicitly deferred by owner', 'No full-grid reproduction',
                             'No remote CI or push', 'No LaTeX compilation; exported tabular fragments only']}
    (evidence / 'manifest.json').write_text(json.dumps(acceptance, indent=2) + '\n')
    checklist = ''.join(f'<tr><td>{escape(name)}</td><td>Passed</td><td><a href="logs/{name}.log">Log</a></td></tr>' for name in required)
    logs = ''.join(f'<li><a href="{escape(c["log"])}">{escape(c["name"])} · exit {c["exit_code"]}</a></li>' for c in finished)
    images = ''.join(f'<section><h2>{title}</h2><p>{caption}</p><a href="gallery/index.html"><img src="gallery/{name}.svg" alt="{title}"></a></section>' for name, title, caption in (
        ('figure3', 'Nine panels with explicit missing coverage', 'GNB at 26 and 31 filters exercises all block/fraction combinations. Empty ranges remain empty. Figure 4 uses the same layout at 0 dB AWGN.'),
        ('figure5', 'SNR units and missing-point gaps', 'Three saved GNB evaluations use -8, 0 and 15 dB. No line bridges unmeasured quarter-decibel grid points.'),
        ('diagnostic_confusion', 'Eight classifiers, one labelled diagnostic', 'Table 2 subset confusion matrices use the same 0–1 row-recall scale. Historical SVM iteration limits and recorded convergence warnings are retained.'),
        ('diagnostic_open_recall', 'Open-set metrics retain original identities', 'Registered recall and attack rejection remain separate. The gallery exports Tables 4/5 with all ten GNB trials; missing models have no fabricated trial mean.')))
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P5 · Figures and researcher documentation</title><style>body{{font:16px/1.6 system-ui,sans-serif;color:#243847;background:#f2f5f7;margin:0}}main{{max-width:1180px;margin:auto;padding:40px 24px}}section{{background:white;padding:24px;margin:24px 0;border-radius:12px}}img{{width:100%;height:auto}}a{{color:#175e96}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #dce4e9}}.status{{color:#806019;font-weight:bold}}code{{background:#edf2f5;padding:3px}}</style>
<main><p class="status">P5 COMPLETE UNDER THE AGREED SCOPE · Container execution pending</p>
<h1>From saved results to inspectable research figures</h1><p>Paper Figures 2–5, Tables 2–5, diagnostics, English tutorials and two exercised extension examples are implemented. The CPU Dockerfile is prepared; its runtime smoke test was explicitly deferred by the owner.</p>
<p><a href="gallery/index.html">Open the complete figure and table gallery</a> · <a href="manifest.json">Acceptance record</a> · <a href="commands.jsonl">Command ledger</a></p>
<section><h2>What was actually executed</h2><p>53 new gallery tasks completed: 32 GNB fits and 21 evaluations reusing clean models. Combining these with immutable P4 eight-model and slow-fit results yields 62 distinct configurations; one identical duplicate is merged with both sources retained. Plotting verified artifact checksums and recomputed metrics from prediction arrays. It did not load audio/features, deserialize models or train.</p><img src="figures/coverage.svg" alt="Counts of verified configurations by model and paper preset"><p>Both Python 3.14.8 and a clean non-editable Python 3.13.16 install pass 143 tests. The full 8,440-task plan was only dry-run. All displayed scientific measurements are validation-subset results, not reproduced paper values.</p></section>
{images}
<section><h2>Exports researchers can reuse</h2><p>The gallery provides six figure groups × PNG/SVG/PDF, five table groups × Markdown/LaTeX, plus plot data, metrics, table JSON and provenance. Table 3 records frozen partitions. The extra Table 5 comparison is explicitly pooled rejection, separate from the primary per-identity macro definition. Percentages and units are explicit; missing cells are em dashes. Plot data contains nulls for missing points, never guessed values.</p><p><a href="gallery/table2.md">Table 2</a> · <a href="gallery/table3.md">Table 3</a> · <a href="gallery/table4.md">Table 4</a> · <a href="gallery/table5.md">Table 5</a> · <a href="gallery/table5_pooled_comparison.md">Pooled comparison</a> · <a href="gallery/plot-data.json">Plot-ready data</a> · <a href="gallery/metrics.json">Metrics and result sources</a></p></section>
<section><h2>Executed tutorial checklist</h2><p>A fresh primary environment and independent compatibility wheel were exercised. Synthetic and public-data quickstarts succeeded. The feature example produced 49, 98 and 147 columns using the shared workflow. The figure example read saved JSON only.</p><table><tr><th>Check</th><th>Outcome</th><th>Evidence</th></tr>{checklist}<tr><td>Docker build and container smoke</td><td>Pending, explicitly deferred</td><td><a href="container-status.json">Runtime probe</a></td></tr></table><p><a href="custom-figure/custom-series.svg">Customized series</a> · <a href="tests-py314.xml">Python 3.14 JUnit</a> · <a href="tests-py313.xml">Python 3.13 JUnit</a></p></section>
<section><h2>Integrity, readability and remaining limits</h2><p>All 257 original WAVs retain their P3 inventory filesystem signatures and read-only permissions. Both original repositories remain clean. Plot input files retain hashes, sizes and mtimes. New model dependencies use relative paths, tested after moving the complete run tree; existing P4 absolute references remain readable. The dependency lock is unchanged and no package was added.</p><p>The implementation separates verified result collection from presentation using ordinary dictionaries and direct functions. English guides map each research change to its module. README now leads with the maintained Ubuntu workflow; old instructions remain clearly marked historical. Local documentation links were checked. Figures were visually inspected for units, legends, nine-panel order and clipping.</p><p>The Docker base digest was resolved from the registry, but no image was built or run. Python tests cannot prove container behavior. Full-paper sweeps, remote CI and publication remain P6 work. LaTeX fragments were exported but not compiled. Development checks initially failed: a test counted the explanatory em dash as a table cell, and the external integrity script used the wrong inventory key. The report also initially expected pure JSON from dry-run stdout; its startup banner is now retained in a separate raw transcript and the JSON payload is parsed explicitly. These checks were corrected; original logs remain available.</p><p>Source revision: <code>{escape(previous.get('source_commit', 'pending local phase commit'))}</code>. Validation used modified source based on P4 <code>46d3ffc</code>. <a href="source.patch">Source patch</a> · <a href="integrity.json">Input and documentation integrity</a> · <a href="package-check.json">Distribution checks</a> · <a href="gallery/manifest.json">Gallery provenance and checksums</a></p></section>
<section><h2>Commands and retained logs</h2><ul>{logs}</ul></section></main></html>'''
    (evidence / 'report.html').write_text(html, encoding='utf-8')
    for target in re.findall(r'(?:href|src)="([^"]+)"', html):
        assert (evidence / target).is_file(), target
    print(json.dumps({'status': acceptance['status'], 'tests': tests, 'gallery_results': 62,
                      'report': str(evidence / 'report.html')}, indent=2))


if __name__ == '__main__':
    main()
