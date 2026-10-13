"""Headless paper figures and tables from verified saved results only."""

from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .config import MODELS
from .data import sha256
from .results import PRESETS, collect_results, figure_series, paper_tables
from .storage import atomic_json, create_output

# Colors follow the author's private paper plots; styles are keyed by model name.
COLORS = ('#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#b28a00', '#a6761d', '#666666')
LINES = ('-', '--', '-.', ':', '-', '--', '-.', ':')
MARKERS = ('o', 's', '^', 'v', 'D', 'x', '>', '*')
STYLES = {m: {'color': c, 'linestyle': l, 'marker': marker, 'markersize': 3, 'linewidth': 1.2}
          for m, c, l, marker in zip(MODELS, COLORS, LINES, MARKERS)}


def preview_label(bundle, preset):
    count = sum(r['provenance']['experiment'] == preset for r in bundle['results'].values())
    total = bundle['expected_counts'][preset]
    if not bundle['full_dataset']:
        return f'INCOMPLETE PREVIEW — validation subset; {count}/{total} grid points'
    return f'Corrected full-data rerun; {count}/{total} grid points' if count == total else f'INCOMPLETE PREVIEW — {count}/{total} grid points'


def draw_paper_figure(bundle, preset):
    """Figures 3/4: feature blocks by row, coefficient fractions by column."""
    series = figure_series(bundle, preset)
    panels = [(b, p) for b in (1, 2, 3) for p in (1, 2, 3)] if preset in ('figure3', 'figure4') else [(1, 3)]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), squeeze=False) if len(panels) == 9 else plt.subplots(1, 1, figsize=(10, 5.6), squeeze=False)
    blocks_names = ('MFCC', 'MFCC + delta', 'MFCC + delta + delta-delta')
    portions = ('one third', 'two thirds', 'all')
    for index, ((blocks, portion), ax) in enumerate(zip(panels, axes.flat)):
        panel = [s for s in series if (s['blocks'], s['portion']) == (blocks, portion)]
        for line in panel:
            y = [np.nan if v is None else v for v in line['y']]
            ax.plot(line['x'], y, label=line['model'], **STYLES[line['model']])
        ax.set(xlabel={'figure2': 'Frame length (ms)', 'figure5': 'SNR (dB)'}.get(preset, 'Number of filters'),
               ylabel='Accuracy (%)', ylim=(0, 100), xlim=(panel[0]['x'][0], panel[0]['x'][-1]))
        if len(panels) == 9:
            ax.set_title(f"({chr(97+index)}) {blocks_names[blocks-1]} · {portions[portion-1]}", fontsize=10)
        if not any(v is not None for s in panel for v in s['y']):
            ax.text(0.5, 0.5, 'No completed results', transform=ax.transAxes, ha='center', color='#697781')
        ax.grid(alpha=0.22)
    title = {'figure2': 'Figure 2 · Frame length', 'figure3': 'Figure 3 · Filter count, clean test audio',
             'figure4': 'Figure 4 · Filter count, 0 dB test AWGN', 'figure5': 'Figure 5 · Test AWGN'}[preset]
    fig.suptitle(title + '\n' + preview_label(bundle, preset), fontsize=12)
    handles = [Line2D([], [], label=model, **STYLES[model]) for model in MODELS]
    fig.legend(handles=handles, loc='lower center', ncol=8, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    return fig, series


def diagnostics(bundle):
    """Confusion matrices and open-set per-identity recall supplement the paper plots."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), layout='constrained')
    image = None
    for model, ax in zip(MODELS, axes.flat):
        result = next((r for r in bundle['results'].values() if r['provenance']['experiment'] == 'table2'
                       and r['configuration']['model'] == model), None)
        ax.set_title(model)
        if result is None:
            ax.text(0.5, 0.5, 'No completed result', ha='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue
        counts = np.asarray(result['confusion_matrix'])
        normalized = np.divide(counts, counts.sum(axis=1, keepdims=True), out=np.zeros_like(counts, dtype=float), where=counts.sum(axis=1, keepdims=True) != 0)
        image = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
        ticks = [0, 5, 11, 17, 23]
        ax.set(xticks=ticks, xticklabels=[result['labels'][i] for i in ticks],
               yticks=ticks, yticklabels=[result['labels'][i] for i in ticks], xlabel='Predicted identity', ylabel='True identity')
    if image is not None:
        fig.colorbar(image, ax=list(axes.flat), label='Recall within true identity', shrink=0.65)
    fig.suptitle('Diagnostic · Row-normalized confusion matrices\n' + preview_label(bundle, 'table2'), fontsize=12)
    figures = {'diagnostic_confusion': fig}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    candidates = sorted((r for r in bundle['results'].values() if r['provenance']['experiment'] == 'tables4_5'),
                        key=lambda r: (r['configuration']['partition']['trial'], MODELS.index(r['configuration']['model'])))
    if candidates:
        result = candidates[0]
        for ax, field, title in zip(axes, ('registered_recall_by_identity', 'attack_rejection_by_identity'), ('Registered recall', 'Attack rejection')):
            values = result['metrics'][field]
            ax.bar(list(values), [100*v for v in values.values()], color=STYLES[result['configuration']['model']]['color'])
            ax.set(xlabel='Original identity', ylabel='Recall (%)', ylim=(0, 100), title=title)
        title = f"Trial {result['configuration']['partition']['trial']} · {result['configuration']['model']}"
    else:
        title = 'No completed open-set results'
        for ax in axes:
            ax.text(0.5, 0.5, title, transform=ax.transAxes, ha='center')
            ax.set_axis_off()
    fig.suptitle('Diagnostic · ' + title + '\n' + preview_label(bundle, 'tables4_5'), fontsize=12)
    figures['diagnostic_open_recall'] = fig
    return figures


def cell(value):
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:.2f}'
    if isinstance(value, list):
        return ', '.join(map(str, value))
    return str(value)


def export_tables(tables, output, scope):
    """Write machine-readable tables plus standalone Markdown and LaTeX tabulars."""
    for name, table in tables.items():
        columns = table['columns']
        label = 'Frozen Table 3 protocol; identity numbers, not measured results.' if name == 'table3' else scope + ' Values are percentages; — means missing.'
        rows = [[cell(row[key]) for key in columns] for row in table['rows']]
        markdown = f'# {name.replace("_", " ").title()}\n\n{label}\n\n'
        markdown += '| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join('---' for _ in columns) + ' |\n'
        markdown += '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows) + '\n'
        (output / f'{name}.md').write_text(markdown, encoding='utf-8')
        latex = '% ' + label.replace('—', '--') + '\n\\begin{tabular}{' + 'l' * len(columns) + '}\n\\hline\n'
        for row in [columns, *rows]:
            latex += ' & '.join(v.replace('_', r'\_').replace('%', r'\%').replace('—', '--') for v in row) + r' \\' + '\n'
        latex += '\\hline\n\\end{tabular}\n'
        (output / f'{name}.tex').write_text(latex, encoding='utf-8')
    atomic_json(output / 'tables.json', {'scope': scope, 'units': 'percent except Table 3 identity IDs', 'tables': tables})


def render_results(runs: list[Path], paper_dir: Path, output: Path) -> dict:
    """Create a fresh external gallery, with plot-ready data and provenance."""
    bundle = collect_results(runs, paper_dir)
    if any(output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve()) for root in runs):
        raise ValueError('Figure output and input sweep directories must be separate')
    output = create_output(output)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    series = []
    for preset in ('figure2', 'figure3', 'figure4', 'figure5'):
        fig, rows = draw_paper_figure(bundle, preset)
        series.extend(rows)
        for suffix in ('png', 'svg', 'pdf'):
            fig.savefig(output / f'{preset}.{suffix}', dpi=160, bbox_inches='tight')
        plt.close(fig)
    for name, fig in diagnostics(bundle).items():
        for suffix in ('png', 'svg', 'pdf'):
            fig.savefig(output / f'{name}.{suffix}', dpi=160, bbox_inches='tight')
        plt.close(fig)
    tables = paper_tables(bundle)
    complete = bundle['full_dataset'] and all(sum(r['provenance']['experiment'] == p for r in bundle['results'].values()) == bundle['expected_counts'][p] for p in PRESETS)
    scope = 'Corrected full-protocol rerun.' if complete else 'INCOMPLETE PREVIEW — results are a subset of the full paper protocol.'
    export_tables(tables, output, scope)
    atomic_json(output / 'plot-data.json', {'scope': scope, 'series': series})
    metrics = [{'config_id': key, 'experiment': r['provenance']['experiment'], 'model': r['configuration']['model'],
                'configuration': r['configuration'], 'split': 'test', 'snr_db': None if r['configuration']['test_snr_quarter_db'] is None else r['configuration']['test_snr_quarter_db']/4,
                'metrics': r['metrics'], 'n_samples': r['test_rows'], 'seed': r['configuration']['seed'],
                'implementation_version': r['environment']['packages']['drone-acoustic-fingerprint'], 'sources': bundle['sources'][key]}
               for key, r in bundle['results'].items()]
    atomic_json(output / 'metrics.json', {'scope': scope, 'results': metrics})
    sections = ''.join(f'<section><h2>{name.replace("_", " ").title()}</h2><img src="{name}.svg" alt="{name.replace("_", " ")}"><p>'
                       + ' · '.join(f'<a href="{name}.{ext}">{ext.upper()}</a>' for ext in ('png', 'svg', 'pdf')) + '</p></section>'
                       for name in ('figure2', 'figure3', 'figure4', 'figure5', 'diagnostic_confusion', 'diagnostic_open_recall'))
    for name, table in tables.items():
        header = ''.join('<th>' + escape(c) + '</th>' for c in table['columns'])
        body = ''.join('<tr>' + ''.join('<td>' + escape(cell(row[c])) + '</td>' for c in table['columns']) + '</tr>' for row in table['rows'])
        sections += f'<section><h2>{name.replace("_", " ").title()}</h2><div class="scroll"><table><tr>{header}</tr>{body}</table></div><p><a href="{name}.md">Markdown</a> · <a href="{name}.tex">LaTeX</a></p></section>'
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Acoustic fingerprint result gallery</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;background:#f2f5f7;color:#233844;margin:0}}main{{max-width:1250px;margin:auto;padding:32px 20px}}section{{background:white;margin:24px 0;padding:24px;border-radius:12px}}img{{width:100%;height:auto}}a{{color:#165e94}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px;border-bottom:1px solid #dce5eb;text-align:left}}.scope{{font-weight:700;color:#865716}}</style>
<main><h1>Acoustic fingerprint result gallery</h1><p class="scope">{scope}</p><p>Saved results only; no training, audio loading or interpolation. Missing grid points break curves and remain null in plot data. Model styling and nine-panel order follow the author's original plotting scripts.</p><p>Table 3 records fixed protocol identities. Tables 2, 4 and 5 contain measured percentages. Table 5 uses the unweighted mean over original attack identities; pooled rejection is a separate comparison. Trial means require all ten trials.</p><p><a href="plot-data.json">Plot data</a> · <a href="metrics.json">Metrics and sources</a> · <a href="tables.json">Tables</a> · <a href="manifest.json">Provenance</a></p>{sections}</main></html>'''
    (output / 'index.html').write_text(html, encoding='utf-8')
    manifest = {'schema_version': 1, 'status': 'passed', 'scope': scope, 'generated_at': datetime.now(timezone.utc).isoformat(),
                'selection': bundle['selection'], 'sources': bundle['sources'], 'skipped': bundle['skipped'],
                'duplicate_results_merged': bundle['duplicates'], 'verified_results': len(bundle['results']),
                'full_dataset': bundle['full_dataset'], 'full_grid': complete,
                'coverage': {p: {'completed': sum(r['provenance']['experiment'] == p for r in bundle['results'].values()), 'expected': bundle['expected_counts'][p]} for p in PRESETS},
                'plot_source_sha256': {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name('results.py'))},
                'artifacts': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sorted(output.iterdir())}}
    atomic_json(output / 'manifest.json', manifest)
    return manifest
