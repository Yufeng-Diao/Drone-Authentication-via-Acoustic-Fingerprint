"""Compare a verified full gallery with Figures 2–5 in the author's ACSAC PDF.

Presentation follows varTime_plot.py, varFilt_plot_all.py and varNoise_plot.py.
No historical CSV, audio, feature arrays, model loading or training is needed.
"""

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from drone_fingerprint.config import MODELS
from drone_fingerprint.data import sha256
from drone_fingerprint.storage import atomic_json, create_output

# Fixed PDF coordinates must not silently crop a different manuscript revision.
PAPER_SHA256 = '38d4d1630c0e143a2ce19478ce7a575b67b52ca106d8cccd0b903b356dbcd762'
CROPS = {'figure2': (5, (327, 94, 550, 232)),
         'figure3': (7, (62, 84, 551, 373)),
         'figure4': (8, (62, 84, 551, 373)),
         'figure5': (8, (68, 489, 284, 627))}
COLORS = ('#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666')
FORMATS = ('-', '--', '|-', '1:', '.-', 'x-', '>-', '*-')
LIMITS = {'figure2': ((0, 2520), (15, 100)), 'figure3': ((21, 272), (25, 100)),
          'figure4': ((21, 272), (25, 100)), 'figure5': ((-8, 15), (10, 100))}
NOTES = {
    'figure2': 'The longer-frame trend and high QDA/LDA/LSVM plateau remain. The most visible difference is RBF-SVM: its rise over short and intermediate frames is slower in the new run, while several other curves are much closer. Similar overall shape does not imply pointwise equality.',
    'figure3': 'The largest visible change is KNN in panels (d)–(i): the new derivative-feature curves stay much higher than in the paper. Base-MFCC panels (a)–(c) are substantially closer in shape. RBF-SVM remains variable. The corrected derivative and normalization definitions are documented; this comparison alone does not isolate their individual effects.',
    'figure4': 'LDA remains strong under 0 dB noise. Inspect KNN and RBF-SVM in derivative panels (d)–(i), plus the low-filter end of (a), (d) and (g). The paper-scale view clips some new values below 25%; the complete 0–100% export is provided explicitly.',
    'figure5': 'The increasing-SNR trend and high-SNR plateaus remain. Compare the low-SNR QDA/LDA/LSVM curves closely: the new paths do not perfectly match the published curves. These are closed-set noise experiments; the open-set background-label diagnostic is not used here.',
}


def read_series(gallery):
    """Require complete, hash-verified percentages on each frozen grid."""
    manifest = json.loads((gallery / 'manifest.json').read_text())
    if not (manifest['status'] == 'passed' and manifest['full_grid'] and manifest['full_dataset']):
        raise ValueError('Paper comparison requires a verified full-data, full-grid gallery')
    record = manifest['artifacts']['plot-data.json']
    if sha256(gallery / 'plot-data.json') != record['sha256']:
        raise ValueError('Plot-data checksum does not match the gallery manifest')
    rows = json.loads((gallery / 'plot-data.json').read_text())['series']
    indexed = {(r['preset'], r['blocks'], r['portion'], r['model']): r for r in rows}
    assert len(indexed) == len(rows) == 160
    for preset in CROPS:
        grid = list(range(20, 2521, 50)) if preset == 'figure2' else [v/4 for v in range(-32, 61)] if preset == 'figure5' else list(range(26, 272, 5))
        panels = [(b, p) for b in (1, 2, 3) for p in (1, 2, 3)] if preset in ('figure3', 'figure4') else [(1, 3)]
        for b, p in panels:
            for model in MODELS:
                row = indexed[preset, b, p, model]
                assert row['x'] == grid and len(row['y']) == len(grid)
                assert row['y_unit'] == 'percent' and row['x_unit'] == ('ms' if preset == 'figure2' else 'dB' if preset == 'figure5' else 'filters')
                assert all(v is not None and np.isfinite(v) and 0 <= v <= 100 for v in row['y'])
    assert sum(len(row['y']) for row in rows) == 8352
    return indexed, manifest


def draw_panel(ax, rows, preset, blocks, portion, letter=None, full_range=False):
    for model, color, fmt in zip(MODELS, COLORS, FORMATS):
        row = rows[preset, blocks, portion, model]
        line, = ax.plot(row['x'], row['y'], fmt, color=color, linewidth=1.5 if letter else 2, markersize=6, label=model)
        np.testing.assert_array_equal(line.get_xdata(), row['x'])
        np.testing.assert_array_equal(line.get_ydata(), row['y'])
    xlabel = {'figure2': 'Frame Length (ms)', 'figure5': 'SNR (dB)'}.get(preset, 'Number of Filters (n)')
    ax.set(xlabel=xlabel + (f'\n({letter})' if letter else ''), ylabel='Accuracy (%)',
           xlim=LIMITS[preset][0], ylim=(0, 100) if full_range else LIMITS[preset][1])
    if preset == 'figure5':
        ax.set_xticks(np.arange(-8, 16, 2))
    ax.grid(True, color='#b0b0b0', linewidth=.6)
    ax.legend(loc='lower right', ncol=3, fontsize=12 if letter else 16, framealpha=.5 if letter else 1)


def save(fig, output, name, extensions=('png', 'svg', 'pdf')):
    fig.tight_layout(pad=.7)
    for suffix in extensions:
        fig.savefig(output / f'{name}.{suffix}', dpi=160)
    plt.close(fig)


def crop_pdf(paper, page, box, output, commands):
    # Poppler renders the actual PDF region; no reference curve is digitized.
    x0, y0, x1, y1 = box
    command = ['pdftoppm', '-f', str(page), '-l', str(page), '-singlefile', '-r', '288',
               '-x', str(x0*4), '-y', str(y0*4), '-W', str((x1-x0)*4), '-H', str((y1-y0)*4),
               '-png', str(paper), str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    commands.append({'command': command, 'exit_code': 0, 'page': page, 'box_pdf_points': box})


def image_pair(left, right, title):
    return f'<div class="pair"><figure><figcaption>Published paper</figcaption><a href="{left}"><img loading="lazy" src="{left}" alt="Published {escape(title)}"></a></figure><figure><figcaption>New full-run results · original plotting style</figcaption><a href="{right}"><img loading="lazy" src="{right}" alt="New {escape(title)}"></a></figure></div>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery', type=Path, required=True)
    parser.add_argument('--paper-pdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.paper_pdf) != PAPER_SHA256:
        raise ValueError('Expected the author-supplied ACSAC_Yufeng.pdf; crop coordinates are version-specific')
    if shutil.which('pdftoppm') is None:
        raise ValueError('Install Ubuntu poppler-utils to render the original PDF figures')
    rows, manifest = read_series(args.gallery)
    output = create_output(args.output)
    font = 'Liberation Serif' if any(f.name == 'Liberation Serif' for f in font_manager.fontManager.ttflist) else 'DejaVu Serif'
    plt.rcParams.update({'font.family': font, 'font.size': 16, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    shutil.copyfile(args.gallery / 'plot-data.json', output / 'plot-data.json')
    commands, sections, clipping = [], [], {}
    for preset, (page, box) in CROPS.items():
        print(f'Rendering {preset}: published PDF and new measurements', flush=True)
        multi = preset in ('figure3', 'figure4')
        panels = [(b, p) for b in (1, 2, 3) for p in (1, 2, 3)] if multi else [(1, 3)]
        clipping[preset] = sum(v < LIMITS[preset][1][0] for r in rows.values() if r['preset'] == preset for v in r['y'])
        fig, axes = plt.subplots(3, 3, figsize=(20, 12)) if multi else plt.subplots(1, 1, figsize=(12, 6), squeeze=False)
        for i, ((b, p), ax) in enumerate(zip(panels, axes.flat)):
            draw_panel(ax, rows, preset, b, p, chr(97+i) if multi else None)
        save(fig, output, preset)
        crop_pdf(args.paper_pdf.resolve(), page, box, output / f'{preset}-published', commands)
        # A labelled standalone comparison is convenient to share outside the HTML report.
        pair, axes = plt.subplots(1, 2, figsize=(20, 6.7 if multi else 6))
        for ax, name, title in zip(axes, (f'{preset}-published', preset), ('Published paper', 'New full-run results')):
            ax.imshow(plt.imread(output / f'{name}.png')); ax.set_axis_off(); ax.set_title(title)
        save(pair, output, f'{preset}-comparison', extensions=('png', 'pdf'))
        title = preset.replace('figure', 'Figure ')
        links = ' · '.join(f'<a href="{preset}.{ext}">New {ext.upper()}</a>' for ext in ('png', 'svg', 'pdf'))
        section = f'<section id="{preset}"><h2>{title}</h2><p>{NOTES[preset]}</p><p>{links} · <a href="{preset}-comparison.pdf">Side-by-side PDF</a></p>'
        section += image_pair(f'{preset}-published.png', f'{preset}.svg', title)
        section += f'<p>Historical vertical range: {LIMITS[preset][1][0]}–100%. New points below this range: <strong>{clipping[preset]}</strong>. Source: PDF page {page}, printed page {657+page}.</p>'
        if clipping[preset]:
            fig, axes = plt.subplots(3, 3, figsize=(20, 12))
            for i, ((b, p), ax) in enumerate(zip(panels, axes.flat)):
                draw_panel(ax, rows, preset, b, p, chr(97+i), full_range=True)
            save(fig, output, preset+'-full-range')
            section += f'<p><a href="{preset}-full-range.svg">View every new point on the complete 0–100% scale</a> · <a href="{preset}-full-range.pdf">Full-range PDF</a></p>'
        if multi:
            section += '<h3>Compare individual panels</h3><p>Rows: MFCC; MFCC + delta; MFCC + delta + delta-delta. Columns: one third; two thirds; all coefficients. Expand a panel, then click either image to enlarge.</p>'
            for i, (b, p) in enumerate(panels):
                letter = chr(97+i); name = f'{preset}-{letter}'
                fig, ax = plt.subplots(figsize=(7, 4.2)); draw_panel(ax, rows, preset, b, p, letter)
                save(fig, output, name, extensions=('png', 'svg'))
                xs, ys = [62, 226, 387, 551], [84, 181, 277, 373]
                crop_pdf(args.paper_pdf.resolve(), page, (xs[p-1], ys[b-1], xs[p], ys[b]), output / f'{name}-published', commands)
                section += f'<details><summary>Panel ({letter}) · {("MFCC", "MFCC + delta", "MFCC + delta + delta-delta")[b-1]} · {("one third", "two thirds", "all coefficients")[p-1]}</summary>' + image_pair(f'{name}-published.png', f'{name}.svg', name) + '</details>'
        sections.append(section+'</section>')
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Published paper vs new experiments</title><style>
body{margin:0;background:#f2f4f6;color:#20343d;font:16px/1.65 system-ui,sans-serif}main{max-width:1800px;margin:auto;padding:24px}h1{line-height:1.2}section{background:white;padding:24px;margin:24px 0;border-radius:10px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}figure{margin:0;min-width:0}img{width:100%;height:auto}figcaption{font-weight:600;margin:12px 0}a{color:#17627d}nav{display:flex;gap:24px;flex-wrap:wrap}.note{padding:16px;border-left:4px solid #ae7a28;background:#fff3db}summary{cursor:pointer;padding:12px;font-weight:600}details{border-top:1px solid #dce3e5}details .pair{padding-bottom:20px}@media(max-width:850px){.pair{grid-template-columns:1fr}main{padding:12px}section{padding:14px}}
</style></head><body><main><h1>Published paper vs new experiments</h1><p>Figures 2–5 · full P6 measurements · original colors, line styles, markers, legends and axis ranges</p><nav><a href="#figure2">Figure 2</a><a href="#figure3">Figure 3</a><a href="#figure4">Figure 4</a><a href="#figure5">Figure 5</a><a href="plot-data.json">New plot data</a><a href="manifest.json">Provenance</a></nav>
<p class="note">Left: actual figure regions rendered from the paper PDF. Right: newly plotted measurements, never copied from the published curves. All 8,352 new points are retained unchanged. Historical curve CSV files are unavailable: comparisons with the original curves are visual, not numerical error estimates. New outputs use the frozen primary experiment, not the separate open-set label-order diagnostic.</p>
<p>Presentation is adapted from the author's <code>varTime_plot.py</code>, <code>varFilt_plot_all.py</code> and <code>varNoise_plot.py</code>. Model names replace positional CSV columns. A portable serif font replaces Times New Roman; pixel-identical typography is not claimed. Full figures are displayed side by side on wide screens and stacked on phones. Each image opens at full size.</p>''' + ''.join(sections) + '</main></body></html>'
    (output / 'index.html').write_text(html)
    atomic_json(output / 'manifest.json', {'status': 'passed', 'generated_at': datetime.now(timezone.utc).isoformat(),
        'gallery': str(args.gallery.resolve()), 'gallery_manifest_sha256': sha256(args.gallery/'manifest.json'),
        'plot_data_sha256': sha256(output/'plot-data.json'), 'paper_pdf_sha256': PAPER_SHA256,
        'generator_sha256': sha256(Path(__file__)), 'verified_points': 8352, 'verified_series': 160,
        'unchanged_plot_values': True, 'training_performed': False, 'font': font, 'points_below_historical_axes': clipping,
        'crop_commands': commands, 'gallery_scope': manifest['scope'],
        'artifacts': {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(output.iterdir())}})
    print(f'Comparison: {output / "index.html"}')


if __name__ == '__main__':
    main()
