"""Presentation helpers must reject incomplete or altered measurements."""

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from drone_fingerprint.data import sha256

SPEC = importlib.util.spec_from_file_location(
    'paper_comparison', Path(__file__).resolve().parents[1] / 'tools/report_paper_comparison.py')
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def gallery(tmp_path):
    rows = []
    for preset in ('figure2', 'figure3', 'figure4', 'figure5'):
        panels = [(b, p) for b in (1, 2, 3) for p in (1, 2, 3)] if preset in ('figure3', 'figure4') else [(1, 3)]
        x = list(range(20, 2521, 50)) if preset == 'figure2' else [n / 4 for n in range(-32, 61)] if preset == 'figure5' else list(range(26, 272, 5))
        for b, p in panels:
            for i, model in enumerate(comparison.MODELS):
                rows.append({'preset': preset, 'blocks': b, 'portion': p, 'model': model,
                             'x_unit': 'ms' if preset == 'figure2' else 'dB' if preset == 'figure5' else 'filters',
                             'y_unit': 'percent', 'x': x, 'y': [30 + i + b + p] * len(x)})
    (tmp_path / 'plot-data.json').write_text(json.dumps({'series': rows}))
    manifest = {'status': 'passed', 'full_grid': True, 'full_dataset': True,
                'artifacts': {'plot-data.json': {'sha256': sha256(tmp_path / 'plot-data.json')}}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return tmp_path


def test_curves_keep_model_mapping_and_percentage_units(tmp_path):
    rows, _ = comparison.read_series(gallery(tmp_path))
    fig, ax = plt.subplots()
    try:
        comparison.draw_panel(ax, rows, 'figure3', 2, 1, 'd')
        for line, model in zip(ax.lines, comparison.MODELS):
            assert line.get_label() == model
            np.testing.assert_array_equal(line.get_ydata(), rows['figure3', 2, 1, model]['y'])
        assert ax.get_ylim() == (25, 100)
    finally:
        plt.close(fig)


@pytest.mark.parametrize('problem', ['checksum', 'subset', 'missing_value', 'duplicate_model'])
def test_untrustworthy_gallery_is_rejected(tmp_path, problem):
    root = gallery(tmp_path)
    path = root / 'plot-data.json'; meta = root / 'manifest.json'
    data = json.loads(path.read_text()); manifest = json.loads(meta.read_text())
    if problem == 'subset':
        manifest['full_dataset'] = False
    elif problem == 'duplicate_model':
        data['series'][1] = data['series'][0]
    else:
        data['series'][0]['y'][0] = None
    path.write_text(json.dumps(data))
    if problem != 'checksum':
        manifest['artifacts']['plot-data.json']['sha256'] = sha256(path)
    meta.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, AssertionError)):
        comparison.read_series(root)
