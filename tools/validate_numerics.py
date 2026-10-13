"""Generate external P2 numerical measurements and figures from fixed references.

Run after installing the package. Historical comparisons replay documented
operations on the current stack; they are not recovered 2022 paper results.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from toolbox.mfcc_base import delta

from drone_fingerprint.config import Settings, load_config
from drone_fingerprint.data import read_recording, select_recordings, split_intervals
from drone_fingerprint.features import extract_base, extract_segment, select_features
from drone_fingerprint.metrics import classification_metrics
from drone_fingerprint.models import make_model
from drone_fingerprint.noise import add_noise
from drone_fingerprint.workflow import create_output, environment


def replay_features(segments, stage, filters=50, portion=2):
    """One correction at a time: locality, derivative, count, normalization."""
    groups = [np.concatenate(segments)] if stage == 0 else segments
    result = []
    count = (filters if stage < 3 else filters - 1) * portion // 3
    for base in groups:
        first = delta(base, 1)
        second = delta(base, 2) if stage < 2 else delta(first, 1)
        blocks = [base, first, second]
        if stage < 4:
            selected = [normalize(b)[:, 1:1 + count] for b in blocks]
            result.append(np.concatenate(selected, axis=1))
        else:
            result.append(select_features(base, Settings(n_filters=filters, blocks=3, portion=portion)))
    return np.concatenate(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--audio-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = create_output(args.output, args.audio_root)
    (output / 'figures').mkdir()
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            fig.savefig(output / 'figures' / f'{name}.{suffix}', dpi=170)
        plt.close(fig)

    signal = np.tile(np.array([-1000, 1000], dtype=np.int16), 500_000)
    legacy_power = float(np.sum(abs(signal) ** 2, dtype=np.int64) / len(signal))
    snr_rows = []
    for requested in (-8, -4, 0, 5, 10, 15):
        noisy, info = add_noise(signal, 'a' * 64, 2022, requested * 4)
        old_noise = (noisy - signal) * np.sqrt(legacy_power / info['signal_power'])
        old_measured = float(10 * np.log10(1_000_000 / np.mean(old_noise ** 2)))
        assert abs(info['measured_full_recording_snr_db'] - requested) < 0.05
        snr_rows.append(dict(info, historical_measured_snr_db=old_measured))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = [r['requested_snr_db'] for r in snr_rows]
    ax.plot(xs, xs, '--', color='gray', label='Requested')
    ax.plot(xs, [r['measured_full_recording_snr_db'] for r in snr_rows], 'o-', label='Float64 power')
    ax.plot(xs, [r['historical_measured_snr_db'] for r in snr_rows], 's-', label='Historical int16 power')
    ax.set(xlabel='Requested SNR (dB)', ylabel='Measured full-recording SNR (dB)', title='B1: squaring before casting understates noise power')
    ax.legend()
    save(fig, 'snr')

    slope = np.arange(20.)[:, None]
    segments = [np.ones((10, 26)), np.full((10, 26), 100.)]
    joined = np.concatenate(segments)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].plot(delta(slope, 2), label='Historical delta(base, 2)')
    axes[0].plot(delta(delta(slope, 1), 1), label='True delta(delta(base, 1), 1)')
    axes[0].set(title='B3: linear input has zero interior acceleration', xlabel='Frame', ylabel='Second-block value')
    axes[1].plot(delta(joined, 1)[:, 1], label='Historical joined segments')
    axes[1].plot(np.concatenate([delta(b, 1) for b in segments])[:, 1], label='Corrected separate segments')
    axes[1].axvline(9.5, color='gray', linestyle='--')
    axes[1].set(title='B4: two constant segments, different levels', xlabel='Frame', ylabel='First derivative')
    for ax in axes: ax.legend(fontsize=9)
    save(fig, 'derivatives')

    dimensions = []
    t = np.arange(11025) / 44100
    for filters in (26, 50, 201, 271):
        for portion in (1, 2, 3):
            for blocks in (1, 2, 3):
                features = extract_segment(1000 * np.sin(2 * np.pi * 450 * t), Settings(n_filters=filters, portion=portion, blocks=blocks, frame_ms=100))
                expected = ((filters - 1) * portion // 3) * blocks
                error = float(np.max(abs(np.linalg.norm(features, axis=1) - 1)))
                assert features.shape[1] == expected and error < 1e-12
                dimensions.append(dict(filters=filters, portion=portion, blocks=blocks, actual=features.shape[1], expected=expected, max_norm_error=error))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for portion in (1, 2, 3):
        rows = [r for r in dimensions if r['portion'] == portion and r['blocks'] == 3]
        ax.plot([r['filters'] for r in rows], [r['actual'] for r in rows], 'o-', label=f'{portion}/3 of available coefficients')
    ax.set(xlabel='Extracted filters / base coefficients', ylabel='Final columns, three blocks', title='B6: all 36 dimension and normalization cases passed')
    ax.legend()
    save(fig, 'dimensions')

    settings, manifest, subset = load_config(args.config)
    settings = replace(settings, identities=tuple(range(1, 9)), n_filters=50, blocks=3, portion=2, test_snr_quarter_db=None, partition=None)
    records = select_recordings(manifest, subset, settings.identities)
    bases, labels = {'train': [], 'test': []}, {'train': [], 'test': []}
    for index, record in enumerate(records, 1):
        print(f"Reference extraction {index}/{len(records)}: {record['path']}", flush=True)
        samples = read_recording(args.audio_root, record)
        for split, intervals in split_intervals(len(samples)).items():
            for start, end in intervals:
                base = extract_base(samples[start:end], settings)
                bases[split].append(base)
                labels[split].extend([record['identity']] * len(base))
    names = ['Historical PKL operations', '+ Segment-local deltas', '+ True second derivative', '+ Post-drop fraction count', '+ Final concatenated L2']
    comparisons, norm_rows = [], []
    for stage, name in enumerate(names):
        train, test = (replay_features(bases[s], stage) for s in ('train', 'test'))
        model = make_model('GNB', 2022).fit(train, labels['train'])
        prediction = model.predict(test)
        comparisons.append(dict(stage=name, columns=train.shape[1], **classification_metrics(labels['test'], prediction, list(settings.identities))))
        norm_rows.append(np.linalg.norm(train, axis=1))
    raw = np.concatenate(bases['train'])
    direct_norm = np.linalg.norm(np.concatenate([normalize(b[:, 1:]) for b in (raw, delta(raw, 1), delta(raw, 2))], axis=1), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(names, [r['macro_f1'] for r in comparisons], color=['#8b96a5'] + ['#278b83'] * 4)
    axes[0].set(xlabel='Macro F1 (eight-identity validation subset)', xlim=(0, 1.05), title='Sequential correction comparison on the current stack')
    axes[0].invert_yaxis()
    for i, row in enumerate(comparisons): axes[0].text(row['macro_f1'] + 0.01, i, f"{row['macro_f1']:.3f}", va='center')
    axes[1].boxplot([norm_rows[0], direct_norm, norm_rows[-1]], tick_labels=['PKL / 2 thirds', 'Direct / all', 'Corrected / 2 thirds'], showfliers=False)
    axes[1].axhline(1, color='gray', linestyle='--')
    axes[1].set(ylabel='Final row L2 norm', title='B5: historical conventions differ')
    axes[1].tick_params(axis='x', labelrotation=15)
    save(fig, 'corrections')
    result = {'status': 'passed', 'environment': environment(), 'legacy_power': legacy_power, 'corrected_power': 1_000_000,
              'snr': snr_rows, 'dimensions': dimensions, 'sequential_comparison': comparisons,
              'reference_recordings': records, 'comparison_train_rows': len(labels['train']), 'comparison_test_rows': len(labels['test']),
              'comparison_scope': 'Current-stack replay of historical PKL operations in frozen manifest order; clean DS1 validation subset, GNB, 50 filters, 1s frames, three blocks, two-thirds selection. Sequential changes are order-dependent, not isolated causal effects or recovered paper results.',
              'norm_ranges': {'historical_pkl': [float(norm_rows[0].min()), float(norm_rows[0].max())], 'historical_direct_all': [float(direct_norm.min()), float(direct_norm.max())], 'corrected': [float(norm_rows[-1].min()), float(norm_rows[-1].max())]}}
    (output / 'measurements.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
