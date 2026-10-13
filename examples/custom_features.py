"""Compare base, delta and delta-delta blocks using the same selected WAVs."""

import argparse
from dataclasses import replace
from pathlib import Path

from drone_fingerprint.config import load_config
from drone_fingerprint.data import select_recordings, sha256
from drone_fingerprint.storage import atomic_json, create_output
from drone_fingerprint.workflow import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--audio-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    args = parser.parse_args()
    settings, manifest, subset = load_config(args.config)
    if settings.partition is not None or settings.test_snr_quarter_db is not None:
        raise ValueError('This example compares clean closed-set feature blocks')
    records = select_recordings(manifest, subset, settings.identities)
    output = create_output(args.output, args.audio_root)
    provenance = {'dataset': 'Researcher feature-block example; not a paper benchmark',
                  'recordings_manifest_sha256': sha256(manifest), 'subset_manifest_sha256': sha256(subset)}
    comparison = []
    for blocks in (1, 2, 3):
        candidate = replace(settings, blocks=blocks, portion=3)
        result = run_experiment(records, args.audio_root, candidate, output / f'blocks-{blocks}', provenance,
                                cache_root=args.cache_root)
        comparison.append({'blocks': blocks, 'columns': result['feature_columns'],
                           'metrics': result['metrics'], 'task_id': result['task_id']})
    atomic_json(output / 'comparison.json', comparison)
    print(f'Comparison: {output / "comparison.json"}')


if __name__ == '__main__':
    main()
