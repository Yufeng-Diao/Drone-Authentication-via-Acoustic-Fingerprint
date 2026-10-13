"""Customize one saved Figure 2 series, preserving gaps in the original grid."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from drone_fingerprint.storage import create_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.gallery / 'plot-data.json').read_text())
    line = next(row for row in data['series'] if row['preset'] == 'figure2' and row['model'] == 'GNB')
    output = create_output(args.output)
    y = np.asarray([np.nan if value is None else value for value in line['y']])
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(line['x'], y, 'o-', color='#1b9e77', label='GNB')
    ax.set(xlabel='Frame length (ms)', ylabel='Accuracy (%)', ylim=(0, 100), title=data['scope'])
    ax.legend()
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'custom-series.{extension}', dpi=160)
    plt.close(fig)
    print(f'Custom figure: {output}')


if __name__ == '__main__':
    main()
