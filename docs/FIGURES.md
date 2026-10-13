# Figures and tables from saved experiments

Use the locked [Ubuntu environment](QUICKSTART.md). Plotting reads `plan.json`,
`sweep.json`, the current completed `result.json`, prediction arrays and artifact
checksums. It neither reads audio/features nor deserializes models or trains.
Preserve the complete sweep, including clean model dependencies.

```bash
uv run --locked drone-auth plot \
  --runs ../refactor_artifacts/runs/paper-smoke-001 \
  --output ../refactor_evidence/paper-smoke-001-figures
```

Open the output's `index.html` locally. Every figure has PNG, SVG and PDF exports.
Tables have Markdown, LaTeX `tabular` and JSON exports. `metrics.json` retains
configuration, sample counts, seed and result sources; `plot-data.json` contains
the actual axis values and percentages. `manifest.json` records source hashes,
coverage, skipped tasks and export checksums. No intermediate CSV is required.
The LaTeX exports are tabular fragments for insertion into a document, not full
standalone documents; Table 3 may require a landscape page or smaller type.

To exercise every layout using a bounded public-data run:

```bash
uv run --locked drone-auth paper \
  --config config/validation/paper_gallery.json --dry-run
uv run --locked drone-auth paper \
  --config config/validation/paper_gallery.json \
  --audio-root ../refactor_artifacts/data/acsac2022/extracted \
  --output ../refactor_artifacts/runs/paper-gallery-001 \
  --cache-root ../refactor_artifacts/cache/features \
  --log-root ../refactor_evidence/paper-gallery-001
uv run --locked drone-auth plot \
  --runs ../refactor_artifacts/runs/paper-smoke-001 \
         ../refactor_artifacts/runs/paper-gallery-001 \
  --output ../refactor_evidence/paper-gallery-001-figures
```

This runs 53 tasks with GNB, including all nine filter panels and ten open-set
trials. It is a validation subset, not a reproduction of paper accuracy. Supplying
multiple runs merges distinct configurations only when data, protocol, seed and
runtime agree. Identical repeated predictions are deduplicated with every source
retained; conflicting repeats fail instead of being averaged. Different runtime
or dataset runs must be plotted separately. `--paper-dir` defaults to
`config/paper` and may point to that directory from another working directory.

## Scientific mapping

| Output | Axes or metric | Selection |
| --- | --- | --- |
| Figure 2 | Frame length (ms), accuracy (%) | DS1; 51 frame lengths |
| Figure 3 | Filter count, accuracy (%) | DS1 clean; nine feature panels |
| Figure 4 | Filter count, accuracy (%) | Same panels; 0 dB test AWGN, clean fitted models |
| Figure 5 | SNR (dB), accuracy (%) | DS2; 93 quarter-decibel values from -8 to 15 |
| Table 2 | Accuracy, macro precision/recall/F1 (%) | DS2, eight classifiers |
| Table 3 | Identity numbers | Frozen registered/background/attack partitions |
| Table 4 | Registered macro recall (%) | Ten trials and unweighted ten-trial mean |
| Table 5 | Attack macro rejection (%) | Mean over original attack identities; ten trials |

Figure 3/4 rows are base MFCC, MFCC + delta, then MFCC + delta + delta-delta.
Columns retain one third, two thirds, then all coefficients. Panels (a)–(i)
follow row-major order. Coefficient zero is excluded per block before prefix
selection. Eight model styles use stable names, colors, line styles and markers.
The axes show the full frozen grid and 0–100% accuracy; fonts use portable
DejaVu Sans. These presentation choices replace historical positional CSV
columns, absolute paths and machine-specific fonts.

Missing points remain JSON `null`, break lines and are never interpolated.
No result is treated as zero. Tables display missing cells as an em dash; a
ten-trial mean appears only when all ten observations exist. All subset plots
say **INCOMPLETE PREVIEW**, even when a selected model's subset grid is complete.
Full-data plots describe a corrected rerun, not exact historical reproduction.

`table5_pooled_comparison` keeps collapsed, sample-weighted rejection separate
from the primary per-identity macro definition. Diagnostics include eight
row-normalized Table 2 confusion matrices and per-identity registered/attack
recall for the first available trial/model, identified in its title. Table 3
is protocol metadata and must not be presented as measured accuracy.

Failed, running and pending tasks remain missing. Only the latest attempt of a
currently completed task is used. Stale successful attempts are not recovered
by scanning directories. Checksums, configuration, provenance, prediction counts,
metrics and confusion matrices are checked before creating output. A fresh
output directory is required; existing figures and input sweeps are preserved.
