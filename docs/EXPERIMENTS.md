# Paper experiments, progress and recovery

The `paper` command expands the frozen protocol into ordinary `Settings` records
and calls the same `run_experiment` function as `run`. No task scheduler service,
database or extra dependency is required. Models run sequentially, with one
numerical thread. All commands below assume the locked Ubuntu environment from
[the quickstart](QUICKSTART.md).

## Start with a plan and a bounded example

```bash
uv run --locked drone-auth paper \
  --config config/validation/paper_smoke.json --dry-run
uv run --locked drone-auth paper \
  --config config/validation/paper_smoke.json \
  --audio-root ../refactor_artifacts/data/acsac2022/extracted \
  --output ../refactor_artifacts/runs/paper-smoke-001 \
  --cache-root ../refactor_artifacts/cache/paper-features \
  --log-root ../refactor_evidence/paper-smoke-001
```

This example trains all eight models on the frozen 48-recording validation
subset, using the Table 2 feature configuration. It does not reproduce Table 2:
the full protocol uses 240 drone recordings. Open-set validation additionally
uses the first 60 seconds of room noise; full open-set training uses the entire
room-noise recording. None of these commands creates sliced or noisy WAV files.

`paper_representative.json` exercises every preset with GNB, two frame lengths,
one filter/block/fraction choice, three noise levels and all ten open-set trials.
`paper_slow_fit.json` uses 20 ms DS1 frames with the two SVMs to exercise real
training visibility. It can take much longer than the basic example; there is
no wall-time limit or automatic cancellation.

`--model GNB RF` overrides the configured model list. `--preset figure5` overrides
the preset list, retaining selectors for presets already in the file and using
the complete frozen grid for other named presets. CLI overrides do not rewrite
configuration files. Use `--dry-run` to inspect the resolved tasks first.

## Full paper coverage

```bash
uv run --locked drone-auth paper \
  --config config/paper/full_sweep.json --dry-run
uv run --locked drone-auth paper \
  --config config/paper/full_sweep.json \
  --audio-root ../refactor_artifacts/data/acsac2022/extracted \
  --output ../refactor_artifacts/runs/full-paper-001 \
  --cache-root ../refactor_artifacts/cache/paper-features \
  --log-root ../refactor_evidence/full-paper-001/attempts
```

The second command starts the full experiment. For the result-by-result mapping,
monitor setup and plotting commands, see the [README workflow](../README.md#reproduce-the-paper-experiments).

| Preset | Fits | Evaluation-only tasks | Configuration |
| --- | ---: | ---: | --- |
| `figure2` | 408 | 0 | 51 frame lengths × 8 models |
| `figure3` | 3,600 | 0 | 50 filter counts × 3 block counts × 3 fractions × 8 models |
| `figure4` | 0 | 3,600 | Figure 3 models evaluated at 0 dB |
| `table2` | 8 | 0 | 24 identities, 201 filters, base coefficients |
| `figure5` | 0 | 744 | Table 2 models × 93 SNR levels |
| `tables4_5` | 80 | 0 | 10 frozen 8/8/8 partitions × 8 models |
| Total | 4,096 | 4,344 | 8,440 unique tasks, excluding retries |

Rows count each clean fit once across the full plan. Figure 4 alone needs 3,600
clean fits plus 3,600 evaluations; Figure 5 alone needs eight clean fits plus
744 evaluations.

Selecting Figure 4 or Figure 5 automatically includes its required clean fits,
without duplicating a clean task already selected in the same plan. Evaluation
results reference the original model with its task ID and checksum; they do not
fit or write another copy of the model. Preserve the complete output directory.
New evaluation results store relative model-reference paths, so the complete
sweep can move together. Earlier P4 results retain absolute references and need
their original paths. References are always checked; substitute models are not searched for.

`paper_gallery.json` is a bounded GNB example: 32 fits and 21 evaluations,
covering all nine Figure 3/4 panels at two adjacent filter counts, three frame
lengths, three SNR values, Table 2 and all ten open-set trials. Render saved
sweeps with [the plot command](FIGURES.md); it does not start experiments.

The dry run reads JSON manifests only. It reports train/test row counts, selected
columns, task dependencies and raw base-plus-selected array bytes for each task.
These are per-task sizes before cache sharing, not a prediction of total disk,
peak estimator memory or runtime. The feature writer checks available disk space
before allocating an array. Estimators can allocate additional memory. The
completed full run took about 11 hours on the validation host, retained 30.06 GiB
of results plus 26.14 GiB of cache, and reached 1.94 GiB peak process RSS. These
are observations, not limits or guarantees; see the [resource report](REPRODUCTION_RESULTS.md#reproducibility-and-measured-resources).

## Configuration for researchers

A sweep JSON has exactly these fields:

```json
{
  "schema_version": 1,
  "protocol": "../paper/protocol.json",
  "recordings_manifest": "../paper/recordings.json",
  "subset_manifest": "../paper/validation_subset.json",
  "partitions_manifest": "../paper/attack_partitions.json",
  "models": ["GNB"],
  "presets": {
    "figure3": {"n_filters": [201], "blocks": [1, 2, 3], "portion": [1, 2, 3]},
    "figure5": {"snr_quarter_db": [-32, 0, 60]},
    "tables4_5": {"trial": [1, 10]}
  }
}
```

Paths resolve relative to the JSON file. A preset's empty selector `{}` means
its entire grid. Selectors are `frame_ms`, `n_filters`, `blocks`, `portion`,
`trial` and `snr_quarter_db`; only values on that preset's frozen grid are
accepted. `portion=1,2,3` means one third, two thirds and all retained coefficients.
`snr_quarter_db` stores exact integer quarter-decibels, so `-32` is -8 dB.
Noise and trial selectors apply only to presets that define them. The complete
selection is `config/paper/full_subset.json`, with 240 drone WAV paths and the
full room-noise interval. For scientific changes beyond the paper grids, start
with a single `run` configuration and document the changed protocol.

The readable execution path is `cli.py` → `experiments.py::load_plan` /
`paper_tasks` → `run_plan` → `workflow.py::run_experiment`. The planner returns
plain dictionaries. Feature extraction, metrics, model constructors and cache
validation remain in their existing modules. `tests/test_experiments.py` checks
coverage, dependencies, failure handling, interruption and result reuse.

## Observe a slow model

For a browser dashboard with live task counts, phase, CPU/memory and attempt logs,
see [the live monitor guide](LIVE_MONITOR.md). It works locally or over an SSH
port forward and runs independently of the trainer.

Each task prints its preset, model, feature settings, trial, SNR and completed
task count. The header SNR value is in integer quarter-decibels; `None` means
clean audio. Feature loading, model initialization, training, evaluation and
serialization have separate start/end messages. Phases lasting over ten seconds
emit UTC timestamps, PID, elapsed time, process CPU, current RSS and Linux
`read_bytes` / `write_bytes` counters. The I/O values are process totals; they
are not physical SSD wear measurements. Activity is diagnostic information,
not proof of convergence. There is no fabricated fit percentage or ETA.

Warnings are shown and recorded, including SVM convergence warnings at the
preserved 1,000-iteration limit. A completed task can contain such warnings;
completion means the configured procedure returned, not that its estimator
converged. QDA rank warnings are also retained. Never increase iteration limits
silently to hide a warning or assume a silent fit is stuck.

## Interrupt, resume and inspect

Press Ctrl-C to interrupt; the CLI records an interrupted attempt and exits 130.
Repeat the identical command with `--resume` to continue. Native estimator code
can delay delivery of Ctrl-C until it returns control to Python; the phase
heartbeat can continue during this interval. Do not launch a duplicate fit. SIGKILL or power loss
can leave an attempt marked running; after the operating system releases the
run lock, resume treats that unfinished attempt as interrupted. A fit restarts
from its beginning, with the same seed and reusable feature cache. There are no
estimator checkpoints. Default execution stops on the first failure and returns
a nonzero exit code, preserving the failure for inspection.

| Location | Contents |
| --- | --- |
| `<output>/plan.json` | Resolved task grid, inputs and resource estimates |
| `<output>/sweep.json` | One entry per task; state and all attempt histories, source revision/dirty state |
| `<output>/tasks/<id>/attempt-NNN/` | Immutable completed artifacts or preserved incomplete/failed artifacts |
| `progress.json` in each attempt | Last observable phase, status, PID and UTC timestamp; updated at phase boundaries |
| `<log-root>/*.log` | Separate per-attempt terminal transcripts, including heartbeats and warnings |
| `<cache-root>/` | Shared checksummed NumPy features and recording index |

The four roots (audio, output, cache, logs) must be disjoint and generated files
must be outside source checkouts. A nonblocking Linux lock rejects a second
runner targeting the same output directory. It does not lock unrelated output
directories: intentionally starting the same plan under another output name
starts another experiment.

Resume validates source content, installed runtime, selected grid, manifest
identity, current audio signatures/checksums and completed artifact checksums.
It rejects changed plans and corrupted completions instead of guessing that a
partial directory is usable. A crash after result publication but before ledger
publication can recover the verified result without refitting. No append-only
CSV of metrics is maintained: read only the single current completed attempt
for each task in `sweep.json`, retaining older failures solely as history.
Heartbeat output is low frequency and status JSON is written only at boundaries.
