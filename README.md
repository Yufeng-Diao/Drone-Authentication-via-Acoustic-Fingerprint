# Drone Authentication via Acoustic Fingerprint

Research code for the [ACSAC 2022 paper](https://doi.org/10.1145/3564625.3564653)
by Yufeng Diao, Yichi Zhang, Guodong Zhao and Mohamed Khamis. The maintained
workflow targets **Ubuntu 24.04**, with **Python 3.14.8** and **3.13.16** compatibility
checks. Exact scientific dependencies are recorded in `uv.lock`.

The pipeline reads original WAV files, extracts MFCC features in bounded batches,
trains the eight paper classifiers, and exports figures and tables from saved
results. Shared binary arrays replace per-frame CSV/PKL files. Long initialization
and training phases show activity in the terminal and an optional live webpage.

The full updated experiment supports the paper's central findings: acoustic
fingerprints distinguish the evaluated drone configurations, and QDA remains a
strong balanced choice for closed-set and open-set authentication. The updated
implementation also provides further observations on feature settings and some
classifier behaviors. See [results and interpretation](#results-and-interpretation)
for the measured comparisons and their scope.

## Choose a deployment method

For running experiments without installing host Python, use the
[Docker and Compose workflow](docs/CONTAINER.md). One image provides training,
port-8765 live monitoring, result plotting and paper-style comparisons. Data,
models and logs stay in explicit host directories. Copy `docker.env.example`
outside the checkout and set your paths. The prebuilt Linux/amd64 image is available
on [Docker Hub](https://hub.docker.com/r/eidos1970/drone-acoustic-fingerprint):

```bash
docker pull eidos1970/drone-acoustic-fingerprint:2026.10.18
```

The release was verified with anonymous registry access and matches the validated
image. See the [container guide](docs/CONTAINER.md#image-availability-and-validation)
for its immutable digest, source revision and validation scope.

The native workflow below remains useful for modifying and debugging the research
code. Both methods use the same implementation and dependency lock.

## Workflow

1. [Install the locked environment and set paths](#install-and-set-paths).
2. [Prepare and verify the public dataset](#prepare-the-public-dataset).
3. [Select a configuration](#configure-an-experiment).
4. [Run the paper experiments](#reproduce-the-paper-experiments), optionally with
   [live monitoring](#watch-training-and-resume).
5. [Export figures, tables and a paper comparison](#generate-figures-and-tables).

Run commands from the root of a checkout of the maintained `main` branch.
The original release remains available separately; see
[version history](#maintained-and-original-versions).

## Install and set paths

Install the pinned **uv 0.13.0** in a separate bootstrap environment. On Ubuntu,
install `python3-venv` first if it is unavailable (`sudo apt-get install python3-venv`).
This keeps the system Python packages separate from the project environment.

```bash
# Change these two roots to absolute paths on a volume with sufficient free space.
export DRONE_ARTIFACTS="$(realpath -m "$PWD/../refactor_artifacts")"
export DRONE_EVIDENCE="$(realpath -m "$PWD/../refactor_evidence")"
mkdir -p "$DRONE_ARTIFACTS/environments" "$DRONE_EVIDENCE"
python3 -m venv "$DRONE_ARTIFACTS/environments/uv-bootstrap"
"$DRONE_ARTIFACTS/environments/uv-bootstrap/bin/python" -m pip install uv==0.13.0
export PATH="$DRONE_ARTIFACTS/environments/uv-bootstrap/bin:$PATH"

export UV_PROJECT_ENVIRONMENT="$DRONE_ARTIFACTS/environments/default"
export UV_CACHE_DIR="$DRONE_ARTIFACTS/cache/uv"
export UV_PYTHON_INSTALL_DIR="$DRONE_ARTIFACTS/environments/python"
export MPLCONFIGDIR="$DRONE_ARTIFACTS/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=""
uv sync --locked
uv run --locked drone-auth --help
uv run --locked drone-auth synthetic --output "$DRONE_ARTIFACTS/runs/synthetic-001"
```

`uv` installs the interpreter selected by `.python-version` and dependencies from
`uv.lock`. Synthetic accuracy is an installation check. Choose a fresh output name
for each new run; existing results are preserved. For the same lock with venv/pip,
see [installation alternatives](docs/QUICKSTART.md#standard-venvpip-installation-from-the-same-lock).
A [CPU Docker workflow](docs/CONTAINER.md) is also available.

Set the experiment paths in the same shell:

```bash
export AUDIO_ROOT="$DRONE_ARTIFACTS/data/acsac2022/extracted"
export CACHE_ROOT="$DRONE_ARTIFACTS/cache/paper-features"
export RUN_ROOT="$DRONE_ARTIFACTS/runs/full-paper-001"
export LOG_ROOT="$DRONE_EVIDENCE/full-paper-001/attempts"
export FIGURE_ROOT="$DRONE_EVIDENCE/full-paper-001/figures"
mkdir -p "$DRONE_EVIDENCE/full-paper-001"
```

| Path | Stored content |
| --- | --- |
| `AUDIO_ROOT` | Extracted, original WAV files; treated as read-only |
| `CACHE_ROOT` | Shared recording index and checksummed NumPy feature arrays |
| `RUN_ROOT` | Plan, task ledger, predictions, metrics and trained models |
| `LOG_ROOT` | Separate per-attempt training logs and warnings |
| `FIGURE_ROOT` | Offline HTML gallery, figures, tables and export metadata |

Keep all generated files outside source checkouts. Audio, run, cache and log roots
must be disjoint (none inside another). A model's path is
`$RUN_ROOT/tasks/<task-id>/attempt-NNN/model.joblib`; `result.json` and
`predictions.npz` are alongside it. Evaluation-only tasks reference a clean model
in the same sweep. Preserve the whole run directory when moving results. See
[artifact layout and recovery](docs/EXPERIMENTS.md#interrupt-resume-and-inspect).

Reapply the environment and path exports in every new terminal, including the
monitor terminal. The CLI does not read these shell variable names automatically;
the commands below pass their values explicitly.

## Prepare the public dataset

Download the [CC BY 4.0 public dataset](https://doi.org/10.5525/gla.researchdata.1348)
and extract the archive so that `AUDIO_ROOT` directly contains `1to8`, `9to16`,
`17to24` and `noise`. Point `AUDIO_ROOT` at that level, below any archive wrapper
folder. There is no need to split, rename, normalize or make noisy copies of WAVs.

```bash
uv run --locked python tools/validate_p0.py --audio-root "$AUDIO_ROOT" \
  > "$DRONE_EVIDENCE/full-paper-001/dataset-validation.json"
uv run --locked drone-auth index \
  --manifest config/paper/recordings.json \
  --audio-root "$AUDIO_ROOT" --cache-root "$CACHE_ROOT"
```

Continue after both commands succeed. Validation checks the 257 WAV files against
the manifest, including hashes, format, distance pairs and Table 1 durations.
The paper selection uses 240 drone recordings; open-set training additionally
uses the full room-noise recording. The other 16 drone recordings remain on disk
but are excluded by the historical selection. See the [data specification](docs/PAPER_PROTOCOL.md#dataset-and-fixed-validation-subset).

## Configure an experiment

| Configuration | Purpose |
| --- | --- |
| [`config/smoke/closed_set.json`](config/smoke/closed_set.json) | One small two-identity GNB run using `drone-auth run` |
| [`config/validation/paper_smoke.json`](config/validation/paper_smoke.json) | Eight classifiers at the Table 2 settings, on the 48-recording validation subset |
| [`config/validation/paper_gallery.json`](config/validation/paper_gallery.json) | 53 GNB tasks covering every plot layout on the validation subset |
| [`config/paper/full_sweep.json`](config/paper/full_sweep.json) | All paper grids, eight classifiers and the complete paper dataset |

Subset configurations check the workflow; their scores are not full-paper
measurements. For a small first public-data check, run:

```bash
uv run --locked drone-auth paper \
  --config config/validation/paper_smoke.json --model GNB \
  --audio-root "$AUDIO_ROOT" \
  --output "$DRONE_ARTIFACTS/runs/paper-smoke-001" \
  --cache-root "$CACHE_ROOT" --log-root "$DRONE_EVIDENCE/paper-smoke-001"
```

For a custom selection, copy `config/paper/full_sweep.json` to
`config/paper/my_study.json` and edit `models` and `presets`. For example, this
complete configuration selects a small part of the **full dataset** grid:

```json
{
  "schema_version": 1,
  "protocol": "../paper/protocol.json",
  "recordings_manifest": "../paper/recordings.json",
  "subset_manifest": "../paper/full_subset.json",
  "partitions_manifest": "../paper/attack_partitions.json",
  "models": ["QDA", "GNB"],
  "presets": {
    "figure3": {"n_filters": [201], "blocks": [1, 2, 3], "portion": [3]},
    "figure5": {"snr_quarter_db": [-32, 0, 60]}
  }
}
```

Paths inside JSON resolve relative to that JSON file; update them if the file is
moved. CLI paths resolve relative to the working directory. An empty preset
selector `{}` requests its entire frozen grid. `blocks` selects MFCC alone (1),
MFCC + delta (2), or MFCC + delta + delta-delta (3); `portion` selects one third
(1), two thirds (2), or all retained coefficients (3). `snr_quarter_db` is in
integer quarter-decibels: `-32`, `0`, `60` mean -8, 0 and 15 dB. `frame_ms`,
`n_filters` and `trial` select the corresponding preset's allowed values.

Inspect changes with `drone-auth paper --config config/paper/my_study.json --dry-run`.
Use `--model` or `--preset` for temporary overrides; they do not edit the file.
The split, feature definitions, seeds and hyperparameters are specified in the
[frozen protocol](docs/PAPER_PROTOCOL.md); the full CLI/config contract and
restrictions are in [experiment configuration](docs/EXPERIMENTS.md#configuration-for-researchers).
For changes beyond those grids, use a single `run` configuration as described in
[the quickstart](docs/QUICKSTART.md#run-the-prepared-public-data-example) and
[numerical settings](docs/NUMERICAL_VALIDATION.md). Label such studies separately.

## Reproduce the paper experiments

The full command below computes every supported experimental figure and table
in one sweep, sharing the required clean models:

```bash
uv run --locked drone-auth paper --config config/paper/full_sweep.json --dry-run \
  > "$DRONE_EVIDENCE/full-paper-001/plan-preview.log"
uv run --locked drone-auth paper \
  --config config/paper/full_sweep.json \
  --audio-root "$AUDIO_ROOT" --output "$RUN_ROOT" \
  --cache-root "$CACHE_ROOT" --log-root "$LOG_ROOT"
```

The plan has **4,096 fits + 4,344 evaluations = 8,440 tasks**. The completed local
run took about 11 hours, with peak process RSS 1.94 GiB; it retained 30.06 GiB of
run artifacts and 26.14 GiB of cache. Provide additional space for data, environment,
logs and temporary writes. These are [observations from one host](docs/REPRODUCTION_RESULTS.md#reproducibility-and-measured-resources),
not resource guarantees. Initial feature preparation and some SVM fits can take
substantial time. A persistent terminal or the [documented `nohup` workflow](docs/LIVE_MONITOR.md#start-the-page)
keeps remote training running after SSH disconnects.

To reproduce only a particular result, use its preset in the following table.
Task counts include dependencies when that preset is run alone.

| Paper result | Preset / operation | Work and interpretation |
| --- | --- | --- |
| Figure 1 | Recording setup in the paper | Physical setup illustration; no model-generated output |
| Table 1 | Dataset validation above | Durations from WAV metadata, not predictions |
| Figure 2 | `figure2` | 408 fits: DS1 frame lengths, 20–2520 ms |
| Figure 3 | `figure3` | 3,600 fits: DS1 filter counts, feature blocks and fractions |
| Figure 4 | `figure4` | 3,600 clean fits + 3,600 evaluations at 0 dB |
| Table 2 | `table2` | 8 fits: DS2 closed-set metrics across 24 identities |
| Figure 5 | `figure5` | 8 clean fits + 744 evaluations from -8 to 15 dB |
| Table 3 | Exported by `plot` | Frozen identity partitions from `attack_partitions.json` |
| Tables 4 and 5 | `tables4_5` | 80 fits: registered recall and unknown rejection over ten trials |

For example, run and plot Figure 5 with all eight classifiers:

```bash
uv run --locked drone-auth paper --config config/paper/full_sweep.json \
  --preset figure5 --dry-run
uv run --locked drone-auth paper --config config/paper/full_sweep.json \
  --preset figure5 --audio-root "$AUDIO_ROOT" \
  --output "$DRONE_ARTIFACTS/runs/figure5-001" \
  --cache-root "$CACHE_ROOT" --log-root "$DRONE_EVIDENCE/figure5-001/attempts"
uv run --locked drone-auth plot --runs "$DRONE_ARTIFACTS/runs/figure5-001" \
  --output "$DRONE_EVIDENCE/figure5-001/figures"
```

Replace `figure5` in the preset and output/log names with the desired preset.
You can select several with `--preset figure3 figure4`. Clean fits are deduplicated
within a plan; independent sweeps fit their own models. Use the full sweep once
if you need all results. Plotting a single preset leaves unrelated panels visibly
incomplete; the selected full grid is still available. Table 3 records the
experimental design, not a measured result.

## Watch training and resume

In another terminal with the same environment/path exports, start the monitor
before or during the full run:

```bash
uv run --locked drone-auth monitor --run "$RUN_ROOT" --log-root "$LOG_ROOT" --port 8765
```

Open the complete URL printed by the monitor. For a remote Ubuntu host, forward
port **8765** in VS Code, or run this on your own computer:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@TRAINING_HOST
```

Use the printed URL's `#token` fragment in your local browser. It changes whenever
the monitor restarts. The page shows task completion, current phase, CPU, memory
and logs; it cannot estimate a model's internal fit percentage or completion time.
The monitor runs independently and does not keep a foreground trainer alive after
SSH disconnection. See [monitoring and remote sessions](docs/LIVE_MONITOR.md).

To continue an interrupted sweep, repeat the **same training command** with
`--resume`. Keep the same config, paths, source and locked environment. Verified
completed tasks are reused; unfinished fits restart from the beginning using the
cache. Warnings remain visible, including SVM warnings at the preserved iteration
limit. Completion records execution of the configured procedure, not a guarantee
of estimator convergence.

## Generate figures and tables

After the full sweep, generate the gallery from saved results:

```bash
uv run --locked drone-auth plot --runs "$RUN_ROOT" --output "$FIGURE_ROOT"
```

Open `$FIGURE_ROOT/index.html`. The gallery includes Figures 2–5, Tables 2–5,
confusion matrices and per-identity recall. Figures export as **PNG, SVG and PDF**;
tables as **Markdown, LaTeX and JSON**. `plot-data.json` retains plotted values,
`metrics.json` records source measurements and `manifest.json` records coverage
and hashes. No CSV intermediate, audio access or retraining is required. To plot
the small GNB check, use its `paper-smoke-001` run and a fresh figure output path;
its plots are explicitly marked incomplete previews.

For figures styled like the paper and a side-by-side comparison, first complete
all full-data grids and export the full gallery. The comparison helper requires
all **8,352 figure points** and the exact author-provided `ACSAC_Yufeng.pdf` revision:

```bash
# Optional comparison only: install Poppler if pdftoppm is unavailable.
sudo apt-get install poppler-utils
uv run --locked python tools/report_paper_comparison.py \
  --gallery "$FIGURE_ROOT" --paper-pdf ../ACSAC_Yufeng.pdf \
  --output "$DRONE_EVIDENCE/full-paper-001/paper-comparison"
```

Open that output's `index.html`. It restores the historical colors, line styles,
markers, panel layout and axis ranges, and places the new figures beside the
paper's figures. It also exports individual panels and full-range views where
needed. The PDF is not included in this repository and its checksum is verified;
other PDF revisions need reviewed crop coordinates. Without it, the ordinary
figure gallery is fully usable. See [figure generation and comparison](docs/FIGURES.md)
for multiple compatible runs, panel mapping and export details.

## Results and interpretation

The completed updated run continues to support the main findings in the original
evaluation setting. QDA closed-set accuracy is **96.2034%** (published: 96.20%), and
its mean registered-drone recall is **94.1884%** (published: 94.19%). QDA remains a
strong balanced authentication choice; KNN retains its lead over the sampled
low-SNR range. Across all Table 2 metrics, the largest absolute difference is
0.3306 percentage points.

The maintained implementation incorporates numerical fixes, explicit experimental
conventions and updated dependencies. Some feature-specific observations, exact
rankings and threshold boundaries consequently differ. Open-set RBF-SVM behavior
is particularly sensitive to the background-label ordering; a separate diagnostic
using the historical ordering closely recovers the published behavior. These
implementation-specific observations complement the central findings and are
reported separately so researchers can choose and cite a clear baseline.

- [Full-run report](docs/REPRODUCTION_RESULTS.md): published and updated values,
  resource measurements, repeat checks and controlled label-order diagnostics.
- [Detailed interpretation by paper claim](docs/CLAIM_ASSESSMENT.md): retained
  findings and the scope of updated feature/classifier observations.
- [Numerical changes](docs/NUMERICAL_VALIDATION.md) and
  [protocol decisions](docs/PAPER_PROTOCOL.md): confirmed fixes versus explicit
  conventions; these are not interchangeable explanations for every difference.

Published measurements, the updated primary run and diagnostics retain their own
identities. Cite the paper together with the software commit/release, lockfile and
configuration used. The evidence concerns the paper's indoor, within-recording
protocol; unseen rooms, held-out recordings and replay attacks remain outside
this rerun's scope. Exact historical numerical identity is not a prerequisite for
using the maintained implementation, and is not claimed for every configuration.

## Maintained and original versions

Use the workflow above for new experiments. The original public implementation
is preserved on the public [`legacy/acsac2022` branch](https://github.com/Yufeng-Diao/Drone-Authentication-via-Acoustic-Fingerprint/tree/legacy/acsac2022), at the fixed
[pre-refactor commit `88d0098`](https://github.com/Yufeng-Diao/Drone-Authentication-via-Acoustic-Fingerprint/tree/88d0098b189b8cb48da3ec05ab81f7890b36d6b5).
Its original code and documentation remain unchanged in that snapshot. `main`
provides the maintained workflow, with the original and refactoring commit history
preserved. See [release status](docs/RELEASE_STATUS.md) for remote CI and the
Docker image baseline.

The historical snapshot supports inspection and comparison using its original
entry points. Legacy scripts, YAML configurations, the Conda environment and
generated-output placeholder folders have been removed from the maintained tree.
Use `drone-auth` for the workflow above; use the
[historical workflow reference](docs/LEGACY_WORKFLOW.md) to inspect the original release.

The maintained source tree is organized as follows:

| Path | Purpose |
| --- | --- |
| `src/drone_fingerprint/` | CLI, features, models, experiment execution, monitoring and plotting |
| `toolbox/` | Shared MFCC numerical core and its license |
| `config/paper/`, `config/smoke/`, `config/validation/` | Frozen paper protocol and executable JSON configurations |
| `examples/`, `docs/` | Research extensions and usage guides |
| `tests/`, `tools/` | Regression tests, validation and evidence-generation utilities |

Store audio, caches, models, results and logs outside this tree using the paths
configured above.

## Further guides

- [Full experiment and release status](docs/RELEASE_STATUS.md)
- [Data cache design and resource measurements](docs/DATA_PIPELINE.md)
- [Model compatibility, including the historical QDA SVD fit](docs/MODEL_MIGRATION.md)
- [Modify features and figures, with executable examples](docs/DEVELOPMENT.md)
- [Tests and code map](docs/QUICKSTART.md#tests-and-code-map)

## License

Project code is Apache-2.0. The reused MFCC core retains its
[MIT license](toolbox/LICENSE_python_speech_features); the QDA compatibility fit
retains [scikit-learn BSD-3-Clause attribution](LICENSE_sklearn_QDA).
The dataset is distributed separately under CC BY 4.0.
