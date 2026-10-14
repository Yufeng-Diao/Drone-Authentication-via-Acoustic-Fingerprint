# Drone Authentication via Acoustic Fingerprint

Maintained research code for the [ACSAC 2022 paper](https://doi.org/10.1145/3564625.3564653)
by Yufeng Diao, Yichi Zhang, Guodong Zhao and Mohamed Khamis. The maintained
workflow runs on **Ubuntu**, with **Python 3.14.8** and **3.13.16** compatibility
checks. Exact scientific dependencies are recorded in `uv.lock`.

The pipeline reads the original WAV files, extracts MFCC features in bounded
batches, trains the eight paper classifiers, and generates figures and tables
from saved results. It uses shared NumPy arrays instead of per-frame CSV/PKL
files. Slow phases show elapsed time, CPU and memory; interrupted paper tasks
restart in new attempts while verified completions are reused.

## Relationship to the published paper

The updated implementation continues to support the paper's central feasibility
findings and QDA as a strong balanced choice for the evaluated closed-set and
open-set tasks. Some feature-extraction findings, exact rankings and numerical
thresholds have changed. The current results do not reproduce every published
number or conclusion.

The full rerun completed all **8,440 tasks**. Confirmed bug fixes, changes to
experimental conventions and dependency updates are documented separately.
Published values, frozen rerun results and controlled diagnostics remain distinct.
The evidence applies to the paper's indoor, within-recording evaluation; it does
not establish unseen-room, held-out-recording or replay-attack robustness.

Read the detailed reports:

- [Claim-by-claim assessment](docs/CLAIM_ASSESSMENT.md): which conclusions remain
  supported, which need qualification, and the evidence for each judgment.
- [Full-run results and scientific differences](docs/REPRODUCTION_RESULTS.md):
  measured values, reproducibility checks, resources and label-order diagnostics.
- [Numerical corrections](docs/NUMERICAL_VALIDATION.md) and
  [experimental conventions](docs/PAPER_PROTOCOL.md): what changed and why.
- [Figures and direct paper comparisons](docs/FIGURES.md): generate the figure
  gallery and compare the new measurements with the author-provided paper PDF.

When reporting experiments, cite the paper and identify the software release or
commit, dependency lock and configuration used. Label baselines by their actual
implementation version rather than treating new measurements as published values.

## Maintained and original versions

Use the workflow below for new experiments. The original public implementation
is preserved separately as `legacy/acsac2022`, at the fixed
[pre-refactor commit `88d0098`](https://github.com/Yufeng-Diao/Drone-Authentication-via-Acoustic-Fingerprint/tree/88d0098b189b8cb48da3ec05ab81f7890b36d6b5).
Its original code and documentation remain unchanged in that snapshot. The branch
is prepared locally; remote publication and the transition of the maintained
implementation to `main` are tracked in [release status](docs/RELEASE_STATUS.md).

The historical snapshot is for inspection and comparison. Its old environment
and entry points are outside the maintained Ubuntu workflow, and preserving the
snapshot does not certify exact recovery of the original paper execution.
See the [historical workflow reference](docs/LEGACY_WORKFLOW.md).

## Start here

Install **uv 0.13.0**, clone this repository, and run these commands from its root.
The [quickstart](docs/QUICKSTART.md) also covers standard venv/pip installation.

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/../refactor_artifacts/environments/default"
export UV_CACHE_DIR="$PWD/../refactor_artifacts/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/../refactor_artifacts/environments/python"
export MPLCONFIGDIR="$PWD/../refactor_artifacts/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=""
uv sync --locked
uv run --locked drone-auth synthetic --output ../refactor_artifacts/runs/synthetic-001
```

Choose fresh output names; commands preserve previous results. Keep environments,
models, caches, figures and logs outside the source checkout. Synthetic accuracy
checks installation only and is not a paper measurement.

## Public data and paper experiments

Download the [CC BY 4.0 public dataset](https://doi.org/10.5525/gla.researchdata.1348).
The [data guide](docs/QUICKSTART.md#run-the-prepared-public-data-example) explains
extraction and verification. The frozen manifests identify 257 WAV files; the
paper uses 240 selected drone recordings and one room-noise recording.

```bash
uv run --locked drone-auth paper --config config/validation/paper_smoke.json --dry-run
uv run --locked drone-auth paper \
  --config config/validation/paper_smoke.json \
  --audio-root ../refactor_artifacts/data/acsac2022/extracted \
  --output ../refactor_artifacts/runs/paper-smoke-001 \
  --cache-root ../refactor_artifacts/cache/features \
  --log-root ../refactor_evidence/paper-smoke-001
uv run --locked drone-auth plot \
  --runs ../refactor_artifacts/runs/paper-smoke-001 \
  --output ../refactor_evidence/paper-smoke-001-figures
```

Open the generated `index.html`. The gallery exports Figures 2–5, Tables 2–5,
confusion matrices and per-identity recall diagnostics as applicable. Missing
results are visible; subset figures are explicitly labelled incomplete previews.
No training runs during plotting. See [figures and tables](docs/FIGURES.md) for
multiple runs, units, panel mapping and Markdown/LaTeX exports.

## Guides for researchers

- [Full experiment and release status](docs/RELEASE_STATUS.md)
- [Live training webpage and SSH port forwarding](docs/LIVE_MONITOR.md)
- [Experiment selection, progress and recovery](docs/EXPERIMENTS.md)
- [Frozen paper protocol](docs/PAPER_PROTOCOL.md) and [numerical corrections](docs/NUMERICAL_VALIDATION.md)
- [Model compatibility decisions](docs/MODEL_MIGRATION.md), including the historical QDA SVD fit
- [Data cache and resource measurements](docs/DATA_PIPELINE.md)
- [Modify features and figures](docs/DEVELOPMENT.md), with executable examples
- [CPU container](docs/CONTAINER.md), using the same Python and dependency lock
- [Historical workflow reference](docs/LEGACY_WORKFLOW.md), for the original release only

The full plan contains 4,096 fits plus 4,344 evaluations reusing clean models.
Use `--dry-run` before selecting it. See [release status](docs/RELEASE_STATUS.md)
for the distinction between completed local validation and pending publication.

## License

Project code is Apache-2.0. The reused MFCC core retains its
[MIT license](toolbox/LICENSE_python_speech_features); the QDA compatibility fit
retains [scikit-learn BSD-3-Clause attribution](LICENSE_sklearn_QDA).
The dataset is distributed separately under CC BY 4.0.
