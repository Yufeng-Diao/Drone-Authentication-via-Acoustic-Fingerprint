# Modern installation and a small closed-set run

P1 provides a real WAV-to-result workflow with base MFCCs and one classifier.
It does not yet implement noisy sweeps, deltas, open-set evaluation, shared caches,
or complete paper plots. Use the [frozen protocol](PAPER_PROTOCOL.md) for the
scientific specification. Python 3.8 and the old Conda environment are unnecessary.

The tested primary interpreter is CPython **3.14.8**, with **3.13.16** as the
compatibility target. Both use ordinary GIL-enabled builds. `uv.lock` contains
the resolved versions and distribution hashes; `.python-version` selects the
primary interpreter. Python 3.15 and free-threaded builds are outside this range.
Linux is tested locally. The Linux/Windows CI matrix in `.github/workflows/ci.yml`
is provided for verification after publication; Windows is not yet a verified
platform. macOS is also unverified.

## Install with uv

Install **uv 0.13.0** using the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
For example, with an existing Python/pip installation, run
`python -m pip install --user uv==0.13.0` and ensure its scripts directory is on
PATH. The project's uv version requirement rejects a different uv version.
Run the following from the repository root. Keep data, environments, caches,
outputs and logs in sibling directories. `uv sync --locked` must not rewrite
the lockfile. Its default editable install is useful when changing the code.

Linux/macOS shell (macOS commands are documented but unverified):

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/../refactor_artifacts/environments/default"
export UV_CACHE_DIR="$PWD/../refactor_artifacts/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/../refactor_artifacts/environments/python"
export MPLCONFIGDIR="$PWD/../refactor_artifacts/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
uv sync --locked
uv run --locked drone-auth --help
uv run --locked drone-auth synthetic --output ../refactor_artifacts/runs/synthetic-001
```

Windows PowerShell:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$((Get-Location).Path)/../refactor_artifacts/environments/default"
$env:UV_CACHE_DIR = "$((Get-Location).Path)/../refactor_artifacts/cache/uv"
$env:UV_PYTHON_INSTALL_DIR = "$((Get-Location).Path)/../refactor_artifacts/environments/python"
$env:MPLCONFIGDIR = "$((Get-Location).Path)/../refactor_artifacts/cache/matplotlib"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONPATH = ""
uv sync --locked
uv run --locked drone-auth --help
uv run --locked drone-auth synthetic --output ../refactor_artifacts/runs/synthetic-001
```

Use a new run directory each time. Existing directories are rejected to preserve
earlier results. Output inside a Git checkout, Python source project or audio
directory is rejected. The synthetic command creates four 20-second WAV fixtures
and a small model run. Its accuracy checks installation behavior only.

The explicit Python-path isolation prevents unrelated ROS, Conda, or user-site
packages from entering this environment. One P1 test attempt exposed ROS pytest
plugins inherited from the host shell; it was resolved by isolating the process,
not by adding ROS dependencies to this project.

## Run the prepared public-data example

Download and extract the [CC BY 4.0 dataset](https://doi.org/10.5525/gla.researchdata.1348)
to an external directory with `1to8`, `9to16`, `17to24` and `noise` subdirectories.
Validate the complete dataset using `tools/validate_p0.py` as described in the
protocol guide. The run itself rechecks each selected WAV's SHA-256, size and
format. It never modifies the original WAV files.

```bash
uv run --locked drone-auth run \
  --config config/smoke/closed_set.json \
  --audio-root ../refactor_artifacts/data/acsac2022/extracted \
  --output ../refactor_artifacts/runs/public-smoke-001
```

In PowerShell, put the same arguments on one line or use PowerShell's backtick
line continuation. Manifest paths in JSON resolve relative to the config file;
CLI data and output paths resolve relative to the working directory.

This configuration projects the P0 fixed validation subset onto identities
1 and 2, retaining both distances and complete recordings: four WAV files in
total. It uses 1-second frames, 50% overlap, 50 filters, 49 retained coefficients,
no deltas, final L2 normalization and GaussianNB. Splitting precedes framing and
pre-emphasis. The sample intervals remain the frozen 70/30 protocol. This is a
small installation/workflow check, not the paper's 24-class experiment or a
recording-held-out generalization test.

The JSON configuration accepts exactly the documented fields. Unknown options,
unsupported feature settings, stale manifests and incomplete distance pairs
fail explicitly. Change `model` to select another paper estimator, but recognize
that the P1 real-data smoke evidence covers GNB; the eight-model fit acceptance
belongs to P4. Change `frame_ms`, `n_filters` or `identities` for a small clean
base-MFCC study and use a fresh output directory. Deltas, AWGN and partial
coefficient selection are not accepted as silent no-op options.

## What the run writes and displays

| File | Contents |
| --- | --- |
| `result.json` | Completed-run status, resolved settings, manifest hashes, exact package versions, effective model parameters, warning records, sample intervals, row counts, timings, metrics and confusion matrix |
| `predictions.npz` | Truth/prediction integer arrays; load with `numpy.load(..., allow_pickle=False)` |
| `model.joblib` | The model trained by this run; load only trusted artifacts in the recorded environment |
| `failure.json` | Written after an execution failure; a failed run has no successful `result.json` |

No intermediate audio or feature files are generated. One recording's samples
are held at a time; compact feature rows accumulate in memory. P1 reads a selected
WAV for checksum verification and again for decoding. Read reduction and cache
reuse are P3 work and have not been measured here.

The terminal prints each recording and the current stage. Operations lasting
more than 10 seconds emit elapsed-time and process-CPU heartbeats. There is no
wall-time cutoff and no invented fit percentage. SVM iteration budgets remain
1000; warnings are printed and recorded. The P1 heartbeat test uses a silent
operation; an actual slow estimator trace and recovery behavior remain P4 work.

Capture terminal logs outside the source tree, for example with shell redirection
to a new file under `../refactor_evidence/`. Timing values are individual smoke
observations and exclude import/startup from the per-stage figures; they are not
benchmark speedups or estimates of the full 4,096-fit workload.

## Standard venv/pip installation from the same lock

Export runtime requirements and build the project, directing generated files
outside the repository:

```bash
uv export --locked --no-dev --no-emit-project --format requirements-txt \
  --output-file ../refactor_artifacts/runtime-requirements.txt
uv build --out-dir ../refactor_artifacts/dist
python3.14 -m venv ../refactor_artifacts/environments/pip-example
../refactor_artifacts/environments/pip-example/bin/python -m pip install \
  --require-hashes -r ../refactor_artifacts/runtime-requirements.txt
../refactor_artifacts/environments/pip-example/bin/python -m pip install --no-deps \
  ../refactor_artifacts/dist/drone_acoustic_fingerprint-0.1.0-py3-none-any.whl
../refactor_artifacts/environments/pip-example/bin/python -I -B -m drone_fingerprint.cli \
  synthetic --output ../refactor_artifacts/runs/pip-synthetic-001
```

Use the pinned 3.14.8 interpreter for the primary installation; 3.13.16 can be
checked separately. On Windows use `py -3.14 -m venv ...` and
`.../Scripts/python.exe` instead of the POSIX interpreter paths. This export is
generated from `uv.lock`; do not maintain a second dependency version list.
The wheel includes the reused MFCC core and both project/third-party licenses.
It does not include audio, models, execution logs or paper results.

## Tests and code map

Create the external temporary parent before using pytest's `--basetemp`:

```bash
mkdir -p ../refactor_artifacts/test-tmp ../refactor_evidence/local-tests
uv run --locked python -I -B -m pytest \
  --basetemp ../refactor_artifacts/test-tmp/check-001 \
  --junitxml ../refactor_evidence/local-tests/check-001.xml
uv run --locked python -I -B tools/audit_models.py \
  > ../refactor_evidence/local-tests/model-defaults.json
```

Pytest recreates its chosen temporary directory; choose a dedicated test-only
path. Its source-tree cache is disabled. To exercise 3.13 independently, point
`UV_PROJECT_ENVIRONMENT` at another external environment and supply
`--python 3.13.16` to both `uv sync --locked` and `uv run --locked`.

The direct code path is `cli.py` → `config.py` / `data.py` → `features.py` →
`models.py` → `workflow.py`. These modules live in `src/drone_fingerprint`.
`features.py` calls the original `toolbox/mfcc_base.py` and `mfcc_sigproc.py`;
their numerical core and attribution remain intact. `toolbox/__init__.py` now
loads the legacy `CMatrix` export on demand so numerical imports do not pull in
unrelated plotting packages. The historical scripts remain reference code and
are not certified against the new minimal dependency set.
