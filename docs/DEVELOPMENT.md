# Extend the research workflow

Use an editable locked installation from [the quickstart](QUICKSTART.md).
The maintained implementation uses plain functions and dataclasses. The
historical `experiment/`, `runners/` and legacy toolbox wrappers are reference
code, not the maintained CLI. No plugin framework is needed to modify a study.

| Research question | Start here |
| --- | --- |
| Which recordings and split intervals? | `config/paper/`, `data.py`, `config.py` |
| Which coefficients and derivative context? | `features.py`, `toolbox/mfcc_base.py`, `toolbox/mfcc_sigproc.py` |
| Which cache keys and bounded reads? | `pipeline.py`, `data.py` |
| Which estimator and parameters? | `models.py`, `qda.py` |
| How are metrics defined? | `metrics.py` |
| Which task grid, dependencies and recovery? | `experiments.py`, `workflow.py`, `artifacts.py` |
| How are figures assembled? | `results.py`, `plotting.py` |

Python modules above live in `src/drone_fingerprint`, except the retained MFCC
core. Read [the numerical contract](NUMERICAL_VALIDATION.md) before changing a
scientific definition. Keep frozen paper manifests intact for comparisons;
give a changed protocol its own configuration and record why it differs.

## Example: compare feature blocks

`examples/custom_features.py` changes a `Settings` record with `dataclasses.replace`
and calls the existing `run_experiment` function. It compares base, delta and
delta-delta blocks on the same four public WAVs. Feature extraction, split rules,
normalization, cache reuse and metrics remain in the shared implementation.

```bash
uv run --locked python -I -B examples/custom_features.py \
  --config config/smoke/closed_set.json \
  --audio-root ../refactor_artifacts/data/acsac2022/extracted \
  --output ../refactor_artifacts/runs/custom-features-001 \
  --cache-root ../refactor_artifacts/cache/features
```

Inspect `comparison.json`: feature widths are 49, 98 and 147. Each child output
also records predictions, metrics and model provenance. This small example is
not a paper benchmark or a claim that adding derivatives improves accuracy.
Do not pass its arbitrary research runs to the paper plot command; that command
intentionally accepts only verified frozen paper sweep tasks.

## Example: customize a saved series

After generating a [gallery](FIGURES.md), customize its Figure 2 GNB series:

```bash
uv run --locked python -I -B examples/plot_one_series.py \
  --gallery ../refactor_evidence/paper-gallery-001-figures \
  --output ../refactor_evidence/custom-series-001
```

The example reads `plot-data.json`, preserves missing values as gaps, labels
units and scope, and writes PNG/SVG/PDF. It does not access audio or models.
Modify this short script for presentation changes; modify `results.py` only when
the mapping from experiment metadata to scientific output must change.

## Verify a change

Run the tests with external temporary/JUnit paths as documented in the
quickstart. Choose tests matching the change: feature changes need numerical
fixtures and cache invalidation checks; metric changes need known labels;
plot changes need saved-result fixtures, grid/panel/unit checks and headless
rendering. Keep artifact validation and failure histories intact. Never suppress
convergence warnings or fabricate a training percentage.

Review names, units, docstrings, the public CLI and one realistic example.
Tests cannot prove readability. Keep comments focused on scientific reasons;
prefer a direct function over an additional abstraction. All maintained code,
documentation and generated evidence use English.
