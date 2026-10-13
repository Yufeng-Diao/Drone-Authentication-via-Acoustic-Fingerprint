# Corrected numerical workflow (P2)

P3 retains these numerical definitions while changing data access and storage.
For current cache behavior, verification and atomic completion, read the
[data pipeline guide](DATA_PIPELINE.md). The P2 baseline section below describes
the archived pre-optimization implementation.

The maintained entry point is `drone-auth`. Ubuntu 24.04, CPython 3.14.8 and
3.13.16, and the existing `uv.lock` are the acceptance targets. The historical
`runners/`, `experiment/`, and PKL/CSV generation scripts remain reference code;
they have not been repaired or certified for the new environment. In particular,
do not use their filename checks, noise generation or derivative calculations
as the corrected workflow. The shared MFCC numerical core and its attribution
remain intact.

## Reproducible small runs

Install as described in [QUICKSTART.md](QUICKSTART.md), then run from the checkout:

```bash
uv run --locked drone-auth run \
  --config config/validation/closed_delta.json \
  --audio-root ../refactor_artifacts/data/acsac2022/extracted \
  --output ../refactor_artifacts/runs/delta-001 --save-features
```

`closed_delta.json` uses identities 1 and 2, both distances, three feature blocks,
50 filters and two-thirds of the 49 available coefficients: 32 columns per block,
96 total. `closed_noisy.json` adds 0 dB AWGN to test samples only. Both use the
complete recordings from the frozen validation subset and GaussianNB.

`open_set_trial_01.json` selects the first frozen Table 3 partition through
`partitions_manifest` and `trial`. It uses all 24 identities from the validation
subset, 201 filters, base MFCCs and GaussianNB. Registered identities retain their
class labels; background drones and room noise train label 0. Only registered
and attack drone test intervals enter evaluation. Attack labels map to 0, while
`predictions.npz` retains `original_identity` for per-identity rejection metrics.
The validation subset explicitly uses the first 60 seconds of room noise. A
full-protocol subset must declare `[0, noise_frame_count]`; the runner does not
silently split or cap this recording. Full dataset preset orchestration is available through the [P4 paper command](EXPERIMENTS.md).

All examples are implementation validation, not paper accuracy benchmarks.
They preserve the paper's within-recording 70/30 intervals, not recording-held-out
generalization. The real validation currently exercises GNB fitting; all eight
constructors are audited, and P4 validates all eight fits on the fixed subset;
see [model compatibility](MODEL_MIGRATION.md).

## Configuration and feature shape

The required schema-1 fields from P1 remain valid. Additional fields are strict:

| Field | Accepted values | Meaning |
| --- | --- | --- |
| `blocks` | 1, 2, 3; default 1 | Base, base + delta, base + delta + delta-delta |
| `portion` | 1, 2, 3; default 3 | Fraction `portion / 3` of coefficients after dropping coefficient 0 |
| `test_snr_quarter_db` | Integer -32 through 60, or null | Requested SNR is this integer divided by 4; null means clean |
| `trial`, `partitions_manifest` | Both supplied, or both absent | Load exactly one numbered 8/8/8 partition from strict schema-1 JSON |

For `F` filters, each block has `floor((F - 1) * portion / 3)` columns. Blocks are
concatenated in base/delta/delta-delta order and the final vector is L2-normalized.
Zero vectors remain zero. Example per-block counts:

| Filters | One third | Two thirds | All |
| --- | --- | --- | --- |
| 26 | 8 | 16 | 25 |
| 50 | 16 | 32 | 49 |
| 201 | 66 | 133 | 200 |
| 271 | 90 | 180 | 270 |

`feature_metadata` in `result.json` records both extracted and selected counts,
framing, FFT, filter band, pre-emphasis, window, DCT, lifter, selection and
normalization semantics. Derivatives are recursively computed with neighborhood
width 1 and repeated edge values, inside each continuous interval. Pre-emphasis,
framing and final-frame padding also stay inside that interval. There is no
borrowing across train/test gaps, recordings or room-noise intervals.

## Numerical changes and historical comparisons

| Issue | Maintained behavior and evidence |
| --- | --- |
| B1: int16 overflow in noise power | Cast to float64 before squaring. A +/-1000 input has power 1,000,000; historical int16 squaring gives 16,960. Seed a local PCG64 generator using master seed, recording SHA-256 and integer quarter-dB SNR. Use complete-recording power and noise timeline, select test intervals afterward, and never clip or round back to PCM. Zero/nonfinite power fails. |
| B2: filename parameter checks | No filename-based feature matching. Verify manifests, complete pairs, WAV content and structured model metadata; report mismatch, missing or corrupt artifacts explicitly. Both closed/open-set workflows share these checks. |
| B3: derivative order | `delta(base, 2)` is a wider first derivative. Use `delta(delta(base, 1), 1)` for acceleration. Hand references cover constant and linear sequences and replicated edges. |
| B4: segment boundaries | Extract/derive each continuous interval separately. Two different constant segments must have zero derivatives, including their boundaries. |
| B5: normalization | Drop coefficient 0, select, concatenate, then normalize the final row. Historical PKL and direct paths normalize blocks at different times. These are scientific convention changes, not guaranteed accuracy improvements. |
| B6: fractions | Count available coefficients after removal and use half-open slicing. Actual extraction is checked for all 36 filter/fraction/block combinations above. |
| B7: repeated rows / unrecorded identity | One immutable directory and JSON completion marker per task, explicit seeds and resolved partition. Stable task identity binds inputs, settings, source hashes, runtime and snapshot option. Checked completed-task reuse is supported; interrupted training recovery is provided by the P4 `paper --resume` command. |
| B8: room noise | Room noise is a declared training-only interval. Background drones contribute only training intervals; attack drones contribute only test intervals. Original drone identities remain available. |
| B9: F1 equation | Use standard per-class `2PR/(P+R)`, then macro-average with `zero_division=0`. Equation (4), printed page 663, shows its reciprocal; this does not establish that the paper's table used the wrong formula. |
| B10: boundaries and model identity | Validate mono little-endian PCM16 at 44.1 kHz, size, checksum, frame count, paths and configuration. Errors exit nonzero. Model loading validates complete feature and training configuration, even for equal column counts. |

Open-set output includes registered macro recall, attack macro rejection, each
identity's recall/rejection, and separately labelled pooled attack rejection.
The latter weights identities by their sample counts and can differ from the
primary macro result. Collapsed-class accuracy and macro P/R/F1 are also reported,
but do not replace those open-set metrics. No background or room-noise test rows
are permitted. Trial aggregation into paper tables belongs to P5.

Requested SNR is defined over the full recording. A particular held-out interval
can have a different measured SNR because its signal power varies. Random noise
is not rescaled to force an exact empirical SNR. The regression tolerance is
0.05 dB on one million samples and the recorded full-timeline seed makes the
measurement repeatable. Global NumPy random state is unaffected.

## Completed-task reuse and model loading

Repeat the same `run` command with `--resume` to reuse a matching complete task.
The runner verifies task identity and model/prediction/snapshot hashes, then
validates raw WAVs through the P3 recording index (or fresh hashes with
`--verify-inputs`) before returning; it does not fit again or rewrite artifacts. Settings,
partition, source or runtime changes require a new directory. Incomplete and
corrupt runs fail, and are not silently resumed. No directory is overwritten.
The full task ID includes evaluation noise; the same clean trained model may
still be loaded for a different SNR using the checked loader:

```python
from pathlib import Path
from drone_fingerprint.artifacts import load_model
from drone_fingerprint.config import load_config
from drone_fingerprint.features import extract_segment

settings, _, _ = load_config(Path("config/validation/closed_noisy.json"))
model = load_model(Path("../refactor_artifacts/runs/delta-001"), settings)
# samples must already be one continuous mono 44.1 kHz test interval.
# predictions = model.predict(extract_segment(samples, settings))
```

Only load trusted local joblib files in the recorded dependency environment.
Checksums detect accidental corruption, not malicious serialized objects.
`load_model` compares feature and training settings, including labels, partition,
model choice and seed; evaluation SNR is deliberately excluded from that check.
P1 model directories lack the new metadata and require a fresh P2 run.

## Corrected baseline for P3

Default runs write only `model.joblib`, `predictions.npz` and `result.json`.
`--save-features` additionally writes a single uncompressed `features.npz` with
`train`, `test`, `train_labels`, and `test_labels`, using float64 features. Load
arrays with `numpy.load(path, allow_pickle=False)`. This opt-in validation snapshot
is not a cache and is never automatically used as an input. No CSV/PKL feature
tables or sliced WAVs are generated. A failed run writes `failure.json`; a success
marker is written last. Atomic writes and reusable feature caching belong to P3.

P2 intentionally retains the unoptimized read path: hash each selected complete
WAV, read its payload again, and convert to float64 before interval selection.
Even the 60-second noise subset currently decodes the complete 3,628-second room
recording. This makes its memory cost visible for the P3 baseline. Do not describe
P2 memory/I/O measurements as optimized or as evidence about hardware failure.

Generate an external visual numerical comparison with:

```bash
uv run --locked python -I -B tools/validate_numerics.py \
  --config config/smoke/closed_set.json \
  --audio-root ../refactor_artifacts/data/acsac2022/extracted \
  --output ../refactor_evidence/numerical-check-001
```

This writes JSON measurements and PNG/SVG figures for SNR, derivatives,
dimensions and normalization, plus a sequential correction comparison on the
8-identity fixed subset. The comparison replays historical operations on the
current stack in frozen manifest order. Effects depend on correction order;
they are not independent causal estimates or recovered historical results.
Numeric correctness is the acceptance criterion, not an increase in accuracy.
Tests also compare the retained MFCC core against an explicit single-frame FFT,
mel-triangle and matrix-DCT reference, and exercise silence, short padding,
nonfinite audio, strict exits, corrupt artifacts and idempotent reuse.
