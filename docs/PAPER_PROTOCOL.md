# Frozen paper protocol

P0 establishes the data and experimental specification for **acsac2022-modern-v1**.
It does not implement the new training pipeline or claim to reproduce the paper's
numerical results. The legacy entry points do not consume these JSON files yet.
P1 provides a locked modern environment; see the [quickstart](QUICKSTART.md).
The platform target is Ubuntu. P2 implements and validates corrected numerical
rules for individual closed/open-set tasks; see [numerical validation](NUMERICAL_VALIDATION.md).
P4 connects the complete experiment runner to these presets; see
[experiment execution and recovery](EXPERIMENTS.md). The frozen JSON status text
records the P0 snapshot, not current implementation progress. These stages do
not establish full paper reproduction. P4 also retains the historical QDA SVD
fit through an [explicit compatibility implementation](MODEL_MIGRATION.md#p4-qda-fit-compatibility).

The scientific reference is [Drone Authentication via Acoustic Fingerprint](https://doi.org/10.1145/3564625.3564653),
ACSAC 2022. Printed page numbers below refer to that paper. Historical implementation
references use public revision `88d0098b189b8cb48da3ec05ab81f7890b36d6b5`.

## Files to read and edit

| File | Purpose |
| --- | --- |
| [protocol.json](../config/paper/protocol.json) | Six experiment presets, feature contract, split, noise, metrics, and classifier migration constraints |
| [recordings.json](../config/paper/recordings.json) | Portable filenames, SHA-256 hashes, sample counts, identity labels, distance pairs, and historical date membership |
| [identities.json](../config/paper/identities.json) | P3 adds the Table 1 body/propeller relation, joined by identity without changing the frozen recording manifest |
| [validation_subset.json](../config/paper/validation_subset.json) | Fixed validation subset bound to the recording manifest's hash |
| [full_subset.json](../config/paper/full_subset.json) | P4 full selection: 240 drone recordings and the entire room-noise interval |
| [full_sweep.json](../config/paper/full_sweep.json) | Executable complete paper plan; use `--dry-run` before starting a full sweep |
| [attack_partitions.json](../config/paper/attack_partitions.json) | All ten Table 3 partitions; do not resample them |
| [table1_durations.json](../config/paper/table1_durations.json) | Printed Table 1 values and comparison tolerance |
| [validate_p0.py](../tools/validate_p0.py) | Standard-library checks for the specification and downloaded WAV files |
| [report_p0.py](../tools/report_p0.py) | Offline HTML and PNG/SVG figures from an existing P0 evidence bundle |

These are versioned research inputs, not execution logs. Generated measurements,
figures, logs, data, features, and models belong outside the repository. Paths in
the manifest are relative to the extracted audio root. No private source files,
absolute workstation paths, audio files, or model results are included here.

## Dataset and fixed validation subset

Use the [published dataset](https://doi.org/10.5525/gla.researchdata.1348), licensed
CC BY 4.0. The archive SHA-256 is recorded in `recordings.json`. The archive contains
256 drone WAV files and one room-noise WAV, all mono 16-bit PCM at 44.1 kHz.

The date selection in historical `toolbox/name_set.py` retains 240 drone files:
five distance pairs for each of 24 identities. Their total duration is
29,285.791746 seconds, matching Table 1's 29,285.79 seconds. Sixteen additional
files dated 2022-02-27 remain in the manifest with `legacy_date_selected: false`.
They are retained on disk but excluded from the paper presets. Room noise is
3,628.024399 seconds and is accounted for separately. The printed per-distance
totals differ slightly from conventional rounding of the measured totals; the
validator exposes every difference and allows at most 0.01 seconds per value.

The validation subset selects the lexicographically first historical `pair_key`
for every identity, retaining both distances and each complete drone recording:
48 drone files plus the first 60 seconds of the room-noise file. Selection is by
metadata, never by model performance. It is not a paper benchmark. No sliced WAV
copies are created. Future subset runs must read intervals directly from the
original files and record this subset's name and manifest hash.

For each drone recording with N samples, boundaries are `floor(N*p/100)` for
p = 0, 20, 35, 65, 80, 100. Alternating intervals are train, test, train, test,
train. Endpoints are exclusive. This retains the historical 70/30 time split,
with at most sample-rounding differences, and identical boundaries for both
distances. The paper specifies 70/30; these exact intervals come from the code.
Frames, pre-emphasis, and deltas must be computed within each continuous interval.
Pair names and sample counts do not independently establish acoustic alignment.
This protocol does not measure recording-held-out or date-held-out generalization.

## Decisions and their scientific status

The following choices resolve ambiguity for the modern implementation. They are
explicit interpretations or corrections, not claims about unspecified paper
details. Changes require a protocol revision and invalidation of dependent caches
and results. P2 must compare relevant legacy and corrected behavior.

| ID | Evidence / issue | Frozen decision | Validation still required |
| --- | --- | --- | --- |
| D01 | Section 3, p. 661; historical date filter | Use the 240-file historical selection for DS1/DS2 and both distances. DS1 contains identities 1–8; DS2 contains 1–24. | P0 validates membership, hashes, coverage and Table 1 durations. |
| D02 | Paper states 70/30; code provides interval positions | Preserve the alternating intervals above; split before signal processing. | P0 validates integer interval coverage. P2 checks actual frames and delta boundaries. |
| D03 | `mfcc_base.py`, `mfcc_sigproc.py`, `MFCC_extract.py` | Retain Hamming, pre-emphasis 0.97, per-window maximum-absolute-value normalization after windowing, power-of-two FFT, 0–8 kHz filters, orthonormal DCT-II, lifter 22, energy replacement, round-half-up framing and segment-local zero padding. | P2 reference calculations, including silence and short segments. |
| D04 / B3–B4 | Historical `delta(base, 2)` changes window width, not derivative order; concatenation crosses segments | Use `delta(delta(base, 1), 1)` for the second derivative, with edge replication inside each continuous segment. | P2 constant/linear sequences and discontinuity regression cases. |
| D05 / B5 | Historical direct and PKL paths normalize differently | Compute base/delta blocks; remove each first coefficient; select coefficients; concatenate in base/delta/delta-delta order; L2-normalize the final vector. Zero vectors remain zero. | P2 norm, ordering and historical-path comparison. |
| D06 / B6 | Paper's fractional coefficient selection lacks exact indexing rules | From F filters, retain floor((F−1)×fraction) coefficients per block, using half-open slices after coefficient 0 is removed. Fractions are 1/3, 2/3, 1. | P0 records expected dimensions; P2 validates actual extraction dimensions. |
| D07 / B8 | General preprocessing splits room noise; dedicated attack preprocessing uses all of it | Train the background class on the entire room-noise recording and training intervals of background drones. Test only registered/attack drone test intervals. The validation subset alone caps room noise at 60 seconds. | P2/P4 label provenance and train/test membership checks. |
| D08 | Table 5 averaging language and historical collapsed-label calculation differ | Report per-attack-identity rejection recall, macro attack recall as primary, and pooled attack recall as the historical comparison. Table 4 uses macro recall over eight registered identities. Preserve original identities before assigning background label 0. Average trials equally. | P2 unequal-class-size metric fixtures; P5 distinctly labelled comparison columns. |
| D09 / B9 | Equation (4), p. 663, prints reciprocal F1; legacy evaluation uses sklearn reports | Use standard per-class F1, then macro-average. Undefined precision/recall/F1 is zero and must be disclosed. Do not infer that the paper's numerical table used the printed equation. | P2 hand-computed metric fixtures. |
| D10 / B1 | Legacy int16 squaring overflows; unrecorded noise randomness | Compute power in float64 over the full clean recording. Generate floating-point AWGN with recorded PCG64 seed derived from master seed 2022, recording SHA-256 and integer quarter-dB SNR. Reuse noise across model/feature comparisons. Never clip to int16. Zero-power SNR input fails explicitly. | P2 power/SNR/reproducibility checks. |
| D11 | Historical model constructors contain both explicit and implicit defaults | Preserve explicit settings; map RF classifier `auto` to `sqrt`; retain DT seed 1 and assign seed 2022 to previously unseeded stochastic models. Audit all implicit defaults against the P1 locked environment before fitting. | P1 dependency/default audit; P4 constructor and convergence records. |
| D12 | Some estimators spend a long time initializing or fitting | Keep SVM `max_iter=1000` and convergence warnings visible. Silence is not a timeout or failure signal. Provide low-frequency elapsed-time/liveness/resource heartbeats; any wall-time budget must be explicit. | P4 slow-fit transcript and interruption/resume checks. |

Expected coefficient counts per block (specification only):

| Filters | One third | Two thirds | All |
| --- | ---: | ---: | ---: |
| 26 | 8 | 16 | 25 |
| 50 | 16 | 32 | 49 |
| 201 | 66 | 133 | 200 |
| 271 | 90 | 180 | 270 |

Multiply each count by 1, 2, or 3 for the corresponding feature-block variant.
No numerical implementation is certified by this arithmetic table.

## Paper coverage and planned workload

| Output | Preset | Data / grid | New fits |
| --- | --- | --- | ---: |
| Figure 2 | `figure2` | DS1; 20–2520 ms in 50 ms steps; 50 filters; 49 coefficients; no deltas | 408 |
| Figure 3 | `figure3` | DS1; 26–271 filters in steps of 5; 1 s frames; three block variants × three fractions | 3,600 |
| Figure 4 | `figure4` | DS1N at 0 dB; reuse Figure 3 clean models | 0 |
| Table 2 | `table2` | DS2; 1 s frames; 201 filters; 200 coefficients; no deltas | 8 |
| Figure 5 | `figure5` | DS2N at 93 levels, −8 to 15 dB by 0.25; reuse Table 2 clean models | 0 |
| Tables 3–5 | `tables4_5` | Ten fixed Table 3 partitions; Table 2 feature settings; registered/background training and registered/attack testing | 80 |

All frame settings use 50% overlap. The 4,096 planned fits are a workload count,
not a runtime estimate. Noisy evaluations add prediction/feature work without
retraining. Figure 1 is the recording setup and needs no model-generated figure.
Table 1 comes from WAV metadata, not predictions. Table 3 is validated structurally
and was cross-checked against the published table.

## Validate without training

Run from the repository root using Python 3.11 or newer. This standalone validator
uses only the standard library. The training environment separately targets
CPython 3.14.8 with 3.13.16 compatibility on Ubuntu, as verified in P1/P2.

```bash
mkdir -p ../refactor_evidence/P0/local-validation
python3 -B tools/validate_p0.py \
  --audio-root ../refactor_artifacts/data/acsac2022/extracted \
  > ../refactor_evidence/P0/local-validation/validation.json \
  2> ../refactor_evidence/P0/local-validation/validation.log
python3 -B -m unittest discover -s tests -p 'test_p0_*.py' -v \
  > ../refactor_evidence/P0/local-validation/tests.log 2>&1
```

Use a new output directory for each run to retain earlier evidence. Without
`--audio-root`, the validator checks only specifications and explicitly reports
audio verification as not run. With it, every published WAV is checked against
the manifest. Missing, unexpected, corrupt or mismatched files fail with a
nonzero exit. Checks stream each file once for hashing and read its WAV header;
they do not write to the audio root or generate feature files. This is a logical
read check, not a measurement of physical SSD wear or the cause of a past failure.

Maintainers can regenerate an existing evidence report with
`python3 -B tools/report_p0.py --evidence-dir /path/to/external/P0/run`.
This requires Matplotlib and the bundle's `manifest.json`,
`measurements/protocol-validation.json`, logs and source patch. The generator
rejects output inside the repository and detects configuration changes since
validation. Plotting dependency versions are recorded in each run manifest;
the committed lockfile specifies the modern reporting environment.


## P6 label-order finding

The modern implementation assigns background label 0 and preserves registered
identity labels 1–24. Historical PKL training assigned background the largest
label instead. Class order affects KNN vote ties and the configured capped
RBF-SVM solver in controlled full-data comparisons. The frozen rerun retains its
original encoding; a separately labelled diagnostic restores background-last
ordering without changing feature values. See [full results and differences](REPRODUCTION_RESULTS.md).
Do not treat background renumbering as a numerically invariant operation.
