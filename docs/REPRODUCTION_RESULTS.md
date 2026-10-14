# Full corrected rerun and differences from the paper

The full Ubuntu run completed on 14 October 2026 using the frozen implementation
at `62f6d2e`, Python 3.14.8 and the committed dependency lock. All 8,440 planned
tasks completed in one attempt each: 4,096 fits and 4,344 evaluations of existing
clean models. The complete Figures 2–5 contain 8,352 measured points. Tables 2–5
and the diagnostic plots are exported from verified saved predictions.

This establishes execution of the corrected protocol. It does **not** establish
exact recovery of every score in the historical paper. Keep the original paper,
primary rerun and controlled diagnostics distinct when reporting results.

## Closed-set authentication

| Classifier | Published Table 2 accuracy (%) | Corrected rerun (%) | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| QDA | 96.20 | 96.2034 | +0.0034 |
| LDA | 92.43 | 92.4298 | -0.0002 |
| LSVM | 93.68 | 93.6781 | -0.0019 |
| RBF-SVM | 66.05 | 66.0435 | -0.0065 |
| KNN | 90.49 | 90.4970 | +0.0070 |
| DT | 62.83 | 62.7589 | -0.0711 |
| RF | 83.73 | 84.0543 | +0.3243 |
| GNB | 67.15 | 67.1595 | +0.0095 |

Across accuracy, macro precision, macro recall and macro F1, the largest absolute
Table 2 difference is 0.3306 percentage points. QDA, LDA and LSVM agree with all
four printed metrics at two-decimal precision. Printed values are rounded; the
new unrounded values are retained in the external results.

## Open-set label ordering is consequential

The frozen modern protocol uses registered identities 1–24 and background label
0. The historical PKL trainer used drone labels 0–23 and background label 24.
Background therefore sorts first in the modern encoding and last in the old one.
This is not a neutral change for every configured estimator:

- KNN uses eight neighbors. A tied vote can change when class order changes; a
  small regression fixture demonstrates that behavior explicitly.
- The configured RBF-SVM retains `max_iter=1000`. Controlled full-data experiments
  show that changing background class order changes its fitted predictions while
  all features, row order, seeds and hyperparameters remain identical. These fits
  retain convergence warnings; completion does not establish convergence.

The primary ten-trial RBF-SVM mean registered recall is **79.2360%**, versus
**13.05%** in the paper. Macro attack rejection is **30.9563%**, versus **98.10%**.
Its pooled rejection is **30.9591%**; changing metric aggregation alone does not
explain the discrepancy. These primary results are preserved without replacement.

`tools/validate_label_order.py` performs a separate diagnostic on all ten trials
for RBF-SVM and KNN. It relabels background 0 to 25 during fitting and maps predicted
25 back to 0 before evaluation. This restores the historical relative class order
while retaining the modern features and runtime. It is not a complete old-code or
old-environment replay. The first trial also repeats the unchanged procedure and
requires prediction equality before the changed-label comparison.

The ten-trial diagnostic means are:

| Classifier and metric | Frozen background-first (%) | Diagnostic background-last (%) | Published (%) |
| --- | ---: | ---: | ---: |
| RBF-SVM registered macro recall | 79.2360 | 13.0362 | 13.05 |
| RBF-SVM macro attack rejection | 30.9563 | 97.9357 | 98.10 |
| KNN registered macro recall | 91.6965 | 92.7188 | 92.71 |
| KNN macro attack rejection | 61.4638 | 56.9300 | 56.92 |

Background-last KNN pooled rejection is 56.9207%. The label-order experiment
explains the major direction and scale of the gap, while remaining differences
are retained. It does not isolate every historical numerical or library change.
The external diagnostic report contains every trial, warning and prediction.
A future choice to change the primary label convention requires a documented
protocol change and new results; it must not silently overwrite this rerun.

## Figures and numerical corrections

Visual review finds the broad longer-frame trend in Figure 2 and the SNR trend
in Figure 5. The derivative panels of Figures 3/4 differ materially, particularly
for KNN. The modern pipeline uses true recursive, segment-local derivatives and
normalizes the concatenated vector; these are documented numerical corrections.
The full-grid visual comparison does not isolate the contribution of each change.
Original plotting CSV files are unavailable, so no pointwise paper-curve error is
claimed. Do not recover missing reference values by treating a plot image as an
exact numerical dataset.

## Reproducibility and measured resources

All eight Table 2 classifiers were fitted again on the full dataset with the same
locked runtime. Predictions, metrics, confusion matrices, scientific settings
and serialized model hashes match exactly. This is not a second full sweep or a
cross-machine equivalence guarantee. All 257 original WAVs and 1,934 cached arrays
were verified by SHA-256 after the full run, as were all saved model/prediction
artifacts and the gallery. Original WAVs remain read-only.

The full training/evaluation process took **39,519.61 seconds** (10 h 58 m 40 s),
with a kernel-reported peak RSS of **1.9367 GiB**. Run artifacts occupy **30.06 GiB**
in 29,419 files; the shared feature cache occupies **26.14 GiB** in 3,870 files.
Most result storage is the 4,096 trained models. There are no generated sliced or
noisy WAV files and no per-frame feature files. These observations describe this
host and protocol, not guaranteed requirements on every system.

The run retained 494 RBF-SVM convergence warnings and 512 SVC deprecation warnings.
OS-accounted I/O includes publication and repeated atomic ledger updates and can
exceed retained storage. It does not measure physical NAND writes or diagnose the
author's historical SSD failure. The supervisor records process usage separately
from later verification and diagnostic work.

## Reusing the evidence tools

All outputs must be outside the checkout. Use the locked environment matching the
saved results. The release acceptance bundle records exact commands, paths,
checksums and failed checks; `tools/validate_full_run.py` verifies that bundle and
produces table comparisons and resource inventory. `tools/report_release.py`
renders its offline closure report after validation and review pass. These tools
are phase-evidence helpers; ordinary users generate their own scientific gallery
with the documented `drone-auth paper` and `drone-auth plot` commands.

Ubuntu local acceptance is complete. Remote CI, repository push and publication
are separate release actions. Existing container smoke checks cover Linux/amd64;
the full sweep ran natively on Ubuntu. LaTeX table fragments were exported but not
compiled. The within-recording split does not test unseen recordings, rooms or
adversarial replay attacks.
