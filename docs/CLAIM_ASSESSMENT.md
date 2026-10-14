# Paper conclusions: claim-by-claim assessment

Assessment date: 14 October 2026. Measurements: frozen full P6 run using
scientific implementation `62f6d2e`. Paper locations refer to printed page
numbers in the author-provided ACSAC manuscript (PDF page = printed page − 657).

The central feasibility result and QDA as a strong balanced choice remain supported within the original evaluation setting. An unqualified statement that every conclusion is unchanged is not supported: the derivative-feature interpretation, exact rankings, RBF-SVM open-set behavior and several universal or boundary statements require revision.

All 8,352 plot points were checked against per-task exported metrics and gallery hashes. Paired derivative comparisons match filter count, coefficient fraction, classifier, test noise and dataset, changing the feature-block setting; each contrast covers 150 related configurations. This audits the maintained implementation and does not isolate individual old-to-new code changes. Grid averages and ten shared-data partitions are descriptive, not independent replicates. No statistical equivalence margin, independent multi-seed replication, significance test or new training is introduced.

## C01 — Core feasibility

Paper: §1 p.659; §5.1 p.663; §7 p.667

Claim: MFCC fingerprints distinguish individual same-model drone configurations in the closed-set experiment.

Assessment: **Supported within the evaluated protocol**

QDA accuracy is 96.2034%. All four Table 2 metrics exceed 96%. The data support feasibility in this indoor, within-recording evaluation, not universal physical uniqueness or deployment security.

## C02 — Body and propeller combinations

Paper: §1 p.659; §5.1 p.663; §7 p.667

Claim: A new combination of body and propellers can be treated as a distinct identity.

Assessment: **Supported as observed separability**

QDA macro recall within the original, replacement-x and replacement-y groups is 98.46%, 95.93% and 94.20%. The 24-class result still supports separability. This is not an isolated causal test of motor versus propeller contributions or a held-out-session test.

## C03 — Clean 8-drone classifier preference

Paper: §4.2 p.663; §7 p.667

Claim: QDA, LDA and LSVM outperform the alternatives without AWGN.

Assessment: **Supported as a dominant group**

One of these three wins in all 450 clean configurations. They strictly occupy all top-three positions in 448/450. At 26 filters with one-third coefficients, RF enters the top three in two configurations. A universal strict ranking would be too strong.

## C04 — Clean 24-drone metrics

Paper: Table 2 p.663; §7 p.667

Claim: QDA, LDA, LSVM and KNN each exceed 90% on accuracy, precision, recall and F1; QDA is best.

Assessment: **Supported**

All 16 model/metric comparisons pass the stated 90% threshold, and QDA leads all four metrics. Accuracies: QDA 96.2034%, LDA 92.4298%, LSVM 93.6781%, KNN 90.4970%. This is a numerical ranking, not a new statistical significance claim.

## C05 — KNN at low SNR

Paper: §5.2 p.664; §7 p.667

Claim: KNN is the most effective method when SNR is below −3 dB.

Assessment: **Supported across the tested range**

KNN wins all 20/20 sampled points from −8 to −3.25 dB. Its smallest lead over the best other classifier is 1.9156 pp. This does not extrapolate below −8 dB or to untested continuous SNR values.

## C06 — LDA under 0 dB AWGN

Paper: §4.2.4 p.663; §7 p.667

Claim: LDA has the highest accuracy in all configurations under 0 dB AWGN.

Assessment: **Dominance retained; universal wording needs qualification**

LDA is highest in 440/450 configurations. Ten low-filter/one-third configurations are exceptions. The conclusion that LDA is a strong choice remains supported; the word all is not supported by the new run.

## C07 — QDA on registered drones

Paper: Abstract p.658; §5.4 p.665–666; §7 p.667

Claim: QDA has the highest mean recall on registered drones, about 94.19%.

Assessment: **Supported**

Ten-trial macro recall is 94.1884%, rank 1. The statement concerns the ten-trial mean: QDA is not first in every individual trial.

## C08 — QDA on unknown drones

Paper: Abstract p.658; §5.4.2 p.666; §7 p.667

Claim: QDA has the third-highest mean recall on unregistered drones, 82.35%.

Assessment: **Magnitude retained; exact rank and number change**

Macro rejection is 82.1973% (−0.1527 pp from the printed value), rank 2 in the frozen primary run. Pooled rejection is 82.2340%. QDA remains strong, but third and 82.35% cannot label the new measurements. The complementary macro miss rate is about 17.80%; rejection is not perfect.

## C09 — QDA as an overall choice

Paper: §5.4.2 p.666; §7 p.667

Claim: QDA offers the best overall authentication performance.

Assessment: **Supported as a trade-off, not a universal optimum**

QDA has higher registered recall and unknown rejection than 6 of the other seven classifiers on the two ten-trial means. RF still rejects more unknowns (84.4099% versus 82.1973%), while recognizing fewer registered drones (75.4722% versus 94.1884%). QDA is a strong balanced choice; no predeclared utility or error-cost function establishes an unconditional single best model.

## C10 — Derivative effect on RBF-SVM and KNN

Paper: §4.2.2–4 p.662–663

Claim: Adding derivatives reduces RBF-SVM and KNN accuracy; KNN works well only with MFCC.

Assessment: **Clean-set direction retained; magnitude and generality change**

In the clean grid, each added-feature variant is lower than its matched MFCC-only result in all 150/150 configurations for both classifiers. For KNN, mean losses are 1.8470 pp and 2.1647 pp. At 201 filters/all coefficients it remains 95.1630% or 95.0589%, versus 96.1165% with MFCC only. Thus the direction survives but good only with MFCC overstates the new degradation. At 0 dB there are also configurations with small gains; the decrease is not universal.

## C11 — No positive effect from derivatives

Paper: §4.2.4 p.663

Claim: DMFCC and DDMFCC do not have a positive effect on drone authentication.

Assessment: **Not supported as a general claim**

MFCC + delta + delta-delta improves LDA and LSVM in every matched configuration: 150/150 each on clean data and 150/150 each at 0 dB. At 201 filters/all coefficients, LDA at 0 dB rises from 93.7240% to 97.0700% (+3.3460 pp); LSVM rises from 90.6207% to 92.6318% (+2.0111 pp). These are observed gains under the revised feature definitions, not evidence of statistical significance or an attribution to one individual code fix.

## C12 — Practical feature configuration

Paper: §4.1–4.2 p.662–663; §7 p.667

Claim: Use 1000 ms frames, 50% overlap, 201 filters and MFCC coefficients 2–201.

Assessment: **Effective baseline retained; optimality not established**

The unchanged DS2 setting still produces QDA accuracy 96.2034% and strong open-set results. It remains a useful baseline. The derivative results do not establish MFCC-only as optimal. Figure 2 samples 970 and 1020 ms, not exactly 1000 ms; it supports a broad longer-frame trend. Overlap was held fixed, so its optimality was not tested.

## C13 — More filters and a plateau near 200

Paper: §4.2 p.662–663

Claim: More filters improve accuracy, with a bottleneck around 200 filters.

Assessment: **Broad trend retained; plateau needs qualification**

All 72 curves in each clean/noisy grid end higher at 271 than at 26 filters; this is not strict stepwise monotonicity. In the clean grid, the largest improvement available beyond 201 filters is 2.3232 pp across the scanned configurations. At 0 dB, QDA with one-third MFCC coefficients gains 10.4196 pp from 201 to 271 filters. A universal noisy-data plateau at 200 is not supported.

## C14 — The 2 dB and high-SNR thresholds

Paper: §5.2 p.664; §6 p.666

Claim: QDA/LDA/LSVM/KNN exceed 80% from 2 dB, with almost no further growth after 10 dB.

Assessment: **Broad robustness retained; boundaries need qualification**

At exactly 2 dB, accuracies are 80.0104%, 84.1118%, 79.8723% and 79.6192%, respectively. All four exceed 80% at every sampled point strictly above 2 dB (2.25–15 dB). Section 6 says greater than 2 dB and remains supported in that sampled sense; at least 2 dB is slightly too strong. From 10 to 15 dB, gains range from 0.1150 pp (RBF-SVM) to 4.6422 pp (DT), so a universal no-growth statement is too strong.

## C15 — RBF-SVM rejects almost everything

Paper: §5.4.1–2 p.665–666

Claim: RBF-SVM achieves 98.10% unknown rejection by classifying almost all drones as unknown, with only 13.05% registered recall.

Assessment: **Not supported for the frozen primary encoding**

The primary run instead gives 79.2360% registered recall and 30.9563% rejection. A separate background-last control gives 13.0362% and 97.9357%, close to the historical behavior. This establishes a configured-procedure sensitivity to class order. It is not a replacement result, a converged-SVM guarantee, or a rerun of all eight models under the alternate encoding.

## C16 — Novelty, physical cause and deployment scope

Paper: §1 p.659; §6 p.666–667

Claim: Novelty and physical-source interpretation; wear/reassembly observations; outdoor and replay robustness identified as future work.

Assessment: **Not established by this rerun**

A computational rerun cannot verify first-work priority or isolate the physical source of the fingerprint. Wear/reassembly observations were not repeated. Outdoor noise, held-out recording sessions, other drone models and replay attacks were not tested; the paper itself identifies several of these as future work. These limits are not newly demonstrated failures caused by the refactor.


## Reproduction and interpretation

This analysis uses saved results without training. The numerical helper
`tools/assess_paper_claims.py` verifies the gallery hashes and reconciles all 8,352
plotted values with per-task exported metrics. It produces the claim ledger,
matched-configuration differences and supporting figures in an external directory.
Its numerical annotations and interpretations are intentionally bound to the
frozen P6 input hashes; another result version needs a newly reviewed assessment.

```bash
uv run --locked python tools/assess_paper_claims.py \
  --full-evidence ../refactor_evidence/P6/20261013T232406Z-full-paper \
  --output ../refactor_evidence/paper-claims-new
```

The evidence bundle is local and not shipped with the source repository. For
ordinary experiments, use `drone-auth paper` and `drone-auth plot`. See
[the full-run findings](REPRODUCTION_RESULTS.md),
[numerical changes](NUMERICAL_VALIDATION.md), and
[figure comparisons](FIGURES.md).

A suitable summary is: the updated implementation continues to support the
paper's central feasibility findings and QDA as a strong balanced choice for the
evaluated tasks. Some feature-extraction findings, exact rankings and numerical
thresholds differ. It is not accurate to state that every published conclusion
is unchanged.
