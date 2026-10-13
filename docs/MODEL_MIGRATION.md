# Model defaults in the modern environment

P1 audits all eight constructors before any real-data fit. It compares the
official scikit-learn **1.0.2** constructor signatures with **1.9.1**, overlays
the paper repository's explicit settings, and checks the resulting modern
parameters. The historical signature snapshot and its exact upstream URLs are
in `config/paper/sklearn_1_0_2_defaults.json`. Run `tools/audit_models.py` to obtain
the full machine-readable comparison. An unreviewed parameter difference fails.

| Model / parameter | Migration decision |
| --- | --- |
| LinearSVC `dual` | Set `True` explicitly to preserve the historical solver; the modern default is `auto`. |
| RF `max_features` | Replace historical classifier `auto` with its `sqrt` equivalent. |
| SVC `probability` | Set historical `False` explicitly instead of inheriting the modern deprecation sentinel. Any resulting deprecation warning remains visible. |
| Previously unseeded LSVM, SVC, RF | Record seed 2022; retain DT's explicit historical seed 1. SVC probability calibration is disabled, so its RNG setting does not enable calibration. |
| QDA new parameters | Accept `solver='svd'`, `shrinkage=None`, `covariance_estimator=None`; do not enable a new covariance method or regularization. |
| DT/RF new `monotonic_cst` | Keep `None`; no constraints added. |
| Remaining shared defaults | Require equality with the historical signature plus explicit repository settings. |

The [LinearSVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
describes the solver default change. The [Random Forest documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
documents the `sqrt` default. Exact installed parameters are saved in each run;
the versioned signature comparison guards future dependency updates.

Both SVMs retain `max_iter=1000`; the iteration limit is separate from wall time.
No timeout is imposed because startup or fitting is quiet. Convergence warnings
are part of the run record and are not suppressed to make a run appear healthy.

This constructor audit does not establish numerical equivalence of old and new
sklearn implementations. P1 fits GNB on a small two-identity closed-set example;
P4 validates all eight model fits, and P6 evaluates the full paper protocol.

## P3 execution limits

P3 sets RF and KNN `n_jobs=1` and limits BLAS/OpenMP work to one thread during
preparation, fitting and evaluation. This is a deliberate resource-policy change
from RF's historical `n_jobs=-1`; the constructor audit records it explicitly.
Seeds, model formulas, splitting and SVM iteration limits are unchanged.

## P4 QDA fit compatibility

The first eight-model real-data run failed at QDA before producing a result.
Constructor equality had missed two implementation changes: the installed 1.9.1
SVD fit divides squared singular values by `n`, and rejects covariance scales
below its rank threshold. The historical 1.0.2 fit divides by `n-1` and warns
when the centered data's singular values are below the threshold. L2-normalized
MFCC features exposed the new failure on the first class. The current API text
still describes warning behavior; this decision follows the actual installed
source and observed exception, not that description alone.

`qda.py::PaperQDA` keeps the historical SVD fit explicitly, using modern input
validation and inherited prediction methods. It adds no regularization or
shrinkage and preserves `tol=1e-4`. Rank warnings remain visible. Exactly zero
or nonfinite covariance scales, and nonfinite decision scores, fail instead of
publishing arbitrary predictions. This compatibility class is limited to the
historical SVD solver; other covariance methods are separate scientific choices.
The adapted fit retains upstream [BSD-3-Clause attribution](../LICENSE_sklearn_QDA).

Sources: [scikit-learn 1.0.2 implementation](https://github.com/scikit-learn/scikit-learn/blob/1.0.2/sklearn/discriminant_analysis.py),
[1.9.1 implementation](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/discriminant_analysis.py),
and [current QDA API](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html).
The phase evidence retains the original failure, source snapshots and a direct
real-feature replay of the historical fit. Unit tests also compare scores to
independently computed Gaussian densities with unbiased sample covariance.
A future scikit-learn upgrade must rerun both constructor and numerical checks.
