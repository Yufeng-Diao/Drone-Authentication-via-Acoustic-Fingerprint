"""Compare installed estimator parameters with official sklearn 1.0.2 defaults.

The historical signature snapshot is data extracted from the URLs it records;
no historical Python environment is required. No estimators are fitted here.
"""

import json
from pathlib import Path

import sklearn
from drone_fingerprint.config import MODELS
from drone_fingerprint.models import make_model

CONFIG = Path(__file__).resolve().parents[1] / "config/paper"


def audit():
    historical = json.loads((CONFIG / "sklearn_1_0_2_defaults.json").read_text())
    explicit = json.loads((CONFIG / "protocol.json").read_text())["legacy_explicit_classifier_parameters"]
    additions = {"QDA": {"solver": "svd", "shrinkage": None, "covariance_estimator": None},
                 "DT": {"monotonic_cst": None}, "RF": {"monotonic_cst": None}}
    rows = []
    for name in MODELS:
        estimator = make_model(name)
        old_defaults = historical["constructors"][type(estimator).__name__]
        expected = dict(old_defaults, **explicit[name])
        decisions = []
        if name in ("RF", "KNN"):
            expected["n_jobs"] = 1
            decisions.append("P3 bounds internal model parallelism to one worker.")
        if name == "RF":
            expected["max_features"] = "sqrt"
            decisions.append("Historical classifier auto maps to sqrt.")
        if name in ("LSVM", "RBF-SVM", "RF"):
            expected["random_state"] = 2022
            decisions.append("Previously unspecified RNG now uses recorded seed 2022.")
        if name == "LSVM":
            decisions.append("Keep historical dual=True; do not adopt the modern auto solver.")
        if name == "RBF-SVM":
            decisions.append("Keep probability=False explicitly; modern default uses a deprecation sentinel.")
        expected.update(additions.get(name, {}))
        actual = estimator.get_params(deep=False)
        if expected != actual:
            raise ValueError(f"Unreviewed classifier parameter change for {name}: expected={expected}, actual={actual}")
        current_defaults = type(estimator)().get_params(deep=False)
        changed_defaults = {key: {"historical": value, "modern": current_defaults.get(key)}
                            for key, value in old_defaults.items() if value != current_defaults.get(key)}
        rows.append({"model": name, "status": "passed", "changed_library_defaults": changed_defaults,
                     "new_parameters": additions.get(name, {}), "decisions": decisions, "effective_parameters": actual})
    return {"status": "passed", "historical_version": "1.0.2", "installed_version": sklearn.__version__,
            "scope": "Constructor/default audit only; eight-model fitting belongs to P4", "models": rows,
            "historical_sources": historical["source_urls"]}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
