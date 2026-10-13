"""Paper QDA using the scikit-learn 1.0.2 SVD covariance convention.

Adapted from scikit-learn (BSD-3-Clause); see LICENSE_sklearn_QDA. Modern
scikit-learn provides validation, prediction and persistence. Keep this small
compatibility fit explicit: adding regularization would change the experiment.
"""

import warnings

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import validate_data


class PaperQDA(QuadraticDiscriminantAnalysis):
    """Retain n-1 covariance scaling and warning-only singular-value rank checks."""

    def fit(self, X, y):
        self._validate_params()
        if self.solver != 'svd' or self.shrinkage is not None or self.covariance_estimator is not None:
            raise ValueError('PaperQDA supports the historical SVD solver only')
        X, y = validate_data(self, X, y, dtype=np.float64)
        check_classification_targets(y)
        self.classes_, labels = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError('QDA requires at least two classes')
        self.priors_ = np.bincount(labels) / len(y) if self.priors is None else np.asarray(self.priors)
        if (self.priors_.shape != (len(self.classes_),) or not np.isfinite(self.priors_).all()
                or np.any(self.priors_ <= 0) or not np.isclose(self.priors_.sum(), 1)):
            raise ValueError('QDA priors must be positive and sum to one')
        means, scalings, rotations, covariances = [], [], [], []
        for index, label in enumerate(self.classes_):
            group = X[labels == index]
            if len(group) < 2:
                raise ValueError(f'QDA class {label} needs at least two training samples')
            mean = group.mean(axis=0)
            _, singular, rotation = np.linalg.svd(group - mean, full_matrices=False)
            if np.count_nonzero(singular > self.tol) < X.shape[1]:
                warnings.warn(f'QDA class {label}: variables are collinear under the historical rank check', UserWarning)
            scaling = (1 - self.reg_param) * singular**2 / (len(group) - 1) + self.reg_param
            if not np.isfinite(scaling).all() or np.any(scaling <= 0):
                raise ValueError(f'QDA class {label} has zero or nonfinite covariance scales; no regularization is added automatically')
            means.append(mean)
            scalings.append(scaling)
            rotations.append(rotation.T)
            if self.store_covariance:
                covariances.append(np.dot(scaling * rotation.T, rotation))
        self.means_, self.scalings_, self.rotations_ = np.asarray(means), scalings, rotations
        if self.store_covariance:
            self.covariance_ = covariances
        return self

    def _decision_function(self, X):
        scores = super()._decision_function(X)
        if not np.isfinite(scores).all():
            raise ValueError('QDA produced nonfinite scores; no valid predictions can be published')
        return scores
