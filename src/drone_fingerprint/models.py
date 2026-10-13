"""The eight paper estimators, with documented migration choices."""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


def make_model(name: str, seed: int = 2022):
    """Construct an estimator; retain historical budgets and LSVM's dual solver."""
    if name == "QDA":
        return QuadraticDiscriminantAnalysis()
    if name == "LDA":
        return LinearDiscriminantAnalysis()
    if name == "LSVM":
        return LinearSVC(C=1, max_iter=1000, dual=True, random_state=seed)
    if name == "RBF-SVM":
        return SVC(C=1, kernel="rbf", gamma="auto", decision_function_shape="ovo", max_iter=1000,
                   probability=False, random_state=seed)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=8, n_jobs=1)
    if name == "DT":
        return DecisionTreeClassifier(max_depth=20, min_samples_leaf=15, min_samples_split=15, random_state=1)
    if name == "RF":
        return RandomForestClassifier(max_depth=15, min_samples_leaf=10, min_samples_split=10,
                                      max_features="sqrt", n_jobs=1, random_state=seed)
    if name == "GNB":
        return GaussianNB()
    raise ValueError(f"Unknown model: {name}")
