"""Classifier and one-class pipelines for spectra.

Every model is a scikit-learn ``Pipeline``. Scaling and any dimensionality
reduction are therefore fitted inside each cross-validation fold, never on
data that is later used for testing.

References
----------
Barker, M., Rayens, W. (2003). Partial least squares for discrimination.
    Journal of Chemometrics 17, 166-173. https://doi.org/10.1002/cem.785
Schölkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J., Williamson, R.C.
    (2001). Estimating the support of a high-dimensional distribution.
    Neural Computation 13, 1443-1471. https://doi.org/10.1162/089976601750264965
Breiman, L. (2001). Random forests. Machine Learning 45, 5-32.
    https://doi.org/10.1023/A:1010933404324
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM

__all__ = ["PLSDA", "make_classifier", "make_one_class"]


class PLSDA(ClassifierMixin, BaseEstimator):
    """Partial least squares discriminant analysis (Barker & Rayens 2003).

    PLS regression on one-hot class indicators; the predicted class is the
    one with the largest fitted indicator. ``decision_function`` returns those
    indicators (they are scores, not probabilities).
    """

    def __init__(self, n_components: int = 5):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, dtype="float64")
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        Y = (y[:, None] == self.classes_[None, :]).astype("float64")
        k = max(1, min(self.n_components, X.shape[1], X.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=k, scale=False).fit(X, Y)
        return self

    def decision_function(self, X):
        return self.pls_.predict(np.asarray(X, dtype="float64"))

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


def make_classifier(kind: str, pca_components: int | None = None, random_state: int = 0, **params) -> Pipeline:
    """A classification pipeline: standardise -> optional PCA -> model.

    Parameters
    ----------
    kind : {"plsda", "svm", "rf", "logistic"}
        ``plsda``: :class:`PLSDA` (``n_components``); ``svm``: RBF support
        vector machine with balanced class weights (``C``, ``gamma``); ``rf``:
        random forest with balanced class weights (Breiman 2001); ``logistic``:
        L2 logistic regression, a simple baseline.
    pca_components : int, optional
        Insert PCA after scaling (fitted per fold).
    **params
        Passed to the model.
    """
    steps = [("scale", StandardScaler())]
    if pca_components:
        steps.append(("pca", PCA(n_components=pca_components, random_state=random_state)))
    if kind == "plsda":
        model = PLSDA(**params)
    elif kind == "svm":
        model = SVC(**{"kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": "balanced", **params})
    elif kind == "rf":
        model = RandomForestClassifier(**{"n_estimators": 300, "class_weight": "balanced",
                                          "random_state": random_state, **params})  # pass n_jobs=-1 for speed
    elif kind == "logistic":
        model = LogisticRegression(**{"max_iter": 5000, "class_weight": "balanced", **params})
    else:
        raise ValueError("kind must be 'plsda', 'svm', 'rf' or 'logistic'")
    steps.append(("model", model))
    return Pipeline(steps)


def make_one_class(nu: float = 0.1, gamma="scale", pca_components: int | None = None) -> Pipeline:
    """One-class SVM (Schölkopf et al. 2001) for anomaly screening.

    Fit it on **healthy** spectra only. ``-decision_function(X)`` is then an
    anomaly score: higher means less like the healthy training leaves. ``nu``
    bounds the share of training spectra treated as outliers.
    """
    steps = [("scale", StandardScaler())]
    if pca_components:
        steps.append(("pca", PCA(n_components=pca_components)))
    steps.append(("model", OneClassSVM(nu=nu, gamma=gamma)))
    return Pipeline(steps)
