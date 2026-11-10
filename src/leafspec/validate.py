"""Validation that respects how the data were collected.

Pixels of the same leaf are near-copies of each other. If a random split puts
pixels of one leaf in both the training and the test set, a model can score
well by recognising the leaf rather than the disease, and accuracy is
overestimated. Validation must therefore hold out whole leaves (grouped
cross-validation), and, where conditions differ between dates or sites, whole
dates or sites (Roberts et al. 2017; Ploton et al. 2020).

References
----------
Roberts, D.R. et al. (2017). Cross-validation strategies for data with
    temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40,
    913-929. https://doi.org/10.1111/ecog.02881
Ploton, P. et al. (2020). Spatial validation reveals poor predictive
    performance of large-scale ecological mapping models. Nature
    Communications 11, 4540. https://doi.org/10.1038/s41467-020-18321-y
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

__all__ = ["grouped_folds", "random_folds", "out_of_fold", "scores", "compare_splits", "leave_one_group_out",
           "one_class_out_of_fold"]


def grouped_folds(y, groups, n_splits: int, seed: int = 0) -> list:
    """Folds that keep every group (e.g. leaf) entirely in train or in test, stratified by class."""
    y, groups = np.asarray(y), np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(f"{n_groups} groups for {n_splits} folds")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def random_folds(y, n_splits: int, seed: int = 0) -> list:
    """Stratified random folds that ignore groups (shown for comparison: leaks between pixels of one leaf)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), np.asarray(y)))


def out_of_fold(model, X, y, folds) -> np.ndarray:
    """Predictions for every row from a model that did not see that row's fold."""
    X, y = np.asarray(X, dtype="float64"), np.asarray(y)
    pred = np.empty(len(y), dtype=y.dtype)
    for train, test in folds:
        pred[test] = clone(model).fit(X[train], y[train]).predict(X[test])
    return pred


def scores(y, pred) -> dict:
    """Balanced accuracy, macro F1, recall per class and the confusion matrix (rows = reference)."""
    y, pred = np.asarray(y), np.asarray(pred)
    labels = np.unique(np.concatenate([y, pred]))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "recall": pd.Series(recall_score(y, pred, labels=labels, average=None, zero_division=0), index=labels),
        "confusion": pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels),
    }


def compare_splits(model, X, y, groups, n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Score the same model with a random split and with a split by group.

    The gap between the two rows estimates how much a random (e.g.
    pixel-level) split overstates performance.
    """
    rows = {}
    for name, folds in [("random split (ignores leaves)", random_folds(y, n_splits, seed)),
                        ("grouped split (whole leaves held out)", grouped_folds(y, groups, n_splits, seed))]:
        s = scores(y, out_of_fold(model, X, y, folds))
        rows[name] = {"balanced_accuracy": s["balanced_accuracy"], "macro_f1": s["macro_f1"]}
    return pd.DataFrame(rows).T


def leave_one_group_out(model, X, y, groups) -> pd.DataFrame:
    """Train on all other groups (e.g. dates), test on one; one row per held-out group.

    Balanced accuracy is NaN where the held-out group has a single class.
    """
    X, y, groups = np.asarray(X, dtype="float64"), np.asarray(y), np.asarray(groups)
    rows = []
    for g in np.unique(groups):
        test = groups == g
        if len(np.unique(y[~test])) < 2:
            rows.append({"held_out": g, "n_test": int(test.sum()), "balanced_accuracy": np.nan})
            continue
        pred = clone(model).fit(X[~test], y[~test]).predict(X[test])
        ba = balanced_accuracy_score(y[test], pred) if len(np.unique(y[test])) > 1 else np.nan
        rows.append({"held_out": g, "n_test": int(test.sum()), "balanced_accuracy": ba,
                     "accuracy": float(np.mean(pred == y[test]))})
    return pd.DataFrame(rows).set_index("held_out")


def one_class_out_of_fold(model, X, is_healthy, groups, n_splits: int = 5, seed: int = 0):
    """Anomaly scores from one-class models trained on healthy rows of the training groups only.

    In each grouped fold the model is fitted on the healthy rows of the
    training groups and scores all rows of the test groups. Diseased rows are
    never used for fitting.

    Returns
    -------
    score : numpy.ndarray
        Out-of-fold anomaly score (higher = less like healthy).
    auc : float
        ROC AUC of the score for separating diseased from healthy rows.
    """
    X = np.asarray(X, dtype="float64")
    healthy = np.asarray(is_healthy, dtype=bool)
    score = np.full(len(X), np.nan)
    for train, test in grouped_folds(healthy, groups, n_splits, seed):
        fit_rows = train[healthy[train]]
        m = clone(model).fit(X[fit_rows])
        score[test] = -m.decision_function(X[test])
    return score, float(roc_auc_score(~healthy, score))
