"""Synthetic data, models, validation and band importance."""

import numpy as np
import pytest

from leafspec import aggregate, bands, models, synthetic, validate


@pytest.fixture(scope="module")
def pixels():
    return synthetic.make_leaves(pixels_per_leaf=40, seed=3)


@pytest.fixture(scope="module")
def leaves(pixels):
    return aggregate.leaf_spectra(pixels, by="leaf")


def test_synthetic_design_and_reproducibility(pixels):
    assert pixels.meta.groupby("label")["leaf"].nunique().to_dict() == {"esca": 30, "healthy": 30,
                                                                      "powdery_mildew": 30}
    again = synthetic.make_leaves(pixels_per_leaf=40, seed=3)
    assert np.array_equal(again.values, pixels.values)
    assert (pixels.meta.loc[pixels.meta.label == "healthy", "severity"] == 0).all()
    assert not pixels.meta.loc[pixels.meta.label == "healthy", "lesion"].any()
    conf = synthetic.make_leaves(synthetic.CONFOUNDED, pixels_per_leaf=5, seed=0)
    assert conf.meta.groupby("date")["label"].agg(lambda s: (s == "healthy").mean()).is_monotonic_decreasing
    with pytest.raises(ValueError, match="unknown label"):
        synthetic.make_leaves({"2021-01-01": {"rust": 1}}, pixels_per_leaf=2)


def test_grouped_folds_never_split_a_leaf(pixels):
    y, g = pixels.meta["label"].to_numpy(), pixels.meta["leaf"].to_numpy()
    folds = validate.grouped_folds(y, g, n_splits=5, seed=1)
    seen = []
    for train, test in folds:
        assert not set(g[train]) & set(g[test])
        seen += list(test)
    assert sorted(seen) == list(range(len(y)))
    with pytest.raises(ValueError, match="groups for"):
        validate.grouped_folds(y[:3], g[:3], n_splits=5)


def test_random_split_overstates_accuracy(pixels):
    y, g = pixels.meta["label"].to_numpy(), pixels.meta["leaf"].to_numpy()
    tab = validate.compare_splits(models.make_classifier("rf", n_estimators=100), pixels.values, y, g, n_splits=5)
    random_ba, grouped_ba = tab["balanced_accuracy"].to_numpy()
    assert random_ba > grouped_ba + 0.05  # leakage between pixels of the same leaf


def test_classifiers_learn_at_leaf_level(leaves):
    y, g = leaves.meta["label"].to_numpy(), leaves.meta["leaf"].to_numpy()
    folds = validate.grouped_folds(y, g, 5, seed=0)
    for kind, params in [("plsda", {"n_components": 6}), ("svm", {}), ("rf", {"n_estimators": 100}),
                         ("logistic", {})]:
        s = validate.scores(y, validate.out_of_fold(models.make_classifier(kind, **params), leaves.values, y, folds))
        assert s["balanced_accuracy"] > 0.6, kind
        assert s["confusion"].to_numpy().sum() == len(y)
    with pytest.raises(ValueError, match="kind must be"):
        models.make_classifier("cnn")


def test_one_class_trained_on_healthy_only(leaves):
    y, g = leaves.meta["label"].to_numpy(), leaves.meta["leaf"].to_numpy()
    score, auc = validate.one_class_out_of_fold(models.make_one_class(nu=0.1), leaves.values, y == "healthy", g, 5)
    assert np.isfinite(score).all() and auc > 0.8
    assert score[y != "healthy"].mean() > score[y == "healthy"].mean()


def test_leave_one_date_out_and_confounding():
    conf = aggregate.leaf_spectra(synthetic.make_leaves(synthetic.CONFOUNDED, pixels_per_leaf=20, seed=2), by="leaf")
    y = conf.meta["label"].to_numpy()
    out = validate.leave_one_group_out(models.make_classifier("logistic"), conf.values, y, conf.meta["date"])
    assert np.isnan(out.loc["2021-05-28", "balanced_accuracy"])  # one class only on that date
    # healthy leaves alone reveal the date: the season is learnable, so date can stand in for disease
    healthy = conf.meta["label"].to_numpy() == "healthy"
    d = conf.meta["date"].to_numpy()[healthy]
    early_late = np.isin(d, ["2021-05-28", "2021-08-13"])
    xs, ds = conf.values[healthy][early_late], d[early_late]
    pred = validate.out_of_fold(models.make_classifier("logistic"), xs, ds,
                                validate.random_folds(ds, n_splits=2, seed=0))
    assert np.mean(pred == ds) > 0.7


def test_window_importance_finds_the_informative_region():
    rng = np.random.default_rng(0)
    wl = np.arange(400.0, 1000.0, 5.0)
    n = 120
    y = np.repeat(["a", "b"], n // 2)
    X = rng.normal(0, 1, (n, wl.size))
    X[np.ix_(y == "b", (wl >= 660) & (wl < 680))] += 2.0  # only 660-680 nm differs
    g = np.arange(n)
    imp = bands.window_importance(models.make_classifier("logistic"), X, y, g, wl, width_nm=20, n_splits=4,
                                  n_repeats=3)
    best = imp.loc[imp["mean_drop"].idxmax()]
    assert best["start_nm"] <= 660 < best["end_nm"]
