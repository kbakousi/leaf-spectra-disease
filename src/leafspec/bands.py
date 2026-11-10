"""Which parts of the spectrum a model relies on: grouped, window-wise permutation importance.

Neighbouring bands are almost perfectly correlated, so permuting a single
band hardly changes a model that can use its neighbours instead, and the
importance of the whole region is underestimated. Here contiguous windows of
bands (e.g. 20 nm) are permuted together, on held-out leaves only.

References
----------
Breiman, L. (2001). Random forests. Machine Learning 45, 5-32.
    https://doi.org/10.1023/A:1010933404324 (permutation importance)
Mahlein, A.-K. (2016). Plant disease detection by imaging sensors - parallels
    and specific demands for precision agriculture and plant phenotyping.
    Plant Disease 100, 241-251. https://doi.org/10.1094/PDIS-03-15-0340-FE
    (spectral regions affected by plant diseases)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score

from .validate import grouped_folds

__all__ = ["window_importance"]


def window_importance(model, X, y, groups, wavelengths, width_nm: float = 20.0, n_splits: int = 5,
                      n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Drop in held-out balanced accuracy when a wavelength window is permuted.

    For each grouped fold the model is fitted on the training leaves; the
    columns of one window are then shuffled jointly across the test rows and
    the fall in balanced accuracy is recorded, ``n_repeats`` times per
    window.

    Returns
    -------
    pandas.DataFrame with ``start_nm``, ``end_nm``, ``centre_nm``,
    ``mean_drop`` and ``sd_drop`` (over folds x repeats), sorted by wavelength.
    """
    X, y = np.asarray(X, dtype="float64"), np.asarray(y)
    wl = np.asarray(wavelengths, dtype="float64")
    edges = np.arange(wl.min(), wl.max() + width_nm, width_nm)
    windows = [(lo, hi, np.flatnonzero((wl >= lo) & (wl < hi))) for lo, hi in zip(edges[:-1], edges[1:],
                                                                                    strict=True)]
    windows = [w for w in windows if w[2].size]
    rng = np.random.default_rng(seed)
    drops = {i: [] for i in range(len(windows))}
    for train, test in grouped_folds(y, groups, n_splits, seed):
        m = clone(model).fit(X[train], y[train])
        base = balanced_accuracy_score(y[test], m.predict(X[test]))
        for i, (_, _, cols) in enumerate(windows):
            for _ in range(n_repeats):
                Xp = X[test].copy()
                Xp[:, cols] = Xp[rng.permutation(len(test))][:, cols]
                drops[i].append(base - balanced_accuracy_score(y[test], m.predict(Xp)))
    return pd.DataFrame([
        {"start_nm": lo, "end_nm": hi, "centre_nm": (lo + hi) / 2, "mean_drop": float(np.mean(drops[i])),
         "sd_drop": float(np.std(drops[i], ddof=1))}
        for i, (lo, hi, _) in enumerate(windows)
    ])
