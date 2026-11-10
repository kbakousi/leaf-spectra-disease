"""From pixel spectra to one spectrum per leaf (or region of interest).

Labels in leaf-disease studies are usually given per leaf, not per pixel, and
pixels of one leaf are strongly correlated. Summarising each leaf before
modelling (or at least validating by leaf, see :mod:`leafspec.validate`)
keeps the unit of analysis equal to the unit that was labelled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .spectra import Spectra

__all__ = ["leaf_spectra"]


def leaf_spectra(spec: Spectra, by="leaf", stat: str = "median", min_pixels: int = 1,
                 carry=("label", "date")) -> Spectra:
    """One spectrum per group (e.g. leaf), with the pixel count and spread.

    Parameters
    ----------
    by : str or list of str
        Metadata column(s) identifying a leaf.
    stat : {"median", "mean"}
        The median is robust to remaining edge or lesion pixels.
    min_pixels : int
        Groups with fewer pixels are dropped.
    carry : tuple of str
        Metadata columns copied to the leaf table; each must be constant
        within a leaf, otherwise a ``ValueError`` is raised (e.g. a leaf with
        two labels).

    Returns
    -------
    Spectra with metadata ``by`` columns, ``carry`` columns, ``n_pixels`` and
    ``mean_iqr`` (the mean over bands of the within-leaf interquartile range).
    """
    keys = [by] if isinstance(by, str) else list(by)
    if stat not in ("median", "mean"):
        raise ValueError("stat must be 'median' or 'mean'")
    carry = [c for c in carry if c in spec.meta.columns and c not in keys]
    groups = spec.meta.groupby(keys, sort=True, dropna=False).indices
    rows, vals = [], []
    for key, idx in groups.items():
        if len(idx) < min_pixels:
            continue
        block = spec.values[idx]
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        for c in carry:
            u = spec.meta[c].iloc[idx].unique()
            if len(u) != 1:
                raise ValueError(f"leaf {row} has {len(u)} different values of '{c}': {list(u)[:3]}")
            row[c] = u[0]
        q75, q25 = np.nanpercentile(block, [75, 25], axis=0)
        row["n_pixels"] = len(idx)
        row["mean_iqr"] = float(np.nanmean(q75 - q25))
        rows.append(row)
        vals.append(np.nanmedian(block, axis=0) if stat == "median" else np.nanmean(block, axis=0))
    if not rows:
        raise ValueError("no group has at least min_pixels pixels")
    return Spectra(np.vstack(vals), spec.wavelengths, pd.DataFrame(rows))
