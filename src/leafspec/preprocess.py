"""Spectral preprocessing: band trimming, quality checks, smoothing, derivatives, SNV, continuum removal.

The order used in the tutorial is: quality mask -> trim noisy edge bands ->
(aggregate pixels to leaves) -> optional smoothing / derivative / SNV.
Each step is a documented choice; none is applied silently.

References
----------
Savitzky, A., Golay, M.J.E. (1964). Smoothing and differentiation of data by
    simplified least squares procedures. Analytical Chemistry 36, 1627-1639.
    https://doi.org/10.1021/ac60214a047
Barnes, R.J., Dhanoa, M.S., Lister, S.J. (1989). Standard normal variate
    transformation and de-trending of near-infrared diffuse reflectance
    spectra. Applied Spectroscopy 43, 772-777.
    https://doi.org/10.1366/0003702894202201
Clark, R.N., Roush, T.L. (1984). Reflectance spectroscopy: quantitative
    analysis techniques for remote sensing applications. Journal of
    Geophysical Research 89, 6329-6340. https://doi.org/10.1029/JB089iB07p06329
Rinnan, A., van den Berg, F., Engelsen, S.B. (2009). Review of the most
    common pre-processing techniques for near-infrared spectra. TrAC Trends
    in Analytical Chemistry 28, 1201-1222. https://doi.org/10.1016/j.trac.2009.07.007
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .spectra import Spectra

__all__ = ["trim", "quality_mask", "savgol", "snv", "continuum_removed"]


def trim(spec: Spectra, lo: float, hi: float) -> Spectra:
    """Keep bands with ``lo <= wavelength <= hi`` nm.

    Imaging spectrometers usually have a lower signal-to-noise ratio at the
    ends of their range; plot the mean spectra of your own data to choose
    ``lo`` and ``hi`` (the tutorial uses 430-900 nm for Specim IQ-like data).
    """
    keep = (spec.wavelengths >= lo) & (spec.wavelengths <= hi)
    if keep.sum() < 2:
        raise ValueError("fewer than two bands left after trimming")
    return Spectra(spec.values[:, keep], spec.wavelengths[keep], spec.meta)


def quality_mask(spec: Spectra, *, max_reflectance: float, min_nir: float, min_ndvi: float,
                 nir_nm: float = 800.0, red_nm: float = 670.0) -> pd.DataFrame:
    """Flag spectra that are not usable leaf pixels.

    All thresholds are required, documented choices.

    Returns
    -------
    pandas.DataFrame with boolean columns ``non_finite``, ``saturated``
    (any band above ``max_reflectance``), ``shadow`` (reflectance at
    ``nir_nm`` below ``min_nir``), ``not_leaf`` (NDVI from ``nir_nm`` and
    ``red_nm`` below ``min_ndvi``: background, soil, stems) and ``ok`` (none
    of these).
    """
    v = spec.values
    out = pd.DataFrame(index=spec.meta.index)
    out["non_finite"] = ~np.isfinite(v).all(axis=1)
    with np.errstate(invalid="ignore"):
        out["saturated"] = np.nanmax(np.where(np.isfinite(v), v, -np.inf), axis=1) > max_reflectance
        nir, red = spec.at(nir_nm), spec.at(red_nm)
        out["shadow"] = ~(nir >= min_nir)
        den = nir + red
        ndvi = np.where(np.abs(den) > 1e-12, (nir - red) / np.where(den == 0, 1, den), np.nan)
        out["not_leaf"] = ~(ndvi >= min_ndvi)
    out["ok"] = ~out[["non_finite", "saturated", "shadow", "not_leaf"]].any(axis=1)
    return out


def _spacing(wl):
    d = np.diff(wl)
    if np.max(np.abs(d - d.mean())) > 0.05 * d.mean():
        warnings.warn("band spacing varies by more than 5 %; Savitzky-Golay assumes even spacing", UserWarning,
                      stacklevel=3)
    return float(d.mean())


def savgol(spec: Spectra, window: int, polyorder: int, deriv: int = 0) -> Spectra:
    """Savitzky-Golay smoothing (``deriv=0``) or derivative along wavelength.

    Derivatives are per nm (the mean band spacing is used). ``window`` is in
    bands and must be odd and larger than ``polyorder``.
    """
    if window % 2 == 0 or window <= polyorder:
        raise ValueError("window must be odd and larger than polyorder")
    delta = _spacing(spec.wavelengths)
    out = savgol_filter(spec.values, window_length=window, polyorder=polyorder, deriv=deriv, delta=delta, axis=1)
    return spec.with_values(out)


def snv(spec: Spectra) -> Spectra:
    """Standard normal variate: centre and scale each spectrum by its own mean and SD (Barnes et al. 1989).

    Removes multiplicative illumination and scatter differences between
    spectra, but also any real overall brightness difference.
    """
    v = spec.values
    mu = v.mean(axis=1, keepdims=True)
    sd = v.std(axis=1, ddof=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(sd > 0, (v - mu) / sd, np.nan)
    return spec.with_values(out)


def _upper_hull(x, y):
    """Indices of the upper convex hull of points sorted by x (monotone chain)."""
    hull: list[int] = []
    for i in range(len(x)):
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            cross = (x[a] - x[o]) * (y[i] - y[o]) - (y[a] - y[o]) * (x[i] - x[o])
            if cross >= 0:  # a is not above the line o-i: drop it
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def continuum_removed(spec: Spectra, lo: float, hi: float) -> Spectra:
    """Continuum removal over ``[lo, hi]`` nm (Clark & Roush 1984).

    Each spectrum is divided by its upper convex hull over the range, so
    absorption features become dips below 1 with the continuum at 1. Bands
    outside the range are dropped.
    """
    sub = trim(spec, lo, hi)
    x = sub.wavelengths
    out = np.full(sub.values.shape, np.nan)
    for r, y in enumerate(sub.values):
        if not np.isfinite(y).all():
            continue
        h = _upper_hull(x, y)
        cont = np.interp(x, x[h], y[h])
        out[r] = np.where(cont > 0, y / cont, np.nan)
    return sub.with_values(out)
