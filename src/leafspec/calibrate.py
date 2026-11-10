"""Radiometric checks and corrections for imaging-spectrometer data (e.g. Specim IQ).

Reflectance from a push-broom camera is computed from the raw image, a dark
reference and a white reference::

    R = (raw - dark) / (white - dark) * panel_reflectance

This is exact only if the white panel received the same light as the
target. In the field it often did not: a panel in shade or at another angle
than sunlit leaves makes every reflectance too high, and a custom white
reference recorded earlier under different light shifts a whole session.
Such errors cannot be removed exactly afterwards. This module provides:

- :func:`reflectance_from_raw`: recompute reflectance from raw data, e.g.
  with another panel reflectance or another white reference;
- :func:`saturation_mask`: pixels whose raw signal is clipped in any band
  (their spectral shape is distorted and should not be used);
- :func:`spectral_angle` and :func:`find_reference_panel`: locate a
  spectrally neutral white panel in the image (spectral angle mapper);
- :func:`vector_normalise`: unit-length spectra, removing a per-spectrum
  brightness or gain factor;
- :func:`session_reference_ratio`: divide each spectrum by the median
  healthy (reference) spectrum of its own session, e.g. date x block. A
  calibration error shared by a session (one custom white reference, the
  same sky) and the seasonal change of healthy leaves both cancel; what
  remains is how each leaf differs from healthy leaves measured under the
  same conditions.

References
----------
Behmann, J. et al. (2018). Specim IQ: evaluation of a new, miniaturized
    handheld hyperspectral camera and its application for plant phenotyping
    and disease detection. Sensors 18, 441. https://doi.org/10.3390/s18020441
Kruse, F.A. et al. (1993). The spectral image processing system (SIPS) -
    interactive visualization and analysis of imaging spectrometer data.
    Remote Sensing of Environment 44, 145-163.
    https://doi.org/10.1016/0034-4257(93)90013-N (spectral angle mapper)
Rinnan, A., van den Berg, F., Engelsen, S.B. (2009). Review of the most
    common pre-processing techniques for near-infrared spectra. TrAC Trends
    in Analytical Chemistry 28, 1201-1222. https://doi.org/10.1016/j.trac.2009.07.007
    (normalisation of spectra)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .spectra import Spectra

__all__ = ["reflectance_from_raw", "saturation_mask", "spectral_angle", "find_reference_panel", "vector_normalise",
           "session_reference_ratio"]


def _ref(arr, n_bands):
    """Dark/white reference as (samples, bands) or (bands,), broadcastable over (lines, samples, bands)."""
    a = np.asarray(arr, dtype="float64")
    if a.ndim == 3:  # (lines, samples, bands): average the lines
        a = a.mean(axis=0)
    if a.shape[-1] != n_bands:
        raise ValueError(f"reference has {a.shape[-1]} bands, image has {n_bands}")
    return a


def reflectance_from_raw(raw, dark, white, panel_reflectance=1.0) -> np.ndarray:
    """``(raw - dark) / (white - dark) * panel_reflectance`` for a (lines, samples, bands) cube.

    Parameters
    ----------
    raw : array (lines, samples, bands)
    dark, white : array
        (lines, samples, bands), (samples, bands) or (bands,); several lines
        are averaged. ``white`` is the signal of the reference panel.
    panel_reflectance : float or array (bands,)
        Reflectance of the panel (1.0 for an ideal white panel; use the
        calibration certificate of the panel if available).

    Returns NaN where ``white - dark <= 0``.
    """
    raw = np.asarray(raw, dtype="float64")
    b = raw.shape[-1]
    d, w = _ref(dark, b), _ref(white, b)
    den = w - d
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (raw - d) / np.where(den > 0, den, np.nan)
    return out * np.asarray(panel_reflectance, dtype="float64")


def saturation_mask(raw, max_dn: float) -> np.ndarray:
    """Boolean (lines, samples): True where any band reaches ``max_dn`` (e.g. 4080 for a 12-bit sensor)."""
    return np.asarray(raw >= max_dn).any(axis=-1)


def spectral_angle(spectra, reference) -> np.ndarray:
    """Angle in degrees between each spectrum (last axis) and ``reference`` (Kruse et al. 1993).

    The angle ignores overall brightness: 0 means the same spectral shape.
    """
    x = np.asarray(spectra, dtype="float64")
    r = np.asarray(reference, dtype="float64")
    num = x @ r
    den = np.linalg.norm(x, axis=-1) * np.linalg.norm(r)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(den > 0, num / den, np.nan)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def find_reference_panel(raw, dark, white, max_angle_deg: float = 3.0, band_mask=None) -> np.ndarray:
    """Pixels whose dark-corrected signal has the spectral shape of the white reference.

    A white or grey panel is spectrally flat in reflectance, so its raw
    signal has the same shape as the white-reference signal. Leaves, soil
    and shadows do not.

    Parameters
    ----------
    band_mask : boolean array (bands,), optional
        Bands to compare (e.g. exclude noisy edge bands).

    Returns
    -------
    Boolean (lines, samples) mask.
    """
    raw = np.asarray(raw, dtype="float64")
    b = raw.shape[-1]
    d, w = _ref(dark, b), _ref(white, b)
    sel = np.ones(b, dtype=bool) if band_mask is None else np.asarray(band_mask, dtype=bool)
    shape = (w - d)
    shape = shape.mean(axis=0) if shape.ndim == 2 else shape
    return spectral_angle((raw - d)[..., sel], shape[sel]) < max_angle_deg


def vector_normalise(spec: Spectra) -> Spectra:
    """Divide each spectrum by its Euclidean norm (unit-length spectra).

    Removes a brightness or gain factor per spectrum (illumination intensity,
    white-reference scale); the spectral shape is kept.
    """
    v = spec.values
    n = np.linalg.norm(v, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return spec.with_values(np.where(n > 0, v / n, np.nan))


def session_reference_ratio(spec: Spectra, session_cols, reference, min_reference: int = 1,
                            stat: str = "median") -> tuple[Spectra, pd.DataFrame]:
    """Divide every spectrum by the reference spectrum of its own session.

    Parameters
    ----------
    session_cols : str or list of str
        Metadata columns that define a session (e.g. ``["date", "block"]``).
    reference : array_like of bool, or str
        Which rows are reference (healthy) spectra: a boolean array, or a
        metadata column name holding booleans.
    min_reference : int
        Sessions with fewer reference spectra raise a ``ValueError``, so that
        no session is normalised by an unreliable reference.
    stat : {"median", "mean"}

    Returns
    -------
    ratio : Spectra
        Spectra divided band by band by their session's reference spectrum
        (healthy rows centre on 1).
    references : pandas.DataFrame
        One row per session: the number of reference spectra used.
    """
    cols = [session_cols] if isinstance(session_cols, str) else list(session_cols)
    ref = spec.meta[reference].to_numpy(dtype=bool) if isinstance(reference, str) else np.asarray(reference, bool)
    if ref.shape != (len(spec),):
        raise ValueError("reference must have one value per spectrum")
    out = np.full(spec.values.shape, np.nan)
    rows = []
    for key, idx in spec.meta.groupby(cols, sort=True, dropna=False).indices.items():
        r_idx = idx[ref[idx]]
        key = key if isinstance(key, tuple) else (key,)
        if len(r_idx) < min_reference:
            raise ValueError(f"session {dict(zip(cols, key, strict=True))} has {len(r_idx)} reference spectra "
                             f"(< {min_reference})")
        block = spec.values[r_idx]
        r = np.nanmedian(block, axis=0) if stat == "median" else np.nanmean(block, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[idx] = spec.values[idx] / np.where(r != 0, r, np.nan)
        rows.append({**dict(zip(cols, key, strict=True)), "n_reference": len(r_idx), "n_total": len(idx)})
    return spec.with_values(out), pd.DataFrame(rows)
