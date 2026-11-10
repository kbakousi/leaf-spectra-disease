"""Narrow-band vegetation indices from leaf spectra.

Each index uses the band nearest to each stated wavelength; a ``ValueError``
is raised if no band lies within ``max_distance`` nm, so an index is never
silently computed from the wrong part of the spectrum.

Disease-specific spectral indices published for other crops (e.g. the
sugar-beet indices of Mahlein et al. 2013) are deliberately not included as
ready-made functions: they were derived for one host and pathogen. Use
:func:`normalized_difference` to test a published band pair on your data.

References
----------
Rouse, J.W., Haas, R.H., Schell, J.A., Deering, D.W. (1974). Monitoring
    vegetation systems in the Great Plains with ERTS. NASA SP-351, 309-317.
Gamon, J.A., Penuelas, J., Field, C.B. (1992). A narrow-waveband spectral
    index that tracks diurnal changes in photosynthetic efficiency. Remote
    Sensing of Environment 41, 35-44. https://doi.org/10.1016/0034-4257(92)90059-S
Horler, D.N.H., Dockray, M., Barber, J. (1983). The red edge of plant leaf
    reflectance. International Journal of Remote Sensing 4, 273-288.
    https://doi.org/10.1080/01431168308948546
Mahlein, A.-K. et al. (2013). Development of spectral indices for detecting
    and identifying plant diseases. Remote Sensing of Environment 128, 21-30.
    https://doi.org/10.1016/j.rse.2012.09.019
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .spectra import Spectra

__all__ = ["normalized_difference", "ndvi", "pri", "red_edge_position", "index_table"]


def normalized_difference(spec: Spectra, a_nm: float, b_nm: float, max_distance: float = 5.0) -> np.ndarray:
    """``(R_a - R_b) / (R_a + R_b)`` per spectrum, from the bands nearest to ``a_nm`` and ``b_nm``."""
    a, b = spec.at(a_nm, max_distance), spec.at(b_nm, max_distance)
    den = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(den) > 1e-12, (a - b) / den, np.nan)


def ndvi(spec: Spectra, nir_nm: float = 800.0, red_nm: float = 670.0) -> np.ndarray:
    """Narrow-band NDVI, ``(R800 - R670) / (R800 + R670)`` by default (after Rouse et al. 1974)."""
    return normalized_difference(spec, nir_nm, red_nm)


def pri(spec: Spectra) -> np.ndarray:
    """Photochemical Reflectance Index, ``(R531 - R570) / (R531 + R570)`` (Gamon et al. 1992)."""
    return normalized_difference(spec, 531.0, 570.0)


def red_edge_position(spec: Spectra, lo: float = 680.0, hi: float = 750.0, window: int = 7) -> np.ndarray:
    """Wavelength (nm) of the steepest rise of reflectance between ``lo`` and ``hi``.

    The red-edge inflection point (Horler et al. 1983), located as the
    maximum of the Savitzky-Golay first derivative, refined with a parabola
    through the maximum and its two neighbours. It shifts to shorter
    wavelengths when chlorophyll content falls.
    """
    sel = (spec.wavelengths >= lo) & (spec.wavelengths <= hi)
    wl = spec.wavelengths[sel]
    if wl.size < window:
        raise ValueError(f"need at least {window} bands between {lo} and {hi} nm")
    d = savgol_filter(spec.values[:, sel], window, 2, deriv=1, delta=float(np.mean(np.diff(wl))), axis=1)
    k = np.clip(np.nanargmax(np.where(np.isfinite(d), d, -np.inf), axis=1), 1, wl.size - 2)
    rows = np.arange(len(k))
    y0, y1, y2 = d[rows, k - 1], d[rows, k], d[rows, k + 1]
    den = y0 - 2 * y1 + y2
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = np.where(np.abs(den) > 1e-15, 0.5 * (y0 - y2) / den, 0.0)
    step = float(np.mean(np.diff(wl)))
    out = wl[k] + np.clip(shift, -1, 1) * step
    out[~np.isfinite(spec.values[:, sel]).all(axis=1)] = np.nan
    return out


def index_table(spec: Spectra) -> pd.DataFrame:
    """NDVI, PRI and red-edge position for every spectrum, with the metadata."""
    return spec.meta.assign(ndvi=ndvi(spec), pri=pri(spec), red_edge_nm=red_edge_position(spec))
