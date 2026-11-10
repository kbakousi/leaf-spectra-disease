"""Illustrative synthetic leaf spectra for tests, demos and teaching.

The generator produces pixel spectra of leaves imaged one per image, with:

- a smooth healthy leaf curve: low visible reflectance with a green peak,
  chlorophyll absorption in the red, a red edge and a near-infrared plateau
  with a water absorption feature near 970 nm;
- **leaf-to-leaf variation** (chlorophyll, red-edge position, NIR level,
  water) and a per-image illumination factor;
- an optional **date effect** (seasonal change of healthy leaves), so date and
  disease can be confounded on purpose;
- two disease patterns with a per-leaf severity, affecting only part of a
  diseased leaf's pixels (lesions): ``esca`` (loss of chlorophyll, shift of
  the red edge to shorter wavelengths, lower NIR, browning) and
  ``powdery_mildew`` (a whitish surface layer raising visible reflectance,
  mild chlorophyll loss);
- band-dependent noise, larger at both ends of the sensor range.

**The size and shape of the disease effects are illustrative choices.** They
follow the general direction of changes described for chlorophyll loss
(Horler et al. 1983) and for symptomatic tissue (Mahlein 2016). They are not
fitted to measured grapevine leaves and must not be used to draw
conclusions about real disease spectra.

References
----------
Horler, D.N.H., Dockray, M., Barber, J. (1983). The red edge of plant leaf
    reflectance. International Journal of Remote Sensing 4, 273-288.
    https://doi.org/10.1080/01431168308948546
Mahlein, A.-K. (2016). Plant disease detection by imaging sensors - parallels
    and specific demands for precision agriculture and plant phenotyping.
    Plant Disease 100, 241-251. https://doi.org/10.1094/PDIS-03-15-0340-FE
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .spectra import Spectra

__all__ = ["SPECIM_IQ_LIKE_WAVELENGTHS", "BALANCED", "CONFOUNDED", "leaf_curve", "make_leaves"]

#: 204 evenly spaced bands from 397 to 1004 nm, like a Specim IQ cube (illustrative).
SPECIM_IQ_LIKE_WAVELENGTHS = np.round(np.linspace(397.3, 1003.6, 204), 2)

#: Every date has every class: date and disease are independent.
BALANCED = {
    "2021-05-28": {"healthy": 10, "esca": 10, "powdery_mildew": 10},
    "2021-07-23": {"healthy": 10, "esca": 10, "powdery_mildew": 10},
    "2021-08-13": {"healthy": 10, "esca": 10, "powdery_mildew": 10},
}

#: Healthy leaves mostly early in the season, diseased leaves mostly late: date and disease are confounded.
CONFOUNDED = {
    "2021-05-28": {"healthy": 20},
    "2021-07-23": {"healthy": 8, "esca": 2, "powdery_mildew": 2},
    "2021-08-13": {"healthy": 2, "esca": 18, "powdery_mildew": 18},
}


def _logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def leaf_curve(wl, chl: float = 1.0, rep: float = 718.0, nir: float = 0.48, water: float = 1.0,
               white: float = 0.0, brown: float = 0.0) -> np.ndarray:
    """Reflectance (0-1) of one leaf at wavelengths ``wl`` (nm).

    ``chl`` scales chlorophyll (lower = paler, higher red and green
    reflectance), ``rep`` is the red-edge inflection (nm), ``nir`` the
    plateau level, ``water`` the depth of the 970 nm feature, ``white`` a
    broadband surface brightening (visible-weighted) and ``brown`` the
    weight of a flat brown necrotic spectrum.
    """
    wl = np.asarray(wl, dtype="float64")
    pale = max(1.0 / max(chl, 0.05) - 1.0, 0.0)
    visible = (0.035 + 0.07 * np.exp(-(((wl - 552.0) / 32.0) ** 2)) * (1 + 0.6 * pale)
               + 0.09 * pale * np.exp(-(((wl - 640.0) / 45.0) ** 2))
               + 0.01 * _logistic((440.0 - wl) / 8.0))
    plateau = nir - 0.035 * water * np.exp(-(((wl - 970.0) / 22.0) ** 2))
    s = _logistic((wl - rep) / 11.0)
    r = visible * (1 - s) + plateau * s
    r = r + white * (0.10 * np.exp(-(((wl - 450.0) / 180.0) ** 2)) + 0.02)
    necrotic = 0.06 + 0.30 * _logistic((wl - 600.0) / 60.0)
    return (1 - brown) * r + brown * necrotic


def _noise_sd(wl):
    return 0.004 + 0.02 * np.exp(-(wl - wl.min()) / 12.0) + 0.02 * np.exp((wl - wl.max()) / 18.0)


def make_leaves(design: dict | None = None, pixels_per_leaf: int = 150, seed: int = 0,
                severity_range=(0.25, 1.0), date_effect: float = 1.0, wavelengths=None) -> Spectra:
    """Pixel spectra for a set of leaves, one leaf per image.

    Parameters
    ----------
    design : dict
        ``{date: {label: n_leaves}}``; see :data:`BALANCED` (default) and
        :data:`CONFOUNDED`. Labels: ``healthy``, ``esca``, ``powdery_mildew``.
    pixels_per_leaf : int
    seed : int
    severity_range : (low, high)
        Per-leaf severity of diseased leaves is uniform in this range; low
        values give mild, hard-to-detect leaves.
    date_effect : float
        Strength of the seasonal change of healthy leaves (0 = none).
    wavelengths : array_like, optional
        Default: :data:`SPECIM_IQ_LIKE_WAVELENGTHS`.

    Returns
    -------
    Spectra with metadata ``leaf``, ``x``, ``y``, ``label``, ``date``,
    ``severity`` (0 for healthy) and ``lesion`` (pixel-level truth).
    """
    design = BALANCED if design is None else design
    wl = SPECIM_IQ_LIKE_WAVELENGTHS if wavelengths is None else np.asarray(wavelengths, dtype="float64")
    rng = np.random.default_rng(seed)
    dates = sorted(design)
    noise = _noise_sd(wl)
    side = int(np.ceil(np.sqrt(pixels_per_leaf)))
    values, metas, k = [], [], 0
    for d_idx, date in enumerate(dates):
        season = d_idx / max(len(dates) - 1, 1)  # 0 early -> 1 late
        for label, n in design[date].items():
            if label not in ("healthy", "esca", "powdery_mildew"):
                raise ValueError(f"unknown label {label!r}")
            for _ in range(n):
                k += 1
                base = {
                    "chl": rng.normal(1.0, 0.10) + 0.15 * date_effect * season,
                    "rep": rng.normal(717.0, 2.0) + 3.0 * date_effect * season,
                    "nir": rng.normal(0.48, 0.035) - 0.03 * date_effect * season,
                    "water": rng.normal(1.0, 0.15),
                }
                sev = 0.0 if label == "healthy" else rng.uniform(*severity_range)
                lesion = rng.random(pixels_per_leaf) < (0.85 * sev if sev else 0.0)
                illum = rng.uniform(0.85, 1.15)
                rows = np.empty((pixels_per_leaf, wl.size))
                for p in range(pixels_per_leaf):
                    q = dict(base)
                    q["chl"] *= rng.normal(1.0, 0.03)
                    s = sev if lesion[p] else 0.15 * sev  # mild systemic effect outside lesions
                    white = brown = 0.0
                    if label == "esca":
                        q["chl"] *= 1 - 0.6 * s
                        q["rep"] -= 12.0 * s
                        q["nir"] *= 1 - 0.15 * s
                        brown = 0.35 * s
                    elif label == "powdery_mildew":
                        q["chl"] *= 1 - 0.15 * s
                        white = 0.6 * s
                    rows[p] = illum * leaf_curve(wl, white=white, brown=brown, **q) + rng.normal(0, noise)
                values.append(rows)
                pix = np.arange(pixels_per_leaf)
                metas.append(pd.DataFrame({"leaf": f"L{k:03d}", "x": pix % side, "y": pix // side, "label": label,
                                           "date": date, "severity": round(sev, 3), "lesion": lesion}))
    return Spectra(np.vstack(values), wl, pd.concat(metas, ignore_index=True))
