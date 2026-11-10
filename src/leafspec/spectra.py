"""A small container that keeps spectra, wavelengths and per-spectrum metadata together."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["Spectra"]


@dataclass
class Spectra:
    """Reflectance spectra with their wavelengths and one metadata row per spectrum.

    Attributes
    ----------
    values : numpy.ndarray, shape (n, b)
        Reflectance (0-1) or a transformed quantity (e.g. SNV, derivative).
    wavelengths : numpy.ndarray, shape (b,)
        Band centres in nm, strictly increasing.
    meta : pandas.DataFrame, n rows
        E.g. ``leaf`` (or ``image``), ``x``, ``y``, ``label``, ``date``. Index is reset to 0..n-1.
    """

    values: np.ndarray
    wavelengths: np.ndarray
    meta: pd.DataFrame

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype="float64")
        self.wavelengths = np.asarray(self.wavelengths, dtype="float64")
        if self.values.ndim != 2:
            raise ValueError("values must be 2-D (spectra x bands)")
        if self.wavelengths.shape != (self.values.shape[1],):
            raise ValueError(f"{self.values.shape[1]} bands but {self.wavelengths.size} wavelengths")
        if self.wavelengths.size > 1 and not np.all(np.diff(self.wavelengths) > 0):
            raise ValueError("wavelengths must be strictly increasing")
        meta = pd.DataFrame(index=range(len(self.values))) if self.meta is None else self.meta
        if len(meta) != len(self.values):
            raise ValueError(f"{len(self.values)} spectra but {len(meta)} metadata rows")
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.values)

    def subset(self, rows) -> Spectra:
        """Rows selected by a boolean mask or integer positions."""
        rows = np.asarray(rows)
        idx = np.flatnonzero(rows) if rows.dtype == bool else rows
        return Spectra(self.values[idx], self.wavelengths, self.meta.iloc[idx])

    def band_index(self, nm: float, max_distance: float = 5.0) -> int:
        """Index of the band nearest to ``nm``; raises if none is within ``max_distance`` nm."""
        i = int(np.argmin(np.abs(self.wavelengths - nm)))
        if abs(self.wavelengths[i] - nm) > max_distance:
            raise ValueError(f"no band within {max_distance} nm of {nm} nm (nearest: {self.wavelengths[i]:.2f})")
        return i

    def at(self, nm: float, max_distance: float = 5.0) -> np.ndarray:
        """Values of the band nearest to ``nm`` (one per spectrum)."""
        return self.values[:, self.band_index(nm, max_distance)]

    def with_values(self, values, wavelengths=None) -> Spectra:
        """Same metadata, new values (and optionally new wavelengths)."""
        return Spectra(values, self.wavelengths if wavelengths is None else wavelengths, self.meta)

    @classmethod
    def concat(cls, parts) -> Spectra:
        """Stack several Spectra with identical wavelengths."""
        parts = list(parts)
        wl = parts[0].wavelengths
        for p in parts[1:]:
            if p.wavelengths.shape != wl.shape or not np.allclose(p.wavelengths, wl):
                raise ValueError("all parts must have the same wavelengths")
        return cls(np.vstack([p.values for p in parts]), wl, pd.concat([p.meta for p in parts], ignore_index=True))
