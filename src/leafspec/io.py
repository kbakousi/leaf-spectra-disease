"""Reading hyperspectral data: ENVI cubes (e.g. Specim IQ) and pixel-spectra tables.

ENVI files are read with numpy only (a header parser and a memory map), so no
imaging library is required. Specim IQ writes calibrated reflectance as
``results/REFLECTANCE_<name>.hdr`` with a ``.dat`` or ``.raw`` data file,
204 bands from about 397 to 1003 nm, band-interleaved-by-line (BIL)
(Behmann et al. 2018).

Pixel-spectra tables are the usual export of regions of interest: one row per
pixel with ``image, x, y, label`` and one column per wavelength. Exports often
start with extra lines (the source file path, generic column names) and may
repeat the header for each region; :func:`read_pixel_table` handles both.

References
----------
Behmann, J. et al. (2018). Specim IQ: evaluation of a new, miniaturized
    handheld hyperspectral camera and its application for plant phenotyping
    and disease detection. Sensors 18, 441. https://doi.org/10.3390/s18020441
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .spectra import Spectra

__all__ = ["read_envi_header", "read_envi", "cube_to_spectra", "read_pixel_table"]

#: ENVI ``data type`` codes.
ENVI_DTYPES = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4", 14: "i8", 15: "u8"}


def read_envi_header(path) -> dict:
    """Parse an ENVI ``.hdr`` file into a dict (lower-case keys; lists for ``{...}`` values).

    Numeric scalars (samples, lines, bands, data type, byte order, header
    offset) become ``int``; ``wavelength`` becomes a float array.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if not text.lstrip().startswith("ENVI"):
        raise ValueError(f"{path} is not an ENVI header")
    out: dict = {}
    for key, value in re.findall(r"^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)", text, flags=re.MULTILINE):
        key = key.strip().lower()
        value = value.strip()
        if value.startswith("{"):
            items = [v.strip() for v in value.strip("{}").replace("\n", " ").split(",")]
            out[key] = [v for v in items if v]
        else:
            out[key] = value
    for k in ("samples", "lines", "bands", "data type", "byte order", "header offset"):
        if k in out:
            out[k] = int(out[k])
    if "wavelength" in out:
        out["wavelength"] = np.array([float(v) for v in out["wavelength"]])
    return out


def _data_file(hdr: Path) -> Path:
    for ext in (".dat", ".raw", ".img", ".bil", ".bsq", ".bip", ""):
        p = hdr.with_suffix(ext)
        if p.exists() and p != hdr:
            return p
    raise FileNotFoundError(f"no data file found next to {hdr}")


def read_envi(path, memmap: bool = True):
    """Read an ENVI cube as ``(lines, samples, bands)`` and its wavelengths.

    Parameters
    ----------
    path : str or Path
        The ``.hdr`` file; the data file is found next to it.
    memmap : bool
        Map the file instead of reading it (fast, low memory; read-only).

    Returns
    -------
    cube : numpy.ndarray (lines, samples, bands)
    wavelengths : numpy.ndarray or None
    header : dict
    """
    hdr = Path(path)
    h = read_envi_header(hdr)
    try:
        dtype = np.dtype(ENVI_DTYPES[h["data type"]]).newbyteorder(">" if h.get("byte order", 0) == 1 else "<")
    except KeyError as exc:
        raise ValueError(f"unsupported ENVI data type {h.get('data type')}") from exc
    lines, samples, bands = h["lines"], h["samples"], h["bands"]
    interleave = h.get("interleave", "bsq").lower()
    shapes = {"bsq": (bands, lines, samples), "bil": (lines, bands, samples), "bip": (lines, samples, bands)}
    if interleave not in shapes:
        raise ValueError(f"unsupported interleave {interleave!r}")
    data_file = _data_file(hdr)
    offset = h.get("header offset", 0)
    count = lines * samples * bands
    if memmap:
        raw = np.memmap(data_file, dtype=dtype, mode="r", offset=offset, shape=(count,))
    else:
        raw = np.fromfile(data_file, dtype=dtype, count=count, offset=offset)
    arr = raw.reshape(shapes[interleave])
    order = {"bsq": (1, 2, 0), "bil": (0, 2, 1), "bip": (0, 1, 2)}[interleave]
    return arr.transpose(order), h.get("wavelength"), h


def cube_to_spectra(cube, wavelengths, mask=None, image: str | None = None, label=None) -> Spectra:
    """Pixel spectra from a cube, optionally only where ``mask`` (lines, samples) is True.

    Metadata columns: ``image``, ``x`` (sample), ``y`` (line) and ``label``.
    """
    cube = np.asarray(cube)
    lines, samples, _ = cube.shape
    if mask is None:
        mask = np.ones((lines, samples), dtype=bool)
    yy, xx = np.nonzero(mask)
    meta = pd.DataFrame({"image": image, "x": xx, "y": yy, "label": label})
    return Spectra(cube[yy, xx, :].astype("float64"), wavelengths, meta)


def _is_number(s) -> bool:
    try:
        return bool(np.isfinite(float(s)))
    except (TypeError, ValueError):
        return False


def read_pixel_table(path, meta_columns=("image", "x", "y", "label")) -> Spectra:
    """Read a pixel-spectra export (CSV or XLSX) into :class:`Spectra`.

    The header row is the first row that contains ``label`` and numeric
    column names (the wavelengths). Lines above it are skipped, repeated
    header lines and file-path lines further down are dropped, and so are
    rows without spectral values.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path, header=None, dtype=object)
    else:
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
            rows = list(csv.reader(fh))  # rows may differ in length (e.g. a lone file-path line)
        width = max((len(r) for r in rows), default=0)
        raw = pd.DataFrame([r + [None] * (width - len(r)) for r in rows], dtype=object).replace("", None)
    header_row = None
    for i, row in raw.iterrows():
        cells = [str(c).strip() for c in row.tolist()]
        if "label" in cells and sum(_is_number(c) for c in cells) >= 2:
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"{path}: no header row with 'label' and wavelength columns found")
    header = [str(c).strip() for c in raw.iloc[header_row].tolist()]
    body = raw.iloc[header_row + 1 :].copy()
    body.columns = header
    body = body.loc[:, [c for c in header if c and c.lower() != "nan" and not c.startswith("Unnamed")]]
    band_cols = [c for c in body.columns if _is_number(c)]
    repeated_header = body["label"].astype(str).str.strip().eq("label") if "label" in body else False
    body = body.loc[~repeated_header]
    values = body[band_cols].apply(pd.to_numeric, errors="coerce")
    keep = values.notna().all(axis=1).to_numpy()
    meta = body.loc[keep, [c for c in meta_columns if c in body.columns]].reset_index(drop=True)
    for c in ("x", "y"):
        if c in meta:
            meta[c] = pd.to_numeric(meta[c], errors="coerce").astype("Int64")
    wl = np.array([float(c) for c in band_cols])
    order = np.argsort(wl)
    return Spectra(values.to_numpy()[keep][:, order], wl[order], meta)
