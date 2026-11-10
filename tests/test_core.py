"""Reading, preprocessing, indices and aggregation."""

import numpy as np
import pandas as pd
import pytest

from leafspec import Spectra, aggregate, indices, io, preprocess, synthetic

WL = np.linspace(400, 1000, 121)  # 5 nm steps


def spectra(values, meta=None):
    values = np.atleast_2d(values)
    return Spectra(values, WL[: values.shape[1]], meta if meta is not None else pd.DataFrame(index=range(len(values))))


# --------------------------------------------------------------------------- container


def test_spectra_validates_shapes_and_order():
    with pytest.raises(ValueError, match="wavelengths"):
        Spectra(np.zeros((2, 3)), [400, 410], None)
    with pytest.raises(ValueError, match="increasing"):
        Spectra(np.zeros((1, 3)), [400, 420, 410], None)
    s = spectra(np.arange(121.0))
    assert s.band_index(702) == 60 and s.at(700)[0] == 60
    with pytest.raises(ValueError, match="no band within"):
        s.band_index(1100)


# --------------------------------------------------------------------------- io


@pytest.mark.parametrize("interleave", ["bsq", "bil", "bip"])
def test_envi_round_trip(tmp_path, interleave):
    rng = np.random.default_rng(0)
    cube = rng.random((4, 5, 3)).astype("float32")  # lines, samples, bands
    layout = {"bsq": (2, 0, 1), "bil": (0, 2, 1), "bip": (0, 1, 2)}[interleave]
    cube.transpose(layout).tofile(tmp_path / "img.dat")
    (tmp_path / "img.hdr").write_text(
        "ENVI\ndescription = {\ntest cube\n}\nsamples = 5\nlines = 4\nbands = 3\nheader offset = 0\n"
        f"data type = 4\ninterleave = {interleave}\nbyte order = 0\nwavelength = {{\n500.0,\n600.5,\n700.25\n}}\n"
    )
    arr, wl, header = io.read_envi(tmp_path / "img.hdr")
    assert np.array_equal(arr, cube) and wl.tolist() == [500.0, 600.5, 700.25]
    assert header["description"] == ["test cube"]
    mask = np.zeros((4, 5), bool)
    mask[1, 2] = mask[3, 0] = True
    s = io.cube_to_spectra(arr, wl, mask, image="img", label="healthy")
    assert s.meta[["x", "y"]].values.tolist() == [[2, 1], [0, 3]]
    assert np.allclose(s.values[0], cube[1, 2])


def test_pixel_table_with_path_line_and_repeated_headers(tmp_path):
    p = tmp_path / "export.csv"
    p.write_text(
        "D:/somewhere/REFLECTANCE_a.hdr\n"  # a lone path line, as written by common ROI exports
        "image,x,y,label,500.0,600.0,700.0\n"
        "a,1,2,7,0.1,0.2,0.3\n"
        "a,2,2,7,0.11,0.21,0.31\n"
        "D:/somewhere/REFLECTANCE_b.hdr\n"
        "image,x,y,label,500.0,600.0,700.0\n"
        "b,5,5,8,0.12,0.22,0.32\n"
    )
    s = io.read_pixel_table(p)
    assert len(s) == 3 and s.wavelengths.tolist() == [500.0, 600.0, 700.0]
    assert s.meta["image"].tolist() == ["a", "a", "b"] and s.meta["x"].tolist() == [1, 2, 5]
    assert s.values[2].tolist() == [0.12, 0.22, 0.32]


def test_pixel_table_xlsx(tmp_path):
    pytest.importorskip("openpyxl")
    rows = [["Column1", "Column2", "Column3", "Column4", "Column5", "Column6"],
            ["C:/x/REFLECTANCE_c.hdr", None, None, None, None, None],
            ["image", "x", "y", "label", "450.5", "550.5"],
            ["c", 3, 4, "002", 0.05, 0.12]]
    pd.DataFrame(rows).to_excel(tmp_path / "c.xlsx", header=False, index=False)
    s = io.read_pixel_table(tmp_path / "c.xlsx")
    assert len(s) == 1 and s.wavelengths.tolist() == [450.5, 550.5] and s.meta["label"].iloc[0] == "002"


# --------------------------------------------------------------------------- preprocessing


def test_trim_snv_and_savgol():
    s = spectra(np.vstack([np.linspace(0.1, 0.5, 121), 2 * np.linspace(0.1, 0.5, 121)]))
    t = preprocess.trim(s, 450, 900)
    assert t.wavelengths.min() >= 450 and t.wavelengths.max() <= 900
    z = preprocess.snv(s).values
    assert np.allclose(z.mean(axis=1), 0) and np.allclose(z.std(axis=1, ddof=1), 1)
    assert np.allclose(z[0], z[1])  # SNV removes the multiplicative difference
    slope = preprocess.savgol(s, window=7, polyorder=2, deriv=1).values
    assert np.allclose(slope[0], 0.4 / 600)  # per nm
    quad = spectra(1e-6 * (WL - 700) ** 2)
    assert np.allclose(preprocess.savgol(quad, 9, 2).values, quad.values)


def test_continuum_removal():
    y = 0.5 - 0.2 * np.exp(-(((WL - 670) / 30) ** 2)) + 0.0002 * (WL - 400)
    cr = preprocess.continuum_removed(spectra(y), 500, 800).values[0]
    assert np.all(cr <= 1 + 1e-12) and np.isclose(cr.max(), 1)
    assert np.argmin(cr) == np.argmin(np.abs(np.linspace(500, 800, cr.size) - 670))


def test_quality_mask_flags():
    leaf = synthetic.leaf_curve(WL)
    soil = 0.1 + 0.15 * (WL - 400) / 600
    s = spectra(np.vstack([leaf, soil, leaf * 0.2, np.where(WL > 900, 1.2, leaf), leaf]))
    s.values[4, 3] = np.nan
    q = preprocess.quality_mask(s, max_reflectance=1.0, min_nir=0.2, min_ndvi=0.3)
    assert q["ok"].tolist() == [True, False, False, False, False]
    assert q.loc[1, "not_leaf"] and q.loc[2, "shadow"] and q.loc[3, "saturated"] and q.loc[4, "non_finite"]


# --------------------------------------------------------------------------- indices


def test_indices_on_known_curves():
    s = spectra(np.vstack([synthetic.leaf_curve(WL, rep=r) for r in (705.0, 718.0, 730.0)]))
    rep = indices.red_edge_position(s)
    assert np.allclose(rep, [705, 718, 730], atol=2.0)
    flat = spectra(np.full(121, 0.3))
    assert indices.ndvi(flat)[0] == 0 and indices.pri(flat)[0] == 0
    tab = indices.index_table(s)
    assert list(tab.columns) == ["ndvi", "pri", "red_edge_nm"] and (tab["ndvi"] > 0.6).all()


def test_paler_leaf_has_lower_ndvi_and_shorter_red_edge():
    s = spectra(np.vstack([synthetic.leaf_curve(WL, chl=1.0), synthetic.leaf_curve(WL, chl=0.5, rep=708)]))
    n = indices.ndvi(s)
    assert n[1] < n[0] and indices.red_edge_position(s)[1] < indices.red_edge_position(s)[0]


# --------------------------------------------------------------------------- aggregation


def test_leaf_spectra_median_counts_and_label_conflict():
    meta = pd.DataFrame({"leaf": ["a", "a", "a", "b"], "label": ["h", "h", "h", "d"]})
    s = Spectra(np.array([[1.0, 2], [3, 4], [100, 200], [5, 6]]), [500, 600], meta)
    leaves = aggregate.leaf_spectra(s, by="leaf")
    assert leaves.values.tolist() == [[3.0, 4.0], [5.0, 6.0]]
    assert leaves.meta["n_pixels"].tolist() == [3, 1] and leaves.meta["label"].tolist() == ["h", "d"]
    assert len(aggregate.leaf_spectra(s, by="leaf", min_pixels=2)) == 1
    meta.loc[1, "label"] = "d"
    with pytest.raises(ValueError, match="different values of 'label'"):
        aggregate.leaf_spectra(Spectra(s.values, s.wavelengths, meta), by="leaf")
