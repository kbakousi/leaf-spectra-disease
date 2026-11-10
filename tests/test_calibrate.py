"""Radiometric helpers on scenes with a known answer."""

import numpy as np
import pandas as pd
import pytest

from leafspec import Spectra, calibrate, synthetic

WL = np.linspace(400, 1000, 61)


def scene(panel_light=1.0, seed=0):
    """Raw cube of leaves (true reflectance known), a white panel patch and two saturated pixels.

    ``panel_light`` < 1 means the panel received less light than the leaves (e.g. in shade).
    """
    rng = np.random.default_rng(seed)
    lines, samples = 20, 30
    true = np.empty((lines, samples, WL.size))
    for i in range(lines):
        for j in range(samples):
            true[i, j] = synthetic.leaf_curve(WL, chl=rng.normal(1, 0.1), nir=rng.normal(0.48, 0.03))
    true[2:6, 3:9] = 0.99  # white panel area, true reflectance 0.99
    dark = np.full((samples, WL.size), 100.0)
    gain = 3000 * np.exp(-(((WL - 650) / 300) ** 2))  # sensor/illumination response
    raw = dark + true * gain
    raw[2:6, 3:9] = dark[None, 3:9] + 0.99 * gain * panel_light  # panel lit by panel_light x the leaf light
    white = dark + 0.99 * gain * panel_light  # what the camera measured on the panel
    raw[10, 10, 30] = raw[11, 12, 5] = 4095
    return raw, dark, white, true


def test_reflectance_from_raw_is_exact_when_panel_and_leaves_share_the_light():
    raw, dark, white, true = scene()
    r = calibrate.reflectance_from_raw(raw, dark, white, panel_reflectance=0.99)
    ok = ~calibrate.saturation_mask(raw, 4080)
    assert np.allclose(r[ok], true[ok])


def test_shaded_panel_inflates_reflectance_by_a_constant_factor():
    raw, dark, white, true = scene(panel_light=0.5)
    r = calibrate.reflectance_from_raw(raw, dark, white, panel_reflectance=0.99)
    leaf = np.ones(r.shape[:2], bool)
    leaf[2:6, 3:9] = False
    leaf &= ~calibrate.saturation_mask(raw, 4080)
    ratio = r[leaf] / true[leaf]
    assert np.allclose(ratio, 2.0)  # every leaf reflectance doubled
    n_true = true[leaf] / np.linalg.norm(true[leaf], axis=1, keepdims=True)
    n_r = r[leaf] / np.linalg.norm(r[leaf], axis=1, keepdims=True)
    assert np.allclose(n_true, n_r)  # ...but the spectral shape is unchanged


def test_saturation_and_panel_detection():
    raw, dark, white, _ = scene(panel_light=0.7)
    sat = calibrate.saturation_mask(raw, 4080)
    assert sat.sum() == 2 and sat[10, 10] and sat[11, 12]
    panel = calibrate.find_reference_panel(raw, dark, white, max_angle_deg=1.0)
    expected = np.zeros(panel.shape, bool)
    expected[2:6, 3:9] = True
    assert np.array_equal(panel, expected)


def test_spectral_angle_ignores_brightness():
    a = synthetic.leaf_curve(WL)
    assert calibrate.spectral_angle(np.vstack([a, 3 * a]), a) == pytest.approx([0, 0], abs=1e-5)
    assert calibrate.spectral_angle(np.ones(5), np.array([1, 0, 0, 0, 0.0])) == pytest.approx(63.43, abs=0.01)


def test_vector_normalise():
    s = Spectra(np.array([[3.0, 4.0], [0.0, 0.0]]), [500, 600], None)
    v = calibrate.vector_normalise(s).values
    assert np.allclose(v[0], [0.6, 0.8]) and np.isnan(v[1]).all()


def test_session_reference_ratio_cancels_session_bias_and_season():
    rng = np.random.default_rng(1)
    rows, meta = [], []
    healthy, diseased = synthetic.leaf_curve(WL), synthetic.leaf_curve(WL, chl=0.6, rep=708)
    for session, bias in [("may", 1.0 + 0.0 * WL), ("july", 2.5 * (1 + 0.0005 * (WL - 700))), ("aug", 1.7 + 0 * WL)]:
        for label, curve, n in [("healthy", healthy, 6), ("esca", diseased, 3)]:
            for _ in range(n):
                rows.append(curve * bias * rng.normal(1, 0.002, WL.size))
                meta.append({"session": session, "label": label})
    s = Spectra(np.vstack(rows), WL, pd.DataFrame(meta))
    ratio, refs = calibrate.session_reference_ratio(s, "session", s.meta["label"].eq("healthy").to_numpy())
    assert refs["n_reference"].tolist() == [6, 6, 6]
    h = ratio.values[s.meta["label"].eq("healthy").to_numpy()]
    assert np.allclose(h, 1.0, atol=0.01)
    d = ratio.values[s.meta["label"].eq("esca").to_numpy()]
    expected = diseased / healthy
    assert np.allclose(d, expected[None, :], rtol=0.02)  # identical in every session despite different biases
    with pytest.raises(ValueError, match="reference spectra"):
        calibrate.session_reference_ratio(s, "session", s.meta["label"].eq("healthy").to_numpy(), min_reference=7)
