# leaf-spectra-disease

[![Tests](https://github.com/kbakousi/leaf-spectra-disease/actions/workflows/ci.yml/badge.svg)](https://github.com/kbakousi/leaf-spectra-disease/actions/workflows/ci.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24118758.svg)](https://doi.org/10.5281/zenodo.24118758)

Leaf-level hyperspectral analysis for plant disease detection. The package covers reading Specim IQ / ENVI data and pixel-spectra exports, spectral preprocessing, narrow-band indices, leaf-level summaries, classifiers and one-class screening, and **validation that holds out whole leaves and dates**. Grapevine leaves with esca-like and powdery-mildew-like symptoms are the worked example.

- **Methods follow the literature.** Each module is an independent implementation of the methods described in the publications cited in its docstring (see [References](#references)).
- **Honest validation is the point.** Pixels of one leaf are near-copies, so random pixel splits overstate accuracy. Disease and sampling date are often confounded in field campaigns. Both problems are measured here, not hidden.
- **No hidden choices.** Quality thresholds, band ranges and model settings are explicit arguments.
- **Light dependencies.** numpy, pandas, scipy and scikit-learn. ENVI files are read without an imaging library.

## Install

```bash
pip install -e .            # core
pip install -e ".[xlsx]"    # + reading .xlsx exports (openpyxl)
pip install -e ".[dev]"     # + pytest, ruff, notebook tools
```

## Modules

| Module | What it does | Method references |
|---|---|---|
| `Spectra` | Container keeping spectra, wavelengths and per-spectrum metadata together; nearest-band lookup with a distance check | — |
| `io` | ENVI header parser and reader (BSQ, BIL, BIP; memory-mapped), cube → pixel spectra, pixel-spectra exports (CSV/XLSX with path lines and repeated headers) | Behmann et al. 2018 |
| `calibrate` | Reflectance from raw, dark and white references; raw saturation mask; white-panel detection by spectral angle; vector (L2) normalisation; **division by the healthy reference of the same session** (cancels session-wide calibration and seasonal effects) | Behmann et al. 2018; Kruse et al. 1993; Rinnan et al. 2009 |
| `preprocess` | Band trimming; quality mask (non-finite, saturated, shadow, not leaf); Savitzky–Golay smoothing and derivatives; standard normal variate; continuum removal | Savitzky & Golay 1964; Barnes et al. 1989; Clark & Roush 1984; Rinnan et al. 2009 |
| `indices` | Narrow-band NDVI, PRI, red-edge position (maximum first derivative, sub-band refinement), generic normalised difference | Rouse et al. 1974; Gamon et al. 1992; Horler et al. 1983 |
| `aggregate` | Pixel → leaf spectra (median or mean), pixel count, within-leaf spread; refuses leaves with conflicting labels | — |
| `models` | Pipelines (scaling and optional PCA fitted per fold): PLS-DA, RBF SVM, random forest, logistic regression; one-class SVM for anomaly screening | Barker & Rayens 2003; Breiman 2001; Schölkopf et al. 2001 |
| `validate` | Grouped folds (whole leaves held out), random vs grouped comparison, leave-one-date-out, one-class scores from models trained on healthy leaves only | Roberts et al. 2017; Ploton et al. 2020 |
| `bands` | Window-wise permutation importance on held-out leaves (correlated neighbouring bands are permuted together) | Breiman 2001; Mahlein 2016 |
| `synthetic` | Illustrative leaf spectra: leaf-to-leaf variation, seasonal effect, esca-like and mildew-like lesions with per-leaf severity, band-dependent noise; balanced and confounded designs | Horler et al. 1983; Mahlein 2016 (directions only; not fitted to measured leaves) |

## Quick start

```python
from leafspec import aggregate, models, preprocess, synthetic, validate

px = synthetic.make_leaves(synthetic.BALANCED, pixels_per_leaf=120, seed=4)
q = preprocess.quality_mask(px, max_reflectance=1.0, min_nir=0.1, min_ndvi=0.2)
px = preprocess.trim(px.subset(q["ok"].to_numpy()), 430, 900)

leaves = aggregate.leaf_spectra(px, by="leaf")                  # one spectrum per leaf
y, g = leaves.meta["label"].to_numpy(), leaves.meta["leaf"].to_numpy()
pred = validate.out_of_fold(models.make_classifier("plsda", n_components=6), leaves.values, y,
                            validate.grouped_folds(y, g, n_splits=5))
print(validate.scores(y, pred)["balanced_accuracy"])
```

For real Specim IQ data, start from `io.read_envi("…/results/REFLECTANCE_<name>.hdr")` or from `io.read_pixel_table(export)`.

## Tutorial

[`tutorials/01_leaf_disease_walkthrough.ipynb`](tutorials/01_leaf_disease_walkthrough.ipynb) runs offline in about a minute. On synthetic leaves it shows:

- how a strict leaf mask removes the very lesions being studied;
- a **random pixel split scoring 0.98 against 0.75 with whole leaves held out**, and 0.90 for leaf spectra;
- a model trained on a campaign where **disease and date are confounded**: 0.97 in cross-validation, 0.69 on an independent campaign, with most healthy late-season leaves flagged as diseased;
- one-class screening trained on healthy leaves only;
- which wavelength windows the model relies on.

## Scientific notes

- **Validate by leaf (or plant), never by pixel.** A random pixel split measures how well a model recognises leaves it has already seen.
- **Check the design before the accuracy.** If healthy leaves come mostly from one date and diseased leaves from another, compare classes within a date or hold out whole dates.
- **Early detection needs repeated measurements of the same leaves or vines before symptoms appear.** Separating symptomatic from healthy leaves on one date is a different, easier claim.
- **Field labels are often visual ("suspected").** Report them as such; laboratory confirmation changes what can be claimed.
- **Reflectance level and calibration.** A healthy leaf has a near-infrared plateau of roughly 0.4–0.5. Values far above or below point to calibration. A white panel in shade, or a custom white reference recorded under other light, scales a whole capture or session. That cannot be undone exactly afterwards. Remove saturated pixels, use shape features (vector normalisation, SNV), and compare leaves with healthy leaves of the **same session** (`calibrate.session_reference_ratio`). Exported spectra may already be normalised: check their norms before treating them as reflectance.
- **Published disease indices are host-specific.** The sugar-beet indices of Mahlein et al. (2013) are not built in. Test a published band pair with `indices.normalized_difference` and validate it on your own leaves.

## Tests

```bash
pytest
```

The tests run offline. They cover:
- ENVI round trips in all three interleaves;
- pixel-table exports with path lines and repeated headers (CSV and XLSX);
- SNV, Savitzky–Golay and continuum removal against known curves;
- calibration: exact recovery of reflectance from raw, a shaded panel inflating reflectance but not spectral shape, panel and saturation detection, and session-reference ratios cancelling session biases;
- quality flags;
- index values and red-edge position on curves with a known inflection;
- leaf aggregation, including label conflicts;
- grouped folds that never split a leaf;
- the random-vs-grouped gap;
- classifiers and one-class screening;
- confounding by date;
- window importance finding an injected spectral difference;
- the tutorial notebook, which is executed as a test.

## References

- Barker, M., Rayens, W. (2003). Partial least squares for discrimination. *Journal of Chemometrics* 17, 166–173. https://doi.org/10.1002/cem.785
- Barnes, R.J., Dhanoa, M.S., Lister, S.J. (1989). Standard normal variate transformation and de-trending of near-infrared diffuse reflectance spectra. *Applied Spectroscopy* 43, 772–777. https://doi.org/10.1366/0003702894202201
- Behmann, J. et al. (2018). Specim IQ: evaluation of a new, miniaturized handheld hyperspectral camera and its application for plant phenotyping and disease detection. *Sensors* 18, 441. https://doi.org/10.3390/s18020441
- Breiman, L. (2001). Random forests. *Machine Learning* 45, 5–32. https://doi.org/10.1023/A:1010933404324
- Clark, R.N., Roush, T.L. (1984). Reflectance spectroscopy: quantitative analysis techniques for remote sensing applications. *Journal of Geophysical Research* 89, 6329–6340. https://doi.org/10.1029/JB089iB07p06329
- Gamon, J.A., Peñuelas, J., Field, C.B. (1992). A narrow-waveband spectral index that tracks diurnal changes in photosynthetic efficiency. *Remote Sensing of Environment* 41, 35–44. https://doi.org/10.1016/0034-4257(92)90059-S
- Horler, D.N.H., Dockray, M., Barber, J. (1983). The red edge of plant leaf reflectance. *International Journal of Remote Sensing* 4, 273–288. https://doi.org/10.1080/01431168308948546
- Kruse, F.A. et al. (1993). The spectral image processing system (SIPS) – interactive visualization and analysis of imaging spectrometer data. *Remote Sensing of Environment* 44, 145–163. https://doi.org/10.1016/0034-4257(93)90013-N
- Mahlein, A.-K. (2016). Plant disease detection by imaging sensors – parallels and specific demands for precision agriculture and plant phenotyping. *Plant Disease* 100, 241–251. https://doi.org/10.1094/PDIS-03-15-0340-FE
- Mahlein, A.-K. et al. (2013). Development of spectral indices for detecting and identifying plant diseases. *Remote Sensing of Environment* 128, 21–30. https://doi.org/10.1016/j.rse.2012.09.019
- Ploton, P. et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications* 11, 4540. https://doi.org/10.1038/s41467-020-18321-y
- Rinnan, Å., van den Berg, F., Engelsen, S.B. (2009). Review of the most common pre-processing techniques for near-infrared spectra. *TrAC Trends in Analytical Chemistry* 28, 1201–1222. https://doi.org/10.1016/j.trac.2009.07.007
- Roberts, D.R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40, 913–929. https://doi.org/10.1111/ecog.02881
- Rouse, J.W., Haas, R.H., Schell, J.A., Deering, D.W. (1974). Monitoring vegetation systems in the Great Plains with ERTS. *Third ERTS Symposium*, NASA SP-351, 309–317.
- Savitzky, A., Golay, M.J.E. (1964). Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry* 36, 1627–1639. https://doi.org/10.1021/ac60214a047
- Schölkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J., Williamson, R.C. (2001). Estimating the support of a high-dimensional distribution. *Neural Computation* 13, 1443–1471. https://doi.org/10.1162/089976601750264965

Further reading on grapevine diseases and leaf spectra: Junges et al. (2020), https://doi.org/10.1007/s40858-020-00387-0; Pérez-Roncal et al. (2020), https://doi.org/10.3390/agronomy10010088; Pithan et al. (2021), https://doi.org/10.1080/01431161.2021.1929542; Knauer et al. (2017), https://doi.org/10.1186/s13007-017-0198-y; Thomas et al. (2017), https://doi.org/10.1007/s41348-017-0124-6.

## Licence and citation

Apache-2.0 (see `LICENSE` and `NOTICE`). To cite, use `CITATION.cff` ("Cite this repository" on GitHub).

Archived on Zenodo: [doi:10.5281/zenodo.24118758](https://doi.org/10.5281/zenodo.24118758) (all versions; each release also has its own DOI on that page).
