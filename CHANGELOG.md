# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-10

### Added
- `Spectra` container (values, wavelengths, metadata) with nearest-band lookup.
- `io`: ENVI header parser and reader (BSQ/BIL/BIP, memory-mapped), cube to pixel spectra, pixel-spectra exports (CSV/XLSX).
- `calibrate`: reflectance from raw/dark/white, saturation mask, panel detection (spectral angle),
  vector normalisation, session-reference ratio.
- `preprocess`: trimming, quality mask, Savitzky-Golay smoothing and derivatives, SNV, continuum removal.
- `indices`: NDVI, PRI, red-edge position, generic normalised difference.
- `aggregate`: pixel to leaf spectra with label-conflict checks.
- `models`: PLS-DA, SVM, random forest and logistic pipelines; one-class SVM.
- `validate`: grouped folds, random vs grouped comparison, leave-one-group-out, one-class out-of-fold scores.
- `bands`: window-wise permutation importance on held-out leaves.
- `synthetic`: illustrative leaf spectra with balanced and date-confounded designs.
- Tutorial notebook `tutorials/01_leaf_disease_walkthrough.ipynb`, executed by the test suite.
