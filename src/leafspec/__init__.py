"""leafspec: leaf-level hyperspectral analysis for plant disease detection.

Each module is an independent implementation of the methods described in the
publications cited in its docstring.
"""

from . import aggregate, bands, calibrate, indices, io, models, preprocess, synthetic, validate
from .spectra import Spectra

__all__ = [
    "Spectra", "aggregate", "bands", "calibrate", "indices", "io", "models", "preprocess", "synthetic", "validate",
]
__version__ = "0.1.0"
