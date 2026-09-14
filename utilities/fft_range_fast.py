"""Range FFT from ``modules/fft/fft_range_fast.m``."""

from __future__ import annotations

import numpy as np
from scipy.fft import fft


def fft_range_fast(Xcube, fft_Rang: int, Is_Windowed=0) -> np.ndarray:
    """FFT along samples (axis 0); ``Is_Windowed`` is ignored in the source.

    Expected cube layout is ``(samples, receivers, chirps)``. SciPy's FFT,
    like MATLAB's, is unnormalized in the forward direction and pads or
    truncates to the supplied FFT length.
    """
    return fft(np.asarray(Xcube), n=fft_Rang, axis=0)
