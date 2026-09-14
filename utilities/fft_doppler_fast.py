"""Doppler FFT from ``modules/fft/fft_doppler_fast.m``."""

from __future__ import annotations

import numpy as np
from scipy.fft import fft, fftshift


def fft_doppler_fast(Xcube, fft_Vel: int, Is_Windowed=0) -> np.ndarray:
    """FFT and FFT shift along chirps (axis 2), without applying a window.

    ``Is_Windowed`` is intentionally ignored, matching the active MATLAB
    code. Trailing singleton dimensions omitted by MATLAB are restored for
    a one-dimensional or two-dimensional input.
    """
    cube = np.asarray(Xcube)
    if cube.ndim < 3:
        cube = cube.reshape(cube.shape + (1,) * (3 - cube.ndim))
    return fftshift(fft(cube, n=fft_Vel, axis=2), axes=2)

