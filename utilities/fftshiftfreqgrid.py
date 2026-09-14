"""Frequency-grid construction matching ``modules/fft/fftshiftfreqgrid.m``."""

from __future__ import annotations

import numpy as np


def fftshiftfreqgrid(N: int, Fs: float) -> np.ndarray:
    """Return the shifted frequency grid as a NumPy one-dimensional array.

    The explicit Nyquist assignments preserve the source arithmetic for
    odd lengths. For ``N == 1`` the MATLAB code expands its vector to two
    elements; that unusual behavior is retained (the returned grid is
    ``[Fs, 0]``). The radar configuration uses ``N == 100``.
    """
    if isinstance(N, bool) or int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    N = int(N)
    freq_res = Fs / N
    freq_grid = np.arange(N, dtype=np.float64) * freq_res
    nyquist = Fs / 2
    half_res = freq_res / 2
    if N % 2:
        halfpts = (N + 1) // 2
        freq_grid[halfpts - 1] = nyquist - half_res
        if N == 1:
            freq_grid = np.append(freq_grid, nyquist + half_res)
        else:
            freq_grid[halfpts] = nyquist + half_res
        negative_count = (N - 1) // 2
    else:
        freq_grid[N // 2] = nyquist
        negative_count = N // 2
    freq_grid = np.fft.fftshift(freq_grid)
    freq_grid[:negative_count] -= Fs
    return freq_grid
