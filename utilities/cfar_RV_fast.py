"""Translate ``cfar_RV_fast.m`` without changing its CFAR bin conventions."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d


def cfar_RV_fast(RD, guard: int, train: int, Pfa: float) -> np.ndarray:
    """Return detections as rows ``[doppler_bin, range_bin, power]``.

    The two bin rows deliberately contain **one-based MATLAB indices**.
    Columns follow MATLAB ``find`` order (range varies fastest). As in the
    source, the first ``guard + train`` edge bins and the last
    ``guard + train + 1`` edge bins are discarded on both dimensions.
    ``RD`` is used directly; it is not squared or converted to decibels.
    """
    rd = np.asarray(RD)
    if rd.ndim != 2 or np.iscomplexobj(rd):
        raise ValueError("RD must be a two-dimensional real-valued matrix")
    if isinstance(guard, bool) or int(guard) != guard or guard < 0:
        raise ValueError("guard must be a nonnegative integer")
    if isinstance(train, bool) or int(train) != train or train < 1:
        raise ValueError("train must be a positive integer")
    if not 0 < Pfa <= 1:
        raise ValueError("Pfa must be in (0, 1]")
    guard, train = int(guard), int(train)
    radius = guard + train
    width = 2 * radius + 1
    window = np.ones((width, width), dtype=np.float64)
    window[train : train + 2 * guard + 1, train : train + 2 * guard + 1] = 0
    num_train = float(window.sum())
    if rd.size == 0:
        return np.empty((3, 0), dtype=np.float64)
    sum_train = convolve2d(rd, window, mode="same", boundary="fill", fillvalue=0)
    noise_mean = sum_train / num_train
    alpha = num_train * (float(Pfa) ** (-1 / num_train) - 1)
    detected = rd > noise_mean * alpha
    detected[:radius, :] = False
    detected[-(radius + 1) :, :] = False
    detected[:, :radius] = False
    detected[:, -(radius + 1) :] = False
    linear = np.flatnonzero(detected.ravel(order="F"))
    row, col = np.unravel_index(linear, rd.shape, order="F")
    return np.vstack((col + 1, row + 1, rd[row, col])).astype(np.float64, copy=False)
