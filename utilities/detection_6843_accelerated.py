"""Faster CFAR and peak grouping with the original bin/order conventions.

The reference functions in cfar_RV_fast.py and peakGrouping.py are retained.
Numerically ambiguous CFAR thresholds and unusual grouping inputs use those
reference calculations instead of silently changing detection decisions.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import correlate1d
from scipy.signal import convolve2d

from .cfar_RV_fast import cfar_RV_fast as _cfar_reference
from .peakGrouping import peakGrouping as _group_reference


def _box_sum(rd: np.ndarray, width: int) -> np.ndarray:
    """Sum a centered square using two short one-dimensional filters."""
    weights = np.ones(width, dtype=np.float64)
    horizontal = correlate1d(rd, weights, axis=1, mode="constant", cval=0.0)
    return correlate1d(horizontal, weights, axis=0, mode="constant", cval=0.0)


def cfar_RV_accelerated(RD, guard: int, train: int, Pfa: float) -> np.ndarray:
    """Return the same one-based [doppler, range, power] columns as CFAR.

    The training ring is an outer square minus the inner guard square.
    Separable square sums avoid the original full two-dimensional convolution.
    A conservative floating-point uncertainty band triggers its exact original
    convolution when an interior cell is too close to its decision threshold.
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
    # The original asymmetric edge exclusion leaves no interior in these cases.
    if rd.size == 0:
        return np.empty((3, 0), dtype=np.float64)
    # Preserve reference behavior for NaN/Inf, negative values, extended precision,
    # and non-numeric arrays, outside the magnitude-spectrum fast path.
    if (rd.dtype.kind not in "biuf" or rd.dtype.itemsize > 8
            or not np.all(np.isfinite(rd)) or np.any(rd < 0)):
        return _cfar_reference(rd, guard, train, Pfa)
    if min(rd.shape) <= 2 * radius + 1:
        return np.empty((3, 0), dtype=np.float64)
    data = rd.astype(np.float64, copy=False)
    max_power = float(np.max(data))
    if max_power == 0:
        return np.empty((3, 0), dtype=np.float64)
    width = 2 * radius + 1
    inner_width = 2 * guard + 1
    num_train = float(width * width - inner_width * inner_width)
    alpha = num_train * (float(Pfa) ** (-1 / num_train) - 1)
    # Very large values can overflow intermediate box sums in a different order.
    if (max_power > np.finfo(np.float64).max / (width * width * 4)
            or max_power < np.finfo(np.float64).tiny * width * width):
        return _cfar_reference(rd, guard, train, Pfa)
    sum_train = _box_sum(data, width) - _box_sum(data, inner_width)
    threshold = (sum_train / num_train) * alpha
    interior = (slice(radius, -(radius + 1)), slice(radius, -(radius + 1)))
    values = data[interior]
    thresholds = threshold[interior]
    detected = values > thresholds
    # This intentionally loose bound covers accumulation in both separable sums
    # and the reference width**2 convolution, subtraction and threshold scaling.
    # Near a threshold correctness matters more than avoiding one slow fallback.
    uncertainty = (np.finfo(np.float64).eps * 32.0 * width ** 4
                   * max_power * max(1.0, abs(alpha / num_train)))
    ambiguous = np.abs(values - thresholds) <= uncertainty
    if np.any(ambiguous):
        window = np.ones((width, width), dtype=np.float64)
        window[train:train + inner_width, train:train + inner_width] = 0
        reference_sum = convolve2d(rd, window, mode="same", boundary="fill", fillvalue=0)
        reference_threshold = (reference_sum[interior] / num_train) * alpha
        detected[ambiguous] = values[ambiguous] > reference_threshold[ambiguous]
    linear = np.flatnonzero(detected.ravel(order="F"))
    row, col = np.unravel_index(linear, detected.shape, order="F")
    row, col = row + radius, col + radius
    return np.vstack((col + 1, row + 1, rd[row, col])).astype(np.float64, copy=False)


def peakGrouping_accelerated(detMat) -> np.ndarray:
    """Group peaks with stable power ordering, duplicate bins and tied maxima.

    On the normal integer radar grid, indexed arrays replace a Python dictionary
    lookup for every neighbor of every detection. Missing neighbors remain zero.
    Sparse wide grids or fractional/nonfinite coordinates use the original code.
    """
    detections = np.asarray(detMat)
    if detections.size == 0:
        return np.empty((3, 0), dtype=np.float64)
    if detections.ndim != 2 or detections.shape[0] != 3:
        raise ValueError("detMat must have shape (3, K)")
    coords = detections[:2]
    if (detections.dtype.kind not in "biuf" or detections.dtype.itemsize > 8
            or not np.all(np.isfinite(coords))
            or np.any(coords < -(2 ** 52)) or np.any(coords > 2 ** 52)
            or np.any(coords != np.floor(coords))):
        return _group_reference(detections)
    integer_coords = coords.astype(np.int64)
    lo = integer_coords.min(axis=1)
    hi = integer_coords.max(axis=1)
    shape = tuple(int(value) + 3 for value in hi - lo)
    area = shape[0] * shape[1]
    if area > min(4_000_000, max(65_536, detections.shape[1] * 256)):
        return _group_reference(detections)
    order = np.argsort(-detections[2], kind="stable")
    detections = detections[:, order]
    positions = integer_coords[:, order] - lo[:, None] + 1
    keys = positions[0] * shape[1] + positions[1]
    # Only the first copy supplies a neighbor's power. The candidate's own power
    # is used separately, so a weaker duplicate can survive just as in MATLAB.
    unique_keys, first = np.unique(keys, return_index=True)
    grid = np.zeros(area, dtype=detections.dtype)
    grid[unique_keys] = detections[2, first]
    maximum = detections[2].copy()
    for doppler_offset in (-1, 0, 1):
        for range_offset in (-1, 0, 1):
            if doppler_offset or range_offset:
                neighbor_keys = keys + doppler_offset * shape[1] + range_offset
                np.maximum(maximum, grid[neighbor_keys], out=maximum)
    return detections[:, detections[2] == maximum]


__all__ = ["cfar_RV_accelerated", "peakGrouping_accelerated"]
