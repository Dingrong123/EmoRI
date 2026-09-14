"""Local peak grouping corresponding to ``modules/cluster/peakGrouping.m``."""

from __future__ import annotations

import numpy as np


def peakGrouping(detMat) -> np.ndarray:
    """Keep 3-by-3 local maxima, preserving tied peaks and one-based bins.

    ``detMat`` has shape ``(3, K)`` and rows ``[doppler, range, power]``.
    Missing neighbors have power zero, just as in the MATLAB kernel.
    Output columns are sorted by decreasing power with stable tie order.
    Empty input produces a consistently shaped ``(3, 0)`` array.
    """
    detections = np.asarray(detMat)
    if detections.size == 0:
        return np.empty((3, 0), dtype=np.float64)
    if detections.ndim != 2 or detections.shape[0] != 3:
        raise ValueError("detMat must have shape (3, K)")
    order = np.argsort(-detections[2], kind="stable")
    detections = detections[:, order]
    # find(...)(1) in the source selects the first (highest-power) copy of
    # a duplicate bin. A dictionary avoids repeatedly scanning all targets.
    neighbors = {}
    for doppler, range_bin, power in detections.T:
        neighbors.setdefault((doppler, range_bin), power)
    offsets = tuple((d, r) for d in (-1, 0, 1) for r in (-1, 0, 1) if d or r)
    keep = []
    for i, (doppler, range_bin, power) in enumerate(detections.T):
        kernel = [power]
        kernel.extend(neighbors.get((doppler + d, range_bin + r), 0) for d, r in offsets)
        if power == np.max(kernel):
            keep.append(i)
    return detections[:, keep]

