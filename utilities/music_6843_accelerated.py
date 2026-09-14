"""Batched MUSIC angle estimation, preserving naive_music_6843 semantics.

Only the angle-estimation path used by the accelerated reader is included.
This specialization uses the original 3-by-2 covariance data matrix, SVD,
256-point FFT, and first-maximum rule for each four-antenna vector.
"""

from __future__ import annotations

import numpy as np


def _peak_direction_accelerated(antenna: np.ndarray) -> np.ndarray:
    """Evaluate the original four-element MUSIC problem in small batches."""
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = antenna / np.abs(antenna)
    whole = np.iscomplexobj(normalized)
    signal = np.asarray(normalized, dtype=np.complex128 if whole else np.float64).T
    if not np.all(np.isfinite(signal)):
        raise ValueError("pmusic input contains NaN or Inf; MATLAB SVD also rejects it")
    result = np.empty(signal.shape[0], dtype=np.float64)
    # Bound FFT workspace even for unusually large detection lists.
    for first in range(0, signal.shape[0], 512):
        last = min(first + 512, signal.shape[0])
        data = np.lib.stride_tricks.sliding_window_view(signal[first:last], 2, axis=1)[:, :, ::-1]
        data = data / np.sqrt(3.0)
        _, _, vh = np.linalg.svd(data, full_matrices=False)
        # Original pmusic chooses the one-column signal subspace because
        # signal and noise subspaces have equal size, then complements it.
        vectors = vh.conj().swapaxes(-1, -2)[:, :, :1]
        response = np.fft.fft(vectors, n=256, axis=1)
        denominator = np.sum(np.abs(response) ** 2, axis=2)
        denominator = np.abs(2 - denominator)
        with np.errstate(divide="ignore", invalid="ignore"):
            spectrum = 1.0 / denominator
        if not whole:
            spectrum = spectrum[:, :129]
        direction = (np.argmax(np.abs(spectrum), axis=1) + 1) / 256.0 * 2.0
        direction[direction > 1.0] -= 2.0
        result[first:last] = direction
    return result


def naive_music_6843_accelerated(
    Xcube: np.ndarray,
    detout: np.ndarray,
    num_tx=None,
    num_rx=None,
    fft_Ang=None,
    Is_Windowed=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y, z, RCS using the unchanged angle bins and conventions.

    Signature and validation match ``naive_music_6843``. The final four
    arguments remain unused, as in the original. Detection indices are
    MATLAB one-based; a negative y squared value retains its complex root.
    """
    cube = np.asarray(Xcube)
    detections = np.asarray(detout)
    if cube.ndim != 3 or cube.shape[1] < 11:
        raise ValueError("Xcube must have shape (range, at least 11 antennas, Doppler)")
    if detections.ndim != 2 or detections.shape[0] < 2:
        raise ValueError("detout must be a two-dimensional array with at least two rows")
    count = detections.shape[1]
    if count == 0:
        return tuple(np.empty(0, dtype=np.float64) for _ in range(4))
    indices = detections[:2]
    if not np.all(np.isfinite(indices)) or not np.all(indices == np.floor(indices)):
        raise ValueError("detout range and Doppler indices must be integers")
    doppler = indices[0].astype(np.intp) - 1
    ranges = indices[1].astype(np.intp) - 1
    if np.any(doppler < 0) or np.any(doppler >= cube.shape[2]):
        raise IndexError("detout Doppler indices lie outside Xcube")
    if np.any(ranges < 0) or np.any(ranges >= cube.shape[0]):
        raise IndexError("detout range indices lie outside Xcube")
    virtual_ant = cube[ranges, :, doppler].T
    amplitudes = np.sum(np.abs(virtual_ant), axis=0)
    peak = np.max(amplitudes)
    if peak == 0 and count > 1:
        rcs = np.zeros(count, dtype=np.float64)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            rcs = (amplitudes - np.min(amplitudes)) / peak
    # Stack both antenna groups to share the SVD/FFT batch dispatch.
    azimuth_and_elevation = np.concatenate(
        (virtual_ant[[3, 2, 7, 6], :], virtual_ant[[8, 10, 4, 6], :]), axis=1
    )
    directions = _peak_direction_accelerated(azimuth_and_elevation)
    z_vector, x_vector = directions[:count], directions[count:]
    y_vector = np.lib.scimath.sqrt(1.0 - x_vector ** 2 - z_vector ** 2)
    return x_vector, y_vector, z_vector, rcs


__all__ = ["naive_music_6843_accelerated"]
