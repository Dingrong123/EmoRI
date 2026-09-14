"""Faster, order-preserving implementation of utilities.MSTC.

The original function remains unchanged. NumPy is the only required dependency;
Numba, when installed, speeds up the greedy distance check. No fast-math or
parallel reduction is used. Random samples, thresholds and tie-breaking follow
the original function, including its exclusion of the three seed points from
the returned static cloud.
"""

from functools import lru_cache
import math
import random

import numpy as np

_NUMBA_ERROR = None
try:
    import numba as _numba
except Exception as _exc:  # Optional acceleration must not prevent NumPy use.
    _numba = None
    _NUMBA_ERROR = str(_exc)


@lru_cache(maxsize=512)
def _samples(point_count):
    # Local RNGs give the same samples without modifying application RNG state.
    population = list(range(point_count))
    return np.asarray(
        [random.Random(seed).sample(population, 3) for seed in range(20)],
        dtype=np.int64,
    )


# Dependencies now load through the directory-independent _emori_runtime package.
# Keep this definition at its new source line: Numba includes that line in cache
# filenames, so old caches referencing python_version.utilities are not loaded.
# Subsequent runs and directory renames can reuse the new stable-module cache.
def _greedy_distance(xyz, seed_indices, eligible):
    """Greedy original-order selection; compiled when Numba is available."""
    accepted = np.empty(len(eligible), dtype=np.int64)
    selected = np.empty(len(eligible) + 3, dtype=np.int64)
    selected[:3] = seed_indices
    count = 0
    boundary = False
    for k in eligible:
        allowed = True
        for j in range(count + 3):
            other = selected[j]
            dx = xyz[other, 0] - xyz[k, 0]
            dy = xyz[other, 1] - xyz[k, 1]
            dz = xyz[other, 2] - xyz[k, 2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            # At the exact threshold, use NumPy's original 1-D norm below.
            # Ordinary samples never pay for the pairwise fallback.
            if abs(distance - 0.5) < 1e-14:
                boundary = True
            if distance < 0.5:
                allowed = False
                break
        if allowed:
            accepted[count] = k
            selected[count + 3] = k
            count += 1
    return accepted[:count], boundary


def _normal_equations(directions, doppler, seeds, accepted):
    """Accumulate in the same order as the original scalar implementation."""
    aa, bb, cc = seeds
    alphas = directions[:, 0]
    betas = directions[:, 1]
    gammas = directions[:, 2]
    a2 = alphas[aa] ** 2 + alphas[bb] ** 2 + alphas[cc] ** 2
    b2 = betas[aa] ** 2 + betas[bb] ** 2 + betas[cc] ** 2
    c2 = gammas[aa] ** 2 + gammas[bb] ** 2 + gammas[cc] ** 2
    ac = alphas[aa] * gammas[aa] + alphas[bb] * gammas[bb] + alphas[cc] * gammas[cc]
    bc = betas[aa] * gammas[aa] + betas[bb] * gammas[bb] + betas[cc] * gammas[cc]
    ab = alphas[aa] * betas[aa] + alphas[bb] * betas[bb] + alphas[cc] * betas[cc]
    ar = alphas[aa] * -doppler[aa] + alphas[bb] * -doppler[bb] + alphas[cc] * -doppler[cc]
    br = betas[aa] * -doppler[aa] + betas[bb] * -doppler[bb] + betas[cc] * -doppler[cc]
    cr = gammas[aa] * -doppler[aa] + gammas[bb] * -doppler[bb] + gammas[cc] * -doppler[cc]
    for k in accepted:
        a2 = a2 + alphas[k] ** 2
        b2 = b2 + betas[k] ** 2
        c2 = c2 + gammas[k] ** 2
        ab = ab + alphas[k] * betas[k]
        bc = bc + gammas[k] * betas[k]
        ac = ac + gammas[k] * alphas[k]
        ar = ar + alphas[k] * (-doppler[k])
        br = br + betas[k] * (-doppler[k])
        cr = cr + gammas[k] * (-doppler[k])
    lhs = np.array([[a2, ab, ac], [ab, b2, bc], [ac, bc, c2]])
    rhs = np.array([[ar], [br], [cr]])
    return lhs, rhs


if _numba is not None:
    _greedy_numba = _numba.njit(cache=True, fastmath=False)(_greedy_distance)
else:
    _greedy_numba = None


def available_backends():
    """Return explicitly selectable backends, excluding the 'auto' selector."""
    return ("numpy", "numba") if _numba is not None else ("numpy",)


def get_backend_info():
    """Report optional compiler availability without starting compilation."""
    return {
        "available_backends": list(available_backends()),
        "auto_backend": "numba" if _numba is not None else "numpy",
        "numba_version": getattr(_numba, "__version__", None),
        "numba_error": _NUMBA_ERROR,
        "fastmath": False,
    }


def _greedy_numpy(close, seeds, eligible):
    blocked = np.any(close[seeds], axis=0)
    accepted = []
    for k in eligible:
        if not blocked[k]:
            accepted.append(k)
            blocked |= close[k]
    return np.asarray(accepted, dtype=np.int64)


def _pairwise_close(xyz):
    differences = xyz[:, None, :] - xyz[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    close = distances < 0.5
    # Vectorized reductions can differ from a 1-D BLAS dot by one ULP.
    # Recheck only pairs where this could affect the strict distance gate.
    near = np.argwhere(np.abs(distances - 0.5) < 1e-14)
    for first, second in near:
        close[first, second] = np.linalg.norm(differences[first, second]) < 0.5
    return close


def MSTC_fast(data, indices, *, backend="auto", progress=False):
    """Return ``(velocities, frame_indices, static_pcs, static_indices)``.

    ``backend`` may be ``'auto'`` (optional Numba), ``'numba'`` or ``'numpy'``.
    ``progress=True`` prints one update per 100 frames, plus completion. The
    first Numba use compiles the distance kernel; following runs reuse its cache.

    Frame numbers in ``static_pcs[:, 3]`` refer to ORIGINAL frames, including
    unsuccessful frames. Empty-result shapes and the empty static-cloud list
    follow utilities.MSTC for compatibility.
    """
    if backend not in ("auto", "numpy", "numba"):
        raise ValueError("backend must be 'auto', 'numpy', or 'numba'")
    if backend == "numba" and _numba is None:
        raise RuntimeError("Numba is unavailable: " + str(_NUMBA_ERROR))
    use_numba = backend != "numpy" and _numba is not None
    velocities = []
    frame_indices = []
    static_chunks = []
    static_indices = [0]
    frame_count = len(indices) - 1

    for frame in range(frame_count):
        current = data[indices[frame]:indices[frame + 1], :]
        count = len(current)
        best_count = 3
        best_seeds = None
        best_accepted = None
        if count > 3:
            xyz = np.ascontiguousarray(current[:, :3])
            doppler = current[:, 3]
            directions = np.empty((count, 3), dtype=np.float64)
            # Preserve math.cos(math.acos(...)) from the original rather than
            # silently replacing it by the normalized coordinate itself.
            for k in range(count):
                divisor = math.sqrt(xyz[k, 0] ** 2 + xyz[k, 1] ** 2 + xyz[k, 2] ** 2 + 0.0001)
                for axis in range(3):
                    directions[k, axis] = math.cos(math.acos(xyz[k, axis] / divisor))

            close = None
            if not use_numba:
                close = _pairwise_close(xyz)
            forward = current[:, 1] > 3
            for seeds in _samples(count):
                A = directions[seeds]
                if np.linalg.matrix_rank(A) != 3:
                    continue
                # Keep inverse and a column RHS to follow np.matrix A.I * VR.
                vx, vy, vz = (np.linalg.inv(A) @ (-doppler[seeds, None])).ravel()
                residual = np.abs(
                    vx * directions[:, 0] + vy * directions[:, 1]
                    + vz * directions[:, 2] + doppler
                )
                eligible = np.flatnonzero(forward & (residual < 0.05))
                if use_numba:
                    accepted, boundary = _greedy_numba(xyz, seeds, eligible)
                    if boundary:
                        if close is None:
                            close = _pairwise_close(xyz)
                        accepted = _greedy_numpy(close, seeds, eligible)
                else:
                    accepted = _greedy_numpy(close, seeds, eligible)
                # Strict comparison preserves the first hypothesis on a tie.
                if len(accepted) > best_count:
                    best_count = len(accepted)
                    best_seeds = seeds
                    best_accepted = accepted

            if best_count >= 4:
                # Keep the final scalar accumulation in NumPy/Python. Numba's
                # floating-point power lowering can differ by one ULP even
                # with fastmath=False; only geometry selection is compiled.
                lhs, rhs = _normal_equations(directions, doppler, best_seeds, best_accepted)
                velocity = (np.linalg.inv(lhs) @ rhs).ravel()
                frame_indices.append(len(velocities))
                velocities.append(velocity)
                static = current[best_accepted, :4].copy()
                static[:, 3] = frame
                static_chunks.append(static)
                static_indices.append(static_indices[-1] + len(static))
            else:
                frame_indices.append(-1)
        else:
            frame_indices.append(-1)
        if progress and ((frame + 1) % 100 == 0 or frame + 1 == frame_count):
            print(f"MSTC: {frame + 1}/{frame_count} frames", flush=True)

    static_pcs = np.concatenate(static_chunks, axis=0) if static_chunks else []
    return (
        np.asarray(velocities), np.asarray(frame_indices),
        static_pcs, np.asarray(static_indices),
    )
