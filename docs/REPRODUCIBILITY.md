# Reproducing the included examples

## Scope and environment

These checks establish that the released offline Python path runs with the two provided samples and preserves the current implementation's output. They do not reproduce the paper's full 98-trace evaluation, baseline comparisons, full 6-DoF behavior, or missing-velocity experiments.

The checked runtime is Windows, 64-bit CPython 3.13.9, NumPy 2.3.5, SciPy 1.16.3, Matplotlib 3.10.6, and Numba 0.62.1. The requirements pin the direct package versions from that runtime; this is not a full lockfile for every transitive dependency or system library. Other OS/runtime combinations have not been checked. No GPU, MATLAB, ROS, or pretrained weights are required for these sample runs.

## Procedure

From the project root (any directory name) after installing the requirements:

```bash
python EmoRI.py --envs 1 2 --experiments 1 --backend numpy --no-plot --output-dir data/reproduce_numpy
```

Optionally install `requirements-accelerated.txt`, then run:

```bash
python EmoRI.py --envs 1 2 --experiments 1 --backend numba --no-plot --output-dir data/reproduce_numba
python EmoRI.py --envs 1 --experiments 1 --backend numpy --no-show --save-plot --output-dir data/reproduce_plot
```

Each backend directory should contain two sets of `nradar_motion_E_1.npy`, `lidar_motion_E_1.npy`, `nradar_slam_E_1.npy`, and `timing_E_1.json`. The plot directory also contains `trajectory_1_1.png`.

| Sequence | Trajectory / GT shape | Map shape, including placeholder | Valid radar frames | Position RMSE |
| --- | --- | --- | ---: | ---: |
| `(1, 1)` | `(1200, 3)` | `(4296, 4)` | 1200 | 0.19124392166440765 m |
| `(2, 1)` | `(1200, 3)` | `(2932, 4)` | 1200 | 0.677321 approximately |

Full-precision recorded metrics, file sizes, hashes, and backend differences are in [reproduction.json](reproduction.json). Both sequences have no missing radar velocity; testing the acceleration fallback requires another targeted input.

To compare backend outputs, from the repository root:

```python
from pathlib import Path
import numpy as np

for env in (1, 2):
    for prefix in ("nradar_motion", "lidar_motion", "nradar_slam"):
        filename = f"{prefix}_{env}_1.npy"
        a = np.load(Path("data/reproduce_numpy") / filename, allow_pickle=False)
        b = np.load(Path("data/reproduce_numba") / filename, allow_pickle=False)
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
```

The stated tolerance was met on the checked environment. CPU math libraries and versions can affect floating-point output; a changed point selection deserves investigation rather than an automatic assertion of cross-platform bitwise identity.

## Metric definition

The reported field `position_rmse_m` is:

```python
position_rmse_m = np.sqrt(np.mean(np.sum((trajectory - truth) ** 2, axis=1)))
```

This is a root mean square of **3D Euclidean** position errors, without an SE(3), Sim(3), or SVD registration. The stored fixed mounting rotation and yaw correction are used unchanged. Their provenance is the supplied code; the sample results are not presented as performance on a held-out extrinsic-calibration test set.

The paper defines ATE as the mean Euclidean position error and DE as endpoint error divided by traveled path length. Thus the program's RMSE is not the paper's ATE. The overview figure uses a top-down trajectory projection while its error panels use all three coordinates.

## What was checked for this release candidate

- The two entry points and local Python dependencies were copied to a separate `python_version` directory with only the eight sample NPY inputs.
- Both samples were run with NumPy and Numba. Map, estimated trajectory, and GT arrays agreed within the tolerance above and matched the existing local reference outputs.
- Commands were launched from an unrelated working directory to verify code-relative CLI paths.
- The plotting path was run with `--no-show --save-plot`.
- Local raw-capture preflight validated eight files / 2,400 radar frames. The large raw inputs are not included in the prepared source/sample archive.
- A newly created virtual environment installed the pinned requirements from PyPI, then ran both samples with both backends and saved a figure. A 10-frame raw-reader check with two spawned workers matched the provided sample points and indices; exact package versions and differences are included in `reproduction.json`.
- The initial documentation preparation preserved the source files. The subsequent directory-name compatibility update changes entry-point imports and the Numba cache identity only; its checks are described below.

The reader's previous local NPY migration check also regenerated the full `(2, 1)` sequence from raw captures and compared its point and index arrays with the original MAT arrays. That historical check is not a fresh, cross-platform rerun of all paper data and its machine-specific logs are excluded from the release.

## Directory-name compatibility checks

The import update was checked with the same sample arrays in directories named `python_version`, `EmoRI`, and `EmoRI-main 中文 测试.v1`. Direct scripts were launched from another working directory; module launch and the public Python API were also exercised. The raw reader processed ten frames with two spawned workers through direct, module, and API entry paths.

A complete 1,200-frame sample was compared with the existing trajectory, map, and GT reference using NumPy and Numba. The largest output difference was approximately `3.6e-15`; radar preview points differed by at most `1.2e-16`, and frame boundaries were equal. The ASTs of all 44 functions/methods were unchanged. Existing legacy Numba caches were retained; the updated cache was compiled once and then successfully loaded after two directory renames. See [import_portability.json](import_portability.json) for the recorded checks.

## Performance measurements

For the raw reader, vary `--workers` and `--chunk-frames` while keeping captures, frame ranges, and PFA fixed. For MSTC, compare `numpy` and `numba` using the same sample and output settings. The stage timings are in the generated JSON reports.

Report interpreter/package versions, hardware, worker count, backend, cache state, repetitions, and whether figures are rendered. Separate imports and the first Numba compilation from repeated processing. Do not compare a cold run with a warm run as an algorithmic speedup. No new speedup factor or paper-runtime replication is claimed by this documentation.
