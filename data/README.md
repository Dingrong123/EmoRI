# Data contract

This release contains two preprocessed example sequences, `(1, 1)` and `(2, 1)`. Each has 1,200 radar frames, 12,000 aligned IMU rows, and 1,200 aligned ground-truth positions. The eight numeric NPY files total 18,050,832 bytes. Hashes and exact shapes are recorded in [the sample manifest](../docs/reproduction.json).

The examples come from the local dataset supplied with this code. Their exact mapping to the paper's 98 individual traces, platform labels, environments, and moving-object conditions has not been provided. Do not infer those labels from `env=1` or `env=2`. No complete paper dataset is bundled. A public raw-capture download URL and dataset license are not yet specified for this release candidate.

## Input files

| Pattern | Shape | dtype | Units / meaning |
| --- | --- | --- | --- |
| `point_cloud/pypc_E_X.npy` | `(P, 5)` | `float64` | Radar x/y/z in m, radial Doppler in m/s, relative amplitude |
| `point_cloud/pypc_E_X_indices.npy` | `(N+1, 1)` | `int32` | Zero-based cumulative point counts |
| `IMU_and_GT/0516env_aligned_imu_E_X.npy` | `(M, 6)` | `float64` | `[ax, ay, az]` in g, `[gx, gy, gz]` in degrees/s |
| `IMU_and_GT/0516env_aligned_gt_E_X.npy` | `(N, 3)` | `float64` | World-frame `[x, y, z]` in m |

`E` is the environment ID; `X` is the experiment ID. The current examples have `M = 10N`, following the code's 100 Hz IMU and 10 Hz radar assumptions. NPY files contain the arrays directly, not dictionaries or MATLAB structures. No object/pickle arrays are needed.

Point counts: `(1, 1)` has 213,362 points; `(2, 1)` has 207,403. Each index array has shape `(1201, 1)`. Each IMU array has shape `(12000, 6)` and each GT array has shape `(1200, 3)`.

## Radar point conventions

- Positions are in the radar frame. The angle estimator constructs the forward `y` component from the selected direction cosines. Retain the code's antenna ordering and sign conventions.
- For a stationary return with unit line of sight `u`, the fitted velocity obeys `u @ v_R = -radial_velocity`.
- The fifth column is named `RCS` in parts of the source but represents relative amplitude, not calibrated RCS in square meters.
- `boundaries[k]:boundaries[k+1]` selects frame `k`. The first boundary is 0; the last equals the total point count. Boundaries must be nondecreasing. Equal boundaries preserve empty frames and their time positions.
- The reader's legacy all-empty-sequence return has shape `(0, 0)`, rather than `(0, 5)`. Downstream consumers should detect an all-empty recording before assuming five columns. The included samples are nonempty.

From the repository root, inspect a sample with:

```python
from pathlib import Path
import numpy as np

root = Path("data")
points = np.load(root / "point_cloud/pypc_1_1.npy", allow_pickle=False)
b = np.load(root / "point_cloud/pypc_1_1_indices.npy", allow_pickle=False).ravel()
imu = np.load(root / "IMU_and_GT/0516env_aligned_imu_1_1.npy", allow_pickle=False)
gt = np.load(root / "IMU_and_GT/0516env_aligned_gt_1_1.npy", allow_pickle=False)
assert points.dtype == np.float64 and points.shape[1] == 5
assert b.dtype == np.int32 and b[0] == 0 and b[-1] == len(points)
assert np.all(np.diff(b) >= 0)
assert imu.shape == (10 * (len(b) - 1), 6)
assert gt.shape == (len(b) - 1, 3)
```

## Alignment and new datasets

The estimator uses index-based association rather than timestamps. IMU rows must already be ordered and synchronized to the radar frame sequence, including capture concatenation. Do not drop empty radar frames. The GT trajectory must use the expected world frame, initial heading, scale, and sequence length: the program performs no GT registration.

The mounting rotation and the constant heading correction are supplied in the code. They are not estimated from a new recording. The current zero lever arm and yaw-only world integration are additional assumptions. See [implementation details](../docs/IMPLEMENTATION.md) before replacing the samples.

The estimator requires GT files to run its current evaluation entry point, although GT is not used to estimate velocity or pose. Acquisition, radar/IMU clock alignment, conversion of other IMU log formats, and LiDAR ground-truth estimation are not included.

## Raw captures

Optional input folder: `radar_raw_data/`. Filenames follow `0516env_new_E_X_Raw_K.bin`; `K` is a capture number. For the supplied acquisition settings, each file contains 300 frames and occupies 737,280,000 bytes. Four files make one example sequence; eight local files across both examples total 5,898,240,000 bytes. These raw files are excluded from Git and the prepared source/sample archive.

The decoder consumes reconstructed binary ADC data, not an arbitrary UDP packet dump, ROS bag, or TI point-cloud output. It expects little-endian signed 16-bit `I0, I1, Q0, Q1` packing, 3 TX, 4 RX, 512 samples per chirp, and 100 loops per frame.

Use `read_6843_accelerated.py --check-inputs` to validate selected capture files before processing. It checks file existence and size/chirp constraints, not acquisition metadata or synchronization quality.

## Outputs and repository policy

The reader writes `point_cloud/pypc_E_X.npy`, `point_cloud/pypc_E_X_indices.npy`, and its timing report. The estimator writes map, trajectory, GT copy, timing JSON, and optional PNG files to `slam_results/` by default.

The `.gitignore` explicitly includes only the eight supplied sample NPY files and data documentation. Raw captures, MAT backups, migration records, new datasets, preview folders, and generated SLAM outputs remain local. To release another dataset, deliberately update the data manifest and allowlist.
