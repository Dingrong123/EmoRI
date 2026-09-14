# Implementation notes

This document describes the Python code shipped in this repository. The research context is **EmoRI**, introduced in [the SenSys '23 paper](https://doi.org/10.1145/3625687.3625795). The present offline implementation preserves the selected processing path and numerical conventions of the supplied code; it is not a complete reproduction of every system component, experiment, or baseline in the paper.

## Module map

| Stage | Entry point / implementation | Current behavior |
| --- | --- | --- |
| Capture decoding | [`read_6843_accelerated.py`](../read_6843_accelerated.py), [`capture_io.py`](../utilities/capture_io.py) | Stream fixed-format IWR6843 ADC frames, with optional ordered multiprocessing. |
| Radar signal processing | [`fft_range_fast.py`](../utilities/fft_range_fast.py), [`fft_doppler_fast.py`](../utilities/fft_doppler_fast.py), [`detection_6843_accelerated.py`](../utilities/detection_6843_accelerated.py), [`music_6843_accelerated.py`](../utilities/music_6843_accelerated.py) | Range/Doppler FFT, CFAR, peak grouping, and the selected MUSIC angle estimator. |
| Radar egovelocity | [`radar_slam_mstc_fast.py`](../utilities/radar_slam_mstc_fast.py) | Deterministic three-point velocity hypotheses with Doppler consistency and spatial separation. |
| Translation and IMU fusion | `Fusion_g_fast` in [`radar_slam_motion_fast.py`](../utilities/radar_slam_motion_fast.py) | Radar velocity directly supplies translation when available; an offline gravity/acceleration routine fills missing radar estimates. |
| Orientation and trajectory | `integrate_motion_fast` in [`radar_slam_motion_fast.py`](../utilities/radar_slam_motion_fast.py) | Integrate IMU **yaw only**, apply fixed extrinsics, and accumulate world-frame displacements. |
| Map placement | `EmoRI_SLAM_fast` in [`radar_slam_motion_fast.py`](../utilities/radar_slam_motion_fast.py) | Transform accepted stationary points using the estimated trajectory and orientation. |
| Evaluation and output | [`EmoRI.py`](../EmoRI.py) | Load aligned ground truth, save arrays, report position RMSE, and optionally plot. |

## ADC to radar point cloud

The reader expects the acquisition configuration in [`get_params_value.py`](../utilities/get_params_value.py), together with constants in [`capture_io.py`](../utilities/capture_io.py). Changing a few parameters does not make the current decoder generic: antenna order, channel signs, and the 100-loop layout also appear in the processing code.

| Setting | Value used by the code |
| --- | --- |
| Carrier frequency | 62 GHz |
| Transmitters / receivers | 3 / 4 |
| ADC samples per chirp | 512 |
| ADC sample rate | 10 MHz |
| Sweep slope | 66.011 THz/s |
| Chirp interval parameter | 160 microseconds; Doppler grid uses three TDM slots |
| Loops per frame | 100 |
| Range / Doppler FFT lengths | 512 / 100 |
| CFAR parameters | Guard half-width 5, training-band width 5, default `pfa=0.03` |
| Raw storage | Little-endian signed 16-bit samples, groups ordered `I0, I1, Q0, Q1` |
| Raw frame size | 300 chirps × 8,192 bytes = 2,457,600 bytes |

The decoder returns one complex array with shape `(4, 300 * 512)`. Receiver channels 0 and 2 are negated. The TDM slots are interpreted as **TX1, TX3, TX2**, and this order defines the 12 virtual antenna channels. Range FFT runs along ADC samples; Doppler FFT runs along slow-time loops and is shifted to a centered Doppler grid. Neither FFT applies a window in the selected implementation.

CFAR operates on the **mean magnitude** of the four TX3 receiver channels, not squared magnitude. Detection bins retain MATLAB's one-based convention internally; this is separate from the zero-based frame slicing in the saved index file. The angle estimator phase-normalizes four-element antenna vectors, forms reversed adjacent windows, uses batched SVD and a 256-point FFT, and keeps the first maximum. Its selected virtual antenna indices are `[3, 2, 7, 6]` for the `z` direction component and `[8, 10, 4, 6]` for `x`, using zero-based Python indexing. It then computes `y = sqrt(1 - x² - z²)` with a complex-capable square root and takes the real part when assembling output points.

Each point-cloud row is `[x, y, z, radial_velocity, relative_amplitude]`, with position in meters and radial velocity in meters/second. The last column is called `RCS` in the code but is a frame-relative amplitude indicator, **not calibrated physical radar cross section**. The saved cumulative index column has `N + 1` entries for `N` radar frames:

```python
points = np.load("data/point_cloud/pypc_1_1.npy", allow_pickle=False)
indices = np.load("data/point_cloud/pypc_1_1_indices.npy", allow_pickle=False).ravel()
k = 0
frame_k = points[indices[k]:indices[k + 1]]
```

Run this standalone example from the repository root. The NumPy calls above follow the Python process's working directory; the entry scripts' CLI path resolution is described separately below. A repeated boundary represents an empty frame. Frame order follows the provided capture-number order, including when workers process separate blocks concurrently.

## Radar velocity estimation

For each radar frame, `MSTC_fast` evaluates 20 candidate triples generated by local random generators with seeds 0 through 19. These generators do not alter application-wide random state. Each full-rank triple yields a candidate velocity from the stationary-object constraint:

```text
U v_R = -v_doppler
```

`U` is built from the point directions. The implementation preserves the source's `sqrt(x² + y² + z² + 0.0001)` normalization and `cos(acos(...))` evaluation order. A candidate's additional points must satisfy `y > 3 m`, an absolute Doppler residual below `0.05 m/s`, and greedy pairwise spatial separation of at least `0.5 m` from already selected points, including the three seeds. Selection proceeds in original point order.

The winning hypothesis must accept at least **four additional points beyond its three seeds**. Ties keep the first hypothesis. A normal-equation solve using the seeds and accepted points refines the velocity. Returned stationary clouds contain the additional accepted points; the three seed points are omitted for compatibility. A failed frame receives index `-1`, while successful velocities are stored compactly. The fourth column of the returned stationary cloud is overwritten with the **original radar frame number**, replacing its input Doppler value.

Numba accelerates the greedy distance check when installed. The NumPy backend implements the same selection convention. Neither fast-math nor parallel floating-point reduction is enabled. The first Numba run can include compilation time; subsequent runs can reuse the local cache. These optimizations do not add an IMU measurement or ground-truth input to the radar velocity estimator.

## Coordinates, fixed extrinsics, and IMU fusion

Use the column-vector convention `v_R = R_IMU_TO_RADAR @ v_I`, where `R` and `I` denote the radar and IMU frames. The stored rotation in [`EmoRI.py`](../EmoRI.py) is:

```python
R_IMU_TO_RADAR = np.array([
    [0.06280492776998943, 0.9890933401528303, 0.1332287713413427],
    [0.994246658175081, -0.07361323317790894, 0.07781178965155619],
    [0.0867705235404837, 0.12757529685018243, -0.9880260218628343],
])
R_RADAR_TO_IMU = R_IMU_TO_RADAR.T
```

These are fixed, mounting-specific numbers. They are not estimated from ground truth or recalibrated each run. Translation is assumed to be zero, so the code does not implement the `omega × lever_arm` correction in the paper's velocity-transfer equation. Reusing these values for a different sensor mounting is not a new calibration.

The input IMU array has columns `[ax, ay, az, gx, gy, gz]`, with accelerations in **g** and angular rates in **degrees/second**. The pipeline assumes already aligned streams at **100 Hz IMU and 10 Hz radar**, i.e. ten IMU samples per radar frame. It does not infer timestamps, estimate clock offset, or resample asynchronous data.

### Frames with valid radar velocity

For a successful frame, translation in radar coordinates is simply:

```text
delta_p_R[k] = v_R[k] / 10
```

Accelerometer samples do not affect that frame's displacement. When all radar frames are valid, `Fusion_g_fast` returns immediately without using IMU acceleration at all. The gyroscope still affects trajectory orientation in the next stage.

For zero-based radar frame `k`, the integrated yaw is:

```text
end[k] = min(10 * (k + 1), number_of_IMU_samples)
yaw_deg[k] = sum(gz[0:end[k]]) / 100 + 0.035 * k
yaw_rad[k] = yaw_deg[k] / 180 * 3.1415926
R_WI[k] = R_z(yaw_rad[k])
delta_p_W[k] = R_WI[k] @ R_RADAR_TO_IMU @ delta_p_R[k]
trajectory[k] = sum(delta_p_W[0:k + 1])
```

`W` is the integration/world frame. Only the sixth IMU column contributes to the final orientation; roll and pitch are not integrated into `R_WI`. The `0.035` degrees/frame term is a fixed existing yaw correction, approximately `0.35` degrees/second, not an online drift estimator. The first output row already includes one radar-period displacement; an explicit origin row is not prepended to the trajectory. Limiting the sample endpoint prevents an indexing overflow but does not recover missing IMU samples.

### Missing radar velocity

If any frame has index `-1`, the entry point converts acceleration from g to meters/second² using `9.81`. The fusion routine copies the IMU array and then performs two passes:

1. Between successive valid radar velocity anchors, compare the radar-derived velocity change (rotated into IMU coordinates) with integrated IMU acceleration. Three-axis gyro increments provide rotations used to estimate and subtract a gravity vector over that interval.
2. Keep `v_R / 10` for valid frames. For a missing frame `i`, integrate corrected accelerations over samples `[10*(i-1):10*i]`, rotate the velocity increment back into radar coordinates, add it to the last velocity estimate, and multiply by `0.1 s`.

The first pass uses a later valid radar frame to correct preceding samples. Consequently, this is **offline gap reconstruction**, not the paper's causal propagation of the latest gravity anchor. Three-axis rotations inside this helper affect gravity compensation only; they do not change the yaw-only final trajectory.

The preserved edge cases also matter: the initial left anchor is frame 0 with zero velocity; a missing first frame normally integrates an empty slice and produces zero displacement; trailing gaps with no later valid radar anchor do not receive the first pass's gravity subtraction. The two supplied examples have no radar-velocity gaps and therefore do not validate this branch.

## Map and ground truth

Map placement uses the same estimated pose and fixed rotation as trajectory integration:

```text
p_W = R_WI[frame] @ R_RADAR_TO_IMU @ p_R + trajectory[frame]
```

Before translation, points must have range in `[0.1, 6] m` and rotated relative height strictly between `-0.5` and `2 m`. This height test is relative to the sensor position after rotation, not an absolute world-height filter. Original frame IDs select poses even after missing radar estimates. The saved map has columns `[x_W, y_W, z_W, frame_id]`; its initial all-zero row is a compatibility placeholder, not a measured point. This stage accumulates points; it does not perform loop closure or pose-graph optimization.

Ground truth is loaded by the evaluation entry point and must have shape `(N, 3)`. It is required by the current CLI because plotting and reporting are integrated with estimation, but it is not fed into the radar velocity, fusion, yaw integration, or map-placement algorithms. The source performs no SVD, SE(3), or Sim(3) trajectory alignment at runtime. Already aligned input trajectories still need a compatible coordinate convention and time origin; the code does not solve those for new recordings. Historical API keys ending in `_aligned` are retained names and do not imply that a ground-truth fit is applied.

The JSON report computes:

```text
position_rmse_m = sqrt(mean(sum((estimated_position - ground_truth_position)**2, axis=1)))
```

This is an unaligned 3D position RMSE in meters. The paper defines ATE as the mean Euclidean position distance, which differs from this square-root-of-mean-squares statistic. The current report does not calculate the paper's endpoint drift error. Label reproduced results with their actual metric.
