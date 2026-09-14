"""Fast motion integration and map placement for Radar_SLAM_v2_fast.py.

The calibration is supplied by the caller and is never estimated here.
Original functions in utilities.py are left unchanged. Only NumPy and Python's
standard library are imported. Input arrays are never modified.
"""

import math

import numpy as np


def _rotation_array(rotation):
    # 外参采用列向量约定：雷达→IMU。这里只检查形状，不重新估计或调整矩阵。
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("radar_to_imu must be a 3-by-3 rotation")
    return rotation


def Fusion_g_fast(mmwave_indices, mmwave_vel, imus, radar_to_imu):
    """Return radar-frame displacements using the fixed mounting rotation.

    Valid radar frames use velocity / 10. Only missing frames (-1) need the
    IMU acceleration and gravity compensation, operating on a private copy.
    IMU acceleration is in m/s^2 and angular velocity is in degrees/second.
    """
    # indices 长度为原始雷达帧数 N；有效值指向紧凑的速度数组，-1 表示测速失败。
    indices = np.asarray(mmwave_indices)
    velocities = np.asarray(mmwave_vel)
    if indices.ndim != 1:
        raise ValueError("mmwave_indices must be one-dimensional")
    radar_to_imu = _rotation_array(radar_to_imu)
    if np.all(indices != -1):
        # 正常分支：10 Hz 对应每帧 0.1 s，因此位移为 v/10，单位 m。
        # 这里不读取 IMU；高级索引产生新数组，也不会改动原速度数据。
        return velocities[indices] / 10
    # 重力扣除会修改加速度，因此向缺帧分支传入副本，保护调用者的 IMU 数据。
    return _fusion_g_gap_reference(
        indices, velocities, np.array(imus, copy=True), radar_to_imu=radar_to_imu
    )


def integrate_motion_fast(displacements_radar, imus, radar_to_imu):
    """Return (trajectory[N,3], world_from_imu[N,3,3]) as in the original.

    Preserve 100 Hz IMU / 10 Hz radar sampling, yaw-only integration, the
    existing 0.035 degrees/frame correction, and the original pi constant.
    Prefix sums replace repeated sums over the growing IMU history. The IMU
    columns and units are unchanged: column 5 is yaw rate in degrees/second.
    """
    displacements = np.asarray(displacements_radar, dtype=float)
    imu = np.asarray(imus)
    radar_to_imu = _rotation_array(radar_to_imu)
    if displacements.ndim != 2 or displacements.shape[1] != 3:
        raise ValueError("displacements_radar must have shape (N, 3)")
    if imu.ndim != 2 or imu.shape[1] < 6:
        raise ValueError("imus must have shape (M, 6) or more columns")
    count = len(displacements)
    # prefix[t] 是前 t 个 IMU 样本的 z 轴角速度之和；一次累计代替逐帧重复求和。
    # 只使用第六列 gz，最终轨迹没有积分 roll/pitch。
    prefix = np.empty(len(imu) + 1, dtype=float)
    prefix[0] = 0.0
    np.cumsum(imu[:, 5], out=prefix[1:])
    # 第 k 帧（从 0 计）使用前 10*(k+1) 个 IMU 样本，不按时间戳插值。
    # minimum 仅防止越界；IMU 长度不足时不会自动补出缺少的角速度样本。
    endpoints = np.minimum((np.arange(count) + 1) * 10, len(imu))
    # 累积角速度 /100 得到角度；0.035*k 是原有固定修正，约为 0.35°/s，非在线估计。
    yaw_degrees = prefix[endpoints] / 100 + 0.035 * np.arange(count)
    yaw = yaw_degrees / 180 * 3.1415926
    cosine, sine = np.cos(yaw), np.sin(yaw)
    # 每帧构造绕 z 轴的 R_WI；第三轴保持不变，数组形状为 (N, 3, 3)。
    rotations = np.zeros((count, 3, 3), dtype=float)
    rotations[:, 0, 0] = cosine
    rotations[:, 0, 1] = -sine
    rotations[:, 1, 0] = sine
    rotations[:, 1, 1] = cosine
    rotations[:, 2, 2] = 1.0
    # 列向量公式：Δp_W = R_WI @ R_IR @ Δp_R。
    # displacements 的每一行存一个向量，所以第一步在右侧乘 R_IR.T。
    displacement_imu = displacements @ radar_to_imu.T
    displacement_world = np.matmul(rotations, displacement_imu[:, :, None])[:, :, 0]
    # 以零位置为积分起点；返回数组的第 0 行已包含第一个雷达周期的位移。
    trajectory = np.cumsum(displacement_world, axis=0)
    return trajectory, rotations


def EmoRI_SLAM_fast(EmoRI, data, indices, rotates, radar_to_imu,
                    chunk_size=65536):
    """Place calibrated static radar points into the trajectory's world frame.

    Preserves the calibrated EmoRI_SLAM output's initial zero row, original
    frame numbers, point order, and range/height gates. Uses each point's actual
    frame ID, so a missing radar estimate never shifts later map points.
    ``indices`` has the cumulative static-point counts returned by MSTC.
    Rotations must include the same trajectory alignment as ``EmoRI``.

    The fixed radar_to_imu mounting rotation is required.
    """
    radar_to_imu = _rotation_array(radar_to_imu)
    trajectory = np.asarray(EmoRI)
    rotations = np.asarray(rotates)
    data = np.asarray(data)
    boundaries = np.asarray(indices)
    # data 每行为 [x_R, y_R, z_R, 原始雷达帧号]；boundaries 划分各组静态点。
    # 第四列是帧号，已不是原始多普勒；必须用原始帧号取位姿，避免缺帧后错位。
    if boundaries.ndim != 1:
        raise ValueError("indices must be one-dimensional cumulative point counts")
    if not np.all(np.isfinite(boundaries)) or not np.all(boundaries == boundaries.astype(np.intp)):
        raise ValueError("Static point boundaries must be finite integers")
    boundaries = boundaries.astype(np.intp)
    if np.any(boundaries < 0) or np.any(boundaries > len(data)) or np.any(np.diff(boundaries) < 0):
        raise ValueError("Static point boundaries must be increasing and within data")
    if int(chunk_size) != chunk_size or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    chunk_size = int(chunk_size)
    # 首行全零是旧输出格式保留的占位行；即使没有静态点也返回这一行。
    chunks = [np.zeros((1, 4))]
    if len(boundaries) < 2 or boundaries[0] == boundaries[-1]:
        return chunks[0]

    selected = data[boundaries[0]:boundaries[-1]]
    frame_values = selected[:, 3]
    if not np.all(np.isfinite(frame_values)) or not np.all(frame_values == frame_values.astype(np.intp)):
        raise ValueError("Static point frame numbers must be finite integers")
    frame_ids = frame_values.astype(np.intp)
    if np.any(frame_ids < 0) or np.any(frame_ids >= len(trajectory)) or np.any(frame_ids >= len(rotations)):
        raise IndexError("Static point frame number is outside the trajectory")
    group_lengths = np.diff(boundaries)
    starts = boundaries[:-1][group_lengths > 0] - boundaries[0]
    expected = np.repeat(frame_ids[starts], group_lengths[group_lengths > 0])
    if not np.array_equal(frame_ids, expected):
        raise ValueError("Each static point group must belong to one radar frame")

    # 合成雷达→世界旋转 R_WR = R_WI @ R_IR，外参平移按零处理。
    world_from_radar = rotations @ radar_to_imu
    for start in range(0, len(selected), chunk_size):
        stop = min(start + chunk_size, len(selected))
        ids = frame_ids[start:stop]
        # 分块批量旋转，限制临时数组大小；这里只计算相对雷达位置的世界系偏移。
        offsets = np.matmul(
            selected[start:stop, None, :3],
            world_from_radar[ids].transpose(0, 2, 1),
        )[:, 0, :]
        distances = np.linalg.norm(offsets, axis=1)
        # 保留距离 [0.1, 6] m、旋转后相对高度 (-0.5, 2) m 的点。
        # 高度筛选在加上轨迹平移之前执行，不是世界坐标中的绝对高度限制。
        keep = ((distances <= 6) & (distances >= 0.1)
                & (offsets[:, 2] < 2) & (offsets[:, 2] > -0.5))
        kept_ids = ids[keep]
        # 最终坐标 p_W = R_WR @ p_R + t_W；输出第四列继续保存原始雷达帧号。
        points_world = offsets[keep] + trajectory[kept_ids]
        chunks.append(np.column_stack((points_world, kept_ids)))
    return np.concatenate(chunks, axis=0)


# 仅由 Fusion_g_fast 调用，外参已检查，IMU 已复制；以下保留原缺帧算法的索引约定。
def _fusion_g_gap_reference(mmwave_indices, mmwave_vel, imus, radar_to_imu):
    """Preserve the original two-pass gap compensation and sample indexing.

    First estimate and remove gravity between valid radar velocity anchors.
    Then integrate corrected acceleration only for missing radar frames.
    """
    # 第一遍：以相邻有效雷达帧作为速度锚点，估计区间内需要扣除的重力项。
    # 初始左锚点沿用旧算法的“帧 0、零速度”，不是从第 0 帧雷达测速初始化。
    anchor0 = np.array([0, 0, 0])
    idx_anchor0 = 0
    for i in range(len(mmwave_indices)):
        if mmwave_indices[i] == -1 or i == idx_anchor0:
            continue
        idx_anchor1 = i
        anchor1 = mmwave_vel[mmwave_indices[i], :]
        # 雷达速度差从雷达系转到 IMU 系，与加速度积分得到的速度增量作比较。
        delta_mmv = radar_to_imu @ (anchor1 - anchor0)

        Rt = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        R_res = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        Rs = []
        for j in range(idx_anchor0 * 10, idx_anchor1 * 10):
            # 单个 IMU 周期为 0.01 s；三轴角速度先乘时间，再从角度转为弧度。
            # 这里的三轴旋转只用于缺帧重力补偿，不改变上面的 yaw-only 轨迹积分。
            ax, ay, az = np.deg2rad([imus[j, 3] / 100, imus[j, 4] / 100, imus[j, 5] / 100])
            Rx = np.matrix([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
            Ry = np.matrix([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
            Rz = np.matrix([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
            R0 = Rz * Ry * Rx
            Rt = R0 * Rt
            # 按旧算法累积并转置旋转；Rs 保存各采样点的重力变换，R_res 为其和。
            Rg = Rt.transpose()
            Rs.append(Rg)
            R_res = R_res + Rg

        # 模型约定：sum_V - Δv_radar = 0.01 * R_res * g；据此求区间重力向量 g。
        # 这里保留原有直接求逆和乘法顺序，以维持原来的数值结果。
        sum_V = np.sum(imus[idx_anchor0 * 10:idx_anchor1 * 10, 0:3], axis=0) / 100
        g = R_res.I * np.matrix(sum_V - delta_mmv).transpose() * 100
        for j in range(idx_anchor0 * 10, idx_anchor1 * 10):
            # 将估计的重力投到各 IMU 采样坐标，原位扣除的是私有副本中的加速度。
            v_I = Rs[j - idx_anchor0 * 10] * g
            imus[j, 0:3] = imus[j, 0:3] - v_I.transpose()
        idx_anchor0 = i
        anchor0 = anchor1

    # 第二遍：有效帧仍直接使用雷达速度；缺帧才积分加速度并接续上一时刻速度。
    # 第一遍利用了后续有效锚点，因此这是离线补偿；末尾无后续锚点的区间未扣重力。
    ys = []
    last_v = np.array([0, 0, 0])
    for i in range(len(mmwave_indices)):
        if mmwave_indices[i] != -1:
            last_v = mmwave_vel[mmwave_indices[i], :]
            ys.append(last_v / 10)
        else:
            # 沿用旧切片 [10*(i-1), 10*i)，不要与航向积分的结束索引混为一谈。
            # 若首帧 i=0 就缺失，此切片通常为空，配合初始 last_v=0 得到零位移。
            vs = np.sum(imus[10 * (i - 1):10 * i, 0:3], axis=0) / 100
            # 加速度积分得到 IMU 系 Δv；转回雷达系后更新速度，再乘 0.1 s 得到位移。
            vs = radar_to_imu.T @ vs
            last_v = vs + last_v
            ys.append(last_v * 0.1)
    return np.array(ys)
