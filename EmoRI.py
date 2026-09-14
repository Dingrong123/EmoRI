"""Accelerated counterpart of Radar_SLAM_v2.py; original code is preserved.

Same fixed extrinsic, 20 MSTC hypotheses, filtering and IMU yaw correction.
GT trajectory alignment remains disabled as in the source before relocation.
Run --help for input/output and plotting options.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

_IMPORT_STARTED = perf_counter()
import numpy as np

# 按脚本位置加载本地包；内部名称固定，但磁盘上的主目录可任意命名。
# 直接运行、python -m 和 Python API 使用同一依赖名称，供 spawn / Numba 恢复模块。
import importlib.util as _importlib_util
import sys as _sys

_PACKAGE_NAME = "_emori_runtime"
_PACKAGE_DIR = Path(__file__).resolve().parent
_runtime_package = _sys.modules.get(_PACKAGE_NAME)
if _runtime_package is None:
    _package_spec = _importlib_util.spec_from_file_location(
        _PACKAGE_NAME, _PACKAGE_DIR / "__init__.py",
        submodule_search_locations=[str(_PACKAGE_DIR)],
    )
    _runtime_package = _importlib_util.module_from_spec(_package_spec)
    _sys.modules[_PACKAGE_NAME] = _runtime_package
    _package_spec.loader.exec_module(_runtime_package)
elif Path(getattr(_runtime_package, "__file__", "")).resolve() != _PACKAGE_DIR / "__init__.py":
    raise ImportError("A different EmoRI checkout is already loaded; use a new Python process for this copy.")
__package__ = _PACKAGE_NAME
from .utilities.radar_slam_mstc_fast import MSTC_fast, get_backend_info
from .utilities.radar_slam_motion_fast import Fusion_g_fast, integrate_motion_fast, EmoRI_SLAM_fast
from .utilities.data_paths import DEFAULT_DATA_DIR, resolve_path, relative_path
_IMPORT_SECONDS = perf_counter() - _IMPORT_STARTED

# 命令行未指定 --envs / --experiments 时使用这里的实验选择。
DEFAULT_ENVS = (1,)
DEFAULT_EXPERIMENTS = (1,)

# 固定安装外参：按列向量约定，v_R = R_IMU_TO_RADAR @ v_I。
# 这里只读取已标定数值，不在运行时重新拟合；雷达与 IMU 的平移外参按零处理。
R_IMU_TO_RADAR = np.array([
    [0.06280492776998943, 0.9890933401528303, 0.1332287713413427],
    [0.994246658175081, -0.07361323317790894, 0.07781178965155619],
    [0.0867705235404837, 0.12757529685018243, -0.9880260218628343],
], dtype=np.float64)
R_RADAR_TO_IMU = R_IMU_TO_RADAR.T


def plot_results(trajectory, truth, map_points, *, show=True, save_path=None):
    """Preserve the current script's two map layers, trajectory and GT view."""
    if not show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(10, 20))
    axes = figure.add_subplot(111, projection='3d')
    axes.scatter(map_points[:, 0], map_points[:, 1], map_points[:, 2],
                 c=np.arange(len(map_points)), s=3, marker='.')
    axes.scatter(map_points[:, 0], map_points[:, 1], map_points[:, 2],
                 c='b', s=0.5, marker='.')
    axes.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                 c=np.arange(len(trajectory)), s=2, marker='.')
    axes.scatter(truth[:, 0], truth[:, 1], truth[:, 2], c='r', s=2, marker='.')
    axes.view_init(elev=90, azim=-90)
    axes.set_xlabel('X (m)')
    axes.set_ylabel('Y (m)')
    axes.set_zlabel('Z (m)')
    axes.axis('equal')
    # 单独统计图像渲染时间，等待用户关闭窗口的时间不计入算法处理耗时。
    figure.canvas.draw()
    if save_path is not None:
        save_path = resolve_path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=120)
    return figure, plt


def process_sequence(env=2, experiment=2, *, data_dir=None, project_dir=None,
                     output_dir=None, backend='auto', progress=False,
                     plot=True, show=True, save_plot=False,
                     return_intermediates=False):
    """Process one complete recording and save the same three array formats.

    NPY inputs are in DATA_DIR/point_cloud and DATA_DIR/IMU_and_GT. Outputs default
    to DATA_DIR/slam_results. project_dir is retained only as a compatibility
    option that locates PROJECT_DIR/python_version/data in the new layout.
    All relative path arguments are based on the entry scripts' directory.
    """
    started = perf_counter()
    # 默认输入集中在 data 下；所有相对目录以入口脚本所在目录为基准。
    if data_dir is None:
        data_dir = Path(project_dir) / 'python_version' / 'data' if project_dir is not None else DEFAULT_DATA_DIR
    data_dir = resolve_path(data_dir)
    output_dir = resolve_path(output_dir) if output_dir is not None else data_dir / 'slam_results'
    timings = {}
    step = perf_counter()
    data_path = data_dir / 'point_cloud' / f'pypc_{env}_{experiment}.npy'
    index_path = data_dir / 'point_cloud' / f'pypc_{env}_{experiment}_indices.npy'
    imu_path = data_dir / 'IMU_and_GT' / f'0516env_aligned_imu_{env}_{experiment}.npy'
    truth_path = data_dir / 'IMU_and_GT' / f'0516env_aligned_gt_{env}_{experiment}.npy'
    # 点云 data 的每行为 [x, y, z, 多普勒速度, RCS]，坐标单位 m、速度单位 m/s。
    # indices 有 N+1 项，第 k 帧点云为 data[indices[k]:indices[k+1]]。
    # NPY 文件直接保存数组；索引落盘仍为 int32 列向量，读取后展平便于按帧切片。
    data = np.real(np.load(data_path, allow_pickle=False))
    indices = np.load(index_path, allow_pickle=False).reshape(-1)
    # IMU 六列按 [ax, ay, az, gx, gy, gz] 解释：输入加速度以 g 为单位，角速度为 °/s。
    # 此处读取的是已对齐的记录；后续算法固定按 100 Hz IMU / 10 Hz 雷达索引配对。
    imus = np.asarray(np.load(imu_path), dtype=np.float64)
    truth = np.asarray(np.load(truth_path), dtype=np.float64)
    if truth.shape != (len(indices) - 1, 3):
        raise ValueError(f'GT shape {truth.shape} does not match {len(indices)-1} radar frames.')
    timings['load_seconds'] = perf_counter() - step

    step = perf_counter()
    # 1. 根据静态散射点的多普勒估计雷达坐标系速度，不使用 IMU 或真值。
    # velocities 只存有效速度；frame_indices 为每个原始帧对应的速度行号，失败为 -1。
    velocities, frame_indices, static_points, static_indices = MSTC_fast(
        data, indices, backend=backend, progress=progress,
    )
    timings['mstc_seconds'] = perf_counter() - step

    step = perf_counter()
    # 2. 有效帧直接使用速度 × 0.1 s；只有缺失速度的帧才需要 IMU 加速度补偿。
    if np.any(frame_indices == -1):
        # 缺帧补偿内部以 m/s² 计算；无缺帧时不读取加速度，省去这次单位转换。
        imus[:, :3] *= 9.81
    displacements = Fusion_g_fast(frame_indices, velocities, imus, radar_to_imu=R_RADAR_TO_IMU)
    timings['fusion_seconds'] = perf_counter() - step

    step = perf_counter()
    # 3. 雷达位移先转到 IMU 坐标系，再按陀螺仪积分出的航向转到世界坐标系并累加。
    # orientations 是每帧的 IMU→世界旋转；当前只积分 yaw，保留固定航向修正。
    trajectory, orientations = integrate_motion_fast(displacements, imus, R_RADAR_TO_IMU)
    timings['yaw_and_integration_seconds'] = perf_counter() - step

    step = perf_counter()
    # 4. 用同一组位姿放置静态点云，保证地图与轨迹使用相同的坐标变换。
    # 当前不执行 GT 轨迹配准；真值仅用于后续绘图和误差评价。
    map_points = EmoRI_SLAM_fast(
        trajectory, static_points, static_indices, orientations, radar_to_imu=R_RADAR_TO_IMU,
    )
    timings['mapping_seconds'] = perf_counter() - step

    step = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    point_path = output_dir / f'nradar_slam_{env}_{experiment}.npy'
    trajectory_path = output_dir / f'nradar_motion_{env}_{experiment}.npy'
    truth_out_path = output_dir / f'lidar_motion_{env}_{experiment}.npy'
    np.save(point_path, map_points)
    np.save(trajectory_path, trajectory)
    np.save(truth_out_path, truth)
    timings['save_seconds'] = perf_counter() - step
    timings['processing_seconds'] = perf_counter() - started
    timings['module_import_seconds'] = _IMPORT_SECONDS

    plt = figure = None
    if plot:
        step = perf_counter()
        figure, plt = plot_results(
            trajectory, truth, map_points, show=show,
            save_path=(output_dir / f'trajectory_{env}_{experiment}.png') if save_plot else None,
        )
        timings['plot_render_seconds'] = perf_counter() - step
    else:
        timings['plot_render_seconds'] = 0.0

    backend_info = get_backend_info()
    backend_info['requested_backend'] = backend
    backend_info['selected_backend'] = backend_info['auto_backend'] if backend == 'auto' else backend
    report = {
        'env': int(env), 'experiment': int(experiment), 'frames': len(frame_indices),
        'valid_radar_frames': int(np.count_nonzero(frame_indices >= 0)),
        'static_point_rows': len(static_points), 'map_rows': len(map_points),
        # 每帧三维位置误差的均方根，单位 m；这里没有先做 SVD 或其他轨迹对齐。
        'position_rmse_m': float(np.sqrt(np.mean(np.sum((trajectory - truth)**2, axis=1)))),
        'backend': backend_info, 'timings': timings,
        'path_base': 'directory containing the entry scripts',
        'data_dir': relative_path(data_dir), 'input_format': 'npy', 'input_point_file': relative_path(data_path),
        'input_index_file': relative_path(index_path), 'input_imu_file': relative_path(imu_path), 'input_gt_file': relative_path(truth_path),
        'output_files': [relative_path(point_path), relative_path(trajectory_path), relative_path(truth_out_path)],
        'extrinsic': 'Fixed R_IMU_TO_RADAR constant; no recalibration',
        'evaluation': 'GT alignment disabled; trajectory compared without SVD alignment',
    }
    report_path = output_dir / f'timing_{env}_{experiment}.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Env {env}, experiment {experiment}: {len(frame_indices)} frames, '
          f'processing {timings["processing_seconds"]:.3f}s, '
          f'MSTC {timings["mstc_seconds"]:.3f}s, RMSE {report["position_rmse_m"]:.6f}m.', flush=True)
    print(f'Results (relative to script directory): {relative_path(output_dir)}', flush=True)
    if plot:
        if show:
            plt.show()
        else:
            plt.close(figure)
    if return_intermediates:
        # 带 _aligned 的键名沿用旧接口；当前返回的轨迹和姿态并未用 GT 校正。
        return report, {'radar_velocities': velocities, 'frame_indices': frame_indices,
                        'static_pcs': static_points, 'static_indices': static_indices,
                        'displacements': displacements, 'trajectory_aligned': trajectory,
                        'gt': truth, 'map': map_points, 'world_from_imu_aligned': orientations}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog='Relative paths are based on the directory containing this script, not the working directory.')
    parser.add_argument('--data-dir', type=Path, default=None, help=f'Common data directory; default is {DEFAULT_DATA_DIR}')
    parser.add_argument('--project-dir', type=Path, default=None, help='Compatibility option: use PROJECT_DIR/python_version/data')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory; default is DATA_DIR/slam_results')
    parser.add_argument('--envs', type=int, nargs='+', default=list(DEFAULT_ENVS), help='Environment numbers')
    parser.add_argument('--experiments', '--exps', type=int, nargs='+', default=list(DEFAULT_EXPERIMENTS), help='Experiment numbers')
    parser.add_argument('--backend', choices=['auto', 'numpy', 'numba'], default='auto', help='MSTC acceleration backend')
    parser.add_argument('--progress', action='store_true', help='Display frame progress')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting and matplotlib imports')
    parser.add_argument('--no-show', action='store_true', help='Render without opening or waiting for a plot window')
    parser.add_argument('--save-plot', action='store_true', help='Save the rendered trajectory figure to the output directory')
    args = parser.parse_args(argv)
    if args.no_plot and args.save_plot:
        parser.error('--save-plot cannot be used with --no-plot')
    try:
        for env in args.envs:
            for experiment in args.experiments:
                process_sequence(env, experiment, data_dir=args.data_dir, project_dir=args.project_dir, output_dir=args.output_dir,
                                 backend=args.backend, progress=args.progress, plot=not args.no_plot,
                                 show=not args.no_show, save_plot=args.save_plot)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
