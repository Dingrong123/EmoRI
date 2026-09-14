"""Accelerated IWR6843 reader; local dependencies are contained in utilities.

Default CLI: environment 1, experiment 1, captures 0..3, 300 frames/file.
Inputs and NPY point-cloud outputs live under the local data/ directory. Use --workers 1 for
serial processing or --help for selection and parallel-processing options.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Iterable

_IMPORT_STARTED = time.perf_counter()
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
from .utilities.capture_io import (
    BYTES_PER_CHIRP, NUM_ADC_SAMPLES, NUM_RX, _validate_file,
    DEFAULT_INPUT_DIR, DEFAULT_PATTERN, _save_npy,
)
from .utilities.get_params_value import get_params_value
from .utilities.fft_range_fast import fft_range_fast
from .utilities.fft_doppler_fast import fft_doppler_fast
from .utilities.detection_6843_accelerated import cfar_RV_accelerated, peakGrouping_accelerated
from .utilities.music_6843_accelerated import naive_music_6843_accelerated
from .utilities.data_paths import DEFAULT_DATA_DIR, resolve_path, relative_path

_IMPORT_SECONDS = time.perf_counter() - _IMPORT_STARTED
HERE = Path(".")
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "point_cloud"
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)
DEFAULT_ENVS = (1,)
DEFAULT_EXPS = (1,)
DEFAULT_NUMS = (0, 1, 2, 3)
_WORKER_PARAMS = None
_WORKER_PFA = None


def decode_adc_data_accelerated(raw: np.ndarray) -> np.ndarray:
    """Decode I0,I1,Q0,Q1 directly into complex128 real/imaginary fields.

    Avoids the reference decoder's intermediate float64 and complex arrays.
    The returned (4, chirps*512) frame owns its storage; input is not modified.
    """
    raw = np.asarray(raw)
    if raw.ndim != 1 or raw.size % (NUM_ADC_SAMPLES * NUM_RX * 2):
        raise ValueError("ADC data must be a flat array containing complete 8192-byte chirps.")
    # 文件按 chirp、接收天线、采样对排列；每四个 int16 为 I0、I1、Q0、Q1。
    # 转置后为 (Rx, chirps, samples/2, 4)，把同一采样的 I、Q 写入复数实部和虚部。
    source = raw.reshape(-1, NUM_RX, NUM_ADC_SAMPLES // 2, 4).transpose(1, 0, 2, 3)
    frame = np.empty((NUM_RX, source.shape[1], NUM_ADC_SAMPLES // 2, 2), dtype=np.complex128)
    frame.real = source[..., :2]
    frame.imag = source[..., 2:]
    # 每行对应一个接收天线，列方向按 chirp 顺序连续存放 512 个复数采样。
    return frame.reshape(NUM_RX, -1)


def iter_data_frames_accelerated(file_name, *, tx=3, loops=100,
                                 start_frame=0, frame_count=None):
    """Stream the same complete frames with the accelerated ADC decoder."""
    if tx <= 0 or loops <= 0 or start_frame < 0:
        raise ValueError("tx and loops must be positive; start_frame must be nonnegative.")
    path, size = _validate_file(file_name)
    # 一帧包含 tx * loops 个 chirp（默认 3 * 100）；起始帧通过字节偏移直接定位。
    bytes_per_frame = BYTES_PER_CHIRP * tx * loops
    available = size // bytes_per_frame
    if frame_count is None:
        frame_count = available - start_frame
    if frame_count < 0 or start_frame + frame_count > available:
        raise ValueError(f"{relative_path(path)} contains {available} complete frames; requested [{start_frame}, {start_frame+frame_count}).")
    with path.open("rb") as stream:
        stream.seek(start_frame * bytes_per_frame)
        for _ in range(frame_count):
            # 原始 ADC 为小端有符号 16 位整数；只读取当前帧，控制内存占用。
            raw = np.fromfile(stream, dtype="<i2", count=bytes_per_frame // 2)
            if raw.size * 2 != bytes_per_frame:
                raise EOFError(f"Capture was truncated while reading {relative_path(path)}.")
            yield decode_adc_data_accelerated(raw)


def _process_frame_owned(frame, params, pfa, return_intermediates=False):
    """Process a private writable complex128 frame; never given caller storage."""
    expected = (params.Rx, params.Tx * params.samples * 100)
    if frame.shape != expected:
        raise ValueError(f"Expected decoded frame shape {expected}; got {frame.shape}.")
    if params.Tx != 3 or params.Rx != 4:
        raise ValueError("This port preserves the original three-TX/four-RX antenna configuration.")
    # 延续原采集配置的通道相位约定：第 1、3 个 Rx 通道反号（相位差 pi）。
    frame[[0, 2], :] *= -1
    data_chirp = frame.reshape(params.Rx, params.Tx * 100, params.samples).transpose(0, 2, 1)
    # TDM 时隙依次对应 TX1、TX3、TX2；每路整理为 (samples, Rx, loops)。
    # 此次序决定后续 12 个虚拟天线的位置，不能直接按 TX 编号重新排序。
    chirps = [data_chirp[:, :, k::3].transpose(1, 0, 2) for k in range(3)]
    # 距离 FFT 沿采样轴，多普勒 FFT 沿慢时间轴并 fftshift；当前实现均不加窗。
    # 每路输出形状为 (距离 bin, 4 个 Rx, 多普勒 bin)。
    ranged = [fft_range_fast(chirp, params.fft_Rang, 1) for chirp in chirps]
    doppler = [fft_doppler_fast(cube, params.fft_Vel, 0) for cube in ranged]
    # 仅用 TX3 的四路幅值均值做检测（不是幅值平方）；保持原算法的检测尺度。
    rd = np.mean(np.abs(doppler[1]), axis=1)
    # 二维 CFAR 的保护区半宽为 5、外侧训练带厚度为 5，pfa 为目标虚警率。
    # 检测列依次为 [多普勒 bin, 距离 bin, 幅值]，bin 保留 MATLAB 的 1 起始编号。
    detections = cfar_RV_accelerated(rd, 5, 5, pfa)
    grouped = peakGrouping_accelerated(detections) if detections.size else np.empty((3, 0))
    points = np.empty((0, 5), dtype=np.float64)
    merged = None
    if grouped.size or return_intermediates:
        # 有检测点或要求中间结果时，合并三路 Tx，构造 MUSIC 使用的 12 通道虚拟阵列。
        merged = np.concatenate(doppler, axis=1)
    if grouped.size:
        # MUSIC 给出方向分量，乘距离后得到雷达坐标系位置；查表时 bin 需减 1。
        x, y, z, rcs = naive_music_6843_accelerated(merged, grouped)
        velocities = np.asarray(params.vel_grid).reshape(-1)[grouped[0].astype(np.intp)-1]
        ranges = np.asarray(params.rng_grid).reshape(-1)[grouped[1].astype(np.intp)-1]
        # 点云五列为 [x(m), y(m), z(m), 径向速度(m/s), 相对强度]。
        # rcs 沿用原变量名，实际为帧内归一化幅值指标，并非标定后的物理 RCS。
        points = np.real(np.column_stack((x * ranges, y * ranges, z * ranges, velocities, rcs)))
    if return_intermediates:
        return points, {
            "data_frame": frame, "data_chirp": data_chirp,
            "Rangedata_tx1": ranged[0], "Rangedata_tx3": ranged[1], "Rangedata_tx2": ranged[2],
            "Dopplerdata_tx1": doppler[0], "Dopplerdata_tx3": doppler[1], "Dopplerdata_tx2": doppler[2],
            "Dopdata_sum": rd, "Resl_indx": detections, "detout": grouped,
            "Dopdata_merge": merged, "pc_body": points,
        }
    return points


def process_frame_accelerated(data_frame: np.ndarray, params: Any = None,
                              angle_search_matrix=None, pfa=3e-2, *,
                              return_intermediates=False):
    """Convert a frame to [x,y,z,velocity,RCS], preserving caller input.

    The angle_search_matrix argument is retained for compatibility; the
    original selected MUSIC implementation also leaves it unused.
    """
    params = get_params_value() if params is None else params
    # 核心处理会原地反号；公开接口先复制，避免修改调用者提供的 ADC 帧。
    frame = np.array(data_frame, dtype=np.complex128, copy=True)
    return _process_frame_owned(frame, params, pfa, return_intermediates)


def _join_blocks(blocks):
    chunks, offsets = [], [0]
    # offsets 是从 0 起始的累计点数；第 k 帧使用 points[offsets[k]:offsets[k+1]]。
    # 将各块局部偏移加上此前总点数，保留空帧的重复边界，防止帧与 IMU 错位。
    for points, block_offsets in blocks:
        if points.size:
            chunks.append(points)
        offset = offsets[-1]
        offsets.extend(int(value) + offset for value in block_offsets[1:, 0])
    if offsets[-1] > np.iinfo(np.int32).max:
        raise OverflowError("The original int32 point offsets cannot represent this capture.")
    result = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float64)
    return result, np.asarray(offsets, dtype=np.int32).reshape(-1, 1)


def _process_capture_serial(filename, start_frame, frames, params, pfa):
    chunks, offsets = [], [0]
    for frame in iter_data_frames_accelerated(filename, tx=params.Tx, loops=100,
                                             start_frame=start_frame, frame_count=frames):
        # 解码器已创建独占帧数组，可直接原地处理，省去公开接口的整帧复制。
        points = _process_frame_owned(frame, params, pfa)
        if points.size:
            chunks.append(points)
        offsets.append(offsets[-1] + len(points))
    if offsets[-1] > np.iinfo(np.int32).max:
        raise OverflowError("The original int32 point offsets cannot represent this capture.")
    result = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float64)
    return result, np.asarray(offsets, dtype=np.int32).reshape(-1, 1)


def _initialize_worker(params, pfa):
    # 每个子进程初始化一次参数副本；逐块任务仅传文件名和帧范围，减少序列化。
    global _WORKER_PARAMS, _WORKER_PFA
    _WORKER_PARAMS, _WORKER_PFA = params, pfa


def _process_block(task):
    filename, start_frame, frames = task
    return _process_capture_serial(filename, start_frame, frames, _WORKER_PARAMS, _WORKER_PFA)


def _pool(workers, params, pfa):
    # 显式使用 spawn，兼容 Windows；子进程自行读取 ADC，不传输大型 FFT 数据块。
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                               initializer=_initialize_worker, initargs=(params, pfa))


def process_capture_accelerated(filename, *, start_frame=0, frames=300, pfa=3e-2,
                                params=None, angle_search_matrix=None,
                                workers=DEFAULT_WORKERS, chunk_frames=16, _executor=None):
    """Process frames serially or in ordered blocks; offsets stay int32 columns.

    Workers read their own blocks and return only point arrays and offsets,
    avoiding transmission of large ADC/FFT cubes. Result order is deterministic.
    In standalone scripts, invoke multiprocessing below an __main__ guard;
    interactive environments may instead select workers=1.
    """
    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    if isinstance(chunk_frames, bool) or int(chunk_frames) != chunk_frames or chunk_frames < 1:
        raise ValueError("chunk_frames must be a positive integer")
    if int(frames) != frames or frames < 0 or int(start_frame) != start_frame or start_frame < 0:
        raise ValueError("frames and start_frame must be nonnegative integers")
    workers, chunk_frames, frames, start_frame = map(int, (workers, chunk_frames, frames, start_frame))
    params = get_params_value() if params is None else params
    path, size = _validate_file(filename)
    available = size // (BYTES_PER_CHIRP * params.Tx * 100)
    if start_frame + frames > available:
        raise ValueError(f"{relative_path(path)} contains {available} frames; requested [{start_frame}, {start_frame+frames}).")
    if workers == 1 or frames <= chunk_frames:
        return _process_capture_serial(path, start_frame, frames, params, pfa)
    # 连续帧分块：摊薄进程通信开销，同时限制每个工作任务的内存占用。
    tasks = [(str(path), first, min(chunk_frames, start_frame + frames - first))
             for first in range(start_frame, start_frame + frames, chunk_frames)]
    context = nullcontext(_executor) if _executor is not None else _pool(int(workers), params, pfa)
    with context as executor:
        # map 按任务提交顺序返回；即使各块完成先后不同，点云帧顺序也保持不变。
        return _join_blocks(executor.map(_process_block, tasks, chunksize=1))


def run_accelerated(input_dir=None, output_dir=None, *, data_dir=DEFAULT_DATA_DIR,
                    envs: Iterable[int] = DEFAULT_ENVS, exps: Iterable[int] = DEFAULT_EXPS,
                    nums: Iterable[int] = DEFAULT_NUMS, frames_per_file=300, start_frame=0,
                    pfa=3e-2, pattern=DEFAULT_PATTERN, check_inputs=False,
                    workers=DEFAULT_WORKERS, chunk_frames=16):
    """Preserve env/experiment/capture/frame order and save two NPY arrays.

    Relative directories are based on the entry scripts' directory.
    """
    if (int(frames_per_file) != frames_per_file or frames_per_file <= 0
            or int(start_frame) != start_frame or start_frame < 0 or not 0 < pfa < 1):
        raise ValueError("frames_per_file must be positive, start_frame >= 0, and 0 < pfa < 1.")
    if (isinstance(workers, bool) or int(workers) != workers or workers < 1
            or isinstance(chunk_frames, bool) or int(chunk_frames) != chunk_frames or chunk_frames < 1):
        raise ValueError("workers and chunk_frames must be positive integers")
    workers, chunk_frames, frames_per_file, start_frame = map(int, (workers, chunk_frames, frames_per_file, start_frame))
    envs, exps, nums = list(envs), list(exps), list(nums)
    if not envs or not exps or not nums:
        raise ValueError("At least one environment, experiment, and capture number is required.")
    for values, name in ((envs, "envs"), (exps, "exps"), (nums, "nums")):
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {name} would repeat frames or overwrite a result.")
    # 相对路径以入口脚本目录为基准；默认点云目录同时也是 SLAM 的输入目录。
    data_dir = resolve_path(data_dir)
    input_dir = resolve_path(input_dir) if input_dir is not None else data_dir / 'radar_raw_data'
    output_dir = resolve_path(output_dir) if output_dir is not None else data_dir / 'point_cloud'
    captures = {(env, exp): [input_dir / pattern.format(env=env, exp=exp, num=num) for num in nums]
                for env in envs for exp in exps}
    bytes_required = (start_frame + frames_per_file) * BYTES_PER_CHIRP * 3 * 100
    errors = []
    for files in captures.values():
        for path in files:
            if not path.is_file():
                errors.append(f"Missing capture: {relative_path(path)}")
            else:
                size = path.stat().st_size
                if size < bytes_required or size % BYTES_PER_CHIRP:
                    errors.append(f"Invalid capture size: {relative_path(path)} ({size} bytes; need >= {bytes_required} and complete chirps)")
    if errors:
        raise ValueError("Input validation failed:\n" + "\n".join(errors))
    total = len(captures) * len(nums) * frames_per_file
    print(f"Validated {len(captures)*len(nums)} captures; {total} frames; {workers} workers.", flush=True)
    if check_inputs:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    params = get_params_value()
    written, capture_timings = [], []
    started = time.perf_counter()
    context = _pool(int(workers), params, pfa) if workers > 1 and frames_per_file > chunk_frames else nullcontext(None)
    # 整次运行复用同一进程池，避免每个采集文件都支付 Windows 进程启动开销。
    with context as executor:
        for (env, exp), files in captures.items():
            blocks = []
            for path in files:
                step = time.perf_counter()
                points, offsets = process_capture_accelerated(
                    path, start_frame=start_frame, frames=frames_per_file, pfa=pfa,
                    params=params, workers=workers, chunk_frames=chunk_frames, _executor=executor,
                )
                elapsed = time.perf_counter() - step
                blocks.append((points, offsets))
                capture_timings.append({"capture": relative_path(path), "frames": frames_per_file,
                                        "points": len(points), "seconds": elapsed})
                print(f"Processed {path.name}: {frames_per_file} frames, {len(points)} points, {elapsed:.3f} s.", flush=True)
            # 按 nums 的输入顺序拼接采集文件，输出点表及 (总帧数+1, 1) 的 int32 边界表。
            # 两个 NPY 文件各保存一个数组，np.load 直接读取，无需按 MAT 变量名取值。
            all_data_list, all_data_index = _join_blocks(blocks)
            point_path = output_dir / f"pypc_{env}_{exp}.npy"
            index_path = output_dir / f"pypc_{env}_{exp}_indices.npy"
            _save_npy(point_path, all_data_list)
            _save_npy(index_path, all_data_index)
            written.extend((point_path, index_path))
            print(f"Finished! {env} {exp}: {len(all_data_list)} points -> {relative_path(point_path)}", flush=True)
    elapsed = time.perf_counter() - started
    report = {"path_base": "directory containing the entry scripts",
              "data_dir": relative_path(data_dir), "input_dir": relative_path(input_dir), "output_dir": relative_path(output_dir),
              "envs": envs, "exps": exps, "nums": nums, "frames_per_file": frames_per_file,
              "start_frame": start_frame, "frames_total": total, "pfa": pfa,
              "workers": workers, "chunk_frames": chunk_frames,
              "processing_and_saving_seconds": elapsed, "module_import_seconds": _IMPORT_SECONDS,
              "captures": capture_timings, "output_files": list(map(relative_path, written)),
              "output_format": "npy",
              "algorithm": "Original FFT/CFAR/peak/MUSIC conventions and complex128 precision",
              "order": "Input order preserved regardless of worker completion order"}
    (output_dir / "read_6843_accelerated_timing.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Completed in {elapsed:.3f} s. Results (relative to script directory): {relative_path(output_dir)}", flush=True)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog='Relative paths are based on the directory containing this script, not the working directory.')
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Common data directory for both entry points")
    parser.add_argument("--input-dir", type=Path, default=None, help="Override DATA_DIR/radar_raw_data")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override DATA_DIR/point_cloud")
    parser.add_argument("--envs", type=int, nargs="+", default=list(DEFAULT_ENVS))
    parser.add_argument("--exps", type=int, nargs="+", default=list(DEFAULT_EXPS))
    parser.add_argument("--nums", type=int, nargs="+", default=list(DEFAULT_NUMS))
    parser.add_argument("--frames-per-file", type=int, default=300)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--pfa", type=float, default=3e-2)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Worker processes; 1 disables multiprocessing")
    parser.add_argument("--chunk-frames", type=int, default=16, help="Frames per ordered work block")
    parser.add_argument("--check-inputs", action="store_true", help="Validate captures without processing or writing results")
    args = parser.parse_args(argv)
    try:
        run_accelerated(**vars(args))
    except (OSError, ValueError, OverflowError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    # spawn 会重新导入入口模块；主程序保护避免子进程再次启动整个处理流程。
    mp.freeze_support()
    raise SystemExit(main())
