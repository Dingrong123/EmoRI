"""Capture constants, file validation and atomic NPY output used by the reader.

Extracted from get_data_6843.py and read_6843_fast.py so the accelerated
pipeline does not import the unused reference processing functions.
"""
from pathlib import Path

import numpy as np

from .data_paths import DEFAULT_DATA_DIR, resolve_path, relative_path

NUM_ADC_SAMPLES = 512
NUM_RX = 4
BYTES_PER_CHIRP = NUM_ADC_SAMPLES * NUM_RX * 2 * 2
DEFAULT_INPUT_DIR = DEFAULT_DATA_DIR / "radar_raw_data"
DEFAULT_PATTERN = "0516env_new_{env}_{exp}_Raw_{num}.bin"


def _validate_file(file_name: str | Path) -> tuple[Path, int]:
    path = resolve_path(file_name)
    size = path.stat().st_size
    if size % BYTES_PER_CHIRP:
        raise ValueError(f"Incomplete ADC chirp in {relative_path(path)}: {size} bytes is not divisible by {BYTES_PER_CHIRP}.")
    return path, size


def _save_npy(path: Path, value) -> None:
    """Write one numeric array as NPY, then replace its destination."""
    path = resolve_path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        # 传入文件对象，避免 np.save 给临时文件再追加 .npy；保持数组形状和 dtype。
        # 写入完成后再替换正式文件，避免读取程序看到只写了一部分的数组。
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
