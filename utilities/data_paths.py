"""Relative path settings shared by both entry points.

All relative paths are anchored to the directory containing the entry scripts,
including CLI overrides and direct Python API calls. No drive is configured.
"""
import os
from pathlib import Path

# Only the runtime anchor is absolute; configured paths stay relative.
CODE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("data")


def resolve_path(path: str | Path) -> Path:
    """Locate a code-relative path; explicit absolute paths also remain valid."""
    return (CODE_DIR / Path(path).expanduser()).resolve()


def relative_path(path: str | Path) -> str:
    """Describe paths relative to the entry scripts in logs and JSON reports."""
    resolved = resolve_path(path)
    try:
        return Path(os.path.relpath(resolved, CODE_DIR)).as_posix()
    except ValueError:
        # Windows cannot express an explicitly supplied different drive as a
        # relative path. Preserve that opt-in location rather than fail a run.
        return resolved.as_posix()
