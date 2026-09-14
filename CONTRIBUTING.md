# Contributing

For a reproducibility issue, use the repository's Issues tab and include:

- The entry point, complete command, environment/experiment IDs, and input shapes.
- Python, NumPy, SciPy, Matplotlib, and Numba versions, and the selected backend.
- Expected behavior, actual behavior, and the relevant timing JSON or traceback.
- Whether the data are the included examples or a new sensor recording.

Avoid uploading raw multi-gigabyte captures in an issue. Provide a small input that reproduces the problem when possible.

For changes to numerical processing, explain the algorithmic reason and compare the two sample trajectories and maps with [the recorded outputs](docs/reproduction.json). For an intended algorithm change, report the metric differences. For a performance-only change, check output shapes, frame ordering, numerical agreement, and the affected stage's runtime under comparable cache and backend conditions.

Keep local helpers in `utilities/`, preserve the documented NPY formats, and use `utilities/data_paths.py` for code-relative CLI paths. Check both `--backend numpy` and `--backend numba` when changing MSTC. The provided examples do not cover missing radar velocity, so changes to that branch require a separate targeted case.

The project currently has a prepared-release status; maintainers still need to select software and data licenses before accepting contributions under finalized terms.
