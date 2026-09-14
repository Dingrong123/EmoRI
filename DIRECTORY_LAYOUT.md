# Directory guide

The self-contained project root can have any directory name. `EmoRI/` in the release archive is an example, not an import requirement; names such as `EmoRI-main` and names containing spaces or Chinese characters also work. The two runnable scripts are `read_6843_accelerated.py` and `EmoRI.py`; their local Python dependencies are all in `utilities/` and are located from the entry scripts' own file locations.

All relative runtime CLI paths are resolved relative to the entry-point directory. Inputs are grouped under `data/point_cloud`, `data/IMU_and_GT`, and optional `data/radar_raw_data`. Results default to `data/slam_results`. Move these files together when relocating or renaming the project. Prefer `--data-dir` for a custom dataset; the legacy `--project-dir` option still selects `PROJECT_DIR/python_version/data` according to its original compatibility contract.

Run either script directly from the project root. For Python API use from that same directory, import `from EmoRI import process_sequence` and pass explicit sequence IDs. Optional `python -m EmoRI.EmoRI` launch is available from the parent directory when the actual root name is the valid Python identifier `EmoRI`; a directory containing spaces or hyphens should use direct script launch. Quote paths containing spaces when launching a script from elsewhere.

See [the input-data overview](README.md#input-data), [the data contract](data/README.md), and [implementation details](docs/IMPLEMENTATION.md). The old MATLAB project and external helper directories are not required.

The local working copy may retain MAT backups, raw recordings, caches, and historical validation reports. They are excluded by `.gitignore` and are not part of the prepared GitHub source/sample archive.
