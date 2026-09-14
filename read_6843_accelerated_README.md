# Raw radar reader

`read_6843_accelerated.py` decodes the original TI IWR6843 acquisition format and writes native NumPy point clouds plus cumulative frame boundaries. Run `python read_6843_accelerated.py --help` for the complete CLI.

```bash
python read_6843_accelerated.py --envs 1 --exps 1 --nums 0 1 2 3 --check-inputs
python read_6843_accelerated.py --envs 1 --exps 1 --nums 0 1 2 3 --workers 4
```

The default input is `data/radar_raw_data/`; output is `data/point_cloud/`. All CLI path overrides are relative to the entry-point directory. The output pair is `pypc_1_1.npy` and `pypc_1_1_indices.npy`, loaded directly with `np.load(..., allow_pickle=False)`.

Processing preserves the capture and frame order, double precision, channel signs, TX1/TX3/TX2 ordering, unwindowed FFT convention, CFAR settings, and selected angle-estimation path. Multiprocessing uses ordered frame blocks and a reused process pool. Use `--workers 1` for serial or interactive operation.

For a short check, select `--nums 0 --frames-per-file 10 --output-dir data/preview/point_cloud`. Partial processing must use a separate output directory and matching IMU/GT if subsequently used for motion estimation. Existing sequence outputs are overwritten.

See [raw-capture input requirements](data/README.md#raw-captures), [data formats](data/README.md), and [algorithm details](docs/IMPLEMENTATION.md). Raw capture acquisition and synchronization are outside this release.
