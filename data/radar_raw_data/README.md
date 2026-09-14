# Optional raw radar captures

Download the original reconstructed ADC captures from the [raw-data-v1 release](https://github.com/Dingrong123/EmoRI/releases/tag/raw-data-v1) and place the BIN files in this folder (`data/radar_raw_data/`), preserving their filenames. Download all four parts for the environment you want to process, or all eight files for both environments. These files are GitHub Release assets and are excluded from Git history and the prepared source/sample archive. The source checkout includes the two prepared NPY examples, which run without raw captures.

Each file must be exactly **737,280,000 bytes** and contains **300 radar frames**. Four parts make one 1,200-frame sequence (2,949,120,000 bytes); all eight total **5,898,240,000 bytes (5.90 GB)**. Here, `E` is the environment ID, `X` is the experiment ID, and `K` is the capture part, processed in ascending order `0, 1, 2, 3`.

| Sequence `(E, X)` | Part `K` | Download |
| --- | --- | --- |
| `(1, 1)` | 0 | [0516env_new_1_1_Raw_0.bin](https://github.com/Dingrong123/EmoRI/releases/download/raw-data-v1/0516env_new_1_1_Raw_0.bin) |
| `(1, 1)` | 1 | [0516env_new_1_1_Raw_1.bin](https://github.com/Dingrong123/EmoRI/releases/download/raw-data-v1/0516env_new_1_1_Raw_1.bin) |
| `(1, 1)` | 2 | [0516env_new_1_1_Raw_2.bin](https://github.com/Dingrong123/EmoRI/releases/download/raw-data-v1/0516env_new_1_1_Raw_2.bin) |
| `(1, 1)` | 3 | [0516env_new_1_1_Raw_3.bin](https://github.com/Dingrong123/EmoRI/releases/download/raw-data-v1/0516env_new_1_1_Raw_3.bin) |
| `(2, 1)` | 0 | [0516env_new_2_1_Raw_0.bin](https://github.com/Dingrong123/EmoRI/releases/download/raw-data-v1/0516env_new_2_1_Raw_0.bin) |
| `(2, 1)` | 1 | [0516env_new_2_1_Raw_1.bin](https://github.com/Dingrong123/EmoRI/releases/download/raw-data-v1/0516env_new_2_1_Raw_1.bin) |
| `(2, 1)` | 2 | [0516env_new_2_1_Raw_2.bin](https://github.com/Dingrong123/EmoRI/releases/download/raw-data-v1/0516env_new_2_1_Raw_2.bin) |
| `(2, 1)` | 3 | [0516env_new_2_1_Raw_3.bin](https://github.com/Dingrong123/EmoRI/releases/download/raw-data-v1/0516env_new_2_1_Raw_3.bin) |

Download [SHA256SUMS.txt](https://github.com/Dingrong123/EmoRI/releases/download/raw-data-v1/SHA256SUMS.txt) for the expected SHA-256 hashes, and [raw_capture_manifest.json](https://github.com/Dingrong123/EmoRI/releases/download/raw-data-v1/raw_capture_manifest.json) for filenames, sizes, hashes, and capture metadata.

## Validate the downloads

Compare each downloaded BIN file's SHA-256 hash with its entry in `SHA256SUMS.txt`. From this folder, PowerShell users can calculate the hashes with:

```powershell
Get-FileHash -Algorithm SHA256 -Path .\0516env_new_*_1_Raw_*.bin
```

After installing the repository requirements, run this preflight command from the repository root to validate all eight files:

```bash
python read_6843_accelerated.py --envs 1 2 --exps 1 --nums 0 1 2 3 --frames-per-file 300 --check-inputs
```

If you downloaded only one environment, replace `--envs 1 2` with `--envs 1` or `--envs 2`. The command checks file existence and size/chirp constraints without processing data or writing outputs. It does not check hashes, acquisition metadata, or synchronization quality.

File packing and synchronization assumptions are documented in [the data guide](../README.md#raw-captures) and [the main README](../../README.md#raw-radar-recordings). See [the raw reader guide](../../read_6843_accelerated_README.md) for processing options.
