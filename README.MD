<!-- Top‐level Badges -->
[![PyPI version](https://img.shields.io/pypi/v/canml.svg)](https://pypi.org/project/canml/)
[![Build Status](https://github.com/cosminmemetea/canml/actions/workflows/ci.yml/badge.svg)](https://github.com/cosminmemetea/canml/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/canml/badge/?version=latest)](https://canml.readthedocs.io/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/canml?style=flat-square)](https://pypistats.org/packages/canml)

# canml 

**canml** is a Python toolkit for decoding of CAN bus logs (BLF) using CAN.DBC definitions. It streams large BLF files into pandas DataFrames—either in chunks or all at once—and offers robust CSV and Parquet export, signal‐level filtering, DBC merging, and progress reporting.

---

## Key Features
**Configurable BLF** loading via CanmlConfig (chunk size, progress bars, uniform timing, interpolation, sorting)

**DBC management with caching**: merge one or many .dbc files, auto-detect collisions, optional signal-name prefixing, and LRU caching for repeated loads

**Streaming BLF decode** with iter_blf_chunks()—filter by arbitration ID or by signal name, and emit tidy pandas chunks without blowing up memory

**One-call full-file load** with load_blf()—ID or signal filters, uniform timestamp spacing (with raw backup), missing-signal injection (dtype-safe), linear interpolation, and automatic enum→Categorical conversion

**Rich metadata support**: pull any custom DBC attributes into df.attrs['signal_attributes'] and carry them through to exports

**Incremental CSV & Parquet export**: auto-create directories, write in append mode, and side-dump your signal_attributes JSON alongside your data

**Built-in logging & progress bars** (Python logging + tqdm) to keep you informed without clutter

---

## Installation

```bash
pip install canml
```

**Dependencies**:

- Python ≥ 3.8, < 4.0
- cantools ≥ 39.4.4
- python-can ≥ 4.4.0
- pandas ≥ 2.2.2
- numpy ≥ 1.26.4
- tqdm ≥ 4.0.0
- pyarrow ≥ 11.0.0

## Usage Quickstart

```bash

from canml.canmlio import load_dbc_files, load_blf, to_csv, to_parquet, CanmlConfig

# 1️⃣ Load your DBC(s) (namespace-collision safe)
#    If you have multiple, pass a list; prefix_signals avoids any name clashes.
db = load_dbc_files("vehicle.dbc", prefix_signals=True)

# 2️⃣ Stream-decode the BLF in memory (no CSV yet)
#    Use CanmlConfig to tweak chunk size, uniform-timing, interpolation, etc.
cfg = CanmlConfig(
    chunk_size=5000,        # rows per chunk
    force_uniform_timing=True,
    interval_seconds=0.01,
    progress_bar=True
)
df = load_blf(
    blf_path="drive.blf",
    db=db,
    config=cfg,
    expected_signals=["Engine_RPM", "Brake_Active"]
)

# 3️⃣ Inspect your DataFrame
print(df.head())
#     timestamp  Engine_RPM  Brake_Active  raw_timestamp
# 0        0.00       1200.0           0.0       162523.1
# 1        0.01       1230.0           0.0       162523.2
# …

# 4️⃣ Export to CSV (with metadata JSON)
to_csv(
    df,
    output_path="drive_data.csv",
    metadata_path="drive_data_signals.json"
)

# 5️⃣ Or write Parquet (faster reads/writes + metadata)
to_parquet(
    df,
    output_path="drive_data.parquet",
    metadata_path="drive_data_signals.json"
)


```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes.
4. To update the docu:
 
 ```bash
pip install sphinx sphinx-rtd-theme
```

Please open an issue to discuss major changes before starting work.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- Inspired by `cantools` and `python-can` for CAN bus parsing.
- Built using [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [scikit-learn](https://scikit-learn.org/stable/), and [matplotlib](https://matplotlib.org/) for data manipulation, machine learning, and visualization.
- Special thanks to the Python community for their open-source contributions.

## Contact

For questions or support, please open an issue on the [GitHub repository](https://github.com/cosminmemetea/canml).