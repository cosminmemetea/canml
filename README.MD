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

- **Merge DBC files**  
  Load one or multiple `.dbc` files into a single `cantools` Database, with optional signal‐name prefixing to avoid collisions.

- **Chunked streaming**  
  Decode arbitrarily large BLF logs in fixed‐size pandas DataFrame chunks, with an optional progress bar.

- **Full‐file mode**  
  Load an entire BLF into one DataFrame, with optional message‐ID filtering, uniform timestamp spacing, and injection of expected signals (NaN‐filled if missing).

- **Flexible export**  
  Incremental CSV export (`to_csv`) or single‐shot Parquet export (`to_parquet`).

- **Signal & message filtering**  
  Only decode specified CAN IDs or automatically add missing signals for downstream consistency.

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
    from canml.canmlio import (
        load_dbc_files,
        iter_blf_chunks,
        load_blf,
        to_csv,
        to_parquet
    )

    # 1. Merge multiple DBCs (with optional signal‐prefixing)
    db = load_dbc_files(["powertrain.dbc", "chassis.dbc"], prefix_signals=True)

    # 2. Stream‐decode a large BLF in 50k‐row chunks, filtering only IDs 0x100 & 0x200
    for idx, df_chunk in enumerate(iter_blf_chunks(
            blf_path="vehicle.blf",
            db=db,
            chunk_size=50_000,
            filter_ids={0x100, 0x200}
        )):
        to_parquet(df_chunk, f"shard-{idx:03}.parquet")

    # 3. Load a smaller BLF fully, enforce uniform 10 ms timestamps,
    #    and ensure specific signals (even if missing) appear as NaN
    df_full = load_blf(
        blf_path="session0.blf",
        db=db,
        message_ids={0x100, 0x200},
        expected_signals=["EngineData_EngineRPM", "BrakeStatus_ABSActive"],
        force_uniform_timing=True,
        interval_seconds=0.01
    )

    # 4. Export to CSV
    to_csv(df_full, "session0_decoded.csv")

```

## API Reference

**load_dbc_files(dbc_paths, prefix_signals=False) → Database**
      
    Load one or more DBC files into a merged cantools Database.

    dbc_paths – single path or list of .dbc file paths

    prefix_signals – if True, renames signals to "<MessageName>_<SignalName>".

**iter_blf_chunks(blf_path, db, chunk_size=10000, filter_ids=None) → Iterator[DataFrame]**

    Stream‐decode a BLF into DataFrame chunks.

    blf_path – path to .blf file

    db – Database from load_dbc_files

    chunk_size – max rows per DataFrame

    filter_ids – set of CAN IDs to include

**load_blf(blf_path, db, message_ids=None, expected_signals=None, force_uniform_timing=False, interval_seconds=0.01) → DataFrame**

    Decode an entire BLF into one DataFrame.

    message_ids – restrict to given CAN IDs

    expected_signals – list of columns to inject as NaN if missing

    force_uniform_timing – override raw timestamps with uniform spacing

**to_csv(df_or_iter, output_path, mode='w', header=True) → None**

    Write DataFrame or iterator of DataFrames to CSV.

    to_parquet(df, output_path, compression='snappy') → None
    
    Write DataFrame to Parquet (pyarrow engine).


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