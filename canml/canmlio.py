"""
canmlio: Enhanced CAN BLF processing toolkit for production use.

Features:
  - Merge multiple DBCs with namespace collision avoidance (optional prefixing).
  - Stream‐decode large BLF files in pandas DataFrame chunks.
  - Full‐file loading with optional uniform timestamp spacing.
  - Signal‐ and message‐level filtering.
  - Automatic injection of expected signals (NaN‐filled if missing).
  - Incremental CSV export and Parquet export.
  - Progress bars via tqdm.

Example:
```python
from canml.canmlio import (
    load_dbc_files,
    load_blf,
    to_csv
)

# Merge powertrain and chassis DBCs, prefix signals to avoid collisions
db = load_dbc_files(["pt.dbc","chassis.dbc"], prefix_signals=True)

# Load BLF, only messages 0x100 & 0x200, inject expected signals
df = load_blf(
    blf_path="log.blf",
    db=db,
    message_ids={0x100,0x200},
    expected_signals=["EngineData_EngineRPM","BrakeStatus_ABSActive"],
    force_uniform_timing=True
)

to_csv(df, "decoded.csv")
```
"""
import logging
from pathlib import Path
from typing import List, Optional, Union, Iterator, Set, Dict, Any

import pandas as pd
import cantools
from cantools.database import Database
from can.io.blf import BLFReader
from tqdm import tqdm

__all__ = [
    "load_dbc_files",
    "iter_blf_chunks",
    "load_blf",
    "to_csv",
    "to_parquet"
]

# Configure module logger
glogger = logging.getLogger(__name__)
if not glogger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    glogger.addHandler(handler)
glogger.setLevel(logging.INFO)


def load_dbc_files(
    dbc_paths: Union[str, List[str]],
    prefix_signals: bool = False
) -> Database:
    """
    Load and merge one or more DBC files into a single Database.
    Optionally prefix signal names with message names to avoid collisions.

    Args:
        dbc_paths: Path or list of paths to DBC files.
        prefix_signals: If True, automatically rename signals to "<MessageName>_<SignalName>".

    Returns:
        A cantools Database with all definitions loaded.

    Raises:
        FileNotFoundError: If any DBC file is missing.
        ValueError: If loading fails.
    """
    paths = [dbc_paths] if isinstance(dbc_paths, str) else dbc_paths
    db = Database()
    for p in paths:
        pth = Path(p)
        if not pth.is_file():
            raise FileNotFoundError(f"DBC file not found: {pth}")
        try:
            glogger.info(f"Loading DBC: {pth}")
            if prefix_signals:
                # load with custom signal naming
                db.add_dbc_file(str(pth))
                # rename signals in messages
                for msg in db.messages:
                    for sig in msg.signals:
                        sig.name = f"{msg.name}_{sig.name}"
            else:
                db.add_dbc_file(str(pth))
        except Exception as e:
            glogger.error(f"Failed to load DBC {pth}: {e}")
            raise ValueError(f"Invalid DBC file: {pth}, {e}")
    return db


def iter_blf_chunks(
    blf_path: str,
    db: Database,
    chunk_size: int = 10000,
    filter_ids: Optional[Set[int]] = None
) -> Iterator[pd.DataFrame]:
    """
    Stream‐decode a BLF file in manageable pandas DataFrame chunks.

    Args:
        blf_path: Path to the BLF log.
        db: cantools Database with message definitions (possibly prefixed).
        chunk_size: Rows per DataFrame chunk.
        filter_ids: If set, only decode messages with these arbitration IDs.

    Yields:
        DataFrame chunks with decoded signals + timestamp column.

    Raises:
        FileNotFoundError: If BLF file not found.
    """
    p = Path(blf_path)
    if not p.is_file():
        raise FileNotFoundError(f"BLF file not found: {p}")
    reader = BLFReader(str(p))
    buffer: List[Dict[str, Any]] = []
    for msg in tqdm(reader, desc=f"Reading {p.name}"):
        if filter_ids and msg.arbitration_id not in filter_ids:
            continue
        try:
            decoded = db.decode_message(msg.arbitration_id, msg.data)
        except (cantools.database.errors.DecodeError, KeyError):
            continue
        rec = decoded.copy()
        rec["timestamp"] = msg.timestamp
        buffer.append(rec)
        if len(buffer) >= chunk_size:
            yield pd.DataFrame(buffer)
            buffer.clear()
    reader.stop()
    if buffer:
        yield pd.DataFrame(buffer)


def load_blf(
    blf_path: str,
    db: Union[Database, str, List[str]],
    message_ids: Optional[Set[int]] = None,
    expected_signals: Optional[List[str]] = None,
    force_uniform_timing: bool = False,
    interval_seconds: float = 0.01
) -> pd.DataFrame:
    """
    Load an entire BLF file into a DataFrame, with optional filters and signal injection.

    Args:
        blf_path: Path to the BLF log.
        db: Database instance or DBC path(s).
        message_ids: Set of arbitration IDs to include (default all).
        expected_signals: List of signal names to ensure exist (fill NaN if missing).
        force_uniform_timing: If True, override timestamps with uniform spacing.
        interval_seconds: Interval for uniform timing.

    Returns:
        A DataFrame with 'timestamp' + decoded signal columns.

    Raises:
        FileNotFoundError: If files missing.
    """
    # Prepare Database
    if isinstance(db, Database):
        database = db
    else:
        database = load_dbc_files(db)
    # Decode in chunks
    dfs: List[pd.DataFrame] = []
    for chunk in iter_blf_chunks(blf_path, database, filter_ids=message_ids):
        dfs.append(chunk)
    if not dfs:
        df = pd.DataFrame()
    else:
        df = pd.concat(dfs, ignore_index=True)
    # Uniform timing
    if force_uniform_timing and not df.empty:
        df["timestamp"] = df.index * interval_seconds
    # Inject expected signals
    if expected_signals:
        for sig in expected_signals:
            if sig not in df.columns:
                glogger.warning(f"Expected signal '{sig}' not found; filling with NaN")
                df[sig] = pd.NA
    # Reorder to put timestamp first
    if "timestamp" in df.columns:
        cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
        df = df[cols]
    return df


def to_csv(
    df_or_iter: Union[pd.DataFrame, Iterator[pd.DataFrame]],
    output_path: str,
    mode: str = "w",
    header: bool = True
) -> None:
    """
    Write DataFrame or iterator of DataFrames to CSV incrementally.

    Args:
        df_or_iter: DataFrame or iterator of DataFrames.
        output_path: Destination CSV file.
        mode: Write mode ('w' or 'a').
        header: Write header for first block.

    Raises:
        ValueError: If write fails.
    """
    p = Path(output_path)
    try:
        if hasattr(df_or_iter, "__iter__") and not isinstance(df_or_iter, pd.DataFrame):
            first = True
            for chunk in df_or_iter:
                chunk.to_csv(
                    p,
                    mode=mode if first else "a",
                    header=header if first else False,
                    index=False
                )
                first = False
        else:
            df_or_iter.to_csv(p, mode=mode, header=header, index=False)
        glogger.info(f"CSV written to {output_path}")
    except Exception as e:
        glogger.error(f"Failed to write CSV {output_path}: {e}")
        raise ValueError(f"Failed to export CSV: {e}")


def to_parquet(
    df: pd.DataFrame,
    output_path: str,
    compression: str = "snappy"
) -> None:
    """
    Write a DataFrame to Parquet.

    Args:
        df: pandas DataFrame.
        output_path: '.parquet' file path.
        compression: Parquet codec.

    Raises:
        ValueError: If write fails.
    """
    p = Path(output_path)
    try:
        df.to_parquet(p, engine="pyarrow", compression=compression)
        glogger.info(f"Parquet written to {output_path}")
    except Exception as e:
        glogger.error(f"Failed to write Parquet {output_path}: {e}")
        raise ValueError(f"Failed to export Parquet: {e}")
