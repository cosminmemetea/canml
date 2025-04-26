"""
canmlio: Production-quality CAN BLF processing toolkit.

Features:
  - Merge multiple DBCs into one database.
  - Stream-decoding of BLF files in fixed-size pandas DataFrame chunks.
  - Full-file loading with optional uniform timestamp spacing.
  - Incremental CSV export and Parquet export.
  - Optional filtering by CAN ID.
  - Progress bars via tqdm.

Example usage:

```python
from canml.canmlio import (
    load_dbc_files,
    iter_blf_chunks,
    load_blf,
    to_csv,
    to_parquet
)

# Merge multiple DBCs
db = load_dbc_files(["pt.dbc", "chassis.dbc"]);

# Stream large BLF in chunks and write to Parquet
for idx, df_chunk in enumerate(
        iter_blf_chunks(
            blf_path="huge.blf",
            db=db,
            chunk_size=50000,
            filter_ids={0x100, 0x200}
        )
):
    to_parquet(df_chunk, f"shard-{idx:03}.parquet")

# Load smaller BLF entirely with uniform timestamps and export to CSV
df = load_blf(
    blf_path="small.blf",
    dbc_paths=["pt.dbc", "chassis.dbc"],
    force_uniform_timing=True,
    interval_seconds=0.01
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

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def load_dbc_files(dbc_paths: Union[str, List[str]]) -> Database:
    """
    Load one or more DBC files into a single cantools Database.

    Args:
        dbc_paths: Path or list of paths to .dbc files.

    Returns:
        Merged cantools Database containing all message definitions.

    Raises:
        FileNotFoundError: If any DBC file does not exist.
        ValueError: If loading a DBC file fails.
    """
    paths = [dbc_paths] if isinstance(dbc_paths, str) else dbc_paths
    db = Database()
    for path in paths:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"DBC file not found: {p}")
        try:
            logger.info(f"Loading DBC: {p}")
            db.add_dbc_file(str(p))
        except Exception as e:
            logger.error(f"Failed to load DBC {p}: {e}")
            raise ValueError(f"Invalid DBC file: {p}, {e}")
    return db


def iter_blf_chunks(
    blf_path: str,
    db: Database,
    chunk_size: int = 10000,
    filter_ids: Optional[Set[int]] = None
) -> Iterator[pd.DataFrame]:
    """
    Stream-decode a BLF file into pandas DataFrame chunks.

    Args:
        blf_path: Path to the BLF log file.
        db: cantools Database with message definitions.
        chunk_size: Maximum number of rows per yielded DataFrame.
        filter_ids: If provided, only decode messages with these arbitration IDs.

    Yields:
        pandas.DataFrame with decoded signals and raw timestamps columns.

    Raises:
        FileNotFoundError: If the BLF file is missing.
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
        record = decoded.copy()
        record["timestamp"] = msg.timestamp
        buffer.append(record)
        if len(buffer) >= chunk_size:
            yield pd.DataFrame(buffer)
            buffer.clear()
    reader.stop()
    if buffer:
        yield pd.DataFrame(buffer)


def load_blf(
    blf_path: str,
    dbc_paths: Union[str, List[str]],
    force_uniform_timing: bool = False,
    interval_seconds: float = 0.01,
    filter_ids: Optional[Set[int]] = None
) -> pd.DataFrame:
    """
    Load a BLF file into a single pandas DataFrame.

    Args:
        blf_path: Path to the BLF log file.
        dbc_paths: Path or list of paths to DBC files.
        force_uniform_timing: If True, replace raw timestamps with uniform spacing.
        interval_seconds: Interval in seconds for uniform timestamp spacing.
        filter_ids: If provided, only decode messages with these IDs.

    Returns:
        pandas.DataFrame with a "timestamp" column and one column per signal.

    Raises:
        FileNotFoundError: If BLF or DBC files are missing.
        ValueError: If DBC loading fails.
    """
    db = load_dbc_files(dbc_paths)
    dfs: List[pd.DataFrame] = []
    for chunk in iter_blf_chunks(blf_path, db, filter_ids=filter_ids):
        dfs.append(chunk)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    if force_uniform_timing:
        df["timestamp"] = df.index * interval_seconds
    cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
    return df[cols]


def to_csv(
    df_or_iter: Union[pd.DataFrame, Iterator[pd.DataFrame]],
    output_path: str,
    mode: str = "w",
    header: bool = True
) -> None:
    """
    Export DataFrame or DataFrame iterator to CSV file.

    Args:
        df_or_iter: A single DataFrame or iterator of DataFrames (for chunks).
        output_path: Destination CSV file path.
        mode: File write mode ("w" or "a").
        header: Whether to write header row (only for first chunk).

    Raises:
        ValueError: If writing fails.
    """
    p = Path(output_path)
    try:
        if hasattr(df_or_iter, "__iter__") and not isinstance(df_or_iter, pd.DataFrame):
            first = True
            for chunk in df_or_iter:
                chunk.to_csv(p, mode=mode if first else "a", header=header if first else False, index=False)
                first = False
        else:
            df_or_iter.to_csv(p, mode=mode, header=header, index=False)
        logger.info(f"CSV written to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write CSV {output_path}: {e}")
        raise ValueError(f"Failed to export CSV: {e}")


def to_parquet(
    df: pd.DataFrame,
    output_path: str,
    compression: str = "snappy"
) -> None:
    """
    Write a DataFrame to Parquet format.

    Args:
        df: pandas DataFrame to write.
        output_path: Destination .parquet file path.
        compression: Compression codec (e.g., "snappy", "gzip").

    Raises:
        ValueError: If writing fails.
    """
    p = Path(output_path)
    try:
        df.to_parquet(p, engine="pyarrow", compression=compression)
        logger.info(f"Parquet written to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write Parquet {output_path}: {e}")
        raise ValueError(f"Failed to export Parquet: {e}")