
"""
canmlio: Enhanced CAN BLF processing toolkit for production use.
Module: canml/canmlio.py
Features:
  - Merge multiple DBCs with namespace collision avoidance (optional prefixing).
  - Stream‐decode large BLF files in pandas DataFrame chunks.
  - Full‐file loading with optional uniform timestamp spacing.
  - Signal‐ and message‐level filtering.
  - Automatic injection of expected signals (NaN‐filled if missing).
  - Incremental CSV export and Parquet export.
  - Progress bars via tqdm.
"""
import logging
from pathlib import Path
from typing import List, Optional, Union, Iterator, Set, Dict, Any
from collections.abc import Iterable

import numpy as np
import pandas as pd
import cantools
from cantools.database.can import Database as CantoolsDatabase
from can.io.blf import BLFReader
from tqdm import tqdm

__all__ = [
    "load_dbc_files",
    "iter_blf_chunks",
    "load_blf",
    "to_csv",
    "to_parquet",
]

# Module logger: ensure single handler
glogger = logging.getLogger(__name__)
glogger.handlers.clear()  # Prevent duplicate handlers
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
glogger.addHandler(handler)
glogger.setLevel(logging.INFO)


def load_dbc_files(
    dbc_paths: Union[str, List[str]],
    prefix_signals: bool = False
) -> CantoolsDatabase:
    """
    Load and merge one or more DBC files into a single Database.
    Optionally prefix signal names with message names to avoid collisions.

    Args:
        dbc_paths: Path or list of paths to DBC files.
        prefix_signals: If True, rename signals to "<MessageName>_<SignalName>".

    Returns:
        A cantools Database with all definitions loaded.

    Raises:
        FileNotFoundError: If any DBC file is missing.
        ValueError: If loading fails or duplicate names are detected.
    """
    paths = [dbc_paths] if isinstance(dbc_paths, str) else list(dbc_paths or [])
    if not paths:
        raise ValueError("At least one DBC file must be provided")

    db = CantoolsDatabase()
    for p in paths:
        pth = Path(p)
        if not pth.is_file():
            raise FileNotFoundError(f"DBC file not found: {pth}")
        try:
            glogger.info(f"Loading DBC: {pth}")
            db.add_dbc_file(str(pth))
        except cantools.database.errors.ParseError as e:
            glogger.error(f"Invalid DBC format in {pth}: {e}", exc_info=True)
            raise ValueError(f"Failed to parse DBC file {pth}") from e
        except Exception as e:
            glogger.error(f"Failed to load DBC {pth}: {e}", exc_info=True)
            raise ValueError(f"Invalid DBC file {pth}") from e

    # Check for duplicate signal names when not prefixing
    all_signal_names = [sig.name for msg in db.messages for sig in msg.signals]
    if not prefix_signals:
        if len(all_signal_names) != len(set(all_signal_names)):
            raise ValueError(
                "Duplicate signal names found across messages; "
                "use prefix_signals=True to avoid collisions"
            )
    else:
        # Ensure message names are unique before prefixing
        message_names = [m.name for m in db.messages]
        if len(message_names) != len(set(message_names)):
            raise ValueError("Duplicate message names found; cannot prefix uniquely")
        glogger.debug("Renaming signals with message prefixes")
        for msg in db.messages:
            for sig in msg.signals:
                sig.name = f"{msg.name}_{sig.name}"

    return db


def iter_blf_chunks(
    blf_path: str,
    db: CantoolsDatabase,
    chunk_size: int = 10000,
    filter_ids: Optional[Set[int]] = None,
    progress_bar: bool = True
) -> Iterator[pd.DataFrame]:
    """
    Stream-decode a BLF file in pandas DataFrame chunks.

    Args:
        blf_path: Path to the BLF log.
        db: cantools Database with message definitions.
        chunk_size: Rows per DataFrame chunk.
        filter_ids: If set, only decode messages with these arbitration IDs.
        progress_bar: If True, show a tqdm progress bar.

    Yields:
        DataFrame chunks with decoded signals + timestamp column.

    Raises:
        FileNotFoundError: If BLF file not found.
        ValueError: If chunk_size is invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    p = Path(blf_path)
    if not p.is_file():
        raise FileNotFoundError(f"BLF file not found: {p}")

    try:
        reader = BLFReader(str(p))
        buffer: List[Dict[str, Any]] = []
        iterator = tqdm(reader, desc=f"Reading {p.name}", mininterval=0.5) if progress_bar else reader
        for msg in iterator:
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

        if buffer:
            yield pd.DataFrame(buffer)
    finally:
        try:
            reader.stop()
        except (NameError, Exception) as e:
            glogger.warning(f"Failed to close BLF reader: {e}", exc_info=True)


def load_blf(
    blf_path: str,
    db: Union[CantoolsDatabase, str, List[str]],
    message_ids: Optional[Set[int]] = None,
    expected_signals: Optional[List[str]] = None,
    force_uniform_timing: bool = False,
    interval_seconds: float = 0.01,
    dtype_map: Optional[Dict[str, Union[str, np.dtype]]] = None,
    sort_timestamps: bool = False
) -> pd.DataFrame:
    """
    Load an entire BLF file into a DataFrame, with optional filters, signal injection,
    and dtype control for injected signals.

    Notes:
      - If force_uniform_timing=True, the original timestamps are saved in "raw_timestamp".
      - Concatenates chunks iteratively to reduce memory usage.

    Args:
        blf_path: Path to the BLF log.
        db: Database instance or DBC path(s).
        message_ids: Set of arbitration IDs to include (default all).
        expected_signals: List of signal names to ensure exist.
        force_uniform_timing: If True, override timestamps with uniform spacing.
        interval_seconds: Interval for uniform timing.
        dtype_map: Optional mapping from signal name to dtype for injected columns.
        sort_timestamps: If True, sort by timestamp before processing.

    Returns:
        A DataFrame with 'timestamp' + decoded signal columns.

    Raises:
        FileNotFoundError: If files missing.
        ValueError: For invalid parameters or processing errors.
    """
    if force_uniform_timing and interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if message_ids is not None and not message_ids:
        glogger.warning("Empty message_ids provided; no messages will be decoded")
    if expected_signals and len(expected_signals) != len(set(expected_signals)):
        raise ValueError("Duplicate names found in expected_signals")
    if dtype_map:
        for sig, dt in dtype_map.items():
            try:
                pd.Series(dtype=dt)
            except Exception as e:
                raise ValueError(f"Invalid dtype for signal {sig}: {dt}") from e

    database = db if isinstance(db, CantoolsDatabase) else load_dbc_files(db)
    df = None
    try:
        for chunk in iter_blf_chunks(blf_path, database, chunk_size=10000, filter_ids=message_ids):
            if df is None:
                df = chunk
            else:
                df = pd.concat([df, chunk], ignore_index=True)
    except Exception as e:
        glogger.error("Failed to process BLF chunks", exc_info=True)
        raise ValueError(f"Failed to process BLF data: {e}") from e

    if df is None or df.empty:
        cols = ["timestamp"] + (expected_signals or [])
        df = pd.DataFrame(columns=cols)
        df["timestamp"] = pd.Series(dtype=float)
        glogger.warning(f"No data decoded from {blf_path}; returning empty DataFrame")

    if sort_timestamps and not df.empty and "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    if force_uniform_timing and not df.empty:
        df["raw_timestamp"] = df["timestamp"]
        df["timestamp"] = pd.Series(range(len(df))) * interval_seconds

    dtype_map = dtype_map or {}
    if expected_signals:
        for sig in expected_signals:
            if sig not in df.columns:
                dt = dtype_map.get(sig, float)
                glogger.warning(f"Expected signal '{sig}' not found; filling with NaN as {dt}")
                df[sig] = pd.Series(dtype=dt)

    if "timestamp" in df.columns:
        cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
        df = df[cols]

    return df


def to_csv(
    df_or_iter: Union[pd.DataFrame, Iterable[pd.DataFrame]],
    output_path: str,
    mode: str = "w",
    header: bool = True,
    pandas_kwargs: Optional[Dict[str, Any]] = None,
    columns: Optional[List[str]] = None
) -> None:
    """
    Write a DataFrame or iterable of DataFrames to CSV incrementally,
    enforcing a canonical column order if provided.

    Args:
        df_or_iter: DataFrame or iterable of DataFrames.
        output_path: Destination CSV file.
        mode: Write mode ('w' or 'a').
        header: Write header for first block.
        pandas_kwargs: Additional kwargs for pandas.to_csv.
        columns: Optional canonical column list; each chunk will be reindexed to this.

    Raises:
        ValueError: If columns are invalid.
        TypeError: If input is not a DataFrame or iterable.
    """
    p = Path(output_path)
    pandas_kwargs = pandas_kwargs or {}

    if columns:
        if not all(isinstance(c, str) for c in columns):
            raise ValueError("All column names must be strings")
        if len(columns) != len(set(columns)):
            raise ValueError("Duplicate column names provided")

    if isinstance(df_or_iter, pd.DataFrame):
        df = df_or_iter
        if columns:
            missing = set(columns) - set(df.columns)
            if missing:
                glogger.warning(f"Columns {missing} not found in data; filling with NaN")
            df = df.reindex(columns=columns)
        df.to_csv(p, mode=mode, header=header, index=False, **pandas_kwargs)
        glogger.info(f"CSV written to {output_path}")
        return

    if not isinstance(df_or_iter, Iterable):
        raise TypeError("Input must be a DataFrame or iterable of DataFrames")

    first = True
    wrote_data = False
    for chunk in df_or_iter:
        if columns:
            missing = set(columns) - set(chunk.columns) if first else set()
            if missing:
                glogger.warning(f"Columns {missing} not found in first chunk; filling with NaN")
            chunk = chunk.reindex(columns=columns)
        chunk.to_csv(
            p,
            mode=mode if first else "a",
            header=header if first else False,
            index=False,
            **pandas_kwargs
        )
        wrote_data = True
        first = False

    if not wrote_data:
        glogger.warning(f"No data to write to {output_path}; creating empty file")
        empty = pd.DataFrame(columns=columns) if columns else pd.DataFrame()
        empty.to_csv(p, mode=mode, header=header, index=False, **pandas_kwargs)

    glogger.info(f"CSV written to {output_path}")


def to_parquet(
    df: pd.DataFrame,
    output_path: str,
    compression: str = "snappy",
    pandas_kwargs: Optional[Dict[str, Any]] = None
) -> None:
    """
    Write a DataFrame to Parquet.

    Args:
        df: pandas DataFrame.
        output_path: '.parquet' file path.
        compression: Parquet codec.
        pandas_kwargs: Additional kwargs for pandas.to_parquet.

    Raises:
        ValueError: If write fails.
    """
    p = Path(output_path)
    pandas_kwargs = pandas_kwargs or {}
    try:
        df.to_parquet(p, engine="pyarrow", compression=compression, **pandas_kwargs)
        glogger.info(f"Parquet written to {output_path}")
    except Exception as e:
        glogger.error(f"Failed to write Parquet {output_path}: {e}", exc_info=True)
        raise ValueError(f"Failed to export Parquet: {e}") from e