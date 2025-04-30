"""
canmlio: Enhanced CAN BLF processing toolkit for production use.
Module: canml/canmlio.py

Features:
  - Merge multiple DBCs with namespace collision avoidance and caching.
  - Stream-decode large BLF files into pandas DataFrame chunks.
  - Full-file loading with uniform timestamp spacing, interpolation, and sorting.
  - Signal/message filtering by CAN ID or signal name.
  - Automatic injection of missing signals with dtype preservation.
  - Incremental CSV & Parquet export with side-car metadata JSON.
  - Generic support for enums and custom signal attributes.
  - Configurable progress bars via tqdm.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from collections import Counter

import numpy as np
import pandas as pd
import cantools
from cantools.database.can import Database as CantoolsDatabase
from can.io.blf import BLFReader
from tqdm import tqdm

__all__ = [
    "CanmlConfig",
    "load_dbc_files",
    "iter_blf_chunks",
    "load_blf",
    "to_csv",
    "to_parquet",
]

# ------------------------------------------------------------------------------
# Module logger: single StreamHandler with timestamped format
# ------------------------------------------------------------------------------
glogger = logging.getLogger(__name__)
glogger.handlers.clear()
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
glogger.addHandler(_handler)
glogger.setLevel(logging.INFO)

T = Any

# ------------------------------------------------------------------------------
# Configuration dataclass
# ------------------------------------------------------------------------------
@dataclass
class CanmlConfig:
    """
    Configuration for BLF processing.

    Attributes:
        chunk_size:       Number of messages per DataFrame chunk.
        progress_bar:     Show tqdm progress bar if True.
        dtype_map:        Mapping from signal name to pandas dtype.
        sort_timestamps:  Sort final DataFrame by timestamp.
        force_uniform_timing: Override timestamps with uniform spacing.
        interval_seconds: Interval between timestamps for uniform timing.
        interpolate_missing: Interpolate missing signal values if True.
    """
    chunk_size: int = 10000
    progress_bar: bool = True
    dtype_map: Optional[Dict[str, Any]] = None
    sort_timestamps: bool = False
    force_uniform_timing: bool = False
    interval_seconds: float = 0.01
    interpolate_missing: bool = False

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")

# ------------------------------------------------------------------------------
# DBC loading with caching and collision detection
# ------------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _load_dbc_files_cached(
    dbc_paths: Union[str, Tuple[str, ...]], prefix_signals: bool
) -> CantoolsDatabase:
    """
    Internal helper to load and merge DBC files, cached for performance.
    """
    # Normalize to list of paths
    paths = [dbc_paths] if isinstance(dbc_paths, str) else list(dbc_paths)
    if not paths:
        raise ValueError("At least one DBC file must be provided")

    db = CantoolsDatabase()
    for p in paths:
        pth = Path(p)
        if pth.suffix.lower() != ".dbc":
            raise ValueError(f"File {pth} is not a .dbc file")
        if not pth.is_file():
            raise FileNotFoundError(f"DBC file not found: {pth}")
        glogger.debug(f"Loading DBC: {pth}")
        try:
            db.add_dbc_file(str(pth))
        except cantools.database.errors.ParseError as e:
            raise ValueError(f"Invalid DBC format in {pth}: {e}") from e
        except Exception as e:
            raise ValueError(f"Invalid DBC file {pth}: {e}") from e

    # Collect all signal names to detect duplicates
    all_names = [sig.name for msg in db.messages for sig in msg.signals]
    if not prefix_signals:
        dupes = [n for n, cnt in Counter(all_names).items() if cnt > 1]
        if dupes:
            raise ValueError(
                f"Duplicate signal names: {sorted(dupes)}; use prefix_signals=True"
            )
    else:
        # Ensure message names are unique before prefixing
        msg_names = [msg.name for msg in db.messages]
        if len(msg_names) != len(set(msg_names)):
            raise ValueError("Duplicate message names found; cannot prefix uniquely")
        for msg in db.messages:
            for sig in msg.signals:
                sig.name = f"{msg.name}_{sig.name}"

    return db

def load_dbc_files(
    dbc_paths: Union[str, List[str]], prefix_signals: bool = False
) -> CantoolsDatabase:
    """
    Public API: Load and merge DBC file(s) into a CantoolsDatabase.

    Args:
        dbc_paths: Path or list of paths to .dbc files.
        prefix_signals: If True, prefix each signal name with its message name.

    Returns:
        A CantoolsDatabase containing all loaded definitions.
    """
    key = tuple(dbc_paths) if isinstance(dbc_paths, list) else dbc_paths
    return _load_dbc_files_cached(key, prefix_signals)

# ------------------------------------------------------------------------------
# BLFReader context manager
# ------------------------------------------------------------------------------
@contextmanager
def blf_reader(path: str) -> Iterator[BLFReader]:
    """
    Ensure BLFReader.stop() is always called to free resources.
    """
    reader = BLFReader(str(path))
    try:
        yield reader
    finally:
        try:
            reader.stop()
        except Exception:
            glogger.debug("Error closing BLF reader", exc_info=True)

# ------------------------------------------------------------------------------
# Streamed BLF decoding into pandas DataFrame chunks
# ------------------------------------------------------------------------------
def iter_blf_chunks(
    blf_path: str,
    db: CantoolsDatabase,
    config: CanmlConfig,
    filter_ids: Optional[Set[int]] = None,
    filter_signals: Optional[Iterable[Any]] = None,
) -> Iterator[pd.DataFrame]:
    """
    Stream-decode a BLF file into DataFrame chunks.

    Args:
        blf_path: Path to .blf log.
        db: Loaded CantoolsDatabase.
        config: CanmlConfig for chunk_size & progress_bar.
        filter_ids: Optional set of arbitration IDs to include.
        filter_signals: Optional iterable of signal identifiers to include.

    Yields:
        pandas.DataFrame chunks containing decoded signals + 'timestamp'.
    """
    p = Path(blf_path)
    if p.suffix.lower() != ".blf" or not p.is_file():
        raise FileNotFoundError(f"Valid BLF file not found: {p}")

    # Build a safe set of signal names (strings), handling unhashable items
    if filter_signals is not None:
        sig_set: Set[str] = set()
        for s in filter_signals:
            try:
                sig_set.add(str(s))
            except Exception:
                # fallback: skip problematic entries
                continue
    else:
        sig_set = None

    buffer: List[Dict[str, Any]] = []
    with blf_reader(blf_path) as reader:
        iterator = tqdm(reader, desc=p.name) if config.progress_bar else reader
        for msg in iterator:
            # Filter by arbitration ID if requested
            if filter_ids and msg.arbitration_id not in filter_ids:
                continue
            try:
                rec = db.decode_message(msg.arbitration_id, msg.data)
            except Exception:
                # Skip messages that cannot be decoded
                continue
            # Filter by signal names if requested
            if sig_set is not None:
                rec = {k: v for k, v in rec.items() if k in sig_set}
            if rec:
                rec["timestamp"] = msg.timestamp
                buffer.append(rec)
            # Yield a chunk when buffer is full
            if len(buffer) >= config.chunk_size:
                yield pd.DataFrame(buffer)
                buffer.clear()
        # Yield any remaining records
        if buffer:
            yield pd.DataFrame(buffer)

# ------------------------------------------------------------------------------
# High-level BLF→DataFrame loader with full-featured options
# ------------------------------------------------------------------------------
def load_blf(
    blf_path: str,
    db: Union[CantoolsDatabase, str, List[str]],
    config: Optional[CanmlConfig] = None,
    message_ids: Optional[Set[int]] = None,
    expected_signals: Optional[Iterable[Any]] = None,
) -> pd.DataFrame:
    """
    Load a BLF log into a pandas DataFrame with decoding, filtering,
    timing normalization, missing-signal injection, and metadata.

    Args:
        blf_path: Path to .blf log file.
        db: CantoolsDatabase instance or path(s) to DBC file(s).
        config: CanmlConfig instance (optional).
        message_ids: Set of arbitration IDs to include (None = all).
        expected_signals: Iterable of signals to include (None = all).

    Returns:
        pandas.DataFrame with columns ['timestamp', ...signals].
    """
    config = config or CanmlConfig()

    # 1) Normalize expected_signals to list of strings and detect duplicates
    if expected_signals is not None:
        exp_list: List[str] = []
        seen: Set[str] = set()
        for s in expected_signals:
            name = str(s)
            if name in seen:
                raise ValueError("Duplicate names in expected_signals")
            seen.add(name)
            exp_list.append(name)
    else:
        exp_list = None

    # 2) Load or reuse DBC database
    dbobj = db if isinstance(db, CantoolsDatabase) else load_dbc_files(db)

    # 3) Warning for explicit empty message_ids
    if message_ids is not None and not message_ids:
        glogger.warning("Empty message_ids provided; no messages will be decoded")

    # 4) Determine the full set of expected signals
    all_sigs = [s.name for msg in dbobj.messages for s in msg.signals]
    expected = exp_list if exp_list is not None else all_sigs

    # 5) Validate dtype_map entries
    dtype_map = config.dtype_map or {}
    for sig in dtype_map:
        if str(sig) not in expected:
            raise ValueError(f"dtype_map contains unknown signal: {sig}")

    # 6) Decode BLF into chunks
    try:
        chunks = list(iter_blf_chunks(
            blf_path,
            dbobj,
            config,
            message_ids,
            expected  # pass list of str names
        ))
    except FileNotFoundError:
        raise
    except Exception as e:
        glogger.error("Failed to process BLF chunks", exc_info=True)
        raise ValueError(f"Failed to process BLF data: {e}") from e

    # 7) Build the DataFrame (or an empty one if no chunks)
    if not chunks:
        glogger.warning(f"No data decoded from {blf_path}; returning empty DataFrame")
        df = pd.DataFrame({
            "timestamp": pd.Series(dtype=float),
            **{sig: pd.Series(dtype=dtype_map.get(sig, float)) for sig in expected}
        })
    else:
        df = pd.concat(chunks, ignore_index=True)

    # 8) Retain only timestamp + expected columns
    keep = [c for c in ["timestamp"] + expected if c in df.columns]
    df = df[keep]

    # 9) Optional timestamp sorting
    if config.sort_timestamps:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # 10) Optional uniform timing
    if config.force_uniform_timing:
        df["raw_timestamp"] = df["timestamp"]
        df["timestamp"] = np.arange(len(df)) * config.interval_seconds

    # 11) Inject missing signals with correct dtype or interpolation
    for sig in expected:
        if sig not in df.columns:
            dt = np.dtype(dtype_map.get(sig, float))
            if config.interpolate_missing and sig in all_sigs:
                series = pd.Series(np.nan, index=df.index, dtype=dt)
                df[sig] = series.interpolate(method="linear", limit_direction="both")
            elif np.issubdtype(dt, np.integer):
                df[sig] = np.zeros(len(df), dtype=dt)
            else:
                df[sig] = pd.Series(np.nan, index=df.index, dtype=dt)

    # 12) Collect custom attributes and convert enums
    df.attrs["signal_attributes"] = {
        s.name: getattr(s, "attributes", {})
        for msg in dbobj.messages for s in msg.signals
        if s.name in df.columns
    }
    for msg in dbobj.messages:
        for s in msg.signals:
            if s.name in df.columns and getattr(s, "choices", None):
                df[s.name] = pd.Categorical(
                    df[s.name].map(s.choices),
                    categories=list(s.choices.values())
                )

    # 13) Ensure timestamp is the first column
    return df[["timestamp"] + [c for c in df.columns if c != "timestamp"]]

# ------------------------------------------------------------------------------
# Incremental CSV export with metadata
# ------------------------------------------------------------------------------
def to_csv(
    df_or_iter: Union[pd.DataFrame, Iterable[pd.DataFrame]],
    output_path: str,
    mode: str = "w",
    header: bool = True,
    pandas_kwargs: Optional[Dict[str, Any]] = None,
    columns: Optional[List[str]] = None,
    metadata_path: Optional[str] = None,
) -> None:
    """
    Write DataFrame or iterable of DataFrames to CSV, exporting metadata JSON.

    Args:
        df_or_iter: Single DataFrame or iterable of chunks.
        output_path: CSV file path.
        mode: 'w' or 'a'.
        header: Write header row if True.
        pandas_kwargs: Extra pandas.to_csv kwargs.
        columns: Optional list of columns (subset/order).
        metadata_path: Path to side-car JSON with df.attrs["signal_attributes"].
    """
    import json

    p = Path(output_path)
    pandas_kwargs = pandas_kwargs or {}

    # Validate columns list
    if columns and len(columns) != len(set(columns)):
        raise ValueError("Duplicate columns specified")

    # Ensure output directory exists
    p.parent.mkdir(parents=True, exist_ok=True)

    def _write(df: pd.DataFrame, write_mode: str, write_header: bool, write_meta: bool):
        # Write CSV chunk
        df.to_csv(p, mode=write_mode, header=write_header,
                  index=False, columns=columns, **pandas_kwargs)
        # Write metadata JSON once (first chunk or single-DF)
        if metadata_path and write_meta:
            m = Path(metadata_path)
            m.parent.mkdir(parents=True, exist_ok=True)
            sig_attrs = df.attrs.get("signal_attributes") or {c: {} for c in df.columns}
            m.write_text(json.dumps(sig_attrs))

    # Single DataFrame vs. chunks
    if isinstance(df_or_iter, pd.DataFrame):
        _write(df_or_iter, mode, header, True)
    else:
        first = True
        for chunk in df_or_iter:
            _write(chunk, mode if first else "a",
                   header if first else False, first)
            first = False

    glogger.info(f"CSV written to {output_path}")

# ------------------------------------------------------------------------------
# Parquet export with metadata
# ------------------------------------------------------------------------------
def to_parquet(
    df: pd.DataFrame,
    output_path: str,
    compression: str = "snappy",
    pandas_kwargs: Optional[Dict[str, Any]] = None,
    metadata_path: Optional[str] = None,
) -> None:
    """
    Write DataFrame to Parquet with optional metadata export.

    Args:
        df: DataFrame to write.
        output_path: Path to .parquet file.
        compression: Parquet codec (snappy, gzip, zstd).
        pandas_kwargs: Extra pandas.to_parquet kwargs.
        metadata_path: Path to side-car JSON with df.attrs["signal_attributes"].
    """
    import json

    p = Path(output_path)
    pandas_kwargs = pandas_kwargs or {}

    # Ensure directory exists
    p.parent.mkdir(parents=True, exist_ok=True)

    # Write Parquet
    try:
        df.to_parquet(p, engine="pyarrow", compression=compression, **pandas_kwargs)
    except Exception as e:
        glogger.error(f"Failed to write Parquet {p}: {e}", exc_info=True)
        raise ValueError(f"Failed to export Parquet: {e}") from e

    # Write metadata JSON if requested
    if metadata_path:
        m = Path(metadata_path)
        m.parent.mkdir(parents=True, exist_ok=True)
        sig_attrs = df.attrs.get("signal_attributes") or {c: {} for c in df.columns}
        m.write_text(json.dumps(sig_attrs))
        glogger.info(f"Metadata written to {m}")

    glogger.info(f"Parquet written to {p}")
