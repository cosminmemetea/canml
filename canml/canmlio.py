"""
canmlio: Enhanced CAN BLF processing toolkit for production use.

This module provides end-to-end functionality for decoding CAN bus logs in BLF
format into pandas DataFrames, handling DBC file loading and merging,
streaming large logs, full-file loading with filtering, timing alignment,
missing-signal injection, and exporting to CSV or Parquet with accompanying
metadata. It also supports enums and custom signal attributes, all configurable
via a single `CanmlConfig` object.

Dependencies:
  - numpy
  - pandas
  - cantools
  - python-can
  - tqdm
  - pyarrow (for Parquet export)

Example:
    from canml.canmlio import load_dbc_files, load_blf, to_csv, CanmlConfig

    # 1️⃣ Load DBC
    db = load_dbc_files("vehicle.dbc", prefix_signals=True)

    # 2️⃣ Configure BLF load
    cfg = CanmlConfig(
        chunk_size=5000,
        progress_bar=True,
        force_uniform_timing=True,
        interval_seconds=0.02,
        interpolate_missing=True,
        dtype_map={"Engine_RPM": "int32"}
    )

    # 3️⃣ Load BLF file
    df = load_blf(
        blf_path="drive.blf",
        db=db,
        config=cfg,
        message_ids={0x100, 0x200},
        expected_signals=["Engine_RPM", "Brake_Active"]
    )

    # 4️⃣ Export
    to_csv(df, "drive.csv", metadata_path="drive_meta.json")
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

# ----------------------------------------------------------------------------
# Logger setup
# ----------------------------------------------------------------------------

glogger = logging.getLogger(__name__)
glogger.handlers.clear()
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
glogger.addHandler(_handler)
glogger.setLevel(logging.INFO)

T = Any

# ----------------------------------------------------------------------------
# Configuration dataclass
# ----------------------------------------------------------------------------
@dataclass
class CanmlConfig:
    """
    Configuration options for BLF processing.

    Args:
        chunk_size (int): Number of messages per chunk when streaming. Defaults to 10000.
            Example: chunk_size=5000 for smaller, more frequent chunks.
        progress_bar (bool): Show a tqdm progress bar if True. Defaults to True.
        dtype_map (Optional[Dict[str, Any]]): Map signal names to pandas dtypes.
            Example: dtype_map={"Speed": "float32"} ensures Speed column is float32.
        sort_timestamps (bool): Sort final DataFrame by timestamp. Defaults to False.
        force_uniform_timing (bool): Override timestamps with uniform spacing. Defaults to False.
        interval_seconds (float): Spacing interval in seconds for uniform timing. Defaults to 0.01.
        interpolate_missing (bool): Interpolate missing signal values if True. Defaults to False.

    Raises:
        ValueError: If chunk_size <= 0 or interval_seconds <= 0.
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

# ----------------------------------------------------------------------------
# DBC loading and merging
# ----------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _load_dbc_files_cached(
    dbc_paths: Union[str, Tuple[str, ...]], prefix_signals: bool
) -> CantoolsDatabase:
    """
    Internal: Load and merge .dbc files into a single CantoolsDatabase.

    Args:
        dbc_paths (str or tuple): Path(s) to .dbc file(s).
        prefix_signals (bool): If True, prefix each signal with its message name
            or message_name_frame_id on duplicate messages.

    Returns:
        CantoolsDatabase: Loaded database.

    Raises:
        FileNotFoundError: If any .dbc path is missing.
        ValueError: On invalid extension, parse errors, or duplicate signal names.
    """
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

    names = [sig.name for msg in db.messages for sig in msg.signals]
    if not prefix_signals:
        dup = [n for n, c in Counter(names).items() if c > 1]
        if dup:
            raise ValueError(f"Duplicate signal names: {sorted(dup)}; use prefix_signals=True")
    else:
        msg_names = [m.name for m in db.messages]
        dup_msg = [n for n, c in Counter(msg_names).items() if c > 1]
        if dup_msg:
            glogger.warning(
                f"Duplicate message names {sorted(dup_msg)}; "
                "using <name>_<frame_id> prefix"
            )
            for msg in db.messages:
                for sig in msg.signals:
                    sig.name = f"{msg.name}_{msg.frame_id}_{sig.name}"
        else:
            for msg in db.messages:
                for sig in msg.signals:
                    sig.name = f"{msg.name}_{sig.name}"
    return db

def load_dbc_files(
    dbc_paths: Union[str, List[str]], prefix_signals: bool = False
) -> CantoolsDatabase:
    """
    Load and merge one or more DBC files, caching results.

    Args:
        dbc_paths (str or list): Path or list of DBC file paths.
        prefix_signals (bool): If True, prefix signals with message names.

    Returns:
        CantoolsDatabase: Merged database.

    Example:
        db = load_dbc_files("vehicle.dbc", prefix_signals=True)
    """
    key = tuple(dbc_paths) if isinstance(dbc_paths, list) else dbc_paths
    return _load_dbc_files_cached(key, prefix_signals)

# ----------------------------------------------------------------------------
# BLFReader context manager
# ----------------------------------------------------------------------------
@contextmanager
def blf_reader(path: str) -> Iterator[BLFReader]:
    """
    Context manager that ensures BLFReader.stop() is called.

    Args:
        path (str): Path to BLF file.

    Yields:
        BLFReader: Active reader.
    """
    reader = BLFReader(str(path))
    try:
        yield reader
    finally:
        try:
            reader.stop()
        except Exception:
            glogger.debug("Error closing BLF reader", exc_info=True)

# ----------------------------------------------------------------------------
# Stream-decode BLF in chunks
# ----------------------------------------------------------------------------

def iter_blf_chunks(
    blf_path: str,
    db: CantoolsDatabase,
    config: CanmlConfig,
    filter_ids: Optional[Set[int]] = None,
    filter_signals: Optional[Iterable[Any]] = None,
) -> Iterator[pd.DataFrame]:
    """
    Stream-decode a BLF file into pandas DataFrame chunks.

    Args:
        blf_path (str): Path to BLF file.
        db (CantoolsDatabase): Database for message definitions.
        config (CanmlConfig): Chunk size, progress bar, etc.
        filter_ids (set[int], optional): Only decode these arbitration IDs.
        filter_signals (iterable, optional): Only include these signal names.

    Yields:
        pd.DataFrame: Decoded signals with a 'timestamp' column.

    Example:
        for chunk in iter_blf_chunks("drive.blf", db, cfg, filter_ids={0x123}):
            print(chunk.head())
    """
    p = Path(blf_path)
    if p.suffix.lower() != ".blf" or not p.is_file():
        raise FileNotFoundError(f"Valid BLF file not found: {p}")

    sig_set: Optional[Set[str]] = None
    if filter_signals is not None:
        sig_set = set()
        for s in filter_signals:
            try:
                sig_set.add(str(s))
            except Exception:
                continue

    buffer: List[Dict[str, Any]] = []
    with blf_reader(blf_path) as reader:
        it = tqdm(reader, desc=p.name) if config.progress_bar else reader
        for msg in it:
            if filter_ids and msg.arbitration_id not in filter_ids:
                continue
            try:
                rec = db.decode_message(msg.arbitration_id, msg.data)
            except Exception:
                continue
            if sig_set is not None:
                rec = {k: v for k, v in rec.items() if k in sig_set}
            if rec:
                rec["timestamp"] = msg.timestamp
                buffer.append(rec)
            if len(buffer) >= config.chunk_size:
                yield pd.DataFrame(buffer)
                buffer.clear()
        if buffer:
            yield pd.DataFrame(buffer)

# ----------------------------------------------------------------------------
# Full-file load with filtering, timing, injection, metadata
# ----------------------------------------------------------------------------

def load_blf(
    blf_path: str,
    db: Union[CantoolsDatabase, str, List[str]],
    config: Optional[CanmlConfig] = None,
    message_ids: Optional[Set[int]] = None,
    expected_signals: Optional[Iterable[Any]] = None,
) -> pd.DataFrame:
    """
    Load a BLF log into a pandas DataFrame, with full-featured options.

    Args:
        blf_path (str): Path to BLF file.
        db (CantoolsDatabase or str/list): Database instance or DBC path(s).
        config (CanmlConfig, optional): Processing options.
        message_ids (set[int], optional): Filter by CAN IDs. Example: {0x123, 0x200}.
        expected_signals (iterable, optional): Signals to include. Example: ["Engine_RPM"].

    Returns:
        pd.DataFrame: Columns ['timestamp', ...signals], dtype-safe, enums as Categorical.

    Example:
        df = load_blf(
            blf_path="drive.blf",
            db="vehicle.dbc",
            config=cfg,
            expected_signals=["Speed", NameSignalValue(...)],
        )
    """
    config = config or CanmlConfig()

    # Normalize and dedupe expected_signals
    exp_list: Optional[List[str]] = None
    if expected_signals is not None:
        seen: Set[str] = set()
        exp_list = []
        for s in expected_signals:
            nm = str(s)
            if nm in seen:
                raise ValueError("Duplicate names in expected_signals")
            seen.add(nm)
            exp_list.append(nm)

    # Load database
    dbobj = db if isinstance(db, CantoolsDatabase) else load_dbc_files(db)
    if message_ids is not None and not message_ids:
        glogger.warning("Empty message_ids provided; no messages decoded")

    # Determine expected signals
    all_sigs = [sig.name for msg in dbobj.messages for sig in msg.signals]
    expected = exp_list if exp_list is not None else all_sigs

    # Validate dtype_map
    dtype_map = config.dtype_map or {}
    for sig in dtype_map:
        if str(sig) not in expected:
            raise ValueError(f"dtype_map contains unknown signal: {sig}")

    # Decode chunks
    try:
        chunks = list(iter_blf_chunks(
            blf_path, dbobj, config, message_ids, expected
        ))
    except FileNotFoundError:
        raise
    except Exception as e:
        glogger.error("Failed to process BLF chunks", exc_info=True)
        raise ValueError(f"Failed to process BLF data: {e}") from e

    # Assemble DataFrame
    if not chunks:
        glogger.warning(f"No data decoded from {blf_path}; returning empty DataFrame")
        df = pd.DataFrame({
            "timestamp": pd.Series(dtype=float),
            **{sig: pd.Series(dtype=dtype_map.get(sig, float)) for sig in expected}
        })
    else:
        df = pd.concat(chunks, ignore_index=True)

    # Filter columns
    keep = [c for c in ["timestamp"] + expected if c in df.columns]
    df = df[keep]

    # Sort timestamps
    if config.sort_timestamps:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Uniform timing
    if config.force_uniform_timing:
        df["raw_timestamp"] = df["timestamp"]
        df["timestamp"] = np.arange(len(df)) * config.interval_seconds

    # Inject missing signals
    for sig in expected:
        if sig not in df.columns:
            dt = np.dtype(dtype_map.get(sig, float))
            if config.interpolate_missing and sig in all_sigs:
                srs = pd.Series(np.nan, index=df.index, dtype=dt)
                df[sig] = srs.interpolate(method="linear", limit_direction="both")
            elif np.issubdtype(dt, np.integer):
                df[sig] = np.zeros(len(df), dtype=dt)
            else:
                df[sig] = pd.Series(np.nan, index=df.index, dtype=dt)

    # Metadata attributes
    df.attrs["signal_attributes"] = {
        sig.name: getattr(sig, "attributes", {})
        for msg in dbobj.messages for sig in msg.signals
        if sig.name in df.columns
    }

    # Enum conversion
    for msg in dbobj.messages:
        for sig in msg.signals:
            if sig.name in df.columns and getattr(sig, "choices", None):
                safe_map = {str(k): v for k, v in sig.choices.items()}
                df[sig.name] = (
                    df[sig.name].astype(str)
                                .map(lambda x: safe_map.get(x, x))
                )
                df[sig.name] = pd.Categorical(df[sig.name], categories=list(safe_map.values()))

    return df[["timestamp"] + [c for c in df.columns if c != "timestamp"]]

# ----------------------------------------------------------------------------
# CSV export
# ----------------------------------------------------------------------------
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
    Write DataFrame or chunks to CSV with metadata JSON.

    Args:
        df_or_iter (DataFrame or iterable): Data to write.
        output_path (str): CSV path.
        mode ("w"/"a"): Write or append mode.
        header (bool): Include header row.
        pandas_kwargs (dict, optional): Extra pandas.to_csv kwargs.
        columns (list, optional): Subset of columns to write.
        metadata_path (str, optional): JSON path for signal_attributes.

    Example:
        to_csv(df, "out.csv", metadata_path="out_meta.json")
    """
    import json

    p = Path(output_path)
    pandas_kwargs = pandas_kwargs or {}
    if columns and len(columns) != len(set(columns)):
        raise ValueError("Duplicate columns specified")
    p.parent.mkdir(parents=True, exist_ok=True)

    def _write(df: pd.DataFrame, m, h, wmeta):
        df.to_csv(p, mode=m, header=h, index=False, columns=columns, **pandas_kwargs)
        if metadata_path and wmeta:
            mpath = Path(metadata_path)
            mpath.parent.mkdir(parents=True, exist_ok=True)
            attrs = df.attrs.get("signal_attributes", {c: {} for c in df.columns})
            mpath.write_text(json.dumps(attrs))

    if isinstance(df_or_iter, pd.DataFrame):
        _write(df_or_iter, mode, header, True)
    else:
        first = True
        for chunk in df_or_iter:
            _write(chunk, mode if first else "a",
                   header if first else False, first)
            first = False

    glogger.info(f"CSV written to {output_path}")

# ----------------------------------------------------------------------------
# Parquet export
# ----------------------------------------------------------------------------
def to_parquet(
    df: pd.DataFrame,
    output_path: str,
    compression: str = "snappy",
    pandas_kwargs: Optional[Dict[str, Any]] = None,
    metadata_path: Optional[str] = None,
) -> None:
    """
    Write a DataFrame to Parquet with optional metadata JSON.

    Args:
        df (DataFrame): Data to write.
        output_path (str): .parquet file path.
        compression (str): Codec e.g. snappy, gzip.
        pandas_kwargs (dict, optional): Extra pandas.to_parquet kwargs.
        metadata_path (str, optional): JSON path for signal_attributes.

    Example:
        to_parquet(df, "data.parquet", metadata_path="data_meta.json")
    """
    import json

    p = Path(output_path)
    pandas_kwargs = pandas_kwargs or {}
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p, engine="pyarrow", compression=compression, **pandas_kwargs)
    except Exception as e:
        glogger.error(f"Failed Parquet {p}: {e}", exc_info=True)
        raise ValueError(f"Failed to export Parquet: {e}") from e

    if metadata_path:
        mpath = Path(metadata_path)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        attrs = df.attrs.get("signal_attributes", {c: {} for c in df.columns})
        mpath.write_text(json.dumps(attrs))
        glogger.info(f"Metadata written to {mpath}")
    glogger.info(f"Parquet written to {output_path}")
