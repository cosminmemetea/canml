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

    # 1. Load DBC with safe prefixing
    db = load_dbc_files("vehicle.dbc", prefix_signals=True)

    # 2. Configure BLF loading
    cfg = CanmlConfig(
        chunk_size=5000,
        progress_bar=True,
        sort_timestamps=True,
        force_uniform_timing=True,
        interval_seconds=0.02,
        interpolate_missing=True,
        dtype_map={"Engine_RPM": "int32"}
    )

    # 3. Load BLF file into DataFrame
    df = load_blf(
        blf_path="drive.blf",
        db=db,
        config=cfg,
        message_ids={0x100, 0x200},
        expected_signals=["Engine_RPM", "Brake_Active"]
    )

    # 4. Export results
    to_csv(df, "drive.csv", metadata_path="drive_meta.json")
"""
import logging
import json
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

# ----------------------------------------------------------------------------
# Configuration dataclass
# ----------------------------------------------------------------------------
@dataclass
class CanmlConfig:
    """
    Configuration options for BLF processing.

    Args:
        chunk_size (int): Number of messages per chunk. Defaults to 10000.
        progress_bar (bool): Show tqdm bar if True. Defaults to True.
        dtype_map (Optional[Dict[str, Any]]): Signal-to-dtype map. Defaults to None.
        sort_timestamps (bool): Sort by timestamp. Defaults to False.
        force_uniform_timing (bool): Uniform spacing of timestamps. Defaults to False.
        interval_seconds (float): Uniform interval seconds. Defaults to 0.01.
        interpolate_missing (bool): Interpolate missing signals. Defaults to False.

    Raises:
        ValueError: If chunk_size or interval_seconds <= 0.
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
# DBC loading and merging with safe signal prefixing
# ----------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _load_dbc_files_cached(
    dbc_paths: Union[str, Tuple[str, ...]], prefix_signals: bool
) -> CantoolsDatabase:
    """
    Internal: Load and merge .dbc files into a single CantoolsDatabase.
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

    # Prefixing logic
    all_signals = [sig.name for msg in db.messages for sig in msg.signals]
    if not prefix_signals:
        dupes = [n for n, c in Counter(all_signals).items() if c > 1]
        if dupes:
            raise ValueError(
                f"Duplicate signal names: {sorted(dupes)}; use prefix_signals=True"
            )
    else:
        msg_names = [msg.name for msg in db.messages]
        dup_msgs = [n for n, c in Counter(msg_names).items() if c > 1]
        for idx, msg in enumerate(db.messages):
            if dup_msgs:
                if hasattr(msg, 'frame_id'):
                    key = msg.frame_id
                elif hasattr(msg, 'arbitration_id'):
                    key = msg.arbitration_id
                else:
                    key = idx
                prefix = f"{msg.name}_{key}"
            else:
                prefix = msg.name
            for sig in msg.signals:
                sig.name = f"{prefix}_{sig.name}"
    return db


def load_dbc_files(
    dbc_paths: Union[str, List[str]], prefix_signals: bool = False
) -> CantoolsDatabase:
    """
    Load and merge DBC files with optional prefixing.
    """
    key = tuple(dbc_paths) if isinstance(dbc_paths, list) else dbc_paths
    return _load_dbc_files_cached(key, prefix_signals)

# ----------------------------------------------------------------------------
# BLFReader context manager
# ----------------------------------------------------------------------------
@contextmanager
def blf_reader(path: str) -> Iterator[BLFReader]:
    reader = BLFReader(str(path))
    try:
        yield reader
    finally:
        try:
            reader.stop()
        except Exception:
            glogger.debug("Error closing BLF reader", exc_info=True)

# ----------------------------------------------------------------------------
# Stream-decode BLF in chunks with drop summary
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

    Logs total vs dropped message counts.
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

    total = 0
    dropped = 0
    buffer: List[Dict[str, Any]] = []
    with blf_reader(blf_path) as reader:
        it = tqdm(reader, desc=p.name) if config.progress_bar else reader
        for msg in it:
            total += 1
            if filter_ids and msg.arbitration_id not in filter_ids:
                dropped += 1
                continue
            try:
                rec = db.decode_message(msg.arbitration_id, msg.data)
            except Exception:
                dropped += 1
                continue
            if sig_set is not None:
                rec = {k: v for k, v in rec.items() if k in sig_set}
            if rec:
                rec["timestamp"] = msg.timestamp
                buffer.append(rec)
            else:
                dropped += 1
            if len(buffer) >= config.chunk_size:
                yield pd.DataFrame(buffer)
                buffer.clear()
        if buffer:
            yield pd.DataFrame(buffer)
    glogger.info(f"Decoded {total-dropped}/{total} messages ({dropped} dropped)")

# ----------------------------------------------------------------------------
# Full-file load with filtering, timing, injection, metadata, enums
# ----------------------------------------------------------------------------

def load_blf(
    blf_path: str,
    db: Union[CantoolsDatabase, str, List[str]],
    config: Optional[CanmlConfig] = None,
    message_ids: Optional[Set[int]] = None,
    expected_signals: Optional[Iterable[Any]] = None,
) -> pd.DataFrame:
    """
    Load an entire BLF file into a DataFrame.

    Supports:
      - ID and signal filtering
      - Timestamp sorting and uniform spacing
      - Missing signal injection with dtype preservation
      - Metadata attributes and enum conversion
    """
    config = config or CanmlConfig()

    # Normalize expected_signals
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
    # Prevent collision with timing
    if exp_list and ("timestamp" in exp_list or "raw_timestamp" in exp_list):
        raise ValueError("'timestamp' or 'raw_timestamp' cannot be expected_signals")

    # Load or reuse database
    dbobj = db if isinstance(db, CantoolsDatabase) else load_dbc_files(db)
    if message_ids is not None and not message_ids:
        glogger.warning("Empty message_ids provided; no messages will be decoded")

    # Determine signals to include
    all_sigs: List[str] = [sig.name for msg in dbobj.messages for sig in msg.signals]
    expected: List[str] = exp_list if exp_list is not None else all_sigs

    # Validate dtype_map
    dtype_map: Dict[str, Any] = config.dtype_map or {}
    for sig, dt in dtype_map.items():
        if sig not in expected:
            raise ValueError(f"dtype_map contains unknown signal: {sig}")
        try:
            pd.Series(dtype=dt)
        except Exception:
            raise ValueError(f"Invalid dtype '{dt}' for signal '{sig}'")

    # Stream decode
    try:
        chunks = list(iter_blf_chunks(
            blf_path, dbobj, config, message_ids, expected
        ))
    except FileNotFoundError:
        raise
    except Exception as e:
        glogger.error("Failed to process BLF chunks", exc_info=True)
        raise ValueError(f"Failed to process BLF data: {e}") from e

    # Concatenate or create empty
    if not chunks:
        glogger.warning(f"No data decoded from {blf_path}; returning empty DataFrame")
        df = pd.DataFrame({
            "timestamp": pd.Series(dtype=float),
            **{sig: pd.Series(dtype=dtype_map.get(sig, float)) for sig in expected}
        })
    else:
        df = pd.concat(chunks, ignore_index=True)

    # Keep only timestamp + expected signals
    cols_keep = [c for c in ["timestamp"] + expected if c in df.columns]
    df = df[cols_keep]

    # Sort and uniform timing
    if config.sort_timestamps:
        df = df.sort_values("timestamp").reset_index(drop=True)
    if config.force_uniform_timing:
        df["raw_timestamp"] = df["timestamp"]
        df["timestamp"] = np.arange(len(df)) * config.interval_seconds

    # Inject missing signals
    reserved = {"timestamp", "raw_timestamp"}
    for sig in expected:
        if sig in reserved:
            continue
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

    # Enum conversion: safely map values to string labels
    for msg in dbobj.messages:
        for sig in msg.signals:
            if sig.name in df.columns and getattr(sig, "choices", None):
                choices = sig.choices  # raw -> label
                cats = [str(lab) for lab in choices.values()]
                def _map_label(x):
                    raw = getattr(x, 'value', x)
                    lbl = choices.get(raw, raw)
                    return str(lbl)
                df[sig.name] = df[sig.name].apply(_map_label)
                df[sig.name] = pd.Categorical(df[sig.name], categories=cats)

    return df

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
    Write DataFrame or chunks to CSV with side-car metadata JSON.

    Args:
        df_or_iter (DataFrame or iterable): Data to write.
        output_path (str): Destination CSV file path.
        mode (str): Write mode 'w' or 'a'.
        header (bool): Include header in CSV.
        pandas_kwargs (dict): Extra pandas.to_csv args.
        columns (list): Subset of columns to write.
        metadata_path (str): Path to JSON for signal_attributes.
    """
    import json

    p = Path(output_path)
    pandas_kwargs = pandas_kwargs or {}
    if columns and len(columns) != len(set(columns)):
        raise ValueError("Duplicate columns specified")
    p.parent.mkdir(parents=True, exist_ok=True)

    def _write(block: pd.DataFrame, m, h, wmeta):
        block.to_csv(p, mode=m, header=h, index=False, columns=columns, **pandas_kwargs)
        if metadata_path and wmeta:
            mpath = Path(metadata_path)
            mpath.parent.mkdir(parents=True, exist_ok=True)
            attrs = block.attrs.get("signal_attributes", {c: {} for c in block.columns})
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
    Write DataFrame to Parquet with side-car metadata JSON.

    Args:
        df (DataFrame): Data to write.
        output_path (str): Destination .parquet file path.
        compression (str): Parquet codec.
        pandas_kwargs (dict): Extra pandas.to_parquet args.
        metadata_path (str): JSON path for signal_attributes.
    """
    import json

    p = Path(output_path)
    pandas_kwargs = pandas_kwargs or {}
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p, engine="pyarrow", compression=compression, **pandas_kwargs)
    except Exception as e:
        glogger.error(f"Failed to export Parquet {p}: {e}", exc_info=True)
        raise ValueError(f"Failed to export Parquet: {e}")

    if metadata_path:
        mpath = Path(metadata_path)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        attrs = df.attrs.get("signal_attributes", {c: {} for c in df.columns})
        mpath.write_text(json.dumps(attrs))
        glogger.info(f"Metadata written to {mpath}")
    glogger.info(f"Parquet written to {output_path}")
