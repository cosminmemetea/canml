"""
canmlio: Enhanced CAN BLF processing toolkit for production use.
Module: canml/canmlio.py
Features:
  - Merge multiple DBCs with namespace collision avoidance.
  - Stream-decode large BLF files into pandas DataFrame chunks.
  - Full-file loading with uniform timestamp spacing and interpolation.
  - Signal/message filtering by ID or name.
  - Automatic injection of expected signals with dtype preservation.
  - Incremental CSV/Parquet export with metadata.
  - Generic support for enums and custom attributes.
  - Progress bars and caching.
"""
import logging
from pathlib import Path
from typing import List, Optional, Union, Iterator, Set, Dict, Any, Iterable, Tuple
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

# Configure logger
glogger = logging.getLogger(__name__)
glogger.handlers.clear()
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
glogger.addHandler(handler)
glogger.setLevel(logging.INFO)

T = Any

@dataclass
class CanmlConfig:
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

@lru_cache(maxsize=32)
def _load_dbc_files_cached(
    dbc_paths: Union[str, Tuple[str, ...]], prefix_signals: bool
) -> CantoolsDatabase:
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
        dupes = [n for n, c in Counter(names).items() if c > 1]
        if dupes:
            raise ValueError(f"Duplicate signal names: {sorted(dupes)}; use prefix_signals=True")
    else:
        msg_names = [m.name for m in db.messages]
        if len(msg_names) != len(set(msg_names)):
            raise ValueError("Duplicate message names found; cannot prefix uniquely")
        for msg in db.messages:
            for sig in msg.signals:
                sig.name = f"{msg.name}_{sig.name}"
    return db

def load_dbc_files(
    dbc_paths: Union[str, List[str]], prefix_signals: bool = False
) -> CantoolsDatabase:
    paths = tuple(dbc_paths) if isinstance(dbc_paths, list) else dbc_paths  # for caching
    return _load_dbc_files_cached(paths, prefix_signals)

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

def iter_blf_chunks(
    blf_path: str,
    db: CantoolsDatabase,
    config: CanmlConfig,
    filter_ids: Optional[Set[int]] = None,
    filter_signals: Optional[Set[str]] = None,
) -> Iterator[pd.DataFrame]:
    p = Path(blf_path)
    if p.suffix.lower() != ".blf" or not p.is_file():
        raise FileNotFoundError(f"Valid BLF file not found: {p}")

    buffer: List[Dict[str, T]] = []
    with blf_reader(blf_path) as reader:
        iterator = tqdm(reader, desc=p.name) if config.progress_bar else reader
        for msg in iterator:
            if filter_ids and msg.arbitration_id not in filter_ids:
                continue
            try:
                rec = db.decode_message(msg.arbitration_id, msg.data)
            except Exception:
                continue
            if filter_signals:
                rec = {k: v for k, v in rec.items() if k in filter_signals}
            if rec:
                rec["timestamp"] = msg.timestamp
                buffer.append(rec)
            if len(buffer) >= config.chunk_size:
                yield pd.DataFrame(buffer)
                buffer.clear()
        if buffer:
            yield pd.DataFrame(buffer)

def load_blf(
    blf_path: str,
    db: Union[CantoolsDatabase, str, List[str]],
    config: Optional[CanmlConfig] = None,
    message_ids: Optional[Set[int]] = None,
    expected_signals: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Load BLF file into a DataFrame with robust signal handling and metadata.

    Args:
        blf_path: Path to BLF file.
        db: Cantools Database object or DBC file path(s).
        config: Configuration object (optional).
        message_ids: Arbitration IDs to include.
        expected_signals: Signals to include in output (adds missing signals).

    Returns:
        DataFrame with decoded signals and metadata.
    """
    config = config or CanmlConfig()

    # Early duplicate check
    if expected_signals is not None:
        exp_list = list(expected_signals)
        if len(exp_list) != len(set(exp_list)):
            raise ValueError("Duplicate names in expected_signals")
    else:
        exp_list = None

    # Load or reuse database
    dbobj = db if isinstance(db, CantoolsDatabase) else load_dbc_files(db)

    # Warn if the user explicitly passed an empty filter set
    if message_ids is not None and not message_ids:
        glogger.warning("Empty message_ids provided; no messages will be decoded")

    # Decide which signals we expect
    all_sigs = [s.name for m in dbobj.messages for s in m.signals]
    expected = exp_list if exp_list is not None else all_sigs

    # Validate dtype_map keys
    dtype_map = config.dtype_map or {}
    for sig in dtype_map:
        if sig not in expected:
            raise ValueError(f"dtype_map contains unknown signal: {sig}")

    # Stream in chunks (catch any error and wrap)
    try:
        chunks = list(iter_blf_chunks(blf_path, dbobj, config, message_ids, set(expected)))
    except FileNotFoundError:
        raise
    except Exception as e:
        glogger.error("Failed to process BLF chunks", exc_info=True)
        raise ValueError(f"Failed to process BLF data: {e}") from e

    # Build the DataFrame
    if not chunks:
        glogger.warning(f"No data decoded from {blf_path}; returning empty DataFrame")
        df = pd.DataFrame({
            "timestamp": pd.Series(dtype=float),
            **{sig: pd.Series(dtype=dtype_map.get(sig, float)) for sig in expected}
        })
    else:
        df = pd.concat(chunks, ignore_index=True)

    # Always keep only timestamp + the expected signal columns
    cols_keep = [c for c in ["timestamp"] + expected if c in df.columns]
    df = df[cols_keep]

    # Optional sorting
    if config.sort_timestamps:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Optional uniform spacing
    if config.force_uniform_timing:
        df["raw_timestamp"] = df["timestamp"]
        df["timestamp"] = np.arange(len(df)) * config.interval_seconds

    # Inject any missing signals, preserving int dtype or float NaNs
    for sig in expected:
        if sig not in df.columns:
            npdt = np.dtype(dtype_map.get(sig, float))
            if config.interpolate_missing and sig in all_sigs:
                df[sig] = df[sig].interpolate(method="linear", limit_direction="both")
            elif np.issubdtype(npdt, np.integer):
                df[sig] = np.zeros(len(df), dtype=npdt)
            else:
                df[sig] = pd.Series([np.nan] * len(df), dtype=npdt)

    # Collect metadata and convert enums
    df.attrs["signal_attributes"] = {
        s.name: getattr(s, "attributes", {})
        for m in dbobj.messages for s in m.signals
        if s.name in df.columns
    }
    for m in dbobj.messages:
        for s in m.signals:
            if s.name in df.columns and getattr(s, "choices", None):
                df[s.name] = pd.Categorical(
                    df[s.name].map(s.choices),
                    categories=list(s.choices.values())
                )

    return df[["timestamp"] + [c for c in df.columns if c != "timestamp"]]

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
    Write DataFrame or chunks to CSV with optional metadata export.
    """
    import json

    p = Path(output_path)
    pandas_kwargs = pandas_kwargs or {}

    # Validate requested columns
    if columns and len(columns) != len(set(columns)):
        raise ValueError("Duplicate columns specified")

    # Ensure output directory exists
    p.parent.mkdir(parents=True, exist_ok=True)

    def _write(df: pd.DataFrame, write_mode: str, write_header: bool, write_meta: bool):
        # 1) CSV
        df.to_csv(p, mode=write_mode, header=write_header, index=False, columns=columns, **pandas_kwargs)
        # 2) Metadata
        if metadata_path and write_meta:
            m = Path(metadata_path)
            m.parent.mkdir(parents=True, exist_ok=True)
            # If the DataFrame has signal_attributes, use them; otherwise default to empty dict per column
            sig_attrs = df.attrs.get("signal_attributes") or {col: {} for col in df.columns}
            m.write_text(json.dumps(sig_attrs))

    # Single DataFrame vs. chunked
    if isinstance(df_or_iter, pd.DataFrame):
        _write(df_or_iter, mode, header, True)
    else:
        first = True
        for chunk in df_or_iter:
            _write(chunk, mode if first else "a", header if first else False, first)
            first = False

    glogger.info(f"CSV written to {output_path}")

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
        output_path: Path to the .parquet file.
        compression: Parquet codec (snappy, gzip, etc.).
        pandas_kwargs: Additional arguments for pandas.to_parquet.
        metadata_path: Optional JSON file path to save df.attrs["signal_attributes"].
    """
    from pathlib import Path
    import json

    p = Path(output_path)
    pandas_kwargs = pandas_kwargs or {}

    # 1) Ensure output directory exists
    p.parent.mkdir(parents=True, exist_ok=True)

    # 2) Write the Parquet
    try:
        df.to_parquet(p, engine="pyarrow", compression=compression, **pandas_kwargs)
    except Exception as e:
        glogger.error(f"Failed to write Parquet {p}: {e}", exc_info=True)
        raise ValueError(f"Failed to export Parquet: {e}") from e

    # 3) Write metadata JSON (always create its dir too)
    if metadata_path:
        m = Path(metadata_path)
        m.parent.mkdir(parents=True, exist_ok=True)
        # Export existing signal_attributes or empty dict per column
        sig_attrs = df.attrs.get("signal_attributes") or {col: {} for col in df.columns}
        m.write_text(json.dumps(sig_attrs))
        glogger.info(f"Metadata written to {m}")

    glogger.info(f"Parquet written to {p}")
