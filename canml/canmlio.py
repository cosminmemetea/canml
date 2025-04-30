"""
canmlio: Enhanced CAN BLF processing toolkit for production use.
Module: canml/canmlio.py

Features:
  - Merge multiple DBCs with namespace collision avoidance.
  - Stream-decode large BLF files into pandas DataFrame chunks.
  - Full-file loading with uniform timestamp spacing and interpolation.
  - Signal/message filtering by ID or signal name.
  - Automatic injection of expected signals with dtype preservation.
  - Incremental CSV/Parquet export with metadata support.
  - Generic handling for enums and custom signal attributes.
  - Progress bars via tqdm and caching for DBC loading.

Dependencies:
  numpy, pandas, cantools, python-can, tqdm, pyarrow

Usage:
  from canml.canmlio import load_dbc_files, iter_blf_chunks, load_blf, to_csv, to_parquet, CanmlConfig

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

# Module logger: single StreamHandler
glogger = logging.getLogger(__name__)
glogger.handlers.clear()
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
glogger.addHandler(_handler)
glogger.setLevel(logging.INFO)

T = Any

@dataclass
class CanmlConfig:
    """
    Configuration for BLF processing.

    Attributes:
        chunk_size: number of messages per DataFrame chunk.
        progress_bar: show tqdm progress bar if True.
        dtype_map: mapping from signal name to desired pandas dtype.
        sort_timestamps: sort final DataFrame by timestamp if True.
        force_uniform_timing: override timestamps with uniform spacing if True.
        interval_seconds: interval between timestamps when uniform timing enabled.
        interpolate_missing: interpolate missing signals if True.
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

@lru_cache(maxsize=32)
def _load_dbc_files_cached(
    dbc_paths: Union[str, Tuple[str, ...]], prefix_signals: bool
) -> CantoolsDatabase:
    """
    Internal cached loader for DBC files.

    Args:
        dbc_paths: single path or tuple of paths to .dbc files.
        prefix_signals: if True, prefix signal names with their message name.

    Returns:
        CantoolsDatabase with all DBC definitions loaded.

    Raises:
        ValueError: for invalid inputs or parsing errors.
        FileNotFoundError: if any path does not exist.
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

    # Detect duplicate signal names
    names = [sig.name for msg in db.messages for sig in msg.signals]
    if not prefix_signals:
        dupes = [n for n, c in Counter(names).items() if c > 1]
        if dupes:
            raise ValueError(
                f"Duplicate signal names: {sorted(dupes)}; use prefix_signals=True"
            )
    else:
        # Ensure unique message names before prefixing
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
    """
    Load and optionally prefix one or more DBC files into a Cantools database.

    Args:
        dbc_paths: path or list of paths to .dbc files.
        prefix_signals: if True, prefix signal names with their message name.

    Returns:
        Cached CantoolsDatabase instance.
    """
    key = tuple(dbc_paths) if isinstance(dbc_paths, list) else dbc_paths
    return _load_dbc_files_cached(key, prefix_signals)

@contextmanager
def blf_reader(path: str) -> Iterator[BLFReader]:
    """
    Context manager for BLFReader to ensure reader.stop() on exit.
    """
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
    """
    Stream-decode BLF file into DataFrame chunks.

    Args:
        blf_path: .blf file path.
        db: loaded CantoolsDatabase.
        config: CanmlConfig instance.
        filter_ids: set of arbitration IDs to include.
        filter_signals: set of signal names to include.

    Yields:
        pandas.DataFrame chunks of decoded messages.
    """
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
    Load an entire BLF file into a pandas DataFrame, with robust decoding,
    filtering, timing normalization, missing‐signal injection, and metadata.

    Args:
        blf_path: Path to the .blf log file.
        db: Either a CantoolsDatabase instance or path(s) to DBC file(s).
        config: CanmlConfig instance controlling chunking, timing, dtypes, etc.
        message_ids: Optional set of CAN arbitration IDs to include (None = all).
        expected_signals: Optional iterable of signal names to include (None = all signals in DBC).

    Returns:
        DataFrame with columns ["timestamp", …signals], enriched with:
          - raw_timestamp (if uniform timing)
          - df.attrs["signal_attributes"] mapping signal→custom attributes
          - enum signals as pandas.Categorical
    """
    # 1️⃣ Prepare config
    config = config or CanmlConfig()

    # 2️⃣ Normalize expected_signals and check duplicates
    if expected_signals is not None:
        exp_list = [str(s) for s in expected_signals]
        if len(exp_list) != len(set(exp_list)):
            raise ValueError("Duplicate names in expected_signals")
    else:
        exp_list = None

    # 3️⃣ Load or reuse CantoolsDatabase
    dbobj = db if isinstance(db, CantoolsDatabase) else load_dbc_files(db)

    # 4️⃣ Warn if user passed an explicit empty message_ids
    if message_ids is not None and not message_ids:
        glogger.warning("Empty message_ids provided; no messages will be decoded")

    # 5️⃣ Determine which signals to expect
    all_sigs = [s.name for msg in dbobj.messages for s in msg.signals]
    expected = exp_list if exp_list is not None else all_sigs

    # 6️⃣ Validate dtype_map keys
    dtype_map = config.dtype_map or {}
    for sig in dtype_map:
        if sig not in expected:
            raise ValueError(f"dtype_map contains unknown signal: {sig}")

    # 7️⃣ Build a safe filter_signals set (avoid unhashable items)
    try:
        filter_set = set(expected)
    except TypeError:
        filter_set = {str(s) for s in expected}

    # 8️⃣ Decode in chunks
    try:
        chunks = list(iter_blf_chunks(
            blf_path,
            dbobj,
            config,
            message_ids,
            filter_set
        ))
    except FileNotFoundError:
        raise
    except Exception as e:
        glogger.error("Failed to process BLF chunks", exc_info=True)
        raise ValueError(f"Failed to process BLF data: {e}") from e

    # 9️⃣ Assemble DataFrame
    if not chunks:
        glogger.warning(f"No data decoded from {blf_path}; returning empty DataFrame")
        df = pd.DataFrame({
            "timestamp": pd.Series(dtype=float),
            **{sig: pd.Series(dtype=dtype_map.get(sig, float)) for sig in expected}
        })
    else:
        df = pd.concat(chunks, ignore_index=True)

    # 1️⃣0️⃣ Retain only timestamp + expected columns
    cols_keep = [c for c in ["timestamp"] + expected if c in df.columns]
    df = df[cols_keep]

    # 1️⃣1️⃣ Optional sorting
    if config.sort_timestamps:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # 1️⃣2️⃣ Optional uniform timing
    if config.force_uniform_timing:
        df["raw_timestamp"] = df["timestamp"]
        df["timestamp"] = np.arange(len(df)) * config.interval_seconds

    # 1️⃣3️⃣ Inject missing signals with correct dtype
    for sig in expected:
        if sig not in df.columns:
            npdt = np.dtype(dtype_map.get(sig, float))
            if config.interpolate_missing and sig in all_sigs:
                # create NaN series then interpolate
                df[sig] = pd.Series([np.nan] * len(df), dtype=npdt) \
                             .interpolate(method="linear", limit_direction="both")
            elif np.issubdtype(npdt, np.integer):
                df[sig] = np.zeros(len(df), dtype=npdt)
            else:
                df[sig] = pd.Series([np.nan] * len(df), dtype=npdt)

    # 1️⃣4️⃣ Collect custom attributes and convert enums
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

    # 1️⃣5️⃣ Ensure timestamp is first column
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
    Write DataFrame or iterable of DataFrames to CSV, exporting metadata.

    Args:
        df_or_iter: single DataFrame or iterable of chunks.
        output_path: CSV file path.
        mode: 'w' or 'a'.
        header: write header row.
        pandas_kwargs: extra pandas.to_csv kwargs.
        columns: subset/order of columns to write.
        metadata_path: JSON file path to save df.attrs["signal_attributes"].
    """
    import json

    p = Path(output_path)
    pandas_kwargs = pandas_kwargs or {}

    # Validate columns
    if columns and len(columns) != len(set(columns)):
        raise ValueError("Duplicate columns specified")
    # Ensure CSV dir exists
    p.parent.mkdir(parents=True, exist_ok=True)

    def _write(df: pd.DataFrame, mode_: str, header_: bool, write_meta: bool):
        df.to_csv(p, mode=mode_, header=header_, index=False, columns=columns, **pandas_kwargs)
        if metadata_path and write_meta:
            m = Path(metadata_path)
            m.parent.mkdir(parents=True, exist_ok=True)
            sig_attrs = df.attrs.get("signal_attributes") or {c: {} for c in df.columns}
            m.write_text(json.dumps(sig_attrs))

    if isinstance(df_or_iter, pd.DataFrame):
        _write(df_or_iter, mode, header, True)
    else:
        first = True
        for chunk in df_or_iter:
            _write(chunk, mode if first else "a",
                   header if first else False, first)
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
        output_path: .parquet file path.
        compression: Parquet codec.
        pandas_kwargs: kwargs for pandas.to_parquet.
        metadata_path: JSON path for signal_attributes metadata.
    """
    import json

    p = Path(output_path)
    pandas_kwargs = pandas_kwargs or {}
    # Ensure dir exists
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p, engine="pyarrow", compression=compression, **pandas_kwargs)
    except Exception as e:
        glogger.error(f"Failed to write Parquet {p}: {e}", exc_info=True)
        raise ValueError(f"Failed to export Parquet: {e}") from e
    # Metadata export
    if metadata_path:
        m = Path(metadata_path)
        m.parent.mkdir(parents=True, exist_ok=True)
        sig_attrs = df.attrs.get("signal_attributes") or {c: {} for c in df.columns}
        m.write_text(json.dumps(sig_attrs))
        glogger.info(f"Metadata written to {m}")
    glogger.info(f"Parquet written to {p}")
