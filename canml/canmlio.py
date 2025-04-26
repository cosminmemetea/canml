# canml/canmlio.py
"""
canmlio: A library for processing CAN BLF files with DBC decoding.

This module provides functions to load CAN BLF files, decode messages using a DBC file,
and export the results to various formats (e.g., CSV). It is designed to be robust, flexible,
and suitable for automotive CAN data analysis.

Dependencies:
    - python-can (for BLF reading)
    - cantools (for DBC decoding)
    - pandas (for DataFrame handling)

Usage example:
    ```python
    from canml import canmlio

    # Load BLF file with DBC
    df = canmlio.load_blf("data.blf", "signals.dbc", use_raw_timestamps=True)

    # Export to CSV
    canmlio.to_csv(df, "output.csv")
    ```
"""
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import cantools
from can.io.blf import BLFReader

# Public API
__all__: List[str] = ["load_blf", "to_csv"]

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_blf(
    blf_path: str,
    dbc_path: str,
    force_uniform_timing: bool = True,
    interval_seconds: float = 0.01
) -> pd.DataFrame:
    """
    Load a BLF file and decode CAN messages using a DBC file.

    Arguments:
        blf_path: Path to the BLF log file.
        dbc_path: Path to the DBC file defining CAN signals.
        force_uniform_timing: If True, override timestamps with uniform spacing.
        interval_seconds: Interval in seconds for uniform timing (default 0.01s).

    Returns:
        A pandas DataFrame containing a "timestamp" column plus one column per signal.

    Raises:
        FileNotFoundError: BLF or DBC file not found.
        ValueError: Invalid BLF or DBC file.
    """
    blf_file = Path(blf_path)
    dbc_file = Path(dbc_path)

    if not blf_file.is_file():
        raise FileNotFoundError(f"BLF file not found: {blf_path}")
    if not dbc_file.is_file():
        raise FileNotFoundError(f"DBC file not found: {dbc_path}")

    # Load DBC
    try:
        logger.info(f"Loading DBC: {dbc_path}")
        db = cantools.database.load_file(str(dbc_file))
    except Exception as e:
        logger.error(f"Invalid DBC file {dbc_path}: {e}")
        raise ValueError(f"Invalid DBC file: {e}")

    # Open BLF
    try:
        logger.info(f"Opening BLF: {blf_path}")
        reader = BLFReader(str(blf_file))
    except Exception as e:
        logger.error(f"Invalid BLF file {blf_path}: {e}")
        raise ValueError(f"Invalid BLF file: {e}")

    records: List[dict] = []
    raw_timestamps: List[float] = []

    for msg in reader:
        try:
            decoded = db.decode_message(msg.arbitration_id, msg.data)
        except (cantools.database.errors.DecodeError, KeyError):
            # Skip messages that can't be decoded
            logger.debug(f"Skipping CAN ID {msg.arbitration_id}")
            continue

        records.append(decoded)
        raw_timestamps.append(msg.timestamp)

    # Close reader
    reader.stop()

    # Build DataFrame
    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No CAN messages decoded from BLF")
        return df

    if force_uniform_timing:
        # Uniform spacing by record index
        df["timestamp"] = df.index * interval_seconds
    else:
        df["timestamp"] = raw_timestamps

    # Ensure timestamp is first column
    columns = ["timestamp"] + [col for col in df.columns if col != "timestamp"]
    return df[columns]


def to_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Export DataFrame to CSV.

    Args:
        df: pandas DataFrame with decoded CAN data.
        output_path: Destination CSV file path.

    Raises:
        ValueError: If DataFrame is empty or write fails.
    """
    if df.empty:
        raise ValueError("Cannot export empty DataFrame: DataFrame is empty")

    output_file = Path(output_path)
    try:
        logger.info(f"Writing CSV: {output_path}")
        df.to_csv(output_file, index=False)
    except Exception as e:
        logger.error(f"Failed to write CSV {output_path}: {e}")
        raise ValueError(f"Failed to export CSV: {e}")
