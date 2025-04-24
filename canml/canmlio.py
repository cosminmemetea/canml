import cantools
import pandas as pd
from python_can import Bus, Message
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_blf(blf_path: str, dbc_path: str) -> pd.DataFrame:
    """
    Load a BLF file, decode CAN messages using a DBC file, and return a pandas DataFrame.

    Args:
        blf_path (str): Path to the BLF file.
        dbc_path (str): Path to the CAN.DBC file.

    Returns:
        pd.DataFrame: DataFrame with timestamps and decoded signal values.

    Raises:
        FileNotFoundError: If BLF or DBC file is not found.
        ValueError: If BLF file is invalid or cannot be decoded.
    """
    # Validate file paths
    blf_file = Path(blf_path)
    dbc_file = Path(dbc_path)
    if not blf_file.exists():
        raise FileNotFoundError(f"BLF file not found: {blf_path}")
    if not dbc_file.exists():
        raise FileNotFoundError(f"DBC file not found: {dbc_path}")

    try:
        # Load DBC file
        logger.info(f"Loading DBC file: {dbc_path}")
        db = cantools.database.load_file(dbc_path)

        # Initialize data storage
        data = []
        timestamps = []

        # Read BLF file
        logger.info(f"Reading BLF file: {blf_path}")
        bus = Bus(blf_path, interface='vector', receive_own_messages=False)

        # Process each message
        for msg in bus:
            try:
                # Decode message using DBC
                decoded = db.decode_message(msg.arbitration_id, msg.data)
                # Append decoded signals and timestamp
                data.append(decoded)
                timestamps.append(msg.timestamp)
            except cantools.database.errors.DecodeError:
                # Skip messages that can't be decoded
                logger.debug(f"Skipping undecodable message with ID {msg.arbitration_id}")
                continue
            except KeyError:
                # Skip messages not defined in DBC
                logger.debug(f"Skipping message with ID {msg.arbitration_id} not in DBC")
                continue

        # Close bus
        bus.shutdown()

        # Create DataFrame
        logger.info("Converting decoded data to DataFrame")
        df = pd.DataFrame(data)
        df['timestamp'] = timestamps

        # Reorder columns to have timestamp first
        cols = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
        df = df[cols]

        # Ensure DataFrame is not empty
        if df.empty:
            logger.warning("No valid data decoded from BLF file")
            return pd.DataFrame()

        logger.info(f"Successfully decoded {len(df)} messages")
        return df

    except Exception as e:
        logger.error(f"Error processing BLF file: {str(e)}")
        raise ValueError(f"Failed to process BLF file: {str(e)}")

def to_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Export a pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing decoded CAN data.
        output_path (str): Path to save the CSV file.

    Raises:
        ValueError: If DataFrame is empty or output path is invalid.
    """
    if df.empty:
        raise ValueError("Cannot export empty DataFrame to CSV")

    output_file = Path(output_path)
    logger.info(f"Exporting data to CSV: {output_path}")
    try:
        df.to_csv(output_file, index=False)
        logger.info(f"Successfully exported data to {output_path}")
    except Exception as e:
        logger.error(f"Error exporting to CSV: {str(e)}")
        raise ValueError(f"Failed to export CSV: {str(e)}")