#!/usr/bin/env python3
"""
generate_blf.py

Generate a BLF log file with random CAN messages defined in a DBC file.
"""
import argparse
import time
import random
import logging
from pathlib import Path

import cantools
import can
from can.io.blf import BLFWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def generate_signal_value(signal):
    """
    Generate a random physical value within the signal's DBC-defined range.
    Handles enumerations, scaling, and offset.
    """
    # Compute raw limits based on bit length
    bit_length = signal.length
    max_raw = (1 << bit_length) - 1

    # Determine physical min/max
    phys_min = signal.minimum if signal.minimum is not None else (0 * signal.scale + signal.offset)
    phys_max = signal.maximum if signal.maximum is not None else (max_raw * signal.scale + signal.offset)

    # Enumerated signals
    if signal.choices:
        return random.choice(list(signal.choices.keys()))

    # Convert physical bounds to raw
    raw_min = int((phys_min - signal.offset) / signal.scale)
    raw_max = int((phys_max - signal.offset) / signal.scale)
    raw_min = max(0, raw_min)
    raw_max = min(max_raw, raw_max)

    raw = random.randint(raw_min, raw_max)
    phys = raw * signal.scale + signal.offset
    return phys


def main():
    parser = argparse.ArgumentParser(
        description="Generate a BLF file with random CAN messages based on a DBC file."
    )
    parser.add_argument(
        "-d", "--dbc", required=True,
        type=Path,
        help="Path to the DBC file"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        type=Path,
        help="Output BLF filename"
    )
    parser.add_argument(
        "-n", "--num-messages", type=int, default=100,
        help="Number of frames per message type (default=100)"
    )
    parser.add_argument(
        "-i", "--interval", type=float, default=0.01,
        help="Timestamp interval between frames in seconds (default=0.01)"
    )

    args = parser.parse_args()

    # Validate DBC file
    if not args.dbc.exists():
        logger.error(f"DBC file not found: {args.dbc}")
        return 1

    # Load database
    try:
        db = cantools.database.load_file(str(args.dbc))
    except Exception as e:
        logger.error(f"Failed to load DBC: {e}")
        return 2

    # Initialize BLF writer
    try:
        writer = BLFWriter(str(args.output), channel=1, compression_level=-1)
    except Exception as e:
        logger.error(f"Could not create BLFWriter: {e}")
        return 3

    # Select messages by name
    message_names = [
        "EngineData",
        "VehicleDynamics",
        "BrakeStatus",
        "EnvironmentData"
    ]
    messages = []
    for name in message_names:
        msg = db.get_message_by_name(name)
        if msg is None:
            logger.warning(f"Message '{name}' not found in DBC, skipping.")
        else:
            messages.append(msg)

    if not messages:
        logger.error("No valid message definitions found in DBC. Exiting.")
        writer.stop()
        return 4

    start_ts = time.time()
    total_expected = len(messages) * args.num_messages
    count = 0

    # Generate frames
    for idx in range(args.num_messages):
        for msg_def in messages:
            payload = {
                sig.name: generate_signal_value(sig)
                for sig in msg_def.signals
            }
            try:
                data = msg_def.encode(payload)
            except Exception as e:
                logger.error(f"Encoding {msg_def.name} failed: {e}")
                continue

            ts = start_ts + idx * args.interval
            can_msg = can.Message(
                arbitration_id=msg_def.frame_id,
                data=data,
                is_extended_id=False,
                timestamp=ts
            )
            try:
                writer.on_message_received(can_msg)
            except Exception as e:
                logger.error(f"Writing message {msg_def.name} failed: {e}")
            count += 1

    # Finalize
    writer.stop()
    logger.info(f"Wrote {count}/{total_expected} frames to {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
