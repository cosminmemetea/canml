#!/usr/bin/env python3
"""
Generate a BLF file from one or multiple DBCs using random signal values.

Usage:
    python generate_blf.py \
        -d powertrain.dbc -d chassis.dbc \
        -o output.blf -n 100 -i 0.01
"""
import argparse
import time
import random
from pathlib import Path

import cantools
import can
from can.io.blf import BLFWriter


def generate_signal_value(signal):
    """Generate a random physical value for a cantools signal."""
    bit_length = signal.length
    max_raw = (1 << bit_length) - 1
    # Physical min/max from DBC
    min_val = signal.minimum if signal.minimum is not None else 0
    max_val = signal.maximum if signal.maximum is not None else max_raw * signal.scale
    # Enumerated choices
    if signal.choices:
        return random.choice(list(signal.choices.keys()))
    # Scaled values
    if signal.scale != 1:
        raw_min = int(min_val / signal.scale)
        raw_max = min(int(max_val / signal.scale), max_raw)
        raw = random.randint(raw_min, raw_max)
        return raw * signal.scale
    # Unscaled integer
    return random.randint(int(min_val), int(max_val))


def main():
    parser = argparse.ArgumentParser(description="Generate BLF from DBC(s)")
    parser.add_argument('-d', '--dbc', action='append', required=True,
                        help="Path to a DBC file (can specify multiple)")
    parser.add_argument('-o', '--output', required=True,
                        help="Output BLF file path")
    parser.add_argument('-n', '--num-msgs', type=int, default=100,
                        help="Number of messages per DBC message type")
    parser.add_argument('-i', '--interval', type=float, default=0.01,
                        help="Interval in seconds between messages")
    args = parser.parse_args()

    # Resolve paths
    db_files = [Path(d) for d in args.dbc]
    for p in db_files:
        if not p.is_file():
            parser.error(f"DBC file not found: {p}")
    out_file = Path(args.output)

    # Load DBCs
    db = cantools.database.Database()
    for p in db_files:
        print(f"Loading DBC: {p}")
        db.add_dbc_file(str(p))
    # Prepare writer
    writer = BLFWriter(str(out_file), channel=1)

    # Message definitions
    msg_defs = db.messages
    start = time.time()

    # Generate messages
    for i in range(args.num_msgs):
        for msg_def in msg_defs:
            data = {}
            for sig in msg_def.signals:
                data[sig.name] = generate_signal_value(sig)
            try:
                payload = msg_def.encode(data)
            except Exception as e:
                print(f"Error encoding {msg_def.name}: {e}")
                continue
            msg = can.Message(
                arbitration_id=msg_def.frame_id,
                data=payload,
                is_extended_id=False,
                timestamp=start + i * args.interval
            )
            writer.on_message_received(msg)

    # Close
    writer.stop()
    print(f"Generated BLF: {out_file}")


if __name__ == "__main__":
    main()
