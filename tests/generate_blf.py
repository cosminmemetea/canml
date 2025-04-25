import time
import can
from can.io.blf import BLFWriter
import cantools
import random

# Load the DBC file
try:
    db = cantools.database.load_file('test.dbc')
except Exception as e:
    print(f"Error loading DBC file: {e}")
    exit(1)

# Create a BLF writer
blf_file = "output-2.blf"
try:
    writer = BLFWriter(blf_file, channel=1, compression_level=-1)
except Exception as e:
    print(f"Error creating BLF writer: {e}")
    exit(1)

# Function to generate random signal values within DBC-defined ranges and bit constraints
def generate_signal_value(signal):
    bit_length = signal.length
    max_bit_value = (1 << bit_length) - 1  # Max raw value for unsigned integer
    min_val = signal.minimum if signal.minimum is not None else 0
    max_val = signal.maximum if signal.maximum is not None else max_bit_value
    
    # Ensure min/max are within bit constraints
    min_val = max(min_val, 0)  # Unsigned signals can't be negative
    max_val = min(max_val, max_bit_value / signal.scale)  # Adjust for scale
    
    # Handle enumerated types (e.g., GearPosition)
    if signal.choices:
        choices = list(signal.choices.keys())
        return random.choice(choices)
    
    # Handle scaled signals (e.g., VehicleSpeed with scale 0.1)
    if signal.scale != 1:
        # Calculate raw min/max based on bit length and DBC range
        raw_min = int(min_val / signal.scale)
        raw_max = min(int(max_val / signal.scale), max_bit_value)
        raw_value = random.randint(raw_min, raw_max)
        physical_value = raw_value * signal.scale
        # Debug: Print raw and physical values
        print(f"Signal: {signal.name}, Raw: {raw_value}, Physical: {physical_value}")
        return physical_value
    
    # Generate integer for unscaled signals
    return random.randint(int(min_val), int(max_val))

# Simulate CAN messages
start_time = time.time()
num_messages = 100  # Number of messages per message type
messages = [
    db.get_message_by_name('EngineData'),
    db.get_message_by_name('VehicleDynamics'),
    db.get_message_by_name('BrakeStatus'),
    db.get_message_by_name('EnvironmentData')
]

for i in range(num_messages):
    for msg_def in messages:
        # Generate random data for each signal
        data = {}
        for signal in msg_def.signals:
            data[signal.name] = generate_signal_value(signal)
        
        # Debug: Print signal values
        print(f"Message: {msg_def.name}, Signals: {data}")
        
        # Encode the message
        try:
            encoded_data = msg_def.encode(data)
            # Debug: Print encoded data
            print(f"Encoded data (hex): {encoded_data.hex()}")
        except Exception as e:
            print(f"Error encoding message {msg_def.name}: {e}")
            print(f"Signal values: {data}")
            continue
        
        # Create a CAN message
        can_msg = can.Message(
            arbitration_id=msg_def.frame_id,
            data=encoded_data,
            is_extended_id=False,
            timestamp=start_time + (i * 0.01)  # 10ms interval
        )
        
        # Write to BLF file
        try:
            writer.on_message_received(can_msg)
        except Exception as e:
            print(f"Error writing CAN message to BLF: {e}")
            continue

# Close the BLF file
writer.stop()

print(f"BLF file generated: {blf_file}")