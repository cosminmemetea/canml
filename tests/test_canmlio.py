import unittest
import os
import sys
import pandas as pd
import cantools
import can
from can.io.blf import BLFWriter
import logging

# Adjust sys.path to include the project_root for importing canmlio
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from canmlio import canmlio

class TestCanmlio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Paths to test files
        cls.dbc_file = os.path.join("tests", "test.dbc")
        cls.blf_file = os.path.join("tests", "output-2.blf")
        
        # Ensure test files exist
        if not os.path.exists(cls.dbc_file):
            raise FileNotFoundError(f"DBC file {cls.dbc_file} not found")
        if not os.path.exists(cls.blf_file):
            raise FileNotFoundError(f"BLF file {cls.blf_file} not found")
        
        # Load DBC for reference
        cls.db = cantools.database.load_file(cls.dbc_file)
        
        # Expected message IDs and signals from test.dbc
        cls.expected_signals = {
            "EngineData": ["EngineRPM", "ThrottlePosition", "CoolantTemp"],
            "VehicleDynamics": ["VehicleSpeed", "GearPosition", "TransmissionTemp"],
            "BrakeStatus": ["BrakePressure", "ABSActive", "WheelSpeedFL"],
            "EnvironmentData": ["AmbientTemp", "CabinTemp", "RainfallRate"]
        }
        
        # Signal ranges from test.dbc
        cls.signal_ranges = {
            "EngineRPM": (0, 8000),
            "ThrottlePosition": (0, 100),
            "CoolantTemp": (-40, 215),
            "VehicleSpeed": (0, 300),
            "GearPosition": (0, 8),
            "TransmissionTemp": (-40, 200),
            "BrakePressure": (0, 1000),
            "ABSActive": (0, 1),
            "WheelSpeedFL": (0, 300),
            "AmbientTemp": (-40, 60),
            "CabinTemp": (-40, 60),
            "RainfallRate": (0, 25.5)  # 8-bit limit (255 * 0.1)
        }
        
        # Set up logging capture
        cls.log_capture = logging.StreamHandler()
        cls.log_capture.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(cls.log_capture)
        
    @classmethod
    def tearDownClass(cls):
        # Remove logging handler
        logging.getLogger('').removeHandler(cls.log_capture)

    def test_load_blf(self):
        """Test that load_blf reads BLF file and returns a valid DataFrame."""
        df = canmlio.load_blf(self.blf_file, self.dbc_file)
        
        # Check DataFrame is not empty
        self.assertFalse(df.empty, "DataFrame is empty")
        
        # Check expected columns
        expected_columns = ['timestamp'] + [
            signal for signals in self.expected_signals.values() for signal in signals
        ]
        self.assertTrue(set(expected_columns).issubset(df.columns), 
                        f"Missing columns: {set(expected_columns) - set(df.columns)}")
        
        # Check signal values are within ranges
        for signal, (min_val, max_val) in self.signal_ranges.items():
            if signal in df.columns:
                values = df[signal].dropna()
                self.assertTrue(values.between(min_val, max_val).all(),
                               f"{signal} values out of range [{min_val}, {max_val}]")

    def test_load_blf_timestamps(self):
        """Test that timestamps are correctly loaded and in ascending order."""
        df = canmlio.load_blf(self.blf_file, self.dbc_file)
        
        # Check timestamp column exists
        self.assertIn("timestamp", df.columns, "Timestamp column missing")
        
        # Check timestamps are in ascending order
        timestamps = df["timestamp"]
        self.assertTrue(timestamps.is_monotonic_increasing,
                        "Timestamps are not in ascending order")
        
        # Check timestamp range (100 messages * 4 types * 10ms = 4s)
        delta = timestamps.max() - timestamps.min()
        self.assertAlmostEqual(delta, 4.0, delta=0.1,
                              msg="Timestamp range incorrect")

    def test_to_csv(self):
        """Test that to_csv exports DataFrame to CSV correctly."""
        df = canmlio.load_blf(self.blf_file, self.dbc_file)
        csv_file = os.path.join("tests", "test_output.csv")
        
        try:
            canmlio.to_csv(df, csv_file)
            self.assertTrue(os.path.exists(csv_file), "CSV file not created")
            
            # Read CSV and verify contents
            csv_df = pd.read_csv(csv_file)
            self.assertEqual(list(df.columns), list(csv_df.columns),
                            "CSV columns do not match DataFrame")
            self.assertEqual(len(df), len(csv_df),
                            "CSV row count does not match DataFrame")
        finally:
            if os.path.exists(csv_file):
                os.remove(csv_file)

    def test_empty_blf_file(self):
        """Test load_blf with an empty BLF file."""
        empty_blf = os.path.join("tests", "empty.blf")
        
        # Create empty BLF file
        with BLFWriter(empty_blf, channel=1) as writer:
            pass
        
        try:
            df = canmlio.load_blf(empty_blf, self.dbc_file)
            self.assertTrue(df.empty, "Expected empty DataFrame for empty BLF")
        finally:
            if os.path.exists(empty_blf):
                os.remove(empty_blf)

    def test_invalid_dbc_file(self):
        """Test load_blf with an invalid DBC file."""
        invalid_dbc = os.path.join("tests", "invalid.dbc")
        
        # Create invalid DBC file
        with open(invalid_dbc, "w") as f:
            f.write("INVALID DBC CONTENT")
        
        try:
            with self.assertRaises(ValueError, msg="Expected ValueError for invalid DBC"):
                canmlio.load_blf(self.blf_file, invalid_dbc)
        finally:
            if os.path.exists(invalid_dbc):
                os.remove(invalid_dbc)

    def test_missing_blf_file(self):
        """Test load_blf with a non-existent BLF file."""
        with self.assertRaises(FileNotFoundError, msg="Expected FileNotFoundError for missing BLF"):
            canmlio.load_blf(os.path.join("tests", "nonexistent.blf"), self.dbc_file)

    def test_empty_dataframe_to_csv(self):
        """Test to_csv with an empty DataFrame."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError, msg="Expected ValueError for empty DataFrame"):
            canmlio.to_csv(empty_df, os.path.join("tests", "empty.csv"))

    def test_undecodable_messages(self):
        """Test load_blf with a BLF file containing undecodable messages."""
        custom_blf = os.path.join("tests", "custom.blf")
        
        # Create BLF with one valid and one undecodable message
        with BLFWriter(custom_blf, channel=1) as writer:
            # Valid EngineData message
            msg_def = self.db.get_message_by_name("EngineData")
            data = {"EngineRPM": 5000, "ThrottlePosition": 50, "CoolantTemp": 80}
            encoded_data = msg_def.encode(data)
            valid_msg = can.Message(
                arbitration_id=msg_def.frame_id,
                data=encoded_data,
                is_extended_id=False,
                timestamp=0
            )
            writer.on_message_received(valid_msg)
            
            # Undecodable message (invalid ID)
            invalid_msg = can.Message(
                arbitration_id=999,  # Not in DBC
                data=[0x00] * 8,
                is_extended_id=False,
                timestamp=0.01
            )
            writer.on_message_received(invalid_msg)
        
        try:
            df = canmlio.load_blf(custom_blf, self.dbc_file)
            self.assertEqual(len(df), 1, "Expected one valid message in DataFrame")
            self.assertIn("EngineRPM", df.columns, "Expected EngineRPM in DataFrame")
            self.assertEqual(df["EngineRPM"].iloc[0], 5000, "Incorrect EngineRPM value")
        finally:
            if os.path.exists(custom_blf):
                os.remove(custom_blf)

if __name__ == "__main__":
    unittest.main()
