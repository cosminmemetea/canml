import unittest
import os
import tempfile
import pandas as pd
import numpy as np
import cantools
import can
from can.io.blf import BLFWriter
from canml.canmlio import (
    load_dbc_files,
    iter_blf_chunks,
    load_blf,
    to_csv,
    to_parquet
)

class TestCanmlioAdvanced(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Paths
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        cls.dbc_path = os.path.join(tests_dir, 'test.dbc')
        if not os.path.exists(cls.dbc_path):
            raise FileNotFoundError(f"DBC fixture not found: {cls.dbc_path}")
        cls.db = cantools.database.load_file(cls.dbc_path)
        # Standard message definitions
        cls.msg_eng = cls.db.get_message_by_name('EngineData')
        cls.msg_brk = cls.db.get_message_by_name('BrakeStatus')

    def test_load_dbc_files_prefix_signals(self):
        # Load same DBC twice to force merge, prefix signals
        db_merged = load_dbc_files([self.dbc_path, self.dbc_path], prefix_signals=True)
        for msg in db_merged.messages:
            for sig in msg.signals:
                self.assertTrue(sig.name.startswith(msg.name + '_'),
                                f"Signal {sig.name} not prefixed by {msg.name}_")

    def test_iter_blf_chunks_and_filtering(self):
        # Create small BLF with two frames
        fd, blf_file = tempfile.mkstemp(suffix='.blf')
        os.close(fd)
        with BLFWriter(blf_file, channel=1) as w:
            # Write one EngineData
            data1 = {s.name: (0 if s.scale==1 else 0.0) for s in self.msg_eng.signals}
            enc1 = self.msg_eng.encode(data1)
            w.on_message_received(can.Message(arbitration_id=self.msg_eng.frame_id,
                                             data=enc1, timestamp=0))
            # Write one BrakeStatus
            data2 = {s.name: (0 if s.scale==1 else 0.0) for s in self.msg_brk.signals}
            enc2 = self.msg_brk.encode(data2)
            w.on_message_received(can.Message(arbitration_id=self.msg_brk.frame_id,
                                             data=enc2, timestamp=0.01))
        try:
            # Filter for only EngineData and chunk size=1
            chunks = list(iter_blf_chunks(blf_file, self.db, chunk_size=1,
                                          filter_ids={self.msg_eng.frame_id}))
            # Expect one chunk with one row
            self.assertEqual(len(chunks), 1)
            df0 = chunks[0]
            self.assertEqual(len(df0), 1)
            # Only EngineData signals present
            self.assertIn('EngineRPM', df0.columns)
            self.assertNotIn('BrakePressure', df0.columns)
        finally:
            os.remove(blf_file)

    def test_load_blf_expected_and_message_filter(self):
        # Create BLF with one EngineData frame
        fd, blf_file = tempfile.mkstemp(suffix='.blf')
        os.close(fd)
        with BLFWriter(blf_file, channel=1) as w:
            data = {s.name: 1 for s in self.msg_eng.signals}
            enc = self.msg_eng.encode(data)
            w.on_message_received(can.Message(arbitration_id=self.msg_eng.frame_id,
                                             data=enc, timestamp=0))
        try:
            # Only EngineData, but expect a BrakeStatus signal injected
            df = load_blf(
                blf_path=blf_file,
                db=self.db,
                message_ids={self.msg_eng.frame_id},
                expected_signals=['BrakeStatus_ABSActive']
            )
            # EngineData signals present
            self.assertIn('EngineRPM', df.columns)
            # Expected BrakeStatus signal injected
            self.assertIn('BrakeStatus_ABSActive', df.columns)
            # Values should be NaN
            self.assertTrue(df['BrakeStatus_ABSActive'].isna().all())
        finally:
            os.remove(blf_file)

    def test_to_parquet_roundtrip(self):
        # Simple DataFrame
        df = pd.DataFrame({
            'timestamp': [0, 0.01, 0.02],
            'foo': [1, 2, 3]
        })
        fd, pq_file = tempfile.mkstemp(suffix='.parquet')
        os.close(fd)
        try:
            to_parquet(df, pq_file)
            df2 = pd.read_parquet(pq_file)
            pd.testing.assert_frame_equal(df, df2)
        finally:
            os.remove(pq_file)

if __name__ == '__main__':
    unittest.main()
