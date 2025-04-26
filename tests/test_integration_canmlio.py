import unittest
import os
import pandas as pd
import logging

from canml import load_blf, to_csv
from can.io.blf import BLFWriter
from canml.canmlio import load_dbc_files, to_parquet, iter_blf_chunks  # used to regenerate fixtures if needed

class TestCanmlioIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Directory containing test fixtures
        cls.tests_dir = os.path.dirname(os.path.abspath(__file__))
        cls.dbc = os.path.join(cls.tests_dir, "test.dbc")
        cls.blfs = [os.path.join(cls.tests_dir, f"output-{i}.blf") for i in range(3)]

        # Verify fixtures exist
        for path in [cls.dbc] + cls.blfs:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required fixture not found: {path}")

        # Capture logs
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(handler)
        cls.log_handler = handler

    @classmethod
    def tearDownClass(cls):
        logging.getLogger('').removeHandler(cls.log_handler)

    def test_load_dbc_files(self):
        # Ensure DBC loading works and returns Database
        db = load_dbc_files(self.dbc)
        self.assertTrue(hasattr(db, 'messages'))
        self.assertGreater(len(db.messages), 0)

    def test_iter_blf_chunks(self):
        # Test chunked decoding and message filtering
        db = load_dbc_files(self.dbc)
        chunks = list(iter_blf_chunks(self.blfs[0], db, chunk_size=5,
                                       filter_ids={db.get_message_by_name('EngineData').frame_id}))
        self.assertTrue(len(chunks) >= 1)
        for df in chunks:
            self.assertIsInstance(df, pd.DataFrame)
            # Only EngineData frames
            self.assertIn('EngineRPM', df.columns)
            break

    def test_load_blf_with_expected(self):
        # Load full BLF and inject expected signal
        expected = ['EngineData_EngineRPM', 'NonExistentSignal']
        df = load_blf(
            blf_path=self.blfs[0],
            db=self.dbc,
            expected_signals=expected,
            force_uniform_timing=True
        )
        # Check injection
        self.assertIn('NonExistentSignal', df.columns)
        self.assertTrue(df['NonExistentSignal'].isna().all())

    def test_to_csv_and_parquet(self):
        # Roundtrip CSV and Parquet
        df = load_blf(self.blfs[0], self.dbc, force_uniform_timing=True)
        csv_out = os.path.join(self.tests_dir, 'int.csv')
        pq_out = os.path.join(self.tests_dir, 'int.parquet')
        try:
            to_csv(df, csv_out)
            self.assertTrue(os.path.exists(csv_out))
            df_csv = pd.read_csv(csv_out)
            pd.testing.assert_frame_equal(df.reset_index(drop=True), df_csv)

            to_parquet(df, pq_out)
            self.assertTrue(os.path.exists(pq_out))
            df_pq = pd.read_parquet(pq_out)
            pd.testing.assert_frame_equal(df.reset_index(drop=True), df_pq)
        finally:
            for f in (csv_out, pq_out):
                if os.path.exists(f): os.remove(f)

if __name__ == '__main__':
    unittest.main()
