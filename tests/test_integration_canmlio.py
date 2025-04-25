import unittest
import os
import sys
import pandas as pd
import cantools
import can
from can.io.blf import BLFWriter
import logging

# Ensure project root is on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from canml.canmlio import load_blf, to_csv

class TestCanmlioIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tests_dir = os.path.dirname(os.path.abspath(__file__))
        cls.dbc = os.path.join(cls.tests_dir, "test.dbc")
        cls.blfs = [os.path.join(cls.tests_dir, f"output-{i}.blf") for i in range(3)]

        # Verify resource existence
        for path in [cls.dbc] + cls.blfs:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")

        # Prepare logging capture
        h = logging.StreamHandler()
        h.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(h)
        cls.handler = h

    @classmethod
    def tearDownClass(cls):
        logging.getLogger('').removeHandler(cls.handler)

    def test_single_file_to_csv(self):
        df = load_blf(self.blfs[0], self.dbc, force_uniform_timing=True)
        output = os.path.join(self.tests_dir, "single_decoded.csv")
        try:
            to_csv(df, output)
            self.assertTrue(os.path.exists(output))
            csv_df = pd.read_csv(output)
            self.assertEqual(len(csv_df), len(df))
            self.assertListEqual(list(csv_df.columns), list(df.columns))
        finally:
            os.remove(output)

    def test_multiple_files_to_csv(self):
        dfs = [load_blf(b, self.dbc, force_uniform_timing=True) for b in self.blfs]
        combined = pd.concat(dfs, ignore_index=True)
        output = os.path.join(self.tests_dir, "multiple_decoded.csv")
        try:
            to_csv(combined, output)
            self.assertTrue(os.path.exists(output))
            csv_df = pd.read_csv(output)
            self.assertEqual(len(csv_df), len(combined))
            self.assertListEqual(list(csv_df.columns), list(combined.columns))
        finally:
            pass
            # os.remove(output)

if __name__ == "__main__":
    unittest.main()
