"""
Module: tests/test_to_parquet.py

This test suite verifies the behavior of the `to_parquet` function
in the `canml.canmlio` module. It uses Python’s built-in `unittest`
framework to ensure proper Parquet export for DataFrames,
including file existence, content correctness, and error handling.

Test Cases:
  - Single DataFrame writes and reads back correctly using pyarrow
  - Custom compression argument is accepted and results in a file
  - Underlying write errors raise ValueError with appropriate message

To execute:
    python -m unittest tests/test_to_parquet.py
"""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import canml.canmlio as ci

class TestToParquet(unittest.TestCase):
    """
    TestCase covering to_parquet functionality.
    """
    def setUp(self):
        # Temporary directory and output path
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.outfile = Path(self.tempdir.name) / 'out.parquet'

    def test_single_dataframe_writes_and_reads_back(self):
        """
        to_parquet should write a DataFrame that can be read back identically.
        """
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        # default compression
        ci.to_parquet(df, str(self.outfile))
        # File must exist
        self.assertTrue(self.outfile.exists())
        # Read back and compare
        df2 = pd.read_parquet(self.outfile)
        pd.testing.assert_frame_equal(df, df2)

    def test_custom_compression_writes_file(self):
        """
        to_parquet should accept a custom compression codec.
        """
        df = pd.DataFrame({'X': [10, 20]})
        ci.to_parquet(df, str(self.outfile), compression='gzip')
        self.assertTrue(self.outfile.exists())
        # read back to ensure file integrity
        df2 = pd.read_parquet(self.outfile)
        pd.testing.assert_frame_equal(df, df2)

    def test_error_propagates_as_value_error(self):
        """
        If DataFrame.to_parquet raises, to_parquet should catch and re-raise ValueError.
        """
        df = pd.DataFrame({'A': [1]})
        with patch.object(pd.DataFrame, 'to_parquet', side_effect=Exception('fail')):
            with self.assertRaises(ValueError) as cm:
                ci.to_parquet(df, str(self.outfile))
        self.assertIn('Failed to export Parquet', str(cm.exception))

if __name__ == '__main__':
    unittest.main()
