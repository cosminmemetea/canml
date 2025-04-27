"""
Module: tests/test_to_csv.py

This test suite verifies the behavior of the `to_csv` function
in the `canml.canmlio` module. It uses Python’s built-in `unittest`
framework to ensure proper CSV export for DataFrames and iterators,
as well as error propagation on write failures.

Test Cases:
  - Single DataFrame writes and reads back correctly
  - Iterable of DataFrame chunks appends rows properly
  - Header appears only once when writing chunks
  - Underlying write errors raise ValueError with appropriate message

To execute:
    python -m unittest tests/test_to_csv.py
"""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import canml.canmlio as ci

class TestToCsv(unittest.TestCase):
    """
    TestCase covering to_csv functionality.
    """
    def setUp(self):
        # Temporary directory and output path
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.outfile = Path(self.tempdir.name) / 'out.csv'

    def test_single_dataframe_writes_and_reads_back(self):
        """
        to_csv should write a DataFrame to CSV which can be read back identically.
        """
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        ci.to_csv(df, str(self.outfile))
        # File must exist
        self.assertTrue(self.outfile.exists())
        # Read back and compare
        df2 = pd.read_csv(self.outfile)
        pd.testing.assert_frame_equal(df, df2)

    def test_iterable_chunks_writes_append(self):
        """
        to_csv should accept an iterator of DataFrames and append chunks.
        """
        df1 = pd.DataFrame({'A': [1, 2], 'B': ['a', 'b']})
        df2 = pd.DataFrame({'A': [3], 'B': ['c']})
        ci.to_csv(iter([df1, df2]), str(self.outfile))
        # Read back all rows
        df_all = pd.read_csv(self.outfile)
        expected = pd.concat([df1, df2], ignore_index=True)
        pd.testing.assert_frame_equal(expected, df_all)

    def test_header_written_once(self):
        """
        Header row should appear only once at the top when writing multiple chunks.
        """
        df1 = pd.DataFrame({'C': [0]})
        df2 = pd.DataFrame({'C': [1]})
        ci.to_csv(iter([df1, df2]), str(self.outfile))
        lines = self.outfile.read_text().splitlines()
        # Count occurrences of header line
        header = 'C'
        header_count = sum(1 for line in lines if line.split(',')[0] == header)
        self.assertEqual(header_count, 1)

    def test_error_propagates_as_value_error(self):
        """
        If DataFrame.to_csv raises, to_csv should catch and re-raise ValueError.
        """
        df = pd.DataFrame({'A': [1]})
        with patch.object(pd.DataFrame, 'to_csv', side_effect=Exception('fail')):
            with self.assertRaises(ValueError) as cm:
                ci.to_csv(df, str(self.outfile))
        self.assertIn('Failed to export CSV', str(cm.exception))

if __name__ == '__main__':
    unittest.main()
