"""
Module: tests/test_load_blf.py

This test suite verifies the behavior of the `load_blf` function
in the `canml.canmlio` module. It uses Python’s built-in `unittest` framework
to ensure correct file-not-found behavior, Database instance handling,
how DBC paths are loaded, chunk concatenation, uniform timing,
expected signal injection, and column ordering.

Test Cases:
  - Missing BLF path raises FileNotFoundError
  - Missing DBC path raises FileNotFoundError
  - Passing a Database instance skips load_dbc_files
  - Passing a DBC path string uses load_dbc_files internally
  - Concatenation of DataFrame chunks from iter_blf_chunks
  - force_uniform_timing overwrites timestamps correctly
  - expected_signals injection fills missing columns with NaN
  - Order of columns: timestamp first

To execute:
    python -m unittest tests/test_load_blf.py
"""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
from cantools.database import Database
from canml.canmlio import load_blf

class TestLoadBlf(unittest.TestCase):
    """
    TestCase covering load_blf functionality.
    """
    def setUp(self):
        # Set up temporary directory with BLF and DBC dummy files
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.blf_path = Path(self.tempdir.name) / 'test.blf'
        self.blf_path.write_text('dummy')
        self.dbc_path = Path(self.tempdir.name) / 'test.dbc'
        self.dbc_path.write_text('dummy')

    def test_missing_blf_raises(self):
        """
        Non-existent BLF path should raise FileNotFoundError,
        even when a Database instance is provided.
        """
        missing_blf = Path(self.tempdir.name) / 'no.blf'
        fake_db = Database()
        with self.assertRaises(FileNotFoundError):
            load_blf(str(missing_blf), fake_db)

    def test_missing_dbc_raises(self):
        """
        Non-existent DBC path string should raise FileNotFoundError.
        """
        missing_dbc = Path(self.tempdir.name) / 'no.dbc'
        with self.assertRaises(FileNotFoundError):
            load_blf(str(self.blf_path), str(missing_dbc))

    def test_pass_database_instance_skips_load_dbc(self):
        """
        Passing an existing Database instance should skip load_dbc_files.
        """
        db_inst = Database()
        # Ensure load_dbc_files is not called
        with patch('canml.canmlio.load_dbc_files') as mock_load:
            # Provide no chunks, expect empty DataFrame
            with patch('canml.canmlio.iter_blf_chunks', return_value=[]):
                df = load_blf(str(self.blf_path), db_inst)
        mock_load.assert_not_called()
        self.assertTrue(df.empty)

    def test_dbc_path_string_calls_load_dbc_files(self):
        """
        Passing a DBC path string should invoke load_dbc_files internally.
        """
        fake_db = MagicMock()
        with patch('canml.canmlio.load_dbc_files', return_value=fake_db) as mock_load:
            with patch('canml.canmlio.iter_blf_chunks', return_value=[]):
                df = load_blf(str(self.blf_path), str(self.dbc_path))
        mock_load.assert_called_once_with(str(self.dbc_path))
        self.assertTrue(df.empty)

    def test_concatenate_chunks(self):
        """
        DataFrame chunks from iter_blf_chunks should concatenate correctly.
        """
        c1 = pd.DataFrame({'timestamp': [0.0, 0.1], 'A': [1, 2]})
        c2 = pd.DataFrame({'timestamp': [0.2], 'A': [3]})
        fake_db = MagicMock()
        with patch('canml.canmlio.iter_blf_chunks', return_value=[c1, c2]):
            df = load_blf(str(self.blf_path), fake_db)
        expected = pd.concat([c1, c2], ignore_index=True)
        pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)

    def test_force_uniform_timing(self):
        """
        With force_uniform_timing=True, timestamps should be uniformly spaced.
        """
        c = pd.DataFrame({'timestamp': [5.0, 6.0, 7.0], 'X': [10, 20, 30]})
        fake_db = MagicMock()
        with patch('canml.canmlio.iter_blf_chunks', return_value=[c]):
            df = load_blf(str(self.blf_path), fake_db, force_uniform_timing=True, interval_seconds=0.5)
        self.assertListEqual(list(df['timestamp']), [0.0, 0.5, 1.0])

    def test_expected_signals_injection(self):
        """
        expected_signals should be injected as NaN if missing.
        """
        c = pd.DataFrame({'timestamp': [0], 'A': [100]})
        fake_db = MagicMock()
        with patch('canml.canmlio.iter_blf_chunks', return_value=[c]):
            df = load_blf(str(self.blf_path), fake_db, expected_signals=['A', 'B', 'C'])
        self.assertIn('B', df.columns)
        self.assertIn('C', df.columns)
        self.assertTrue(pd.isna(df.loc[0, 'B']))
        self.assertTrue(pd.isna(df.loc[0, 'C']))
        self.assertEqual(df.loc[0, 'A'], 100)

    def test_column_order_timestamp_first(self):
        """
        The 'timestamp' column should always appear first.
        """
        c = pd.DataFrame({'X': [1], 'timestamp': [0]})
        fake_db = MagicMock()
        with patch('canml.canmlio.iter_blf_chunks', return_value=[c]):
            df = load_blf(str(self.blf_path), fake_db)
        self.assertEqual(list(df.columns)[0], 'timestamp')

if __name__ == '__main__':
    unittest.main()
