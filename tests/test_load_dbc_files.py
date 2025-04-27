"""
Module: tests/test_load_dbc_files.py

This test suite verifies the behavior of the `load_dbc_files` function
in the `canml.canmlio` module. It uses Python's built-in `unittest`
framework with a fake database stub to isolate file-loading logic,
error handling, and optional signal name prefixing.

Test Cases:
  - File not found raises FileNotFoundError
  - Invalid DBC loading raises ValueError
  - Single and multiple DBC file loading calls add_dbc_file correctly
  - prefix_signals=True renames all signals by prefixing with message names
  - prefix_signals=False leaves signal names unchanged

To execute:
    python -m unittest tests/test_load_dbc_files.py
"""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import canml.canmlio as ci
from canml.canmlio import load_dbc_files


def create_dummy_messages(names, signals):
    """
    Generate dummy message instances for testing.

    Each dummy message has:
      - name: the message name
      - signals: list of DummySignal objects initialized with given signal names

    Returns a list of DummyMsg instances.
    """
    class DummySignal:
        """Represents a CAN signal with a mutable name attribute."""
        def __init__(self, name):
            self.name = name

    class DummyMsg:
        """Represents a CAN message holding multiple DummySignal objects."""
        def __init__(self, name, signals):
            self.name = name
            self.signals = [DummySignal(sig) for sig in signals]

    return [DummyMsg(n, signals) for n in names]


class FakeDatabase:
    """
    Stub implementation of cantools.database.Database for load_dbc_files tests.

    Attributes:
      - messages: injected dummy messages for prefixing tests
      - added_paths: record of DBC file paths passed to add_dbc_file
    """
    def __init__(self, dummy_messages=None):
        self.messages = dummy_messages or []
        self.added_paths = []

    def add_dbc_file(self, path):
        """
        Simulate loading a DBC file by recording its path.
        """
        self.added_paths.append(path)


class TestLoadDbcFiles(unittest.TestCase):
    """
    TestCase covering all behaviors of load_dbc_files:
      - File existence checks
      - Invalid DBC handling
      - Single/multiple file loading
      - Signal prefixing logic
    """

    def setUp(self):
        """
        Prepare a temporary DBC file for existence validation.
        """
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.real_dbc = Path(self.tempdir.name) / 'one.dbc'
        # Write a dummy file to satisfy file existence
        self.real_dbc.write_text('dummy')

    def test_missing_file_raises(self):
        """
        Passing a non-existent path should raise FileNotFoundError.
        """
        missing = Path(self.tempdir.name) / 'absent.dbc'
        with self.assertRaises(FileNotFoundError):
            load_dbc_files(str(missing))

    def test_invalid_dbc_raises_value(self):
        """
        Simulate an invalid DBC by patching add_dbc_file to throw,
        expecting a ValueError from load_dbc_files.
        """
        with patch('canml.canmlio.Database', return_value=ci.Database()):
            with patch.object(ci.Database, 'add_dbc_file', side_effect=Exception('invalid')):
                with self.assertRaises(ValueError):
                    load_dbc_files(str(self.real_dbc))

    def test_single_and_multiple_dbc_loading(self):
        """
        Validate that add_dbc_file is called exactly once for a single path,
        and once per path for multiple inputs, returning the stub DB.
        """
        fake_db = FakeDatabase()
        with patch('canml.canmlio.Database', return_value=fake_db):
            # Single-file load
            db1 = load_dbc_files(str(self.real_dbc), prefix_signals=False)
            self.assertIs(db1, fake_db)
            self.assertEqual(fake_db.added_paths, [str(self.real_dbc)])

            # Multi-file load
            second = Path(self.tempdir.name) / 'two.dbc'
            second.write_text('x')
            fake_db.added_paths.clear()
            db2 = load_dbc_files([str(self.real_dbc), str(second)], prefix_signals=False)
            self.assertIs(db2, fake_db)
            self.assertEqual(fake_db.added_paths, [str(self.real_dbc), str(second)])

    def test_prefix_signals_true_renames_all(self):
        """
        With prefix_signals=True, each signal name should be
        '<MessageName>_<OriginalSignalName>'.

        Uses 12 dummy messages and 3 signals each to ensure full coverage.
        """
        names = [f'MSG{i}' for i in range(1, 13)]
        signals = ['A', 'B', 'C']
        dummy_msgs = create_dummy_messages(names, signals)
        fake_db = FakeDatabase(dummy_messages=dummy_msgs)

        with patch('canml.canmlio.Database', return_value=fake_db):
            db = load_dbc_files(str(self.real_dbc), prefix_signals=True)
        self.assertIs(db, fake_db)

        for msg in dummy_msgs:
            for orig, sig in zip(signals, msg.signals):
                expected = f'{msg.name}_{orig}'
                self.assertEqual(
                    sig.name, expected,
                    f"Signal rename mismatch: expected {expected}, got {sig.name}"
                )

    def test_prefix_signals_false_keeps_original(self):
        """
        With prefix_signals=False, signal names remain unmodified.
        """
        names = ['Only']
        signals = ['X', 'Y', 'Z']
        dummy_msgs = create_dummy_messages(names, signals)
        fake_db = FakeDatabase(dummy_messages=dummy_msgs)

        with patch('canml.canmlio.Database', return_value=fake_db):
            db = load_dbc_files(str(self.real_dbc), prefix_signals=False)
        self.assertIs(db, fake_db)

        for orig, sig in zip(signals, dummy_msgs[0].signals):
            self.assertEqual(
                sig.name, orig,
                f"Unexpected rename: signal {orig} became {sig.name}"
            )


if __name__ == '__main__':
    unittest.main()
