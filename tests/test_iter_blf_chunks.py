"""
Module: tests/test_iter_blf_chunks.py

This test suite verifies the behavior of the `iter_blf_chunks` function
in the `canml.canmlio` module. It uses Python’s built-in `unittest` framework
to ensure correct FileNotFoundError handling, chunking behavior, filtering,
skip-on-error logic, and proper resource cleanup.

Test Cases:
  - BLF file missing raises FileNotFoundError
  - Correct chunking based on chunk_size (including final partial chunk)
  - Filtering by arbitration IDs via filter_ids
  - Skipping messages that raise DecodeError or KeyError
  - Ensuring BLFReader.stop() is called

To execute:
    python -m unittest tests/test_iter_blf_chunks.py
"""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from canml.canmlio import iter_blf_chunks

# Dummy message to simulate BLFReader outputs
def create_dummy_msgs(count):
    """Generate a list of DummyMsg instances with sequential IDs and timestamps."""
    class DummyMsg:
        """Represents a CAN message record."""
        def __init__(self, arbitration_id, data, timestamp):
            self.arbitration_id = arbitration_id
            self.data = data
            self.timestamp = timestamp
    return [DummyMsg(i, b"", i * 0.1) for i in range(count)]

class DummyReader:
    """
    Simulates can.io.blf.BLFReader interface:
      - Iterable of messages
      - stop() method marks as stopped
    """
    def __init__(self, messages):
        self._messages = list(messages)
        self.stopped = False
    def __iter__(self):
        return iter(self._messages)
    def stop(self):
        self.stopped = True

class FakeDatabase:
    """
    Stub for cantools Database to control decode_message behavior.

    decode_map: dict mapping arbitration_id -> decoded dict or Exception
    """
    def __init__(self, decode_map=None):
        self.decode_map = decode_map or {}
    def decode_message(self, arbitration_id, data):
        """
        Returns a dict if mapping exists, else raises KeyError or propagates Exception.
        """
        result = self.decode_map.get(arbitration_id)
        if isinstance(result, Exception):
            raise result
        if result is None:
            # mimic cantools DecodeError
            from cantools.database.errors import DecodeError
            raise DecodeError(f"Cannot decode ID {arbitration_id}")
        return result

class TestIterBlfChunks(unittest.TestCase):
    """
    Tests for iter_blf_chunks: file checks, chunking, filtering, error skipping, and cleanup.
    """
    def setUp(self):
        # Prepare a dummy BLF file path
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.blf_path = Path(self.tempdir.name) / 'test.blf'
        self.blf_path.write_text('dummy')  # content irrelevant for dummy reader

    def test_missing_file_raises(self):
        """
        Non-existent BLF path should raise FileNotFoundError immediately.
        """
        missing = Path(self.tempdir.name) / 'no.blf'
        with self.assertRaises(FileNotFoundError):
            _ = list(iter_blf_chunks(str(missing), db=FakeDatabase(), chunk_size=1))

    def test_chunking_and_stop(self):
        """
        Verify chunk sizes and that stop() is called on reader.
        """
        msgs = create_dummy_msgs(5)
        fake_db = FakeDatabase({i: {'val': i} for i in range(5)})
        reader = DummyReader(msgs)
        with patch('canml.canmlio.BLFReader', return_value=reader):
            chunks = list(iter_blf_chunks(str(self.blf_path), db=fake_db, chunk_size=2))
        # Should produce 3 chunks: sizes 2,2,1
        self.assertEqual([len(c) for c in chunks], [2, 2, 1])
        # Reader.stop() must be invoked
        self.assertTrue(reader.stopped)
        # Verify data integrity
        df0 = chunks[0]
        self.assertListEqual(df0['val'].tolist(), [0, 1])
        self.assertListEqual([round(x, 1) for x in df0['timestamp']], [0.0, 0.1])

    def test_filter_ids(self):
        """
        Only messages with arbitration_id in filter_ids should be decoded.
        """
        msgs = create_dummy_msgs(3)
        # Only id 1 and 2 have mapping
        fake_db = FakeDatabase({1: {'A': 10}, 2: {'A': 20}})
        reader = DummyReader(msgs)
        with patch('canml.canmlio.BLFReader', return_value=reader):
            chunk = next(iter_blf_chunks(str(self.blf_path), db=fake_db, chunk_size=10, filter_ids={1, 2}))
        self.assertListEqual(chunk['A'].tolist(), [10, 20])
        # Ensure timestamp field preserved
        self.assertListEqual([round(t,1) for t in chunk['timestamp']], [0.1, 0.2])

    def test_skip_errors(self):
        """
        Messages causing DecodeError or KeyError are skipped, valid ones included.
        """
        from cantools.database.errors import DecodeError
        # msg 0 -> KeyError, msg1 -> DecodeError, msg2 -> valid
        msgs = [create_dummy_msgs(1)[0] for _ in range(3)]
        # override arbitration IDs
        msgs[0].arbitration_id = 0
        msgs[1].arbitration_id = 1
        msgs[2].arbitration_id = 2
        msgs[0].timestamp = 0.0
        msgs[1].timestamp = 0.1
        msgs[2].timestamp = 0.2
        fake_db = FakeDatabase({0: KeyError('x'), 1: DecodeError('y'), 2: {'VAL': 42}})
        reader = DummyReader(msgs)
        with patch('canml.canmlio.BLFReader', return_value=reader):
            chunk = next(iter_blf_chunks(str(self.blf_path), db=fake_db, chunk_size=5))
        self.assertListEqual(chunk['VAL'].tolist(), [42])
        self.assertListEqual([round(t,1) for t in chunk['timestamp']], [0.2])

if __name__ == '__main__':
    unittest.main()
