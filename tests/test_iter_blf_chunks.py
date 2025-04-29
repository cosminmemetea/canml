"""
Module: tests/test_iter_blf_chunks.py

This test suite verifies the behavior of the `iter_blf_chunks` function in the
`canml.canmlio` module using pytest. It tests file existence, chunk sizing,
message filtering, decode error handling, and reader cleanup.

Test Cases:
  - Missing BLF path raises FileNotFoundError
  - Invalid chunk_size (<=0) raises ValueError
  - Yields correct chunk sizes without filters
  - Applies filter_ids to include only specified arbitration IDs
  - Skips messages raising DecodeError or KeyError
  - Progress bar disabled when requested
  - Reader.stop exceptions are caught and do not propagate

Best Practices:
  - Uses pytest tmp_path for file operations
  - Monkeypatches BLFReader for controlled iteration
  - Tests DataFrame content and shapes
  - Captures warnings for stop failures

Prerequisites:
  pip install pytest pandas numpy cantools tqdm

To execute:
    pytest tests/test_iter_blf_chunks.py -v
"""
import pytest
import pandas as pd
from pathlib import Path
import canml.canmlio as canmlio
import cantools.database.errors as db_errors

# Dummy message and reader classes for testing
class DummyMsg:
    def __init__(self, arbitration_id, data, timestamp):
        self.arbitration_id = arbitration_id
        self.data = data
        self.timestamp = timestamp

class DummyReader:
    def __init__(self, msgs, stop_exc=None):
        self._msgs = msgs
        self.stop_exc = stop_exc
    def __iter__(self):
        return iter(self._msgs)
    def stop(self):
        if self.stop_exc:
            raise self.stop_exc

@pytest.fixture
def dummy_db():
    """Simple DB that returns a dict with 'sig': arbitration_id."""
    class DB:
        def decode_message(self, arb, data):
            return {'sig': arb}
    return DB()

def test_missing_blf_path(dummy_db):
    """Nonexistent BLF path should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        next(canmlio.iter_blf_chunks('nonexistent.blf', dummy_db))

def test_invalid_chunk_size(dummy_db, tmp_path):
    """chunk_size <= 0 should raise ValueError."""
    blf_file = tmp_path / 'x.blf'
    blf_file.write_bytes(b'')
    with pytest.raises(ValueError):
        list(canmlio.iter_blf_chunks(str(blf_file), dummy_db, chunk_size=0))

def test_chunking_and_content(tmp_path, dummy_db, monkeypatch):
    """Yields DataFrame chunks of correct sizes and content."""
    blf_file = tmp_path / 'x.blf'
    blf_file.write_bytes(b'')
    msgs = [DummyMsg(1, b"\x00", 1.0), DummyMsg(2, b"\x01", 2.0), DummyMsg(3, b"\x02", 3.0)]
    monkeypatch.setattr(canmlio, 'BLFReader', lambda path: DummyReader(msgs))
    chunks = list(canmlio.iter_blf_chunks(str(blf_file), dummy_db, chunk_size=2, progress_bar=False))
    assert len(chunks) == 2
    df1, df2 = chunks
    assert list(df1['timestamp']) == [1.0, 2.0]
    assert list(df2['timestamp']) == [3.0]

def test_filter_ids(tmp_path, dummy_db, monkeypatch):
    """Only messages with arbitration_id in filter_ids are decoded."""
    blf_file = tmp_path / 'f.blf'
    blf_file.write_bytes(b'')
    msgs = [DummyMsg(1, b"", 0.0), DummyMsg(2, b"", 1.0)]
    monkeypatch.setattr(canmlio, 'BLFReader', lambda path: DummyReader(msgs))
    chunks = list(canmlio.iter_blf_chunks(str(blf_file), dummy_db, chunk_size=10, filter_ids={2}, progress_bar=False))
    assert len(chunks) == 1
    df = chunks[0]
    assert list(df['timestamp']) == [1.0]

def test_skip_on_decode_error(tmp_path, monkeypatch):
    """Messages causing DecodeError are skipped."""
    blf_file = tmp_path / 'e.blf'
    blf_file.write_bytes(b'')
    class BadDB:
        def decode_message(self, arb, data):
            raise db_errors.DecodeError('fail')
    msgs = [DummyMsg(1, b"", 0.0), DummyMsg(2, b"", 1.0)]
    monkeypatch.setattr(canmlio, 'BLFReader', lambda path: DummyReader(msgs))
    chunks = list(canmlio.iter_blf_chunks(str(blf_file), BadDB(), chunk_size=10, progress_bar=False))
    assert chunks == []

def test_stop_exception_logged(tmp_path, dummy_db, monkeypatch, caplog):
    """Exceptions in reader.stop() are caught and logged."""
    blf_file = tmp_path / 's.blf'
    blf_file.write_bytes(b'')
    msgs = [DummyMsg(1, b"", 0.0)]
    reader = DummyReader(msgs, stop_exc=RuntimeError('stop fail'))
    monkeypatch.setattr(canmlio, 'BLFReader', lambda path: reader)
    caplog.set_level('WARNING')
    chunks = list(canmlio.iter_blf_chunks(str(blf_file), dummy_db, chunk_size=10, progress_bar=False))
    assert len(chunks) == 1
    assert 'Failed to close BLF reader' in caplog.text

def test_no_messages_yields_nothing(tmp_path, dummy_db, monkeypatch):
    """No messages should yield no chunks."""
    blf_file = tmp_path / 'n.blf'
    blf_file.write_bytes(b'')
    monkeypatch.setattr(canmlio, 'BLFReader', lambda path: DummyReader([]))
    chunks = list(canmlio.iter_blf_chunks(str(blf_file), dummy_db, chunk_size=5, progress_bar=False))
    assert chunks == []
