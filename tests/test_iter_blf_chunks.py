"""
Module: tests/test_iter_blf_chunks.py

This test suite verifies the behavior of the `iter_blf_chunks` function in the
`canml.canmlio` module using pytest. It ensures proper streaming of BLF messages
into DataFrame chunks, filtering by ID and signal, chunk sizing, error handling,
and reader cleanup.

Test Cases:
  - Nonexistent BLF path raises FileNotFoundError
  - Wrong extension raises FileNotFoundError
  - Empty reader yields no chunks
  - Single chunk grouping works
  - Chunk splitting by chunk_size
  - filter_ids filters messages by arbitration ID
  - filter_signals filters decoded signal keys
  - stop() exceptions are suppressed on reader close
  - progress_bar toggle does not affect output

Best Practices:
  - Uses pytest tmp_path for dummy BLF files
  - Monkeypatches BLFReader to inject dummy messages
  - Uses a minimal fake DB with decode_message
  - Verifies DataFrame contents and shapes
"""

import pytest
import pandas as pd
import numpy as np
import canml.canmlio as canmlio
from canml.canmlio import iter_blf_chunks, CanmlConfig

# Dummy message and reader classes
class DummyMsg:
    def __init__(self, arbitration_id, data, timestamp):
        self.arbitration_id = arbitration_id
        self.data = data
        self.timestamp = timestamp

class DummyReader:
    msgs = []
    def __init__(self, path):
        # path is ignored
        pass
    def __iter__(self):
        return iter(self.__class__.msgs)
    def stop(self):
        # optionally overridden in tests
        pass

class DummyDB:
    def decode_message(self, arbitration_id, data):
        # echo arbitration_id for testing
        return {"sig": arbitration_id, "val": data}


@pytest.fixture(autouse=True)
def patch_reader(monkeypatch):
    """Monkeypatch BLFReader to use DummyReader."""
    monkeypatch.setattr(canmlio, "BLFReader", DummyReader)
    yield
    DummyReader.msgs = []

@pytest.fixture
def blf_file(tmp_path):
    """Create an empty .blf file for path validation."""
    f = tmp_path / "test.blf"
    f.write_bytes(b"")  # content is irrelevant
    return str(f)

def test_nonexistent_path():
    cfg = CanmlConfig(progress_bar=False)
    with pytest.raises(FileNotFoundError):
        list(iter_blf_chunks("no.such.blf", DummyDB(), cfg))

def test_wrong_extension(tmp_path):
    txt = tmp_path / "file.txt"
    txt.write_text("")
    cfg = CanmlConfig(progress_bar=False)
    with pytest.raises(FileNotFoundError):
        list(iter_blf_chunks(str(txt), DummyDB(), cfg))

def test_empty_reader_yields_no_chunks(blf_file):
    cfg = CanmlConfig(progress_bar=False)
    DummyReader.msgs = []
    chunks = list(iter_blf_chunks(blf_file, DummyDB(), cfg))
    assert chunks == []

def test_single_chunk(blf_file):
    cfg = CanmlConfig(chunk_size=10, progress_bar=False)
    # two messages < chunk_size => single chunk
    DummyReader.msgs = [
        DummyMsg(1, 100, 0.1),
        DummyMsg(2, 200, 0.2),
    ]
    chunks = list(iter_blf_chunks(blf_file, DummyDB(), cfg))
    assert len(chunks) == 1
    df = chunks[0]
    # should have 2 rows, columns 'sig','val','timestamp'
    assert list(df['sig']) == [1, 2]
    assert list(df['val']) == [100, 200]
    assert np.allclose(df['timestamp'], [0.1, 0.2])

def test_chunk_splitting(blf_file):
    cfg = CanmlConfig(chunk_size=1, progress_bar=False)
    DummyReader.msgs = [
        DummyMsg(10, 1, 0.0),
        DummyMsg(20, 2, 0.1),
        DummyMsg(30, 3, 0.2),
    ]
    chunks = list(iter_blf_chunks(blf_file, DummyDB(), cfg))
    # 3 messages, chunk_size=1 => 3 chunks
    assert len(chunks) == 3
    # each DataFrame should have exactly one row
    assert all(len(df) == 1 for df in chunks)

def test_filter_ids(blf_file):
    cfg = CanmlConfig(progress_bar=False)
    DummyReader.msgs = [
        DummyMsg(5, 50, 0.5),
        DummyMsg(6, 60, 0.6),
    ]
    # only include arbitration_id=6
    chunks = list(iter_blf_chunks(blf_file, DummyDB(), cfg, filter_ids={6}))
    df = pd.concat(chunks, ignore_index=True)
    assert list(df['sig']) == [6]
    assert list(df['timestamp']) == [0.6]

def test_filter_signals(blf_file):
    cfg = CanmlConfig(progress_bar=False)
    # DB decode returns two keys; filter_signals picks only 'val'
    class TwoFieldDB(DummyDB):
        def decode_message(self, arbitration_id, data):
            return {"a": arbitration_id, "b": data}
    DummyReader.msgs = [DummyMsg(7, 70, 0.7)]
    chunks = list(iter_blf_chunks(blf_file, TwoFieldDB(), cfg, None, filter_signals={"b"}))
    df = chunks[0]
    assert 'a' not in df.columns
    assert 'b' in df.columns
    assert df.at[0, 'b'] == 70

def test_stop_exception_suppressed(blf_file):
    cfg = CanmlConfig(progress_bar=False)
    # make stop() raise
    def bad_stop(self): raise RuntimeError("stop fail")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(DummyReader, "stop", bad_stop)
    DummyReader.msgs = [DummyMsg(8, 80, 0.8)]
    # should still yield data and not raise
    chunks = list(iter_blf_chunks(blf_file, DummyDB(), cfg))
    assert len(chunks) == 1
    df = chunks[0]
    assert df.at[0, 'sig'] == 8
    monkeypatch.undo()

def test_progress_bar_toggle(blf_file):
    # progress_bar=True vs False should both yield same results
    DummyReader.msgs = [DummyMsg(9, 90, 0.9)]
    cfg1 = CanmlConfig(progress_bar=True)
    cfg2 = CanmlConfig(progress_bar=False)
    chunks1 = list(iter_blf_chunks(blf_file, DummyDB(), cfg1))
    chunks2 = list(iter_blf_chunks(blf_file, DummyDB(), cfg2))
    df1 = pd.concat(chunks1, ignore_index=True)
    df2 = pd.concat(chunks2, ignore_index=True)
    pd.testing.assert_frame_equal(df1, df2)
