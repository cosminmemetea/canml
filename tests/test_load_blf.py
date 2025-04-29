"""
Module: tests/test_load_blf.py

This test suite verifies the behavior of the `load_blf` function in the `canml.canmlio` module
using pytest. It ensures correct handling of BLF and DBC files, Database instance vs. path
inputs, chunk concatenation, uniform timing, signal injection with dtype control, timestamp
sorting, column ordering, and logger configuration.

Test Cases:
  - Missing BLF path raises ValueError
  - Missing DBC path raises FileNotFoundError
  - Empty DBC paths raises ValueError
  - Empty BLF file returns empty DataFrame with correct schema
  - Database instance skips load_dbc_files
  - DBC path string/list calls load_dbc_files
  - Empty message_ids returns empty DataFrame with warning about no data
  - Chunk concatenation from iter_blf_chunks
  - force_uniform_timing with interval_seconds
  - sort_timestamps sorts non-monotonic timestamps
  - expected_signals injection with dtype_map (int32)
  - Duplicate expected_signals raises ValueError
  - Invalid dtype_map raises ValueError
  - Invalid interval_seconds raises ValueError
  - Column order with timestamp first
  - Logger configuration (StreamHandler, INFO level, format)
  - Chunk iteration errors raise ValueError

Best Practices:
  - Uses pytest fixtures for temporary files and mocks
  - Parametrizes tests for input variations
  - Captures logs with caplog
  - Mocks cantools dependencies for portability
  - Organized with clear docstrings and comments
  - Correct imports for canml.canmlio package structure

Prerequisites:
  - Install dependencies: pip install pytest pandas numpy cantools tqdm pytest-cov
  - Ensure project root is in PYTHONPATH: export PYTHONPATH=$PWD
  - Alternatively, install canml as editable: pip install -e .

To execute:
    pytest tests/test_load_blf.py -v
To run with coverage:
    pytest --cov=canml.canmlio tests/test_load_blf.py
"""

import pytest
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from cantools.database.can import Database as CantoolsDatabase
import canml.canmlio as canmlio


@pytest.fixture
def dummy_db(tmp_path, monkeypatch):
    """Provide a dummy CantoolsDatabase and mock load_dbc_files."""
    db = CantoolsDatabase()
    monkeypatch.setattr(db, 'add_dbc_file', lambda x: None)
    return db


@pytest.fixture
def sample_blf(tmp_path, monkeypatch):
    """Create a minimal BLF file and patch BLFReader to yield no messages."""
    blf = tmp_path / "empty.blf"
    blf.write_bytes(b"")
    class DummyReader:
        def __init__(self, path): pass
        def __iter__(self): return iter([])
        def stop(self): pass
    monkeypatch.setattr(canmlio, 'BLFReader', DummyReader)
    return str(blf)


def test_missing_blf_path(dummy_db):
    """Missing BLF path should raise ValueError wrapping FileNotFoundError."""
    with pytest.raises(ValueError) as excinfo:
        canmlio.load_blf('nonexistent.blf', dummy_db)
    assert 'Failed to process BLF data' in str(excinfo.value)


def test_missing_dbc_path(tmp_path):
    """Missing DBC path string should raise FileNotFoundError from load_dbc_files."""
    blf = tmp_path / 'file.blf'
    blf.write_bytes(b'')
    with pytest.raises(FileNotFoundError):
        canmlio.load_blf(str(blf), 'nonexistent.dbc')


def test_empty_dbc_paths():
    """Empty DBC paths list should raise ValueError in load_dbc_files."""
    with pytest.raises(ValueError):
        canmlio.load_dbc_files([])


def test_empty_blf_returns_empty_df(dummy_db, sample_blf, caplog):
    """Empty BLF yields empty DataFrame with correct schema and warning."""
    caplog.set_level('WARNING')
    df = canmlio.load_blf(sample_blf, dummy_db, expected_signals=['A', 'B'])
    assert list(df.columns) == ['timestamp', 'A', 'B']
    assert df.empty
    assert 'No data decoded' in caplog.text


def test_db_instance_skips_load_dbc_files(dummy_db, sample_blf, monkeypatch):
    """Pass CantoolsDatabase instance; should not call load_dbc_files."""
    monkeypatch.setattr(canmlio, 'load_dbc_files', lambda x: (_ for _ in ()).throw(AssertionError("Should not call")))
    df = canmlio.load_blf(sample_blf, dummy_db)
    assert isinstance(df, pd.DataFrame)


def test_dbc_path_invokes_load_dbc_files(tmp_path, sample_blf, monkeypatch, dummy_db):
    """Passing DBC path should call load_dbc_files exactly once."""
    dbc = tmp_path / 'd.dbc'
    dbc.write_text('')
    called = {'count': 0}
    def fake_load(paths):
        called['count'] += 1
        return dummy_db
    monkeypatch.setattr(canmlio, 'load_dbc_files', fake_load)
    df = canmlio.load_blf(sample_blf, str(dbc))
    assert called['count'] == 1
    assert isinstance(df, pd.DataFrame)


def test_empty_message_ids_returns_empty(dummy_db, sample_blf, caplog):
    """Empty message_ids yields empty DataFrame and 'No data decoded' warning."""
    caplog.set_level('WARNING')
    df = canmlio.load_blf(sample_blf, dummy_db, message_ids=set())
    assert df.empty
    assert 'No data decoded' in caplog.text


def test_chunk_concatenation(monkeypatch, dummy_db):
    """Chunks from iter_blf_chunks are concatenated into a single DataFrame."""
    monkeypatch.setattr(canmlio, 'iter_blf_chunks', lambda *args, **kwargs: iter([
        pd.DataFrame([{'timestamp': 1, 'v': 10}]),
        pd.DataFrame([{'timestamp': 2, 'v': 20}])
    ]))
    df = canmlio.load_blf('p.blf', dummy_db)
    assert len(df) == 2
    assert list(df['timestamp']) == [1, 2]
    assert list(df['v']) == [10, 20]


def test_force_uniform_timing_and_interval(dummy_db, sample_blf, monkeypatch):
    """force_uniform_timing transforms timestamps by interval_seconds."""
    monkeypatch.setattr(canmlio, 'iter_blf_chunks', lambda *args, **kwargs: iter([
        pd.DataFrame([{'timestamp': 0.1, 'x': 1}, {'timestamp': 0.2, 'x': 2}]),
        pd.DataFrame([{'timestamp': 0.3, 'x': 3}])
    ]))
    df = canmlio.load_blf('path.blf', dummy_db, force_uniform_timing=True, interval_seconds=0.5)
    assert 'raw_timestamp' in df.columns
    # uniform timestamps: 0.0, 0.5, 1.0
    assert np.allclose(df['timestamp'], [0.0, 0.5, 1.0])


def test_sort_timestamps(monkeypatch, dummy_db):
    """sort_timestamps=True sorts non-monotonic timestamps."""
    monkeypatch.setattr(canmlio, 'iter_blf_chunks', lambda *args, **kwargs: iter([
        pd.DataFrame([{'timestamp': 2, 'a': 1}, {'timestamp': 1, 'a': 2}])
    ]))
    df = canmlio.load_blf('p.blf', dummy_db, sort_timestamps=True)
    assert list(df['timestamp']) == [1, 2]


def test_expected_signals_injection_with_dtype(dummy_db, sample_blf, monkeypatch, caplog):
    """Missing expected_signals are injected with dtype from dtype_map."""
    monkeypatch.setattr(canmlio, 'iter_blf_chunks', lambda *args, **kwargs: iter([
        pd.DataFrame([{'timestamp': 0}])
    ]))
    caplog.set_level('WARNING')
    df = canmlio.load_blf(sample_blf, dummy_db, expected_signals=['S1'], dtype_map={'S1': 'int32'})
    assert 'S1' in df.columns
    # dtype_map specified int32, so injected column should be int32
    assert df['S1'].dtype == np.dtype('int32')
    assert "Expected signal 'S1' not found" in caplog.text


def test_duplicate_expected_signals_raises():
    """Duplicate names in expected_signals should raise ValueError."""
    with pytest.raises(ValueError):
        canmlio.load_blf('p.blf', 'd.dbc', expected_signals=['X','X'])


def test_invalid_dtype_map_raises(dummy_db, sample_blf):
    """Invalid dtype in dtype_map should raise ValueError."""
    with pytest.raises(ValueError):
        canmlio.load_blf(sample_blf, dummy_db, expected_signals=['E'], dtype_map={'E': 'not_a_dtype'})


def test_invalid_interval_seconds_value(dummy_db, sample_blf):
    """interval_seconds <= 0 with force_uniform_timing should raise ValueError."""
    with pytest.raises(ValueError):
        canmlio.load_blf(sample_blf, dummy_db, force_uniform_timing=True, interval_seconds=0)


def test_column_order_timestamp_first(monkeypatch, dummy_db):
    """Result DataFrame columns start with timestamp."""
    monkeypatch.setattr(canmlio, 'iter_blf_chunks', lambda *args, **kwargs: iter([
        pd.DataFrame([{'timestamp': 1, 'b': 2, 'a': 3}])
    ]))
    df = canmlio.load_blf('f.blf', dummy_db)
    assert df.columns[0] == 'timestamp'


def test_logger_configuration():
    """Module logger should have a single StreamHandler at INFO level and proper format."""
    logger = canmlio.glogger
    handlers = logger.handlers
    assert len(handlers) == 1
    handler = handlers[0]
    assert handler.level in (logging.NOTSET, logging.INFO)
    assert handler.formatter._fmt == "%(asctime)s [%(levelname)s] %(message)s"


def test_iter_blf_chunks_error_handling(monkeypatch, tmp_path, dummy_db):
    """Exceptions in iter_blf_chunks should raise ValueError."""
    class BadReader:
        def __init__(self, path): pass
        def __iter__(self): raise RuntimeError("boom")
        def stop(self): raise RuntimeError("stop fail")
    monkeypatch.setattr(canmlio, 'BLFReader', BadReader)
    p = tmp_path / 't.blf'
    p.write_bytes(b'')
    with pytest.raises(ValueError) as excinfo:
        canmlio.load_blf(str(p), dummy_db)
    assert 'Failed to process BLF data' in str(excinfo.value)
