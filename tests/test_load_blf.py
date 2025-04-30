"""
Module: tests/test_load_blf.py

This test suite verifies the behavior of the `load_blf` function in the
`canml.canmlio` module using pytest. It ensures correct handling of BLF files,
DBC inputs, chunk concatenation, uniform timing, signal injection with dtype control,
filtering, metadata, and error handling.

Test Cases:
  - Missing BLF path raises FileNotFoundError
  - Missing DBC path raises FileNotFoundError
  - Empty DBC paths raises ValueError (via load_dbc_files)
  - Empty BLF returns empty DataFrame with expected columns and warning
  - Database instance skips load_dbc_files
  - DBC path string/list invokes load_dbc_files
  - Empty message_ids yields empty DataFrame with warning
  - Chunk concatenation from iter_blf_chunks
  - filter_ids filters messages by ID
  - filter_signals filters decoded fields
  - force_uniform_timing transforms timestamps and preserves raw_timestamp
  - sort_timestamps sorts
  - expected_signals injection preserves integer dtype
  - duplicate expected_signals raises ValueError
  - invalid dtype_map signal raises ValueError
  - invalid interval_seconds raises ValueError
  - timestamp is first column
  - metadata_attrs appear in DataFrame attrs
  - error in iter_blf_chunks propagates as ValueError

Best Practices:
  - Uses pytest fixtures and monkeypatch
  - Captures warnings and logs

Prerequisites:
  pip install pytest pandas numpy cantools tqdm

To execute:
    pytest tests/test_load_blf.py -v
"""
import pytest
import pandas as pd
import numpy as np
import logging

import canml.canmlio as canmlio
from canml.canmlio import load_blf, load_dbc_files, iter_blf_chunks, CanmlConfig
from cantools.database.can import Database as CantoolsDatabase


@pytest.fixture(autouse=True)
def setup_logging(caplog):
    """Capture warnings and info logs from canmlio."""
    caplog.set_level(logging.WARNING, logger=canmlio.__name__)
    yield
    canmlio.glogger.setLevel(logging.INFO)


@pytest.fixture
def dummy_db(monkeypatch):
    """Provide a dummy CantoolsDatabase instance for tests."""
    db = CantoolsDatabase()
    monkeypatch.setattr(canmlio, 'load_dbc_files', lambda x: db)
    return db


@pytest.fixture
def sample_blf(tmp_path, monkeypatch):
    """Create empty BLF and patch BLFReader to yield no messages."""
    f = tmp_path / 'empty.blf'
    f.write_bytes(b'')
    class DummyReader:
        def __init__(self, path): pass
        def __iter__(self): return iter([])
        def stop(self): pass
    monkeypatch.setattr(canmlio, 'BLFReader', DummyReader)
    return str(f)


def test_missing_blf_path(dummy_db):
    """Nonexistent BLF path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_blf('no.blf', dummy_db)


def test_missing_dbc_path(tmp_path):
    """Nonexistent DBC path raises FileNotFoundError."""
    blf = tmp_path / 'a.blf'
    blf.write_bytes(b'')
    with pytest.raises(FileNotFoundError):
        load_blf(str(blf), 'missing.dbc')


def test_empty_dbc_paths():
    """Empty DBC list raises ValueError via load_dbc_files."""
    with pytest.raises(ValueError):
        load_dbc_files([])


def test_empty_blf_returns_empty(sample_blf, dummy_db, caplog):
    """Empty BLF yields empty DF with timestamp+expected columns and warning."""
    cfg = CanmlConfig()
    caplog.set_level(logging.WARNING, logger=canmlio.__name__)
    df = load_blf(sample_blf, dummy_db, config=cfg, expected_signals=['A','B'])
    assert df.empty
    assert list(df.columns) == ['timestamp','A','B']
    assert 'No data decoded' in caplog.text


def test_db_instance_skips_load_dbc_files(sample_blf, dummy_db, monkeypatch):
    """Passing DB instance should not call load_dbc_files."""
    monkeypatch.setattr(canmlio, 'load_dbc_files', lambda x: (_ for _ in ()).throw(AssertionError()))
    df = load_blf(sample_blf, dummy_db)
    assert isinstance(df, pd.DataFrame)


def test_dbc_path_invokes_load_dbc_files(sample_blf, tmp_path, monkeypatch, dummy_db):
    """Passing path invokes load_dbc_files once."""
    dbc = tmp_path / 'd.dbc'
    dbc.write_text('')
    count = {'n':0}
    monkeypatch.setattr(canmlio, 'load_dbc_files', lambda p: (_ for _ in ()).throw(AssertionError()) if count['n'] else (count.update(n=1) or dummy_db))
    df = load_blf(sample_blf, str(dbc))
    assert isinstance(df, pd.DataFrame)


def test_empty_message_ids_warn(sample_blf, dummy_db, caplog):
    """Empty message_ids yields empty DF with warning."""
    cfg = CanmlConfig()
    caplog.set_level(logging.WARNING, logger=canmlio.__name__)
    df = load_blf(sample_blf, dummy_db, config=cfg, message_ids=set())
    assert df.empty and 'Empty message_ids' in caplog.text


def test_chunk_concat_and_filter(monkeypatch, dummy_db):
    """Chunks concatenated and filter_ids works."""
    def dummy_chunks(path, db, config, fids, fsigs):
        yield pd.DataFrame([{'x':1,'timestamp':0.1}])
        yield pd.DataFrame([{'x':2,'timestamp':0.2}])
    monkeypatch.setattr(canmlio, 'iter_blf_chunks', dummy_chunks)
    cfg = CanmlConfig()
    df = load_blf('p.blf', dummy_db, config=cfg, message_ids={1}, expected_signals=['x'])
    assert list(df['x']) == [1,2]


def test_filter_signals(monkeypatch, dummy_db):
    """filter_signals drops unwanted keys."""
    def dummy_chunks(path, db, config, fids, fsigs):
        yield pd.DataFrame([{'a':1,'b':2,'timestamp':0}])
    monkeypatch.setattr(canmlio, 'iter_blf_chunks', dummy_chunks)
    cfg = CanmlConfig()
    df = load_blf('p.blf', dummy_db, config=cfg, expected_signals=['b'])
    assert 'a' not in df.columns and 'b' in df.columns


def test_uniform_timing_and_raw(monkeypatch, dummy_db):
    """force_uniform_timing resets timestamp and adds raw_timestamp."""
    def dummy_chunks(*args, **kwargs):
        yield pd.DataFrame([{'timestamp':0.1,'v':1},{'timestamp':0.5,'v':2}])
    monkeypatch.setattr(canmlio, 'iter_blf_chunks', dummy_chunks)
    cfg = CanmlConfig(force_uniform_timing=True, interval_seconds=0.5)
    df = load_blf('p.blf', dummy_db, config=cfg)
    assert 'raw_timestamp' in df.columns
    assert np.allclose(df['timestamp'], [0.0,0.5])


def test_sort_timestamps(monkeypatch, dummy_db):
    """sort_timestamps orders timestamps."""
    def dummy_chunks(*args, **kwargs): yield pd.DataFrame([{'timestamp':2},{'timestamp':1}])
    monkeypatch.setattr(canmlio, 'iter_blf_chunks', dummy_chunks)
    cfg = CanmlConfig(sort_timestamps=True)
    df = load_blf('p.blf', dummy_db, config=cfg)
    assert list(df['timestamp']) == [1,2]


def test_expected_signals_and_dtype(monkeypatch, dummy_db):
    """Missing signal injected with correct dtype int32."""
    def dummy_chunks(*args, **kwargs): yield pd.DataFrame([{'timestamp':0}])
    monkeypatch.setattr(canmlio, 'iter_blf_chunks', dummy_chunks)
    cfg = CanmlConfig(dtype_map={'S':'int32'})
    df = load_blf('p.blf', dummy_db, config=cfg, expected_signals=['S'])
    assert df['S'].dtype == np.dtype('int32')


def test_duplicate_expected_signals():
    """Duplicate expected_signals raises ValueError before DB load."""
    with pytest.raises(ValueError):
        load_blf('p.blf', 'd.dbc', config=CanmlConfig(), expected_signals=['X','X'])


def test_invalid_dtype_map(dummy_db):
    """dtype_map unknown signal raises ValueError."""
    with pytest.raises(ValueError):
        load_blf('p.blf', dummy_db, config=CanmlConfig(dtype_map={'E':'float'}), expected_signals=['A'])


def test_invalid_interval_values():
    """interval_seconds<=0 raises in config."""
    with pytest.raises(ValueError):
        CanmlConfig(interval_seconds=0)


def test_timestamp_first(monkeypatch, dummy_db):
    """timestamp column is first in result."""
    def dummy_chunks(*args, **kwargs): yield pd.DataFrame([{'timestamp':1,'b':2,'a':3}])
    monkeypatch.setattr(canmlio, 'iter_blf_chunks', dummy_chunks)
    cfg = CanmlConfig()
    df = load_blf('p.blf', dummy_db, config=cfg)
    assert df.columns[0] == 'timestamp'


def test_metadata_attrs(monkeypatch, tmp_path, sample_blf):
    """DataFrame attrs contain signal_attributes metadata."""
    # Fake DB and signals
    class SigObj:
        def __init__(self,name): self.name=name; self.attributes={'u':'m'}; self.choices=None
    class Msg:
        def __init__(self,sigs): self.signals=sigs
    fake_db = type('FakeDB', (), {})()
    fake_db.messages = [Msg([SigObj('a'), SigObj('b')])]
    monkeypatch.setattr(canmlio, 'load_dbc_files', lambda x: fake_db)
    monkeypatch.setattr(canmlio, 'iter_blf_chunks', lambda *args, **kwargs: iter([pd.DataFrame([{'timestamp':0,'a':1,'b':2}])]))
    cfg = CanmlConfig()
    df = load_blf(sample_blf, 'dummy.dbc', config=cfg)
    assert 'signal_attributes' in df.attrs
    assert 'a' in df.attrs['signal_attributes']


def test_iter_error_propagates(monkeypatch, dummy_db):
    """Exceptions in iter_blf_chunks raise ValueError."""
    monkeypatch.setattr(canmlio, 'iter_blf_chunks', lambda *a,**k: (_ for _ in ()).throw(RuntimeError('boom')))
    cfg = CanmlConfig()
    with pytest.raises(ValueError):
        load_blf('p.blf', dummy_db, config=cfg)
