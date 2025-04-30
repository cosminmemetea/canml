"""
Module: tests/test_load_dbc_files.py

This test suite verifies the behavior of the `load_dbc_files` function in the
`canml.canmlio` module using pytest. It ensures correct merging of DBC files,
error handling for missing or invalid files, prefixing signals, collision detection,
file extension validation, and LRU caching.

Test Cases:
  - Single valid DBC path loads correctly and caches result
  - Multiple valid DBC paths merge messages
  - Nonexistent DBC path raises FileNotFoundError
  - Non-DBC file extension raises ValueError
  - Empty DBC paths raises ValueError
  - ParseError in add_dbc_file raises ValueError with parse message
  - Other exceptions in add_dbc_file raises ValueError with invalid message
  - Duplicate signal names without prefix raises ValueError
  - Duplicate message names with prefix_signals=True raises ValueError
  - prefix_signals renames signal names with message prefixes
  - LRU cache reuses parsed DBCs for identical inputs
  - LRU cache does not reuse for different inputs
  - Logging captures DBC loading at debug level

Best Practices:
  - Uses pytest tmp_path fixture for temporary files
  - Monkeypatches CantoolsDatabase for dependency isolation
  - Verifies error messages, states, and logging output
  - Tests caching behavior with LRU cache
  - Uses caplog for logging verification

Prerequisites:
  pip install pytest cantools

To execute:
    pytest tests/test_load_dbc_files.py -v
"""
import pytest
import logging
from pathlib import Path
import cantools.database.errors as db_errors

import canml.canmlio as canmlio

class FakeDB:
    """Fake CantoolsDatabase substitute recording DBC additions and holding messages."""
    def __init__(self):
        self.added = []
        self.messages = []

    def add_dbc_file(self, path):
        self.added.append(path)


@pytest.fixture(autouse=True)
def setup_logging(caplog):
    """Configure logging to capture debug messages."""
    caplog.set_level(logging.DEBUG, logger=canmlio.__name__)
    yield
    # Reset to INFO after test
    canmlio.glogger.setLevel(logging.INFO)


@pytest.fixture(autouse=True)
def disable_and_clear_cache(monkeypatch):
    """Prevent actual CantoolsDatabase usage and clear LRU cache before each test."""
    monkeypatch.setattr(canmlio, "CantoolsDatabase", FakeDB)
    # Clear cache on internal loader
    canmlio._load_dbc_files_cached.cache_clear()
    yield
    canmlio._load_dbc_files_cached.cache_clear()


def test_single_valid(tmp_path):
    """Loading a single valid DBC file should record one add_dbc_file call."""
    p = tmp_path / "a.dbc"
    p.write_text('VERSION "1"')
    db = canmlio.load_dbc_files(str(p))
    assert isinstance(db, FakeDB)
    assert db.added == [str(p)]


def test_multiple_valid(tmp_path):
    """Loading multiple DBCs merges calls in order."""
    p1 = tmp_path / "x.dbc"
    p2 = tmp_path / "y.dbc"
    p1.write_text('VERSION "1"')
    p2.write_text('VERSION "2"')
    db = canmlio.load_dbc_files([str(p1), str(p2)])
    assert isinstance(db, FakeDB)
    assert db.added == [str(p1), str(p2)]


def test_missing_path(tmp_path):
    """Missing DBC path should raise FileNotFoundError."""
    missing = tmp_path / "no.dbc"
    with pytest.raises(FileNotFoundError):
        canmlio.load_dbc_files(str(missing))


def test_non_dbc_extension(tmp_path):
    """Non-DBC file extension should raise ValueError."""
    txt = tmp_path / "file.txt"
    txt.write_text("dummy")
    with pytest.raises(ValueError) as excinfo:
        canmlio.load_dbc_files(str(txt))
    assert "is not a .dbc file" in str(excinfo.value)


def test_empty_paths():
    """Empty DBC paths should raise ValueError."""
    with pytest.raises(ValueError) as excinfo:
        canmlio.load_dbc_files([])
    assert "At least one DBC file must be provided" in str(excinfo.value)


def test_parse_error_wrapped(tmp_path, monkeypatch):
    """ParseError in add_dbc_file should raise ValueError with parse message."""
    p = tmp_path / "bad.dbc"
    p.write_text("")
    def bad_add(self, path):
        raise db_errors.ParseError("bad format")
    monkeypatch.setattr(FakeDB, 'add_dbc_file', bad_add)
    with pytest.raises(ValueError) as excinfo:
        canmlio.load_dbc_files(str(p))
    assert f"Invalid DBC format in {p}" in str(excinfo.value)


def test_other_exception_wrapped(tmp_path, monkeypatch):
    """Generic errors in add_dbc_file should wrap in ValueError invalid message."""
    p = tmp_path / "bad2.dbc"
    p.write_text("")
    def bad_add(self, path):
        raise RuntimeError("oops")
    monkeypatch.setattr(FakeDB, 'add_dbc_file', bad_add)
    with pytest.raises(ValueError) as excinfo:
        canmlio.load_dbc_files(str(p))
    assert f"Invalid DBC file {p}" in str(excinfo.value)


def test_duplicate_signals_without_prefix(tmp_path, monkeypatch):
    """Duplicate signal names without prefix_signals should raise ValueError."""
    p = tmp_path / "d.dbc"
    p.write_text("VERSION\n")
    class Msg:
        def __init__(self, name):
            self.name = name
            self.signals = [type('Sig', (), {'name': 'SIG'})]
    fake = FakeDB()
    fake.messages = [Msg('M1'), Msg('M2')]
    monkeypatch.setattr(canmlio, 'CantoolsDatabase', lambda: fake)
    canmlio._load_dbc_files_cached.cache_clear()
    with pytest.raises(ValueError) as excinfo:
        canmlio.load_dbc_files(str(p), prefix_signals=False)
    assert "Duplicate signal names" in str(excinfo.value)


def test_duplicate_message_names_with_prefix(tmp_path):
    """Duplicate message names with prefix_signals=True should raise ValueError."""
    p = tmp_path / "d.dbc"
    p.write_text("VERSION\n")
    class Msg:
        def __init__(self, name):
            self.name = name
            self.signals = []
    fake = FakeDB()
    fake.messages = [Msg('X'), Msg('X')]
    canmlio.CantoolsDatabase = lambda: fake
    canmlio._load_dbc_files_cached.cache_clear()
    with pytest.raises(ValueError) as excinfo:
        canmlio.load_dbc_files(str(p), prefix_signals=True)
    assert "Duplicate message names" in str(excinfo.value)


def test_prefix_signals_renames(tmp_path):
    """prefix_signals=True should rename signals to Message_Signal."""
    p = tmp_path / "d.dbc"
    p.write_text("VERSION\n")
    class Sig:
        def __init__(self, name): self.name = name
    class Msg:
        def __init__(self, name):
            self.name = name
            self.signals = [Sig('A'), Sig('B')]
    fake = FakeDB()
    fake.messages = [Msg('M')]
    canmlio.CantoolsDatabase = lambda: fake
    canmlio._load_dbc_files_cached.cache_clear()
    db = canmlio.load_dbc_files(str(p), prefix_signals=True)
    sigs = [s.name for s in db.messages[0].signals]
    assert sigs == ['M_A', 'M_B']


def test_lru_cache_reuse(tmp_path):
    """LRU cache should return the same object for identical inputs."""
    p = tmp_path / "c.dbc"
    p.write_text("VERSION\n")
    db1 = canmlio.load_dbc_files(str(p))
    db2 = canmlio.load_dbc_files(str(p))
    assert db1 is db2


def test_lru_cache_different(tmp_path):
    """LRU cache should not reuse for different input sets."""
    p1 = tmp_path / "c1.dbc"
    p2 = tmp_path / "c2.dbc"
    p1.write_text("VERSION\n")
    p2.write_text("VERSION\n")
    db1 = canmlio.load_dbc_files(str(p1))
    db2 = canmlio.load_dbc_files(str(p2))
    assert db1 is not db2


def test_logging_debug_level(tmp_path, caplog):
    """Debug logs should capture each DBC load path."""
    caplog.set_level(logging.DEBUG, logger=canmlio.__name__)
    p1 = tmp_path / "x.dbc"
    p1.write_text("VERSION\n")
    canmlio.load_dbc_files(str(p1))
    assert f"Loading DBC: {p1}" in caplog.text
