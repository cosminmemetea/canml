"""
Module: tests/test_load_dbc_files.py

This test suite verifies the behavior of the `load_dbc_files` function in the
`canml.canmlio` module using pytest. It ensures correct merging of DBC files,
error handling for missing or invalid files, prefixing signals, and collision detection.

Test Cases:
  - Single valid DBC path loads correctly
  - Multiple valid DBC paths merge messages
  - Nonexistent DBC path raises FileNotFoundError
  - Empty DBC paths raises ValueError
  - ParseError in add_dbc_file raises ValueError with parse message
  - Other exceptions in add_dbc_file raises ValueError invalid message
  - Duplicate signal names without prefix raises ValueError
  - Duplicate message names with prefix_signals=True raises ValueError
  - prefix_signals renames signal names with message prefixes

Best Practices:
  - Uses pytest tmp_path fixture for temporary files
  - Monkeypatches CantoolsDatabase for dependency isolation
  - Verifies error messages and states

Prerequisites:
  pip install pytest cantools

To execute:
    pytest tests/test_load_dbc_files.py -v
"""
import pytest
import pytest
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
def disable_cantools_loader(monkeypatch):
    """Prevent actual CantoolsDatabase usage by patching reference in module."""
    monkeypatch.setattr(canmlio, 'CantoolsDatabase', FakeDB)

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
    assert f"Failed to parse DBC file {p}" in str(excinfo.value)

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
    with pytest.raises(ValueError) as excinfo:
        canmlio.load_dbc_files(str(p), prefix_signals=False)
    assert "Duplicate signal names found across messages" in str(excinfo.value)

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
    # bypass default DB creation
    canmlio.CantoolsDatabase = lambda: fake
    with pytest.raises(ValueError) as excinfo:
        canmlio.load_dbc_files(str(p), prefix_signals=True)
    assert "Duplicate message names found; cannot prefix uniquely" in str(excinfo.value)


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
    db = canmlio.load_dbc_files(str(p), prefix_signals=True)
    sigs = [s.name for s in db.messages[0].signals]
    assert sigs == ['M_A', 'M_B']
