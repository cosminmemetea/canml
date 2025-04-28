"""
Module: tests/test_to_parquet.py

This test suite verifies the behavior of the `to_parquet` function in the
`canml.canmlio` module using pytest. It tests successful writes, pandas_kwargs
and compression forwarding, error handling, and logging behavior.

Test Cases:
  - Writing a DataFrame round-trip: file exists and data matches after read
  - Compression and pandas_kwargs are passed correctly to DataFrame.to_parquet
  - Exceptions in to_parquet produce ValueError with appropriate message

Prerequisites:
  - Install dependencies: pip install pytest pandas pyarrow

To execute:
    pytest tests/test_to_parquet.py -v
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import canml.canmlio as canmlio


def test_round_trip_parquet(tmp_path):
    """Writing a DataFrame and reading back yields identical data."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z'],
        'C': [0.1, 0.2, 0.3]
    })
    out = tmp_path / 'data.parquet'
    # Call to_parquet
    canmlio.to_parquet(df, str(out))
    # File should exist
    assert out.is_file(), "Parquet file was not created"
    # Read back and compare
    df2 = pd.read_parquet(str(out))
    # Ensure same columns and dtypes (allowing possible dtype promotions)
    pd.testing.assert_frame_equal(df, df2)


def test_compression_and_kwargs_forwarded(monkeypatch, tmp_path):
    """to_parquet should forward compression and pandas_kwargs to DataFrame.to_parquet."""
    df = pd.DataFrame({'X': [10]})
    out = tmp_path / 'test.parquet'
    calls = {}
    
    def fake_to_parquet(self, path, engine, compression, **kwargs):
        # Record parameters
        calls['path'] = path
        calls['engine'] = engine
        calls['compression'] = compression
        calls['kwargs'] = kwargs
        # create an empty file to satisfy existence if needed
        Path(path).write_bytes(b'')

    # Monkeypatch the DataFrame.to_parquet method
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', fake_to_parquet)
    # Call with custom compression and extra kwargs
    canmlio.to_parquet(df, str(out), compression='gzip', pandas_kwargs={'index': False, 'foo': 'bar'})
    assert Path(calls['path']) == out
    assert calls['engine'] == 'pyarrow'
    assert calls['compression'] == 'gzip'
    # pandas_kwargs forwarded
    assert calls['kwargs']['index'] is False
    assert calls['kwargs']['foo'] == 'bar'


def test_to_parquet_raises_value_error(monkeypatch, tmp_path, caplog):
    """Errors during DataFrame.to_parquet should raise ValueError and log error."""
    df = pd.DataFrame({'Z': [0]})
    out = tmp_path / 'fail.parquet'

    def broken_to_parquet(self, path, engine, compression, **kwargs):
        raise RuntimeError('disk full')

    monkeypatch.setattr(pd.DataFrame, 'to_parquet', broken_to_parquet)
    caplog.set_level('ERROR')
    with pytest.raises(ValueError) as excinfo:
        canmlio.to_parquet(df, str(out))
    # Check the message contains original exception text
    assert 'disk full' in str(excinfo.value)
    # Logger should have an error record
    errors = [rec for rec in caplog.records if rec.levelname == 'ERROR']
    assert any('Failed to write Parquet' in rec.getMessage() for rec in errors)
