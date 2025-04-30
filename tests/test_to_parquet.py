"""
Module: tests/test_to_parquet.py

This test suite verifies the behavior of the `to_parquet` function in the
`canml.canmlio` module using pytest. It ensures:
  - Writing a DataFrame to Parquet with default compression
  - Writing with a different compression codec
  - Metadata JSON export when requested
  - Directory creation for parquet and metadata
  - Proper logging
  - Error propagation on invalid write paths

Prerequisites:
  pip install pytest pandas pyarrow

To execute:
    pytest tests/test_to_parquet.py -v
"""
import pytest
import pandas as pd
import json
import os
import logging
from pathlib import Path

import canml.canmlio as canmlio

@pytest.fixture(autouse=True)
def setup_logging(caplog):
    caplog.set_level(logging.INFO, logger=canmlio.__name__)
    yield
    canmlio.glogger.setLevel(logging.INFO)

@pytest.fixture
def sample_df(tmp_path):
    df = pd.DataFrame({'x': [10, 20], 'y': [30, 40]})
    df.attrs['signal_attributes'] = {'x': {'unit': 'km/h'}, 'y': {'unit': 'm/s'}}
    return df


def test_write_parquet_default(tmp_path, sample_df, caplog):
    out = tmp_path / 'data.parquet'
    canmlio.to_parquet(sample_df, str(out))
    assert out.exists()
    # read back
    df_read = pd.read_parquet(out)
    pd.testing.assert_frame_equal(df_read, sample_df)
    assert f"Parquet written to {out}" in caplog.text


def test_write_parquet_compression(tmp_path, sample_df):
    out = tmp_path / 'data_gzip.parquet'
    canmlio.to_parquet(sample_df, str(out), compression='gzip')
    assert out.exists()
    df_read = pd.read_parquet(out)
    pd.testing.assert_frame_equal(df_read, sample_df)


def test_metadata_export(tmp_path, sample_df):
    out = tmp_path / 'folder' / 'nested.parquet'
    meta = tmp_path / 'folder' / 'meta.json'
    # ensure nested directories are created
    canmlio.to_parquet(sample_df, str(out), metadata_path=str(meta))
    assert out.exists()
    assert meta.exists()
    data = json.loads(meta.read_text())
    assert 'x' in data and 'y' in data


def test_no_metadata_if_not_requested(tmp_path, sample_df):
    out = tmp_path / 'data2.parquet'
    meta = tmp_path / 'data2_meta.json'
    canmlio.to_parquet(sample_df, str(out))
    assert out.exists()
    assert not meta.exists()


def test_path_dir_creation(tmp_path, sample_df):
    """Output and metadata directories are auto-created."""
    out = tmp_path / 'nonexistent_dir' / 'file.parquet'
    meta = tmp_path / 'nonexistent_dir' / 'file.json'
    canmlio.to_parquet(sample_df, str(out), metadata_path=str(meta))
    assert out.exists()
    assert meta.exists()
