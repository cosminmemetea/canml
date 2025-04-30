"""
Module: tests/test_to_csv.py

This test suite verifies the behavior of the `to_csv` function in the
`canml.canmlio` module using pytest. It ensures:
  - Writing a single DataFrame with header and without header
  - Writing multiple chunks to CSV with append mode
  - Column filtering/reordering
  - Duplicate columns detection
  - Metadata JSON export for both DataFrame and chunks
  - Proper logging
  - Error on non-DataFrame/non-iterable input

Prerequisites:
  pip install pytest pandas

To execute:
    pytest tests/test_to_csv.py -v
"""
import pytest
import pandas as pd
import json
import logging

import canml.canmlio as canmlio

@pytest.fixture(autouse=True)
def setup_logging(caplog):
    caplog.set_level(logging.INFO, logger=canmlio.__name__)
    yield
    canmlio.glogger.setLevel(logging.INFO)

@pytest.fixture
def sample_df(tmp_path):
    # Simple DataFrame with two columns and metadata
    df = pd.DataFrame({'a': [1,2], 'b': [3,4]})
    df.attrs['signal_attributes'] = {'a': {'unit': 'v'}, 'b': {'scale':2}}
    return df

@pytest.fixture
def chunks(sample_df):
    # Split sample into two chunks
    return [sample_df.iloc[:1], sample_df.iloc[1:]]


def test_write_single_df(tmp_path, sample_df, caplog):
    out = tmp_path / 'single.csv'
    meta = tmp_path / 'meta.json'
    canmlio.to_csv(sample_df, str(out), metadata_path=str(meta))
    # CSV file exists
    df_read = pd.read_csv(out)
    pd.testing.assert_frame_equal(df_read, sample_df)
    # Metadata written
    meta_data = json.loads(meta.read_text())
    assert 'a' in meta_data and 'b' in meta_data
    assert 'unit' in meta_data['a']
    assert 'scale' in meta_data['b']
    # Log message
    assert f"CSV written to {out}" in caplog.text


def test_write_iterable_chunks(tmp_path, chunks, caplog):
    out = tmp_path / 'iter.csv'
    meta = tmp_path / 'iter_meta.json'
    canmlio.to_csv(chunks, str(out), metadata_path=str(meta))
    df_read = pd.read_csv(out)
    # concatenated equals full
    df_full = pd.concat(chunks, ignore_index=True)
    pd.testing.assert_frame_equal(df_read, df_full)
    # metadata only from first chunk
    meta_data = json.loads(meta.read_text())
    assert meta_data == chunks[0].attrs['signal_attributes']
    assert f"CSV written to {out}" in caplog.text


def test_columns_filter_and_order(tmp_path):
    df = pd.DataFrame({'x': [1], 'y':[2], 'z':[3]})
    out = tmp_path / 'cols.csv'
    canmlio.to_csv(df, str(out), columns=['z','x'])
    df_read = pd.read_csv(out)
    # only z and x and in order
    assert list(df_read.columns) == ['z','x']
    assert df_read.at[0,'z']==3 and df_read.at[0,'x']==1


def test_duplicate_columns_error(tmp_path):
    df = pd.DataFrame({'a':[1]})
    with pytest.raises(ValueError):
        canmlio.to_csv(df, 'dummy.csv', columns=['a','a'])


def test_non_iterable_input_error(tmp_path):
    with pytest.raises(TypeError):
        canmlio.to_csv(123, str(tmp_path/'out.csv'))


def test_append_mode(tmp_path):
    df1 = pd.DataFrame({'c':[5]})
    df2 = pd.DataFrame({'c':[6]})
    out = tmp_path / 'append.csv'
    # first write with header
    canmlio.to_csv(df1, str(out), mode='w', header=True)
    # append without header
    canmlio.to_csv(df2, str(out), mode='a', header=False)
    df_read = pd.read_csv(out)
    assert list(df_read['c']) == [5,6]


def test_metadata_not_written_if_absent(tmp_path):
    df = pd.DataFrame({'a':[1]})
    out = tmp_path / 'no_meta.csv'
    meta = tmp_path / 'no_meta.json'
    canmlio.to_csv(df, str(out))
    assert not meta.exists()


def test_metadata_path_dir_created(tmp_path):
    df = pd.DataFrame({'d':[7]})
    out = tmp_path / 'nested' / 'out.csv'
    meta = tmp_path / 'nested' / 'info.json'
    # ensure directory creation
    canmlio.to_csv(df, str(out), metadata_path=str(meta))
    assert meta.exists()
    data = json.loads(meta.read_text())
    assert 'd' in data
