"""
Module: tests/test_to_csv.py

This test suite verifies the behavior of the `to_csv` function in the
`canml.canmlio` module using pytest. It tests writing a single DataFrame,
iterable of DataFrames, column ordering, pandas_kwargs passthrough,
error handling, and logging behavior.

Test Cases:
  - Writing a single DataFrame creates CSV with correct header and data
  - Iterable of DataFrames writes multiple chunks with header on first only
  - Specifying `columns` warns on missing columns and fills with NaN
  - Invalid column names list raises ValueError
  - Non-iterable input raises TypeError
  - Empty iterable yields empty file with headers
  - pandas_kwargs passed through to `to_csv`

Prerequisites:
  - Install dependencies: pip install pytest pandas numpy

To execute:
    pytest tests/test_to_csv.py -v
"""
import pytest
import pandas as pd
from pathlib import Path

import canml.canmlio as canmlio

def test_single_dataframe_write(tmp_path):
    """Writing a single DataFrame creates CSV with correct header and data."""
    df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
    out = tmp_path / 'out.csv'
    canmlio.to_csv(df, str(out), mode='w', header=True)
    # Read back
    df2 = pd.read_csv(str(out))
    pd.testing.assert_frame_equal(df, df2)

def test_iterable_of_dataframes(tmp_path):
    """Iterable of DataFrames writes multiple chunks with only first header."""
    df1 = pd.DataFrame({'A': [1], 'B': [2]})
    df2 = pd.DataFrame({'A': [3], 'B': [4]})
    out = tmp_path / 'chunks.csv'
    canmlio.to_csv([df1, df2], str(out), mode='w', header=True)
    lines = Path(out).read_text().strip().splitlines()
    # Header + 2 data lines
    assert lines[0] == 'A,B'
    assert lines[1] == '1,2'
    assert lines[2] == '3,4'

def test_missing_columns_warning_and_fill(tmp_path, caplog):
    """Specifying columns with missing names warns and fills them with NaN."""
    df = pd.DataFrame({'A': [1, 2]})
    out = tmp_path / 'missing.csv'
    caplog.set_level('WARNING')
    canmlio.to_csv(df, str(out), mode='w', header=True, columns=['A', 'B'])
    # Warning for missing B
    assert any("Columns {'B'} not found in data; filling with NaN" in rec.getMessage() for rec in caplog.records)
    df2 = pd.read_csv(str(out))
    assert 'B' in df2.columns
    assert df2['B'].isna().all()

def test_invalid_columns_list_raises():
    """Non-string or duplicate column names raise ValueError."""
    df = pd.DataFrame({'X': [0]})
    with pytest.raises(ValueError):
        canmlio.to_csv(df, 'dummy.csv', columns=[123])
    with pytest.raises(ValueError):
        canmlio.to_csv(df, 'dummy.csv', columns=['X', 'X'])

def test_non_iterable_input_raises():
    """Passing non-DataFrame, non-iterable input raises TypeError."""
    with pytest.raises(TypeError):
        canmlio.to_csv(123, 'dummy.csv')

def test_empty_iterable_creates_empty(tmp_path, caplog):
    """Empty iterable yields an empty CSV file with no data but header if columns given."""
    out = tmp_path / 'empty.csv'
    caplog.set_level('WARNING')
    canmlio.to_csv([], str(out), columns=['A', 'B'])
    # Warning for no data
    assert any('No data to write' in rec.getMessage() for rec in caplog.records)
    text = Path(out).read_text().splitlines()
    # Header only
    assert text == ['A,B']

def test_pandas_kwargs_passthrough(tmp_path):
    """Additional pandas_kwargs are forwarded to DataFrame.to_csv via custom separator."""
    # Create DataFrame with two columns to observe separator
    df = pd.DataFrame({'A': [5], 'B': [6]})
    out = tmp_path / 'kwargs.csv'
    # Use a custom separator
    canmlio.to_csv(df, str(out), pandas_kwargs={'sep': '|'})
    content = Path(out).read_text().strip().splitlines()
    # Confirm header uses '|' and data row uses '|'
    assert content[0] == 'A|B'
    assert content[1] == '5|6'

def test_iterable_missing_columns_first_chunk(tmp_path, caplog):
    """Iterable with columns specified warns only on first chunk and fills missing."""
    caplog.set_level('WARNING')
    # First chunk has only 'A', second chunk has both 'A' and 'B'
    df1 = pd.DataFrame({'A': [1]})
    df2 = pd.DataFrame({'A': [2], 'B': [3]})
    out = tmp_path / 'iter_miss.csv'
    canmlio.to_csv([df1, df2], str(out), header=True, columns=['A', 'B'])
    # Warning only once for missing 'B' in first chunk
    warnings = [rec.getMessage() for rec in caplog.records if 'filling with NaN' in rec.getMessage()]
    assert len(warnings) == 1
    assert "Columns {'B'} not found in first chunk; filling with NaN" in warnings[0]
    # Read back and ensure both columns present and second chunk B is preserved
    df_out = pd.read_csv(str(out))
    assert list(df_out.columns) == ['A', 'B']
    # First row B is NaN, second row B equals 3
    assert pd.isna(df_out.loc[0, 'B'])
    assert df_out.loc[1, 'B'] == 3
