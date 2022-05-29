import pandas as pd

from data import read_dataframe


def test_read_dataframe_type() -> None:
    df = read_dataframe()
    assert isinstance(df, pd.DataFrame)


def test_read_dataframe_shape() -> None:
    df = read_dataframe()
    assert df.shape[0] == 9903
    assert df.shape[1] == 10
