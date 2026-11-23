import pandas as pd
import pyarrow as pa
import pytest

from tabicl._sklearn.preprocessing import TransformToNumerical


def test_bool_and_pyarrow_string_columns_are_categorical():
    """Boolean and pyarrow-backed string columns are ordinal-encoded, not dropped."""

    X = pd.DataFrame(
        {
            "numpy_bool": [True, False],
            "pyarrow_bool": pd.Series([True, False], dtype="bool[pyarrow]"),
            "pyarrow_string": pd.Series(["a", "b"], dtype=pd.ArrowDtype(pa.string())),
        }
    )

    X_out = TransformToNumerical().fit_transform(X)

    assert X_out.tolist() == [[1, 1, 0], [0, 0, 1]]


def test_warns_about_columns_dropped_for_their_dtype():
    """Columns that are neither categorical nor numeric are dropped with a warning."""

    X = pd.DataFrame({"num": [1.0, 2.0], "date": pd.to_datetime(["2020-01-01", "2020-01-02"])})

    with pytest.warns(UserWarning, match=r"not used due to their data type: \['date'\]"):
        X_out = TransformToNumerical().fit_transform(X)

    assert X_out.tolist() == [[1.0], [2.0]]
