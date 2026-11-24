import pandas as pd
import pytest

from tabicl._sklearn.preprocessing import TransformToNumerical


def test_categorical_dtype_columns_are_ordinal_encoded():
    """String, object, category, and boolean columns are ordinal-encoded, not dropped."""

    X = pd.DataFrame(
        {
            "inferred_string": ["a", "b"],
            "string": pd.Series(["a", "b"], dtype="string"),
            "object": pd.Series(["a", "b"], dtype=object),
            "category": pd.Series(["a", "b"], dtype="category"),
            "numpy_bool": [True, False],
            "pandas_boolean": pd.Series([True, False], dtype="boolean"),
        }
    )

    X_out = TransformToNumerical().fit_transform(X)

    assert X_out.tolist() == [[0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 0, 0]]


def test_warns_about_columns_dropped_for_their_dtype():
    """Columns that are neither categorical nor numeric are dropped with a warning that lists their dtypes."""

    X = pd.DataFrame({"num": [1.0, 2.0], "date": pd.Series(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")})

    with pytest.warns(UserWarning, match=r"not used due to their data type: \{'date': 'datetime64\[ns\]'\}"):
        X_out = TransformToNumerical().fit_transform(X)

    assert X_out.tolist() == [[1.0], [2.0]]
