import pandas as pd
import pytest

from tabicl._sklearn.preprocessing import TransformToNumerical


def test_columns_are_classified_by_dtype(capsys):
    """String, object, category, and boolean columns are ordinal-encoded and numeric columns are kept.

    Non-regression test for https://github.com/soda-inria/tabicl/pull/190.
    """

    X = pd.DataFrame(
        {
            "inferred_string": ["a", "b"],
            "string": pd.Series(["a", "b"], dtype="string"),
            "object": pd.Series(["a", "b"], dtype=object),
            "category": pd.Series(["a", "b"], dtype="category"),
            "numpy_bool": [True, False],
            "pandas_boolean": pd.Series([True, False], dtype="boolean"),
            "numpy_int": [1, 2],
            "numpy_float": [0.5, 1.5],
            "pandas_int": pd.Series([1, 2], dtype="Int64"),
            "pandas_float": pd.Series([0.5, 1.5], dtype="Float64"),
        }
    )

    X_out = TransformToNumerical(verbose=True).fit_transform(X)

    assert X_out.tolist() == [[0, 0, 0, 0, 1, 1, 1, 0.5, 1, 0.5], [1, 1, 1, 1, 0, 0, 2, 1.5, 2, 1.5]]
    assert capsys.readouterr().out == (
        "Columns classified as categorical: "
        "['inferred_string', 'string', 'object', 'category', 'numpy_bool', 'pandas_boolean']\n"
        "Columns classified as continuous: ['numpy_int', 'numpy_float', 'pandas_int', 'pandas_float']\n"
    )


def test_warns_about_columns_dropped_for_their_dtype():
    """Columns that are neither categorical nor numeric are dropped with a warning that lists their dtypes."""

    X = pd.DataFrame({"num": [1.0, 2.0], "date": pd.Series(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")})

    with pytest.warns(UserWarning, match=r"not used due to their data type: \{'date': 'datetime64\[ns\]'\}"):
        X_out = TransformToNumerical().fit_transform(X)

    assert X_out.tolist() == [[1.0], [2.0]]
