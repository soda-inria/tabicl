"""Tests for SHAP categorical error handling."""

import numpy as np
import pandas as pd
import pytest
import shap  # noqa: F401

from tabicl import TabICLClassifier
from tabicl.shap import get_shap_values

from conftest import model_path


@pytest.mark.parametrize("input_type", ["dataframe", "object_array"])
def test_get_shap_values_gives_clear_error_for_string_categories(input_type):
    """String/object categoricals should give a helpful TypeError.

    This tests the realistic workflow: TabICL handles categorical data fine
    during fit, but SHAP cannot explain categoricals without pre-encoding.
    """
    # Prepare categorical data
    if input_type == "dataframe":
        X = pd.DataFrame({
            "cat": ["a", "b", "c"] * 10,
            "num1": np.random.randn(30),
            "num2": np.random.randn(30),
        })
    else:  # object_array
        X = np.array(
            [["a", 1.0, 4.0], ["b", 2.0, 5.0], ["c", 3.0, 6.0]] * 10,
            dtype=object
        )

    y = np.random.choice([0, 1], size=30)

    # Fit on categorical data (TabICL handles this automatically)
    clf = TabICLClassifier(n_estimators=1, **model_path("classifier"))
    clf.fit(X, y)  # This works fine

    # Try to explain the same categorical data - this is where SHAP limitation hits
    with pytest.raises(TypeError, match="require numeric input"):
        get_shap_values(clf, X)
