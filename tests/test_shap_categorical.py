"""Tests for SHAP categorical error handling."""

import numpy as np
import pandas as pd
import pytest
import shap  # noqa: F401

from tabicl import TabICLClassifier
from tabicl.shap import get_shap_values

from conftest import model_path


def test_get_shap_values_gives_clear_error_for_string_categories():
    """String/object categoricals should give a helpful TypeError.

    This tests the API asymmetry: TabICL accepts categorical DataFrames during
    fit, but the SHAP helper requires numeric input. The error message guides
    users to encode before fitting.
    """
    X = pd.DataFrame({
        "cat": ["a", "b", "c"] * 10,
        "num1": np.random.randn(30),
        "num2": np.random.randn(30),
    })
    y = np.random.choice([0, 1], size=30)

    # Fit on categorical data (TabICL handles this automatically)
    clf = TabICLClassifier(n_estimators=1, **model_path("classifier"))
    clf.fit(X, y)  # This works fine

    # Try to explain the same categorical data - this is where SHAP limitation hits
    with pytest.raises(TypeError, match="require numeric input"):
        get_shap_values(clf, X)
