import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier, is_regressor

from tabicl import TabICLClassifier, TabICLRegressor

@pytest.mark.parametrize("Estimator", [TabICLClassifier, TabICLRegressor])
def test_handles_string_dataframe_without_crashing(Estimator):
    """Estimator should not crash with string-valued columns in a DataFrame."""

    rng = np.random.RandomState(0)
    n = 20

    num1 = rng.randn(n)
    num2 = rng.randn(n)
    obj_str = pd.Series(rng.choice(["a", "b", "c"], size=n), dtype=object)
    str_dtype = pd.Series(rng.choice(["x", "y", "z"], size=n), dtype="string")

    X = pd.DataFrame(
        {
            "num1": num1,
            "num2": num2,
            "obj_str": obj_str,      # object dtype
            "str_dtype": str_dtype,  # pandas string dtype
        }
    )

    assert X["obj_str"].dtype == "object"
    assert str(X["str_dtype"].dtype).startswith("string")

    est = Estimator()

    if is_classifier(est):
        y = rng.randint(0, 2, size=n)
    elif is_regressor(est):
        y = rng.randn(n)
    else:
        raise ValueError(f'Estimator is neither classifier nor regressor')

    est.fit(X, y)
    preds = est.predict(X)

    assert len(preds) == n


@pytest.mark.parametrize("Estimator", [TabICLClassifier, TabICLRegressor])
def test_string_numpy_object_array_raises(Estimator):
    """String-valued NumPy object arrays cannot be reliably typed column-wise,
    so the estimator should raise an informative error pointing users to DataFrames."""

    rng = np.random.RandomState(0)
    n = 20

    X = np.empty((n, 4), dtype=object)
    X[:, 0] = rng.randn(n)
    X[:, 1] = rng.randn(n)
    X[:, 2] = rng.choice(["a", "b", "c"], size=n)
    X[:, 3] = rng.choice(["u", "v", "w"], size=n)
    assert X.dtype == object

    est = Estimator()

    if is_classifier(est):
        y = rng.randint(0, 2, size=n)
    elif is_regressor(est):
        y = rng.randn(n)
    else:
        raise ValueError(f'Estimator is neither classifier nor regressor')

    with pytest.raises(ValueError, match="castable to a numeric dtype"):
        est.fit(X, y)