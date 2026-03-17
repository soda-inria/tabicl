import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier, is_regressor

from tabicl import TabICLClassifier, TabICLRegressor

@pytest.mark.parametrize("container", ["dataframe", "numpy_object"])
@pytest.mark.parametrize("Estimator", [TabICLClassifier, TabICLRegressor])
def test_handles_string_inputs_without_crashing(Estimator, container):
    """Estimator should not crash with string-valued inputs in different containers."""

    rng = np.random.RandomState(0)
    n = 20

    num1 = rng.randn(n)
    num2 = rng.randn(n)
    obj_str = pd.Series(rng.choice(["a", "b", "c"], size=n), dtype=object)
    str_dtype = pd.Series(rng.choice(["x", "y", "z"], size=n), dtype="string")

    if container == "dataframe":
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

    elif container == "numpy_object":
        X = np.empty((n, 4), dtype=object)
        X[:, 0] = num1
        X[:, 1] = num2
        X[:, 2] = obj_str
        X[:, 3] = rng.choice(["u", "v", "w"], size=n)

        assert X.dtype == object

    else:
        raise AssertionError(f"Unknown container: {container}")

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