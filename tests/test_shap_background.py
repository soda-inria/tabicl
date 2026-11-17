"""Check that explicit backgrounds define the reference used for SHAP values."""

import numpy as np
import pandas as pd
import pytest
from conftest import model_path

from tabicl import TabICLClassifier, TabICLRegressor
from tabicl.shap import get_shap_explainer, get_shap_values


class LinearRegressor:
    def predict(self, X):
        return np.nan_to_num(X) @ np.array([2.0, -3.0]) + 5.0


@pytest.mark.parametrize("container", [np.array, list, pd.DataFrame])
def test_background_changes_reference_and_attributions(container):
    X = np.array([[4.0, 5.0], [6.0, 1.0]])
    background = container([[0.0, 2.0], [2.0, 4.0]])
    original = np.asarray(background).copy()
    estimator = LinearRegressor()

    sv = get_shap_values(estimator, X, X_background=background, algorithm="exact")

    expected_values = (X - np.array([1.0, 3.0])) * np.array([2.0, -3.0])
    np.testing.assert_allclose(sv.base_values, -2.0)
    np.testing.assert_allclose(sv.values, expected_values)
    np.testing.assert_allclose(sv.base_values + sv.values.sum(axis=1), estimator.predict(X))
    np.testing.assert_array_equal(np.asarray(background), original)


def test_default_background_preserves_missing_value_reference():
    X = np.array([[4.0, 5.0]])
    estimator = LinearRegressor()
    default = get_shap_values(estimator, X, algorithm="exact")
    explicit = get_shap_values(estimator, X, X_background=None, algorithm="exact")

    np.testing.assert_allclose(default.base_values, 5.0)
    np.testing.assert_allclose(default.values, X * np.array([2.0, -3.0]))
    np.testing.assert_array_equal(default.values, explicit.values)
    explainer = get_shap_explainer(estimator, X, predict_fn="predict")
    assert explainer.masker.data.shape == (1, 2)
    assert np.isnan(explainer.masker.data).all()


@pytest.mark.parametrize("predict_fn", ["predict", LinearRegressor().predict])
def test_explainer_accepts_background_and_prediction_function(predict_fn):
    X = np.array([[4.0, 5.0]])
    background = np.array([[1.0, 3.0]])
    explainer = get_shap_explainer(
        LinearRegressor(), X, predict_fn=predict_fn, X_background=background, algorithm="exact"
    )
    sv = explainer(X)
    np.testing.assert_allclose(sv.base_values, -2.0)
    np.testing.assert_allclose(sv.values, [[6.0, -6.0]])


def test_dataframe_feature_names_are_preserved():
    X = pd.DataFrame({"first": [4.0], "second": [5.0]})
    background = pd.DataFrame({"first": [1.0], "second": [3.0]})
    sv = get_shap_values(LinearRegressor(), X, X_background=background, algorithm="exact")
    assert sv.feature_names == ["first", "second"]
    np.testing.assert_allclose(sv.values, [[6.0, -6.0]])


@pytest.mark.parametrize("helper", [get_shap_values, get_shap_explainer])
@pytest.mark.parametrize(
    ("background", "error", "message"),
    [
        (np.ones(2), ValueError, "two-dimensional"),
        (np.ones((1, 2, 1)), ValueError, "two-dimensional"),
        (np.empty((0, 2)), ValueError, "at least one sample"),
        (np.ones((3, 1)), ValueError, "same number of features"),
        ([["category", 1]], TypeError, "numeric"),
    ],
)
def test_invalid_background_has_actionable_error(helper, background, error, message):
    estimator = LinearRegressor()
    kwargs = {"predict_fn": "predict"} if helper is get_shap_explainer else {}
    with pytest.raises(error, match=message):
        helper(estimator, np.ones((1, 2)), X_background=background, **kwargs)


@pytest.mark.parametrize("kind", ["classifier", "regressor"])
def test_tabicl_background_reconstructs_predictions(kind):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 2))
    y = np.tile([0, 1], 10) if kind == "classifier" else 2 * X[:, 0] - X[:, 1]
    cls = TabICLClassifier if kind == "classifier" else TabICLRegressor
    estimator = cls(n_estimators=1, n_jobs=1, device="cpu", use_amp=False, **model_path(kind)).fit(X, y)
    X_test, background = X[:2], X[2:4]

    sv = get_shap_values(estimator, X_test, X_background=background, algorithm="exact")

    predict = estimator.predict_proba if kind == "classifier" else estimator.predict
    expected_reference = predict(background).mean(axis=0)
    np.testing.assert_allclose(sv.base_values, np.broadcast_to(expected_reference, sv.base_values.shape), atol=1e-5)
    np.testing.assert_allclose(sv.base_values + sv.values.sum(axis=1), predict(X_test), atol=1e-5)
