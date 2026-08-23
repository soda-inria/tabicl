"""Tests for finetuning with string/categorical features (GitHub issue #118)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("transformers", reason="finetune extra not installed")
from tabicl import FinetunedTabICLClassifier, FinetunedTabICLRegressor

from conftest import model_path


def _make_categorical_dataframe(n=80, rng=None):
    """Create a DataFrame with mixed numeric and categorical columns."""
    if rng is None:
        rng = np.random.RandomState(42)
    cat_values = rng.choice(["foo", "bar", "baz"], size=n)
    num1 = rng.randn(n)
    num2 = rng.randn(n)
    X = pd.DataFrame(
        {
            "num1": num1,
            "num2": num2,
            "cat": pd.Series(cat_values, dtype="category"),
        }
    )
    return X


@pytest.fixture
def clf_data():
    rng = np.random.RandomState(42)
    X = _make_categorical_dataframe(n=80, rng=rng)
    y = (X["num1"] > 0).astype(int).values
    return X, y


@pytest.fixture
def reg_data():
    rng = np.random.RandomState(42)
    X = _make_categorical_dataframe(n=80, rng=rng)
    y = X["num1"].values + rng.randn(80) * 0.1
    return X, y


class TestFinetunedClassifierStringCategoricals:
    """FinetunedTabICLClassifier should handle string categoricals like TabICLClassifier."""

    def test_fit_predict_with_category_dtype(self, clf_data):
        X, y = clf_data
        est = FinetunedTabICLClassifier(
            epochs=1,
            n_estimators_finetune=1,
            n_estimators_validation=1,
            n_estimators_inference=1,
            early_stopping=False,
            **model_path("classifier"),
        )
        est.fit(X, y)
        preds = est.predict(X)
        assert len(preds) == len(y)

    def test_fit_predict_with_object_dtype(self, clf_data):
        X, y = clf_data
        X = X.copy()
        X["cat"] = X["cat"].astype(object)
        est = FinetunedTabICLClassifier(
            epochs=1,
            n_estimators_finetune=1,
            n_estimators_validation=1,
            n_estimators_inference=1,
            early_stopping=False,
            **model_path("classifier"),
        )
        est.fit(X, y)
        preds = est.predict(X)
        assert len(preds) == len(y)

    def test_fit_predict_with_string_dtype(self, clf_data):
        X, y = clf_data
        X = X.copy()
        X["cat"] = X["cat"].astype("string")
        est = FinetunedTabICLClassifier(
            epochs=1,
            n_estimators_finetune=1,
            n_estimators_validation=1,
            n_estimators_inference=1,
            early_stopping=False,
            **model_path("classifier"),
        )
        est.fit(X, y)
        preds = est.predict(X)
        assert len(preds) == len(y)

    def test_explicit_validation_set_with_categoricals(self, clf_data):
        X, y = clf_data
        X_train, X_val = X.iloc[:60], X.iloc[60:]
        y_train, y_val = y[:60], y[60:]
        est = FinetunedTabICLClassifier(
            epochs=1,
            n_estimators_finetune=1,
            n_estimators_validation=1,
            n_estimators_inference=1,
            early_stopping=False,
            **model_path("classifier"),
        )
        est.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        preds = est.predict(X)
        assert len(preds) == len(y)

    def test_unseen_validation_categories(self, clf_data):
        """Validation set may contain categories not seen during training."""
        X, y = clf_data
        X_train = X.iloc[:60].copy()
        X_val = X.iloc[60:].copy()
        X_val["cat"] = X_val["cat"].astype(object)
        X_val.iloc[0, X_val.columns.get_loc("cat")] = "unseen_category"
        y_train, y_val = y[:60], y[60:]
        est = FinetunedTabICLClassifier(
            epochs=1,
            n_estimators_finetune=1,
            n_estimators_validation=1,
            n_estimators_inference=1,
            early_stopping=False,
            **model_path("classifier"),
        )
        est.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        preds = est.predict(X_val)
        assert len(preds) == len(y_val)


class TestFinetunedRegressorStringCategoricals:
    """FinetunedTabICLRegressor should handle string categoricals like TabICLRegressor."""

    def test_fit_predict_with_category_dtype(self, reg_data):
        X, y = reg_data
        est = FinetunedTabICLRegressor(
            epochs=1,
            n_estimators_finetune=1,
            n_estimators_validation=1,
            n_estimators_inference=1,
            early_stopping=False,
            **model_path("regressor"),
        )
        est.fit(X, y)
        preds = est.predict(X)
        assert len(preds) == len(y)

    def test_fit_predict_with_object_dtype(self, reg_data):
        X, y = reg_data
        X = X.copy()
        X["cat"] = X["cat"].astype(object)
        est = FinetunedTabICLRegressor(
            epochs=1,
            n_estimators_finetune=1,
            n_estimators_validation=1,
            n_estimators_inference=1,
            early_stopping=False,
            **model_path("regressor"),
        )
        est.fit(X, y)
        preds = est.predict(X)
        assert len(preds) == len(y)

    def test_explicit_validation_set_with_categoricals(self, reg_data):
        X, y = reg_data
        X_train, X_val = X.iloc[:60], X.iloc[60:]
        y_train, y_val = y[:60], y[60:]
        est = FinetunedTabICLRegressor(
            epochs=1,
            n_estimators_finetune=1,
            n_estimators_validation=1,
            n_estimators_inference=1,
            early_stopping=False,
            **model_path("regressor"),
        )
        est.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        preds = est.predict(X)
        assert len(preds) == len(y)
