"""Tests for predict_proba with all-NaN categorical columns (GitHub issue #143)."""

import numpy as np
import pandas as pd
import pytest

from tabicl import TabICLClassifier, TabICLRegressor

from conftest import model_path


class TestPredictAllNanCategorical:
    """predict_proba/predict should not crash when a categorical column is all-NaN."""

    def test_classifier_predict_proba_allnan_categorical(self):
        rng = np.random.default_rng(0)
        n = 100
        X = pd.DataFrame({
            "num": rng.normal(size=n),
            "cat": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
        })
        y = rng.choice([0, 1], size=n)

        clf = TabICLClassifier(n_estimators=1, **model_path("classifier"))
        clf.fit(X, y)

        X_pred = X.head(20).copy()
        X_pred["cat"] = pd.Categorical([None] * 20, categories=["a", "b", "c"])
        proba = clf.predict_proba(X_pred)
        assert proba.shape == (20, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_classifier_predict_allnan_categorical(self):
        rng = np.random.default_rng(0)
        n = 100
        X = pd.DataFrame({
            "num": rng.normal(size=n),
            "cat": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
        })
        y = rng.choice([0, 1], size=n)

        clf = TabICLClassifier(n_estimators=1, **model_path("classifier"))
        clf.fit(X, y)

        X_pred = X.head(20).copy()
        X_pred["cat"] = pd.Categorical([None] * 20, categories=["a", "b", "c"])
        preds = clf.predict(X_pred)
        assert preds.shape == (20,)

    def test_regressor_predict_allnan_categorical(self):
        rng = np.random.default_rng(0)
        n = 100
        X = pd.DataFrame({
            "num": rng.normal(size=n),
            "cat": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
        })
        y = rng.normal(size=n)

        reg = TabICLRegressor(n_estimators=1, **model_path("regressor"))
        reg.fit(X, y)

        X_pred = X.head(20).copy()
        X_pred["cat"] = pd.Categorical([None] * 20, categories=["a", "b", "c"])
        preds = reg.predict(X_pred)
        assert preds.shape == (20,)

    def test_numpy_allnan_column_predict_proba(self):
        """Numpy arrays with all-NaN columns should produce valid predictions."""
        rng = np.random.default_rng(0)
        n = 50
        X = rng.normal(size=(n, 4))
        y = rng.choice([0, 1], size=n)

        clf = TabICLClassifier(n_estimators=1, **model_path("classifier"))
        clf.fit(X, y)

        X_pred = X[:10].copy()
        X_pred[:, 2] = np.nan
        proba = clf.predict_proba(X_pred)
        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_prediction_independent_of_batch_composition(self):
        """A row's prediction must not change based on other rows' NaN patterns."""
        rng = np.random.default_rng(0)
        n = 50
        X = rng.normal(size=(n, 4))
        y = rng.choice([0, 1], size=n)

        clf = TabICLClassifier(n_estimators=1, **model_path("classifier"))
        clf.fit(X, y)

        # Row 0 has a non-NaN value in column 2
        X_row = X[:1].copy()

        # Predict row 0 alone
        proba_alone = clf.predict_proba(X_row)

        # Predict row 0 alongside rows where column 2 is all-NaN
        X_batch = np.vstack([X_row, X[1:5]])
        X_batch[1:, 2] = np.nan
        proba_in_batch = clf.predict_proba(X_batch)

        # Row 0's prediction should be the same regardless of batch composition
        np.testing.assert_allclose(proba_alone[0], proba_in_batch[0], rtol=1e-4, atol=1e-5)
