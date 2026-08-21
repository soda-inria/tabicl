"""Tests for predict_proba with all-NaN categorical columns (GitHub issue #143)."""

import os

import numpy as np
import pandas as pd
import pytest

from tabicl import TabICLClassifier, TabICLRegressor

_CKPT_DIR = os.environ.get("TABICL_CHECKPOINT_DIR")


def _model_path(kind: str):
    if _CKPT_DIR is None:
        return {}
    filenames = {"classifier": "tabicl-classifier-v2-20260212.ckpt", "regressor": "tabicl-regressor-v2-20260212.ckpt"}
    return {"model_path": os.path.join(_CKPT_DIR, filenames[kind])}


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

        clf = TabICLClassifier(n_estimators=1, **_model_path("classifier"))
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

        clf = TabICLClassifier(n_estimators=1, **_model_path("classifier"))
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

        reg = TabICLRegressor(n_estimators=1, **_model_path("regressor"))
        reg.fit(X, y)

        X_pred = X.head(20).copy()
        X_pred["cat"] = pd.Categorical([None] * 20, categories=["a", "b", "c"])
        preds = reg.predict(X_pred)
        assert preds.shape == (20,)

    def test_shap_numpy_allnan_column_still_works(self):
        """SHAP's feature masking (numpy arrays with all-NaN columns) must still work."""
        rng = np.random.default_rng(0)
        n = 50
        X = rng.normal(size=(n, 4))
        y = rng.choice([0, 1], size=n)

        clf = TabICLClassifier(n_estimators=1, **_model_path("classifier"))
        clf.fit(X, y)

        X_pred = X[:10].copy()
        X_pred[:, 2] = np.nan
        proba = clf.predict_proba(X_pred)
        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
