import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.utils.estimator_checks import parametrize_with_checks

from tabicl import TabICLClassifier, TabICLRegressor


# n_estimators=2 ensures the full preprocessing and ensembling pipeline is tested:
# n_estimators=1 skips shuffling and uses only one norm method, while n_estimators=2
# exercises feature/class shuffling, multiple normalization methods, and ensemble averaging.
@parametrize_with_checks([TabICLClassifier(n_estimators=2), TabICLRegressor(n_estimators=2)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


class TestClassifierKVCache:
    @pytest.mark.parametrize("kv_cache", ["kv", "repr"])
    def test_kv_cache(self, kv_cache):
        """Predictions with kv cache should match predictions without cache."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        X_train, X_test = X[:40], X[40:]
        y_train = y[:40]
        clf = TabICLClassifier(n_estimators=2)
        clf.fit(X_train, y_train)
        pred_no_cache = clf.predict_proba(X_test)

        clf.fit(X_train, y_train, kv_cache=kv_cache)
        pred_cached = clf.predict_proba(X_test)

        np.testing.assert_allclose(pred_no_cache, pred_cached, rtol=1e-4, atol=1e-4)


class TestRegressorKVCache:
    @pytest.mark.parametrize("kv_cache", ["kv", "repr"])
    def test_kv_cache(self, kv_cache):
        """Predictions with kv cache should match predictions without cache."""
        X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        X_train, X_test = X[:40], X[40:]
        y_train = y[:40]
        reg = TabICLRegressor(n_estimators=2)
        reg.fit(X_train, y_train)
        pred_no_cache = reg.predict(X_test)

        reg.fit(X_train, y_train, kv_cache=kv_cache)
        pred_cached = reg.predict(X_test)

        # Relaxed tolerance: kv cache changes float32 computation order
        np.testing.assert_allclose(pred_no_cache, pred_cached, rtol=1e-4, atol=1e-4)
