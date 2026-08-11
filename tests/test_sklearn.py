import copy

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.utils.estimator_checks import parametrize_with_checks

from tabicl import TabICLClassifier, TabICLRegressor
from tests.torch_devices import skip_if_device_unusable


# n_estimators=2 ensures the full preprocessing and ensembling pipeline is tested:
# n_estimators=1 skips shuffling and uses only one norm method, while n_estimators=2
# exercises feature/class shuffling, multiple normalization methods, and ensemble averaging.
#
_DEVICES = ["cpu", "cuda", "xpu", "mps"]
_SKLEARN_ESTIMATORS = [
    est
    for device in _DEVICES
    for est in (
        TabICLClassifier(n_estimators=2, device=device),
        TabICLRegressor(n_estimators=2, device=device),
    )
]


@parametrize_with_checks(_SKLEARN_ESTIMATORS)
def test_sklearn_compatible_estimator(estimator, check):
    skip_if_device_unusable(estimator.get_params()["device"])
    check(estimator)


@pytest.mark.parametrize("device", _DEVICES)
def test_serialization(device):
    skip_if_device_unusable(device)
    clf = TabICLClassifier(n_estimators=2, device=device)
    assert not hasattr(clf, "model_")
    assert not hasattr(clf, "model_kv_cache_")
    clone = copy.deepcopy(clf)
    assert not hasattr(clone, "model_")
    assert not hasattr(clf, "model_kv_cache_")
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    clf.fit(X, y)
    assert hasattr(clf, "model_")
    assert hasattr(clf, "model_kv_cache_")
    clone = copy.deepcopy(clf)
    assert hasattr(clone, "model_")
    assert hasattr(clf, "model_kv_cache_")


class TestClassifierKVCache:
    @pytest.mark.parametrize("device", _DEVICES)
    @pytest.mark.parametrize("kv_cache", ["kv", "repr"])
    def test_kv_cache(self, device, kv_cache):
        """Predictions with kv cache should match predictions without cache."""
        skip_if_device_unusable(device)
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        X_train, X_test = X[:40], X[40:]
        y_train = y[:40]
        clf = TabICLClassifier(n_estimators=2, device=device)
        clf.fit(X_train, y_train)
        pred_no_cache = clf.predict_proba(X_test)

        clf_cached = TabICLClassifier(n_estimators=2, device=device, kv_cache=kv_cache)
        clf_cached.fit(X_train, y_train)
        pred_cached = clf_cached.predict_proba(X_test)

        np.testing.assert_allclose(pred_no_cache, pred_cached, rtol=1e-4, atol=1e-4)


class TestRegressorKVCache:
    @pytest.mark.parametrize("device", _DEVICES)
    @pytest.mark.parametrize("kv_cache", ["kv", "repr"])
    def test_kv_cache(self, device, kv_cache):
        """Predictions with kv cache should match predictions without cache."""
        skip_if_device_unusable(device)
        X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        X_train, X_test = X[:40], X[40:]
        y_train = y[:40]
        reg = TabICLRegressor(n_estimators=2, device=device)
        reg.fit(X_train, y_train)
        pred_no_cache = reg.predict(X_test)

        reg_cached = TabICLRegressor(n_estimators=2, device=device, kv_cache=kv_cache)
        reg_cached.fit(X_train, y_train)
        pred_cached = reg_cached.predict(X_test)

        # Relaxed tolerance: kv cache changes float32 computation order
        np.testing.assert_allclose(pred_no_cache, pred_cached, rtol=1e-4, atol=1e-4)
