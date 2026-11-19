import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_classification, make_regression
from sklearn.utils.estimator_checks import parametrize_with_checks

from conftest import model_path
from tabicl import TabICLClassifier, TabICLRegressor
from tests.torch_devices_helpers import skip_if_device_unusable, DEVICES


# n_estimators=2 ensures the full preprocessing and ensembling pipeline is tested:
# n_estimators=1 skips shuffling and uses only one norm method, while n_estimators=2
# exercises feature/class shuffling, multiple normalization methods, and ensemble averaging.
#

_SKLEARN_ESTIMATORS = [
    est
    for device in DEVICES
    for est in (
        TabICLClassifier(n_estimators=2, device=device),
        TabICLRegressor(n_estimators=2, device=device),
    )
]


@parametrize_with_checks(_SKLEARN_ESTIMATORS)
def test_sklearn_compatible_estimator(estimator, check):
    skip_if_device_unusable(estimator.get_params()["device"])
    check(estimator)


@pytest.mark.parametrize("device", DEVICES)
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


def test_all_nan_column():
    """An all-NaN feature column must not break fit or predict_proba (gh-147)."""
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    X = np.hstack([X, np.full((X.shape[0], 1), np.nan)])
    clf = TabICLClassifier(n_estimators=2)
    clf.fit(X[:40], y[:40])
    proba = clf.predict_proba(X[40:])
    assert proba.shape == (10, 2)
    assert np.all(np.isfinite(proba))
    assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("kv_cache", [False, "kv", "repr"])
@pytest.mark.parametrize("scale, shift", [(1.0, 1000.0), (3.0, 0.0)])
def test_regression_variance_target_transform(kv_cache, scale, shift):
    """Variance is invariant to target shifts and scales quadratically (gh-180)."""
    X, y = make_regression(n_samples=50, n_features=5, noise=10.0, random_state=0)
    X_train, X_test, y_train = X[:40], X[40:], y[:40]
    reg = TabICLRegressor(
        n_estimators=2, device="cpu", kv_cache=kv_cache, random_state=0, **model_path("regressor")
    )
    reg.fit(X_train, y_train)
    original = reg.predict(X_test, output_type=["mean", "variance"])
    variance = reg.predict(X_test, output_type="variance")
    assert variance.shape == (len(X_test),)
    assert_allclose(variance, original["variance"])

    reg.fit(X_train, scale * y_train + shift)
    transformed = reg.predict(X_test, output_type=["mean", "variance"])
    assert_allclose(transformed["variance"], scale**2 * original["variance"], rtol=1e-4, atol=1e-4)
    assert_allclose(transformed["mean"], scale * original["mean"] + shift, rtol=1e-4, atol=1e-4)
    for prediction in (original, transformed):
        assert np.all(np.isfinite(prediction["variance"]))
        assert np.all(prediction["variance"] >= 0)
