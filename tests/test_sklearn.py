import copy

import numpy as np
import pytest
import sklearn
from numpy.testing import assert_allclose
from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.fixes import parse_version
from threadpoolctl import threadpool_info

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

# scikit-learn < 1.4 forces aligned memmaps when OpenBLAS reports (or may be) the
# Prescott architecture, and aligned memmaps reject estimator objects.
_SKLEARN_PRESCOTT_MEMMAP_BUG = parse_version(sklearn.__version__) < parse_version("1.4") and any(
    info["internal_api"] == "openblas"
    and info.get("architecture", "prescott").lower() == "prescott"
    for info in threadpool_info()
)


@parametrize_with_checks(_SKLEARN_ESTIMATORS)
def test_sklearn_compatible_estimator(estimator, check):
    skip_if_device_unusable(estimator.get_params()["device"])
    if (
        _SKLEARN_PRESCOTT_MEMMAP_BUG
        and check.func.__name__ == "check_estimators_pickle"
        and check.keywords.get("readonly_memmap")
    ):
        pytest.skip("scikit-learn < 1.4 cannot memmap estimators with Prescott OpenBLAS")
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
