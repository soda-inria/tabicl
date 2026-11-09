import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification, make_regression

from tabicl import TabICLClassifier, TabICLRegressor
from tabicl._model.inference_config import InferenceConfig
from tabicl._model.tabicl import TabICL
from tests.torch_devices_helpers import DEVICES, skip_if_device_unusable


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kv_cache", ["kv", "repr"])
@pytest.mark.parametrize("offload_mode", ["gpu", "disk"])
def test_classifier_kv_cache(device, kv_cache, offload_mode, tmp_path):
    """Predictions with kv cache should match predictions without cache for all offload modes."""
    skip_if_device_unusable(device)
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    X_train, X_test = X[:40], X[40:]
    y_train = y[:40]
    clf = TabICLClassifier(n_estimators=2, device=device, offload_mode="cpu")
    clf.fit(X_train, y_train)
    pred_no_cache = clf.predict_proba(X_test)

    clf_cached = TabICLClassifier(
        n_estimators=2,
        device=device,
        kv_cache=kv_cache,
        offload_mode=offload_mode,
        disk_offload_dir=str(tmp_path),
    )
    clf_cached.fit(X_train, y_train)
    pred_cached = clf_cached.predict_proba(X_test)

    np.testing.assert_allclose(pred_no_cache, pred_cached, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kv_cache", ["kv", "repr"])
@pytest.mark.parametrize("offload_mode", ["gpu", "disk"])
def test_regression_kv_cache(device, kv_cache, offload_mode, tmp_path):
    """Predictions with kv cache should match predictions without cache for all offload modes."""
    skip_if_device_unusable(device)
    X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    X_train, X_test = X[:40], X[40:]
    y_train = y[:40]
    reg = TabICLRegressor(n_estimators=2, device=device, offload_mode="cpu")
    reg.fit(X_train, y_train)
    pred_no_cache = reg.predict(X_test)

    reg_cached = TabICLRegressor(
        n_estimators=2,
        device=device,
        kv_cache=kv_cache,
        offload_mode=offload_mode,
        disk_offload_dir=str(tmp_path),
    )
    reg_cached.fit(X_train, y_train)
    pred_cached = reg_cached.predict(X_test)

    # Relaxed tolerance: kv cache changes float32 computation order
    np.testing.assert_allclose(pred_no_cache, pred_cached, rtol=1e-4, atol=1e-4)
