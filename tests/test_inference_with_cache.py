"""Inference with a stored cache must predict what inference without one predicts.

The estimators store the cache during ``fit`` with ``kv_cache="kv"`` or ``"repr"``.
The row-wise interaction returns its output where the offload settings of
``ROW_CONFIG`` put it, which can be CPU memory while the model runs on an
accelerator; storing a repr cache then adds the label embedding, computed on
the device, to those representations.
"""

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
def test_classifier_kv_cache(device, kv_cache):
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kv_cache", ["kv", "repr"])
def test_regression_kv_cache(device, kv_cache):
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


@pytest.mark.parametrize("device", DEVICES)
def test_repr_cache_from_offloaded_representations(device):
    """Predictions match those with the representations kept on the device.

    Non-regression test for https://github.com/soda-inria/tabicl/pull/181.
    """
    skip_if_device_unusable(device)
    torch.manual_seed(0)
    model = TabICL().eval().to(device)
    X_train = torch.randn(1, 20, 5, device=device)
    y_train = torch.randint(0, 3, (1, 20), device=device).float()
    X_test = torch.randn(1, 10, 5, device=device)

    def predict(row_offload):
        mgr = {"device": device, "use_amp": False}
        config = InferenceConfig(COL_CONFIG=mgr, ROW_CONFIG={**mgr, "offload": row_offload}, ICL_CONFIG=mgr)
        with torch.no_grad():
            return model.forward_with_cache(
                X_train=X_train, y_train=y_train, X_test=X_test, cache_mode="repr", inference_config=config
            )

    torch.testing.assert_close(predict("cpu"), predict("gpu"))
