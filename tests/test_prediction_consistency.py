"""Golden-value checks that predict_proba / mean / quantile outputs stay stable.

Fixtures under ``tests/data/prediction_consistency/`` store the train/test
tables (parquet) and expected predictions (``.npy`` under
``tabicl-vX.Y.Z/``). Tests always load existing fixtures (and raise if any are
missing). To rebuild them after intentional changes, see the docstring of
``tests/generate_data_and_predictions.py``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from conftest import model_path
from tabicl import TabICLClassifier, TabICLRegressor
from tests.generate_data_and_predictions import (
    ESTIMATOR_KWARGS,
    N_TEST,
    QUANTILE_ALPHAS,
    prediction_paths,
    task_paths,
)
from tests.torch_devices_helpers import skip_if_device_unusable, DEVICES

# Cross-device fp32 drift. Those values are aligned with the device↔CPU
# parity tests in test_numpy_inputs.py for the float32 case (`use_amp=False`).
RTOL = 1e-4
ATOL = 1e-4


def _load_task(kind: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    paths = task_paths(kind)
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing prediction-consistency data: {missing}. "
            "See tests/generate_data_and_predictions.py on how to generate them."
        )
    X_train = pd.read_parquet(paths["X_train"])
    X_test = pd.read_parquet(paths["X_test"])
    y_train = pd.read_parquet(paths["y_train"])["y"].to_numpy()
    return X_train, X_test, y_train


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["binary", "multiclass"])
def test_predict_proba_consistency(kind, device):
    """Classification probabilities must match the stored golden snapshot."""
    skip_if_device_unusable(device)
    X_train, X_test, y_train = _load_task(kind)

    clf = TabICLClassifier(device=device, **ESTIMATOR_KWARGS, **model_path("classifier"))
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

    expected_path = prediction_paths(kind)["proba"]
    if not os.path.exists(expected_path):
        raise FileNotFoundError(
            f"Missing golden predictions: {expected_path}. "
            "See tests/generate_data_and_predictions.py on how to generate them."
        )
    expected = np.load(expected_path)
    assert proba.shape == expected.shape
    np.testing.assert_allclose(proba, expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("device", DEVICES)
def test_predict_regression_consistency(device):
    """Regression mean and quantile predictions must match the stored golden snapshots."""
    skip_if_device_unusable(device)
    X_train, X_test, y_train = _load_task("regression")

    reg = TabICLRegressor(device=device, **ESTIMATOR_KWARGS, **model_path("regressor"))
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test, output_type=["mean", "quantiles"], alphas=QUANTILE_ALPHAS)
    mean_pred = preds["mean"]
    quantiles = preds["quantiles"]

    paths = prediction_paths("regression")    
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing golden predictions: {missing}. "
            "See tests/generate_data_and_predictions.py on how to generate them."
        )
    expected_mean = np.load(paths["mean"])
    expected_quantiles = np.load(paths["quantiles"])

    assert mean_pred.shape == (N_TEST,)
    assert mean_pred.shape == expected_mean.shape
    np.testing.assert_allclose(mean_pred, expected_mean, rtol=RTOL, atol=ATOL)

    assert quantiles.shape == (N_TEST, len(QUANTILE_ALPHAS))
    assert quantiles.shape == expected_quantiles.shape
    np.testing.assert_allclose(quantiles, expected_quantiles, rtol=RTOL, atol=ATOL)
