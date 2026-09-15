"""Prediction must restore the caller's PyTorch thread count, including on errors."""

import multiprocessing as mp
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import model_path

from tabicl import TabICLClassifier, TabICLRegressor


@pytest.fixture(autouse=True)
def restore_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(original)


def make_estimator(kind, cached=False, n_jobs=1):
    """Use real estimator preprocessing/aggregation with a stub model forward."""
    cls = TabICLClassifier if kind == "classifier" else TabICLRegressor
    estimator = cls(n_jobs=n_jobs)
    estimator.n_features_in_ = 2
    estimator.X_encoder_ = SimpleNamespace(transform=lambda X: X)
    estimator.y_scaler_ = SimpleNamespace(inverse_transform=lambda X: X)
    estimator.ensemble_generator_ = SimpleNamespace(
        X_=np.zeros((2, 2)),
        transform=lambda X, mode: {"none": (X[None],) if mode == "test" else (X[None], np.zeros((1, 2)))},
        feature_shuffles_={"none": [[0, 1]]},
        class_shuffles_={"none": [[0, 1]]},
    )
    if cached:
        estimator.model_kv_cache_ = {"none": object()}
    output = np.array([[[0.0, 1.0], [1.0, 0.0]]]) if kind == "classifier" else np.array([[1.0, 2.0]])
    estimator._batch_forward = lambda *args, **kwargs: output
    estimator._batch_forward_with_cache = lambda *args, **kwargs: output
    return estimator


def predict(estimator, X):
    if isinstance(estimator, TabICLClassifier):
        return estimator.predict_proba(X)
    return estimator.predict(X)


@pytest.mark.parametrize("kind", ["classifier", "regressor"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("failure", ["validation", "encoding", "inference"])
def test_prediction_failure_restores_thread_count(kind, cached, failure):
    estimator = make_estimator(kind, cached=cached)
    X = np.ones((2, 2))

    def fail(*args, **kwargs):
        assert torch.get_num_threads() == 1
        raise RuntimeError("prediction failed")

    if failure == "validation":
        X = np.ones((2, 3))
        error, message = ValueError, "features"
    else:
        error, message = RuntimeError, "prediction failed"
        if failure == "encoding":
            estimator.X_encoder_.transform = fail
        else:
            estimator._batch_forward = fail
            estimator._batch_forward_with_cache = fail

    with pytest.raises(error, match=message):
        predict(estimator, X)

    assert torch.get_num_threads() == 2


@pytest.mark.parametrize("kind", ["classifier", "regressor"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(("n_jobs", "expected_threads"), [(None, 2), (1, 1), (-1, 4), (-2, 3), (-10, 1)])
def test_success_uses_requested_threads_and_restores_them(kind, cached, n_jobs, expected_threads, monkeypatch):
    monkeypatch.setattr(mp, "cpu_count", lambda: 4)
    estimator = make_estimator(kind, cached=cached, n_jobs=n_jobs)

    def transform(X):
        assert torch.get_num_threads() == expected_threads
        return X

    estimator.X_encoder_.transform = transform
    result = predict(estimator, np.ones((2, 2)))
    assert torch.get_num_threads() == 2
    if kind == "classifier":
        expected = estimator.softmax(np.array([[0.0, 1.0], [1.0, 0.0]]), temperature=estimator.softmax_temperature)
    else:
        expected = np.array([1.0, 2.0])
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("kind", ["classifier", "regressor"])
def test_too_many_threads_still_warns_and_caps_at_cpu_count(kind, monkeypatch):
    monkeypatch.setattr(mp, "cpu_count", lambda: 1)
    estimator = make_estimator(kind, n_jobs=3)

    def transform(X):
        assert torch.get_num_threads() == 1
        return X

    estimator.X_encoder_.transform = transform
    with pytest.warns(UserWarning, match="only 1 logical cores"):
        predict(estimator, np.ones((2, 2)))
    assert torch.get_num_threads() == 2


@pytest.mark.parametrize("kind", ["classifier", "regressor"])
def test_default_n_jobs_does_not_set_threads(kind, monkeypatch):
    estimator = make_estimator(kind, n_jobs=None)
    with monkeypatch.context() as patch:
        patch.setattr(torch, "set_num_threads", lambda count: pytest.fail("n_jobs=None should not set threads"))
        predict(estimator, np.ones((2, 2)))


@pytest.mark.parametrize("kind", ["classifier", "regressor"])
@pytest.mark.parametrize("kv_cache", [False, "kv", "repr"])
def test_real_model_recovers_after_invalid_prediction(kind, kv_cache):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 3))
    y = np.tile([0, 1], 10) if kind == "classifier" else X[:, 0] - 2 * X[:, 1]
    cls = TabICLClassifier if kind == "classifier" else TabICLRegressor
    estimator = cls(
        n_estimators=2, n_jobs=1, device="cpu", use_amp=False, kv_cache=kv_cache, **model_path(kind)
    ).fit(X, y)
    before = predict(estimator, X[:2])
    assert torch.get_num_threads() == 2

    with pytest.raises(ValueError, match="features"):
        predict(estimator, np.ones((2, 4)))

    assert torch.get_num_threads() == 2
    np.testing.assert_array_equal(predict(estimator, X[:2]), before)
    assert torch.get_num_threads() == 2
