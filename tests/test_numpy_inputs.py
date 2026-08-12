import numpy as np
import pytest
import torch
from sklearn.base import clone, is_classifier
from sklearn.datasets import make_classification, make_friedman1
from sklearn.metrics import log_loss, r2_score
from sklearn.model_selection import train_test_split

from src.tabicl import TabICLClassifier, TabICLRegressor
from tabicl._model.inference import InferenceManager
from tabicl._torch_devices import resolve_default_device
from tests.torch_devices import skip_if_device_unusable


_DEVICES = ["cpu", "cuda", "xpu", "mps"]

@pytest.mark.parametrize(
    "estimator",
    [
        TabICLClassifier(random_state=0),
        TabICLRegressor(random_state=0),
    ],
)
def test_tabicl_supports_nans(estimator):
    est = clone(estimator)

    X = np.array(
        [
            [1.0, np.nan, 3.0],
            [4.0, 5.0, np.nan],
            [7.0, 8.0, 9.0],
            [np.nan, 11.0, 12.0],
        ],
        dtype=float,
    )

    if is_classifier(est):
        y = np.array([0, 1, 0, 1])
    else:
        y = np.array([0.1, 1.2, 2.3, 3.4], dtype=float)

    est.fit(X, y)
    y_pred = est.predict(X)

    assert y_pred.shape == y.shape


@pytest.mark.parametrize(
    "X",
    [
        np.array(
            [
                [True, False, True],
                [False, True, False],
                [True, True, False],
                [False, False, True],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 2.5, 3],
                [4, 5.5, 6],
                [7, 8.5, 9],
                [10, 11.5, 12],
            ],
            dtype=object,
        ),
        np.array(
            [
                ["1.0", "2.0", "3.0"],
                ["4.0", "5.0", "6.0"],
                ["7.0", "8.0", "9.0"],
                ["10.0", "11.0", "12.0"],
            ],
            dtype=str,
        ),
    ],
    ids=["bool", "object", "string"],
)
@pytest.mark.parametrize(
    "estimator",
    [
        TabICLClassifier(random_state=0),
        TabICLRegressor(random_state=0),
    ],
)
@pytest.mark.parametrize("device", _DEVICES)
def test_tabicl_supports_bool_object_and_string_inputs(estimator, X, device):
    skip_if_device_unusable(device)

    est = clone(estimator).set_params(device=device)

    if is_classifier(est):
        y = np.array([0, 1, 0, 1])
    else:
        y = np.array([0.1, 1.2, 2.3, 3.4], dtype=float)

    est.fit(X, y)
    y_pred = est.predict(X)

    assert y_pred.shape == y.shape


@pytest.mark.parametrize(
    "estimator_cls",
    [
        TabICLClassifier,
        TabICLRegressor,
    ],
)
@pytest.mark.parametrize(
    "cuda_available, xpu_available, mps_available, expected_device",
    [
        (True, True, True, "cuda"),
        (True, False, True, "cuda"),
        (False, True, True, "xpu"),
        (False, True, False, "xpu"),
        (False, False, True, "mps"),
        (False, False, False, "cpu"),
    ],
)
def test_tabicl_default_device_selection(
    monkeypatch, estimator_cls, cuda_available, xpu_available, mps_available, expected_device
):
    class _FakeXPUBackend:
        @staticmethod
        def is_available():
            return xpu_available

    class _FakeMPSBackend:
        @staticmethod
        def is_available():
            return mps_available

    monkeypatch.setattr(torch, "xpu", _FakeXPUBackend, raising=False)
    monkeypatch.setattr(torch, "mps", _FakeMPSBackend, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    assert resolve_default_device().type == expected_device

    est = estimator_cls(random_state=0)
    est._resolve_device()
    assert est.device_.type == expected_device

    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    mgr.configure(device=None, use_amp=False, use_fa3=False, use_async=False)
    assert mgr.exe_device.type == expected_device


@pytest.mark.parametrize(
    "estimator_cls",
    [
        TabICLClassifier,
        TabICLRegressor,
    ],
)
@pytest.mark.parametrize(
    "device_type, n_samples, n_features, expected_amp",
    [
        ("cpu", 2000, 100, False),
        ("mps", 2000, 100, True),
        ("cuda", 2000, 100, True),
        ("xpu", 2000, 100, True),
        ("mps", 100, 10, False),  # small-data heuristic
        ("cuda", 100, 10, False),
    ],
)
def test_resolve_amp_is_device_aware(
    estimator_cls, device_type, n_samples, n_features, expected_amp
):
    est = estimator_cls(random_state=0, use_amp="auto", use_fa3="auto")
    est.device_ = torch.device(device_type)
    est.n_samples_in_ = n_samples
    est.n_features_in_ = n_features
    use_amp, _ = est._resolve_amp_fa3()
    assert use_amp is expected_amp


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("kv_cache", [False, True])
@pytest.mark.parametrize("use_amp", [False, True])
def test_tabicl_regressor_device_cpu_r2_parity(device, kv_cache, use_amp):
    """Accelerator predictions should roughly match CPU on a small regression task."""
    skip_if_device_unusable(device)

    # Float16 AMP introduces cross-device numeric drift; fp32 should match tightly.
    if use_amp:
        rtol, atol, score_tol = 1e-1, 5e-1, 1e-2
    else:
        rtol, atol, score_tol = 1e-4, 1e-4, 1e-5

    X, y = make_friedman1(n_samples=200, n_features=20, noise=1.0, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0
    )

    preds = {}
    scores = {}
    for d in ("cpu", device):
        reg = TabICLRegressor(
            n_estimators=2,
            device=d,
            kv_cache=kv_cache,
            use_amp=use_amp,
            use_fa3=False,
            random_state=0,
            verbose=False,
        )
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)
        preds[d] = pred
        scores[d] = r2_score(y_test, pred)

    assert abs(scores["cpu"] - scores[device]) < score_tol
    np.testing.assert_allclose(preds["cpu"], preds[device], rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", ["cuda", "xpu", "mps"])
@pytest.mark.parametrize("kv_cache", [False, True])
@pytest.mark.parametrize("use_amp", [False, True])
def test_tabicl_classifier_device_cpu_logloss_parity(device, kv_cache, use_amp):
    """Accelerator probabilities should roughly match CPU on a small classification task."""
    skip_if_device_unusable(device)

    # Float16 AMP introduces cross-device numeric drift; fp32 should match tightly.
    if use_amp:
        rtol, atol, score_tol = 1e-1, 1e-1, 1e-2
    else:
        rtol, atol, score_tol = 1e-4, 1e-4, 1e-5

    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=2,
        n_classes=3,
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0
    )

    probas = {}
    scores = {}
    for d in ("cpu", device):
        clf = TabICLClassifier(
            n_estimators=2,
            device=d,
            kv_cache=kv_cache,
            use_amp=use_amp,
            use_fa3=False,
            random_state=0,
            verbose=False,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        probas[d] = proba
        scores[d] = log_loss(y_test, proba, labels=clf.classes_)

    assert abs(scores["cpu"] - scores[device]) < score_tol
    np.testing.assert_allclose(probas["cpu"], probas[device], rtol=rtol, atol=atol)
