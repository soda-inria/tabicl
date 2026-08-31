import warnings

import numpy as np
import pytest
import torch
from sklearn.base import clone, is_classifier
from sklearn.datasets import make_classification, make_friedman1
from sklearn.metrics import log_loss, r2_score
from sklearn.model_selection import train_test_split

from tabicl import TabICLClassifier, TabICLRegressor
from tabicl._model.inference import InferenceManager
from tabicl._sklearn.preprocessing import UniqueFeatureFilter
from tabicl._torch_devices import (
    MPS_NUMERICS_ISSUE_URL,
    resolve_default_device,
    resolve_torch_device,
)
from tests.torch_devices_helpers import skip_if_device_unusable


_DEVICES = ["cpu", "cuda", "xpu", "mps"]


def _patch_backend_availability(
    monkeypatch, *, cuda_available, xpu_available, mps_available
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

from conftest import model_path


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
    "estimator",
    [
        TabICLClassifier(random_state=0, **model_path("classifier")),
        TabICLRegressor(random_state=0, **model_path("regressor")),
    ],
)
def test_tabicl_supports_float16(estimator):
    """float16 arrays should not crash the Yeo-Johnson normalizer (issue #140)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 4)).astype(np.float16)

    est = clone(estimator)
    if is_classifier(est):
        y = rng.integers(0, 2, size=50)
    else:
        y = rng.standard_normal(50)

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
    _patch_backend_availability(
        monkeypatch,
        cuda_available=cuda_available,
        xpu_available=xpu_available,
        mps_available=mps_available,
    )
    # Preference-order tests assume healthy MPS hardware.
    monkeypatch.setattr(
        "tabicl._torch_devices.mps_possibly_faulty", lambda: False
    )

    assert resolve_default_device().type == expected_device

    est = estimator_cls(random_state=0)
    est._resolve_device()
    assert est.device_.type == expected_device

    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    mgr.configure(device=None, use_amp=False, use_fa3=False, use_async=False)
    assert mgr.exe_device.type == expected_device


def test_resolve_default_device_falls_back_from_faulty_mps(monkeypatch):
    _patch_backend_availability(
        monkeypatch, cuda_available=False, xpu_available=False, mps_available=True
    )
    monkeypatch.setattr("tabicl._torch_devices.mps_possibly_faulty", lambda: True)

    with pytest.warns(
        RuntimeWarning, match=rf"virtualized Apple Silicon.*{MPS_NUMERICS_ISSUE_URL}.*Falling back to CPU"
    ):
        assert resolve_default_device().type == "cpu"

    est = TabICLClassifier(random_state=0, device=None)
    with pytest.warns(RuntimeWarning, match="Falling back to CPU"):
        est._resolve_device()
    assert est.device_.type == "cpu"

    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    with pytest.warns(RuntimeWarning, match="Falling back to CPU"):
        mgr.configure(device=None, use_amp=False, use_fa3=False, use_async=False)
    assert mgr.exe_device.type == "cpu"


def test_resolve_explicit_mps_warns_but_keeps_mps_on_faulty_host(monkeypatch):
    _patch_backend_availability(
        monkeypatch, cuda_available=False, xpu_available=False, mps_available=True
    )
    monkeypatch.setattr("tabicl._torch_devices.mps_possibly_faulty", lambda: True)

    with pytest.warns(
        RuntimeWarning,
        match=rf"device='mps' was requested.*{MPS_NUMERICS_ISSUE_URL}.*device='cpu'",
    ):
        resolved = resolve_torch_device("mps")
    assert resolved.type == "mps"

    est = TabICLClassifier(random_state=0, device="mps")
    with pytest.warns(RuntimeWarning, match="device='mps' was requested"):
        est._resolve_device()
    assert est.device_.type == "mps"

    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    with pytest.warns(RuntimeWarning, match="device='mps' was requested"):
        mgr.configure(device="mps", use_amp=False, use_fa3=False, use_async=False)
    assert mgr.exe_device.type == "mps"


def test_resolve_mps_no_warning_on_healthy_host(monkeypatch):
    _patch_backend_availability(
        monkeypatch, cuda_available=False, xpu_available=False, mps_available=True
    )
    monkeypatch.setattr("tabicl._torch_devices.mps_possibly_faulty", lambda: False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_default_device().type == "mps"
        assert resolve_torch_device("mps").type == "mps"

    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


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


@pytest.mark.parametrize(
    "estimator_cls",
    [
        TabICLClassifier,
        TabICLRegressor,
    ],
)
@pytest.mark.parametrize(
    "device_type, n_samples, n_features, use_amp, expected_fa3",
    [
        # FA3 auto is CUDA-only.
        ("cpu", 20000, 100, "auto", False),
        ("mps", 20000, 100, "auto", False),
        ("xpu", 20000, 100, "auto", False),
        ("cuda", 100, 10, "auto", False),  # small data
        ("cuda", 2000, 100, "auto", False),  # medium + AMP on → FA3 off
        ("cuda", 20000, 100, "auto", True),  # large + AMP on → FA3 on
        ("cuda", 2000, 100, False, True),  # medium + AMP off → FA3 fallback
        ("mps", 2000, 100, False, False),  # AMP off does not enable FA3 off CUDA
    ],
)
def test_resolve_fa3_auto_is_cuda_only(
    estimator_cls, device_type, n_samples, n_features, use_amp, expected_fa3
):
    est = estimator_cls(random_state=0, use_amp=use_amp, use_fa3="auto")
    est.device_ = torch.device(device_type)
    est.n_samples_in_ = n_samples
    est.n_features_in_ = n_features
    _, use_fa3 = est._resolve_amp_fa3()
    assert use_fa3 is expected_fa3


def test_resolve_fa3_explicit_true_preserved_off_cuda():
    est = TabICLClassifier(random_state=0, use_amp=False, use_fa3=True)
    est.device_ = torch.device("mps")
    est.n_samples_in_ = 100
    est.n_features_in_ = 10
    _, use_fa3 = est._resolve_amp_fa3()
    assert use_fa3 is True


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


def test_tabicl_fits_all_constant_features():
    """No feature is informative: one column is kept, so fit falls back to the marginal."""

    X = np.tile(np.array([0.4, 1.0, 0.5, 118.2]), (6, 1))
    y = np.array([-3.2, -1.0, 0.5, 2.0, -0.7, 1.1])

    # Constant features are still dropped as long as one informative feature remains.
    assert UniqueFeatureFilter().fit_transform(np.c_[np.ones(6), y]).shape == (6, 1)

    y_pred = TabICLRegressor(n_estimators=4, random_state=0).fit(X, y).predict(X)

    assert np.allclose(y_pred, y_pred[0])
    assert abs(y_pred[0] - y.mean()) < y.std()
