import numpy as np
import pytest
import torch
from sklearn.base import clone, is_classifier

from src.tabicl import TabICLClassifier, TabICLRegressor


def _device_available(device: str | torch.device | None) -> bool:
    """Return whether a torch device backend is available on this host.

    Uses torch's backend naming convention where a device type maps to
    ``torch.<device_type>`` exposing an ``is_available()`` function.
    """
    if device is None:
        # The default device is always available.
        return True
    try:
        device_type = torch.device(device).type
    except (TypeError, RuntimeError, ValueError):
        return False

    if device_type == "cpu":
        return True

    backend_api = getattr(torch, device_type, None)
    is_available = getattr(backend_api, "is_available", None)
    if not callable(is_available):
        return False
    return bool(is_available())


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
@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu", "mps", None])
def test_tabicl_supports_bool_object_and_string_inputs(estimator, X, device):
    if not _device_available(device):
        pytest.skip(f"{device} device is not available on this host")

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
    "xpu_available, cuda_available, expected_device",
    [
        (True, True, "cuda"),
        (True, False, "xpu"),
        (False, True, "cuda"),
        (False, False, "cpu"),
    ],
)
def test_tabicl_default_device_selection(monkeypatch, estimator_cls, xpu_available, cuda_available, expected_device):
    class _FakeXPUBackend:
        @staticmethod
        def is_available():
            return xpu_available

    monkeypatch.setattr(torch, "xpu", _FakeXPUBackend, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    est = estimator_cls(random_state=0)
    est._resolve_device()

    assert est.device_.type == expected_device