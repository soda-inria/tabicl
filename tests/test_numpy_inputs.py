import os

import numpy as np
import pytest
from sklearn.base import clone, is_classifier

from src.tabicl import TabICLClassifier, TabICLRegressor

# Optional env var to point at a local checkpoint directory (e.g. when HF
# downloads are unavailable). When unset, estimators use default auto-download.
_CKPT_DIR = os.environ.get("TABICL_CHECKPOINT_DIR")


def _model_path(kind: str):
    """Return model_path kwarg dict for the given kind ('classifier'/'regressor')."""
    if _CKPT_DIR is None:
        return {}
    filenames = {"classifier": "tabicl-classifier-v2-20260212.ckpt", "regressor": "tabicl-regressor-v2-20260212.ckpt"}
    return {"model_path": os.path.join(_CKPT_DIR, filenames[kind])}


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
def test_tabicl_supports_bool_object_and_string_inputs(estimator, X):
    est = clone(estimator)

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
        TabICLClassifier(random_state=0, **_model_path("classifier")),
        TabICLRegressor(random_state=0, **_model_path("regressor")),
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
