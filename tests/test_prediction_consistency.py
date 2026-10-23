"""Golden-value checks that predict_proba / mean / quantile outputs stay stable.

Fixtures under ``tests/data/prediction_consistency/`` store the train/test
tables (parquet) and expected predictions (``.npy``). Tables are built with
sklearn ``make_*`` (mixed categorical and numeric features with diverse ranges);
categoricals come from binning informative numerics with light noise.

Tests always load existing fixtures (and raise if any are missing). To rebuild
data and CPU prediction snapshots after intentional changes, call
:func:`regenerate_fixtures` manually from the repo root, e.g.::

    python -c "import sys; sys.path[:0] = ['.', 'tests']; from tests.test_prediction_consistency import regenerate_fixtures; regenerate_fixtures()"
"""

from __future__ import annotations

from pathlib import Path

import torch
# Windows CI disables oneDNN/MKLDNN (TABICL_DISABLE_MKLDNN=1) to avoid flaky
# "could not create a primitive" errors. Force the same here, so snapshots and CPU 
# assertions share one portable backend; Linux would otherwise use oneDNN by default.
torch.backends.mkldnn.enabled = False

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from conftest import model_path
from tabicl import TabICLClassifier, TabICLRegressor
from tests.torch_devices_helpers import skip_if_device_unusable

N_TRAIN = 1000
N_TEST = 50
N_SAMPLES = N_TRAIN + N_TEST
N_FEATURES = 10
# Continuous columns cycle through these scales (tiny → large magnitudes).
NUMERIC_SCALES = (0.01, 1.0, 25.0, 1e3, 1e5)
# Categoricals cycle through these cardinalities after binning.
CAT_CARDINALITIES = (2, 3, 5, 8)
N_CATEGORICAL = len(CAT_CARDINALITIES)
# Noise std relative to each column's std before binning into categoricals.
CAT_NOISE_SCALE = 0.15
N_NUMERIC = N_FEATURES - N_CATEGORICAL
QUANTILE_ALPHAS = [0.1, 0.5, 0.9]

# CPU + MPS are exercised in CI; other accelerators can be checked locally.
_DEVICES = ["cpu", "mps"]

DATA_DIR = Path(__file__).resolve().parent / "data" / "prediction_consistency"

_ESTIMATOR_KW = dict(
    n_estimators=8,
    use_amp=False,
    use_fa3=False,
    random_state=42,
    verbose=False,
)


def _allclose_tol(device: str) -> tuple[float, float]:
    """Return (rtol, atol) for comparing predictions to the CPU golden snapshot."""
    if device == "cpu":
        return 1e-5, 1e-6
    # Cross-device fp32 drift (see device↔CPU parity tests in test_numpy_inputs.py).
    return 1e-4, 1e-4


def _to_mixed_dataframe(X: np.ndarray, seed: int) -> pd.DataFrame:
    """Keep the first columns numeric; bin the rest into predictive categoricals."""
    rng = np.random.RandomState(seed)
    assert X.shape[1] == N_FEATURES
    data: dict[str, pd.Series | np.ndarray] = {}

    for i in range(N_NUMERIC):
        scale = NUMERIC_SCALES[i % len(NUMERIC_SCALES)]
        col = X[:, i]
        data[f"num_{i}"] = ((col - col.mean()) * scale + scale * 10).astype(np.float64)

    for i in range(N_CATEGORICAL):
        n_levels = CAT_CARDINALITIES[i % len(CAT_CARDINALITIES)]
        col = X[:, N_NUMERIC + i]
        col = col + CAT_NOISE_SCALE * col.std() * rng.randn(len(col))
        # Quantile bins keep levels roughly balanced while preserving rank signal.
        codes = pd.qcut(col, q=n_levels, labels=False, duplicates="drop")
        labels = np.array([f"c{i}_{int(c)}" for c in codes], dtype=object)
        data[f"cat_{i}"] = pd.Series(labels, dtype="category")

    return pd.DataFrame(data)


def _generate_synthetic_task(kind: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Build train/test tables for ``binary``, ``multiclass``, or ``regression``."""
    if kind == "binary":
        X, y = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=N_FEATURES,
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            random_state=0,
        )
        seed = 1000
    elif kind == "multiclass":
        X, y = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=N_FEATURES,
            n_redundant=0,
            n_repeated=0,
            n_classes=11, # >10 to also include a case for the mixed-radix encoding
            n_clusters_per_class=1,
            random_state=1,
        )
        seed = 2000
    elif kind == "regression":
        X, y = make_regression(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=N_FEATURES,
            noise=0.1,
            random_state=2,
        )
        seed = 3000
    else:
        raise ValueError(f"Unknown task kind: {kind!r}")

    df = _to_mixed_dataframe(X.astype(np.float64), seed=seed)
    assert sum(c.startswith("cat_") for c in df.columns) == N_CATEGORICAL
    assert sum(c.startswith("num_") for c in df.columns) == N_NUMERIC

    X_train, X_test = df.iloc[:N_TRAIN].copy(), df.iloc[N_TRAIN:].copy()
    y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]
    if kind != "regression":
        assert set(np.unique(y_test)).issubset(set(np.unique(y_train)))
        y_train = y_train.astype(np.int64)
    else:
        y_train = y_train.astype(np.float64)
    return X_train, X_test, y_train


def _task_paths(kind: str) -> dict[str, Path]:
    return {
        "X_train": DATA_DIR / f"{kind}_X_train.parquet",
        "X_test": DATA_DIR / f"{kind}_X_test.parquet",
        "y_train": DATA_DIR / f"{kind}_y_train.parquet",
    }


def _prediction_paths(kind: str) -> dict[str, Path]:
    if kind in ("binary", "multiclass"):
        return {"proba": DATA_DIR / f"{kind}_expected_proba.npy"}
    return {
        "mean": DATA_DIR / "regression_expected_mean.npy",
        "quantiles": DATA_DIR / "regression_expected_quantiles.npy",
    }


def _save_task(kind: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: np.ndarray) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = _task_paths(kind)
    X_train.to_parquet(paths["X_train"], index=False)
    X_test.to_parquet(paths["X_test"], index=False)
    pd.DataFrame({"y": y_train}).to_parquet(paths["y_train"], index=False)


def _load_task(kind: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    paths = _task_paths(kind)
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing prediction-consistency fixtures: {missing}. "
            "Regenerate with regenerate_fixtures() "
            '(e.g. python -c "import sys; sys.path[:0] = [\'.\', \'tests\']; '
            "from tests.test_prediction_consistency import regenerate_fixtures; "
            'regenerate_fixtures()").'
        )
    X_train = pd.read_parquet(paths["X_train"])
    X_test = pd.read_parquet(paths["X_test"])
    y_train = pd.read_parquet(paths["y_train"])["y"].to_numpy()
    return X_train, X_test, y_train


def regenerate_fixtures() -> None:
    """Generate parquet tables and CPU golden predictions for all tasks.

    Call manually from the repo root after changing generation knobs or when
    fixtures are missing::

        python -c "import sys; sys.path[:0] = ['.', 'tests']; from tests.test_prediction_consistency import regenerate_fixtures; regenerate_fixtures()"
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for kind in ("binary", "multiclass"):
        X_train, X_test, y_train = _generate_synthetic_task(kind)
        _save_task(kind, X_train, X_test, y_train)
        clf = TabICLClassifier(device="cpu", **_ESTIMATOR_KW, **model_path("classifier"))
        clf.fit(X_train, y_train)
        np.save(_prediction_paths(kind)["proba"], clf.predict_proba(X_test))

    X_train, X_test, y_train = _generate_synthetic_task("regression")
    _save_task("regression", X_train, X_test, y_train)
    reg = TabICLRegressor(device="cpu", **_ESTIMATOR_KW, **model_path("regressor"))
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test, output_type=["mean", "quantiles"], alphas=QUANTILE_ALPHAS)
    paths = _prediction_paths("regression")
    np.save(paths["mean"], preds["mean"])
    np.save(paths["quantiles"], preds["quantiles"])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("kind", ["binary", "multiclass"])
def test_predict_proba_consistency(kind, device):
    """Classification probabilities must match the stored golden snapshot."""
    skip_if_device_unusable(device)
    X_train, X_test, y_train = _load_task(kind)

    clf = TabICLClassifier(device=device, **_ESTIMATOR_KW, **model_path("classifier"))
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

    expected_path = _prediction_paths(kind)["proba"]
    if not expected_path.exists():
        raise FileNotFoundError(
            f"Missing golden snapshot {expected_path}. "
            "Regenerate with regenerate_fixtures()."
        )
    expected = np.load(expected_path)
    rtol, atol = _allclose_tol(device)
    assert proba.shape == expected.shape
    np.testing.assert_allclose(proba, expected, rtol=rtol, atol=atol)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", _DEVICES)
def test_predict_regression_consistency(device):
    """Regression mean and quantile predictions must match the stored golden snapshots."""
    skip_if_device_unusable(device)
    X_train, X_test, y_train = _load_task("regression")

    reg = TabICLRegressor(device=device, **_ESTIMATOR_KW, **model_path("regressor"))
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test, output_type=["mean", "quantiles"], alphas=QUANTILE_ALPHAS)
    mean_pred = preds["mean"]
    quantiles = preds["quantiles"]

    paths = _prediction_paths("regression")
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing golden snapshots {missing}. Regenerate with regenerate_fixtures()."
        )
    expected_mean = np.load(paths["mean"])
    expected_quantiles = np.load(paths["quantiles"])
    rtol, atol = _allclose_tol(device)

    assert mean_pred.shape == (N_TEST,)
    assert mean_pred.shape == expected_mean.shape
    np.testing.assert_allclose(mean_pred, expected_mean, rtol=rtol, atol=atol)

    assert quantiles.shape == (N_TEST, len(QUANTILE_ALPHAS))
    assert quantiles.shape == expected_quantiles.shape
    np.testing.assert_allclose(quantiles, expected_quantiles, rtol=rtol, atol=atol)
    assert np.all(quantiles[:, 0] <= quantiles[:, 1])
    assert np.all(quantiles[:, 1] <= quantiles[:, 2])
