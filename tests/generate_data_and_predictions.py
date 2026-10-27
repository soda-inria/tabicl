"""Regenerate parquet tables and CPU golden predictions for prediction-consistency tests.

Run from the repo root after changing generation knobs or when fixtures are missing.
Tables are written under ``tests/data/prediction_consistency/``. Prediction
``.npy`` files go in a versioned subdir ``tabicl-vX.Y.Z`` matching the installed
release (e.g. ``tabicl-v2.0.0``).

Pin a specific PyPI release matching . Reuse an existing env's torch/sklearn
via ``--system-site-packages`` so you only need to install ``tabicl`` (replace 
``2.0.0`` for any other release)::

    python -m venv /tmp/tabicl-golden --system-site-packages
    /tmp/tabicl-golden/bin/pip install --force-reinstall --no-deps 'tabicl==2.0.0'
    /tmp/tabicl-golden/bin/python tests/generate_data_and_predictions.py

"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from importlib.metadata import version as pkg_version

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification, make_regression

# Allow ``from conftest import model_path`` when run as a script from the repo root.
sys.path.insert(0, os.path.dirname(__file__))
from conftest import model_path

from tabicl import TabICLClassifier, TabICLRegressor

ESTIMATOR_KWARGS = dict(
    n_estimators=2,
    use_amp=False,
    use_fa3=False,
    random_state=42,
    verbose=False,
)

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

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "prediction_consistency")
# Version whose checked-in prediction goldens the tests compare against.
GOLDEN_TABICL_VERSION = "2.0.0"
PRED_DIR = os.path.join(DATA_DIR, f"tabicl-v{GOLDEN_TABICL_VERSION}")


def task_paths(kind: str) -> dict[str, str]:
    return {
        "X_train": os.path.join(DATA_DIR, f"{kind}_X_train.parquet"),
        "X_test": os.path.join(DATA_DIR, f"{kind}_X_test.parquet"),
        "y_train": os.path.join(DATA_DIR, f"{kind}_y_train.parquet"),
    }


def prediction_paths(kind: str) -> dict[str, str]:
    if kind in ("binary", "multiclass"):
        return {"proba": os.path.join(PRED_DIR, f"{kind}_expected_proba.npy")}
    return {
        "mean": os.path.join(PRED_DIR, "regression_expected_mean.npy"),
        "quantiles": os.path.join(PRED_DIR, "regression_expected_quantiles.npy"),
    }


def _to_mixed_dataframe(X: np.ndarray, seed: int) -> pd.DataFrame:
    """Keep the first columns numeric; bin the rest into predictive categoricals.
    
    Numerical features are further scaled and offset with scales on
    different magnitudes to make the problem harder and ensure that an
    unintended change in the internal numeric feature preprocessing has
    a good chance to make the non-regression test fail.
    """
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
            n_classes=11,  # >10 to also include a case for the mixed-radix encoding
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


def _save_task(kind: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: np.ndarray) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = task_paths(kind)
    X_train.to_parquet(paths["X_train"], index=False)
    X_test.to_parquet(paths["X_test"], index=False)
    pd.DataFrame({"y": y_train}).to_parquet(paths["y_train"], index=False)


def generate_data_and_predictions() -> None:
    """Generate parquet tables and CPU golden predictions for all tasks."""

    if GOLDEN_TABICL_VERSION != pkg_version("tabicl"): 
        raise ValueError(
            "Golden tabicl version mismatch, make sure you installed version "
            f"{GOLDEN_TABICL_VERSION} or update GOLDEN_TABICL_VERSION in "
            "tests/generate_data_and_predictions.py accordingly."
        )
    os.makedirs(PRED_DIR, exist_ok=True)
    print("Generating data and predictions...")

    for kind in ("binary", "multiclass"):
        X_train, X_test, y_train = _generate_synthetic_task(kind)
        _save_task(kind, X_train, X_test, y_train)
        clf = TabICLClassifier(device="cpu", **ESTIMATOR_KWARGS, **model_path("classifier"))
        clf.fit(X_train, y_train)
        np.save(prediction_paths(kind)["proba"], clf.predict_proba(X_test))

    X_train, X_test, y_train = _generate_synthetic_task("regression")
    _save_task("regression", X_train, X_test, y_train)
    reg = TabICLRegressor(device="cpu", **ESTIMATOR_KWARGS, **model_path("regressor"))
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test, output_type=["mean", "quantiles"], alphas=QUANTILE_ALPHAS)
    paths = prediction_paths("regression")
    np.save(paths["mean"], preds["mean"])
    np.save(paths["quantiles"], preds["quantiles"])

    print(f"Wrote data to {DATA_DIR}")
    print(f"Wrote predictions to {PRED_DIR}")


if __name__ == "__main__":
    generate_data_and_predictions()
