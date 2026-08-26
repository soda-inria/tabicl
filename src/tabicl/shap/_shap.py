"""SHAP explanations for TabICL.

This module uses a single all-NaN row as the SHAP background. NaN values
are handled by TabICL's normal preprocessing (mean imputation for numeric
features, missing-category encoding for numerically-encoded categoricals),
providing a natural missing-value baseline for feature attribution.

Note: String/object categorical features must be numerically encoded before
calling get_shap_values (e.g., via pandas' get_dummies or sklearn's
OrdinalEncoder). See tutorials/interpretability.py for an example.

Example::

    from tabicl import TabICLClassifier
    from tabicl.shap import get_shap_values, get_shap_explainer, plot_shap

    clf = TabICLClassifier().fit(X_train, y_train)
    sv = get_shap_values(clf, X_test)
    plot_shap(sv)
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import shap


def get_shap_values(estimator: Any, X_test: np.ndarray, attribute_names: list[str] | None = None, **kwargs: Any) -> Any:
    """Compute SHAP values for a fitted estimator.

    Parameters
    ----------
    estimator : estimator object
        A fitted TabICL estimator (classifier or regressor).

    X_test : array-like or DataFrame
        Samples to explain.

    attribute_names : list of str, optional
        Feature names (inferred from DataFrame columns when possible).

    **kwargs
        Forwarded to :func:`get_shap_explainer`.

    Returns
    -------
    shap.Explanation
    """
    if hasattr(X_test, "columns") and attribute_names is None:
        attribute_names = list(X_test.columns)

    try:
        X_np = np.asarray(X_test, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "SHAP values require numeric input. Categorical features must be "
            "encoded before calling get_shap_values (e.g., via pd.get_dummies "
            "or OrdinalEncoder). See tutorials/interpretability.py for an example."
        ) from exc

    predict_fn = "predict_proba" if hasattr(estimator, "predict_proba") else "predict"
    explainer = get_shap_explainer(estimator, X_np, predict_fn=predict_fn, **kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
        sv = explainer(X_np)

    if attribute_names is not None and hasattr(sv, "feature_names"):
        sv.feature_names = list(attribute_names)
    return sv


def get_shap_explainer(
    estimator: Any, X: np.ndarray, predict_fn: str | Callable = "predict_proba", **kwargs: Any
) -> Any:
    """Build a ``shap.Explainer`` with an all-NaN background.

    Parameters
    ----------
    estimator : estimator object
        A fitted estimator.

    X : array-like
        Used only to infer ``n_features``.

    predict_fn : str or callable, default="predict_proba"
        Prediction method; resolved via ``getattr`` when a string.

    **kwargs
        Forwarded to ``shap.Explainer``.

    Returns
    -------
    shap.Explainer
    """
    if isinstance(predict_fn, str):
        predict_fn = getattr(estimator, predict_fn)

    return shap.Explainer(predict_fn, np.full((1, X.shape[1]), np.nan), **kwargs)


# ── visualisation helpers ───────────────────────────────────────────────


def plot_shap(shap_values: Any, kind: str | tuple[str, ...] = "bar") -> None:
    """Plot SHAP explanations.

    Parameters
    ----------
    shap_values : shap.Explanation
        Typically returned by :func:`get_shap_values`.

    kind : str or tuple of str, default="bar"
        Which plots to show. Any combination of ``"bar"``, ``"beeswarm"``,
        and ``"scatter"``.
    """
    if isinstance(kind, str):
        kind = (kind,)

    # For multi-output (e.g. multi-class), take the first output slice.
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 0]

    if "bar" in kind:
        shap.plots.bar(shap_values=shap_values, show=False)
        plt.title("Aggregate feature importances across the test examples")
        plt.tight_layout()
        plt.show()

    if "beeswarm" in kind:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*NumPy global RNG.*", category=FutureWarning)
            shap.summary_plot(shap_values=shap_values, show=False)
        plt.title("Per-sample feature importances")
        plt.tight_layout()
        plt.show()

    if "scatter" in kind and len(shap_values) > 1:
        top = shap_values.abs.mean(0).values.argsort()[-1]
        plot_shap_feature(shap_values, top)


def plot_shap_feature(shap_values: Any, feature: int | str, n_plots: int = 1) -> None:
    """Scatter plot of a single feature coloured by its top interactions.

    Parameters
    ----------
    shap_values : shap.Explanation

    feature : int or str
        Index or name of the feature to plot.

    n_plots : int, default=1
        How many interaction partners to show.
    """
    inds = shap.utils.potential_interactions(shap_values[:, feature], shap_values)
    for i in range(n_plots):
        shap.plots.scatter(
            shap_values[:, feature],
            color=shap_values[:, inds[i]],
            show=False,
        )
        plt.title(f"Feature {feature} coloured by feature {inds[i]}")
        plt.tight_layout()
