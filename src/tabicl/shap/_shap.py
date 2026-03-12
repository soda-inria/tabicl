"""SHAP explanations for TabICL.

TabICL treats NaN columns as absent features. This module exploits that
by using a single all-NaN row as the SHAP background, so that masked
features are genuinely removed from the model's perspective rather than
replaced by some reference value.

Example::

    from tabicl import TabICLClassifier
    from tabicl.shap import get_shap_values, get_shap_explainer, plot_shap

    clf = TabICLClassifier().fit(X_train, y_train)
    sv = get_shap_values(clf, X_test)
    plot_shap(sv)
"""

from __future__ import annotations
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
    X_np = np.asarray(X_test)

    predict_fn = "predict_proba" if hasattr(estimator, "predict_proba") else "predict"
    explainer = get_shap_explainer(estimator, X_np, predict_fn=predict_fn, **kwargs)
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


def plot_shap(shap_values: Any) -> None:
    """Bar plot + beeswarm + top-feature interaction scatter.

    Parameters
    ----------
    shap_values : shap.Explanation
        Typically returned by :func:`get_shap_values`.
    """
    # For multi-output (e.g. multi-class), take the first output slice.
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 0]

    shap.plots.bar(shap_values=shap_values, show=False)
    plt.title("Aggregate feature importances across the test examples")
    plt.show()

    shap.summary_plot(shap_values=shap_values, show=False)
    plt.title("Feature importances for each feature for each test example " "(a dot is one feature for one example)")
    plt.show()

    if len(shap_values) > 1:
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
