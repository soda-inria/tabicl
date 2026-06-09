"""
Quantile regression with TabICL
===============================

This example shows the TabICL predicted quantiles on a simple 1D regression
problem with heteroscedastic noise.
"""

# %% Imports

from functools import partial
import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tabicl import TabICLRegressor

# %%
# Generate heteroscedastic data
# -----------------------------
#
# We first generate a simple one-dimensional regression task to illustrate the
# natural ability of TabICL to model predictive uncertainty in the presence of
# heteroscedastic noise.
#
# `Heteroscedasticity
# <https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity>`_
# means that the variance of the target random variable `y` is not constant
# over the feature space: there are regions of  `x` for which `y` is much
# harder to predict y given x than for other regions.
#
# The following data generating process is heteroscedastic because `true_y_std`
# is not constant: it is defined as a function of `x`.

rng = np.random.default_rng(0)
n_samples = int(3e3)
x = rng.uniform(low=-3, high=3, size=n_samples)
X = x.reshape((n_samples, 1))


def true_y_mean(x):
    return expit(x) - 0.5 - 0.1 * x


def true_y_std(x):
    return 0.07 * np.exp(-((x - 0.5) ** 2) / 0.9)


y = rng.normal(loc=true_y_mean(x), scale=true_y_std(x))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# %%
#
# Fit TabICL and estimate quantiles
# ---------------------------------
#
# The TabICL model is fitted with `n_estimators=1` to disable ensembling to speed-up
# the execution (at the cost of slightly worse predictions).
#
tabicl = TabICLRegressor(n_estimators=1)
tabicl.fit(X_train, y_train)

# %%
# We then estimate quantiles of the distribution of Y|X for each test data
# point. If the model is good, we expect 80% of the observed values of y to lie
# between the predicted lower (0.1) and upper (0.9) quantile values.
#
# The 0.5 quantile prediction estimates the median of Y|X: we expect 50% of the
# observation to lie above and the remaining 50% to lie below.

quantiles = tabicl.predict(X_test, output_type="quantiles", alphas=[0.10, 0.5, 0.90])
quantiles.shape


# %%
# Plot the quantiles predicted by TabICL
# --------------------------------------
#
# The predictions of the lower and upper quantiles are sorted by increasing
# value of `x` to define the boarder of a shaded area that represents the
# predictive uncertainty of the model on this dataset.


def plot_data_and_quantiles(X_test, y_test, quantiles):
    _, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)

    # Plot test data points
    ax.scatter(x=X_test, y=y_test, alpha=0.15, color="gray")

    # sort test points for a clean line plot (optional)
    order = np.argsort(X_test[:, 0])
    x_sorted = X_test[order, 0]

    # Plot median
    ax.plot(x_sorted, quantiles[order, 1], color="darkgreen", lw=3, label="median")

    # Plot 10-90% interval
    ax.fill_between(
        x_sorted,
        quantiles[order, 0],
        quantiles[order, 2],
        alpha=0.18,
        color="green",
        label="10–90% interval",
    )

    ax.legend(frameon=False)
    ax.set(xlabel="Input feature x", ylabel="Target variable y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("TabICL predicted quantiles")


plot_data_and_quantiles(X_test, y_test, quantiles)

# %%
# For a given `x`, the prediction interval is the range of values that the
# model is confident about the prediction of `y` for that `x`: it can be read
# by drawing a vertical line at `x` and reading the y-axis range that intersects
# with the shaded area in the plot above.
#
# We observe that the width of the prediction intervals is much larger for `x`
# values between -0.5 and 1.5 than elsewhere and naturally adapts to the noise
# level of `y` given `x`.
#
# For `x` values below -1. or above 2., the TabICL model is very confident
# (narrow prediction intervals): indeed we can observe that `y` is nearly a
# deterministic (noise free) function of `x` in those regions.


# %%
#
# Assessing the reliability of the quantile predictions
# -----------------------------------------------------
#
# The reliability of the predictions can be assessed by ploting reliability
# diagrams for each quantile level.
#
# A reliability diagram bins the test data sorted by predicted values and
# computes the observed quantiles for each bin. If the model is reliable, the
# observed quantiles should be close to the predicted quantiles and the
# reliability diagram should be close to the diagonal line.


def _compute_reliability_curve(df, n_bins, functional):
    """Compute mean predicted and observed quantile per bin.

    The bins are sorted by mean predicted value."""
    df_bin = pd.qcut(df["y_predicted"], q=n_bins)

    df_binned = df.assign(bin=df_bin)
    agg_df = df_binned.groupby("bin", observed=True).agg(
        y_predicted=("y_predicted", "mean"),
        y_observed=("y_observed", functional),
    )
    agg_df = agg_df.sort_values("y_predicted").reset_index(drop=True)
    return agg_df["y_predicted"].values, agg_df["y_observed"].values


def plot_quantile_reliability_diagram(
    y_observed,
    y_predicted,
    quantile_level,
    n_bins="auto",
    min_n_bins=5,
    max_n_bins=30,
    n_bootstrap=300,
    confidence_level=0.95,
    random_state=None,
    ax=None,
):
    """Plot reliability diagram with bootstrap confidence intervals."""
    rng = np.random.default_rng(random_state)
    n_samples = len(y_observed)
    functional = partial(np.quantile, q=quantile_level)

    if n_bins == "auto":
        n_bins_sturges = int(np.ceil(np.log2(n_samples) + 1))
        n_bins = max(min_n_bins, min(max_n_bins, n_bins_sturges))

    # Original reliability curve (main estimate)
    df = pd.DataFrame({"y_observed": y_observed, "y_predicted": y_predicted})
    x_orig, y_orig = _compute_reliability_curve(df, n_bins, functional)

    # Full bootstrap: compute entire reliability curve for each resample to
    # assess a confidence interval for the reliability curve itself.
    bootstrap_x = np.full((n_bootstrap, n_bins), np.nan)
    bootstrap_y = np.full((n_bootstrap, n_bins), np.nan)
    for bootstrap_idx in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        df_boot = df.iloc[idx].reset_index(drop=True)
        x_boot, y_boot = _compute_reliability_curve(df_boot, n_bins, functional)
        bootstrap_x[bootstrap_idx] = x_boot
        bootstrap_y[bootstrap_idx] = y_boot

    # Use the mean x across bootstraps as reference (handles variable bin centers).
    x_ref = np.nanmean(bootstrap_x, axis=0)
    alpha = 1 - confidence_level
    y_lower = np.nanquantile(bootstrap_y, alpha / 2, axis=0)
    y_upper = np.nanquantile(bootstrap_y, 1 - alpha / 2, axis=0)

    # Plot the reliability curve and the confidence interval.
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)

    range_min = min(x_orig.min(), y_orig.min())
    range_max = max(x_orig.max(), y_orig.max())
    offset = (range_max - range_min) * 0.05
    extended_range = [range_min - offset, range_max + offset]
    ax.plot(
        extended_range,
        extended_range,
        color="black",
        linestyle="--",
        label="Perfect reliability",
    )
    ax.fill_between(
        x_ref,
        y_lower,
        y_upper,
        alpha=0.2,
        color="C0",
        label=f"{100 * confidence_level:.0f}% bootstrap CI",
    )
    ax.plot(x_orig, y_orig, color="C0", lw=2, label="Model")
    ax.set_xlim(extended_range)
    ax.set_ylim(extended_range)
    ax.set_title(f"Reliability diagram for {quantile_level}-quantile regression")
    ax.set_xlabel(f"Predicted {quantile_level}-quantile")
    ax.set_ylabel(f"Observed {quantile_level}-quantile")
    ax.legend(frameon=False)


fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
plot_quantile_reliability_diagram(y_test, quantiles[:, 0], 0.1, ax=axs[0])
plot_quantile_reliability_diagram(y_test, quantiles[:, 1], 0.5, ax=axs[1])
plot_quantile_reliability_diagram(y_test, quantiles[:, 2], 0.9, ax=axs[2])


# %%
#
# We observe that all 3 quantile predictions produced by TabICL on this
# dataset look quite reliable as they are close to the diagonal line.
#
# Since we have limited test data, we use a bootstrap procedure to estimate the
# confidence interval of the reliability diagram itself. The resulting the
# confidence intervals of the reliability diagrams are still quite wide for
# this small test set.
#
# Increasing the number of training data points should improve the reliability
# of the TabICL quantile predictions.
#
# Increasing the number of test data points should shrink the confidence
# intervals of the reliability diagrams to be able to detect subtle reliability
# problems.
#
# Increasing both the number of training and test data points should therefore
# result in diagonal reliability diagrams with narrow confidence intervals. You
# can check that this is the case by increasing the number of training and test
# data points at the beginning of the notebook and rerunning the notebook.
