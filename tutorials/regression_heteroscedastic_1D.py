"""
TabICL quantiles with heteroscedastic noise
============================================

This example shows the TabICL predicted quantiles on a simple 1D regression
problem with heteroscedastic noise.
"""

# %% Imports

import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from tabicl import TabICLRegressor

# %%
# Generate heteroscedastic data
# -----------------------------
#
# [Heteroscedasticity](https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity) means that the variance
# of the target random variable `y` is not constant over the feature
# space: there are regions of  `X` for which `y` is much harder to
# predict than for other.
rng = np.random.default_rng(0)
n_samples = int(3e3)
x = rng.uniform(low=-3, high=3, size=n_samples)
X = x.reshape((n_samples, 1))


def true_y_mean(x):
    return expit(x) - 0.5 - 0.1 * x


def true_y_std(x):
    return 0.07 * np.exp(-((x - 0.5) ** 2) / 0.9)


y = rng.normal(loc=true_y_mean(x), scale=true_y_std(x))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 2, random_state=0
)


# %%
#
# Fit TabICL
# ----------
tabicl = TabICLRegressor(n_estimators=1)
tabicl.fit(X_train, y_train)

alphas = [0.10, 0.5, 0.90]
quantiles = tabicl.predict(X_test, output_type="quantiles", alphas=alphas)


# %%
# Plot the quantiles predicted by TabICL
# --------------------------------------
#
# The shaded area represents the predictive uncertainty of the model on this
# dataset. We can observe that the width of the prediction interval
# is much larger for `X` values between -0.5 and 1.5 than elsewhere
# and naturally adapts to the noise level of `y` given `X`.

def plot_data_generating_process(
    x,
    y,
    plot_data=True,
    max_scatter_points=1_000,
    plot_010_quantile=True,
    plot_090_quantile=True,
    color="k",
    ax=None,
):
    x_grid = np.linspace(x.min(), x.max(), 100)
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    if plot_data:
        ax.scatter(
            x=x[:max_scatter_points],
            y=y[:max_scatter_points],
            alpha=0.15,
            color="gray"
        )
    if plot_090_quantile:
        ax.plot(
            x_grid,
            true_y_mean(x_grid) + norm.ppf(0.90) * true_y_std(x_grid),
            linestyle="--",
            linewidth=1,
            color=color,
        )
    if plot_010_quantile:
        ax.plot(
            x_grid,
            true_y_mean(x_grid) - norm.ppf(0.90) * true_y_std(x_grid),
            linestyle="--",
            linewidth=1,
            label=r"10-90% quantiles",
            color=color,
        )
    ax.legend(frameon=False)


def plot_tabicl_quantiles(X_test, quantiles, ax=None):
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)

    # sort test points for a clean line plot (optional)
    order = np.argsort(X_test[:, 0])
    x_sorted = X_test[order, 0]

    # Plot median
    ax.plot(x_sorted, quantiles[order, 1],
            color='darkgreen', lw=3, label="median")

    # Plot 10-90% interval
    ax.fill_between(x_sorted, quantiles[order, 0],
                    quantiles[order, 2], alpha=0.18,
                    color="green", label="10–90% interval")

    ax.legend(frameon=False)


# Plot TabICL predicted quantiles
plt.rcParams["font.size"] = 13
fig, ax = plt.subplots()

ax.set(xlabel="Input feature x", ylabel="Target variable y")
ax.spines[['top', 'right']].set_visible(False)

plot_data_generating_process(
    X_test, y_test, ax=ax,
    plot_010_quantile=False,
    plot_090_quantile=False
)

plot_tabicl_quantiles(X_test, quantiles, ax=ax)
ax.set_title("TabICL predicted quantiles")
