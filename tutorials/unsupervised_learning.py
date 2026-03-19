"""
Unsupervised learning with TabICL
==================================

This tutorial demonstrates how to use TabICL for unsupervised tasks: density
estimation, outlier detection, missing-value imputation, and synthetic data
generation.
"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from tabicl import TabICLUnsupervised


# %%
# Generate 2D data and fit the model
# -----------------------------------
#
# We use the classic two-moon dataset — the same family used in the
# classification tutorial — but with only 200 samples so inference stays fast.
#
# ``TabICLUnsupervised`` decomposes the joint density via the chain rule of
# probability. Under the hood, each conditional ``P(X_k | X_{<k})`` is
# predicted by a TabICL classifier (categorical features) or regressor
# (numerical features). Calling ``fit()`` stores the training data and loads
# the shared model weights once.

X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

model = TabICLUnsupervised(
    n_estimators=4,
    categorical_features=[],
    device="cpu",
    random_state=42,
)
model.fit(X)

# Shared axis limits so all 2D scatter plots are directly comparable.
pad = 1.0
xlim = (X[:, 0].min() - pad, X[:, 0].max() + pad)
ylim = (X[:, 1].min() - pad, X[:, 1].max() + pad)


# %%
# Outlier detection with ``score_samples()``
# -------------------------------------------
#
# ``score_samples()`` estimates the joint density by averaging chain-rule
# log-probabilities over several random feature orderings. Higher scores
# indicate more typical data points; lower scores flag outliers.
#
# We first evaluate the density on a 2D grid to visualise where the model
# places probability mass.

h = 0.2
x_min, x_max = xlim
y_min, y_max = ylim
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_grid = np.c_[xx.ravel(), yy.ravel()]

scores_grid = model.score_samples(X_grid, n_permutations=4)

fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
cf = ax.contourf(
    xx,
    yy,
    scores_grid.reshape(xx.shape),
    levels=20,
    cmap="YlOrRd",
    alpha=0.85,
)
ax.scatter(X[:, 0], X[:, 1], c="black", s=10, alpha=0.6, label="Training data")
ax.set(xlabel="Feature 1", ylabel="Feature 2", xlim=xlim, ylim=ylim)
plt.colorbar(cf, ax=ax, label="Density score")
ax.legend(frameon=False)
plt.show()


# %%
#
# Now let's inject a handful of outliers and compare their scores to those of
# the normal training points.

outliers = np.array([[-1.2, 1.2], [2.2, -0.8], [0.5, 1.8], [-1.5, -0.8], [2.5, 1.0]])
X_all = np.vstack([X, outliers])
is_outlier = np.array([False] * len(X) + [True] * len(outliers))

scores_all = model.score_samples(X_all, n_permutations=4)

print(f"Normal score range:  [{scores_all[~is_outlier].min():.4f}, " f"{scores_all[~is_outlier].max():.4f}]")
print(f"Outlier score range: [{scores_all[is_outlier].min():.4f}, " f"{scores_all[is_outlier].max():.4f}]")

fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
sc = ax.scatter(
    X_all[:, 0],
    X_all[:, 1],
    c=np.log1p(scores_all),
    cmap="YlOrRd",
    s=30,
    edgecolors="none",
)
ax.scatter(
    outliers[:, 0],
    outliers[:, 1],
    facecolors="none",
    edgecolors="blue",
    s=120,
    linewidths=2,
    label="Outliers",
)
ax.set(xlabel="Feature 1", ylabel="Feature 2")
ax.set_title("Log-density scores (outliers circled)")
plt.colorbar(sc, ax=ax, label="log(1 + score)")
ax.legend(frameon=False)
plt.show()


# %%
# Synthetic data generation with ``generate()``
# -----------------------------------------------
#
# ``generate()`` autoregressively samples new data from the learned density:
# each feature is drawn from ``P(X_k | X_{<k})`` using the fitted conditionals.
# The ``temperature`` parameter controls diversity — values near 0 give
# near-deterministic (mode) samples, while 1.0 gives full distribution
# sampling.

X_synth = model.generate(n_samples=200, temperature=1.0)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)

axes[0].scatter(X[:, 0], X[:, 1], s=10, alpha=0.6)
axes[0].set_title("Real data")

axes[1].scatter(X_synth[:, 0], X_synth[:, 1], s=10, alpha=0.6, color="C1")
axes[1].set_title("Synthetic data (temperature=1.0)")

for ax in axes:
    ax.set(xlabel="Feature 1", ylabel="Feature 2", xlim=xlim, ylim=ylim)
    ax.spines[["top", "right"]].set_visible(False)

plt.show()


# %%
#
# A quick temperature sweep shows the effect: low temperatures concentrate
# samples around high-density modes, while ``temperature = 1.0`` reproduces the full
# spread.

temperatures = [0.01, 0.5, 1.0]
fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)

for ax, t in zip(axes, temperatures):
    X_t = model.generate(n_samples=200, temperature=t)
    ax.scatter(X_t[:, 0], X_t[:, 1], s=10, alpha=0.6)
    ax.set_title(f"t = {t}")
    ax.set(xlabel="Feature 1", ylabel="Feature 2", xlim=xlim, ylim=ylim)
    ax.spines[["top", "right"]].set_visible(False)

plt.show()


# %%
# Missing-value imputation with ``impute()``
# --------------------------------------------
#
# ``impute()`` fills NaN values by conditioning on all observed features. A low
# temperature (near 0) gives deterministic imputation close to the conditional
# median; temperature 1.0 would draw stochastic samples.
#
# We randomly mask one feature per row for ~50 % of the rows, so the model
# always has a conditioning signal.

rng = np.random.default_rng(0)

# For each selected row, mask exactly one of the two features.
mask = np.zeros(X.shape, dtype=bool)
masked_rows = rng.choice(len(X), size=100, replace=False)
masked_col = rng.integers(0, 2, size=len(masked_rows))
mask[masked_rows, masked_col] = True

X_masked = X.copy()
X_masked[mask] = np.nan

is_observed = ~mask.any(axis=1)
is_partial = mask.any(axis=1)

print(f"Rows: {len(X)} total, {is_partial.sum()} with missing values, " f"{is_observed.sum()} fully observed")
print(f"Cells: {mask.sum()} / {mask.size} missing ({100 * mask.mean():.0f} %)")

X_imputed = model.impute(X_masked, temperature=0.5)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)

axes[0].scatter(X[:, 0], X[:, 1], s=20, alpha=0.7)
axes[0].set_title("Original")

axes[1].scatter(
    X[is_observed, 0],
    X[is_observed, 1],
    s=20,
    alpha=0.5,
    label="observed",
)
axes[1].scatter(
    X[is_partial, 0],
    X[is_partial, 1],
    s=40,
    marker="s",
    alpha=0.7,
    label="partial NaN",
)
axes[1].legend(frameon=False, fontsize=8)
axes[1].set_title("Missing pattern")

axes[2].scatter(
    X_imputed[is_observed, 0],
    X_imputed[is_observed, 1],
    s=20,
    alpha=0.5,
    label="observed",
)
axes[2].scatter(
    X_imputed[is_partial, 0],
    X_imputed[is_partial, 1],
    s=40,
    marker="s",
    alpha=0.7,
    color="C2",
    label="imputed",
)
axes[2].legend(frameon=False, fontsize=8)
axes[2].set_title("After imputation")

for ax in axes:
    ax.set(xlabel="$x_1$", ylabel="$x_2$", xlim=xlim, ylim=ylim)
    ax.spines[["top", "right"]].set_visible(False)

plt.show()

# %%
