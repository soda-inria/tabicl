"""
Probabilistic classification
===================================

This example shows TabICL predicted class probabilities on a simple 2D
classification problem.
"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve
from tabicl import TabICLClassifier


# %%
# Generate 2D classification data
# --------------------------------
# We generate a simple two‑moon 2D dataset with fairly large noise so that the
# classes are not separable. In two dimensions it is easy to visualise the
# output of a classifier.

rng = np.random.default_rng(0)
X, y = make_moons(n_samples=400, noise=0.35, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 2, random_state=0
)

# %%
# Fit TabICL
# -----------

tabicl = TabICLClassifier()
tabicl.fit(X_train, y_train)

# Predict probabilities on test set
y_proba = tabicl.predict_proba(X_test)

# %%
# Plot predicted probabilities
# -----------------------------
# Points are coloured by their true label.
# The black contour line shows the decision boundary at a probability threshold of 0.5.
# The colour shading indicates the estimated probability for class 1.

fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)

# Create a mesh to plot decision boundaries
h = 0.2
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict probabilities on mesh
Z = tabicl.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Plot decision boundary and margins
ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

# Plot training data
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                     cmap='RdYlBu_r', edgecolors='k', s=50, alpha=0.8)

ax.set(xlabel='Feature 1', ylabel='Feature 2')
ax.set_title('TabICL predicted class probabilities (2D)')
plt.colorbar(scatter, ax=ax, label='Probability of class 1')
plt.show()


# %%
# Evaluate model performance
# --------------------------------
# TabICL provides calibrated probabilities.

y_pred = tabicl.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.3f}")

# calibration curve for the positive class (class '1')
prob_true, prob_pred = calibration_curve(y_test, y_proba[:, 1], strategy='quantile', n_bins=7)

# plot calibration curve
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='TabICL')
ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.set_title('Calibration curve')
ax.legend()
plt.show()
