"""
Getting Started with TabICL
============================

This example demonstrates the basic usage of TabICL for classification and
regression tasks using the scikit-learn compatible API.
"""

# %%
# Classification with TabICLClassifier
# -------------------------------------
#
# TabICL provides a scikit-learn compatible classifier that works out of the
# box. Let's use it on a synthetic classification dataset.

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

from tabicl import TabICLClassifier

X, y = make_classification(
    n_samples=300, n_features=10, n_informative=5, random_state=42
)

clf = TabICLClassifier(n_estimators=4, device="cpu")
scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
print(f"Classification accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# %%
# Regression with TabICLRegressor
# --------------------------------
#
# TabICLRegressor follows the same interface for regression tasks.

from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score

from tabicl import TabICLRegressor

X, y = make_regression(
    n_samples=300, n_features=10, n_informative=5, noise=0.5, random_state=42
)

reg = TabICLRegressor(n_estimators=4, device="cpu")
scores = cross_val_score(reg, X, y, cv=5, scoring="r2")
print(f"Regression R² score: {scores.mean():.3f} (+/- {scores.std():.3f})")

# %%
# Using KV caching for faster repeated inference
# ------------------------------------------------
#
# When you need to call ``predict`` multiple times on the same training data
# (e.g., during evaluation), enable KV caching to speed up inference. The
# cache is built once during ``fit`` and reused across ``predict`` calls.

from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=300, n_features=10, n_informative=5, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = TabICLClassifier(n_estimators=4, kv_cache=True, device="cpu")
clf.fit(X_train, y_train)

# Subsequent predict calls reuse the cached context
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
print(f"Predictions shape: {predictions.shape}")
print(f"Probabilities shape: {probabilities.shape}")
