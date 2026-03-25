"""
Model interpretability with TabICL
============================

TabICL comes with a fast approximations of SHAP values. It is much
faster than using black-box shape routines on TabICL which is slow.

Here we demo it on dataset on wages
"""

# %%
# The dataset: wages
# ---------------------

from sklearn.datasets import fetch_openml

survey = fetch_openml(data_id=534, as_frame=True)

X = survey.data[survey.feature_names]

# %%
# A quick glance at the data with skrub's TableReport
import skrub
skrub.TableReport(X)

# %%
# We need to convert the categorical features to numeric ones. We can do this with pandas' get_dummies
import pandas as pd
X = pd.get_dummies(X, drop_first=True)

# %%
# The values to predict: wages
y = survey.target.values.ravel()

# %%
# Split out a test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# %%
# Our TabICL model
# ------------------
#
from tabicl import TabICLRegressor

clf = TabICLRegressor(n_estimators=4, device="cpu")
clf.fit(X_train, y_train)


# %%
# Shap-like interpretability
# ---------------------------
#
# Use TabICL's fast approximations of shap-like values and plot them
#
# This part of the example requires to install the shap extra:
# pip install 'tabicl[shap]

from tabicl.shap import get_shap_values, plot_shap

# Compute the shap values
sv = get_shap_values(clf, X_test)

# %%
# Plot the shap values
plot_shap(sv)

# %%
# Note that these are approximate SHAP values, and not exact ones.
