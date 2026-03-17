"""
Handling strings and dates with skrub
============================

This example demonstrates how skrub can be used to preprocess datasets with strings or dates for TabICL
to make better predictions.
"""

# %%
# Preparing the dataset
# -------------------------------------
#
# Real-world datasets often contain complex heterogeneous data that benefits from more sophisticated preprocessing.
# For these scenarios, we recommend `skrub <https://skrub-data.org/stable/index.html>`_,
# a powerful library designed specifically for advanced tabular data preparation.
#
# Why use skrub?
# - Handles diverse data types (numerical, categorical, text, datetime, etc.)
# - Provides robust preprocessing for dirty data
# - Offers sophisticated feature engineering capabilities
# - Supports multi-table integration and joins.
#
# To install skrub, use ``pip install -U skrub``.
#
# TabICL can handle numerical and categorical columns natively, but will treat string columns as categorical.
# Here, we show how using skrub can provide TabICL with better string encoding. We use the "open payments" dataset.

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import skrub.datasets
from skrub import TableVectorizer, DatetimeEncoder, StringEncoder
import pandas as pd
import time
import numpy as np
from tabicl import TabICLClassifier

data = skrub.datasets.fetch_open_payments()
X, y = data.X, data.y
rng = np.random.RandomState(0)
subset_indices = rng.permutation(y.shape[0])[:600]
X, y = X.iloc[subset_indices], y[subset_indices]  # subsample for fast experiments

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
X

# %%
# TabICL without skrub
# -------------------------------------
#
# When string columns are used with TabICL directly, TabICL will interpret them as categorical columns.
# This means that TabICL doesn't know which strings are similar, it only knows which strings are identical.
# It can still learn from them if they repeat in the dataset,
# but may struggle when the number of different strings is high.
# Note that the runtime here might include the time for downloading the checkpoint.

reg = TabICLClassifier(n_estimators=1, device="cpu")  # 1 estimator for speed
start_time = time.time()
scores = cross_val_score(reg, X, y, cv=2, scoring="roc_auc_ovr")
print(f"ROC AUC without skrub: {scores.mean():.3f} (+/- {scores.std():.3f}), time: {time.time()-start_time:.1f} s")

# %%
# TabICL with skrub
# -------------------------------------
#
# With skrub, we can embed high-cardinality string columns using semantics-aware methods into numerical features.
# The `TableVectorizer <https://skrub-data.org/stable/reference/generated/skrub.TableVectorizer.html>`_
# applies different conversions to columns of a dataframe.
# Here, for efficiency reasons, we use the StringEncoder with lower-dimensional embeddings
# for all string columns with at least 10 distinct values.
# For lower-cardinality string columns, we use "passthrough", so they are directly forwarded to TabICL,
# which then treats them as categoricals.
# (Without "passthrough", they would be one-hot encoded by default,
# which is not the recommended way to handle categoricals for TabICL.)
# We also provide advanced settings for the DatetimeEncoder,
# even though our example dataset here does not contain dates.

pipeline = make_pipeline(
    TableVectorizer(
        low_cardinality="passthrough",  # let TabICL handle low-cardinality categories
        cardinality_threshold=10,
        high_cardinality=StringEncoder(n_components=10),  # fewer components for speed
        datetime=DatetimeEncoder(add_weekday=True, add_day_of_year=True, periodic_encoding='circular'),
    ),
    TabICLClassifier(n_estimators=1, device="cpu")  # 1 estimator for speed
)
start_time = time.time()
scores = cross_val_score(pipeline, X, y, cv=2, scoring="roc_auc_ovr")
print(f"ROC AUC with skrub: {scores.mean():.3f} (+/- {scores.std():.3f}), time: {time.time()-start_time:.1f} s")

# %%
#
# Overall, skrub preprocessing helps TabICL to achieve a larger ROC AUC on this dataset.
# It increases the runtime because strings get encoded into multiple columns.
