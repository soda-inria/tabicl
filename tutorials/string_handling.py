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

from sklearn.model_selection import cross_val_score
import skrub.datasets
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer, DatetimeEncoder, StringEncoder, TextEncoder
import pandas as pd
import time
import numpy as np

from tabicl import TabICLRegressor, TabICLClassifier


# data = skrub.datasets.fetch_bike_sharing()  # only dates, but good improvement
# # data = skrub.datasets.fetch_videogame_sales()  # todo: not shuffled
# # data = skrub.datasets.fetch_employee_salaries()  # RMSE 12.2K vs 14.9K, but very slow (many strings)
# X, y = data.X.iloc[:500], data.y[:500]  # subsample for fast experiments
#
# pd.set_option('display.max_columns', None)
# print(X.head())
#
# reg = TabICLRegressor(n_estimators=1, device="cpu")
# start_time = time.time()
# scores = cross_val_score(reg, X, y, cv=4, scoring="neg_root_mean_squared_error")
# print(f"RMSE without skrub: {-scores.mean():.3f} (+/- {scores.std():.3f}), time: {time.time()-start_time:.1f} s")
#
# pipeline = make_pipeline(
#     TableVectorizer(
#         low_cardinality="passthrough",
#         datetime=DatetimeEncoder(add_weekday=True, add_day_of_year=True, periodic_encoding='circular')
#     ),
#     TabICLRegressor(n_estimators=1, device="cpu")
# )
# start_time = time.time()
# scores = cross_val_score(pipeline, X, y, cv=4, scoring="neg_root_mean_squared_error")
# print(f"RMSE with skrub: {-scores.mean():.3f} (+/- {scores.std():.3f}), time: {time.time()-start_time:.1f} s")


# data = skrub.datasets.fetch_open_payments()
data = skrub.datasets.fetch_toxicity()
X, y = data.X, data.y
np.random.seed(0)
perm = np.random.permutation(y.shape[0])
X, y = X.iloc[perm[:600]], y[perm[:600]]  # subsample for fast experiments

pd.set_option('display.max_columns', None)
print(X.head())

reg = TabICLClassifier(n_estimators=1, device="cpu")
start_time = time.time()
scores = cross_val_score(reg, X, y, cv=2, scoring="roc_auc_ovr")
print(f"ROC AUC without skrub: {scores.mean():.3f} (+/- {scores.std():.3f}), time: {time.time()-start_time:.1f} s")

pipeline = make_pipeline(
    TableVectorizer(
        low_cardinality="passthrough",
        cardinality_threshold=10,
        high_cardinality=StringEncoder(n_components=10),
        datetime=DatetimeEncoder(add_weekday=True, add_day_of_year=True, periodic_encoding='circular'),
    ),
    TabICLClassifier(n_estimators=1, device="cpu")
)
start_time = time.time()
scores = cross_val_score(pipeline, X, y, cv=4, scoring="roc_auc_ovr")
print(f"ROC AUC with skrub: {scores.mean():.3f} (+/- {scores.std():.3f}), time: {time.time()-start_time:.1f} s")
