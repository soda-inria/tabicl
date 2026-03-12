In development
==============

- Add preprocessing for NumPy array inputs, consistent with existing behavior for Pandas arrays: ordinal encoding for categorical features, mean imputation for numerical features, and encoding missing values as a separate category for categorical columns. ([PR#51](https://github.com/soda-inria/tabicl/pull/51))

2.0.3
=====

- Drop Python 3.9 support and now requires Python >= 3.10

- `kv_cache` moved from `fit()` to `__init__()` following scikit-learn convention. `kv_cache` is now a constructor parameter for both `TabICLClassifier` and `TabICLRegressor`.

- `TabICLForecaster` API changes — `output_selection` renamed to `point_estimate`

- Fix KV cache dtype mismatch. When AMP is enabled, cached projections stored in float16 caused errors when loaded on CPU/MPS/CUDA without AMP. The cache is now auto-upcast to float32 during loading.

- Refactor time series forecasting module