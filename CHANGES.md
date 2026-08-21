In development
==============

New features
------------

- Add TabICLv2 pre-training code (`python -m tabicl.train`): quantile regression training via a
  pinball loss (`--regression_method quantile`) in addition to classification, and the Muon
  optimizer (`--muon True`) alongside AdamW. The training CLI now also exposes the `graph_scm`
  prior options, layernorm-without-bias (`--norm_type layernorm_nobias`), SSMax
  (`--col_ssmax`/`--icl_ssmax` with `--ssmax_type`), feature grouping and target-aware embeddings
  (`--col_feature_group`, `--col_target_aware`, `--col_affine`), the RoPE variant
  (`--row_rope_interleaved`; v1 interleaved by default, v2 uses `False`), residual initialization
  (`--zero_init`; v2 uses `False`), and FlashAttention-3 during training (`--use_flash_attn3`;
  the v2 recipe enables it for stages 2 and 3 only). All CLI defaults reproduce the TabICLv1
  model configuration; resuming a run re-seeds the data stream with the current step. Ships the
  three-stage TabICLv2 curriculum scripts, separately for the classifier and regressor
  checkpoints (`scripts/train_v2_{clf,reg}_stage{1,2,3}.sh`).

Bug fixes
---------

- Finetuning now supports string/categorical features in DataFrames, matching the behavior of the base `TabICLClassifier` and `TabICLRegressor`. Previously, `FinetunedTabICLClassifier` and `FinetunedTabICLRegressor` would raise `ValueError: could not convert string to float` when the input contained categorical columns. (Reported by @zehua-jerry-yu in [#118](https://github.com/soda-inria/tabicl/issues/118))

- When unpickling a TabICL estimator, the fitted attributes `device_`, `model_`, etc. are only state if the pickled model was fitted. ([PR#121](https://github.com/soda-inria/tabicl/pull/121))


2.1.0
=====

New features
------------

- Add SHAP and ShapIQ support with NaN-based feature masking to explain TabICL predictions, plus a dedicated tutorial and a faster SHAP path. ([PR#90](https://github.com/soda-inria/tabicl/pull/90))

- Add support for some unsupervised learning tasks. ([PR#82](https://github.com/soda-inria/tabicl/pull/82))

- Add support for raw quantiles (direct outputs of TabICL) in regression, enabling native quantile regression without post-hoc calibration. ([PR#42](https://github.com/soda-inria/tabicl/pull/42))

- Add preprocessing for NumPy array inputs, consistent with existing behavior for Pandas inputs: ordinal encoding for categorical features, mean imputation for numerical features, and encoding missing values as a separate category for categorical columns. ([PR#51](https://github.com/soda-inria/tabicl/pull/51))

API changes
-----------

- Clarify the public vs. private API boundary following scikit-learn conventions. Internal modules are now prefixed with an underscore (`_model`, `_sklearn`, `_unsupervised`, etc.); import public estimators (`TabICLClassifier`, `TabICLRegressor`, `TabICLForecaster`) from the top-level `tabicl` package. ([PR#84](https://github.com/soda-inria/tabicl/pull/84))

Performance
-----------

- Replace broadcasting with `searchsorted` in `QuantileDistribution` for faster quantile evaluation.

Documentation
-------------

- New documentation site built with Sphinx + Sphinx Gallery and published on Read the Docs, including a redesigned landing page. ([PR#52](https://github.com/soda-inria/tabicl/pull/52), [PR#54](https://github.com/soda-inria/tabicl/pull/54), [PR#58](https://github.com/soda-inria/tabicl/pull/58), [PR#60](https://github.com/soda-inria/tabicl/pull/60), [PR#67](https://github.com/soda-inria/tabicl/pull/67), [PR#81](https://github.com/soda-inria/tabicl/pull/81))

- Add a project logo. ([PR#74](https://github.com/soda-inria/tabicl/pull/74))

- New tutorials: quantile regression ([PR#61](https://github.com/soda-inria/tabicl/pull/61)), probabilistic classification ([PR#73](https://github.com/soda-inria/tabicl/pull/73)), time series forecasting ([PR#77](https://github.com/soda-inria/tabicl/pull/77)), and skrub integration with string-handling fixes ([PR#78](https://github.com/soda-inria/tabicl/pull/78)).

Maintenance
-----------

- Tweak dependency management and test against the development versions of dependencies in CI. ([PR#53](https://github.com/soda-inria/tabicl/pull/53), [PR#63](https://github.com/soda-inria/tabicl/pull/63))

- Set explicit read permissions in CI workflows. ([PR#76](https://github.com/soda-inria/tabicl/pull/76))

2.0.3
=====

- Drop Python 3.9 support and now requires Python >= 3.10

- `kv_cache` moved from `fit()` to `__init__()` following scikit-learn convention. `kv_cache` is now a constructor parameter for both `TabICLClassifier` and `TabICLRegressor`.

- `TabICLForecaster` API changes — `output_selection` renamed to `point_estimate`

- Fix KV cache dtype mismatch. When AMP is enabled, cached projections stored in float16 caused errors when loaded on CPU/MPS/CUDA without AMP. The cache is now auto-upcast to float32 during loading.

- Refactor time series forecasting module
