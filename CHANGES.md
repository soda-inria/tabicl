2.2.0
=====

New features
------------

- Add TabICLv2 pre-training code (`python -m tabicl.train`): quantile regression training via a
  pinball loss (`--regression_method quantile`) in addition to classification, and the Muon
  optimizer (`--muon True`) alongside AdamW. The training CLI now also exposes the `graph_scm`
  prior options, layernorm-without-bias (`--norm_type layernorm_nobias`), SSMax
  (`--col_ssmax`/`--icl_ssmax` with `--ssmax_type`), feature grouping and target-aware embeddings
  (`--col_feature_group`, `--col_target_aware`, `--col_affine`), the RoPE variant
  (`--row_rope_interleaved`; v1 interleaved by default, v2 uses `False`), residual initialization
  (`--zero_init`; v2 uses `False`), FlashAttention-3 during training (`--use_flash_attn3`;
  the v2 recipe enables it for stages 2 and 3 only). The cuDNN SDPA backend is now
  automatically disabled during training (slower than Flash Attention on Hopper). `--dtype` now
  supports `bfloat16` in addition to `float16`/`float32`; the GradScaler is only enabled for
  `float16`. All CLI defaults reproduce the TabICLv1 model configuration; resuming a run
  re-seeds the data stream with the current step. The v2 curriculum scripts
  (`scripts/train_v2_{clf,reg}_stage{1,2,3}.sh`) now use `--dtype float16` for faster training.
  ([PR#135](https://github.com/soda-inria/tabicl/pull/135))

- Remove the GluonTS dependency from the forecasting module. ([PR#108](https://github.com/soda-inria/tabicl/pull/108), @daidahao)

- Improve non-CUDA GPU inference reliability and performance (including XPU): inference now consistently runs on the configured backend device, uses backend-appropriate autocast, and queries available memory plus async stream/event primitives through backend-agnostic `torch.<backend>` APIs (with safe synchronous fallbacks when async is unavailable). This fixes pathological auto-batch sizing (e.g. batch size forced to 1) and restores expected accelerated inference behavior on supported non-CUDA GPU backends. When `device=None`, estimators now default to CUDA when available, otherwise XPU, then MPS, and then CPU. ([PR#144](https://github.com/soda-inria/tabicl/pull/144))

- Improve Apple Silicon MPS inference: MPS now uses the same AMP, auto-batching, and memory-aware inference path as other accelerators instead of falling back to the CPU path. `use_amp="auto"` is device-aware (off on CPU; size heuristic on CUDA/XPU/MPS), and float16 KV caches are kept on MPS when AMP is enabled. ([PR#144](https://github.com/soda-inria/tabicl/pull/144))

Bug fixes
---------

- Finetuning now supports string/categorical features in DataFrames, matching the behavior of the base `TabICLClassifier` and `TabICLRegressor`. Previously, `FinetunedTabICLClassifier` and `FinetunedTabICLRegressor` would raise `ValueError: could not convert string to float` when the input contained categorical columns. ([PR#151](https://github.com/soda-inria/tabicl/pull/151); reported by @zehua-jerry-yu in [#118](https://github.com/soda-inria/tabicl/issues/118))

- Fix `DatetimeEncoder` sin/cos encoding off-by-one error that caused the first and last elements of a period to map to identical angles (e.g. Monday and Sunday getting the same encoding). The denominator was incorrectly `p-1` instead of `p`. ([PR#151](https://github.com/soda-inria/tabicl/pull/151); reported by @christophM in [#136](https://github.com/soda-inria/tabicl/issues/136))

- Fix `float16` input arrays crashing during Yeo-Johnson normalization in the preprocessing pipeline. The `PreprocessingPipeline` now upcasts `float16` to `float32` before fitting/transforming, avoiding scipy's narrow-exponent bound error. ([PR#151](https://github.com/soda-inria/tabicl/pull/151); reported by @SebastienMelo in [#140](https://github.com/soda-inria/tabicl/issues/140))

- Fix `predict_proba`/`predict` crashing with `TypeError` when a categorical column is all-NaN in the prediction batch. Removed the batch-global all-NaN feature-masking detection from the prediction path — it was intended for SHAP but did not work correctly with SHAP's coalition batching, and made predictions depend on batch composition. All-NaN columns now flow through normal preprocessing (OrdinalEncoder/SimpleImputer handle NaN natively). ([PR#151](https://github.com/soda-inria/tabicl/pull/151); reported by @Innixma in [#143](https://github.com/soda-inria/tabicl/issues/143))

- Fix PyTorch autograd error when fine-tuning with partial module freezing (e.g. `freeze_col=True, freeze_row=True, freeze_icl=False`). An in-place operation on a tensor view from frozen modules conflicted with autograd; resolved by detaching before the in-place write. ([PR#151](https://github.com/soda-inria/tabicl/pull/151); reported by @denisfouchard in [#128](https://github.com/soda-inria/tabicl/issues/128))

- When unpickling a TabICL estimator, the fitted attributes `device_`, `model_`, etc. are only set if the pickled model was fitted. ([PR#121](https://github.com/soda-inria/tabicl/pull/121), @jeromedockes)

- Fix `get_state`/`set_state` for `model_kv_cache_`. ([PR#124](https://github.com/soda-inria/tabicl/pull/124), @jeromedockes)

- Fix default behaviour on NumPy arrays with string-valued columns. ([PR#123](https://github.com/soda-inria/tabicl/pull/123), @marineLM)

- `n_threads` is now set to the minimum of `n_logical_cores` and `n_jobs`, rather than maximum. ([PR#107](https://github.com/soda-inria/tabicl/pull/107), @douglas-boubert)

- Keep all-NaN columns in `SimpleImputer` so `predict_proba` does not crash on datasets where an entire feature is missing at prediction time. ([PR#148](https://github.com/soda-inria/tabicl/pull/148), @axsaucedo)

- Fix `UnicodeEncodeError` in fine-tuning tutorials on Windows. ([PR#141](https://github.com/soda-inria/tabicl/pull/141), @maxdemarzi)

- Fix `mix_probs` key in `SCMPrior`. ([PR#106](https://github.com/soda-inria/tabicl/pull/106), @nightcityblade)

- Fix `early_stopping=False` silently discarding all fine-tuning updates: `best_state` was only updated inside the early-stopping branch, so disabling early stopping caused the pretrained weights to be restored after training. ([PR#151](https://github.com/soda-inria/tabicl/pull/151), @dholzmueller)

Documentation
-------------

- Document `skrub.tabular_pipeline`'s new support for TabICL. ([PR#139](https://github.com/soda-inria/tabicl/pull/139), @ashwinvis)

- Remove ambiguity on classification/regression in README. ([PR#134](https://github.com/soda-inria/tabicl/pull/134), @MarieSacksick)

- Fix broken star history chart in README. ([PR#149](https://github.com/soda-inria/tabicl/pull/149), @FaintFlower)

Maintenance
-----------

- Rename prior config variables for consistency. ([PR#105](https://github.com/soda-inria/tabicl/pull/105), @ChristopheMuller)

- Add `pytest` to the `test` dependency group. ([PR#122](https://github.com/soda-inria/tabicl/pull/122), @jeromedockes)


2.1.1
=====

New features
------------

- Add fine-tuning support for TabICL via `FinetunedTabICLClassifier` and `FinetunedTabICLRegressor`: full PyTorch training loop with AdamW, cosine-warmup schedule, early stopping, gradient clipping, AMP, DDP, partial module freezing, and checkpointing in the pre-training schema. ([PR#101](https://github.com/soda-inria/tabicl/pull/101), @JingangQu)


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
