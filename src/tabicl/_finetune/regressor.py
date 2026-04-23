"""TabICL regressor fine-tuning wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.validation import check_is_fitted

from tabicl._sklearn.regressor import TabICLRegressor
from tabicl._finetune.base import ValidationMetrics, FinetunedTabICLBase
from tabicl._finetune.data import MetaBatch


def _pinball_loss(quantiles: torch.Tensor, targets: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Compute the pinball (quantile) loss averaged over all quantile levels.

    .. math::
        L_\\alpha(y, \\hat{q}) = \\max(\\alpha (y - \\hat{q}), (\\alpha - 1)(y - \\hat{q}))

    Parameters
    ----------
    quantiles : Tensor, shape ``(E, test_size, Q)``
        Raw quantile predictions from the TabICL head, one prediction per
        quantile level.

    targets : Tensor, shape ``(E, test_size)``
        Ground-truth target values (z-normalized during training).

    alpha : Tensor, shape ``(Q,)``
        Probability levels matching ``quantiles``' last dim.

    Returns
    -------
    Tensor
        Scalar loss, averaged over ensemble members, query samples, and
        quantile levels.
    """
    diff = targets.unsqueeze(-1) - quantiles
    return torch.maximum(alpha * diff, (alpha - 1.0) * diff).mean()


class FinetunedTabICLRegressor(FinetunedTabICLBase, RegressorMixin):
    """Fine-tune a pretrained TabICL for single-dataset regression.

    Subclass of :class:`FinetunedTabICLBase` that trains against pinball (quantile)
    loss applied directly to the raw quantile outputs of TabICL and scores
    validation with MSE / MAE / R² computed in raw y space via the wrapped
    :class:`TabICLRegressor`.

    Minimal usage::

        from tabicl import FinetunedTabICLRegressor
        reg = FinetunedTabICLRegressor(epochs=30, device="cuda", verbose=True)
        reg.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        y_pred = reg.predict(X_test)
        quantiles = reg.predict(X_test, output_type="quantiles", alphas=[0.1, 0.5, 0.9])

    Parameters
    ----------

    **Optimization**

    epochs : int, default=30
        Number of passes through the fine-tuning meta-batches.

    learning_rate : float, default=1e-5
        AdamW learning rate.

    weight_decay : float, default=0.01
        AdamW weight decay.

    grad_clip : float, default=1.0
        Max global gradient norm (``0`` disables).

    amp : bool, default=True
        Use FP16 automatic mixed precision on CUDA.

    use_lr_scheduler : bool, default=True
        Cosine-with-warmup LR schedule.

    warmup_proportion : float, default=0.1
        Warmup fraction of total steps.

    **Data pipeline**

    n_estimators_finetune : int, default=2
        Ensemble size during training meta-batches.

    n_estimators_validation : int, default=2
        Ensemble size during end-of-epoch validation.

    n_estimators_inference : int, default=8
        Ensemble size of the final inner estimator used by :meth:`predict`.

    max_data_size : int, default=10_000
        Max samples per meta-dataset chunk.

    finetune_ctx_query_ratio : float, default=0.2
        Query fraction inside each chunk.

    validation_split_ratio : float, default=0.1
        Size of auto-split validation set when ``X_val`` / ``y_val`` are not
        passed to :meth:`fit`.

    **Early stopping & time budget**

    early_stopping : bool, default=True
        Stop after ``patience`` non-improving epochs.

    patience : int, default=8
        Number of non-improving epochs tolerated.

    min_delta : float, default=1e-4
        Minimum metric improvement that counts as an improvement.

    time_limit : float or None, default=None
        Wall-clock budget in seconds; ``None`` disables.

    save_interval : int, default=1
        Write an interval checkpoint every N epochs; best is always saved.

    **Preprocessing**

    norm_methods : str, list[str] or None, default=None
        Normalization methods forwarded to
        :class:`tabicl._sklearn.preprocessing.EnsembleGenerator`.

    feat_shuffle_method : str, default="latin"
        Feature-permutation strategy for ensemble diversity.

    outlier_threshold : float, default=4.0
        Z-score threshold for outlier clipping during preprocessing.

    **Model loading**

    model_path : str, Path or None, default=None
        Checkpoint file to fine-tune from. ``None`` → download the default
        TabICLv2 regressor checkpoint from Hugging Face Hub.

    allow_auto_download : bool, default=True
        Permit downloading the pretrained checkpoint when it isn't cached.

    checkpoint_version : str, default="tabicl-regressor-v2-20260212.ckpt"
        Pretrained checkpoint version identifier.

    **Freezing**

    freeze_col : bool, default=False
        Freeze the column-embedding sub-module (weights and dropout/BN).

    freeze_row : bool, default=False
        Freeze the row-interaction sub-module.

    freeze_icl : bool, default=False
        Freeze the in-context-learning predictor.

    **Device & logging**

    device : str, torch.device or None, default=None
        Compute device; ``None`` auto-selects ``cuda`` when available.

    random_state : int, default=42
        Seed for data splits and ensemble shuffle patterns.

    verbose : bool, default=False
        Print a tqdm progress bar and one-line per-epoch summary.

    wandb_kwargs : dict or None, default=None
        When provided, enables Weights & Biases tracking by instantiating
        :class:`WandbLogger(**wandb_kwargs)` on rank 0. Supported keys are
        those of :func:`wandb.init` — most commonly ``project``, ``name``
        (the W&B run name), ``entity``, ``tags``, ``notes``, ``group``,
        ``mode`` (``"online" / "offline" / "disabled"``), and ``dir``. All
        keys are forwarded verbatim to ``wandb.init``.

    **Regressor-specific**

    eval_metric : {"mse", "mae", "r2"}, default="mse"
        Primary validation metric driving early stopping and best-weight
        selection. Computed in raw y space via :meth:`TabICLRegressor.predict`.
        ``mse`` and ``mae`` are internally negated so "higher is better"
        holds uniformly.

    extra_regressor_kwargs : dict or None, default=None
        Additional kwargs forwarded to the inner :class:`TabICLRegressor`.
    """

    def __init__(
        self,
        *,
        # Optimization
        epochs: int = 30,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        amp: bool = True,
        use_lr_scheduler: bool = True,
        warmup_proportion: float = 0.1,
        # Data pipeline
        n_estimators_finetune: int = 2,
        n_estimators_validation: int = 2,
        n_estimators_inference: int = 8,
        max_data_size: int = 10_000,
        finetune_ctx_query_ratio: float = 0.2,
        validation_split_ratio: float = 0.1,
        # Early stopping & time budget
        early_stopping: bool = True,
        patience: int = 8,
        min_delta: float = 1e-4,
        time_limit: Optional[float] = None,
        save_interval: int = 1,
        # Preprocessing
        norm_methods=None,
        feat_shuffle_method: str = "latin",
        outlier_threshold: float = 4.0,
        # Model loading
        model_path: Optional[str | Path] = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = "tabicl-regressor-v2-20260212.ckpt",
        # Freezing
        freeze_col: bool = False,
        freeze_row: bool = False,
        freeze_icl: bool = False,
        # Device & logging
        device: Optional[str | torch.device] = None,
        random_state: int = 42,
        verbose: bool = False,
        wandb_kwargs: Optional[dict[str, Any]] = None,
        # Regressor-specific
        eval_metric: Literal["mse", "mae", "r2"] = "mse",
        extra_regressor_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            amp=amp,
            use_lr_scheduler=use_lr_scheduler,
            warmup_proportion=warmup_proportion,
            n_estimators_finetune=n_estimators_finetune,
            n_estimators_validation=n_estimators_validation,
            n_estimators_inference=n_estimators_inference,
            max_data_size=max_data_size,
            finetune_ctx_query_ratio=finetune_ctx_query_ratio,
            validation_split_ratio=validation_split_ratio,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            time_limit=time_limit,
            save_interval=save_interval,
            norm_methods=norm_methods,
            feat_shuffle_method=feat_shuffle_method,
            outlier_threshold=outlier_threshold,
            model_path=model_path,
            allow_auto_download=allow_auto_download,
            checkpoint_version=checkpoint_version,
            freeze_col=freeze_col,
            freeze_row=freeze_row,
            freeze_icl=freeze_icl,
            device=device,
            random_state=random_state,
            verbose=verbose,
            wandb_kwargs=wandb_kwargs,
        )
        # Regressor-specific
        self.eval_metric = eval_metric
        self.extra_regressor_kwargs = extra_regressor_kwargs

    # ---- base hooks ----

    @property
    def _model_type(self) -> Literal["classifier", "regressor"]:
        return "regressor"

    @property
    def _metric_name(self) -> str:
        return self.eval_metric

    def _create_inner_estimator(self, *, n_estimators: int, device: torch.device) -> TabICLRegressor:
        """Construct a fresh :class:`TabICLRegressor` with matching config.

        Note: ``self.verbose`` drives only the outer fine-tuning tqdm bar and
        epoch summary. The inner estimator defaults to ``verbose=False`` so
        its per-inference "Available GPU memory / Offload decision" logs do
        not fire on every end-of-epoch validation pass. Callers who want the
        inner chatter can request it via
        ``extra_regressor_kwargs={"verbose": True}``.
        """
        kwargs = dict(self.extra_regressor_kwargs or {})
        kwargs.setdefault("verbose", False)
        kwargs.update(
            dict(
                n_estimators=n_estimators,
                norm_methods=self.norm_methods,
                feat_shuffle_method=self.feat_shuffle_method,
                outlier_threshold=self.outlier_threshold,
                model_path=self.model_path,
                allow_auto_download=self.allow_auto_download,
                checkpoint_version=self.checkpoint_version,
                device=device,
                random_state=self.random_state,
            )
        )
        return TabICLRegressor(**kwargs)

    def _compute_batch_loss(self, batch: MetaBatch, model) -> torch.Tensor:
        """Pinball loss over the model's raw quantile outputs."""
        # TabICL.forward in train mode emits raw quantiles of shape
        # (E, test_size, Q). ``d`` is omitted — all ensemble members in a
        # fine-tuning batch share the same feature count, so ``d=None``
        # short-circuits the mask math inside TabICL.
        quantiles = model(batch.X, batch.y_train)
        # TabICL's pretrained regressor head uses the default quantile levels
        # ``linspace(0, 1, Q+2)[1:-1]`` (see :class:`QuantileToDistribution`),
        # so we build ``alpha`` on the compute device directly.
        q = quantiles.shape[-1]
        alpha = torch.linspace(0.0, 1.0, q + 2, device=quantiles.device, dtype=quantiles.dtype)[1:-1]
        return _pinball_loss(quantiles, batch.y_query, alpha)

    def _run_validation(
        self,
        inner: TabICLRegressor,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> ValidationMetrics:
        """Fit ``inner`` on train, predict on val, return MSE/MAE/R² metrics."""
        # The caller (:meth:`FinetunedTabICLBase._validate_current_model`) has
        # already switched the underlying module to eval mode.
        try:
            inner.fit(X_train, y_train)
            preds = inner.predict(X_val)
        except (ValueError, RuntimeError) as e:
            if self.verbose:
                import logging

                logging.getLogger(__name__).warning("Validation failed: %s", e)
            return ValidationMetrics(primary=float("nan"))

        y_val_arr = np.asarray(y_val)
        mse = float(mean_squared_error(y_val_arr, preds))
        mae = float(mean_absolute_error(y_val_arr, preds))
        r2 = float(r2_score(y_val_arr, preds))

        secondary = {"mse": mse, "mae": mae, "r2": r2}
        if self.eval_metric == "mse":
            primary = -mse  # higher-is-better sign convention
        elif self.eval_metric == "mae":
            primary = -mae
        elif self.eval_metric == "r2":
            primary = r2
        else:
            raise ValueError(f"Unsupported eval_metric: {self.eval_metric!r}")

        return ValidationMetrics(primary=primary, secondary=secondary)

    def _metric_improved(self, current: float, best: float) -> bool:
        """Higher-is-better comparison (``mse`` / ``mae`` are pre-negated)."""
        return current > best + self.min_delta

    def _initial_best_metric(self) -> float:
        """Sentinel for "no baseline yet" under the higher-is-better convention."""
        return -np.inf

    # ---- prediction surface ----

    def predict(self, X, output_type: str | list[str] = "mean", alphas=None):
        """Predict target values for ``X``.

        Thin wrapper around :meth:`TabICLRegressor.predict` that forwards
        ``output_type`` / ``alphas`` unchanged. See that method for the full
        set of supported output types (``"mean"``, ``"median"``,
        ``"quantiles"``, ``"raw_quantiles"``).
        """
        check_is_fitted(self, "_final_estimator_")
        return self._final_estimator_.predict(X, output_type=output_type, alphas=alphas)
