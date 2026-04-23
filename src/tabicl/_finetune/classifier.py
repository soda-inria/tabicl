"""TabICL classifier fine-tuning wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.utils.validation import check_is_fitted

from tabicl._sklearn.classifier import TabICLClassifier
from tabicl._finetune.base import ValidationMetrics, FinetunedTabICLBase
from tabicl._finetune.data import MetaBatch


class FinetunedTabICLClassifier(ClassifierMixin, FinetunedTabICLBase):
    """Fine-tune a pretrained TabICL for single-dataset classification.

    Subclass of :class:`FinetunedTabICLBase` that implements cross-entropy loss on
    the raw TabICL logits and ROC-AUC / log-loss / accuracy evaluation metrics.

    Minimal usage::

        from tabicl import FinetunedTabICLClassifier
        clf = FinetunedTabICLClassifier(epochs=30, device="cuda", verbose=True)
        clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        y_proba = clf.predict_proba(X_test)

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
        Ensemble size of the final inner estimator used by
        :meth:`predict` / :meth:`predict_proba`.

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
        TabICLv2 classifier checkpoint from Hugging Face Hub.

    allow_auto_download : bool, default=True
        Permit downloading the pretrained checkpoint when it isn't cached.

    checkpoint_version : str, default="tabicl-classifier-v2-20260212.ckpt"
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

    **Classifier-specific**

    class_shuffle_method : str, default="shift"
        Class-label shuffle strategy for ensemble diversity.

    softmax_temperature : float, default=0.9
        Softmax temperature used by the inner :class:`TabICLClassifier` at
        inference time.

    average_logits : bool, default=True
        If True, ensemble averaging is done on logits; else on probabilities.

    support_many_classes : bool, default=True
        Enable TabICL's mixed-radix ensembling when the dataset has more
        classes than the pretrained head's native ``max_classes``.

    eval_metric : {"roc_auc", "log_loss", "accuracy"}, default="roc_auc"
        Primary validation metric driving early stopping and best-weight
        selection. ``log_loss`` is internally negated so "higher is better"
        holds uniformly.

    extra_classifier_kwargs : dict or None, default=None
        Additional kwargs forwarded to the inner :class:`TabICLClassifier`
        (e.g. ``{"kv_cache": "kv"}``).
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
        checkpoint_version: str = "tabicl-classifier-v2-20260212.ckpt",
        # Freezing
        freeze_col: bool = False,
        freeze_row: bool = False,
        freeze_icl: bool = False,
        # Device & logging
        device: Optional[str | torch.device] = None,
        random_state: int = 42,
        verbose: bool = False,
        wandb_kwargs: Optional[dict[str, Any]] = None,
        # Classifier-specific
        class_shuffle_method: str = "shift",
        softmax_temperature: float = 0.9,
        average_logits: bool = True,
        support_many_classes: bool = True,
        eval_metric: Literal["roc_auc", "log_loss", "accuracy"] = "roc_auc",
        extra_classifier_kwargs: Optional[dict[str, Any]] = None,
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
        # Classifier-specific
        self.class_shuffle_method = class_shuffle_method
        self.softmax_temperature = softmax_temperature
        self.average_logits = average_logits
        self.support_many_classes = support_many_classes
        self.eval_metric = eval_metric
        self.extra_classifier_kwargs = extra_classifier_kwargs

    # ---- base hooks ----

    @property
    def _model_type(self) -> Literal["classifier", "regressor"]:
        return "classifier"

    @property
    def _metric_name(self) -> str:
        return self.eval_metric

    def _create_inner_estimator(self, *, n_estimators: int, device: torch.device) -> TabICLClassifier:
        """Construct a fresh :class:`TabICLClassifier` with matching config.

        Note: ``self.verbose`` drives only the outer fine-tuning tqdm bar and
        epoch summary. The inner estimator defaults to ``verbose=False`` so
        its per-inference "Available GPU memory / Offload decision" logs do
        not fire on every end-of-epoch validation pass. Callers who want the
        inner chatter can request it via
        ``extra_classifier_kwargs={"verbose": True}``.
        """
        kwargs = dict(self.extra_classifier_kwargs or {})
        kwargs.setdefault("verbose", False)
        kwargs.update(
            dict(
                n_estimators=n_estimators,
                norm_methods=self.norm_methods,
                feat_shuffle_method=self.feat_shuffle_method,
                class_shuffle_method=self.class_shuffle_method,
                outlier_threshold=self.outlier_threshold,
                softmax_temperature=self.softmax_temperature,
                average_logits=self.average_logits,
                support_many_classes=self.support_many_classes,
                model_path=self.model_path,
                allow_auto_download=self.allow_auto_download,
                checkpoint_version=self.checkpoint_version,
                device=device,
                random_state=self.random_state,
            )
        )
        return TabICLClassifier(**kwargs)

    def _task_skip_batch(self, batch: MetaBatch) -> bool:
        """Skip batches where the query contains classes missing from context.

        Cross-entropy is undefined for a class index the model never saw in
        context (its logits aren't calibrated for it); skipping preserves
        training stability.
        """
        ctx = torch.unique(batch.y_train.reshape(-1))
        qry = torch.unique(batch.y_query.reshape(-1))
        return not bool(torch.isin(qry, ctx, assume_unique=True).all())

    def _compute_batch_loss(self, batch: MetaBatch, model) -> torch.Tensor:
        """Cross-entropy on the TabICL classifier logits."""
        # TabICL.forward signature: (X, y_train) -> (E, test_size, max_classes)
        # where E = n_estimators_finetune (ensemble as the batch dim). The
        # active-feature-count tensor ``d`` is omitted: all ensemble members
        # in a fine-tuning batch share the same feature count, so ``d=None``
        # short-circuits the mask math inside TabICL.
        logits = model(batch.X, batch.y_train.float())
        # Slice down to the number of classes actually in this dataset so
        # cross-entropy targets are valid indices into the sliced logits.
        n_classes = int(batch.y_train.max().item()) + 1
        logits_used = logits[..., :n_classes].reshape(-1, n_classes)
        targets = batch.y_query.long().reshape(-1)
        return F.cross_entropy(logits_used, targets)

    def _run_validation(
        self,
        inner: TabICLClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> ValidationMetrics:
        """Fit ``inner`` on train, predict on val, return ROC-AUC/log-loss/accuracy."""
        # The caller (:meth:`FinetunedTabICLBase._validate_current_model`) has
        # already switched the underlying module to eval mode.
        try:
            inner.fit(X_train, y_train)
            proba = inner.predict_proba(X_val)
        except (ValueError, RuntimeError) as e:
            if self.verbose:
                import logging

                logging.getLogger(__name__).warning("Validation failed: %s", e)
            return ValidationMetrics(primary=float("nan"))

        secondary: dict[str, float] = {}
        try:
            if proba.shape[1] == 2:
                roc = float(roc_auc_score(y_val, proba[:, 1]))
            else:
                roc = float(roc_auc_score(y_val, proba, multi_class="ovr"))
            secondary["roc_auc"] = roc
        except ValueError:
            roc = float("nan")

        ll = float(log_loss(y_val, proba, labels=inner.classes_))
        secondary["log_loss"] = ll
        acc = float(accuracy_score(y_val, np.asarray(inner.classes_)[proba.argmax(axis=1)]))
        secondary["accuracy"] = acc

        if self.eval_metric == "roc_auc":
            primary = roc
        elif self.eval_metric == "log_loss":
            primary = -ll  # higher-is-better sign convention
        elif self.eval_metric == "accuracy":
            primary = acc
        else:
            raise ValueError(f"Unsupported eval_metric: {self.eval_metric!r}")

        return ValidationMetrics(primary=primary, secondary=secondary)

    def _metric_improved(self, current: float, best: float) -> bool:
        """Higher-is-better comparison (``log_loss`` is pre-negated)."""
        return current > best + self.min_delta

    def _initial_best_metric(self) -> float:
        """Sentinel for "no baseline yet" under the higher-is-better convention."""
        return -np.inf

    # ---- prediction surface ----

    def predict_proba(self, X):
        """Predict class probabilities for ``X``.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Probability that each sample belongs to each class, in the order
            given by :attr:`classes_`.
        """
        check_is_fitted(self, "_final_estimator_")
        return self._final_estimator_.predict_proba(X)

    @property
    def classes_(self):
        """Class labels in the order used by :meth:`predict_proba`."""
        check_is_fitted(self, "_final_estimator_")
        return self._final_estimator_.classes_
