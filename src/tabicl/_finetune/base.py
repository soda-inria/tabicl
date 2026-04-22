"""Abstract base class for fine-tuning TabICL on a single dataset.

:meth:`FinetunedTabICLBase.fit` (1) loads the pretrained checkpoint,
(2) runs a PyTorch training loop (AdamW + cosine-with-warmup, optional AMP,
gradient clipping, early stopping, DDP auto-detect) guarded by an initial-eval
safety net that never returns weights worse than the pretrained baseline,
(3) saves checkpoints in the pretraining schema so they load back via
``TabICLClassifier(model_path=...)`` / ``TabICLRegressor(model_path=...)``,
and (4) builds a final inner estimator for ``predict`` / ``predict_proba``.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from tqdm.auto import tqdm

from tabicl._model.tabicl import TabICL
from tabicl._finetune.data import (
    MetaBatch,
    count_chunks,
    iter_epoch_meta_batches,
    move_meta_batch,
)
from tabicl._finetune.logging import FinetuningLogger, NullLogger, WandbLogger
from tabicl.train._optim import get_scheduler

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Result of a single validation pass.

    Attributes
    ----------
    primary : float
        The metric used for early stopping / best-weight selection. Always
        encoded as higher-is-better (MSE-style metrics are negated inside
        :meth:`FinetunedTabICLBase._run_validation`).

    secondary : dict[str, float]
        Additional metrics logged alongside the primary one. Keys are free-
        form strings (e.g. ``"roc_auc"``, ``"log_loss"``, ``"accuracy"``) and
        surface through the experiment logger under ``val/{key}``.
    """

    primary: float
    secondary: dict[str, float] = field(default_factory=dict)


def _ddp_env(device: str | torch.device) -> tuple[bool, int, int, int, bool, torch.device]:
    """Detect torchrun env and, if present, initialize the process group.

    Returns ``(using_ddp, rank, world_size, local_rank, master, device)``.
    On a single-process run this returns ``(False, 0, 1, 0, True, device)``
    with zero side effects.
    """
    if int(os.environ.get("RANK", -1)) == -1:
        resolved = torch.device(device) if not isinstance(device, torch.device) else device
        return False, 0, 1, 0, True, resolved

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    resolved = torch.device(f"cuda:{local_rank}")
    master = rank == 0
    if master:
        logger.info("DDP enabled: world_size=%d, rank=%d, device=%s", world_size, rank, resolved)
    return True, rank, world_size, local_rank, master, resolved


class FinetunedTabICLBase(BaseEstimator, ABC):
    """Abstract base for TabICL single-dataset fine-tuning estimators.

    Subclasses (:class:`FinetunedTabICLClassifier`,
    :class:`FinetunedTabICLRegressor`) plug in the task-specific loss, inner
    estimator factory, and validation metric; the base owns the full training
    loop, checkpointing, DDP handling, and the pickle / persistence surface.

    Subclass contract
    -----------------
    * :meth:`_compute_batch_loss` — given a :class:`MetaBatch` and the
      routed model, return a scalar loss tensor.
    * :meth:`_create_inner_estimator` — construct an unfitted
      :class:`~tabicl.TabICLClassifier` / :class:`~tabicl.TabICLRegressor`
      that the base composes for both validation and final inference.
    * :meth:`_run_validation` — fit the inner estimator on the training
      split and score it on the validation split, returning
      :class:`ValidationMetrics`.
    * :meth:`_metric_improved` / :meth:`_initial_best_metric` — define the
      ordering of the primary metric (higher-is-better is the convention;
      MSE-style metrics are negated in :meth:`_run_validation`).
    * :attr:`_model_type` / :attr:`_metric_name` — task flags used in data
      preprocessing and logging.

    Parameters
    ----------

    **Optimization**

    epochs : int, default=30
        Number of passes through the fine-tuning meta-batches.

    learning_rate : float, default=1e-5
        AdamW learning rate. Small values stabilize fine-tuning; values above
        ~5e-4 typically destabilize the pretrained weights.

    weight_decay : float, default=0.01
        AdamW weight decay.

    grad_clip : float, default=1.0
        Max global gradient norm. Set to ``0`` or a negative value to disable.

    amp : bool, default=True
        Use FP16 automatic mixed precision on CUDA. Silently ignored on
        CPU/MPS.

    use_lr_scheduler : bool, default=True
        If True, wrap the optimizer in a cosine-with-warmup schedule sized to
        ``epochs * chunks_per_epoch``.

    warmup_proportion : float, default=0.1
        Fraction of total steps spent linearly warming the LR up to
        ``learning_rate`` before the cosine decay.

    **Data pipeline**

    n_estimators_finetune : int, default=2
        Number of ensemble variants packed into each training meta-batch
        (serves as the effective batch dim passed to the underlying model).

    n_estimators_validation : int, default=2
        Ensemble size used at end-of-epoch validation.

    n_estimators_inference : int, default=8
        Ensemble size of the final inner estimator used by
        :meth:`predict` / :meth:`predict_proba`.

    max_data_size : int, default=10_000
        Maximum samples per meta-dataset chunk. Larger training sets are
        randomly re-chunked per epoch.

    finetune_ctx_query_ratio : float, default=0.2
        Fraction of each chunk used as query (where loss is computed); the
        remainder is the in-context examples.

    validation_split_ratio : float, default=0.1
        When explicit ``X_val`` / ``y_val`` are not passed to :meth:`fit`, this
        much of the training data is carved off as a validation split (stratified
        for classification).

    **Early stopping & time budget**

    early_stopping : bool, default=True
        Stop when the primary validation metric has not improved for
        ``patience`` consecutive epochs.

    patience : int, default=8
        Number of non-improving epochs tolerated before stopping.

    min_delta : float, default=1e-4
        Minimum metric improvement considered an actual improvement.

    time_limit : float or None, default=None
        Wall-clock budget in seconds. If projected time for one more epoch
        would exceed the budget, training halts.

    save_interval : int, default=1
        Write an interval checkpoint every N epochs. The best-metric
        checkpoint is always saved regardless of this value.

    **Preprocessing**

    norm_methods : str, list[str] or None, default=None
        Normalization methods forwarded to
        :class:`tabicl._sklearn.preprocessing.EnsembleGenerator`. ``None`` uses
        ``["none", "power"]``.

    feat_shuffle_method : str, default="latin"
        Feature-permutation strategy for ensemble diversity.

    outlier_threshold : float, default=4.0
        Z-score threshold for outlier clipping during preprocessing.

    **Model loading**

    model_path : str, Path or None, default=None
        Checkpoint file to fine-tune from. If ``None`` (default) the subclass-
        specific default checkpoint version is downloaded from Hugging Face
        Hub the first time.

    allow_auto_download : bool, default=True
        Permit downloading the pretrained checkpoint when it is not already
        cached locally.

    checkpoint_version : str or None, default=None
        Pretrained checkpoint version identifier; subclasses set the task-
        appropriate default.

    **Freezing**

    freeze_col : bool, default=False
        Freeze the column-embedding sub-module. Its parameters retain
        ``requires_grad=False`` and its dropout / batch-norm stay in eval mode
        across the entire training loop.

    freeze_row : bool, default=False
        Freeze the row-interaction sub-module.

    freeze_icl : bool, default=False
        Freeze the in-context-learning predictor.

    **Device & logging**

    device : str, torch.device or None, default=None
        Compute device. ``None`` auto-selects ``cuda`` when available,
        otherwise ``cpu``.

    random_state : int, default=42
        Seed for data splits, shuffle-pattern generation, and per-epoch
        chunking.

    verbose : bool, default=False
        If True, print a tqdm progress bar and a per-epoch one-line summary
        (train loss, val metric, wall-clock time).

    wandb_kwargs : dict or None, default=None
        If provided, a :class:`~tabicl._finetune.logging.WandbLogger` is
        instantiated with these kwargs (e.g. ``{"project": "tabicl-ft"}``) and
        used for experiment tracking. ``None`` disables W&B integration.
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
        checkpoint_version: Optional[str] = None,
        # Freezing
        freeze_col: bool = False,
        freeze_row: bool = False,
        freeze_icl: bool = False,
        # Device & logging
        device: Optional[str | torch.device] = None,
        random_state: int = 42,
        verbose: bool = False,
        wandb_kwargs: Optional[dict[str, Any]] = None,
    ):
        # Optimization
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.amp = amp
        self.use_lr_scheduler = use_lr_scheduler
        self.warmup_proportion = warmup_proportion
        # Data pipeline
        self.n_estimators_finetune = n_estimators_finetune
        self.n_estimators_validation = n_estimators_validation
        self.n_estimators_inference = n_estimators_inference
        self.max_data_size = max_data_size
        self.finetune_ctx_query_ratio = finetune_ctx_query_ratio
        self.validation_split_ratio = validation_split_ratio
        # Early stopping & time budget
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.time_limit = time_limit
        self.save_interval = save_interval
        # Preprocessing
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.outlier_threshold = outlier_threshold
        # Model loading
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.checkpoint_version = checkpoint_version
        # Freezing
        self.freeze_col = freeze_col
        self.freeze_row = freeze_row
        self.freeze_icl = freeze_icl
        # Device & logging
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.wandb_kwargs = wandb_kwargs

    # ---------------- abstract surface ----------------

    @property
    @abstractmethod
    def _model_type(self) -> Literal["classifier", "regressor"]: ...

    @abstractmethod
    def _create_inner_estimator(self, *, n_estimators: int, device: torch.device):
        """Return an unfitted TabICLClassifier/Regressor configured for inference."""

    @abstractmethod
    def _compute_batch_loss(self, batch: MetaBatch, model: nn.Module) -> torch.Tensor:
        """Run ``model`` forward on ``batch`` and return a scalar loss tensor.

        The caller passes the correctly-routed ``model``: the DDP wrapper when
        DDP is active (so backward hooks fire) and the raw TabICL module
        otherwise. Subclasses should call ``model(batch.X, batch.y_train)`` and
        compute the task-specific loss on the returned tensor.
        """

    @abstractmethod
    def _run_validation(
        self,
        inner,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> ValidationMetrics:
        """Fit ``inner`` on train, predict on val, return :class:`ValidationMetrics`."""

    @abstractmethod
    def _metric_improved(self, current: float, best: float) -> bool: ...

    @abstractmethod
    def _initial_best_metric(self) -> float: ...

    @property
    @abstractmethod
    def _metric_name(self) -> str: ...

    # ---------------- public API ----------------

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        output_dir: Optional[str | Path] = None,
    ) -> "FinetunedTabICLBase":
        """Fine-tune the pretrained TabICL on ``(X, y)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training inputs.

        y : array-like of shape (n_samples,)
            Training targets (class labels for classification, numeric for
            regression).

        X_val, y_val : array-like, optional
            Explicit validation set used for early stopping and best-metric
            selection. If either is ``None``, a single stratified split of
            ``(X, y)`` is taken using ``validation_split_ratio``.

        output_dir : str or Path, optional
            Directory for checkpoint files. When provided, the best-metric
            checkpoint is written to ``{output_dir}/best.ckpt`` and
            intermediate checkpoints to ``{output_dir}/epoch{N}.ckpt``. Both
            share the TabICL checkpoint schema, so either file can be loaded
            directly by ``TabICLClassifier(model_path=<path>)`` /
            ``TabICLRegressor(model_path=<path>)``.

            When ``None`` (default), no checkpoints are saved and fine-tuning
            progress is lost if the run is interrupted — a ``UserWarning`` is
            emitted in that case.

        Returns
        -------
        self : FinetunedTabICLBase
            Fitted estimator (chainable).

        Notes
        -----
        The loop compares every epoch's validation metric against the
        pretrained checkpoint's metric (computed before any weight update).
        The best weights encountered — possibly the pretrained ones if no
        epoch improves on them — are restored before ``predict`` becomes
        available, so fine-tuning never returns a model worse than the
        baseline on the validation set.
        """
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            warnings.warn(
                "`output_dir` is not set; no checkpoints will be saved and all "
                "fine-tuning progress is lost if the run is interrupted.",
                UserWarning,
                stacklevel=2,
            )

        try:
            self._fit_impl(X, y, X_val=X_val, y_val=y_val, output_dir=output_dir)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        return self

    def predict(self, X):
        """Predict targets for ``X`` using the fine-tuned inner estimator.

        Subclasses may override to forward task-specific keyword arguments
        (e.g. the regressor's ``output_type``).
        """
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "_final_estimator_")
        return self._final_estimator_.predict(X)

    # ---------------- internal helpers ----------------

    def _resolve_device(self, device_override: Optional[torch.device] = None) -> torch.device:
        """Resolve the compute device.

        ``device_override`` (if given) wins; otherwise we honor ``self.device``
        (``None`` → auto-select ``cuda`` when available, else ``cpu``).
        """
        if device_override is not None:
            return device_override
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device) if isinstance(self.device, str) else self.device

    def _make_experiment_logger(self) -> FinetuningLogger:
        """Return a configured experiment logger.

        ``WandbLogger(**self.wandb_kwargs)`` when ``wandb_kwargs`` is set,
        else a :class:`NullLogger` (no-op). This is called once per
        :meth:`fit` on the master rank.
        """
        if self.wandb_kwargs is None:
            return NullLogger()
        kwargs = dict(self.wandb_kwargs)
        return WandbLogger(**kwargs)

    def _frozen_submodules(self, model: TabICL) -> list[nn.Module]:
        """Return the list of TabICL sub-modules that should stay frozen."""
        out: list[nn.Module] = []
        if self.freeze_col:
            out.append(model.col_embedder)
        if self.freeze_row:
            out.append(model.row_interactor)
        if self.freeze_icl:
            out.append(model.icl_predictor)
        return out

    def _apply_freezing(self, model: TabICL) -> bool:
        """Zero ``requires_grad`` on frozen sub-modules.

        Mode (train/eval) is handled separately by :meth:`_set_training_mode`
        so that calling ``model.train()`` elsewhere doesn't silently re-enable
        dropout / BN updates on the frozen parts.

        Returns
        -------
        bool
            True if at least one sub-module was frozen.
        """
        frozen_modules = self._frozen_submodules(model)
        for sub in frozen_modules:
            for p in sub.parameters():
                p.requires_grad = False
        return bool(frozen_modules)

    def _set_training_mode(self, training: bool) -> None:
        """Switch ``self.model_`` between train/eval while honoring freezes.

        ``nn.Module.train()`` is recursive, so naively calling
        ``self.model_.train()`` flips frozen sub-modules (and their dropout /
        BN) back into training mode. We first toggle the whole model, then
        snap frozen sub-modules back to ``eval()``.
        """
        self.model_.train(training)
        if training:
            for sub in self._frozen_submodules(self.model_):
                sub.eval()

    def _load_pretrained(self, device: torch.device) -> None:
        """Compose with the sklearn estimator's ``_load_model`` to pull the
        pretrained checkpoint, then steal the resulting ``model_``.
        """
        loader = self._create_inner_estimator(n_estimators=self.n_estimators_finetune, device=device)
        loader._resolve_device()
        loader._load_model()
        self.model_ = loader.model_
        self.model_config_ = loader.model_config_
        self.model_path_ = loader.model_path_

    def _build_inner_estimator(
        self,
        model: TabICL,
        n_estimators: int,
        device: torch.device,
    ):
        """Return an inner sklearn estimator pre-loaded with ``model``.

        Neutralizes ``_load_model`` on the instance so the subsequent ``.fit``
        does not re-download or re-load weights. Accessing ``inner.model_``
        returns the same Python object as ``model`` (no copy).
        """
        inner = self._create_inner_estimator(n_estimators=n_estimators, device=device)
        inner.model_ = model
        inner.model_config_ = self.model_config_
        inner.model_path_ = self.model_path_
        # Shadow the method with an instance attribute so inner.fit() skips reload.
        inner._load_model = lambda: None
        return inner

    # ---------------- checkpointing ----------------

    def _save_checkpoint(
        self,
        *,
        output_dir: Path,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        best_metric: float,
        is_best: bool,
        save_interval: bool,
    ) -> None:
        """Write a TabICL-compatible checkpoint file.

        The dict layout matches :meth:`tabicl.train._run.Trainer.save_checkpoint`
        so the resulting file loads directly via
        ``TabICLClassifier(model_path=<path>)`` / ``TabICLRegressor(model_path=<path>)``.

        Parameters
        ----------
        output_dir : Path
            Target directory. The best-metric snapshot is written to
            ``{output_dir}/best.ckpt`` and interval snapshots to
            ``{output_dir}/epoch{N}.ckpt``.

        epoch : int
            1-indexed epoch count at save time (stored under both ``epoch`` and
            ``curr_step`` for pretraining-schema compatibility).

        optimizer, scheduler : torch objects
            Their ``state_dict()`` is embedded so a future run could resume
            from this checkpoint.

        best_metric : float
            The current best validation metric; stored under ``best_metric``.

        is_best : bool
            If True, (over)write ``best.ckpt``.

        save_interval : bool
            If True, write the per-epoch ``epoch{N}.ckpt`` file. Set
            independently from ``is_best`` so both can be written in the same
            call at the best-and-interval-aligned epoch.
        """
        # ``self.model_`` is always the raw TabICL module: the DDP wrapper, if
        # any, is stored separately as ``self._ddp_model_``.
        ckpt = {
            "config": self.model_config_,
            "state_dict": {k: v.cpu() for k, v in self.model_.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            # ``curr_step`` matches the pretraining checkpoint schema so that
            # fine-tuned weights are loadable via ``TabICLClassifier(model_path=...)``.
            "curr_step": epoch,
            "epoch": epoch,
            "best_metric": float(best_metric),
        }
        if is_best:
            torch.save(ckpt, output_dir / "best.ckpt")
        if save_interval:
            torch.save(ckpt, output_dir / f"epoch{epoch}.ckpt")

    def _validate_current_model(
        self,
        device: torch.device,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> ValidationMetrics:
        """Build an inner estimator with current weights and run one validation pass.

        Flips the shared ``self.model_`` into eval mode while delegating to the
        subclass-specific :meth:`_run_validation` (which fits the inner
        estimator and computes metrics), then restores training mode with
        freeze flags honored.
        """
        inner = self._build_inner_estimator(self.model_, self.n_estimators_validation, device)
        self.model_.eval()
        try:
            return self._run_validation(inner, X_train, y_train, X_val, y_val)
        finally:
            # Only flip back to train mode if we're still inside a fit loop;
            # ``is_fitted_`` is set after the loop exits.
            if not getattr(self, "is_fitted_", False):
                self._set_training_mode(True)

    # ---------------- AMP ----------------

    def _make_amp(self, device: torch.device):
        """Build the AMP scaler + autocast-context factory for this device.

        AMP is only engaged on CUDA with ``self.amp=True``; in every other
        case (``amp=False``, CPU, MPS) ``scaler`` becomes a disabled
        :class:`torch.GradScaler` whose scale/step/update operations are
        no-ops, and ``amp_ctx_factory`` returns :class:`contextlib.nullcontext`
        so the training loop stays branch-free.

        Returns
        -------
        tuple
            ``(use_amp, scaler, amp_ctx_factory)``. ``amp_ctx_factory`` is a
            zero-arg callable that produces a fresh context manager per batch.
        """
        use_amp = self.amp and device.type == "cuda" and torch.cuda.is_available()
        scaler = torch.GradScaler("cuda", enabled=use_amp)
        if use_amp:
            amp_ctx_factory = lambda: torch.autocast(  # noqa: E731
                device_type="cuda",
                dtype=torch.float16,
            )
        else:
            from contextlib import nullcontext

            amp_ctx_factory = nullcontext
        return use_amp, scaler, amp_ctx_factory

    # ---------------- main loop ----------------

    def _fit_impl(self, X, y, *, X_val, y_val, output_dir: Optional[Path]) -> None:
        """Run the fine-tuning loop on already-resolved inputs.

        Invoked by :meth:`fit` inside a ``try/finally`` that guarantees
        ``dist.destroy_process_group`` on exit. The method walks 13 linear
        stages (DDP → logger → inputs → split → model load → freeze → DDP
        wrap → init-eval → scheduler → **main loop** → restore best →
        final checkpoint → inference wiring) without branching on DDP except
        where rank-dependent behavior is needed.
        """
        from sklearn.utils.validation import check_X_y, check_array

        # 1. DDP setup
        using_ddp, _, _, local_rank, master, device = _ddp_env(self._resolve_device())
        self._is_master_ = master

        # 2. Logger (WandbLogger when wandb_kwargs is set, else no-op)
        _logger: FinetuningLogger = NullLogger()
        if master:
            try:
                _logger = self._make_experiment_logger()
                _logger.setup({k: v for k, v in self.get_params().items() if k != "wandb_kwargs"})
            except (OSError, ModuleNotFoundError):
                logger.warning("Experiment logger setup failed; falling back to NullLogger.", exc_info=True)
                _logger = NullLogger()

        # 3. Input validation / stash
        ensure_numeric_y = self._model_type == "regressor"
        X, y = check_X_y(
            X,
            y,
            ensure_all_finite=False,
            y_numeric=ensure_numeric_y,
            dtype=None,
        )
        self.X_raw_ = X
        self.y_raw_ = y

        # 4. Val split
        if X_val is not None and y_val is not None:
            X_train_arr, y_train_arr = X, y
            X_val_arr = check_array(X_val, ensure_all_finite=False, dtype=None)
            y_val_arr = np.asarray(y_val)
        else:
            stratify = y if self._model_type == "classifier" else None
            X_train_arr, X_val_arr, y_train_arr, y_val_arr = train_test_split(
                X,
                y,
                test_size=self.validation_split_ratio,
                random_state=self.random_state,
                stratify=stratify,
            )

        # 5. Load pretrained model
        self._load_pretrained(device)
        self.model_.to(device)

        # 6. Freezing
        any_frozen = self._apply_freezing(self.model_)

        # 7. DDP wrap
        if using_ddp:
            ddp_model = DDP(
                self.model_,
                device_ids=[local_rank],
                broadcast_buffers=False,
                find_unused_parameters=any_frozen,
            )
            self._ddp_model_ = ddp_model
        else:
            self._ddp_model_ = None

        params = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        _, scaler, amp_ctx_factory = self._make_amp(device)

        # 8. Initial-eval safety net (rank 0 only)
        if master:
            baseline_result = self._validate_current_model(device, X_train_arr, y_train_arr, X_val_arr, y_val_arr)
            best_metric = baseline_result.primary
            msg = f"Baseline val {self._metric_name}: {best_metric:.4f}"
            if self.verbose:
                tqdm.write(msg)
            logger.info(msg)
        else:
            best_metric = self._initial_best_metric()
        best_metric = self._broadcast_metric(best_metric, using_ddp, device)

        best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}

        # 9. Estimate steps per epoch for scheduler
        chunks_per_epoch = max(1, count_chunks(len(y_train_arr), self.max_data_size))
        total_steps = max(1, self.epochs * chunks_per_epoch)
        scheduler = None
        if self.use_lr_scheduler:
            sched_cfg = SimpleNamespace(
                max_steps=total_steps,
                warmup_proportion=self.warmup_proportion,
                warmup_steps=0,
                scheduler="cosine_warmup",
            )
            scheduler = get_scheduler(sched_cfg, optimizer)

        # 10. Main loop
        global_step = 0
        patience_counter = 0
        start_time = time.monotonic()
        show_bar = master and self.verbose
        epoch_iter = tqdm(
            range(self.epochs),
            desc="Fine-tune",
            disable=not show_bar,
            leave=True,
        )

        for epoch in epoch_iter:
            epoch_loss_sum = 0.0
            epoch_n_batches = 0
            epoch_seed = self.random_state + epoch
            epoch_start = time.monotonic()

            # 10.1  Prepare this epoch's meta-batch iterator (fresh random
            #       chunking + context/query splits via ``epoch_seed``).
            self._set_training_mode(True)
            meta_iter = iter_epoch_meta_batches(
                X_train_arr,
                y_train_arr,
                classification=self._model_type == "classifier",
                n_estimators=self.n_estimators_finetune,
                max_chunk_size=self.max_data_size,
                query_ratio=self.finetune_ctx_query_ratio,
                epoch_seed=epoch_seed,
                preprocessing_seed=self.random_state,
                norm_methods=self.norm_methods,
                feat_shuffle_method=self.feat_shuffle_method,
                class_shuffle_method=getattr(self, "class_shuffle_method", "shift"),
                outlier_threshold=self.outlier_threshold,
            )
            # Only show the nested per-batch bar when there's more than one
            # chunk in the epoch — otherwise the outer epoch bar is enough.
            batch_iter = (
                tqdm(
                    meta_iter,
                    total=chunks_per_epoch,
                    desc=f"  epoch {epoch + 1}/{self.epochs}",
                    leave=False,
                    disable=not show_bar,
                )
                if chunks_per_epoch > 1
                else meta_iter
            )

            # 10.2  Step through each meta-batch: forward + loss + backward
            #       + grad-clip + optimizer + scheduler + step logging.
            for batch in batch_iter:
                if self._skip_batch(batch, using_ddp, device):
                    continue
                batch = move_meta_batch(batch, device)

                optimizer.zero_grad(set_to_none=True)
                forward_model = self._ddp_model_ if using_ddp else self.model_
                with amp_ctx_factory():
                    loss = self._compute_batch_loss(batch, forward_model)

                scaler.scale(loss).backward()
                if self.grad_clip is not None and self.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

                loss_val = float(loss.detach().item())
                epoch_loss_sum += loss_val
                epoch_n_batches += 1
                global_step += 1
                if master:
                    cur_lr = scheduler.get_last_lr()[0] if scheduler is not None else self.learning_rate
                    _logger.log_step(
                        {"train/loss": loss_val, "train/lr": cur_lr, "train/epoch": epoch},
                        step=global_step,
                    )
                    if show_bar and hasattr(batch_iter, "set_postfix"):
                        batch_iter.set_postfix(loss=f"{loss_val:.4f}", lr=f"{cur_lr:.2e}")

            mean_train_loss = epoch_loss_sum / epoch_n_batches if epoch_n_batches > 0 else None
            epoch_time = time.monotonic() - epoch_start

            # 10.3  End-of-epoch validation on rank 0, broadcast primary
            #       metric to all ranks so they agree on early-stopping.
            if master:
                eval_result = self._validate_current_model(device, X_train_arr, y_train_arr, X_val_arr, y_val_arr)
                primary = eval_result.primary
                epoch_metrics = {
                    "train/epoch": epoch,
                    f"val/{self._metric_name}": primary,
                }
                if mean_train_loss is not None:
                    epoch_metrics["train/mean_loss"] = mean_train_loss
                for k, v in eval_result.secondary.items():
                    epoch_metrics[f"val/{k}"] = v
                _logger.log_epoch(epoch_metrics, step=global_step)
                train_loss_str = f"{mean_train_loss:.4f}" if mean_train_loss is not None else "N/A"
                summary = (
                    f"epoch {epoch + 1}/{self.epochs} | "
                    f"train_loss={train_loss_str} | "
                    f"val_{self._metric_name}={primary:.4f} | "
                    f"time={epoch_time:.1f}s"
                )
                logger.info(summary)
                if show_bar:
                    display_best = (
                        primary
                        if not np.isnan(primary) and self._metric_improved(primary, best_metric)
                        else best_metric
                    )
                    tqdm.write(summary)
                    epoch_iter.set_postfix(
                        {
                            "train_loss": train_loss_str,
                            f"val_{self._metric_name}": f"{primary:.4f}",
                            "best": f"{display_best:.4f}",
                            "s/epoch": f"{epoch_time:.1f}",
                        }
                    )
            else:
                primary = self._initial_best_metric()

            primary = self._broadcast_metric(primary, using_ddp, device)

            # 10.4  Decide whether to checkpoint (best or interval-aligned).
            is_best = not np.isnan(primary) and self._metric_improved(primary, best_metric)
            save_this_interval = (
                output_dir is not None and self.save_interval > 0 and ((epoch + 1) % self.save_interval == 0)
            )
            if output_dir is not None and master and (is_best or save_this_interval):
                self._save_checkpoint(
                    output_dir=output_dir,
                    epoch=epoch + 1,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric=primary if is_best else best_metric,
                    is_best=is_best,
                    save_interval=save_this_interval,
                )

            # 10.5  Update best-state / patience, maybe early-stop.
            if self.early_stopping and not np.isnan(primary):
                if is_best:
                    best_metric = primary
                    patience_counter = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                else:
                    patience_counter += 1
                if patience_counter >= self.patience:
                    if master:
                        logger.info(
                            "Early stopping at epoch %d (best %s=%.4f)",
                            epoch,
                            self._metric_name,
                            best_metric,
                        )
                    break

            # 10.6  Wall-clock time-budget check — stop if the *next* epoch
            #       would push us past ``self.time_limit``.
            if self.time_limit is not None:
                elapsed = time.monotonic() - start_time
                epochs_done = epoch + 1
                projected = elapsed + (elapsed / epochs_done)
                if projected > self.time_limit:
                    if master:
                        logger.info(
                            "Time limit reached (%.1fs > %.1fs); stopping.",
                            projected,
                            self.time_limit,
                        )
                    break

        # 11. Restore best weights
        self.model_.load_state_dict(best_state)
        self._best_metric_ = best_metric

        # 11b. Always write a final best checkpoint so users have a single file
        # to point TabICLClassifier(model_path=...) at. If no epoch ever
        # improved on the baseline, this is the baseline weights — which is
        # exactly what the final estimator will use.
        if output_dir is not None and master:
            self._save_checkpoint(
                output_dir=output_dir,
                epoch=self.epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                best_metric=best_metric,
                is_best=True,
                save_interval=False,
            )

        # 12. Finalize logger
        if master:
            _logger.finish()

        # 13. Build final inner estimator on full training data. Switch the
        # underlying module to eval mode so `_final_estimator_.predict` routes
        # through the inference forward (which emits n_classes-wide logits /
        # proper quantile tensors), not the training forward.
        if master:
            self.model_.eval()
            self._final_estimator_ = self._build_inner_estimator(self.model_, self.n_estimators_inference, device)
            self._final_estimator_.fit(self.X_raw_, self.y_raw_)

        self.is_fitted_ = True
        # Drop heavy training-only refs to keep pickle clean
        self._ddp_model_ = None

    # ---------------- hooks / small overrides ----------------

    def _skip_batch(self, batch: MetaBatch, using_ddp: bool, device: torch.device) -> bool:
        """Decide whether ``batch`` should be skipped, consistently across ranks.

        Subclasses override :meth:`_task_skip_batch` to express a local
        decision (e.g. classifier skipping batches with unseen query labels).
        Under DDP we ``all_reduce(MAX)`` the flag so every rank skips together,
        which is required to keep DDP's backward all-reduce from deadlocking.
        """
        should_skip = self._task_skip_batch(batch)
        if using_ddp:
            t = torch.tensor([int(should_skip)], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            should_skip = bool(t.item())
        return should_skip

    def _task_skip_batch(self, batch: MetaBatch) -> bool:
        """Per-task skip predicate; default is never skip.

        Classifier overrides to skip batches whose query contains classes
        missing from the context.
        """
        del batch
        return False

    @staticmethod
    def _broadcast_metric(value: float, using_ddp: bool, device: torch.device) -> float:
        """Broadcast a scalar metric from rank 0 to all ranks.

        No-op outside DDP. Used for early-stopping / time-limit consensus so
        every rank takes the same stop decision.
        """
        if not using_ddp:
            return float(value)
        t = torch.tensor([value], dtype=torch.float64, device=device)
        dist.broadcast(t, src=0)
        return float(t.item())

    # ---------------- pickle ----------------

    def __getstate__(self):
        """Strip training-only refs before pickling.

        The heavy :class:`TabICL` module and the DDP wrapper are dropped;
        inference relies solely on ``self._final_estimator_``, whose own
        ``__getstate__`` handles the weights.
        """
        state = self.__dict__.copy()
        state["model_"] = None
        state.pop("_ddp_model_", None)
        return state

    def __setstate__(self, state):
        """Default unpickling — no device re-resolution needed because the
        inference path lives entirely inside ``self._final_estimator_``.
        """
        self.__dict__.update(state)
