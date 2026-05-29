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
from sklearn.preprocessing import LabelEncoder
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
        """在冻结的子模块上设置 requires_grad 为 False。

        通过 :meth:`_set_training_mode` 分别处理模式（train/eval），

        这样在其他地方调用 ``model.train()`` 时不会静默地重新启用

        冻结部分的 dropout / BN 更新。

        返回值

        -------

        bool

        如果至少有一个子模块被冻结，则返回 True。
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
        
        这是微调的核心循环。被外部的 fit() 方法调用。
        整个过程严格按照 13 个线性阶段执行，除了涉及多卡分布式训练 (DDP) 需要根据显卡序号(rank)决定行为外，
        其他逻辑尽可能保持一致，避免复杂的条件分支。
        """
        # 导入 scikit-learn 的数据校验工具，确保输入的数据格式合法
        from sklearn.utils.validation import check_X_y, check_array

        # ---------------------------------------------------------
        # 1. DDP setup (分布式数据并行设置)
        # 获取当前运行环境是否在使用多显卡并行，以及主节点(master)、设备(device)信息
        # ---------------------------------------------------------
        using_ddp, rank, world_size, local_rank, master, device = _ddp_env(self._resolve_device())
        self._is_master_ = master # 标记当前进程是否为主进程 (通常 rank=0 为主进程负责打印和存盘)

        # ---------------------------------------------------------
        # 2. Logger (日志记录器初始化)
        # 初始化实验记录器 (比如 Weights & Biases)，只有主进程(master)需要记录日志
        # ---------------------------------------------------------
        _logger: FinetuningLogger = NullLogger() # 默认使用空日志器，什么都不做
        if master:
            try:
                # 尝试初始化真实的日志记录器，并记录当前的超参数配置
                _logger = self._make_experiment_logger()
                _logger.setup({k: v for k, v in self.get_params().items() if k != "wandb_kwargs"})
            except (OSError, ModuleNotFoundError):
                # 如果没安装 wandb 或报错，则回退到空日志器
                logger.warning("Experiment logger setup failed; falling back to NullLogger.", exc_info=True)
                _logger = NullLogger()

        # ---------------------------------------------------------
        # 3. Input validation / stash (输入数据校验与暂存)
        # ---------------------------------------------------------
        # 如果是回归任务，确保目标变量 y 是数值型
        ensure_numeric_y = self._model_type == "regressor"
        X, y = check_X_y(
            X,
            y,
            ensure_all_finite=False, # 允许有缺失值
            y_numeric=ensure_numeric_y,
            dtype=None,
        )
        self.X_raw_ = X
        self.y_raw_ = y  # 暂存原始数据，最终拟合完整模型时会用到

        # 3b. 标签编码 (分类任务特有)
        # 分类任务的标签可能是字符串(如"猫","狗")或者不连续的数字(-1, 5)。
        # 神经网络只能接收 0 到 K-1 的整数，所以需要用 LabelEncoder 转换一下。
        if self._model_type == "classifier":
            self._label_encoder_ = LabelEncoder().fit(y)
            y_fit = self._label_encoder_.transform(y).astype(np.int64)
        else:
            y_fit = y # 回归任务不需要转换标签

        # ---------------------------------------------------------
        # 4. Val split (划分验证集)
        # ---------------------------------------------------------
        if X_val is not None and y_val is not None:
            # 如果用户自己提供了验证集，直接使用并进行必要的校验和标签转换
            X_train_arr, y_train_arr = X, y_fit
            X_val_arr = check_array(X_val, ensure_all_finite=False, dtype=None)
            y_val_arr = np.asarray(y_val)
            if self._model_type == "classifier":
                y_val_arr = self._label_encoder_.transform(y_val_arr).astype(np.int64)
        else:
            # 如果用户没提供，则自动从训练集中切分一部分作为验证集
            # 分类任务会根据标签分布(stratify)进行分层抽样，保证类别比例一致
            stratify = y_fit if self._model_type == "classifier" else None
            X_train_arr, X_val_arr, y_train_arr, y_val_arr = train_test_split(
                X,
                y_fit,
                test_size=self.validation_split_ratio,
                random_state=self.random_state,
                stratify=stratify,
            )

        # ---------------------------------------------------------
        # 5. Load pretrained model (加载预训练的神经网络模型)
        # 将模型放到指定的设备(CPU或GPU)上
        # ---------------------------------------------------------
        self._load_pretrained(device)
        self.model_.to(device)

        # ---------------------------------------------------------
        # 6. Freezing (冻结参数)
        # 微调时通常不需要更新所有网络层的参数。这里会把一些底层特征提取层的参数冻结(不可变)，
        # 只训练最上面的几层，以防过拟合。
        # ---------------------------------------------------------
        any_frozen = self._apply_freezing(self.model_)

        # ---------------------------------------------------------
        # 7. DDP wrap & Optimizer (多卡包装和优化器设置)
        # ---------------------------------------------------------
        if using_ddp:
            # 如果是多卡训练，用 DDP (DistributedDataParallel) 把模型包起来
            ddp_model = DDP(
                self.model_,
                device_ids=[local_rank],
                broadcast_buffers=False,
                find_unused_parameters=any_frozen, # 如果有冻结的层，这里必须是 True
            )
            self._ddp_model_ = ddp_model
        else:
            self._ddp_model_ = None

        # 找出所有需要计算梯度的(未冻结的)参数，传给 AdamW 优化器
        params = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        # 初始化 AMP (自动混合精度)，用于加速训练并节省显存
        _, scaler, amp_ctx_factory = self._make_amp(device)

        # ---------------------------------------------------------
        # 8. Initial-eval safety net (微调前的基准测试)
        # 先不训练，直接用预训练模型在验证集上跑一下，记录一个基准分数
        # ---------------------------------------------------------
        if master:
            baseline_result = self._validate_current_model(device, X_train_arr, y_train_arr, X_val_arr, y_val_arr)
            best_metric = baseline_result.primary # 初始化最佳指标
            msg = f"Baseline val {self._metric_name}: {best_metric:.4f}"
            if self.verbose:
                tqdm.write(msg)
            logger.info(msg)
        else:
            best_metric = self._initial_best_metric()
        
        # 将主进程的基准分数同步给其他所有进程，保证大家步调一致
        best_metric = self._broadcast_metric(best_metric, using_ddp, device)
        # 把当前的(未微调的)模型权重拷贝一份作为“目前最佳状态”保存
        best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}

        # ---------------------------------------------------------
        # 9. Scheduler setup (学习率调度器设置)
        # 计算总步数，用于配置学习率的变化曲线 (比如先预热，后平滑下降)
        # ---------------------------------------------------------
        chunks_per_epoch = max(
            1,
            count_chunks(len(y_train_arr), self.max_data_size, rank=rank, world_size=world_size),
        )
        if using_ddp and master:
            global_chunks = count_chunks(len(y_train_arr), self.max_data_size)
            if global_chunks < world_size:
                warnings.warn(
                    f"Fine-tune produced {global_chunks} chunk(s)/epoch but "
                    f"world_size={world_size}; falling back to chunk replication "
                    f"(no DDP speedup). Increase n_samples or decrease "
                    f"max_data_size to benefit from DDP.",
                    UserWarning,
                    stacklevel=2,
                )
        total_steps = max(1, self.epochs * chunks_per_epoch)
        scheduler = None
        if self.use_lr_scheduler:
            sched_cfg = SimpleNamespace(
                max_steps=total_steps,
                warmup_proportion=self.warmup_proportion,
                warmup_steps=0,
                scheduler="cosine_warmup", # 使用带有预热的余弦退火学习率
            )
            scheduler = get_scheduler(sched_cfg, optimizer)

        # ---------------------------------------------------------
        # 10. Main loop (核心训练大循环)
        # ---------------------------------------------------------
        global_step = 0
        patience_counter = 0 # 早停计数器 (多少个 epoch 没进步了)
        start_time = time.monotonic()
        show_bar = master and self.verbose # 是否显示进度条
        
        # 外层循环：迭代各个 Epoch (整个训练集过一遍叫一个 Epoch)
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

            # 10.1 准备这一个 Epoch 的数据生成器
            # TabICL 比较特殊，它的输入往往是 "Meta-batch"(元批次)，包含上下文(context)和查询(query)
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
                rank=rank,
                world_size=world_size,
            )
            # 内层进度条 (遍历每个 Epoch 里面的数据块 batch)
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

            # 10.2 Step through each batch (网络前向传播、算损失、反向传播、更新参数)
            for batch in batch_iter:
                if self._skip_batch(batch, using_ddp, device):
                    continue
                batch = move_meta_batch(batch, device) # 把数据丢到 GPU

                optimizer.zero_grad(set_to_none=True) # 梯度清零
                forward_model = self._ddp_model_ if using_ddp else self.model_
                
                # 开启混合精度上下文
                with amp_ctx_factory():
                    loss = self._compute_batch_loss(batch, forward_model) # 计算 Loss

                # 反向传播计算梯度
                scaler.scale(loss).backward()
                
                # 梯度裁剪 (防止梯度爆炸)
                if self.grad_clip is not None and self.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                    
                # 优化器更新参数
                scaler.step(optimizer)
                scaler.update()
                
                # 更新学习率
                if scheduler is not None:
                    scheduler.step()

                # 记录这一个 batch 的损失值
                loss_val = float(loss.detach().item())
                epoch_loss_sum += loss_val
                epoch_n_batches += 1
                global_step += 1
                
                # 主进程打印当前进度和信息到 W&B 记录器
                if master:
                    cur_lr = scheduler.get_last_lr()[0] if scheduler is not None else self.learning_rate
                    _logger.log_step(
                        {"train/loss": loss_val, "train/lr": cur_lr, "train/epoch": epoch},
                        step=global_step,
                    )
                    if show_bar and hasattr(batch_iter, "set_postfix"):
                        batch_iter.set_postfix(loss=f"{loss_val:.4f}", lr=f"{cur_lr:.2e}")

            # 计算这一个 Epoch 的平均训练损失
            mean_train_loss = epoch_loss_sum / epoch_n_batches if epoch_n_batches > 0 else None
            epoch_time = time.monotonic() - epoch_start

            # 10.3 End-of-epoch validation (每个 Epoch 结束后，在验证集上评估一次模型)
            if master:
                eval_result = self._validate_current_model(device, X_train_arr, y_train_arr, X_val_arr, y_val_arr)
                primary = eval_result.primary # 核心评估指标 (如 准确率, R平方 等)
                
                # 记录 Epoch 级别的指标到日志器
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

            # 同步验证集结果到所有显卡进程，确保所有进程对于“模型变好了没有”这件事认知一致
            primary = self._broadcast_metric(primary, using_ddp, device)

            # 10.4 决定是否要将当前模型的权重存成物理文件 (Checkpoint)
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

            # 10.5 Early-stopping (早停机制：如果模型连续若干个 Epoch 在验证集上都没进步，就提前结束训练)
            if self.early_stopping and not np.isnan(primary):
                if is_best:
                    # 如果创新高了，更新最高记录，重置耐心计数器，把当前权重备份到内存(best_state)
                    best_metric = primary
                    patience_counter = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                else:
                    # 如果没进步，耐心计数器 +1
                    patience_counter += 1
                if patience_counter >= self.patience: # 耐心耗尽
                    if master:
                        logger.info(
                            "Early stopping at epoch %d (best %s=%.4f)",
                            epoch,
                            self._metric_name,
                            best_metric,
                        )
                    break # 跳出整个外层训练循环，停止训练

            # 10.6 Wall-clock time-budget check (超时检查)
            # 如果训练时间超出了用户设置的 time_limit，也会提前结束
            if self.time_limit is not None:
                if master:
                    elapsed = time.monotonic() - start_time
                    projected = elapsed + (elapsed / (epoch + 1))
                    should_stop = projected > self.time_limit
                    stop_log = (projected, self.time_limit) if should_stop else None
                else:
                    should_stop = False
                    stop_log = None
                should_stop = self._broadcast_metric(float(should_stop), using_ddp, device) > 0.5
                if should_stop:
                    if master and stop_log is not None:
                        logger.info(
                            "Time limit reached (%.1fs > %.1fs); stopping.",
                            stop_log[0],
                            stop_log[1],
                        )
                    break

        # ---------------------------------------------------------
        # 11. Restore best weights (训练结束，恢复成表现最好那一次的模型状态)
        # ---------------------------------------------------------
        self.model_.load_state_dict(best_state)
        self._best_metric_ = best_metric

        # 11b. 将最好的状态保存到本地文件中
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

        # ---------------------------------------------------------
        # 12. Finalize logger (关闭实验记录器)
        # ---------------------------------------------------------
        if master:
            _logger.finish()

        # ---------------------------------------------------------
        # 13. Build final inner estimator (封装最终的推理预测器)
        # 微调结束后，把微调好的网络装入到一个 scikit-learn 风格的包装器中。
        # 用所有的原始数据(X_raw_, y_raw_)再对分类器做一些轻量级的校准。
        # ---------------------------------------------------------
        if master:
            self.model_.eval() # 切换为推理模式
            self._final_estimator_ = self._build_inner_estimator(self.model_, self.n_estimators_inference, device)
            self._final_estimator_.fit(self.X_raw_, self.y_raw_)
            # 修改序列化(pickle)配置，确保保存最终模型时能连带权重一起保存
            self._final_estimator_._save_model_weights = True
            self._final_estimator_.__dict__.pop("_load_model", None)

        self.is_fitted_ = True # 标记为已经训练完成
        # Drop heavy training-only refs to keep pickle clean
        self._ddp_model_ = None
    # ---------------- hooks / small overrides ----------------

    def _skip_batch(self, batch: MetaBatch, using_ddp: bool, device: torch.device) -> bool:
        """Decide whether ``batch`` should be skipped, consistently across ranks.

        Subclasses override :meth:`_task_skip_batch` to express a local
        decision (e.g. classifier skipping batches with unseen query labels).
        Under DDP we ``all_reduce(MAX)`` the flag so every rank skips together,
        which is required to keep DDP's backward all-reduce from deadlocking.
        With rank-aware chunk sharding, ranks may inspect different chunks and
        disagree on the local skip flag; MAX consensus means "if any rank
        wants to skip, every rank drops its own shard-chunk this iteration",
        which is slightly more conservative than single-GPU but preserves
        step-count parity.
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
