#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent


def add_sys_path_if_exists(path: Path | None, *, prepend: bool = True) -> None:
    if path is None:
        return
    resolved = path.expanduser().resolve()
    if resolved.exists() and str(resolved) not in sys.path:
        if prepend:
            sys.path.insert(0, str(resolved))
        else:
            sys.path.append(str(resolved))


add_sys_path_if_exists(SCRIPT_DIR / "src", prepend=True)

extensions_path_env = os.environ.get("TABPFN_EXTENSIONS_SRC_DIR", "").strip()
extensions_candidates = [
    Path(extensions_path_env) if extensions_path_env else None,
    SCRIPT_DIR.parent / "tabpfn-extensions" / "src",
    SCRIPT_DIR.parent / "tabpfn-extensions",
    Path.home() / "pythonlibs" / "tabpfn_manyclass" / "src",
    Path.home() / "pythonlibs" / "tabpfn_manyclass",
]
for candidate in extensions_candidates:
    add_sys_path_if_exists(candidate, prepend=False)


def find_tabpfn_extensions_package_dir() -> Path | None:
    for candidate in extensions_candidates:
        if candidate is None:
            continue
        resolved = candidate.expanduser().resolve()
        package_dir = resolved / "tabpfn_extensions"
        if package_dir.is_dir():
            return package_dir
    return None


TABPFN_EXTENSIONS_PACKAGE_DIR = find_tabpfn_extensions_package_dir()


def import_many_class_classifier():
    try:
        from tabpfn_extensions.many_class import ManyClassClassifier

        return ManyClassClassifier
    except Exception:
        if TABPFN_EXTENSIONS_PACKAGE_DIR is None:
            raise

        for module_name in list(sys.modules):
            if module_name == "tabpfn_extensions" or module_name.startswith(
                "tabpfn_extensions."
            ):
                sys.modules.pop(module_name, None)

        package = types.ModuleType("tabpfn_extensions")
        package.__file__ = str(TABPFN_EXTENSIONS_PACKAGE_DIR / "__init__.py")
        package.__package__ = "tabpfn_extensions"
        package.__path__ = [str(TABPFN_EXTENSIONS_PACKAGE_DIR)]
        sys.modules["tabpfn_extensions"] = package

        importlib.invalidate_caches()
        module = importlib.import_module("tabpfn_extensions.many_class")
        return module.ManyClassClassifier

import numpy as np
import pandas as pd

CLASSIFICATION_TASKS = {"binclass", "multiclass"}
CATEGORICAL_MISSING_TOKEN = "__tabicl_missing__"
DEFAULT_MODEL_VERSION = "v2.5"
TABPFN_CLASS_LIMIT = 10
GATED_CLASSIFIER_CACHE_FILES = {
    "v2.5": "tabpfn-v2.5-classifier-v2.5_default.ckpt",
    "v2.6": "tabpfn-v2.6-classifier-v2.6_default.ckpt",
}

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")


@dataclass
class LoadedDataset:
    dataset_name: str
    dataset_dir: Path
    task_type: Optional[str]
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame | None
    y_val: np.ndarray | None
    X_test: pd.DataFrame
    y_test: np.ndarray
    categorical_feature_indices: list[int]

    @property
    def n_val(self) -> int:
        return 0 if self.y_val is None else int(len(self.y_val))

    @property
    def X_train_merged(self) -> pd.DataFrame:
        if self.X_val is None:
            return self.X_train
        return pd.concat([self.X_train, self.X_val], axis=0, ignore_index=True)

    @property
    def y_train_merged(self) -> np.ndarray:
        if self.y_val is None:
            return self.y_train
        return np.concatenate([np.asarray(self.y_train), np.asarray(self.y_val)], axis=0)

    @property
    def n_train_report(self) -> int:
        return int(len(self.y_train_merged))

    @property
    def n_classes(self) -> int:
        pieces = [np.asarray(self.y_train)]
        if self.y_val is not None:
            pieces.append(np.asarray(self.y_val))
        pieces.append(np.asarray(self.y_test))
        return int(len(pd.unique(pd.Series(np.concatenate(pieces, axis=0)))))


@dataclass
class PredictionResult:
    y_pred: Any
    fit_seconds: float
    predict_seconds: float
    n_train_a: int = 0
    n_train_b: int = 0
    n_holdout_c: int = 0
    n_test_d: int = 0
    ttt_loss: Optional[float] = None
    ttt_steps: int = 0
    ttt_lr: Optional[float] = None
    ttt_applied: bool = False
    ttt_update_seconds: float = 0.0
    ttt_split_strategy: Optional[str] = None
    ttt_split_reason: Optional[str] = None


@dataclass
class ResultRow:
    dataset_name: str
    dataset_dir: str
    task_type: Optional[str]
    n_train: int
    n_val: int
    n_test: int
    n_features: int
    n_classes: int
    accuracy: Optional[float]
    fit_seconds: float
    predict_seconds: float
    status: str
    error: Optional[str]
    n_train_a: int = 0
    n_train_b: int = 0
    n_holdout_c: int = 0
    n_test_d: int = 0
    ttt_loss: Optional[float] = None
    ttt_steps: int = 0
    ttt_lr: Optional[float] = None
    ttt_applied: bool = False
    ttt_update_seconds: float = 0.0
    ttt_split_strategy: Optional[str] = None
    ttt_split_reason: Optional[str] = None


@dataclass
class TTTConfig:
    enabled: bool = True
    lr: float = 5e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    train_fraction: float = 0.75
    steps: int = 1
    random_state: int = 42
    n_estimators_finetune: int = 1
    train_input_encoders: bool = False
    micro_batch_size: int = 1
    data_parallel: bool = True
    gpu_group: Optional[str] = None


@dataclass
class TTTSplit:
    b_indices: np.ndarray
    c_indices: np.ndarray
    strategy: str
    reason: str


@dataclass
class TTTUpdateResult:
    applied: bool
    loss: Optional[float]
    steps: int
    update_seconds: float
    reason: Optional[str] = None


RESULT_COLUMNS = list(ResultRow.__annotations__.keys())
OOM_ERROR_MARKERS = (
    "out of memory",
    "oom",
    "cuda error: out of memory",
    "cudnn_status_alloc_failed",
    "tabpfncudaoutofmemoryerror",
    "tabpfnmpsoutofmemoryerror",
)


class SkipDataset(Exception):
    pass


def build_ttt_config(args: argparse.Namespace) -> TTTConfig:
    if int(args.ttt_steps) < 0:
        raise ValueError("--ttt-steps must be >= 0")
    ttt_n_estimators_finetune = (
        int(args.n_estimators)
        if args.ttt_n_estimators_finetune is None
        else int(args.ttt_n_estimators_finetune)
    )
    if ttt_n_estimators_finetune < 1:
        raise ValueError("--ttt-n-estimators-finetune must be >= 1")
    if int(args.ttt_micro_batch_size) < 1:
        raise ValueError("--ttt-micro-batch-size must be >= 1")
    train_fraction = float(args.ttt_train_fraction)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--ttt-train-fraction must be between 0 and 1")

    return TTTConfig(
        enabled=bool(args.ttt_holdout),
        lr=float(args.ttt_lr),
        weight_decay=float(args.ttt_weight_decay),
        grad_clip=float(args.ttt_grad_clip),
        train_fraction=train_fraction,
        steps=int(args.ttt_steps),
        random_state=int(args.random_state),
        n_estimators_finetune=ttt_n_estimators_finetune,
        train_input_encoders=bool(args.ttt_train_input_encoders),
        micro_batch_size=int(args.ttt_micro_batch_size),
        data_parallel=bool(args.ttt_data_parallel),
        gpu_group=getattr(args, "gpu_group", None),
    )


def take_rows(values, indices: np.ndarray):
    if hasattr(values, "iloc"):
        return values.iloc[indices].reset_index(drop=True)
    return np.asarray(values)[indices]


def split_ttt_holdout(y: np.ndarray, config: TTTConfig) -> TTTSplit:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(y), dtype=int)
    if len(indices) < 2:
        return TTTSplit(
            b_indices=indices,
            c_indices=np.asarray([], dtype=int),
            strategy="none",
            reason="Need at least two samples for B/C holdout split.",
        )

    y_array = np.asarray(y)
    try:
        b_idx, c_idx = train_test_split(
            indices,
            train_size=config.train_fraction,
            random_state=config.random_state,
            shuffle=True,
            stratify=y_array,
        )
        return TTTSplit(
            b_indices=np.asarray(b_idx, dtype=int),
            c_indices=np.asarray(c_idx, dtype=int),
            strategy="stratified",
            reason="stratified split",
        )
    except Exception as exc:
        b_idx, c_idx = train_test_split(
            indices,
            train_size=config.train_fraction,
            random_state=config.random_state,
            shuffle=True,
            stratify=None,
        )
        return TTTSplit(
            b_indices=np.asarray(b_idx, dtype=int),
            c_indices=np.asarray(c_idx, dtype=int),
            strategy="random",
            reason=f"stratified split unavailable: {type(exc).__name__}: {exc}",
        )


def _set_requires_grad(module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _configure_ttt_trainable_params(classifier, config: TTTConfig):
    model = classifier.model_
    model.train()
    for param in model.parameters():
        param.requires_grad = False

    if config.train_input_encoders:
        for module_name in ("encoder", "y_encoder"):
            module = getattr(model, module_name, None)
            if module is not None:
                module.train()
                _set_requires_grad(module, True)

    for module_name in (
        "transformer_encoder",
        "transformer_decoder",
        "encoder_compression_layer",
        "decoder_dict",
        "feature_positional_embedding_embeddings",
        "global_att_embeddings_for_compression",
    ):
        module = getattr(model, module_name, None)
        if module is not None:
            module.train()
            _set_requires_grad(module, True)

    return [param for param in model.parameters() if param.requires_grad]


def _get_ttt_base_model(classifier):
    model = classifier.model_
    return getattr(model, "module", model)


def _resolve_ttt_data_parallel_device_ids(
    classifier,
    config: TTTConfig,
) -> tuple[list[int], str | None]:
    if not config.data_parallel:
        return [], "--ttt-data-parallel is disabled"
    if not config.gpu_group:
        return [], "no worker GPU group is configured"

    try:
        gpu_ids = parse_gpu_id_list(normalize_gpu_group(config.gpu_group))
    except Exception as exc:
        return [], f"invalid GPU group {config.gpu_group!r}: {exc}"
    if len(gpu_ids) < 2:
        return [], "GPU group has fewer than 2 GPUs"
    if config.n_estimators_finetune < 2:
        return [], "--ttt-n-estimators-finetune is smaller than 2"

    try:
        import torch
    except Exception as exc:
        return [], f"failed to import torch: {type(exc).__name__}: {exc}"

    if not torch.cuda.is_available():
        return [], "CUDA is not available"

    device = getattr(classifier, "devices_", [None])[0]
    device_type = getattr(device, "type", str(device).split(":")[0])
    if device_type != "cuda":
        return [], f"classifier device is not CUDA: {device}"

    visible_count = int(torch.cuda.device_count())
    local_count = min(len(gpu_ids), visible_count, int(config.n_estimators_finetune))
    if local_count < 2:
        return [], (
            f"fewer than 2 process-local CUDA devices are visible "
            f"(gpu_group={config.gpu_group}, torch_count={visible_count})"
        )

    # The worker has rewritten CUDA_VISIBLE_DEVICES to the physical group, so
    # DataParallel must use process-local logical ids.
    return list(range(local_count)), None


def _build_ttt_forward_models(
    classifier,
    config: TTTConfig,
    *,
    verbose: bool,
) -> tuple[Any, Any | None, int]:
    base_model = _get_ttt_base_model(classifier)
    device_ids, reason = _resolve_ttt_data_parallel_device_ids(classifier, config)
    if not device_ids:
        if verbose and config.data_parallel:
            print(f"[ttt-data-parallel] disabled: {reason}", flush=True)
        return base_model, None, 1

    try:
        import torch

        parallel_model = torch.nn.DataParallel(base_model, device_ids=device_ids, dim=1)
    except Exception as exc:
        if verbose:
            print(
                "[ttt-data-parallel] disabled: "
                f"failed to initialize DataParallel: {type(exc).__name__}: {exc}",
                flush=True,
            )
        return base_model, None, 1

    if verbose:
        print(
            "[ttt-data-parallel] enabled: "
            f"gpu_group={config.gpu_group} device_ids={','.join(map(str, device_ids))} "
            f"micro_batch_size_per_gpu={config.micro_batch_size}",
            flush=True,
        )
    return base_model, parallel_model, len(device_ids)


def _build_fixed_ttt_batch(classifier, X_train, y_train, split: TTTSplit, config: TTTConfig):
    from tabpfn.finetuning.data_util import (
        get_preprocessed_dataset_chunks,
        meta_dataset_collator,
    )

    def fixed_split(X, y, stratify=None):
        return (
            take_rows(X, split.b_indices),
            take_rows(X, split.c_indices),
            take_rows(y, split.b_indices),
            take_rows(y, split.c_indices),
        )

    datasets = get_preprocessed_dataset_chunks(
        calling_instance=classifier,
        X_raw=X_train,
        y_raw=y_train,
        split_fn=fixed_split,
        max_data_size=None,
        model_type="classifier",
        equal_split_size=False,
        data_shuffle_seed=config.random_state,
        preprocessing_random_state=config.random_state,
        shuffle=False,
        force_no_stratify=True,
    )
    if len(datasets) != 1:
        raise RuntimeError(f"Expected one preprocessed TTT batch, got {len(datasets)}")
    return meta_dataset_collator([datasets[0]])


def _compute_ttt_classification_loss_single(classifier, batch):
    import torch
    import torch.nn.functional as F

    logits_qbel = classifier.forward(batch.X_query, return_raw_logits=True)
    if logits_qbel.ndim != 4:
        raise RuntimeError(f"Expected raw logits with shape Q,B,E,L; got {tuple(logits_qbel.shape)}")

    q_size, batch_size, n_estimators, n_logits = logits_qbel.shape
    y_query = batch.y_query.to(classifier.devices_[0]).long()
    if y_query.ndim == 1:
        y_query = y_query.unsqueeze(0)
    if y_query.shape[1] != q_size:
        raise RuntimeError(
            f"Holdout target length mismatch: logits Q={q_size}, targets={tuple(y_query.shape)}"
        )

    logits_blq = logits_qbel.permute(1, 2, 3, 0).reshape(
        batch_size * n_estimators,
        n_logits,
        q_size,
    )
    targets_bq = y_query.repeat(batch_size * n_estimators, 1)
    return F.cross_entropy(logits_blq, targets_bq)


def _module_accepts_kwarg(module: Any, kwarg_name: str) -> bool:
    try:
        from inspect import signature

        return kwarg_name in signature(module.forward).parameters
    except Exception:
        return False


def _first_batch_config(config_entry: Any) -> Any:
    if isinstance(config_entry, (list, tuple)):
        if not config_entry:
            raise RuntimeError("Empty TabPFN ensemble config entry")
        return config_entry[0]
    return config_entry


def _normalise_cat_indices(value: Any) -> list[int]:
    if value is None:
        return []
    return [int(item) for item in value]


def _cat_indices_for_estimators(batch: Any, estimator_indices: list[int]) -> list[list[int]]:
    cat_indices = getattr(batch, "cat_indices", [])
    if not cat_indices:
        return [[] for _ in estimator_indices]

    # meta_dataset_collator produces [dataset_batch][estimator][cat_index].
    if len(cat_indices) == 1 and isinstance(cat_indices[0], list):
        first_dataset = cat_indices[0]
        if not first_dataset or first_dataset[0] is None or isinstance(first_dataset[0], list):
            return [
                _normalise_cat_indices(first_dataset[estimator_idx])
                if estimator_idx < len(first_dataset)
                else []
                for estimator_idx in estimator_indices
            ]

    return [
        _normalise_cat_indices(cat_indices[estimator_idx])
        if estimator_idx < len(cat_indices)
        else []
        for estimator_idx in estimator_indices
    ]


def _apply_classifier_class_permutation(
    logits_qbl: Any,
    config_entry: Any,
    local_batch_idx: int,
    n_classes: int,
):
    import torch

    config = _first_batch_config(config_entry)
    class_permutation = getattr(config, "class_permutation", None)
    estimator_logits = logits_qbl[:, local_batch_idx, :]
    if class_permutation is None:
        return estimator_logits[:, :n_classes]

    permutation = np.asarray(class_permutation, dtype=np.int64)
    if len(permutation) != n_classes:
        use_perm = np.arange(n_classes, dtype=np.int64)
        use_perm[: len(permutation)] = permutation
    else:
        use_perm = permutation

    permutation_tensor = torch.as_tensor(
        use_perm,
        dtype=torch.long,
        device=estimator_logits.device,
    )
    return estimator_logits.index_select(-1, permutation_tensor)


def _compute_ttt_classification_loss_data_parallel(
    classifier,
    batch,
    config: TTTConfig,
    *,
    base_model: Any,
    parallel_model: Any,
    data_parallel_world_size: int,
):
    import torch
    import torch.nn.functional as F
    from contextlib import nullcontext

    device = classifier.devices_[0]
    y_query = batch.y_query.to(device).long()
    if y_query.ndim == 1:
        y_query = y_query.unsqueeze(0)

    n_estimators = len(batch.X_query)
    if n_estimators < 1:
        raise RuntimeError("TTT batch has no estimator views")
    if y_query.shape[0] != 1:
        raise RuntimeError(f"Only batch_size=1 is supported for TTT, got y_query={tuple(y_query.shape)}")

    force_dtype = getattr(getattr(classifier, "executor_", None), "force_inference_dtype", None)
    autocast_enabled = bool(getattr(classifier, "use_autocast_", False))
    device_type = getattr(device, "type", str(device).split(":")[0])
    effective_micro_batch_size = max(1, int(config.micro_batch_size) * int(data_parallel_world_size))
    total_loss = None

    for start_idx in range(0, n_estimators, effective_micro_batch_size):
        end_idx = min(start_idx + effective_micro_batch_size, n_estimators)
        estimator_indices = list(range(start_idx, end_idx))

        compatible_groups: dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], list[int]] = {}
        for estimator_idx in estimator_indices:
            key = (
                tuple(batch.X_context[estimator_idx].shape[1:]),
                tuple(batch.X_query[estimator_idx].shape[1:]),
                tuple(batch.y_context[estimator_idx].shape[1:]),
            )
            compatible_groups.setdefault(key, []).append(estimator_idx)

        for group_indices in compatible_groups.values():
            X_context = torch.cat([batch.X_context[idx] for idx in group_indices], dim=0).to(device)
            X_query = torch.cat([batch.X_query[idx] for idx in group_indices], dim=0).to(device)
            y_context = torch.cat([batch.y_context[idx] for idx in group_indices], dim=0).to(device)
            if force_dtype is not None:
                X_context = X_context.type(force_dtype)
                X_query = X_query.type(force_dtype)
                y_context = y_context.type(force_dtype)

            X_full = torch.cat([X_context, X_query], dim=-2).transpose(0, 1)
            y_context_model = y_context.transpose(0, 1)
            forward_model = (
                parallel_model
                if len(group_indices) >= data_parallel_world_size
                else base_model
            )

            kwargs: dict[str, Any] = {}
            if _module_accepts_kwarg(base_model, "only_return_standard_out"):
                kwargs["only_return_standard_out"] = True
            if _module_accepts_kwarg(base_model, "categorical_inds"):
                kwargs["categorical_inds"] = _cat_indices_for_estimators(batch, group_indices)
            if _module_accepts_kwarg(base_model, "task_type"):
                kwargs["task_type"] = "multiclass"

            autocast_context = (
                torch.autocast(device_type=device_type)
                if autocast_enabled and device_type in {"cuda", "cpu"}
                else nullcontext()
            )
            with autocast_context:
                output = forward_model(X_full, y_context_model, **kwargs)
            if isinstance(output, dict):
                output = output.get("standard", output.get("main"))
            if output is None:
                raise RuntimeError("TabPFN model forward returned no standard output")
            if output.ndim == 2:
                output = output.unsqueeze(1)
            if output.ndim != 3:
                raise RuntimeError(f"Expected model output with shape Q,B,L; got {tuple(output.shape)}")

            logits_by_estimator = []
            for local_idx, estimator_idx in enumerate(group_indices):
                logits_by_estimator.append(
                    _apply_classifier_class_permutation(
                        output,
                        batch.configs[estimator_idx],
                        local_idx,
                        int(classifier.n_classes_),
                    )
                )
            logits_qel = torch.stack(logits_by_estimator, dim=1)
            q_size, estimator_batch_size, n_logits = logits_qel.shape
            if y_query.shape[1] != q_size:
                raise RuntimeError(
                    f"Holdout target length mismatch: logits Q={q_size}, targets={tuple(y_query.shape)}"
                )
            logits_elq = logits_qel.permute(1, 2, 0).reshape(
                estimator_batch_size,
                n_logits,
                q_size,
            )
            targets_eq = y_query.repeat(estimator_batch_size, 1)
            loss = F.cross_entropy(logits_elq, targets_eq)
            scaled_loss = loss * (estimator_batch_size / n_estimators)
            total_loss = scaled_loss if total_loss is None else total_loss + scaled_loss

    if total_loss is None:
        raise RuntimeError("No estimator micro-batches were available for TTT loss")
    return total_loss


def _compute_ttt_classification_loss(
    classifier,
    batch,
    config: TTTConfig,
    *,
    base_model: Any,
    parallel_model: Any | None,
    data_parallel_world_size: int,
):
    if parallel_model is None or data_parallel_world_size < 2:
        return _compute_ttt_classification_loss_single(classifier, batch)
    return _compute_ttt_classification_loss_data_parallel(
        classifier,
        batch,
        config,
        base_model=base_model,
        parallel_model=parallel_model,
        data_parallel_world_size=data_parallel_world_size,
    )


def run_ttt_holdout_update(
    classifier,
    X_train,
    y_train,
    split: TTTSplit,
    config: TTTConfig,
    *,
    dataset_name: str,
    verbose: bool,
) -> TTTUpdateResult:
    if not config.enabled:
        return TTTUpdateResult(False, None, 0, 0.0, reason="TTT disabled")
    if config.steps < 1:
        return TTTUpdateResult(False, None, 0, 0.0, reason="--ttt-steps must be >= 1")
    if len(split.b_indices) == 0 or len(split.c_indices) == 0:
        return TTTUpdateResult(False, None, 0, 0.0, reason="B/C split produced an empty side")

    y_b = np.asarray(y_train)[split.b_indices]
    y_c = np.asarray(y_train)[split.c_indices]
    b_labels = set(pd.Series(y_b).astype(object).tolist())
    c_labels = set(pd.Series(y_c).astype(object).tolist())
    missing_from_context = sorted(str(label) for label in (c_labels - b_labels))
    if missing_from_context:
        return TTTUpdateResult(
            False,
            None,
            0,
            0.0,
            reason="Holdout labels absent from B context: " + ",".join(missing_from_context),
        )

    import torch

    update_started = time.time()
    batch = _build_fixed_ttt_batch(classifier, X_train, y_train, split, config)
    trainable_params = _configure_ttt_trainable_params(classifier, config)
    if not trainable_params:
        return TTTUpdateResult(
            False,
            None,
            0,
            time.time() - update_started,
            reason="No trainable parameters selected for TTT",
        )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    classifier.fit_from_preprocessed(
        batch.X_context,
        batch.y_context,
        batch.cat_indices,
        batch.configs,
    )
    base_model, parallel_model, data_parallel_world_size = _build_ttt_forward_models(
        classifier,
        config,
        verbose=verbose,
    )

    last_loss = None
    for step_idx in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _compute_ttt_classification_loss(
            classifier,
            batch,
            config,
            base_model=base_model,
            parallel_model=parallel_model,
            data_parallel_world_size=data_parallel_world_size,
        )
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip)
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        if verbose:
            print(
                f"[ttt-loss] dataset={dataset_name} step={step_idx} loss={last_loss:.6f}",
                flush=True,
            )

    classifier.model_.eval()
    return TTTUpdateResult(
        applied=True,
        loss=last_loss,
        steps=config.steps,
        update_seconds=time.time() - update_started,
    )


class TabPFNAdapter:
    def __init__(self, args: argparse.Namespace, device: str) -> None:
        self.args = args
        self.device = device
        self.ttt_config = build_ttt_config(args)

    def _classifier_kwargs(
        self,
        categorical_feature_indices: list[int],
        *,
        n_estimators_override: int | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "device": self.device,
            "ignore_pretraining_limits": self.args.ignore_pretraining_limits,
            "n_estimators": (
                int(n_estimators_override)
                if n_estimators_override is not None
                else int(self.args.n_estimators)
            ),
        }
        if categorical_feature_indices:
            kwargs["categorical_features_indices"] = list(categorical_feature_indices)
        return kwargs

    def _make_classifier(
        self,
        categorical_feature_indices: list[int],
        *,
        n_estimators_override: int | None = None,
    ):
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except Exception as exc:
            raise RuntimeError(f"Failed to import local TabPFN: {exc}") from exc

        classifier_kwargs = self._classifier_kwargs(
            categorical_feature_indices,
            n_estimators_override=n_estimators_override,
        )
        if self.args.model_path:
            return TabPFNClassifier(model_path=self.args.model_path, **classifier_kwargs)

        version_map = {
            "v2": ModelVersion.V2,
            "v2.5": ModelVersion.V2_5,
            "v2.6": ModelVersion.V2_6,
        }
        version = version_map[self.args.model_version]
        try:
            return TabPFNClassifier.create_default_for_version(
                version,
                **classifier_kwargs,
            )
        except TypeError:
            classifier = TabPFNClassifier.create_default_for_version(version)
            if hasattr(classifier, "set_params"):
                try:
                    classifier.set_params(
                        device=self.device,
                        ignore_pretraining_limits=self.args.ignore_pretraining_limits,
                        n_estimators=classifier_kwargs["n_estimators"],
                    )
                except Exception:
                    try:
                        classifier.set_params(device=self.device)
                    except Exception:
                        pass
            elif hasattr(classifier, "device"):
                classifier.device = self.device
            if hasattr(classifier, "ignore_pretraining_limits"):
                classifier.ignore_pretraining_limits = self.args.ignore_pretraining_limits
            if hasattr(classifier, "n_estimators"):
                classifier.n_estimators = classifier_kwargs["n_estimators"]
            if categorical_feature_indices and hasattr(
                classifier,
                "categorical_features_indices",
            ):
                classifier.categorical_features_indices = list(categorical_feature_indices)
            return classifier

    def _should_use_many_class(self, n_classes: int) -> bool:
        mode = self.args.many_class
        if mode == "off":
            return False
        if mode == "on":
            return True
        return n_classes > TABPFN_CLASS_LIMIT

    def _wrap_many_class_if_needed(self, classifier, loaded: LoadedDataset):
        if not self._should_use_many_class(loaded.n_classes):
            return classifier
        try:
            ManyClassClassifier = import_many_class_classifier()
        except Exception as exc:
            raise RuntimeError(
                "Official TabPFN many-class inference requires tabpfn-extensions. "
                "Install it with: pip install git+https://github.com/PriorLabs/tabpfn-extensions.git. "
                f"Import failed with {type(exc).__name__}: {exc}"
            ) from exc

        wrapped = ManyClassClassifier(
            estimator=classifier,
            alphabet_size=self.args.many_class_alphabet_size,
            n_estimators=self.args.many_class_n_estimators,
            n_estimators_redundancy=self.args.many_class_redundancy,
            random_state=self.args.random_state,
            verbose=1 if self.args.verbose else 0,
        )
        if loaded.categorical_feature_indices and hasattr(
            wrapped,
            "set_categorical_features",
        ):
            wrapped.set_categorical_features(loaded.categorical_feature_indices)
        if self.args.verbose:
            print(
                f"[many_class] {loaded.dataset_name}: n_classes={loaded.n_classes}, "
                f"alphabet_size={self.args.many_class_alphabet_size}, "
                f"redundancy={self.args.many_class_redundancy}",
                flush=True,
            )
        return wrapped

    def _fit_predict_baseline(
        self,
        loaded: LoadedDataset,
        *,
        ttt_reason: str | None = None,
        ttt_split_strategy: str | None = None,
        n_train_b: int = 0,
        n_holdout_c: int = 0,
    ) -> PredictionResult:
        classifier = self._wrap_many_class_if_needed(
            self._make_classifier(loaded.categorical_feature_indices),
            loaded,
        )
        X_train = loaded.X_train_merged.to_numpy()
        y_train = loaded.y_train_merged
        X_test = loaded.X_test.to_numpy()

        fit_started = time.time()
        classifier.fit(X_train, y_train)
        fit_seconds = time.time() - fit_started

        predict_started = time.time()
        y_pred = classifier.predict(X_test)
        predict_seconds = time.time() - predict_started
        return PredictionResult(
            y_pred=y_pred,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            n_train_a=loaded.n_train_report,
            n_train_b=n_train_b,
            n_holdout_c=n_holdout_c,
            n_test_d=int(len(loaded.y_test)),
            ttt_lr=self.ttt_config.lr if self.ttt_config.enabled else None,
            ttt_split_strategy=ttt_split_strategy,
            ttt_split_reason=ttt_reason,
        )

    def _clone_ttt_classifier_for_full_context(self, classifier, loaded: LoadedDataset):
        from tabpfn import TabPFNClassifier
        from tabpfn.finetuning.train_util import clone_model_for_evaluation

        eval_kwargs = self._classifier_kwargs(loaded.categorical_feature_indices)
        return clone_model_for_evaluation(classifier, eval_kwargs, TabPFNClassifier)

    def _fit_predict_with_ttt(self, loaded: LoadedDataset) -> PredictionResult:
        config = self.ttt_config
        if not config.enabled:
            return self._fit_predict_baseline(loaded)
        if self._should_use_many_class(loaded.n_classes):
            return self._fit_predict_baseline(
                loaded,
                ttt_reason="TTT skipped because ManyClassClassifier is required",
            )

        X_train = loaded.X_train_merged.to_numpy()
        y_train = loaded.y_train_merged
        X_test = loaded.X_test.to_numpy()
        split = split_ttt_holdout(y_train, config)
        n_train_b = int(len(split.b_indices))
        n_holdout_c = int(len(split.c_indices))

        fit_started = time.time()
        try:
            if n_train_b == 0 or n_holdout_c == 0:
                return self._fit_predict_baseline(
                    loaded,
                    ttt_reason=f"{split.reason} | B/C split produced an empty side; fallback baseline",
                    ttt_split_strategy=split.strategy,
                    n_train_b=n_train_b,
                    n_holdout_c=n_holdout_c,
                )

            classifier = self._make_classifier(
                loaded.categorical_feature_indices,
                n_estimators_override=config.n_estimators_finetune,
            )
            ttt_result = run_ttt_holdout_update(
                classifier,
                X_train,
                y_train,
                split,
                config,
                dataset_name=loaded.dataset_name,
                verbose=bool(self.args.verbose),
            )
            ttt_reason = split.reason
            if ttt_result.reason:
                ttt_reason = f"{ttt_reason} | {ttt_result.reason}"

            if not ttt_result.applied:
                baseline = self._fit_predict_baseline(
                    loaded,
                    ttt_reason=f"{ttt_reason}; fallback baseline",
                    ttt_split_strategy=split.strategy,
                    n_train_b=n_train_b,
                    n_holdout_c=n_holdout_c,
                )
                baseline.ttt_update_seconds = float(ttt_result.update_seconds)
                return baseline

            inference_classifier = self._clone_ttt_classifier_for_full_context(classifier, loaded)
            inference_classifier.fit(X_train, y_train)
            fit_seconds = time.time() - fit_started

            predict_started = time.time()
            y_pred = inference_classifier.predict(X_test)
            predict_seconds = time.time() - predict_started
            return PredictionResult(
                y_pred=y_pred,
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
                n_train_a=loaded.n_train_report,
                n_train_b=n_train_b,
                n_holdout_c=n_holdout_c,
                n_test_d=int(len(loaded.y_test)),
                ttt_loss=ttt_result.loss,
                ttt_steps=ttt_result.steps,
                ttt_lr=config.lr,
                ttt_applied=True,
                ttt_update_seconds=float(ttt_result.update_seconds),
                ttt_split_strategy=split.strategy,
                ttt_split_reason=ttt_reason,
            )
        except Exception as exc:
            return self._fit_predict_baseline(
                loaded,
                ttt_reason=(
                    f"{split.reason} | TTT failed with {type(exc).__name__}: {exc}; "
                    "fallback baseline"
                ),
                ttt_split_strategy=split.strategy,
                n_train_b=n_train_b,
                n_holdout_c=n_holdout_c,
            )

    def fit_predict(self, loaded: LoadedDataset) -> PredictionResult:
        return self._fit_predict_with_ttt(loaded)

    def fit_predict_with_forced_ignore_limits(
        self,
        loaded: LoadedDataset,
    ) -> PredictionResult:
        original_flag = self.args.ignore_pretraining_limits
        self.args.ignore_pretraining_limits = True
        try:
            return self.fit_predict(loaded)
        finally:
            self.args.ignore_pretraining_limits = original_flag


def resolve_script_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    script_path = SCRIPT_DIR / path
    repo_path = SCRIPT_DIR.parent.parent / path

    first_part = path.parts[0] if path.parts else ""
    if first_part and first_part not in {".", ".."}:
        if (Path.cwd() / first_part).exists():
            return cwd_path.resolve()
        if (SCRIPT_DIR / first_part).exists():
            return script_path.resolve()
        if (SCRIPT_DIR.parent.parent / first_part).exists():
            return repo_path.resolve()

    if cwd_path.exists():
        return cwd_path.resolve()
    if script_path.exists():
        return script_path.resolve()
    if repo_path.exists():
        return repo_path.resolve()
    return script_path.resolve()


def resolve_model_path(
    value: str | Path | None,
    model_version: str | None = None,
) -> str | None:
    if value is None:
        raw_value = ""
    else:
        raw_value = str(value).strip()

    if not raw_value or raw_value.lower() in {"auto", "none", "null"}:
        if model_version:
            preferred_file = GATED_CLASSIFIER_CACHE_FILES.get(model_version)
            if preferred_file:
                cache_dir = Path(
                    os.environ.get("TABPFN_MODEL_CACHE_DIR", str(Path.home() / ".cache" / "tabpfn"))
                ).expanduser()
                for candidate in (
                    Path.cwd() / preferred_file,
                    SCRIPT_DIR / preferred_file,
                    SCRIPT_DIR.parent.parent / preferred_file,
                    cache_dir / preferred_file,
                ):
                    if candidate.exists():
                        return candidate.resolve().as_posix()
        return None

    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path.as_posix()

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve().as_posix()

    script_path = SCRIPT_DIR / path
    if script_path.exists():
        return script_path.resolve().as_posix()

    repo_path = SCRIPT_DIR.parent.parent / path
    if repo_path.exists():
        return repo_path.resolve().as_posix()

    return script_path.resolve().as_posix()


def load_dataset_info(dataset_dir: Path) -> dict | None:
    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        return None
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_dataset_dirs(data_root: Path) -> list[Path]:
    if (data_root / "info.json").exists():
        return [data_root]
    return [path for path in sorted(data_root.iterdir()) if path.is_dir()]


def normalize_categorical_series(series: pd.Series) -> pd.Series:
    string_series = series.astype("string")
    string_series = string_series.fillna(CATEGORICAL_MISSING_TOKEN)
    return string_series.astype(str)


def make_feature_frame(values, *, kind: str, prefix: str) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        df = values.copy()
    else:
        arr = np.asarray(values)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        df = pd.DataFrame(arr)

    if kind == "numeric":
        df = df.apply(pd.to_numeric, errors="coerce")
    elif kind == "categorical":
        df = df.apply(normalize_categorical_series)
    else:
        raise ValueError(f"Unsupported feature kind: {kind}")

    df.columns = [f"{prefix}_{i}" for i in range(df.shape[1])]
    return df


def make_target_array(values) -> np.ndarray:
    y = np.asarray(values)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    elif y.ndim > 1 and y.shape[0] == 1:
        y = y.squeeze(0)
    return pd.Series(y).values


def load_array(file_path: Path) -> np.ndarray:
    suffix = file_path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        try:
            arr = np.load(file_path, allow_pickle=False)
        except ValueError:
            arr = np.load(file_path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        return np.asarray(arr)
    raise ValueError(f"Unsupported split file type: {file_path}")


def find_by_suffix(files: list[Path], suffix: str) -> Path | None:
    lower_suffix = suffix.lower()
    for file_path in files:
        if file_path.name.lower().endswith(lower_suffix):
            return file_path
    return None


def find_split_files(dataset_dir: Path):
    files = [path for path in dataset_dir.iterdir() if path.is_file()]
    n_train = find_by_suffix(files, "n_train.npy")
    c_train = find_by_suffix(files, "c_train.npy")
    y_train = find_by_suffix(files, "y_train.npy")
    n_val = find_by_suffix(files, "n_val.npy")
    c_val = find_by_suffix(files, "c_val.npy")
    y_val = find_by_suffix(files, "y_val.npy")
    n_test = find_by_suffix(files, "n_test.npy")
    c_test = find_by_suffix(files, "c_test.npy")
    y_test = find_by_suffix(files, "y_test.npy")

    if y_train is None or y_test is None:
        raise FileNotFoundError("Missing y_train.npy or y_test.npy")
    if n_train is None and c_train is None:
        raise FileNotFoundError("Missing both N_train.npy and C_train.npy")
    if n_test is None and c_test is None:
        raise FileNotFoundError("Missing both N_test.npy and C_test.npy")

    train_split = (n_train, c_train, y_train)
    val_split = (n_val, c_val, y_val) if y_val is not None and (n_val is not None or c_val is not None) else None
    test_split = (n_test, c_test, y_test)
    return train_split, val_split, test_split


def stable_feature_prefix(context: str, fallback: str) -> str:
    stem = Path(context or fallback).stem
    for suffix in ("-train", "-test", "-val", "-single"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or fallback


def load_split(
    num_path: Path | None,
    cat_path: Path | None,
    y_path: Path,
    *,
    context: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    features: list[pd.DataFrame] = []
    feature_prefix = stable_feature_prefix(context, y_path.stem)

    if num_path is not None:
        x_num = load_array(num_path)
        features.append(make_feature_frame(x_num, kind="numeric", prefix=f"{feature_prefix}_n"))
    if cat_path is not None:
        x_cat = load_array(cat_path)
        features.append(make_feature_frame(x_cat, kind="categorical", prefix=f"{feature_prefix}_c"))

    if not features:
        raise ValueError("No feature files found for split")

    n_samples = features[0].shape[0]
    for idx, feature_df in enumerate(features):
        if feature_df.shape[0] != n_samples:
            raise ValueError(
                f"Inconsistent number of rows across feature blocks: block {idx} has "
                f"{feature_df.shape[0]} rows but expected {n_samples}"
            )

    X = features[0] if len(features) == 1 else pd.concat(features, axis=1)
    y = make_target_array(load_array(y_path))
    if len(X) != len(y):
        raise ValueError(f"Feature/target row mismatch: X has {len(X)} rows while y has {len(y)}")
    return X, y


def categorical_indices_from_split(train_split) -> list[int]:
    num_path, cat_path, _ = train_split
    n_num = 0
    if num_path is not None:
        arr = load_array(Path(num_path))
        n_num = int(arr.reshape(arr.shape[0], -1).shape[1]) if arr.ndim > 1 else 1
    if cat_path is None:
        return []
    arr = load_array(Path(cat_path))
    n_cat = int(arr.reshape(arr.shape[0], -1).shape[1]) if arr.ndim > 1 else 1
    return list(range(n_num, n_num + n_cat))


def load_classification_dataset(dataset_dir: Path) -> LoadedDataset:
    info = load_dataset_info(dataset_dir)
    task_type = str(info.get("task_type", "")).lower() if info else None
    if task_type not in CLASSIFICATION_TASKS:
        raise SkipDataset(f"Skipped due to task_type={task_type!r}")

    train_split, val_split, test_split = find_split_files(dataset_dir)
    categorical_feature_indices = categorical_indices_from_split(train_split)
    X_train, y_train = load_split(
        train_split[0],
        train_split[1],
        train_split[2],
        context=f"{dataset_dir.name}-train",
    )

    X_val = None
    y_val = None
    if val_split is not None:
        X_val, y_val = load_split(
            val_split[0],
            val_split[1],
            val_split[2],
            context=f"{dataset_dir.name}-val",
        )

    X_test, y_test = load_split(
        test_split[0],
        test_split[1],
        test_split[2],
        context=f"{dataset_dir.name}-test",
    )
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature count mismatch: train has {X_train.shape[1]}, test has {X_test.shape[1]}"
        )
    if X_val is not None and X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            f"Feature count mismatch: train has {X_train.shape[1]}, val has {X_val.shape[1]}"
        )

    return LoadedDataset(
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir,
        task_type=task_type,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        categorical_feature_indices=categorical_feature_indices,
    )


def empty_row_for_dataset(
    dataset_dir: Path,
    status: str,
    error: str,
    *,
    task_type: Optional[str] = None,
) -> ResultRow:
    return ResultRow(
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir.as_posix(),
        task_type=task_type,
        n_train=0,
        n_val=0,
        n_test=0,
        n_features=0,
        n_classes=0,
        accuracy=None,
        fit_seconds=0.0,
        predict_seconds=0.0,
        status=status,
        error=error,
    )


def evaluate_one_dataset(adapter: TabPFNAdapter, dataset_dir: Path) -> ResultRow:
    task_type: Optional[str] = None
    try:
        loaded = load_classification_dataset(dataset_dir)
        task_type = loaded.task_type
        try:
            result = adapter.fit_predict(loaded)
        except Exception as exc:
            if is_pretraining_limit_error(exc):
                result = adapter.fit_predict_with_forced_ignore_limits(loaded)
            else:
                raise
        y_pred = np.asarray(result.y_pred)
        y_test = np.asarray(loaded.y_test)
        if len(y_pred) != len(y_test):
            raise ValueError(f"Prediction length mismatch: got {len(y_pred)}, expected {len(y_test)}")
        accuracy = float(np.mean(y_pred == y_test))
        return ResultRow(
            dataset_name=loaded.dataset_name,
            dataset_dir=loaded.dataset_dir.as_posix(),
            task_type=loaded.task_type,
            n_train=loaded.n_train_report,
            n_val=loaded.n_val,
            n_test=int(len(y_test)),
            n_features=int(loaded.X_train.shape[1]),
            n_classes=loaded.n_classes,
            accuracy=accuracy,
            fit_seconds=float(result.fit_seconds),
            predict_seconds=float(result.predict_seconds),
            status="ok",
            error=None,
            n_train_a=result.n_train_a,
            n_train_b=result.n_train_b,
            n_holdout_c=result.n_holdout_c,
            n_test_d=result.n_test_d,
            ttt_loss=result.ttt_loss,
            ttt_steps=result.ttt_steps,
            ttt_lr=result.ttt_lr,
            ttt_applied=result.ttt_applied,
            ttt_update_seconds=result.ttt_update_seconds,
            ttt_split_strategy=result.ttt_split_strategy,
            ttt_split_reason=result.ttt_split_reason,
        )
    except SkipDataset as exc:
        return empty_row_for_dataset(dataset_dir, "skip", str(exc), task_type=task_type)
    except Exception as exc:
        return empty_row_for_dataset(
            dataset_dir,
            "fail",
            f"{type(exc).__name__}: {exc}",
            task_type=task_type,
        )


def rows_to_frame(rows: Iterable[ResultRow]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(row) for row in rows])
    if frame.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    for column in RESULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[RESULT_COLUMNS]


def ensure_result_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    normalized = frame.copy()
    for column in RESULT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    return normalized[RESULT_COLUMNS]


def is_oom_error_message(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in OOM_ERROR_MARKERS)


def is_pretraining_limit_error(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return (
        "ignore_pretraining_limits" in text
        or "officially supported by tabpfn" in text
        or "pre-training range" in text
    )


def load_failed_dataset_names_from_results_csv(
    results_csv: Path,
    *,
    include_oom_failures: bool,
) -> list[str]:
    if not results_csv.exists():
        raise FileNotFoundError(f"Retry results CSV does not exist: {results_csv}")

    frame = pd.read_csv(results_csv)
    required_columns = {"dataset_name", "status"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Retry results CSV is missing required columns: {', '.join(missing_columns)}"
        )

    failed_df = frame[frame["status"].astype(str).str.lower() == "fail"].copy()
    if not include_oom_failures and "error" in failed_df.columns:
        failed_df = failed_df[~failed_df["error"].map(is_oom_error_message)]

    dataset_names: list[str] = []
    seen_names: set[str] = set()
    for _, row in failed_df.iterrows():
        raw_name = str(row.get("dataset_name", "") or "").strip()
        if not raw_name and "dataset_dir" in failed_df.columns:
            raw_name = Path(str(row.get("dataset_dir", "") or "")).name.strip()
        if not raw_name or raw_name in seen_names:
            continue
        dataset_names.append(raw_name)
        seen_names.add(raw_name)
    return dataset_names


def filter_dataset_dirs_by_name(
    dataset_dirs: list[Path],
    dataset_names: Iterable[str],
) -> tuple[list[Path], list[str]]:
    wanted_names = {str(name).strip() for name in dataset_names if str(name).strip()}
    filtered = [path for path in dataset_dirs if path.name in wanted_names]
    found_names = {path.name for path in filtered}
    missing_names = [name for name in dataset_names if name not in found_names]
    return filtered, missing_names


def merge_result_frames(base_df: pd.DataFrame, updated_df: pd.DataFrame) -> pd.DataFrame:
    base_df = ensure_result_columns(base_df)
    updated_df = ensure_result_columns(updated_df)
    if base_df.empty:
        return updated_df
    if updated_df.empty:
        return base_df

    base_df = base_df.copy()
    updated_df = updated_df.copy()
    order_map = {
        str(dataset_name): idx
        for idx, dataset_name in enumerate(base_df["dataset_name"].astype(str).tolist())
    }
    replacement_names = set(updated_df["dataset_name"].astype(str).tolist())
    base_keep = base_df[~base_df["dataset_name"].astype(str).isin(replacement_names)].copy()

    base_keep["_merge_order"] = base_keep["dataset_name"].astype(str).map(order_map)
    appended_start = len(order_map)
    updated_df["_merge_order"] = [
        order_map.get(str(dataset_name), appended_start + idx)
        for idx, dataset_name in enumerate(updated_df["dataset_name"].tolist())
    ]

    merged = pd.concat([base_keep, updated_df], ignore_index=True)
    merged = merged.sort_values("_merge_order", kind="stable").drop(columns="_merge_order")
    merged = merged.reset_index(drop=True)
    return merged[RESULT_COLUMNS]


def write_summary(summary_path: Path, result_df: pd.DataFrame, dataset_dirs: list[Path], wall_seconds: float) -> None:
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()
    skipped_df = result_df[result_df["status"] == "skip"].copy() if len(result_df) else pd.DataFrame()
    lines = [
        f"discovered_datasets: {len(dataset_dirs)}",
        f"processed_datasets: {len(result_df)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        f"skipped_count: {len(skipped_df)}",
        f"avg_accuracy_ok: {ok_df['accuracy'].mean():.6f}" if len(ok_df) else "avg_accuracy_ok: (none)",
        (
            f"ttt_applied_ok_count: {int(ok_df['ttt_applied'].astype(bool).sum())}"
            if len(ok_df) and "ttt_applied" in ok_df.columns
            else "ttt_applied_ok_count: 0"
        ),
        (
            f"avg_ttt_loss_ok: {pd.to_numeric(ok_df['ttt_loss'], errors='coerce').mean():.6f}"
            if len(ok_df) and "ttt_loss" in ok_df.columns and pd.to_numeric(ok_df["ttt_loss"], errors="coerce").notna().any()
            else "avg_ttt_loss_ok: (none)"
        ),
        f"wall_seconds: {wall_seconds:.3f}",
        "failed_datasets: "
        + (", ".join(failed_df["dataset_name"].astype(str).tolist()) if len(failed_df) else "(none)"),
        "skipped_datasets: "
        + (", ".join(skipped_df["dataset_name"].astype(str).tolist()) if len(skipped_df) else "(none)"),
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_gpu_id_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def normalize_gpu_group(value: int | str) -> str:
    if isinstance(value, int):
        return str(value)
    gpu_group = str(value).strip()
    gpu_ids = parse_gpu_id_list(gpu_group)
    if not gpu_ids:
        raise ValueError(f"GPU group must contain at least one GPU id: {value!r}")
    return ",".join(str(gpu_id) for gpu_id in gpu_ids)


def first_gpu_id_from_group(value: int | str) -> int:
    return parse_gpu_id_list(normalize_gpu_group(value))[0]


def parse_gpu_group_list(value: str) -> list[str]:
    gpu_groups: list[str] = []
    for raw_group in str(value).split(";"):
        raw_group = raw_group.strip()
        if not raw_group:
            continue
        gpu_groups.append(normalize_gpu_group(raw_group))
    return gpu_groups


def detect_env_gpu_ids() -> list[int]:
    for env_name in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        raw_value = os.environ.get(env_name, "").strip()
        if not raw_value:
            continue
        try:
            gpu_ids = parse_gpu_id_list(raw_value)
        except ValueError:
            gpu_ids = []
        if gpu_ids:
            return gpu_ids
    return []


def detect_nvidia_gpu_ids() -> list[int]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    gpu_ids: list[int] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            gpu_ids.append(int(line.split(",")[0].strip()))
        except ValueError:
            continue
    return gpu_ids


def detect_torch_gpu_ids() -> list[int]:
    try:
        import torch

        if torch.cuda.is_available():
            return list(range(int(torch.cuda.device_count())))
    except Exception:
        pass
    return []


def default_tabpfn_cache_dir() -> Path:
    cache_dir = os.environ.get("TABPFN_MODEL_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir).expanduser()
    return Path.home() / ".cache" / "tabpfn"


def validate_model_access(args: argparse.Namespace) -> None:
    if args.model_path:
        model_path = Path(args.model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        return

    cache_file_name = GATED_CLASSIFIER_CACHE_FILES.get(args.model_version)
    if cache_file_name is None:
        return

    cache_path = default_tabpfn_cache_dir() / cache_file_name
    if cache_path.exists() or os.environ.get("TABPFN_TOKEN"):
        return

    raise RuntimeError(
        f"TabPFN {args.model_version} classifier weights are not cached at {cache_path} "
        "and TABPFN_TOKEN is not set. Use --model-version v2.6 if that cache exists, "
        "pass --model-path to a local checkpoint, or export TABPFN_TOKEN after accepting "
        "the PriorLabs license."
    )


def resolve_workers_and_gpu_ids(args: argparse.Namespace) -> tuple[int, list[int]]:
    if args.gpus is None or str(args.gpus).strip().lower() == "auto":
        gpu_ids = detect_env_gpu_ids() or detect_torch_gpu_ids() or detect_nvidia_gpu_ids()
        if not gpu_ids:
            raise RuntimeError("No visible GPU detected. Pass --gpus explicitly if needed.")
    else:
        gpu_ids = parse_gpu_id_list(str(args.gpus))
        if not gpu_ids:
            raise ValueError("--gpus must contain at least one GPU id or use 'auto'")

    workers = len(gpu_ids) if args.workers is None else int(args.workers)
    if workers <= 0:
        raise ValueError("--workers must be positive")
    if len(gpu_ids) != workers:
        raise ValueError(
            f"--gpus must contain exactly --workers ids; got {len(gpu_ids)} ids for {workers} workers"
        )
    return workers, gpu_ids


def resolve_gpu_assignments(args: argparse.Namespace) -> tuple[int, list[int], list[str]]:
    if args.gpu_groups:
        if args.gpus is not None:
            raise ValueError("Use either --gpu-groups or --gpus, not both")
        gpu_groups = parse_gpu_group_list(args.gpu_groups)
        if not gpu_groups:
            raise ValueError("--gpu-groups must contain at least one group")
        workers = len(gpu_groups) if args.workers is None else int(args.workers)
        if workers <= 0:
            raise ValueError("--workers must be positive")
        if len(gpu_groups) != workers:
            raise ValueError(
                f"--gpu-groups must contain exactly --workers groups; "
                f"got {len(gpu_groups)} groups for {workers} workers"
            )
        gpu_ids = [first_gpu_id_from_group(gpu_group) for gpu_group in gpu_groups]
        return workers, gpu_ids, gpu_groups

    workers, gpu_ids = resolve_workers_and_gpu_ids(args)
    return workers, gpu_ids, [str(gpu_id) for gpu_id in gpu_ids]


def apply_worker_environment_updates(gpu_group: int | str) -> str:
    gpu_group_str = normalize_gpu_group(gpu_group)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group_str
    os.environ["ROCR_VISIBLE_DEVICES"] = gpu_group_str
    os.environ["HIP_VISIBLE_DEVICES"] = gpu_group_str
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    return "cuda:0"


def worker_main(
    worker_id: int,
    gpu_id: int,
    gpu_group: str,
    dataset_dirs: list[str],
    ready_queue,
    start_event,
    worker_csv: str,
    args_dict: dict[str, Any],
) -> None:
    rows: list[ResultRow] = []
    try:
        device = apply_worker_environment_updates(gpu_group)
        args = argparse.Namespace(**args_dict)
        args.gpu_group = gpu_group

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU backend is not available in this worker after binding "
                f"gpu_group={gpu_group}. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )

        adapter = TabPFNAdapter(args, device=device)
        ready_queue.put(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "gpu_group": gpu_group,
                "status": "ready",
                "assigned_count": len(dataset_dirs),
            }
        )
        start_event.wait()

        worker_label = f"worker {worker_id} | gpu {gpu_group}"
        for dataset_dir in dataset_dirs:
            row = evaluate_one_dataset(adapter, Path(dataset_dir))
            rows.append(row)
            if args.verbose:
                if row.status == "ok":
                    print(
                        f"[{worker_label}] [ok] "
                        f"{row.dataset_name} accuracy={row.accuracy:.6f} "
                        f"ttt_applied={row.ttt_applied} "
                        f"ttt_loss={row.ttt_loss} "
                        f"ttt_reason={row.ttt_split_reason}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{worker_label}] [{row.status}] "
                        f"{row.dataset_name} error={row.error}",
                        flush=True,
                    )
    except Exception:
        try:
            ready_queue.put(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "gpu_group": gpu_group,
                    "status": "crash",
                    "error": traceback.format_exc(),
                }
            )
        except Exception:
            pass
        rows.append(
            ResultRow(
                dataset_name=f"__WORKER_CRASH__{worker_id}",
                dataset_dir="__worker__",
                task_type=None,
                n_train=0,
                n_val=0,
                n_test=0,
                n_features=0,
                n_classes=0,
                accuracy=None,
                fit_seconds=0.0,
                predict_seconds=0.0,
                status="fail",
                error=traceback.format_exc(),
            )
        )
    rows_to_frame(rows).to_csv(worker_csv, index=False)


def run_benchmark(args: argparse.Namespace) -> None:
    data_root = resolve_script_path(args.data_root)
    out_dir = resolve_script_path(args.out_dir)
    args.model_path = resolve_model_path(args.model_path, args.model_version)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dataset_dirs = find_dataset_dirs(data_root)
    dataset_dirs = list(all_dataset_dirs)
    retry_results_csv: Path | None = None
    merge_results_csv: Path | None = None

    if args.retry_failed_datasets_only:
        retry_results_csv = resolve_script_path(args.reference_results_csv)
        failed_dataset_names = load_failed_dataset_names_from_results_csv(
            retry_results_csv,
            include_oom_failures=args.retry_include_oom_failures,
        )
        dataset_dirs, missing_dataset_names = filter_dataset_dirs_by_name(
            dataset_dirs,
            failed_dataset_names,
        )
        if missing_dataset_names:
            print(
                "warning: retry CSV referenced datasets missing under data_root: "
                + ", ".join(missing_dataset_names),
                file=sys.stderr,
                flush=True,
            )
        if args.verbose:
            print(
                f"retry_failed_datasets_only: selected {len(dataset_dirs)} datasets "
                f"from {retry_results_csv}",
                flush=True,
            )

    if args.merge_results_from_csv:
        merge_results_csv = resolve_script_path(args.merge_results_from_csv)
    elif retry_results_csv is not None:
        merge_results_csv = retry_results_csv

    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: args.max_datasets]
    if not dataset_dirs:
        if retry_results_csv is not None:
            raise FileNotFoundError(
                f"No retry target datasets found under {data_root} for {retry_results_csv}"
            )
        raise FileNotFoundError(f"No dataset directories found under {data_root}")

    args.data_root = str(data_root)
    args.out_dir = str(out_dir)
    args.workers, gpu_ids, gpu_groups = resolve_gpu_assignments(args)
    if args.verbose:
        print(f"resolved_workers: {args.workers}", flush=True)
        print(f"resolved_gpus: {','.join(str(gpu_id) for gpu_id in gpu_ids)}", flush=True)
        print(f"resolved_gpu_groups: {';'.join(gpu_groups)}", flush=True)
        print(
            f"ignore_pretraining_limits: {args.ignore_pretraining_limits}",
            flush=True,
        )
        if merge_results_csv is not None:
            print(f"merge_results_from_csv: {merge_results_csv}", flush=True)

    validate_model_access(args)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    started = time.time()
    worker_csvs: list[Path] = []
    processes: list[mp.Process] = []
    ready_queue: mp.Queue = mp.Queue()
    start_event: mp.Event = mp.Event()
    args_dict = vars(args).copy()

    for worker_id in range(args.workers):
        assigned = [str(path.resolve()) for path in dataset_dirs[worker_id :: args.workers]]
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        worker_csvs.append(worker_csv)
        proc = mp.Process(
            target=worker_main,
            args=(
                worker_id,
                gpu_ids[worker_id],
                gpu_groups[worker_id],
                assigned,
                ready_queue,
                start_event,
                str(worker_csv),
                args_dict,
            ),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    try:
        ready_workers: set[int] = set()
        while len(ready_workers) < args.workers:
            try:
                message = ready_queue.get(timeout=10)
            except Exception:
                dead_workers = [
                    str(idx)
                    for idx, proc in enumerate(processes)
                    if not proc.is_alive() and idx not in ready_workers
                ]
                if dead_workers:
                    raise RuntimeError(
                        "Some workers exited before initialization completed: "
                        + ", ".join(dead_workers)
                    )
                continue

            if message.get("status") == "ready":
                ready_workers.add(int(message["worker_id"]))
                if args.verbose:
                    print(
                        f"[worker {message['worker_id']} | gpu "
                        f"{message.get('gpu_group', message['gpu_id'])}] "
                        f"ready assigned={message.get('assigned_count', '?')}",
                        flush=True,
                    )
                continue

            if message.get("status") == "crash":
                raise RuntimeError(
                    f"Worker {message['worker_id']} on gpu "
                    f"{message.get('gpu_group', message['gpu_id'])} crashed "
                    f"during initialization:\n{message.get('error', '(no traceback)')}"
                )

        start_event.set()

        for proc in processes:
            proc.join()
    except Exception:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        for proc in processes:
            proc.join(timeout=5)
        raise

    frames = [pd.read_csv(path) for path in worker_csvs if path.exists()]
    result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RESULT_COLUMNS)
    result_df = ensure_result_columns(result_df)
    summary_dataset_dirs = dataset_dirs
    if merge_results_csv is not None:
        base_df = ensure_result_columns(pd.read_csv(merge_results_csv))
        result_df = merge_result_frames(base_df, result_df)
        summary_dataset_dirs = all_dataset_dirs
    all_csv = out_dir / "all_classification_results.csv"
    summary_txt = out_dir / "summary.txt"
    result_df.to_csv(all_csv, index=False)
    write_summary(summary_txt, result_df, summary_dataset_dirs, time.time() - started)
    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {summary_txt}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TabPFN holdout-TTT on data178 classification datasets."
    )
    parser.add_argument("--data-root", default="../../data178")
    parser.add_argument("--out-dir", default="pfn_results/ensemble1_pfnv2.5_step8_lr5e-6")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--gpus",
        default=None,
        help=(
            "Comma-separated physical GPU ids, or 'auto' to use detected GPUs. "
            "Defaults to auto when --gpu-groups is not set."
        ),
    )
    parser.add_argument(
        "--gpu-groups",
        default="1,2,3",
        help=(
            "Semicolon-separated worker GPU groups, e.g. '0,1' for one worker "
            "using two GPUs or '0,1;2,3' for two workers. Mutually exclusive "
            "with --gpus."
        ),
    )
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=8,
        help="Final TabPFN ensemble size used for baseline and post-TTT inference.",
    )
    parser.add_argument(
        "--model-version",
        choices=["v2", "v2.5", "v2.6"],
        default=DEFAULT_MODEL_VERSION,
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path to a local TabPFN checkpoint. Relative paths are resolved from the "
            "current working directory if they exist, otherwise from this script's "
            "directory. If omitted, the script first tries a version-matched local "
            "checkpoint filename, then falls back to the --model-version cache/token flow. "
            "Use 'auto' or 'none' to skip local checkpoint discovery explicitly."
        ),
    )
    parser.add_argument(
        "--many-class",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use the official tabpfn-extensions ManyClassClassifier for >10 classes.",
    )
    parser.add_argument(
        "--many-class-alphabet-size",
        type=int,
        default=TABPFN_CLASS_LIMIT,
    )
    parser.add_argument("--many-class-redundancy", type=int, default=4)
    parser.add_argument("--many-class-n-estimators", type=int, default=None)
    parser.add_argument(
        "--retry-failed-datasets-only",
        action="store_true",
        help=(
            "When enabled, only rerun datasets whose rows have status=fail in "
            "--reference-results-csv. Disabled by default so the script can run "
            "full benchmarks for any model version."
        ),
    )
    parser.add_argument(
        "--reference-results-csv",
        default="v2.5_results/all_classification_results.csv",
        help=(
            "Historical results CSV used by --retry-failed-datasets-only to pick "
            "failed datasets. Default points to the local v2.5 results file."
        ),
    )
    parser.add_argument(
        "--retry-include-oom-failures",
        action="store_true",
        help="When retrying previous failures, also rerun failures whose error looks like OOM.",
    )
    parser.add_argument(
        "--merge-results-from-csv",
        default=None,
        help=(
            "Merge the current run into this existing results CSV before saving "
            "all_classification_results.csv. Defaults to --reference-results-csv "
            "when --retry-failed-datasets-only is enabled."
        ),
    )
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--ignore-pretraining-limits",
        dest="ignore_pretraining_limits",
        action="store_true",
        help=(
            "Ignore TabPFN pretraining limits such as large sample/feature checks. "
            "Enabled by default."
        ),
    )
    limit_group.add_argument(
        "--enforce-pretraining-limits",
        dest="ignore_pretraining_limits",
        action="store_false",
        help="Enforce TabPFN pretraining limits and fail on those validation checks.",
    )
    ttt_group = parser.add_mutually_exclusive_group()
    ttt_group.add_argument(
        "--ttt-holdout",
        dest="ttt_holdout",
        action="store_true",
        help="Enable labeled B/C holdout TTT before final test prediction.",
    )
    ttt_group.add_argument(
        "--no-ttt-holdout",
        dest="ttt_holdout",
        action="store_false",
        help="Disable TTT and run the baseline TabPFN fit/predict path.",
    )
    parser.add_argument("--ttt-steps", type=int, default=8)
    parser.add_argument("--ttt-lr", type=float, default=5e-6)
    parser.add_argument("--ttt-weight-decay", type=float, default=0.0)
    parser.add_argument("--ttt-grad-clip", type=float, default=1.0)
    parser.add_argument("--ttt-train-fraction", type=float, default=0.75)
    parser.add_argument(
        "--ttt-n-estimators-finetune",
        type=int,
        default=None,
        help=(
            "Number of ensemble views used to compute the TTT holdout loss. "
            "Defaults to --n-estimators so the TTT update uses the same ensemble "
            "size as final inference. Use 1 with --no-ttt-data-parallel for the "
            "legacy single-view TTT path."
        ),
    )
    parser.add_argument(
        "--ttt-micro-batch-size",
        type=int,
        default=1,
        help=(
            "Number of estimator views processed per GPU during TTT. When "
            "--ttt-data-parallel is active, the effective estimator batch is "
            "this value times the number of GPUs in the worker group."
        ),
    )
    dp_group = parser.add_mutually_exclusive_group()
    dp_group.add_argument(
        "--ttt-data-parallel",
        dest="ttt_data_parallel",
        action="store_true",
        help="Enable intra-worker multi-GPU DataParallel for the TTT update loss.",
    )
    dp_group.add_argument(
        "--no-ttt-data-parallel",
        dest="ttt_data_parallel",
        action="store_false",
        help="Disable intra-worker multi-GPU DataParallel for the TTT update loss.",
    )
    parser.add_argument(
        "--ttt-train-input-encoders",
        action="store_true",
        help=(
            "Also train TabPFN input encoders during TTT. By default only the "
            "transformer/decoder side is trainable."
        ),
    )
    parser.set_defaults(ignore_pretraining_limits=True, ttt_holdout=True, ttt_data_parallel=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
