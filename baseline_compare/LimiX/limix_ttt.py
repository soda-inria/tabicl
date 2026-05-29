#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from itertools import chain, repeat
from pathlib import Path
from typing import Any, Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd

import benchmark_infer as base


@dataclass
class TTTConfig:
    enabled: bool = True
    lr: float = 5e-6
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    epochs: int = 3
    max_chunk_size: int = 20_000
    min_chunk_size: int = 50
    query_ratio: float = 0.2
    early_stopping: bool = True
    patience: int = 8
    min_delta: float = 1e-4
    eval_metric: str = "accuracy"
    validation_fraction: float = 0.1
    validation_n_estimators: int = 1
    n_estimators_finetune: int = 0
    trainable_scope: str = "all"


@dataclass
class TTTUpdateResult:
    applied: bool
    loss: Optional[float]
    steps: int
    update_seconds: float
    reason: Optional[str] = None
    epochs: int = 0
    chunks_per_epoch: int = 0
    val_eval_metric: Optional[str] = None
    val_baseline_metric: Optional[float] = None
    val_best_metric: Optional[float] = None
    val_baseline_accuracy: Optional[float] = None
    val_best_accuracy: Optional[float] = None
    best_epoch: int = 0
    stopped_early: bool = False


@dataclass
class PredictionResult:
    y_pred: Any
    y_proba: Any
    classes: Any
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
    ttt_epochs: int = 0
    ttt_chunks_per_epoch: int = 0
    ttt_val_eval_metric: Optional[str] = None
    ttt_val_baseline_metric: Optional[float] = None
    ttt_val_best_metric: Optional[float] = None
    ttt_val_baseline_accuracy: Optional[float] = None
    ttt_val_best_accuracy: Optional[float] = None
    ttt_best_epoch: int = 0
    ttt_stopped_early: bool = False
    ttt_oom_fallback: bool = False
    ttt_fallback_reason: Optional[str] = None


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
    f1: Optional[float]
    balanced_accuracy: Optional[float]
    roc_auc: Optional[float]
    log_loss: Optional[float]
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
    ttt_epochs: int = 0
    ttt_chunks_per_epoch: int = 0
    ttt_val_eval_metric: Optional[str] = None
    ttt_val_baseline_metric: Optional[float] = None
    ttt_val_best_metric: Optional[float] = None
    ttt_val_baseline_accuracy: Optional[float] = None
    ttt_val_best_accuracy: Optional[float] = None
    ttt_best_epoch: int = 0
    ttt_stopped_early: bool = False
    ttt_oom_fallback: bool = False
    ttt_fallback_reason: Optional[str] = None
    ttt_split_strategy: Optional[str] = None
    ttt_split_reason: Optional[str] = None


RESULT_COLUMNS = list(ResultRow.__annotations__.keys())
OOM_ERROR_MARKERS = base.OOM_ERROR_MARKERS


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def format_exception_for_csv(exc: BaseException) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())


def is_oom_exception(exc: BaseException) -> bool:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        oom_types = [
            getattr(torch, "OutOfMemoryError", None),
            getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),
        ]
        for oom_type in oom_types:
            if oom_type is not None and isinstance(exc, oom_type):
                return True

    text = f"{type(exc).__name__}: {exc}".strip().lower()
    return any(marker in text for marker in OOM_ERROR_MARKERS)


def is_oom_error_message(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and any(marker in text for marker in OOM_ERROR_MARKERS)


def truthy_column_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column].fillna(False)
    if values.dtype == bool:
        return values
    return values.map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def append_ttt_reason(existing: Optional[str], addition: str) -> str:
    if not existing:
        return addition
    return f"{existing} | {addition}"


def force_memory_cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def count_ttt_chunks(n_samples: int, max_chunk_size: int, min_chunk_size: int) -> int:
    if n_samples < max(2, min_chunk_size):
        return 0
    return max(1, (int(n_samples) + int(max_chunk_size) - 1) // int(max_chunk_size))


def iter_epoch_chunk_indices(
    n_samples: int,
    *,
    max_chunk_size: int,
    min_chunk_size: int,
    seed: int,
) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    for start in range(0, n_samples, max_chunk_size):
        chunk = indices[start : start + max_chunk_size]
        if len(chunk) >= min_chunk_size:
            yield chunk


def split_ctx_query(y_chunk: np.ndarray, *, query_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    from sklearn.model_selection import train_test_split

    y_chunk = np.asarray(y_chunk)
    n_samples = int(len(y_chunk))
    n_classes = int(len(np.unique(y_chunk)))
    if n_samples < max(2, n_classes + 1):
        raise ValueError("chunk too small for context/query split")
    query_size = max(n_classes, int(np.ceil(n_samples * query_ratio)))
    query_size = min(query_size, n_samples - 1)
    all_indices = np.arange(n_samples)
    _, counts = np.unique(y_chunk, return_counts=True)
    can_stratify = n_classes > 1 and np.min(counts) >= 2 and query_size >= n_classes
    if can_stratify:
        ctx_idx, qry_idx = train_test_split(
            all_indices,
            test_size=query_size,
            random_state=seed,
            stratify=y_chunk,
        )
        return np.asarray(ctx_idx), np.asarray(qry_idx), "stratified"

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    qry_idx = perm[:query_size]
    ctx_idx = perm[query_size:]
    return np.asarray(ctx_idx), np.asarray(qry_idx), "random"


def split_ttt_validation(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    validation_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame | None, np.ndarray | None, str | None]:
    from sklearn.model_selection import train_test_split

    y_arr = np.asarray(y)
    if len(y_arr) < 4:
        return X, y_arr, None, None, None
    _, counts = np.unique(y_arr, return_counts=True)
    stratify = y_arr if len(counts) > 1 and np.min(counts) >= 2 else None
    try:
        train_idx, val_idx = train_test_split(
            np.arange(len(y_arr)),
            test_size=validation_fraction,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return X, y_arr, None, None, None
    return (
        X.iloc[train_idx].reset_index(drop=True),
        y_arr[train_idx],
        X.iloc[val_idx].reset_index(drop=True),
        y_arr[val_idx],
        f"internal validation_fraction={validation_fraction}",
    )


def normalize_probabilities(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.maximum(scores, 0.0)
    row_sums = scores.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze(1) <= 0
    if np.any(zero_rows):
        scores[zero_rows, :] = 1.0
        row_sums = scores.sum(axis=1, keepdims=True)
    return scores / row_sums


def compute_roc_auc(y_true: Any, y_proba: Any, classes: Any) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score

        y_true_arr = np.asarray(y_true)
        proba_arr = np.asarray(y_proba)
        class_arr = None if classes is None else np.asarray(classes)
        if proba_arr.ndim != 2 or proba_arr.shape[0] != y_true_arr.shape[0] or proba_arr.shape[1] < 2:
            return None
        if len(np.unique(y_true_arr)) < 2:
            return None
        if class_arr is not None:
            if class_arr.ndim != 1 or len(class_arr) != proba_arr.shape[1]:
                class_arr = None
            elif not set(np.unique(y_true_arr).tolist()).issubset(set(class_arr.tolist())):
                return None
        if proba_arr.shape[1] == 2:
            if class_arr is not None and len(class_arr) == 2:
                y_binary = (y_true_arr == class_arr[1]).astype(int)
                if len(np.unique(y_binary)) < 2:
                    return None
                return float(roc_auc_score(y_binary, proba_arr[:, 1]))
            return float(roc_auc_score(y_true_arr, proba_arr[:, 1]))
        if class_arr is None:
            class_arr = np.asarray(sorted(pd.unique(pd.Series(y_true_arr)).tolist()))
            if len(class_arr) != proba_arr.shape[1]:
                return None
        return float(roc_auc_score(y_true_arr, proba_arr, labels=class_arr, multi_class="ovr"))
    except Exception:
        return None


def compute_balanced_accuracy(y_true: Any, y_pred: Any) -> Optional[float]:
    try:
        from sklearn.metrics import balanced_accuracy_score

        return float(balanced_accuracy_score(np.asarray(y_true), np.asarray(y_pred)))
    except Exception:
        return None


def compute_log_loss(y_true: Any, y_proba: Any, classes: Any) -> Optional[float]:
    try:
        from sklearn.metrics import log_loss

        y_true_arr = np.asarray(y_true)
        proba_arr = np.asarray(y_proba)
        if proba_arr.ndim != 2 or proba_arr.shape[0] != len(y_true_arr):
            return None
        class_arr = None if classes is None else np.asarray(classes)
        if class_arr is not None and (
            class_arr.ndim != 1 or len(class_arr) != proba_arr.shape[1]
        ):
            class_arr = None
        if class_arr is not None:
            return float(log_loss(y_true_arr, proba_arr, labels=list(class_arr)))
        return float(log_loss(y_true_arr, proba_arr))
    except Exception:
        return None


class LimiXTTTAdapter(base.LimiXAdapter):
    def __init__(self, args: argparse.Namespace, device: str, ttt_config: TTTConfig) -> None:
        super().__init__(args, device)
        self.ttt_config = ttt_config

    def _inference_n_estimators(self) -> int:
        return max(0, int(getattr(self.args, "n_estimators", 0)))

    def fit_predict_with_flash_attention_disabled(
        self,
        loaded: base.LoadedDataset,
    ) -> PredictionResult:
        disabled = self._disable_flash_attention_runtime()
        self.reset_predictor()
        if self.args.verbose:
            state = "after disabling flash attention" if disabled else "with fresh predictor"
            print(f"[flash_attn_retry] {loaded.dataset_name}: retrying {state}", flush=True)
        return self.fit_predict(loaded)

    def _validate_dataset_limits(self, loaded: base.LoadedDataset, y_train: np.ndarray) -> None:
        n_train_classes = len(pd.unique(pd.Series(y_train)))
        n_dataset_classes = loaded.n_classes
        if n_train_classes < 2:
            raise base.SkipDataset(f"LimiX supports at least 2 train classes; got {n_train_classes}")
        if self.args.max_classes and n_dataset_classes > self.args.max_classes:
            raise base.SkipDataset(
                f"LimiX supports up to {self.args.max_classes} dataset classes; got {n_dataset_classes}"
            )
        if self.args.max_train_rows > 0 and len(y_train) >= self.args.max_train_rows:
            raise base.SkipDataset(
                f"LimiX skips train size >= {self.args.max_train_rows}; got {len(y_train)}"
            )

    def _selected_pipeline_indices(self, predictor, limit: int = 0) -> list[int]:
        n_estimators = int(getattr(predictor, "n_estimators", len(getattr(predictor, "inference_config", []))))
        if limit and limit > 0:
            n_estimators = min(n_estimators, int(limit))
        return list(range(n_estimators))

    def _has_retrieval_config(self, predictor) -> bool:
        for item in getattr(predictor, "inference_config", []):
            retrieval_config = item.get("retrieval_config", {}) if isinstance(item, dict) else {}
            if retrieval_config.get("use_retrieval"):
                return True
        return False

    def _init_cls_encoding(self, predictor, y_train: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import LabelEncoder

        np_rng = np.random.default_rng(int(self.args.random_state))
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(np.asarray(y_train))
        predictor.label_encoder = label_encoder
        predictor.classes = label_encoder.classes_
        predictor.n_classes = len(predictor.classes)

        noise = np_rng.random((predictor.n_estimators * predictor.class_shuffle_factor, predictor.n_classes))
        shufflings = np.argsort(noise, axis=1)
        uniqs = np.unique(shufflings, axis=0)
        balance_count = predictor.n_estimators // len(uniqs)
        predictor.class_permutations = list(chain.from_iterable(repeat(elem, balance_count) for elem in uniqs))
        cout = predictor.n_estimators % len(uniqs)
        if cout > 0:
            predictor.class_permutations += [uniqs[i] for i in np_rng.choice(len(uniqs), size=cout)]
        return np.asarray(y_encoded)

    def _predict_proba_direct(
        self,
        predictor,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        *,
        max_pipelines: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        original_pipelines = predictor.preprocess_pipelines
        original_configs = predictor.inference_config
        original_n_estimators = predictor.n_estimators
        if max_pipelines and max_pipelines > 0:
            keep = int(min(max_pipelines, predictor.n_estimators))
            predictor.preprocess_pipelines = predictor.preprocess_pipelines[:keep]
            predictor.inference_config = predictor.inference_config[:keep]
            predictor.n_estimators = keep
        try:
            prediction = predictor.predict(
                X_train.to_numpy(),
                np.asarray(y_train),
                X_test.to_numpy(),
                task_type="Classification",
            )
            proba = np.asarray(prediction)
            classes = np.asarray(getattr(predictor, "classes", np.arange(proba.shape[1] if proba.ndim == 2 else 0)))
            if proba.ndim != 2:
                one_hot = np.zeros((len(proba), len(classes)), dtype=np.float32)
                class_to_idx = {value: idx for idx, value in enumerate(classes.tolist())}
                for row_idx, value in enumerate(proba):
                    if value in class_to_idx:
                        one_hot[row_idx, class_to_idx[value]] = 1.0
                proba = one_hot
            return normalize_probabilities(proba), classes
        finally:
            predictor.preprocess_pipelines = original_pipelines
            predictor.inference_config = original_configs
            predictor.n_estimators = original_n_estimators

    def _predict_proba_one_vs_rest(
        self,
        predictor,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        *,
        max_pipelines: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        train_classes = pd.unique(pd.Series(np.asarray(y_train)))
        scores: list[np.ndarray] = []
        for class_value in train_classes:
            y_binary = (np.asarray(y_train) == class_value).astype(np.int64)
            if len(np.unique(y_binary)) < 2:
                scores.append(np.zeros(len(X_test), dtype=np.float32))
                continue
            proba, binary_classes = self._predict_proba_direct(
                predictor,
                X_train,
                y_binary,
                X_test,
                max_pipelines=max_pipelines,
            )
            positive_cols = np.where(np.asarray(binary_classes) == 1)[0]
            positive_col = int(positive_cols[0]) if len(positive_cols) else min(1, proba.shape[1] - 1)
            scores.append(np.asarray(proba[:, positive_col], dtype=np.float32))
        return normalize_probabilities(np.column_stack(scores)), np.asarray(train_classes)

    def _predict_proba_with_strategy(
        self,
        predictor,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        *,
        model_max_classes: int,
        test_batch_rows: int,
        max_pipelines: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_train_classes = int(len(pd.unique(pd.Series(np.asarray(y_train)))))
        use_ovr = n_train_classes > model_max_classes
        if test_batch_rows <= 0 or len(X_test) <= test_batch_rows:
            if use_ovr:
                return self._predict_proba_one_vs_rest(
                    predictor,
                    X_train,
                    y_train,
                    X_test,
                    max_pipelines=max_pipelines,
                )
            return self._predict_proba_direct(
                predictor,
                X_train,
                y_train,
                X_test,
                max_pipelines=max_pipelines,
            )

        probas: list[np.ndarray] = []
        classes: np.ndarray | None = None
        for start in range(0, len(X_test), test_batch_rows):
            batch = X_test.iloc[start : start + test_batch_rows].reset_index(drop=True)
            if use_ovr:
                batch_proba, batch_classes = self._predict_proba_one_vs_rest(
                    predictor,
                    X_train,
                    y_train,
                    batch,
                    max_pipelines=max_pipelines,
                )
            else:
                batch_proba, batch_classes = self._predict_proba_direct(
                    predictor,
                    X_train,
                    y_train,
                    batch,
                    max_pipelines=max_pipelines,
                )
            if classes is None:
                classes = np.asarray(batch_classes)
            probas.append(np.asarray(batch_proba))
        return np.concatenate(probas, axis=0), np.asarray(classes)

    def _baseline_predict(
        self,
        loaded: base.LoadedDataset,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        *,
        ttt_reason: str | None = None,
        ttt_oom_fallback: bool = False,
        ttt_fallback_reason: str | None = None,
    ) -> PredictionResult:
        self._validate_dataset_limits(loaded, np.asarray(y_context))
        predictor = self._load_predictor()
        model_max_classes = self._model_max_classes(predictor)
        predict_started = time.time()
        y_proba, classes = self._predict_proba_with_strategy(
            predictor,
            X_context,
            np.asarray(y_context),
            loaded.X_test,
            model_max_classes=model_max_classes,
            test_batch_rows=int(self.args.test_batch_rows),
            max_pipelines=self._inference_n_estimators(),
        )
        predict_seconds = time.time() - predict_started
        y_pred = np.asarray(classes)[np.argmax(np.asarray(y_proba), axis=1)]
        return PredictionResult(
            y_pred=y_pred,
            y_proba=y_proba,
            classes=classes,
            fit_seconds=0.0,
            predict_seconds=float(predict_seconds),
            n_train_a=int(len(y_context)),
            n_train_b=0,
            n_holdout_c=0,
            n_test_d=int(len(loaded.y_test)),
            ttt_lr=self.ttt_config.lr if self.ttt_config.enabled else None,
            ttt_applied=False,
            ttt_split_reason=ttt_reason,
            ttt_oom_fallback=ttt_oom_fallback,
            ttt_fallback_reason=ttt_fallback_reason,
        )

    def _configure_trainable_params(self, predictor) -> list[Any]:
        model = predictor.model
        for param in model.parameters():
            param.requires_grad = False

        scope = self.ttt_config.trainable_scope
        if scope == "all":
            for param in model.parameters():
                param.requires_grad = True
        elif scope == "decoder":
            for name, param in model.named_parameters():
                if name.startswith("cls_y_decoder"):
                    param.requires_grad = True
        elif scope == "last_layer_decoder":
            last_layer_prefix = f"transformer_encoder.layers.{len(model.transformer_encoder.layers) - 1}."
            for name, param in model.named_parameters():
                if name.startswith("cls_y_decoder") or name.startswith(last_layer_prefix):
                    param.requires_grad = True
        else:
            raise ValueError(f"Unsupported --ttt-trainable-scope: {scope}")

        return [param for param in model.parameters() if param.requires_grad]

    def _forward_chunk_loss(
        self,
        predictor,
        X_chunk: pd.DataFrame,
        y_chunk: np.ndarray,
        *,
        epoch_seed: int,
        chunk_idx: int,
    ) -> tuple[Any, str]:
        import torch
        import torch.nn.functional as F
        from inference.inference_method import InferenceAttentionMap
        from inference.preprocess import SubSampleData

        ctx_idx, qry_idx, split_strategy = split_ctx_query(
            np.asarray(y_chunk),
            query_ratio=self.ttt_config.query_ratio,
            seed=epoch_seed + chunk_idx * 7919,
        )
        y_ctx_raw = np.asarray(y_chunk)[ctx_idx]
        y_qry_raw = np.asarray(y_chunk)[qry_idx]
        missing_query_labels = sorted(set(y_qry_raw.tolist()) - set(y_ctx_raw.tolist()))
        if missing_query_labels:
            raise ValueError(
                "query labels absent from context after "
                f"{split_strategy} split: {','.join(str(item) for item in missing_query_labels)}"
            )

        X_ctx = X_chunk.iloc[ctx_idx].reset_index(drop=True)
        X_qry = X_chunk.iloc[qry_idx].reset_index(drop=True)
        x_raw = np.concatenate([X_ctx.to_numpy(), X_qry.to_numpy()], axis=0)
        y_encoded_all = self._init_cls_encoding(predictor, y_ctx_raw)
        y_qry = predictor.label_encoder.transform(y_qry_raw)
        n_classes = int(predictor.n_classes)
        categorical_idx = None
        losses = []
        pipeline_indices = self._selected_pipeline_indices(predictor, self.ttt_config.n_estimators_finetune)

        for id_pipe in pipeline_indices:
            pipe = predictor.preprocess_pipelines[id_pipe]
            x_ = x_raw.copy()
            y_permuted = predictor.class_permutations[id_pipe][y_encoded_all.copy()]
            x_ = predictor.convert_x_dtypes(x_)
            x_ = predictor.convert_category2num(x_)
            x_ = x_.astype(np.float32)
            categorical_idx = predictor.get_categorical_features_indices(x_) if categorical_idx is None else list(categorical_idx)
            categorical_idx_ = list(categorical_idx)
            for id_step, step in enumerate(pipe):
                if isinstance(step, InferenceAttentionMap) or isinstance(step, SubSampleData):
                    raise ValueError("LimiX TTT currently supports only no-retrieval inference configs")
                x_, categorical_idx_ = step.fit_transform(
                    x_,
                    categorical_idx_,
                    predictor.seeds[id_pipe * predictor.preprocess_num + id_step],
                    y=y_permuted,
                )

            x_tensor = torch.from_numpy(x_).float().to(predictor.device)
            y_tensor = torch.from_numpy(y_permuted).float().to(predictor.device)
            y_query_tensor = torch.from_numpy(np.asarray(y_qry)).long().to(predictor.device)
            with torch.autocast(
                device_type=predictor.device.type,
                enabled=bool(predictor.mix_precision and predictor.device.type == "cuda"),
            ):
                output = predictor.model(
                    x=x_tensor.unsqueeze(0),
                    y=y_tensor.unsqueeze(0),
                    eval_pos=len(y_permuted),
                    task_type="cls",
                )
                logits = output if isinstance(output, torch.Tensor) else output["cls_output"]
                logits = logits.squeeze(0)[:, :n_classes]
                if predictor.softmax_temperature != 1:
                    logits = logits.float() / predictor.softmax_temperature
                logits = logits[..., predictor.class_permutations[id_pipe]]
                losses.append(F.cross_entropy(logits.float(), y_query_tensor))

        if not losses:
            raise ValueError("No LimiX pipelines available for TTT loss")
        return torch.stack(losses).mean(), split_strategy

    def _evaluate_validation_accuracy(
        self,
        predictor,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        X_val: pd.DataFrame | None,
        y_val: np.ndarray | None,
    ) -> Optional[float]:
        if X_val is None or y_val is None or len(y_val) == 0:
            return None
        import torch

        model = predictor.model
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                validation_n_estimators = int(self.ttt_config.validation_n_estimators)
                if validation_n_estimators <= 0:
                    validation_n_estimators = self._inference_n_estimators()
                y_proba, classes = self._predict_proba_with_strategy(
                    predictor,
                    X_context,
                    np.asarray(y_context),
                    X_val,
                    model_max_classes=self._model_max_classes(predictor),
                    test_batch_rows=int(self.args.test_batch_rows),
                    max_pipelines=validation_n_estimators,
                )
            y_pred = np.asarray(classes)[np.argmax(np.asarray(y_proba), axis=1)]
            return float(np.mean(np.asarray(y_pred) == np.asarray(y_val)))
        finally:
            model.train(was_training)

    def _run_ttt_update(
        self,
        predictor,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None,
        y_val: np.ndarray | None,
        *,
        dataset_name: str,
    ) -> TTTUpdateResult:
        import torch

        config = self.ttt_config
        update_start = time.time()
        if config.epochs < 1:
            return TTTUpdateResult(False, None, 0, 0.0, reason="--ttt-epochs must be >= 1")
        if config.max_chunk_size < 2:
            raise ValueError("--ttt-max-chunk-size must be >= 2")
        if not 0.0 < config.query_ratio < 1.0:
            raise ValueError("--ttt-query-ratio must be in (0, 1)")
        if self._has_retrieval_config(predictor):
            return TTTUpdateResult(
                False,
                None,
                0,
                time.time() - update_start,
                reason="TTT skipped because retrieval inference configs are not supported in limix_ttt.py",
            )

        y_train = np.asarray(y_train)
        self._init_cls_encoding(predictor, y_train)
        if predictor.n_classes > self._model_max_classes(predictor):
            return TTTUpdateResult(
                False,
                None,
                0,
                time.time() - update_start,
                reason=(
                    f"TTT skipped because n_classes={predictor.n_classes} "
                    f"exceeds model max_classes={self._model_max_classes(predictor)}"
                ),
            )

        chunks_per_epoch = count_ttt_chunks(
            len(y_train),
            max_chunk_size=int(config.max_chunk_size),
            min_chunk_size=int(config.min_chunk_size),
        )
        if chunks_per_epoch == 0:
            return TTTUpdateResult(
                False,
                None,
                0,
                time.time() - update_start,
                reason="Need at least one valid chunk for LimiX TTT",
            )

        model = predictor.model.to(predictor.device)
        model.train()
        trainable_params = self._configure_trainable_params(predictor)
        if not trainable_params:
            return TTTUpdateResult(
                False,
                None,
                0,
                time.time() - update_start,
                reason="No trainable parameters selected for LimiX TTT",
            )

        optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
        baseline_metric = None
        best_metric = None
        baseline_accuracy = None
        best_accuracy = None
        best_state = None
        best_epoch = 0
        patience_counter = 0
        stopped_early = False
        update_steps = 0
        last_loss = None
        skipped_batches = 0
        skip_reasons: dict[str, int] = {}

        if config.early_stopping and X_val is not None and y_val is not None and len(y_val) > 0:
            baseline_metric = self._evaluate_validation_accuracy(predictor, X_train, y_train, X_val, y_val)
            if baseline_metric is not None:
                best_metric = float(baseline_metric)
                baseline_accuracy = float(baseline_metric)
                best_accuracy = float(baseline_metric)
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                if self.args.verbose:
                    print(f"[ttt-val] dataset={dataset_name} baseline_accuracy={best_metric:.6f}", flush=True)

        for epoch_idx in range(config.epochs):
            epoch_loss_sum = 0.0
            epoch_updates = 0
            for chunk_idx, chunk_indices in enumerate(
                iter_epoch_chunk_indices(
                    len(y_train),
                    max_chunk_size=int(config.max_chunk_size),
                    min_chunk_size=int(config.min_chunk_size),
                    seed=int(self.args.random_state) + epoch_idx,
                )
            ):
                X_chunk = X_train.iloc[chunk_indices].reset_index(drop=True)
                y_chunk = y_train[chunk_indices]
                try:
                    loss, _split_strategy = self._forward_chunk_loss(
                        predictor,
                        X_chunk,
                        y_chunk,
                        epoch_seed=int(self.args.random_state) + epoch_idx,
                        chunk_idx=chunk_idx,
                    )
                except ValueError as exc:
                    skipped_batches += 1
                    reason = str(exc)
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip)
                optimizer.step()
                update_steps += 1
                epoch_updates += 1
                last_loss = float(loss.detach().cpu())
                epoch_loss_sum += last_loss

            if self.args.verbose and epoch_updates:
                print(
                    f"[ttt-loss] dataset={dataset_name} epoch={epoch_idx + 1}/{config.epochs} "
                    f"mean_loss={epoch_loss_sum / epoch_updates:.6f} updates={epoch_updates}",
                    flush=True,
                )

            if best_state is not None:
                val_metric = self._evaluate_validation_accuracy(predictor, X_train, y_train, X_val, y_val)
                if val_metric is not None:
                    improved = val_metric > float(best_metric) + float(config.min_delta)
                    if improved:
                        best_metric = float(val_metric)
                        best_accuracy = float(val_metric)
                        best_epoch = epoch_idx + 1
                        patience_counter = 0
                        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    else:
                        patience_counter += 1
                    if self.args.verbose:
                        print(
                            f"[ttt-val] dataset={dataset_name} epoch={epoch_idx + 1}/{config.epochs} "
                            f"accuracy={val_metric:.6f} best={best_metric:.6f} "
                            f"patience={patience_counter}/{config.patience}",
                            flush=True,
                        )
                    if patience_counter >= config.patience:
                        stopped_early = True
                        break

        if update_steps == 0:
            reason = "No valid LimiX TTT chunks produced an optimizer update"
            if skipped_batches:
                top_reasons = sorted(skip_reasons.items(), key=lambda item: (-item[1], item[0]))[:3]
                reason += "; skipped=" + "; ".join(f"{count}x {text}" for text, count in top_reasons)
            return TTTUpdateResult(
                False,
                None,
                0,
                time.time() - update_start,
                reason=reason,
                epochs=config.epochs,
                chunks_per_epoch=chunks_per_epoch,
                val_eval_metric=config.eval_metric,
                val_baseline_metric=baseline_metric,
                val_best_metric=best_metric,
                val_baseline_accuracy=baseline_accuracy,
                val_best_accuracy=best_accuracy,
                best_epoch=best_epoch,
                stopped_early=stopped_early,
            )

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return TTTUpdateResult(
            True,
            last_loss,
            update_steps,
            time.time() - update_start,
            epochs=config.epochs,
            chunks_per_epoch=chunks_per_epoch,
            val_eval_metric=config.eval_metric,
            val_baseline_metric=baseline_metric,
            val_best_metric=best_metric,
            val_baseline_accuracy=baseline_accuracy,
            val_best_accuracy=best_accuracy,
            best_epoch=best_epoch,
            stopped_early=stopped_early,
        )

    def fit_predict(self, loaded: base.LoadedDataset) -> PredictionResult:
        y_baseline_context = np.asarray(loaded.y_train_merged)
        self._validate_dataset_limits(loaded, y_baseline_context)
        predictor = self._load_predictor()
        original_state = {key: value.detach().cpu().clone() for key, value in predictor.model.state_dict().items()}

        if not self.ttt_config.enabled:
            return self._baseline_predict(loaded, loaded.X_train_merged, y_baseline_context)

        X_ttt_train = loaded.X_train
        y_ttt_train = np.asarray(loaded.y_train)
        X_ttt_val = loaded.X_val
        y_ttt_val = np.asarray(loaded.y_val) if loaded.y_val is not None else None
        ttt_validation_reason = "dataset_val_split" if loaded.X_val is not None else None
        final_context_X = loaded.X_train_merged
        final_context_y = y_baseline_context
        if loaded.X_val is None and self.ttt_config.early_stopping:
            X_ttt_train, y_ttt_train, X_ttt_val, y_ttt_val, ttt_validation_reason = split_ttt_validation(
                loaded.X_train,
                loaded.y_train,
                validation_fraction=self.ttt_config.validation_fraction,
                random_state=int(self.args.random_state),
            )
            final_context_X = loaded.X_train
            final_context_y = np.asarray(loaded.y_train)

        ttt_split_reason = "full train set chunked per epoch"
        if ttt_validation_reason:
            ttt_split_reason += f" | validation={ttt_validation_reason}"

        fit_started = time.time()
        ttt_result: TTTUpdateResult | None = None
        try:
            ttt_result = self._run_ttt_update(
                predictor,
                X_ttt_train,
                y_ttt_train,
                X_ttt_val,
                y_ttt_val,
                dataset_name=loaded.dataset_name,
            )
        except Exception as exc:
            if not is_oom_exception(exc):
                predictor.model.load_state_dict(original_state)
                raise
            fallback_reason = (
                "TTT OOM; used original model parameters for inference: "
                f"{format_exception_for_csv(exc)}"
            )
            predictor.model.load_state_dict(original_state)
            force_memory_cleanup()
            result = self._baseline_predict(
                loaded,
                final_context_X,
                final_context_y,
                ttt_reason=append_ttt_reason(ttt_split_reason, fallback_reason),
                ttt_oom_fallback=True,
                ttt_fallback_reason=fallback_reason,
            )
            result.fit_seconds = time.time() - fit_started
            result.n_train_b = int(len(y_ttt_train))
            result.ttt_update_seconds = result.fit_seconds
            return result

        if ttt_result is not None and not ttt_result.applied:
            predictor.model.load_state_dict(original_state)

        predict_started = time.time()
        try:
            y_proba, classes = self._predict_proba_with_strategy(
                predictor,
                final_context_X,
                final_context_y,
                loaded.X_test,
                model_max_classes=self._model_max_classes(predictor),
                test_batch_rows=int(self.args.test_batch_rows),
                max_pipelines=self._inference_n_estimators(),
            )
            predict_seconds = time.time() - predict_started
            y_pred = np.asarray(classes)[np.argmax(np.asarray(y_proba), axis=1)]
        finally:
            predictor.model.load_state_dict(original_state)
            force_memory_cleanup()

        ttt_split_reason_out = ttt_split_reason
        if ttt_result and ttt_result.reason:
            ttt_split_reason_out = append_ttt_reason(ttt_split_reason_out, ttt_result.reason)

        return PredictionResult(
            y_pred=y_pred,
            y_proba=y_proba,
            classes=classes,
            fit_seconds=time.time() - fit_started,
            predict_seconds=float(predict_seconds),
            n_train_a=int(len(final_context_y)),
            n_train_b=int(len(y_ttt_train)),
            n_holdout_c=0,
            n_test_d=int(len(loaded.y_test)),
            ttt_loss=ttt_result.loss if ttt_result else None,
            ttt_steps=ttt_result.steps if ttt_result else 0,
            ttt_lr=self.ttt_config.lr,
            ttt_applied=bool(ttt_result.applied) if ttt_result else False,
            ttt_update_seconds=float(ttt_result.update_seconds) if ttt_result else 0.0,
            ttt_split_strategy="full_train_epoch_chunks",
            ttt_split_reason=ttt_split_reason_out,
            ttt_epochs=ttt_result.epochs if ttt_result else 0,
            ttt_chunks_per_epoch=ttt_result.chunks_per_epoch if ttt_result else 0,
            ttt_val_eval_metric=ttt_result.val_eval_metric if ttt_result else None,
            ttt_val_baseline_metric=ttt_result.val_baseline_metric if ttt_result else None,
            ttt_val_best_metric=ttt_result.val_best_metric if ttt_result else None,
            ttt_val_baseline_accuracy=ttt_result.val_baseline_accuracy if ttt_result else None,
            ttt_val_best_accuracy=ttt_result.val_best_accuracy if ttt_result else None,
            ttt_best_epoch=ttt_result.best_epoch if ttt_result else 0,
            ttt_stopped_early=ttt_result.stopped_early if ttt_result else False,
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
        f1=None,
        balanced_accuracy=None,
        roc_auc=None,
        log_loss=None,
        fit_seconds=0.0,
        predict_seconds=0.0,
        status=status,
        error=error,
    )


def worker_crash_row(dataset_name: str, error: str) -> ResultRow:
    return ResultRow(
        dataset_name=dataset_name,
        dataset_dir="__worker__",
        task_type=None,
        n_train=0,
        n_val=0,
        n_test=0,
        n_features=0,
        n_classes=0,
        accuracy=None,
        f1=None,
        balanced_accuracy=None,
        roc_auc=None,
        log_loss=None,
        fit_seconds=0.0,
        predict_seconds=0.0,
        status="fail",
        error=error,
    )


def evaluate_one_dataset(adapter: LimiXTTTAdapter, dataset_dir: Path) -> ResultRow:
    task_type: Optional[str] = None
    try:
        loaded = base.load_classification_dataset(dataset_dir)
        task_type = loaded.task_type
        try:
            result = adapter.fit_predict(loaded)
        except Exception as exc:
            if base.is_invalid_cuda_config_error(exc):
                result = adapter.fit_predict_with_flash_attention_disabled(loaded)
            else:
                raise
        y_pred = np.asarray(result.y_pred)
        y_test = np.asarray(loaded.y_test)
        if len(y_pred) != len(y_test):
            raise ValueError(f"Prediction length mismatch: got {len(y_pred)}, expected {len(y_test)}")
        from sklearn.metrics import f1_score

        accuracy = float(np.mean(y_pred == y_test))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        balanced_accuracy = compute_balanced_accuracy(y_test, y_pred)
        roc_auc = compute_roc_auc(y_test, result.y_proba, result.classes)
        log_loss_score = compute_log_loss(y_test, result.y_proba, result.classes)
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
            f1=f1,
            balanced_accuracy=balanced_accuracy,
            roc_auc=roc_auc,
            log_loss=log_loss_score,
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
            ttt_epochs=result.ttt_epochs,
            ttt_chunks_per_epoch=result.ttt_chunks_per_epoch,
            ttt_val_eval_metric=result.ttt_val_eval_metric,
            ttt_val_baseline_metric=result.ttt_val_baseline_metric,
            ttt_val_best_metric=result.ttt_val_best_metric,
            ttt_val_baseline_accuracy=result.ttt_val_baseline_accuracy,
            ttt_val_best_accuracy=result.ttt_val_best_accuracy,
            ttt_best_epoch=result.ttt_best_epoch,
            ttt_stopped_early=result.ttt_stopped_early,
            ttt_oom_fallback=result.ttt_oom_fallback,
            ttt_fallback_reason=result.ttt_fallback_reason,
            ttt_split_strategy=result.ttt_split_strategy,
            ttt_split_reason=result.ttt_split_reason,
        )
    except base.SkipDataset as exc:
        return empty_row_for_dataset(dataset_dir, "skip", str(exc), task_type=task_type)
    except Exception as exc:
        return empty_row_for_dataset(dataset_dir, "fail", f"{type(exc).__name__}: {exc}", task_type=task_type)


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


def result_row_from_dict(row_dict: dict[str, Any]) -> ResultRow:
    frame = ensure_result_columns(pd.DataFrame([row_dict]))
    normalized = frame.iloc[0].where(pd.notna(frame.iloc[0]), None).to_dict()
    return ResultRow(**normalized)


def merge_result_frames(base_df: pd.DataFrame, updated_df: pd.DataFrame) -> pd.DataFrame:
    base_df = ensure_result_columns(base_df)
    updated_df = ensure_result_columns(updated_df)
    if base_df.empty:
        return updated_df
    if updated_df.empty:
        return base_df

    order_map = {
        str(dataset_name): idx
        for idx, dataset_name in enumerate(base_df["dataset_name"].astype(str).tolist())
    }
    replacement_names = set(updated_df["dataset_name"].astype(str).tolist())
    base_keep = base_df[~base_df["dataset_name"].astype(str).isin(replacement_names)].copy()
    base_keep["_merge_order"] = base_keep["dataset_name"].astype(str).map(order_map)
    appended_start = len(order_map)
    updated_df = updated_df.copy()
    updated_df["_merge_order"] = [
        order_map.get(str(dataset_name), appended_start + idx)
        for idx, dataset_name in enumerate(updated_df["dataset_name"].tolist())
    ]
    merged = pd.concat([base_keep, updated_df], ignore_index=True)
    merged = merged.sort_values("_merge_order", kind="stable").drop(columns="_merge_order")
    return merged.reset_index(drop=True)[RESULT_COLUMNS]


def write_summary(summary_path: Path, result_df: pd.DataFrame, dataset_dirs: list[Path], wall_seconds: float) -> None:
    result_df = result_df.copy()
    for metric_column in ("accuracy", "f1", "balanced_accuracy", "roc_auc", "log_loss"):
        if metric_column in result_df.columns:
            result_df[metric_column] = pd.to_numeric(result_df[metric_column], errors="coerce")
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()
    skipped_df = result_df[result_df["status"] == "skip"].copy() if len(result_df) else pd.DataFrame()
    oom_fallback_df = ok_df[truthy_column_mask(ok_df, "ttt_oom_fallback")].copy() if len(ok_df) else pd.DataFrame()

    def mean_line(label: str, column: str) -> str:
        if len(ok_df) and column in ok_df.columns and ok_df[column].notna().any():
            return f"{label}: {ok_df[column].mean():.6f}"
        return f"{label}: (none)"

    lines = [
        f"discovered_datasets: {len(dataset_dirs)}",
        f"processed_datasets: {len(result_df)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        f"skipped_count: {len(skipped_df)}",
        f"ttt_oom_fallback_count: {len(oom_fallback_df)}",
        mean_line("avg_accuracy_ok", "accuracy"),
        mean_line("avg_f1_ok", "f1"),
        mean_line("avg_balanced_accuracy_ok", "balanced_accuracy"),
        mean_line("avg_roc_auc_ok", "roc_auc"),
        mean_line("avg_log_loss_ok", "log_loss"),
        f"wall_seconds: {wall_seconds:.3f}",
        "failed_datasets: "
        + (", ".join(failed_df["dataset_name"].astype(str).tolist()) if len(failed_df) else "(none)"),
        "skipped_datasets: "
        + (", ".join(skipped_df["dataset_name"].astype(str).tolist()) if len(skipped_df) else "(none)"),
        "ttt_oom_fallback_datasets: "
        + (", ".join(oom_fallback_df["dataset_name"].astype(str).tolist()) if len(oom_fallback_df) else "(none)"),
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_ttt_config(args: argparse.Namespace) -> TTTConfig:
    if int(args.ttt_epochs) < 1:
        raise ValueError("--ttt-epochs must be >= 1")
    if int(args.ttt_max_chunk_size) < 2:
        raise ValueError("--ttt-max-chunk-size must be >= 2")
    if int(args.ttt_min_chunk_size) < 1:
        raise ValueError("--ttt-min-chunk-size must be >= 1")
    if not 0.0 < float(args.ttt_query_ratio) < 1.0:
        raise ValueError("--ttt-query-ratio must be in (0, 1)")
    if not 0.0 < float(args.ttt_validation_fraction) < 1.0:
        raise ValueError("--ttt-validation-fraction must be in (0, 1)")
    if str(args.ttt_eval_metric) != "accuracy":
        raise ValueError("--ttt-eval-metric currently supports only: accuracy")
    return TTTConfig(
        enabled=bool(args.ttt_enabled),
        lr=float(args.ttt_lr),
        grad_clip=float(args.ttt_grad_clip),
        weight_decay=float(args.ttt_weight_decay),
        epochs=int(args.ttt_epochs),
        max_chunk_size=int(args.ttt_max_chunk_size),
        min_chunk_size=int(args.ttt_min_chunk_size),
        query_ratio=float(args.ttt_query_ratio),
        early_stopping=bool(args.ttt_early_stopping),
        patience=int(args.ttt_patience),
        min_delta=float(args.ttt_min_delta),
        eval_metric=str(args.ttt_eval_metric),
        validation_fraction=float(args.ttt_validation_fraction),
        validation_n_estimators=int(args.ttt_validation_n_estimators),
        n_estimators_finetune=int(args.ttt_n_estimators_finetune),
        trainable_scope=str(args.ttt_trainable_scope),
    )


def _evaluate_one_dataset_child(
    dataset_dir: str,
    gpu_id: int,
    args_dict: dict[str, Any],
    disable_flash_attention: bool,
    result_queue,
) -> None:
    base.bind_worker_gpu(gpu_id)
    args = argparse.Namespace(**args_dict)
    try:
        if disable_flash_attention:
            LimiXTTTAdapter._disable_flash_attention_runtime()
        adapter = LimiXTTTAdapter(args, device="cuda:0", ttt_config=build_ttt_config(args))
        row = evaluate_one_dataset(adapter, Path(dataset_dir))
    except Exception:
        row = worker_crash_row(Path(dataset_dir).name, traceback.format_exc())
    result_queue.put(asdict(row))


def evaluate_one_dataset_in_fresh_process(
    dataset_dir: Path,
    gpu_id: int,
    args_dict: dict[str, Any],
    *,
    disable_flash_attention: bool,
) -> ResultRow:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_evaluate_one_dataset_child,
        args=(str(dataset_dir), gpu_id, args_dict, disable_flash_attention, result_queue),
        daemon=False,
    )
    proc.start()
    proc.join()
    row_dict: dict[str, Any] | None = None
    if not result_queue.empty():
        row_dict = result_queue.get()
    result_queue.close()
    result_queue.join_thread()
    if row_dict is None:
        return worker_crash_row(
            dataset_dir.name,
            f"Child dataset retry exited without a result (exitcode={proc.exitcode})",
        )
    return result_row_from_dict(row_dict)


def worker_main(
    worker_id: int,
    gpu_id: int,
    dataset_dirs: list[str],
    worker_csv: str,
    args_dict: dict[str, Any],
) -> None:
    base.bind_worker_gpu(gpu_id)
    args = argparse.Namespace(**args_dict)
    rows: list[ResultRow] = []
    try:
        adapter = LimiXTTTAdapter(args, device="cuda:0", ttt_config=build_ttt_config(args))
        for dataset_dir in dataset_dirs:
            row = evaluate_one_dataset(adapter, Path(dataset_dir))
            if row.status == "fail" and base.is_invalid_cuda_config_error(row.error):
                row = evaluate_one_dataset_in_fresh_process(
                    Path(dataset_dir),
                    gpu_id,
                    args_dict,
                    disable_flash_attention=True,
                )
                LimiXTTTAdapter._disable_flash_attention_runtime()
                adapter = LimiXTTTAdapter(args, device="cuda:0", ttt_config=build_ttt_config(args))
            rows.append(row)
            rows_to_frame(rows).to_csv(worker_csv, index=False)
            if args.verbose:
                if row.status == "ok":
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] [ok] {row.dataset_name} "
                        f"accuracy={row.accuracy:.6f} f1={row.f1:.6f} "
                        f"roc_auc={row.roc_auc if row.roc_auc is not None else 'None'} "
                        f"ttt_applied={row.ttt_applied} ttt_oom_fallback={row.ttt_oom_fallback}",
                        flush=True,
                    )
                else:
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] [{row.status}] "
                        f"{row.dataset_name} error={row.error}",
                        flush=True,
                    )
    except Exception:
        rows.append(worker_crash_row(f"__WORKER_CRASH__{worker_id}", traceback.format_exc()))
    rows_to_frame(rows).to_csv(worker_csv, index=False)


def run_internal_ddp_dataset(args: argparse.Namespace) -> None:
    if not args.internal_ddp_dataset_dir or not args.internal_ddp_result_path:
        raise ValueError("Internal DDP mode requires --internal-ddp-dataset-dir and --internal-ddp-result-path")
    rank, local_rank = base.initialize_ddp_child()
    try:
        internal_args = argparse.Namespace(**vars(args))
        internal_args.ttt_enabled = False
        adapter = LimiXTTTAdapter(internal_args, device=f"cuda:{local_rank}", ttt_config=build_ttt_config(internal_args))
        row = evaluate_one_dataset(adapter, Path(args.internal_ddp_dataset_dir))
        if rank == 0:
            result_path = Path(args.internal_ddp_result_path)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(asdict(row), ensure_ascii=False), encoding="utf-8")
    finally:
        base.cleanup_ddp_child()


def evaluate_one_dataset_with_ddp(
    args: argparse.Namespace,
    dataset_dir: Path,
    gpu_ids: list[int],
    *,
    out_dir: Path,
) -> ResultRow:
    with tempfile.NamedTemporaryFile(
        prefix=f"{dataset_dir.name}_",
        suffix="_ddp_result.json",
        dir=out_dir,
        delete=False,
    ) as tmp_file:
        result_path = Path(tmp_file.name)

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(gpu_ids)}",
        str(Path(__file__).resolve()),
        "--internal-ddp-dataset-dir",
        str(dataset_dir.resolve()),
        "--internal-ddp-result-path",
        str(result_path),
        "--model-cache-dir",
        str(args.model_cache_dir),
        "--config-path",
        str(args.config_path),
        "--hf-repo",
        str(args.hf_repo),
        "--hf-filename",
        str(args.hf_filename),
        "--hf-endpoint",
        str(args.hf_endpoint),
        "--random-state",
        str(args.random_state),
        "--max-classes",
        str(args.max_classes),
        "--max-train-rows",
        str(args.max_train_rows),
        "--n-estimators",
        str(args.n_estimators),
        "--test-batch-rows",
        str(args.test_batch_rows),
        "--no-ttt",
    ]
    if args.direct_max_classes is not None:
        command.extend(["--direct-max-classes", str(args.direct_max_classes)])
    if args.model_path:
        command.extend(["--model-path", str(args.model_path)])
    if args.local_files_only:
        command.append("--local-files-only")
    if args.disable_mixed_precision:
        command.append("--disable-mixed-precision")
    if args.verbose:
        command.append("--verbose")

    env = os.environ.copy()
    base.bind_ddp_gpu_group(gpu_ids)
    env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    env["ROCR_VISIBLE_DEVICES"] = os.environ["ROCR_VISIBLE_DEVICES"]
    env.pop("HIP_VISIBLE_DEVICES", None)
    env.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
    env.setdefault("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "1"))

    try:
        completed = subprocess.run(
            command,
            cwd=str(Path.cwd()),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if args.verbose and completed.stdout.strip():
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
        if args.verbose and completed.stderr.strip():
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
        if completed.returncode == 0 and result_path.exists():
            row_dict = json.loads(result_path.read_text(encoding="utf-8"))
            return result_row_from_dict(row_dict)
        if completed.returncode == 0 and not result_path.exists():
            return empty_row_for_dataset(
                dataset_dir,
                "fail",
                "DDP subprocess finished successfully but did not produce a result row.",
            )
        return empty_row_for_dataset(
            dataset_dir,
            "fail",
            base.format_subprocess_error(command, completed),
        )
    finally:
        try:
            result_path.unlink(missing_ok=True)
        except Exception:
            pass


def run_retry_failed_datasets_with_ddp(args: argparse.Namespace) -> None:
    data_root = base.resolve_script_path(args.data_root)
    out_dir = base.resolve_script_path(args.out_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.workers is not None and int(args.workers) != 1:
        raise ValueError("--retry-failed-datasets-with-ddp requires --workers 1")
    gpu_ids = base.resolve_requested_gpu_ids(args.gpus)
    if len(gpu_ids) < 2:
        raise ValueError("--retry-failed-datasets-with-ddp requires at least 2 GPU ids")

    reference_results_csv = base.resolve_script_path(args.reference_results_csv)
    merge_results_csv = (
        base.resolve_script_path(args.merge_results_from_csv)
        if args.merge_results_from_csv
        else reference_results_csv
    )
    all_dataset_dirs = base.find_dataset_dirs(data_root)
    oom_dataset_names, skipped_non_oom_names = base.load_oom_failed_dataset_names_from_results_csv(reference_results_csv)
    dataset_dirs, missing_dataset_names = base.filter_dataset_dirs_by_name(all_dataset_dirs, oom_dataset_names)
    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: args.max_datasets]
    if not dataset_dirs:
        raise FileNotFoundError(
            f"No OOM retry target datasets found under {data_root} for {reference_results_csv}"
        )
    if missing_dataset_names:
        print("warning: retry CSV referenced OOM datasets missing under data_root: " + ", ".join(missing_dataset_names), file=sys.stderr, flush=True)
    if skipped_non_oom_names:
        print("warning: retry-failed-datasets-with-ddp skipped non-OOM failed datasets from reference CSV: " + ", ".join(skipped_non_oom_names), file=sys.stderr, flush=True)

    started = time.time()
    rows: list[ResultRow] = []
    worker_csv = out_dir / "worker_0.csv"
    for dataset_dir in dataset_dirs:
        row = evaluate_one_dataset_with_ddp(args, dataset_dir, gpu_ids, out_dir=out_dir)
        rows.append(row)
        rows_to_frame(rows).to_csv(worker_csv, index=False)

    result_df = rows_to_frame(rows)
    base_df = ensure_result_columns(pd.read_csv(merge_results_csv))
    result_df = merge_result_frames(base_df, result_df)
    all_csv = out_dir / "all_classification_results.csv"
    summary_txt = out_dir / "summary.txt"
    result_df.to_csv(all_csv, index=False)
    write_summary(summary_txt, result_df, all_dataset_dirs, time.time() - started)
    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {summary_txt}")


def run_benchmark(args: argparse.Namespace) -> None:
    data_root = base.resolve_script_path(args.data_root)
    out_dir = base.resolve_script_path(args.out_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dataset_dirs = base.find_dataset_dirs(data_root)
    dataset_dirs = list(all_dataset_dirs)
    retry_results_csv: Path | None = None
    merge_results_csv: Path | None = None
    if args.retry_failed_datasets_only:
        retry_results_csv = base.resolve_script_path(args.reference_results_csv)
        failed_dataset_names = base.load_failed_dataset_names_from_results_csv(
            retry_results_csv,
            include_oom_failures=args.retry_include_oom_failures,
        )
        dataset_dirs, missing_dataset_names = base.filter_dataset_dirs_by_name(dataset_dirs, failed_dataset_names)
        if missing_dataset_names:
            print("warning: retry CSV referenced datasets missing under data_root: " + ", ".join(missing_dataset_names), file=sys.stderr, flush=True)
    if args.merge_results_from_csv:
        merge_results_csv = base.resolve_script_path(args.merge_results_from_csv)
    elif retry_results_csv is not None:
        merge_results_csv = retry_results_csv
    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: args.max_datasets]
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset directories found under {data_root}")

    args.data_root = str(data_root)
    args.out_dir = str(out_dir)
    args.workers, gpu_ids = base.resolve_workers_and_gpu_ids(args)
    if args.verbose:
        print(f"resolved_workers: {args.workers}", flush=True)
        print(f"resolved_gpus: {','.join(str(gpu_id) for gpu_id in gpu_ids)}", flush=True)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    started = time.time()
    worker_csvs: list[Path] = []
    processes: list[mp.Process] = []
    args_dict = vars(args).copy()
    for worker_id in range(args.workers):
        assigned = [str(path.resolve()) for path in dataset_dirs[worker_id :: args.workers]]
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        worker_csvs.append(worker_csv)
        proc = mp.Process(
            target=worker_main,
            args=(worker_id, gpu_ids[worker_id], assigned, str(worker_csv), args_dict),
            daemon=False,
        )
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()

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
    parser = argparse.ArgumentParser(description="Run LimiX chunk/query TTT on data178 classification datasets.")
    parser.add_argument("--data-root", default="../../data178")
    parser.add_argument("--out-dir", default="limix_results/limix_ttt_epoch30_chunk150")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpus", default="2,3", help="Comma-separated physical GPU ids, or 'auto'.")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--config-path", default="config/cls_default_noretrieval.json")
    parser.add_argument("--hf-repo", default="stableai-org/LimiX-16M")
    parser.add_argument("--hf-filename", default="LimiX-16M.ckpt")
    parser.add_argument("--model-cache-dir", default="cache")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-classes", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--direct-max-classes", type=int, default=None)
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=0,
        help="Number of LimiX inference pipelines to use for baseline/final prediction; 0 uses all config pipelines.",
    )
    parser.add_argument("--test-batch-rows", type=int, default=0)
    parser.add_argument("--disable-mixed-precision", action="store_true")
    parser.add_argument("--ttt-holdout", dest="ttt_enabled", action="store_true")
    parser.add_argument("--no-ttt", dest="ttt_enabled", action="store_false")
    parser.set_defaults(ttt_enabled=True)
    parser.add_argument("--ttt-trainable-scope", choices=["all", "decoder", "last_layer_decoder"], default="all")
    parser.add_argument("--ttt-lr", type=float, default=5e-6)
    parser.add_argument("--ttt-grad-clip", type=float, default=1.0)
    parser.add_argument("--ttt-weight-decay", type=float, default=0.0)
    parser.add_argument("--ttt-epochs", "--ttt-steps", dest="ttt_epochs", type=int, default=30)
    parser.add_argument("--ttt-max-chunk-size", type=int, default=150)
    parser.add_argument("--ttt-min-chunk-size", type=int, default=10)
    parser.add_argument("--ttt-query-ratio", type=float, default=0.2)
    parser.add_argument("--ttt-early-stopping", type=parse_bool, default=True)
    parser.add_argument("--ttt-patience", type=int, default=8)
    parser.add_argument("--ttt-min-delta", type=float, default=1e-4)
    parser.add_argument("--ttt-eval-metric", choices=["accuracy"], default="accuracy")
    parser.add_argument("--ttt-validation-fraction", type=float, default=0.1)
    parser.add_argument("--ttt-validation-n-estimators", type=int, default=0)
    parser.add_argument("--ttt-n-estimators-finetune", type=int, default=0)
    parser.add_argument("--retry-failed-datasets-with-ddp", action="store_true")
    parser.add_argument("--retry-failed-datasets-only", action="store_true")
    parser.add_argument("--reference-results-csv", default="limix_results/all_classification_results.csv")
    parser.add_argument("--retry-include-oom-failures", action="store_true")
    parser.add_argument("--merge-results-from-csv", default=None)
    parser.add_argument("--internal-ddp-dataset-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-ddp-result-path", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.internal_ddp_dataset_dir or args.internal_ddp_result_path:
        run_internal_ddp_dataset(args)
        return
    if args.retry_failed_datasets_with_ddp and args.retry_failed_datasets_only:
        raise ValueError("--retry-failed-datasets-with-ddp cannot be combined with --retry-failed-datasets-only")
    if args.retry_failed_datasets_with_ddp:
        run_retry_failed_datasets_with_ddp(args)
        return
    run_benchmark(args)


if __name__ == "__main__":
    main()
