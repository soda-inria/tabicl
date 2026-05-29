#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import queue
import re
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_DATA_ROOT = Path("data178")
DEFAULT_MODEL_PATH = "tabicl-classifier-v2-20260212.ckpt"
DEFAULT_CHECKPOINT_VERSION = "tabicl-classifier-v2-20260212.ckpt"
CLASSIFICATION_TASKS = {"binclass", "multiclass"}
CATEGORICAL_MISSING_TOKEN = "__tabicl_missing__"
ADAPTOR_BLOCKS = {"mlp", "dcnv2"}
ADAPTOR_TARGET = "row_interactor_output"

np = None
pd = None


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
    ttt_split_strategy: Optional[str] = None
    ttt_split_reason: Optional[str] = None
    ttt_epochs: int = 0
    ttt_chunks_per_epoch: int = 0
    ttt_batch_mode: Optional[str] = None
    ttt_val_eval_metric: Optional[str] = None
    ttt_val_baseline_metric: Optional[float] = None
    ttt_val_best_metric: Optional[float] = None
    ttt_val_baseline_accuracy: Optional[float] = None
    ttt_val_best_accuracy: Optional[float] = None
    ttt_best_epoch: int = 0
    ttt_stopped_early: bool = False
    ttt_oom_fallback: bool = False
    ttt_fallback_reason: Optional[str] = None
    ttt_adaptor_enabled: bool = False
    ttt_adaptor_block: Optional[str] = None
    ttt_adaptor_bottleneck: int = 0
    ttt_adaptor_trainable_params: int = 0
    ttt_adaptor_target: Optional[str] = None


@dataclass
class TTTConfig:
    enabled: bool = False
    lr: float = 1e-5
    scheduler: str = "cosine_warmup"
    warmup_proportion: float = 0.1
    grad_clip: float = 1.0
    amp: bool | str = "auto"
    dtype: str = "float32"
    micro_batch_size: int = 1
    weight_decay: float = 0.01
    epochs: int = 30
    steps: int = 30
    max_chunk_size: int = 10_000
    min_chunk_size: int = 50
    query_ratio: float = 0.2
    n_estimators_finetune: int = 2
    early_stopping: bool = True
    patience: int = 8
    min_delta: float = 1e-4
    eval_metric: str = "roc_auc"
    validation_fraction: float = 0.1
    validation_n_estimators: int = 2
    freeze_col: bool = False
    freeze_row: bool = False
    freeze_icl: bool = False
    adaptor_enabled: bool = False
    adaptor_block: str = "mlp"
    adaptor_bottleneck: int = 64
    adaptor_dropout: float = 0.0
    adaptor_scale: float = 1.0
    adaptor_train_backbone: bool = False
    train_fraction: float = 0.75
    random_state: int = 42
    data_parallel: bool = False
    gpu_group: Optional[str] = None
    save_ckpt: bool = True
    save_ckpt_every: int = 2
    save_ckpt_start_step: Optional[int] = None
    ckpt_root: Optional[str] = None


@dataclass
class TTTSplit:
    b_indices: Any
    c_indices: Any
    strategy: str
    reason: str


@dataclass
class TTTUpdateResult:
    applied: bool
    loss: Optional[float]
    steps: int
    update_seconds: float
    reason: Optional[str] = None
    epochs: int = 0
    chunks_per_epoch: int = 0
    batch_mode: str = "epoch_chunk"
    val_eval_metric: Optional[str] = None
    val_baseline_metric: Optional[float] = None
    val_best_metric: Optional[float] = None
    val_baseline_accuracy: Optional[float] = None
    val_best_accuracy: Optional[float] = None
    best_epoch: int = 0
    stopped_early: bool = False
    adaptor_enabled: bool = False
    adaptor_block: Optional[str] = None
    adaptor_bottleneck: int = 0
    adaptor_trainable_params: int = 0
    adaptor_target: Optional[str] = None


@dataclass
class TTTValidationResult:
    primary: float
    secondary: Dict[str, float]


@dataclass
class MetaBatch:
    X: Any
    y_train: Any
    y_query: Any
    train_size: int
    skip_reason: Optional[str] = None


@dataclass
class ModelSummaryRow:
    model_name: str
    model_path: str
    gpu_id: int
    datasets_discovered: int
    ok_count: int
    failed_count: int
    skipped_count: int
    ttt_oom_fallback_count: int
    avg_accuracy_ok: Optional[float]
    avg_f1_ok: Optional[float]
    avg_balanced_accuracy_ok: Optional[float]
    avg_roc_auc_ok: Optional[float]
    avg_log_loss_ok: Optional[float]
    avg_fit_seconds_ok: Optional[float]
    avg_predict_seconds_ok: Optional[float]
    avg_dataset_seconds_ok: Optional[float]
    total_dataset_seconds_ok: float
    model_wall_seconds: float
    status: str
    error: Optional[str]
    failed_datasets: str
    ttt_adaptor_enabled_count: int = 0
    ttt_adaptor_block_counts: str = "(none)"
    avg_ttt_adaptor_trainable_params_ok: Optional[float] = None


OOM_ERROR_MARKERS = (
    "out of memory",
    "oom",
    "cuda out of memory",
    "cuda error: out of memory",
    "cudnn_status_alloc_failed",
    "outofmemoryerror",
)


def ensure_runtime_deps() -> None:
    global np
    global pd

    if np is None or pd is None:
        import numpy as _np
        import pandas as _pd

        np = _np
        pd = _pd


def format_optional_float(value: Optional[float], precision: int = 6) -> str:
    if value is None:
        return "None"
    try:
        if pd is not None and pd.isna(value):
            return "None"
    except Exception:
        pass
    return f"{float(value):.{precision}f}"


def compute_weighted_f1(y_true: Any, y_pred: Any) -> Optional[float]:
    ensure_runtime_deps()
    try:
        from sklearn.metrics import f1_score

        return float(
            f1_score(
                np.asarray(y_true),
                np.asarray(y_pred),
                average="weighted",
                zero_division=0,
            )
        )
    except Exception:
        return None


def compute_balanced_accuracy(y_true: Any, y_pred: Any) -> Optional[float]:
    ensure_runtime_deps()
    try:
        from sklearn.metrics import balanced_accuracy_score

        return float(balanced_accuracy_score(np.asarray(y_true), np.asarray(y_pred)))
    except Exception:
        return None


def compute_tabpfn_roc_auc(
    y_true: Any,
    y_proba: Any,
    classes: Any | None = None,
) -> Optional[float]:
    ensure_runtime_deps()
    try:
        from sklearn.metrics import roc_auc_score

        y_true_arr = np.asarray(y_true)
        proba_arr = np.asarray(y_proba)
        if proba_arr.ndim != 2 or proba_arr.shape[0] != y_true_arr.shape[0] or proba_arr.shape[1] < 2:
            return None
        if len(np.unique(y_true_arr)) < 2:
            return None
        class_arr = None if classes is None else np.asarray(classes)
        if class_arr is not None and (
            class_arr.ndim != 1 or len(class_arr) != proba_arr.shape[1]
        ):
            class_arr = None
        if proba_arr.shape[1] == 2:
            if class_arr is not None:
                y_binary = (y_true_arr == class_arr[1]).astype(int)
                if len(np.unique(y_binary)) < 2:
                    return None
                return float(roc_auc_score(y_binary, proba_arr[:, 1]))
            return float(roc_auc_score(y_true_arr, proba_arr[:, 1]))
        if class_arr is not None:
            return float(
                roc_auc_score(
                    y_true_arr,
                    proba_arr,
                    labels=list(class_arr),
                    multi_class="ovr",
                )
            )
        return float(roc_auc_score(y_true_arr, proba_arr, multi_class="ovr"))
    except (ValueError, RuntimeError, AttributeError, TypeError):
        return None


def compute_log_loss(
    y_true: Any,
    y_proba: Any | None,
    classes: Any | None,
) -> Optional[float]:
    if y_proba is None:
        return None
    ensure_runtime_deps()
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


def format_exception_for_csv(exc: BaseException) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())


def append_ttt_reason(existing: Optional[str], addition: str) -> str:
    if not existing:
        return addition
    return f"{existing} | {addition}"


def truthy_column_mask(frame: Any, column: str) -> Any:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)

    values = frame[column].fillna(False)
    if values.dtype == bool:
        return values
    return values.map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def format_adaptor_block_counts(frame: Any) -> str:
    if frame is None or not len(frame) or "ttt_adaptor_block" not in frame.columns:
        return "(none)"

    blocks = frame["ttt_adaptor_block"].dropna().map(lambda value: str(value).strip())
    blocks = blocks[(blocks != "") & (blocks.str.lower() != "none")]
    if not len(blocks):
        return "(none)"

    counts = blocks.value_counts()
    return ", ".join(f"{block}={int(counts[block])}" for block in sorted(counts.index))


def parse_adaptor_block_count_string(value: Any) -> Dict[str, int]:
    text = str(value).strip()
    if not text or text == "(none)":
        return {}

    counts: Dict[str, int] = {}
    for item in text.split(","):
        if "=" not in item:
            continue
        key, raw_count = item.split("=", 1)
        key = key.strip()
        try:
            count = int(float(raw_count.strip()))
        except ValueError:
            continue
        if key:
            counts[key] = counts.get(key, 0) + count
    return counts


def format_adaptor_block_count_totals(values: Any) -> str:
    totals: Dict[str, int] = {}
    for value in values:
        for block, count in parse_adaptor_block_count_string(value).items():
            totals[block] = totals.get(block, 0) + count
    if not totals:
        return "(none)"
    return ", ".join(f"{block}={totals[block]}" for block in sorted(totals))


def format_dataset_result_log(
    worker_label: str,
    row: ResultRow,
    *,
    model_name: str | None = None,
) -> str:
    prefix = f"[{worker_label}]"
    if model_name:
        prefix = f"{prefix} [{model_name}]"

    if row.status == "ok":
        return (
            f"{prefix} [ok] {row.dataset_name} "
            f"accuracy={format_optional_float(row.accuracy)} "
            f"f1={format_optional_float(row.f1)} "
            f"balanced_accuracy={format_optional_float(row.balanced_accuracy)} "
            f"roc_auc={format_optional_float(row.roc_auc)} "
            f"log_loss={format_optional_float(row.log_loss)} "
            f"fit={row.fit_seconds:.3f}s "
            f"predict={row.predict_seconds:.3f}s "
            f"ttt_applied={row.ttt_applied} "
            f"ttt_update={row.ttt_update_seconds:.3f}s "
            f"ttt_loss={format_optional_float(row.ttt_loss)} "
            f"ttt_steps={row.ttt_steps} "
            f"ttt_epochs={row.ttt_epochs} "
            f"ttt_mode={row.ttt_batch_mode} "
            f"ttt_val_metric={row.ttt_val_eval_metric} "
            f"ttt_val_best={format_optional_float(row.ttt_val_best_metric)} "
            f"ttt_best_epoch={row.ttt_best_epoch} "
            f"ttt_stopped_early={row.ttt_stopped_early} "
            f"ttt_oom_fallback={row.ttt_oom_fallback} "
            f"ttt_adaptor_enabled={row.ttt_adaptor_enabled} "
            f"ttt_adaptor_block={row.ttt_adaptor_block} "
            f"ttt_adaptor_params={row.ttt_adaptor_trainable_params}"
        )
    if row.status == "skip":
        return f"{prefix} [skip] {row.dataset_name} reason={row.error}"
    return f"{prefix} [fail] {row.dataset_name} error={row.error}"


def parse_optional_int(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered == "none":
        return None
    return int(value)


def parse_auto_bool(value: str) -> bool | str:
    lowered = value.strip().lower()
    if lowered == "auto":
        return "auto"
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("must be one of: auto, true, false")


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("must be one of: true, false")


def parse_kv_cache(value: str) -> bool | str:
    lowered = value.strip().lower()
    if lowered in {"false", "0", "no", "off"}:
        return False
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"kv", "repr"}:
        return lowered
    raise argparse.ArgumentTypeError("kv_cache must be one of: false, true, kv, repr")


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


def apply_worker_environment_updates(gpu_id: int | str) -> str:
    gpu_id_str = normalize_gpu_group(gpu_id)
    # Set visibility vars for both CUDA and ROCm stacks so each worker sees one GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str
    os.environ["ROCR_VISIBLE_DEVICES"] = gpu_id_str
    os.environ["HIP_VISIBLE_DEVICES"] = gpu_id_str
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    return "cuda:0"


def parse_gpu_id_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_gpu_group_list(value: str) -> List[str]:
    gpu_groups = []
    for raw_group in value.split(";"):
        raw_group = raw_group.strip()
        if not raw_group:
            continue
        gpu_groups.append(normalize_gpu_group(raw_group))
    return gpu_groups


def detect_default_gpu_ids() -> List[int]:
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

    try:
        import torch

        device_count = int(torch.cuda.device_count())
    except Exception:
        device_count = 0

    if device_count <= 0:
        raise RuntimeError("No visible GPU detected; please pass --gpus explicitly")
    return list(range(device_count))


def load_dataset_info(dataset_dir: Path) -> dict | None:
    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        return None
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def should_skip_ttt_for_dataset(dataset_dir: Path, info: dict | None) -> bool:
    dataset_names = {dataset_dir.name.strip().lower()}
    if info:
        info_name = str(info.get("name", "")).strip().lower()
        if info_name:
            dataset_names.add(info_name)
    return "volkert" in dataset_names


def find_dataset_dirs(data_root: Path) -> List[Path]:
    return [path for path in sorted(data_root.iterdir()) if path.is_dir()]


def collect_torch_diagnostics() -> Dict[str, object]:
    import torch

    try:
        device_count = torch.cuda.device_count()
    except Exception as exc:
        device_count = f"error: {exc}"

    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_hip_version": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": device_count,
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def stable_feature_prefix(context: str, fallback: str) -> str:
    stem = Path(context or fallback).stem
    for suffix in ("-train", "-test", "-val", "-single"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or fallback


def normalize_categorical_series(series: pd.Series) -> pd.Series:
    string_series = series.astype("string")
    string_series = string_series.fillna(CATEGORICAL_MISSING_TOKEN)
    return string_series.astype(str)


def make_feature_frame(
    values,
    *,
    kind: str,
    prefix: str,
) -> pd.DataFrame:
    ensure_runtime_deps()

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
    ensure_runtime_deps()

    y = np.asarray(values)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    elif y.ndim > 1 and y.shape[0] == 1:
        y = y.squeeze(0)
    return pd.Series(y).values


def load_array(file_path: Path) -> np.ndarray:
    ensure_runtime_deps()

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


def find_by_suffix(files: List[Path], suffix: str) -> Path | None:
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


def load_split(
    num_path: Path | None,
    cat_path: Path | None,
    y_path: Path,
    *,
    context: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    ensure_runtime_deps()

    features: List[pd.DataFrame] = []
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


def normalize_model_path(model_path: str | None) -> str | None:
    if model_path is None:
        return None
    if str(model_path).strip().lower() == "none":
        return None

    path = Path(model_path).expanduser()
    try:
        if path.exists():
            path = path.resolve()
    except Exception:
        pass
    return str(path)


def extract_last_int(stem: str) -> int | None:
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        return None
    return int(numbers[-1])


def discover_model_paths(models_dir: Path, max_models: int | None = None) -> List[Path]:
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory does not exist: {models_dir}")
    if not models_dir.is_dir():
        raise NotADirectoryError(f"Models directory is not a directory: {models_dir}")

    model_paths = [
        path
        for path in models_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".ckpt", ".pt", ".pth"}
    ]
    model_paths.sort(
        key=lambda path: (
            0 if extract_last_int(path.stem) is not None else 1,
            extract_last_int(path.stem) if extract_last_int(path.stem) is not None else path.stem,
            path.stem,
        )
    )
    if max_models is not None:
        model_paths = model_paths[:max_models]
    return model_paths


def preload_model_once(classifier: Any, worker_label: str, verbose: bool) -> None:
    load_fn = getattr(classifier, "_load_model", None)
    if not callable(load_fn):
        return

    t0 = time.perf_counter()
    load_fn()

    if hasattr(classifier, "_get_model_load_key"):
        try:
            classifier._loaded_model_key = classifier._get_model_load_key()
        except Exception:
            pass

    def _skip_reloading():
        return None

    try:
        classifier._load_model = _skip_reloading
    except Exception:
        pass

    if verbose:
        print(
            f"[{worker_label}] model preloaded once in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )


def release_classifier_resources(classifier: Any) -> None:
    if classifier is None:
        return

    try:
        model = getattr(classifier, "model_", None)
        if model is not None:
            clear_cache = getattr(model, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()
    except Exception:
        pass

    for attr_name in ("model_kv_cache_", "ensemble_generator_", "X_encoder_", "y_encoder_", "model_"):
        if hasattr(classifier, attr_name):
            try:
                setattr(classifier, attr_name, None)
            except Exception:
                pass


def force_memory_cleanup(device_str: str) -> None:
    gc.collect()

    try:
        import torch
    except Exception:
        return

    if not device_str.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        return

    try:
        device = torch.device(device_str)
        with torch.cuda.device(device):
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
    except Exception:
        pass


def build_ttt_config(args: argparse.Namespace) -> TTTConfig:
    ttt_adaptor_block = str(getattr(args, "ttt_adaptor_block", "mlp")).lower()
    if int(args.ttt_save_ckpt_every) < 1:
        raise ValueError("--ttt-save-ckpt-every must be >= 1")
    if args.ttt_save_ckpt_start_step is not None and int(args.ttt_save_ckpt_start_step) < 1:
        raise ValueError("--ttt-save-ckpt-start-step must be >= 1 or None")
    if int(args.ttt_epochs) < 1:
        raise ValueError("--ttt-epochs must be >= 1")
    if int(args.ttt_max_chunk_size) < 2:
        raise ValueError("--ttt-max-chunk-size must be >= 2")
    if int(args.ttt_min_chunk_size) < 1:
        raise ValueError("--ttt-min-chunk-size must be >= 1")
    if not 0.0 < float(args.ttt_query_ratio) < 1.0:
        raise ValueError("--ttt-query-ratio must be in (0, 1)")
    if int(args.ttt_n_estimators_finetune) < 1:
        raise ValueError("--ttt-n-estimators-finetune must be >= 1")
    if int(args.ttt_patience) < 1:
        raise ValueError("--ttt-patience must be >= 1")
    if float(args.ttt_min_delta) < 0:
        raise ValueError("--ttt-min-delta must be >= 0")
    if float(args.ttt_warmup_proportion) < 0:
        raise ValueError("--ttt-warmup-proportion must be >= 0")
    if not 0.0 < float(args.ttt_validation_fraction) < 1.0:
        raise ValueError("--ttt-validation-fraction must be in (0, 1)")
    if int(args.ttt_validation_n_estimators) < 1:
        raise ValueError("--ttt-validation-n-estimators must be >= 1")
    if str(args.ttt_eval_metric) not in {"roc_auc", "log_loss", "accuracy"}:
        raise ValueError("--ttt-eval-metric must be one of: roc_auc, log_loss, accuracy")
    if ttt_adaptor_block not in ADAPTOR_BLOCKS:
        raise ValueError("--ttt-adaptor-block must be one of: mlp, dcnv2")
    if int(args.ttt_adaptor_bottleneck) < 1:
        raise ValueError("--ttt-adaptor-bottleneck must be >= 1")
    if not 0.0 <= float(args.ttt_adaptor_dropout) < 1.0:
        raise ValueError("--ttt-adaptor-dropout must be in [0, 1)")
    if float(args.ttt_adaptor_scale) < 0:
        raise ValueError("--ttt-adaptor-scale must be >= 0")

    return TTTConfig(
        enabled=bool(args.ttt_enabled),
        lr=float(args.ttt_lr),
        scheduler=str(args.ttt_scheduler),
        warmup_proportion=float(args.ttt_warmup_proportion),
        grad_clip=float(args.ttt_grad_clip),
        amp=args.use_amp,
        dtype=str(args.ttt_dtype),
        micro_batch_size=int(args.ttt_micro_batch_size),
        weight_decay=float(args.ttt_weight_decay),
        epochs=int(args.ttt_epochs),
        steps=int(args.ttt_epochs),
        max_chunk_size=int(args.ttt_max_chunk_size),
        min_chunk_size=int(args.ttt_min_chunk_size),
        query_ratio=float(args.ttt_query_ratio),
        n_estimators_finetune=int(args.ttt_n_estimators_finetune),
        early_stopping=bool(args.ttt_early_stopping),
        patience=int(args.ttt_patience),
        min_delta=float(args.ttt_min_delta),
        eval_metric=str(args.ttt_eval_metric),
        validation_fraction=float(args.ttt_validation_fraction),
        validation_n_estimators=int(args.ttt_validation_n_estimators),
        freeze_col=bool(args.ttt_freeze_col),
        freeze_row=bool(args.ttt_freeze_row),
        freeze_icl=bool(args.ttt_freeze_icl),
        adaptor_enabled=bool(args.ttt_adaptor),
        adaptor_block=ttt_adaptor_block,
        adaptor_bottleneck=int(args.ttt_adaptor_bottleneck),
        adaptor_dropout=float(args.ttt_adaptor_dropout),
        adaptor_scale=float(args.ttt_adaptor_scale),
        adaptor_train_backbone=bool(args.ttt_adaptor_train_backbone),
        random_state=int(args.random_state),
        data_parallel=bool(args.ttt_data_parallel),
        save_ckpt=bool(args.ttt_save_ckpt),
        save_ckpt_every=int(args.ttt_save_ckpt_every),
        save_ckpt_start_step=(
            int(args.ttt_save_ckpt_start_step) if args.ttt_save_ckpt_start_step is not None else None
        ),
        ckpt_root=str((Path(args.out_dir).expanduser() / "ttt_ckpts").resolve()),
    )


def take_rows(X, indices):
    if hasattr(X, "iloc"):
        return X.iloc[indices].reset_index(drop=True)
    return np.asarray(X)[indices]


def count_ttt_chunks(n_samples: int, max_chunk_size: int, min_chunk_size: int) -> int:
    if n_samples <= 0:
        return 0
    if n_samples <= max_chunk_size:
        return 1
    n_full = n_samples // max_chunk_size
    remainder = n_samples - n_full * max_chunk_size
    return n_full + (1 if remainder >= min_chunk_size else 0)


def _chunk_indices(
    n_samples: int,
    *,
    max_chunk_size: int,
    min_chunk_size: int,
    rng,
) -> List[Any]:
    perm = rng.permutation(n_samples)
    if n_samples <= max_chunk_size:
        return [perm]
    n_full = n_samples // max_chunk_size
    chunks = [perm[i * max_chunk_size : (i + 1) * max_chunk_size] for i in range(n_full)]
    remainder = n_samples - n_full * max_chunk_size
    if remainder >= min_chunk_size:
        chunks.append(perm[n_full * max_chunk_size :])
    return chunks


def _split_ctx_query(y_chunk, *, query_size: int, seed: int) -> tuple[Any, Any, str]:
    from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

    n = len(y_chunk)
    query_size = max(1, min(int(query_size), n - 1))
    dummy_X = np.zeros((n, 1))
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=query_size, random_state=seed)
        ctx_idx, qry_idx = next(splitter.split(dummy_X, y_chunk))
        return ctx_idx, qry_idx, "stratified"
    except Exception as exc:
        splitter = ShuffleSplit(n_splits=1, test_size=query_size, random_state=seed)
        ctx_idx, qry_idx = next(splitter.split(dummy_X, y_chunk))
        return ctx_idx, qry_idx, f"random_fallback:{type(exc).__name__}"


def split_ttt_validation(
    X,
    y,
    *,
    validation_fraction: float,
    random_state: int,
) -> tuple[Any, Any, Any, Any, str]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(y))
    y_array = np.asarray(y)
    if len(indices) < 3:
        return X, y_array, None, None, "none: fewer than three samples"

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=random_state,
            shuffle=True,
            stratify=y_array,
        )
        strategy = "auto_stratified_validation"
    except Exception as exc:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=random_state,
            shuffle=True,
            stratify=None,
        )
        strategy = f"auto_random_validation:{type(exc).__name__}"

    return (
        take_rows(X, np.asarray(train_idx, dtype=int)),
        y_array[np.asarray(train_idx, dtype=int)],
        take_rows(X, np.asarray(val_idx, dtype=int)),
        y_array[np.asarray(val_idx, dtype=int)],
        strategy,
    )


def _torch_dtype(dtype_name: str):
    import torch

    normalized = dtype_name.strip().lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    raise ValueError("--ttt-dtype must be one of: float32, float16, bfloat16")


def _resolve_amp_enabled(amp_value: bool | str, device) -> bool:
    import torch

    device_type = getattr(device, "type", str(device).split(":")[0])
    if amp_value == "auto":
        return device_type == "cuda" and torch.cuda.is_available()
    return bool(amp_value) and device_type == "cuda" and torch.cuda.is_available()


def _make_ttt_amp(config: TTTConfig, device):
    import torch

    use_amp = _resolve_amp_enabled(config.amp, device)
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


def _set_requires_grad(module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _get_ttt_base_model(classifier):
    model = classifier.model_
    return getattr(model, "module", model)


def _resolve_ttt_data_parallel_device_ids(classifier, config: TTTConfig) -> List[int]:
    if not config.data_parallel or not config.gpu_group:
        return []

    device = getattr(classifier, "device_", None)
    device_type = getattr(device, "type", str(device).split(":")[0])
    if device_type != "cuda":
        return []

    try:
        gpu_ids = parse_gpu_id_list(normalize_gpu_group(config.gpu_group))
    except Exception:
        return []
    if len(gpu_ids) < 2:
        return []

    import torch

    if not torch.cuda.is_available():
        return []

    # Device visibility has already been rewritten per worker, so DataParallel
    # must use process-local logical ids rather than original global ids.
    return list(range(len(gpu_ids)))


def _build_ttt_forward_model(classifier, config: TTTConfig):
    base_model = _get_ttt_base_model(classifier)
    device_ids = _resolve_ttt_data_parallel_device_ids(classifier, config)
    if not device_ids:
        return base_model, 1

    import torch

    try:
        return torch.nn.DataParallel(base_model, device_ids=device_ids), len(device_ids)
    except Exception:
        return base_model, 1


def _remove_ttt_adaptor(classifier) -> None:
    model = _get_ttt_base_model(classifier)
    handle = getattr(model, "_ttt_adaptor_hook_handle", None)
    if handle is not None:
        try:
            handle.remove()
        except Exception:
            pass
    if hasattr(model, "ttt_adaptor"):
        delattr(model, "ttt_adaptor")
    if hasattr(model.row_interactor, "ttt_adaptor"):
        delattr(model.row_interactor, "ttt_adaptor")
    model._ttt_adaptor_hook_handle = None
    model._ttt_adaptor_target = None
    model._ttt_adaptor_block = None


def _install_ttt_adaptor(classifier, config: TTTConfig):
    import torch

    model = _get_ttt_base_model(classifier)
    _remove_ttt_adaptor(classifier)

    dim = int(getattr(model, "embed_dim")) * int(getattr(model, "row_num_cls"))
    bottleneck = min(max(1, int(config.adaptor_bottleneck)), dim)

    class TTTMLPResidualAdaptor(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, dropout: float, scale: float) -> None:
            super().__init__()
            self.scale = float(scale)
            self.net = torch.nn.Sequential(
                torch.nn.LayerNorm(input_dim),
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(hidden_dim, input_dim),
            )
            last_linear = self.net[-1]
            torch.nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                torch.nn.init.zeros_(last_linear.bias)

        def forward(self, x):
            return self.net(x) * self.scale

    class TTTDCNv2ResidualAdaptor(torch.nn.Module):
        def __init__(self, input_dim: int, cross_dim: int, dropout: float, scale: float) -> None:
            super().__init__()
            self.scale = float(scale)
            self.norm = torch.nn.LayerNorm(input_dim)
            self.down = torch.nn.Linear(input_dim, cross_dim)
            self.activation = torch.nn.GELU()
            self.dropout = torch.nn.Dropout(float(dropout))
            self.up = torch.nn.Linear(cross_dim, input_dim)
            torch.nn.init.zeros_(self.up.weight)
            if self.up.bias is not None:
                torch.nn.init.zeros_(self.up.bias)

        def forward(self, x):
            x_norm = self.norm(x)
            cross = self.up(self.dropout(self.activation(self.down(x_norm))))
            return x_norm * cross * self.scale

    adaptor_block = str(config.adaptor_block).lower()
    if adaptor_block == "mlp":
        adaptor = TTTMLPResidualAdaptor(
            input_dim=dim,
            hidden_dim=bottleneck,
            dropout=float(config.adaptor_dropout),
            scale=float(config.adaptor_scale),
        )
    elif adaptor_block == "dcnv2":
        adaptor = TTTDCNv2ResidualAdaptor(
            input_dim=dim,
            cross_dim=bottleneck,
            dropout=float(config.adaptor_dropout),
            scale=float(config.adaptor_scale),
        )
    else:
        raise ValueError("--ttt-adaptor-block must be one of: mlp, dcnv2")
    first_param = next(model.parameters())
    adaptor.to(device=first_param.device, dtype=first_param.dtype)
    model.row_interactor.ttt_adaptor = adaptor
    model._ttt_adaptor_target = ADAPTOR_TARGET
    model._ttt_adaptor_block = adaptor_block

    def _row_output_hook(_module, _inputs, output):
        return output + _module.ttt_adaptor(output)

    model._ttt_adaptor_hook_handle = model.row_interactor.register_forward_hook(_row_output_hook)
    return adaptor


def _get_ttt_adaptor(classifier):
    model = _get_ttt_base_model(classifier)
    return getattr(model.row_interactor, "ttt_adaptor", None)


def _count_trainable_params(params: List[Any]) -> int:
    return int(sum(param.numel() for param in params if getattr(param, "requires_grad", False)))


def _configure_ttt_trainable_params(classifier, config: TTTConfig):
    model = _get_ttt_base_model(classifier)
    model.train()

    for param in model.parameters():
        param.requires_grad = False

    if config.adaptor_enabled:
        adaptor = _install_ttt_adaptor(classifier, config)
        if config.adaptor_train_backbone:
            if config.freeze_col:
                model.col_embedder.eval()
            else:
                model.col_embedder.train()
                _set_requires_grad(model.col_embedder, True)

            if config.freeze_row:
                model.row_interactor.eval()
            else:
                model.row_interactor.train()
                _set_requires_grad(model.row_interactor, True)

            if config.freeze_icl:
                model.icl_predictor.eval()
            else:
                model.icl_predictor.train()
                _set_requires_grad(model.icl_predictor, True)
        else:
            model.col_embedder.train()
            model.row_interactor.train()
            model.icl_predictor.train()

        adaptor.train()
        _set_requires_grad(adaptor, True)
        return [param for param in model.parameters() if param.requires_grad]

    if config.freeze_col:
        model.col_embedder.eval()
    else:
        model.col_embedder.train()
        _set_requires_grad(model.col_embedder, True)

    if config.freeze_row:
        model.row_interactor.eval()
    else:
        model.row_interactor.train()
        _set_requires_grad(model.row_interactor, True)

    if config.freeze_icl:
        model.icl_predictor.eval()
    else:
        model.icl_predictor.train()
        _set_requires_grad(model.icl_predictor, True)

    return [param for param in model.parameters() if param.requires_grad]


def _set_ttt_train_mode(classifier, config: TTTConfig) -> None:
    model = _get_ttt_base_model(classifier)
    model.train()
    adaptor = _get_ttt_adaptor(classifier)
    if adaptor is not None and not config.adaptor_train_backbone:
        model.col_embedder.train()
        model.row_interactor.train()
        model.icl_predictor.train()
        adaptor.train()
        return
    if config.freeze_col or not any(param.requires_grad for param in model.col_embedder.parameters()):
        model.col_embedder.eval()
    else:
        model.col_embedder.train()
    if config.freeze_row:
        model.row_interactor.eval()
    else:
        model.row_interactor.train()
    if config.freeze_icl or not any(param.requires_grad for param in model.icl_predictor.parameters()):
        model.icl_predictor.eval()
    else:
        model.icl_predictor.train()
    if adaptor is not None:
        adaptor.train()


def _fit_preserving_model_weights(classifier, X, y) -> None:
    original_load_model = getattr(classifier, "_load_model", None)

    def _skip_model_reload():
        return None

    try:
        classifier._load_model = _skip_model_reload
        classifier.fit(X, y)
    finally:
        if original_load_model is not None:
            classifier._load_model = original_load_model


def _optional_metric_to_float(value: Optional[float]) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _metric_is_valid(value: Optional[float]) -> bool:
    if value is None:
        return False
    try:
        return not bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _ttt_metric_improved(current: Optional[float], best: Optional[float], min_delta: float) -> bool:
    if not _metric_is_valid(current) or not _metric_is_valid(best):
        return False
    return float(current) > float(best) + float(min_delta)


def _evaluate_ttt_validation_metrics(
    classifier,
    X_context,
    y_context,
    X_val,
    y_val,
    config: TTTConfig,
) -> Optional[TTTValidationResult]:
    if X_val is None or y_val is None or len(y_val) == 0:
        return None

    import torch

    base_model = _get_ttt_base_model(classifier)
    original_n_estimators = getattr(classifier, "n_estimators", None)
    try:
        base_model.eval()
        if original_n_estimators is not None:
            classifier.n_estimators = min(int(original_n_estimators), int(config.validation_n_estimators))
        _fit_preserving_model_weights(classifier, X_context, y_context)
        with torch.inference_mode():
            proba = classifier.predict_proba(X_val)
        proba_arr = np.asarray(proba)
        classes = getattr(getattr(classifier, "y_encoder_", None), "classes_", None)
        class_arr = np.asarray(classes) if classes is not None else None
        if class_arr is not None and class_arr.ndim == 1 and len(class_arr) == proba_arr.shape[1]:
            y_pred = class_arr[proba_arr.argmax(axis=1)]
        else:
            y_pred = proba_arr.argmax(axis=1)

        roc_auc = _optional_metric_to_float(compute_tabpfn_roc_auc(y_val, proba_arr, class_arr))
        log_loss_score = _optional_metric_to_float(compute_log_loss(y_val, proba_arr, class_arr))
        accuracy = float(np.mean(np.asarray(y_pred) == np.asarray(y_val)))
        secondary = {
            "roc_auc": roc_auc,
            "log_loss": log_loss_score,
            "accuracy": accuracy,
        }
        if config.eval_metric == "roc_auc":
            primary = roc_auc
        elif config.eval_metric == "log_loss":
            primary = -log_loss_score if _metric_is_valid(log_loss_score) else float("nan")
        elif config.eval_metric == "accuracy":
            primary = accuracy
        else:
            raise ValueError(f"Unsupported TTT eval metric: {config.eval_metric!r}")
        return TTTValidationResult(primary=float(primary), secondary=secondary)
    except (ValueError, RuntimeError):
        return TTTValidationResult(primary=float("nan"), secondary={})
    finally:
        if original_n_estimators is not None:
            classifier.n_estimators = original_n_estimators


def _build_classification_meta_batch(
    classifier,
    X_chunk,
    y_chunk,
    *,
    config: TTTConfig,
    query_size: int,
    epoch_seed: int,
    chunk_idx: int,
) -> MetaBatch:
    try:
        from tabicl._sklearn.preprocessing import EnsembleGenerator
    except ImportError:
        from tabicl.sklearn.preprocessing import EnsembleGenerator

    if len(y_chunk) < 2:
        return MetaBatch(None, None, None, 0, "chunk has fewer than two samples")

    split_seed = int(epoch_seed + chunk_idx * 7919)
    n_classes_in_chunk = int(np.max(y_chunk)) + 1
    query_size = max(int(query_size), n_classes_in_chunk)
    ctx_idx, qry_idx, split_strategy = _split_ctx_query(y_chunk, query_size=query_size, seed=split_seed)
    y_ctx_raw = np.asarray(y_chunk)[ctx_idx].astype(int)
    y_qry_raw = np.asarray(y_chunk)[qry_idx].astype(int)

    missing_query_labels = sorted(set(y_qry_raw.tolist()) - set(y_ctx_raw.tolist()))
    if missing_query_labels:
        return MetaBatch(
            None,
            None,
            None,
            0,
            "query labels absent from context after "
            f"{split_strategy} split: {','.join(str(item) for item in missing_query_labels)}",
        )

    local_classes = np.asarray(sorted(set(y_ctx_raw.tolist())), dtype=np.int64)
    local_label_map = {int(label): idx for idx, label in enumerate(local_classes.tolist())}
    y_ctx = np.asarray([local_label_map[int(label)] for label in y_ctx_raw], dtype=np.int64)
    y_qry = np.asarray([local_label_map[int(label)] for label in y_qry_raw], dtype=np.int64)
    X_ctx = X_chunk[ctx_idx]
    X_qry = X_chunk[qry_idx]

    gen = EnsembleGenerator(
        classification=True,
        n_estimators=config.n_estimators_finetune,
        norm_methods=getattr(classifier, "norm_methods", None),
        feat_shuffle_method=getattr(classifier, "feat_shuffle_method", "latin"),
        class_shuffle_method=getattr(classifier, "class_shuffle_method", "shift"),
        outlier_threshold=getattr(classifier, "outlier_threshold", 4.0),
        random_state=config.random_state,
    )
    gen.fit(X_ctx, y_ctx)
    variants = gen.transform(X_qry, mode="both")

    X_list = []
    y_train_list = []
    y_query_list = []
    for norm_method, (X_variant, y_variant) in variants.items():
        X_list.append(X_variant)
        y_train_list.append(y_variant)
        shuffle_configs = gen.ensemble_configs_[norm_method]
        if len(shuffle_configs) != X_variant.shape[0]:
            raise RuntimeError(
                f"Ensemble data/class shuffle mismatch for norm_method={norm_method!r}: "
                f"{X_variant.shape[0]} views vs {len(shuffle_configs)} class shuffles"
            )
        for _feat_shuffle, class_shuffle in shuffle_configs:
            if class_shuffle is None:
                y_query_list.append(y_qry)
            else:
                y_query_list.append(np.asarray(class_shuffle, dtype=np.int64)[y_qry.astype(int)])

    import torch

    return MetaBatch(
        X=torch.from_numpy(np.concatenate(X_list, axis=0)).float(),
        y_train=torch.from_numpy(np.concatenate(y_train_list, axis=0)).float(),
        y_query=torch.from_numpy(np.stack(y_query_list, axis=0)).long(),
        train_size=int(len(ctx_idx)),
        skip_reason=None,
    )


def iter_epoch_meta_batches(
    classifier,
    X_encoded,
    y_encoded,
    *,
    config: TTTConfig,
    epoch_seed: int,
) -> Iterator[MetaBatch]:
    rng = np.random.default_rng(epoch_seed)
    chunks = _chunk_indices(
        len(y_encoded),
        max_chunk_size=config.max_chunk_size,
        min_chunk_size=config.min_chunk_size,
        rng=rng,
    )
    for chunk_idx, indices in enumerate(chunks):
        X_chunk = X_encoded[indices]
        y_chunk = y_encoded[indices]
        query_size = max(1, int(len(indices) * config.query_ratio))
        yield _build_classification_meta_batch(
            classifier,
            X_chunk,
            y_chunk,
            config=config,
            query_size=query_size,
            epoch_seed=epoch_seed,
            chunk_idx=chunk_idx,
        )


def move_meta_batch(batch: MetaBatch, device) -> MetaBatch:
    return MetaBatch(
        X=batch.X.to(device, non_blocking=True),
        y_train=batch.y_train.to(device, non_blocking=True),
        y_query=batch.y_query.to(device, non_blocking=True),
        train_size=batch.train_size,
        skip_reason=batch.skip_reason,
    )


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    sanitized = sanitized.strip("._")
    return sanitized or "unknown"


def _build_ttt_ckpt_path(config: TTTConfig, model_name: str, dataset_name: str, step_idx: int) -> Path:
    if not config.ckpt_root:
        raise ValueError("TTT checkpoint root is not configured")
    return (
        Path(config.ckpt_root)
        / _sanitize_path_component(model_name)
        / _sanitize_path_component(dataset_name)
        / f"step_{step_idx}.ckpt"
    )


def _save_ttt_model_ckpt(classifier, config: TTTConfig, model_name: str, dataset_name: str, step_idx: int) -> Path:
    import torch

    model_config = getattr(classifier, "model_config_", None)
    if model_config is None:
        raise RuntimeError("TTT checkpoint save requested before model_config_ is available")

    base_model = _get_ttt_base_model(classifier)
    ckpt_path = _build_ttt_ckpt_path(config, model_name, dataset_name, step_idx)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "config": dict(model_config),
        "state_dict": {name: tensor.detach().cpu() for name, tensor in base_model.state_dict().items()},
        "ttt_metadata": {
            "model_name": str(model_name),
            "dataset_name": str(dataset_name),
            "step": int(step_idx),
            "adaptor_enabled": bool(config.adaptor_enabled),
            "adaptor_block": str(config.adaptor_block).lower() if config.adaptor_enabled else None,
            "adaptor_target": ADAPTOR_TARGET if config.adaptor_enabled else None,
            "adaptor_bottleneck": int(config.adaptor_bottleneck) if config.adaptor_enabled else 0,
            "adaptor_dropout": float(config.adaptor_dropout) if config.adaptor_enabled else 0.0,
            "adaptor_scale": float(config.adaptor_scale) if config.adaptor_enabled else 0.0,
            "adaptor_train_backbone": bool(config.adaptor_train_backbone),
        },
    }
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


def _should_save_ttt_ckpt_step(config: TTTConfig, step_idx: int) -> bool:
    if not config.save_ckpt:
        return False
    if config.save_ckpt_start_step is None:
        return step_idx % config.save_ckpt_every == 0
    return (
        step_idx >= config.save_ckpt_start_step
        and (step_idx - config.save_ckpt_start_step) % config.save_ckpt_every == 0
    )


def _should_save_ttt_final_ckpt(config: TTTConfig, final_step: int) -> bool:
    if not config.save_ckpt:
        return False
    if config.save_ckpt_start_step is None:
        return True
    return final_step >= config.save_ckpt_start_step


def _derive_model_name(model_path: str | None, checkpoint_version: str | None) -> str:
    candidate = model_path if model_path else checkpoint_version
    if not candidate:
        return "tabicl_model"
    return Path(str(candidate)).stem


def _build_ttt_lr_scheduler(optimizer, config: TTTConfig, total_steps: int):
    try:
        from tabicl.train._optim import get_scheduler
    except ModuleNotFoundError as exc:
        if exc.name not in {"transformers", "tabicl.train", "tabicl.train._optim"}:
            raise
        return _build_fallback_ttt_lr_scheduler(optimizer, config, total_steps)

    sched_cfg = SimpleNamespace(
        max_steps=max(1, int(total_steps)),
        warmup_proportion=float(config.warmup_proportion),
        warmup_steps=0,
        scheduler=str(config.scheduler),
    )
    return get_scheduler(sched_cfg, optimizer)


def _build_fallback_ttt_lr_scheduler(optimizer, config: TTTConfig, total_steps: int):
    import math
    import torch

    total_steps = max(1, int(total_steps))
    warmup_steps = float(total_steps) * float(config.warmup_proportion)

    if config.scheduler == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)

    if config.scheduler != "cosine_warmup":
        raise ValueError("--ttt-scheduler must be one of: constant, cosine_warmup")

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1.0, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1.0, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_ttt_epoch_chunk_update(
    classifier,
    X_train,
    y_train,
    X_val,
    y_val,
    config: TTTConfig,
    *,
    model_name: str,
    dataset_name: str,
) -> TTTUpdateResult:
    ensure_runtime_deps()

    if config.scheduler not in {"constant", "cosine_warmup"}:
        raise ValueError("--ttt-scheduler must be one of: constant, cosine_warmup")
    if config.epochs < 1:
        return TTTUpdateResult(
            applied=False,
            loss=None,
            steps=0,
            update_seconds=0.0,
            reason="--ttt-epochs must be >= 1",
            epochs=0,
            chunks_per_epoch=0,
        )
    if config.micro_batch_size < 1:
        raise ValueError("--ttt-micro-batch-size must be >= 1")

    update_start = time.time()
    classifier.fit(X_train, y_train)
    if classifier.n_classes_ > classifier.model_.max_classes:
        return TTTUpdateResult(
            applied=False,
            loss=None,
            steps=0,
            update_seconds=time.time() - update_start,
            reason=(
                f"TTT training skipped because n_classes={classifier.n_classes_} "
                f"exceeds model max_classes={classifier.model_.max_classes}"
            ),
            epochs=0,
            chunks_per_epoch=0,
        )

    import torch
    import torch.nn.functional as F

    base_model = _get_ttt_base_model(classifier)
    trainable_params = _configure_ttt_trainable_params(classifier, config)
    trainable_param_count = _count_trainable_params(trainable_params)
    if not trainable_params:
        return TTTUpdateResult(
            applied=False,
            loss=None,
            steps=0,
            update_seconds=time.time() - update_start,
            reason="No trainable parameters selected for TTT",
            epochs=0,
            chunks_per_epoch=0,
            adaptor_enabled=bool(config.adaptor_enabled),
            adaptor_block=str(config.adaptor_block).lower() if config.adaptor_enabled else None,
            adaptor_bottleneck=int(config.adaptor_bottleneck) if config.adaptor_enabled else 0,
            adaptor_trainable_params=0,
            adaptor_target=ADAPTOR_TARGET if config.adaptor_enabled else None,
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    forward_model, data_parallel_world_size = _build_ttt_forward_model(classifier, config)
    effective_micro_batch_size = config.micro_batch_size * data_parallel_world_size

    X_encoded = classifier.X_encoder_.transform(X_train)
    y_encoded = classifier.y_encoder_.transform(y_train)
    chunks_per_epoch = count_ttt_chunks(
        int(len(y_encoded)),
        max_chunk_size=config.max_chunk_size,
        min_chunk_size=config.min_chunk_size,
    )
    scheduler = _build_ttt_lr_scheduler(
        optimizer,
        config,
        total_steps=max(1, config.epochs * chunks_per_epoch),
    )

    device = classifier.device_
    use_amp, scaler, amp_ctx_factory = _make_ttt_amp(config, device)
    if str(config.dtype).lower() != "float32":
        print(
            f"[ttt-amp] model={model_name} dataset={dataset_name} "
            f"--ttt-dtype={config.dtype} is ignored; TTT AMP follows --use-amp "
            f"and uses float16 autocast on CUDA. use_amp={use_amp}",
            flush=True,
        )

    last_loss = None
    update_steps = 0
    skipped_batches = 0
    skip_reasons: Dict[str, int] = {}
    baseline_metric: Optional[float] = None
    best_metric: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    best_accuracy: Optional[float] = None
    best_epoch = 0
    best_state = None
    patience_counter = 0
    stopped_early = False
    try:
        if X_encoded.shape[0] < 2 or chunks_per_epoch == 0:
            return TTTUpdateResult(
                applied=False,
                loss=None,
                steps=0,
                update_seconds=time.time() - update_start,
                reason="Need at least two encoded training samples for chunk TTT",
                epochs=0,
                chunks_per_epoch=chunks_per_epoch,
            )

        if config.early_stopping and X_val is not None and y_val is not None and len(y_val) > 0:
            baseline_result = _evaluate_ttt_validation_metrics(classifier, X_train, y_train, X_val, y_val, config)
            if baseline_result is not None:
                baseline_metric = float(baseline_result.primary)
                best_metric = float(baseline_result.primary)
                baseline_accuracy = baseline_result.secondary.get("accuracy")
                best_accuracy = baseline_accuracy
                best_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
                print(
                    f"[ttt-val] model={model_name} dataset={dataset_name} "
                    f"baseline_{config.eval_metric}={best_metric:.6f}",
                    flush=True,
                )

        last_saved_step = 0
        for epoch_idx in range(config.epochs):
            _set_ttt_train_mode(classifier, config)
            epoch_seed = config.random_state + epoch_idx
            epoch_loss_sum = 0.0
            epoch_updates = 0
            for batch in iter_epoch_meta_batches(
                classifier,
                X_encoded,
                y_encoded,
                config=config,
                epoch_seed=epoch_seed,
            ):
                if batch.skip_reason:
                    skipped_batches += 1
                    skip_reasons[batch.skip_reason] = skip_reasons.get(batch.skip_reason, 0) + 1
                    continue

                batch = move_meta_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                total_views = int(batch.X.shape[0])

                for start_idx in range(0, total_views, effective_micro_batch_size):
                    end_idx = min(start_idx + effective_micro_batch_size, total_views)
                    X_batch = batch.X[start_idx:end_idx]
                    y_train_batch = batch.y_train[start_idx:end_idx]
                    y_query_batch = batch.y_query[start_idx:end_idx]
                    with amp_ctx_factory():
                        logits = forward_model(X_batch, y_train_batch.float())
                        n_classes = int(y_train_batch.max().item()) + 1
                        logits_used = logits[..., :n_classes].reshape(-1, n_classes)
                        loss = F.cross_entropy(logits_used, y_query_batch.long().reshape(-1))
                        scaled_loss = loss * (X_batch.shape[0] / total_views)

                    scaler.scale(scaled_loss).backward()
                    batch_loss += float(scaled_loss.detach().cpu())

                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                update_steps += 1
                epoch_updates += 1
                epoch_loss_sum += batch_loss
                last_loss = batch_loss
                current_lr = scheduler.get_last_lr()[0]

                if update_steps % 3 == 0:
                    print(
                        f"[ttt-loss] model={model_name} dataset={dataset_name} "
                        f"epoch={epoch_idx + 1}/{config.epochs} step={update_steps} "
                        f"loss={batch_loss:.6f} lr={current_lr:.2e}",
                        flush=True,
                    )
                if _should_save_ttt_ckpt_step(config, update_steps):
                    ckpt_path = _save_ttt_model_ckpt(classifier, config, model_name, dataset_name, update_steps)
                    last_saved_step = update_steps
                    print(
                        f"[ttt-ckpt] saved model={model_name} dataset={dataset_name} "
                        f"step={update_steps} path={ckpt_path}",
                        flush=True,
                    )

            if epoch_updates > 0:
                print(
                    f"[ttt-loss] model={model_name} dataset={dataset_name} "
                    f"epoch={epoch_idx + 1}/{config.epochs} "
                    f"mean_loss={epoch_loss_sum / epoch_updates:.6f} "
                    f"updates={epoch_updates} lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

            if best_state is not None:
                val_result = _evaluate_ttt_validation_metrics(classifier, X_train, y_train, X_val, y_val, config)
                if val_result is not None:
                    val_metric = float(val_result.primary)
                    improved = _ttt_metric_improved(val_metric, best_metric, config.min_delta)
                    if improved:
                        best_metric = val_metric
                        best_accuracy = val_result.secondary.get("accuracy")
                        best_epoch = epoch_idx + 1
                        patience_counter = 0
                        best_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
                    elif _metric_is_valid(val_metric):
                        patience_counter += 1
                    print(
                        f"[ttt-val] model={model_name} dataset={dataset_name} "
                        f"epoch={epoch_idx + 1}/{config.epochs} "
                        f"{config.eval_metric}={val_metric:.6f} best={best_metric:.6f} "
                        f"patience={patience_counter}/{config.patience}",
                        flush=True,
                    )
                    if patience_counter >= config.patience:
                        stopped_early = True
                        print(
                            f"[ttt-early-stop] model={model_name} dataset={dataset_name} "
                            f"epoch={epoch_idx + 1} best_epoch={best_epoch} "
                            f"best_{config.eval_metric}={best_metric:.6f}",
                            flush=True,
                        )
                        break

        if update_steps == 0:
            reason = "No valid epoch chunks produced an optimizer update"
            if skip_reasons:
                top_reasons = sorted(skip_reasons.items(), key=lambda item: (-item[1], item[0]))[:3]
                reason += "; skipped=" + "; ".join(f"{count}x {text}" for text, count in top_reasons)
            return TTTUpdateResult(
                applied=False,
                loss=None,
                steps=0,
                update_seconds=time.time() - update_start,
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
                adaptor_enabled=bool(config.adaptor_enabled),
                adaptor_block=str(config.adaptor_block).lower() if config.adaptor_enabled else None,
                adaptor_bottleneck=int(config.adaptor_bottleneck) if config.adaptor_enabled else 0,
                adaptor_trainable_params=trainable_param_count,
                adaptor_target=ADAPTOR_TARGET if config.adaptor_enabled else None,
            )

        if best_state is not None:
            base_model.load_state_dict(best_state)

        if _should_save_ttt_final_ckpt(config, update_steps) and last_saved_step != update_steps:
            ckpt_path = _save_ttt_model_ckpt(classifier, config, model_name, dataset_name, update_steps)
            print(
                f"[ttt-ckpt] saved model={model_name} dataset={dataset_name} "
                f"step={update_steps} path={ckpt_path}",
                flush=True,
            )
    finally:
        if hasattr(base_model, "clear_cache"):
            base_model.clear_cache()
        classifier.model_kv_cache_ = None
        base_model.eval()

    return TTTUpdateResult(
        applied=True,
        loss=last_loss,
        steps=update_steps,
        update_seconds=time.time() - update_start,
        reason=None,
        epochs=config.epochs,
        chunks_per_epoch=chunks_per_epoch,
        val_eval_metric=config.eval_metric,
        val_baseline_metric=baseline_metric,
        val_best_metric=best_metric,
        val_baseline_accuracy=baseline_accuracy,
        val_best_accuracy=best_accuracy,
        best_epoch=best_epoch,
        stopped_early=stopped_early,
        adaptor_enabled=bool(config.adaptor_enabled),
        adaptor_block=str(config.adaptor_block).lower() if config.adaptor_enabled else None,
        adaptor_bottleneck=int(config.adaptor_bottleneck) if config.adaptor_enabled else 0,
        adaptor_trainable_params=trainable_param_count,
        adaptor_target=ADAPTOR_TARGET if config.adaptor_enabled else None,
    )


def build_model_summary_row(
    model_path: Path,
    gpu_id: int,
    dataset_dirs: List[Path],
    rows: List[ResultRow],
    model_wall_seconds: float,
    error: str | None = None,
) -> ModelSummaryRow:
    ensure_runtime_deps()

    result_df = (
        pd.DataFrame([asdict(row) for row in rows])
        if rows
        else pd.DataFrame(columns=ResultRow.__annotations__.keys())
    )
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()
    skipped_df = result_df[result_df["status"] == "skip"].copy() if len(result_df) else pd.DataFrame()
    oom_fallback_count = int(truthy_column_mask(ok_df, "ttt_oom_fallback").sum()) if len(ok_df) else 0
    adaptor_enabled_count = int(truthy_column_mask(ok_df, "ttt_adaptor_enabled").sum()) if len(ok_df) else 0
    adaptor_block_counts = format_adaptor_block_counts(
        ok_df[truthy_column_mask(ok_df, "ttt_adaptor_enabled")].copy()
        if len(ok_df)
        else pd.DataFrame()
    )

    avg_fit_seconds_ok = float(ok_df["fit_seconds"].mean()) if len(ok_df) else None
    avg_predict_seconds_ok = float(ok_df["predict_seconds"].mean()) if len(ok_df) else None
    total_dataset_seconds_ok = (
        float((ok_df["fit_seconds"] + ok_df["predict_seconds"]).sum())
        if len(ok_df)
        else 0.0
    )
    avg_dataset_seconds_ok = (
        float((ok_df["fit_seconds"] + ok_df["predict_seconds"]).mean())
        if len(ok_df)
        else None
    )
    failed_datasets = ",".join(failed_df["dataset_name"].astype(str).tolist()) if len(failed_df) else ""

    if error is not None:
        return ModelSummaryRow(
            model_name=model_path.stem,
            model_path=model_path.as_posix(),
            gpu_id=gpu_id,
            datasets_discovered=len(dataset_dirs),
            ok_count=0,
            failed_count=len(dataset_dirs),
            skipped_count=0,
            ttt_oom_fallback_count=0,
            avg_accuracy_ok=None,
            avg_f1_ok=None,
            avg_balanced_accuracy_ok=None,
            avg_roc_auc_ok=None,
            avg_log_loss_ok=None,
            avg_fit_seconds_ok=None,
            avg_predict_seconds_ok=None,
            avg_dataset_seconds_ok=None,
            total_dataset_seconds_ok=0.0,
            model_wall_seconds=float(model_wall_seconds),
            status="fail",
            error=error,
            failed_datasets=",".join(path.name for path in dataset_dirs),
            ttt_adaptor_enabled_count=0,
            ttt_adaptor_block_counts="(none)",
            avg_ttt_adaptor_trainable_params_ok=None,
        )

    return ModelSummaryRow(
        model_name=model_path.stem,
        model_path=model_path.as_posix(),
        gpu_id=gpu_id,
        datasets_discovered=len(dataset_dirs),
        ok_count=int(len(ok_df)),
        failed_count=int(len(failed_df)),
        skipped_count=int(len(skipped_df)),
        ttt_oom_fallback_count=oom_fallback_count,
        avg_accuracy_ok=(float(ok_df["accuracy"].mean()) if len(ok_df) else None),
        avg_f1_ok=(float(ok_df["f1"].mean()) if len(ok_df) else None),
        avg_balanced_accuracy_ok=(
            float(ok_df["balanced_accuracy"].mean())
            if len(ok_df) and ok_df["balanced_accuracy"].notna().any()
            else None
        ),
        avg_roc_auc_ok=(
            float(ok_df["roc_auc"].mean())
            if len(ok_df) and ok_df["roc_auc"].notna().any()
            else None
        ),
        avg_log_loss_ok=(
            float(ok_df["log_loss"].mean())
            if len(ok_df) and ok_df["log_loss"].notna().any()
            else None
        ),
        avg_fit_seconds_ok=avg_fit_seconds_ok,
        avg_predict_seconds_ok=avg_predict_seconds_ok,
        avg_dataset_seconds_ok=avg_dataset_seconds_ok,
        total_dataset_seconds_ok=total_dataset_seconds_ok,
        model_wall_seconds=float(model_wall_seconds),
        status="ok" if len(ok_df) else "fail",
        error=None if len(ok_df) else "No successful datasets processed",
        failed_datasets=failed_datasets,
        ttt_adaptor_enabled_count=adaptor_enabled_count,
        ttt_adaptor_block_counts=adaptor_block_counts,
        avg_ttt_adaptor_trainable_params_ok=(
            float(pd.to_numeric(ok_df["ttt_adaptor_trainable_params"], errors="coerce").mean())
            if len(ok_df)
            and "ttt_adaptor_trainable_params" in ok_df.columns
            and pd.to_numeric(ok_df["ttt_adaptor_trainable_params"], errors="coerce").notna().any()
            else None
        ),
    )


class BackgroundPrefetcher:
    def __init__(self, enabled: bool, verbose: bool) -> None:
        self.enabled = bool(enabled)
        self.verbose = verbose
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._seen: set[str] = set()
        self._thread: threading.Thread | None = None

        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def schedule(self, model_paths: List[Path | str]) -> None:
        if not self.enabled:
            return

        for model_path in model_paths:
            key = str(Path(model_path).resolve())
            if key in self._seen:
                continue
            self._seen.add(key)
            self._queue.put(key)

    def close(self) -> None:
        if not self.enabled:
            return
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            model_path = self._queue.get()
            if model_path is None:
                return

            try:
                with open(model_path, "rb") as handle:
                    while handle.read(8 * 1024 * 1024):
                        pass
                if self.verbose:
                    print(f"[prefetch] warmed page cache for {model_path}", flush=True)
            except Exception as exc:
                print(f"[prefetch] warning: failed to warm {model_path}: {exc}", flush=True)


def evaluate_one_dataset(
    classifier,
    dataset_dir: Path,
    ttt_config: TTTConfig | None = None,
    *,
    model_name: str,
) -> ResultRow:
    ensure_runtime_deps()
    if ttt_config is None:
        ttt_config = TTTConfig()

    task_type: Optional[str] = None
    try:
        info = load_dataset_info(dataset_dir)
        task_type = str(info.get("task_type", "")).lower() if info else None
        if task_type not in CLASSIFICATION_TASKS:
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
                status="skip",
                error=f"Skipped due to task_type={task_type!r}",
            )

        train_split, val_split, test_split = find_split_files(dataset_dir)

        X_train, y_train = load_split(
            train_split[0],
            train_split[1],
            train_split[2],
            context=f"{dataset_dir.name}-train",
        )
        val_count = 0
        X_ttt_train = X_train
        y_ttt_train = np.asarray(y_train)
        X_ttt_val = None
        y_ttt_val = None
        ttt_validation_reason = None
        if val_split is not None:
            X_val, y_val = load_split(
                val_split[0],
                val_split[1],
                val_split[2],
                context=f"{dataset_dir.name}-val",
            )
            val_count = int(len(y_val))
            X_ttt_val = X_val
            y_ttt_val = np.asarray(y_val)
            ttt_validation_reason = "dataset_val_split"
            X_train = pd.concat([X_train, X_val], axis=0, ignore_index=True)
            y_train = np.concatenate([np.asarray(y_train), np.asarray(y_val)], axis=0)
        elif ttt_config.enabled and ttt_config.early_stopping:
            X_ttt_train, y_ttt_train, X_ttt_val, y_ttt_val, ttt_validation_reason = split_ttt_validation(
                X_train,
                y_train,
                validation_fraction=ttt_config.validation_fraction,
                random_state=ttt_config.random_state,
            )
            val_count = int(len(y_ttt_val)) if y_ttt_val is not None else 0

        X_test, y_test = load_split(
            test_split[0],
            test_split[1],
            test_split[2],
            context=f"{dataset_dir.name}-test",
        )

        classes = pd.unique(pd.Series(np.concatenate([np.asarray(y_train), np.asarray(y_test)], axis=0)))

        ttt_loss = None
        ttt_steps = 0
        ttt_lr = ttt_config.lr if ttt_config.enabled else None
        ttt_applied = False
        ttt_update_seconds = 0.0
        ttt_split_strategy = None
        ttt_split_reason = None
        ttt_epochs = 0
        ttt_chunks_per_epoch = 0
        ttt_batch_mode = None
        ttt_val_eval_metric = None
        ttt_val_baseline_metric = None
        ttt_val_best_metric = None
        ttt_val_baseline_accuracy = None
        ttt_val_best_accuracy = None
        ttt_best_epoch = 0
        ttt_stopped_early = False
        ttt_oom_fallback = False
        ttt_fallback_reason = None
        ttt_adaptor_enabled = bool(ttt_config.enabled and ttt_config.adaptor_enabled)
        ttt_adaptor_block = str(ttt_config.adaptor_block).lower() if ttt_adaptor_enabled else None
        ttt_adaptor_bottleneck = int(ttt_config.adaptor_bottleneck) if ttt_adaptor_enabled else 0
        ttt_adaptor_trainable_params = 0
        ttt_adaptor_target = ADAPTOR_TARGET if ttt_adaptor_enabled else None
        n_train_b = 0
        n_holdout_c = 0

        t0 = time.time()
        if ttt_config.enabled:
            if should_skip_ttt_for_dataset(dataset_dir, info):
                ttt_split_reason = "TTT skipped for dataset=volkert to avoid OOM"
                classifier.fit(X_train, y_train)
            else:
                ttt_split_strategy = "full_train_epoch_chunks"
                ttt_split_reason = "full train set chunked per epoch"
                if ttt_validation_reason:
                    ttt_split_reason += f" | validation={ttt_validation_reason}"
                n_train_b = int(len(y_ttt_train))
                n_holdout_c = 0
                ttt_attempt_start = time.time()
                try:
                    ttt_result = run_ttt_epoch_chunk_update(
                        classifier,
                        X_ttt_train,
                        y_ttt_train,
                        X_ttt_val,
                        y_ttt_val,
                        ttt_config,
                        model_name=model_name,
                        dataset_name=dataset_dir.name,
                    )
                except Exception as ttt_exc:
                    if not is_oom_exception(ttt_exc):
                        raise
                    ttt_oom_fallback = True
                    ttt_update_seconds = time.time() - ttt_attempt_start
                    ttt_fallback_reason = (
                        "TTT OOM; used original model parameters for inference: "
                        f"{format_exception_for_csv(ttt_exc)}"
                    )
                    ttt_split_reason = append_ttt_reason(ttt_split_reason, ttt_fallback_reason)
                    force_memory_cleanup(str(getattr(classifier, "device_", "cuda:0")))
                    classifier.fit(X_train, y_train)
                else:
                    ttt_loss = ttt_result.loss
                    ttt_steps = ttt_result.steps
                    ttt_applied = ttt_result.applied
                    ttt_update_seconds = float(ttt_result.update_seconds)
                    ttt_epochs = ttt_result.epochs
                    ttt_chunks_per_epoch = ttt_result.chunks_per_epoch
                    ttt_batch_mode = ttt_result.batch_mode
                    ttt_val_eval_metric = ttt_result.val_eval_metric
                    ttt_val_baseline_metric = ttt_result.val_baseline_metric
                    ttt_val_best_metric = ttt_result.val_best_metric
                    ttt_val_baseline_accuracy = ttt_result.val_baseline_accuracy
                    ttt_val_best_accuracy = ttt_result.val_best_accuracy
                    ttt_best_epoch = ttt_result.best_epoch
                    ttt_stopped_early = ttt_result.stopped_early
                    ttt_adaptor_enabled = ttt_result.adaptor_enabled
                    ttt_adaptor_block = ttt_result.adaptor_block
                    ttt_adaptor_bottleneck = ttt_result.adaptor_bottleneck
                    ttt_adaptor_trainable_params = ttt_result.adaptor_trainable_params
                    ttt_adaptor_target = ttt_result.adaptor_target
                    if ttt_result.reason:
                        ttt_split_reason = append_ttt_reason(ttt_split_reason, ttt_result.reason)

                    if ttt_applied:
                        _fit_preserving_model_weights(classifier, X_train, y_train)
                    else:
                        classifier.fit(X_train, y_train)
        else:
            classifier.fit(X_train, y_train)
        fit_seconds = time.time() - t0

        t1 = time.time()
        y_proba = classifier.predict_proba(X_test)
        y_pred_encoded = np.argmax(np.asarray(y_proba), axis=1)
        y_pred = classifier.y_encoder_.inverse_transform(y_pred_encoded)
        proba_classes = getattr(getattr(classifier, "y_encoder_", None), "classes_", None)
        roc_auc = compute_tabpfn_roc_auc(y_test, y_proba, proba_classes)
        log_loss_score = compute_log_loss(y_test, y_proba, proba_classes)
        predict_seconds = time.time() - t1

        accuracy = float(np.mean(np.asarray(y_pred) == np.asarray(y_test)))
        f1 = compute_weighted_f1(y_test, y_pred)
        balanced_accuracy = compute_balanced_accuracy(y_test, y_pred)

        return ResultRow(
            dataset_name=dataset_dir.name,
            dataset_dir=dataset_dir.as_posix(),
            task_type=task_type,
            n_train=int(len(y_train)),
            n_val=val_count,
            n_test=int(len(y_test)),
            n_features=int(X_train.shape[1]),
            n_classes=int(len(classes)),
            accuracy=accuracy,
            f1=f1,
            balanced_accuracy=balanced_accuracy,
            roc_auc=roc_auc,
            log_loss=log_loss_score,
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            status="ok",
            error=None,
            n_train_a=int(len(y_train)),
            n_train_b=n_train_b,
            n_holdout_c=n_holdout_c,
            n_test_d=int(len(y_test)),
            ttt_loss=ttt_loss,
            ttt_steps=ttt_steps,
            ttt_lr=ttt_lr,
            ttt_applied=ttt_applied,
            ttt_update_seconds=ttt_update_seconds,
            ttt_split_strategy=ttt_split_strategy,
            ttt_split_reason=ttt_split_reason,
            ttt_epochs=ttt_epochs,
            ttt_chunks_per_epoch=ttt_chunks_per_epoch,
            ttt_batch_mode=ttt_batch_mode,
            ttt_val_eval_metric=ttt_val_eval_metric,
            ttt_val_baseline_metric=ttt_val_baseline_metric,
            ttt_val_best_metric=ttt_val_best_metric,
            ttt_val_baseline_accuracy=ttt_val_baseline_accuracy,
            ttt_val_best_accuracy=ttt_val_best_accuracy,
            ttt_best_epoch=ttt_best_epoch,
            ttt_stopped_early=ttt_stopped_early,
            ttt_oom_fallback=ttt_oom_fallback,
            ttt_fallback_reason=ttt_fallback_reason,
            ttt_adaptor_enabled=ttt_adaptor_enabled,
            ttt_adaptor_block=ttt_adaptor_block,
            ttt_adaptor_bottleneck=ttt_adaptor_bottleneck,
            ttt_adaptor_trainable_params=ttt_adaptor_trainable_params,
            ttt_adaptor_target=ttt_adaptor_target,
        )
    except Exception as exc:
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
            status="fail",
            error=f"{type(exc).__name__}: {exc}",
        )


def worker_main(
    worker_id: int,
    gpu_id: int,
    gpu_group: str,
    assigned_dataset_dirs: List[str],
    ready_queue,
    start_event,
    worker_out_csv: str,
    model_kwargs: Dict,
    ttt_config: TTTConfig,
    verbose: bool,
) -> None:
    try:
        ensure_runtime_deps()
        device_str = apply_worker_environment_updates(gpu_group)
        worker_label = f"worker {worker_id} | gpu {gpu_group}"
        worker_ttt_config = replace(ttt_config, gpu_group=gpu_group)

        import torch
        from tabicl import TabICLClassifier

        torch_diag = collect_torch_diagnostics()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU backend is not available in this worker. "
                f"Diagnostics: {json.dumps(torch_diag, ensure_ascii=False)}"
            )

        worker_kwargs = dict(model_kwargs)
        worker_kwargs["device"] = device_str
        classifier = TabICLClassifier(**worker_kwargs)
        model_name = _derive_model_name(
            str(worker_kwargs.get("model_path")) if worker_kwargs.get("model_path") is not None else None,
            str(worker_kwargs.get("checkpoint_version")) if worker_kwargs.get("checkpoint_version") is not None else None,
        )

        ready_queue.put(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "gpu_group": gpu_group,
                "status": "ready",
                "assigned_count": len(assigned_dataset_dirs),
            }
        )
        start_event.wait()

        rows: List[ResultRow] = []
        for dataset_dir in assigned_dataset_dirs:
            row = evaluate_one_dataset(
                classifier,
                Path(dataset_dir),
                worker_ttt_config,
                model_name=model_name,
            )
            rows.append(row)
            print(
                format_dataset_result_log(
                    worker_label,
                    row,
                ),
                flush=True,
            )

        pd.DataFrame([asdict(row) for row in rows]).to_csv(worker_out_csv, index=False)
    except Exception:
        try:
            ready_queue.put(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "status": "crash",
                    "error": traceback.format_exc(),
                }
            )
        except Exception:
            pass

        ensure_runtime_deps()
        crash_row = pd.DataFrame(
            [
                asdict(
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
                        f1=None,
                        balanced_accuracy=None,
                        roc_auc=None,
                        log_loss=None,
                        fit_seconds=0.0,
                        predict_seconds=0.0,
                        status="fail",
                        error=traceback.format_exc(),
                    )
                )
            ]
        )
        crash_row.to_csv(worker_out_csv, index=False)


def model_pool_worker_main(
    worker_id: int,
    gpu_id: int,
    gpu_group: str,
    dataset_dirs: List[str],
    ready_queue,
    task_queue,
    result_queue,
    base_model_kwargs: Dict,
    ttt_config: TTTConfig,
    verbose: bool,
) -> None:
    device_str = "cuda:0"
    worker_label = f"worker {worker_id} | gpu {gpu_group}"

    try:
        ensure_runtime_deps()
        device_str = apply_worker_environment_updates(gpu_group)
        worker_ttt_config = replace(ttt_config, gpu_group=gpu_group)

        import torch
        from tabicl import TabICLClassifier

        torch_diag = collect_torch_diagnostics()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU backend is not available in this worker. "
                f"Diagnostics: {json.dumps(torch_diag, ensure_ascii=False)}"
            )

        ready_queue.put(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "gpu_group": gpu_group,
                "status": "ready",
                "assigned_count": len(dataset_dirs),
            }
        )

        resolved_dataset_dirs = [Path(item) for item in dataset_dirs]
        while True:
            task = task_queue.get()
            if task is None:
                return

            model_path = Path(str(task["model_path"]))
            started_at = time.time()
            classifier = None

            try:
                worker_kwargs = dict(base_model_kwargs)
                worker_kwargs["device"] = device_str
                worker_kwargs["model_path"] = normalize_model_path(str(model_path))
                classifier = TabICLClassifier(**worker_kwargs)
                model_name = _derive_model_name(str(model_path), worker_kwargs.get("checkpoint_version"))
                if not worker_ttt_config.enabled:
                    preload_model_once(classifier, worker_label, verbose)

                rows: List[ResultRow] = []
                for dataset_dir in resolved_dataset_dirs:
                    row = evaluate_one_dataset(
                        classifier,
                        dataset_dir,
                        worker_ttt_config,
                        model_name=model_name,
                    )
                    rows.append(row)
                    print(
                        format_dataset_result_log(
                            worker_label,
                            row,
                            model_name=model_path.stem,
                        ),
                        flush=True,
                    )

                summary = build_model_summary_row(
                    model_path=model_path,
                    gpu_id=gpu_id,
                    dataset_dirs=resolved_dataset_dirs,
                    rows=rows,
                    model_wall_seconds=time.time() - started_at,
                )
                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "status": "model_done",
                        "summary": asdict(summary),
                    }
                )
            except Exception as exc:
                summary = build_model_summary_row(
                    model_path=model_path,
                    gpu_id=gpu_id,
                    dataset_dirs=resolved_dataset_dirs,
                    rows=[],
                    model_wall_seconds=time.time() - started_at,
                    error=f"{type(exc).__name__}: {exc}",
                )
                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "status": "model_done",
                        "summary": asdict(summary),
                    }
                )
            finally:
                release_classifier_resources(classifier)
                force_memory_cleanup(device_str)
    except Exception:
        try:
            ready_queue.put(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "status": "crash",
                    "error": traceback.format_exc(),
                }
            )
        except Exception:
            pass

        try:
            result_queue.put(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "status": "worker_crash",
                    "error": traceback.format_exc(),
                }
            )
        except Exception:
            pass


def write_summary(
    summary_path: Path,
    result_df: pd.DataFrame,
    dataset_dirs: List[Path],
    wall_seconds: float,
) -> None:
    ensure_runtime_deps()

    result_df = result_df.copy()
    for metric_column in ("accuracy", "f1", "balanced_accuracy", "roc_auc", "log_loss"):
        if metric_column in result_df.columns:
            result_df[metric_column] = pd.to_numeric(result_df[metric_column], errors="coerce")
    if "ttt_adaptor_trainable_params" in result_df.columns:
        result_df["ttt_adaptor_trainable_params"] = pd.to_numeric(
            result_df["ttt_adaptor_trainable_params"],
            errors="coerce",
        )

    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()
    skipped_df = result_df[result_df["status"] == "skip"].copy() if len(result_df) else pd.DataFrame()
    oom_fallback_df = (
        ok_df[truthy_column_mask(ok_df, "ttt_oom_fallback")].copy()
        if len(ok_df)
        else pd.DataFrame()
    )
    adaptor_enabled_df = (
        ok_df[truthy_column_mask(ok_df, "ttt_adaptor_enabled")].copy()
        if len(ok_df)
        else pd.DataFrame()
    )
    adaptor_block_counts = format_adaptor_block_counts(adaptor_enabled_df)

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
        f"ttt_adaptor_enabled_count: {len(adaptor_enabled_df)}",
        f"ttt_adaptor_block_counts: {adaptor_block_counts}",
        mean_line("avg_accuracy_ok", "accuracy"),
        mean_line("avg_f1_ok", "f1"),
        mean_line("avg_balanced_accuracy_ok", "balanced_accuracy"),
        mean_line("avg_roc_auc_ok", "roc_auc"),
        mean_line("avg_log_loss_ok", "log_loss"),
        mean_line("avg_ttt_adaptor_trainable_params_ok", "ttt_adaptor_trainable_params"),
        f"wall_seconds: {wall_seconds:.3f}",
    ]

    if len(failed_df):
        failed_names = ", ".join(failed_df["dataset_name"].astype(str).tolist())
        lines.append(f"failed_datasets: {failed_names}")
    else:
        lines.append("failed_datasets: (none)")

    if len(skipped_df):
        skipped_names = ", ".join(skipped_df["dataset_name"].astype(str).tolist())
        lines.append(f"skipped_datasets: {skipped_names}")
    else:
        lines.append("skipped_datasets: (none)")

    if len(oom_fallback_df):
        oom_fallback_names = ", ".join(oom_fallback_df["dataset_name"].astype(str).tolist())
        lines.append(f"ttt_oom_fallback_datasets: {oom_fallback_names}")
    else:
        lines.append("ttt_oom_fallback_datasets: (none)")

    if len(adaptor_enabled_df):
        adaptor_names = ", ".join(adaptor_enabled_df["dataset_name"].astype(str).tolist())
        lines.append(f"ttt_adaptor_enabled_datasets: {adaptor_names}")
    else:
        lines.append("ttt_adaptor_enabled_datasets: (none)")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_pool_outputs(
    out_dir: Path,
    model_summaries: List[dict[str, Any]],
    wall_seconds: float,
) -> None:
    ensure_runtime_deps()

    summary_df = (
        pd.DataFrame(model_summaries)
        if model_summaries
        else pd.DataFrame(columns=ModelSummaryRow.__annotations__.keys())
    )
    for column in (
        "avg_accuracy_ok",
        "avg_f1_ok",
        "avg_balanced_accuracy_ok",
        "avg_roc_auc_ok",
        "avg_log_loss_ok",
        "avg_fit_seconds_ok",
        "avg_predict_seconds_ok",
        "avg_dataset_seconds_ok",
        "total_dataset_seconds_ok",
        "model_wall_seconds",
        "ttt_oom_fallback_count",
        "ttt_adaptor_enabled_count",
        "avg_ttt_adaptor_trainable_params_ok",
    ):
        if column in summary_df.columns:
            summary_df[column] = pd.to_numeric(summary_df[column], errors="coerce")
    all_csv = out_dir / "all_models_summary.csv"
    summary_txt = out_dir / "summary.txt"
    summary_df.to_csv(all_csv, index=False)

    ok_df = summary_df[summary_df["status"] == "ok"].copy() if len(summary_df) else pd.DataFrame()
    failed_df = summary_df[summary_df["status"] == "fail"].copy() if len(summary_df) else pd.DataFrame()
    adaptor_block_counts = (
        format_adaptor_block_count_totals(ok_df["ttt_adaptor_block_counts"])
        if len(ok_df) and "ttt_adaptor_block_counts" in ok_df.columns
        else "(none)"
    )

    def mean_line(label: str, column: str) -> str:
        if len(ok_df) and column in ok_df.columns and ok_df[column].notna().any():
            return f"{label}: {ok_df[column].mean():.6f}"
        return f"{label}: (none)"

    lines = [
        f"total_models: {len(summary_df)}",
        f"successful_models: {len(ok_df)}",
        f"failed_models: {len(failed_df)}",
        mean_line("average_avg_dataset_seconds_ok", "avg_dataset_seconds_ok"),
        mean_line("average_avg_accuracy_ok", "avg_accuracy_ok"),
        mean_line("average_avg_f1_ok", "avg_f1_ok"),
        mean_line("average_avg_balanced_accuracy_ok", "avg_balanced_accuracy_ok"),
        mean_line("average_avg_roc_auc_ok", "avg_roc_auc_ok"),
        mean_line("average_avg_log_loss_ok", "avg_log_loss_ok"),
        mean_line("average_ttt_adaptor_enabled_count", "ttt_adaptor_enabled_count"),
        f"ttt_adaptor_block_counts: {adaptor_block_counts}",
        mean_line("average_ttt_adaptor_trainable_params_ok", "avg_ttt_adaptor_trainable_params_ok"),
        f"global_wall_seconds: {wall_seconds:.3f}",
    ]

    if len(failed_df):
        lines.append(
            "failed_models_list: "
            + ", ".join(failed_df["model_name"].astype(str).tolist())
        )
    else:
        lines.append("failed_models_list: (none)")

    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run TabICLv2 classification benchmarks on data178 with "
            "epoch-shuffled chunk TTT, optional row-level residual adaptors, "
            "and AMD/ROCm multi-GPU workers."
        )
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--checkpoint-version", default=DEFAULT_CHECKPOINT_VERSION)
    parser.add_argument("--out-dir", default="1b_result/adaptor_dcnv2_ttt_chunk8000")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--gpu-groups", default="0")
    parser.add_argument("--n-estimators", type=int, default=32)
    parser.add_argument("--batch-size", type=parse_optional_int, default=8)
    parser.add_argument("--kv-cache", type=parse_kv_cache, default=False)
    parser.add_argument("--use-amp", type=parse_auto_bool, default="auto")
    parser.add_argument("--use-fa3", type=parse_auto_bool, default="auto")
    parser.add_argument("--offload-mode", choices=["auto", "gpu", "cpu", "disk"], default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-datasets", type=int, default=40)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--prefetch-models", type=int, default=4)
    parser.add_argument(
        "--ttt-holdout",
        dest="ttt_enabled",
        action="store_true",
        help="Compatibility alias: enable full-train epoch-chunk TTT. No B/C holdout is used.",
    )
    parser.add_argument(
        "--no-ttt",
        dest="ttt_enabled",
        action="store_false",
        help="Disable TTT and run ordinary TabICL inference.",
    )
    parser.set_defaults(ttt_enabled=True)
    parser.add_argument("--ttt-lr", type=float, default=1e-5)
    parser.add_argument("--ttt-scheduler", choices=["constant", "cosine_warmup"], default="cosine_warmup")
    parser.add_argument("--ttt-warmup-proportion", type=float, default=0.1)
    parser.add_argument("--ttt-grad-clip", type=float, default=1.0)
    parser.add_argument("--ttt-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument(
        "--ttt-micro-batch-size",
        type=int,
        default=1,
        help=(
            "Per-step TTT micro-batch size. When TTT data parallel is active, "
            "this value is interpreted per GPU and the effective batch becomes "
            "micro_batch_size x number_of_gpus_in_gpu_group."
        ),
    )
    parser.add_argument("--ttt-weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--ttt-epochs",
        "--ttt-steps",
        dest="ttt_epochs",
        type=int,
        default=30,
        help="Number of epoch-shuffled chunk TTT passes. --ttt-steps is kept as a compatibility alias.",
    )
    parser.add_argument("--ttt-max-chunk-size", type=int, default=8000)
    parser.add_argument("--ttt-min-chunk-size", type=int, default=50)
    parser.add_argument("--ttt-query-ratio", type=float, default=0.2)
    parser.add_argument("--ttt-n-estimators-finetune", type=int, default=2)
    parser.add_argument("--ttt-early-stopping", type=parse_bool, default=True)
    parser.add_argument("--ttt-patience", type=int, default=8)
    parser.add_argument("--ttt-min-delta", type=float, default=1e-4)
    parser.add_argument("--ttt-eval-metric", choices=["roc_auc", "log_loss", "accuracy"], default="roc_auc")
    parser.add_argument("--ttt-validation-fraction", type=float, default=0.1)
    parser.add_argument("--ttt-validation-n-estimators", type=int, default=2)
    parser.add_argument("--ttt-freeze-col", type=parse_bool, default=False)
    parser.add_argument("--ttt-freeze-row", type=parse_bool, default=False)
    parser.add_argument("--ttt-freeze-icl", type=parse_bool, default=False)
    parser.add_argument(
        "--ttt-adaptor",
        type=parse_bool,
        default=True,
        help="Enable a row-representation residual adaptor during the TTT update path.",
    )
    parser.add_argument(
        "--ttt-adaptor-block",
        choices=["mlp", "dcnv2"],
        default="dcnv2",
        help="Adaptor block type. mlp keeps the legacy bottleneck residual adaptor; dcnv2 uses a low-rank cross block.",
    )
    parser.add_argument("--ttt-adaptor-bottleneck", type=int, default=64)
    parser.add_argument("--ttt-adaptor-dropout", type=float, default=0.0)
    parser.add_argument("--ttt-adaptor-scale", type=float, default=1.0)
    parser.add_argument(
        "--ttt-adaptor-train-backbone",
        type=parse_bool,
        default=False,
        help="When the adaptor is enabled, also train the selected TabICL backbone modules.",
    )
    parser.add_argument(
        "--ttt-save-ckpt",
        type=parse_bool,
        default=False,
        help="Whether to save intermediate TabICL checkpoints during the TTT update path.",
    )
    parser.add_argument(
        "--ttt-save-ckpt-every",
        type=int,
        default=30,
        help="Save a TTT checkpoint every N optimizer steps and always save the final step.",
    )
    parser.add_argument(
        "--ttt-save-ckpt-start-step",
        type=parse_optional_int,
        default=None,
        help=(
            "First optimizer step to save a TTT checkpoint. Use None to keep the legacy "
            "multiple-of --ttt-save-ckpt-every schedule."
        ),
    )
    parser.add_argument(
        "--ttt-data-parallel",
        dest="ttt_data_parallel",
        action="store_true",
        help="Enable intra-worker multi-GPU data parallelism for the TTT update path.",
    )
    parser.add_argument(
        "--no-ttt-data-parallel",
        dest="ttt_data_parallel",
        action="store_false",
        help="Disable intra-worker multi-GPU data parallelism for the TTT update path.",
    )
    parser.set_defaults(ttt_data_parallel=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def resolve_gpu_ids(args: argparse.Namespace) -> List[int]:
    gpu_ids = parse_gpu_id_list(args.gpus) if args.gpus else detect_default_gpu_ids()
    if args.workers is None:
        args.workers = len(gpu_ids)
    if len(gpu_ids) != args.workers:
        raise ValueError(f"--gpus must contain exactly {args.workers} ids")
    return gpu_ids


def resolve_gpu_assignments(args: argparse.Namespace) -> tuple[List[int], List[str]]:
    if args.gpu_groups:
        if args.gpus:
            raise ValueError("Use either --gpu-groups or --gpus, not both")
        gpu_groups = parse_gpu_group_list(args.gpu_groups)
        if not gpu_groups:
            raise ValueError("--gpu-groups must contain at least one group")
        if args.workers is None:
            args.workers = len(gpu_groups)
        if len(gpu_groups) != args.workers:
            raise ValueError(f"--gpu-groups must contain exactly {args.workers} groups")
        gpu_ids = [first_gpu_id_from_group(gpu_group) for gpu_group in gpu_groups]
        return gpu_ids, gpu_groups

    gpu_ids = resolve_gpu_ids(args)
    return gpu_ids, [str(gpu_id) for gpu_id in gpu_ids]


def build_common_model_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "n_estimators": args.n_estimators,
        "batch_size": args.batch_size,
        "kv_cache": args.kv_cache,
        "allow_auto_download": True,
        "checkpoint_version": args.checkpoint_version,
        "use_amp": args.use_amp,
        "use_fa3": args.use_fa3,
        "offload_mode": args.offload_mode,
        "random_state": args.random_state,
    }


def run_single_model_mode(
    args: argparse.Namespace,
    dataset_dirs: List[Path],
    gpu_ids: List[int],
    gpu_groups: List[str],
    out_dir: Path,
) -> None:
    model_kwargs = build_common_model_kwargs(args)
    model_kwargs["model_path"] = normalize_model_path(args.model_path or DEFAULT_MODEL_PATH)
    ttt_config = build_ttt_config(args)

    start_time = time.time()
    ready_queue: mp.Queue = mp.Queue()
    start_event = mp.Event()

    worker_csv_paths: List[Path] = []
    processes: List[mp.Process] = []
    for worker_id in range(args.workers):
        assigned_dirs = [str(path.resolve()) for path in dataset_dirs[worker_id::args.workers]]
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        worker_csv_paths.append(worker_csv)

        proc = mp.Process(
            target=worker_main,
            args=(
                worker_id,
                gpu_ids[worker_id],
                gpu_groups[worker_id],
                assigned_dirs,
                ready_queue,
                start_event,
                str(worker_csv),
                dict(model_kwargs),
                ttt_config,
                args.verbose,
            ),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

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
                    f"[worker {message['worker_id']} | gpu {message.get('gpu_group', message['gpu_id'])}] "
                    f"ready assigned={message.get('assigned_count', '?')}"
                )
            continue

        if message.get("status") == "crash":
            raise RuntimeError(
                f"Worker {message['worker_id']} on gpu {message['gpu_id']} crashed "
                f"during initialization:\n{message.get('error', '(no traceback)')}"
            )

    start_event.set()

    for proc in processes:
        proc.join()

    dfs: List[pd.DataFrame] = []
    for worker_csv in worker_csv_paths:
        if worker_csv.exists():
            dfs.append(pd.read_csv(worker_csv))

    all_df = (
        pd.concat(dfs, ignore_index=True)
        if dfs
        else pd.DataFrame(columns=ResultRow.__annotations__.keys())
    )
    all_csv = out_dir / "all_classification_results.csv"
    summary_txt = out_dir / "summary.txt"
    all_df.to_csv(all_csv, index=False)

    wall_seconds = time.time() - start_time
    write_summary(summary_txt, all_df, dataset_dirs, wall_seconds)

    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {summary_txt}")
    print("model_kwargs:")
    print(json.dumps(model_kwargs, indent=2, ensure_ascii=False))
    if ttt_config.enabled:
        print("ttt_config:")
        print(json.dumps(asdict(ttt_config), indent=2, ensure_ascii=False))


def run_multi_model_mode(
    args: argparse.Namespace,
    dataset_dirs: List[Path],
    gpu_ids: List[int],
    gpu_groups: List[str],
    out_dir: Path,
) -> None:
    model_paths = discover_model_paths(Path(args.models_dir), max_models=args.max_models)
    if not model_paths:
        raise FileNotFoundError(f"No checkpoint files found under {args.models_dir}")

    worker_count = min(args.workers, len(model_paths))
    base_model_kwargs = build_common_model_kwargs(args)
    ttt_config = build_ttt_config(args)
    ready_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    task_queues: List[mp.Queue] = []
    processes: List[mp.Process] = []
    prefetcher = BackgroundPrefetcher(enabled=args.prefetch_models > 0, verbose=args.verbose)
    closed_workers: set[int] = set()

    try:
        for worker_id in range(worker_count):
            task_queue: mp.Queue = mp.Queue()
            task_queues.append(task_queue)
            proc = mp.Process(
                target=model_pool_worker_main,
                args=(
                    worker_id,
                    gpu_ids[worker_id],
                    gpu_groups[worker_id],
                    [str(path.resolve()) for path in dataset_dirs],
                    ready_queue,
                    task_queue,
                    result_queue,
                    dict(base_model_kwargs),
                    ttt_config,
                    args.verbose,
                ),
                daemon=False,
            )
            proc.start()
            processes.append(proc)

        ready_workers: set[int] = set()
        while len(ready_workers) < worker_count:
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
                        f"[worker {message['worker_id']} | gpu {message.get('gpu_group', message['gpu_id'])}] "
                        f"model-pool ready datasets={message.get('assigned_count', '?')}",
                        flush=True,
                    )
                continue

            if message.get("status") == "crash":
                raise RuntimeError(
                    f"Worker {message['worker_id']} on gpu {message['gpu_id']} crashed "
                    f"during initialization:\n{message.get('error', '(no traceback)')}"
                )

        next_model_idx = 0
        completed_models = 0
        start_time = time.time()
        collected_summaries: List[dict[str, Any]] = []
        prefetcher.schedule(model_paths[: min(len(model_paths), worker_count + max(0, args.prefetch_models))])

        for worker_id in range(worker_count):
            if next_model_idx >= len(model_paths):
                break
            task_queues[worker_id].put({"model_path": str(model_paths[next_model_idx])})
            next_model_idx += 1
            prefetcher.schedule(model_paths[next_model_idx : next_model_idx + max(0, args.prefetch_models)])

        while completed_models < len(model_paths):
            message = result_queue.get()
            status = message.get("status")

            if status == "worker_crash":
                raise RuntimeError(
                    f"Worker {message.get('worker_id')} on gpu {message.get('gpu_id')} crashed:\n"
                    f"{message.get('error', '(no traceback)')}"
                )

            if status != "model_done":
                continue

            worker_id = int(message["worker_id"])
            summary = dict(message["summary"])
            collected_summaries.append(summary)
            completed_models += 1

            if args.verbose:
                print(
                    f"[worker {worker_id} | gpu {summary['gpu_id']}] "
                    f"finished model={summary['model_name']} status={summary['status']} "
                    f"ok={summary['ok_count']} fail={summary['failed_count']} "
                    f"wall={summary['model_wall_seconds']:.2f}s",
                    flush=True,
                )

            if next_model_idx < len(model_paths):
                task_queues[worker_id].put({"model_path": str(model_paths[next_model_idx])})
                next_model_idx += 1
                prefetcher.schedule(model_paths[next_model_idx : next_model_idx + max(0, args.prefetch_models)])
            elif worker_id not in closed_workers:
                task_queues[worker_id].put(None)
                closed_workers.add(worker_id)

        for worker_id in range(worker_count):
            if worker_id not in closed_workers:
                task_queues[worker_id].put(None)

        for proc in processes:
            proc.join()

        wall_seconds = time.time() - start_time
        write_model_pool_outputs(out_dir, collected_summaries, wall_seconds)

        print(f"saved_all_models_csv: {out_dir / 'all_models_summary.csv'}")
        print(f"saved_summary: {out_dir / 'summary.txt'}")
        print("base_model_kwargs:")
        print(json.dumps(base_model_kwargs, indent=2, ensure_ascii=False))
        if ttt_config.enabled:
            print("ttt_config:")
            print(json.dumps(asdict(ttt_config), indent=2, ensure_ascii=False))
    finally:
        for worker_id in range(len(task_queues)):
            if worker_id not in closed_workers:
                try:
                    task_queues[worker_id].put(None)
                except Exception:
                    pass
        for proc in processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
        prefetcher.close()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    ensure_runtime_deps()

    if args.models_dir is not None and args.model_path is not None:
        raise ValueError("--models-dir and --model-path are mutually exclusive")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = find_dataset_dirs(data_root)
    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: args.max_datasets]
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset directories found under {data_root}")

    gpu_ids, gpu_groups = resolve_gpu_assignments(args)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if args.models_dir is not None:
        run_multi_model_mode(args, dataset_dirs, gpu_ids, gpu_groups, out_dir)
    else:
        run_single_model_mode(args, dataset_dirs, gpu_ids, gpu_groups, out_dir)


if __name__ == "__main__":
    main()
