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
    roc_auc: Optional[float]
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
    ttt_val_baseline_accuracy: Optional[float] = None
    ttt_val_best_accuracy: Optional[float] = None
    ttt_best_epoch: int = 0
    ttt_stopped_early: bool = False
    ttt_oom_fallback: bool = False
    ttt_fallback_reason: Optional[str] = None
    pseudo_enabled: bool = False
    pseudo_source: Optional[str] = None
    pseudo_selected_count: int = 0
    pseudo_selected_ratio: Optional[float] = None
    pseudo_conf_mean: Optional[float] = None
    pseudo_margin_mean: Optional[float] = None
    pseudo_loss_weight: Optional[float] = None
    pseudo_class_hist: str = "{}"
    pseudo_threshold_mode: Optional[str] = None
    pseudo_precision_posthoc: Optional[float] = None


@dataclass
class TTTConfig:
    enabled: bool = False
    lr: float = 2e-6
    scheduler: str = "constant"
    grad_clip: float = 1.0
    dtype: str = "float32"
    micro_batch_size: int = 1
    weight_decay: float = 0.0
    epochs: int = 1
    steps: int = 1
    max_chunk_size: int = 10_000
    min_chunk_size: int = 50
    query_ratio: float = 0.2
    n_estimators_finetune: int = 2
    early_stopping: bool = True
    patience: int = 8
    min_delta: float = 1e-4
    validation_fraction: float = 0.1
    validation_n_estimators: int = 2
    freeze_col: bool = True
    freeze_row: bool = True
    train_fraction: float = 0.75
    random_state: int = 42
    data_parallel: bool = False
    gpu_group: Optional[str] = None
    save_ckpt: bool = True
    save_ckpt_every: int = 2
    save_ckpt_start_step: Optional[int] = None
    ckpt_root: Optional[str] = None
    pseudo_labels: str = "off"
    pseudo_target_val_precision: float = 0.90
    pseudo_min_confidence: float = 0.90
    pseudo_min_margin: float = 0.15
    pseudo_max_ratio: float = 0.30
    pseudo_query_ratio: float = 0.50
    pseudo_loss_weight: float = 0.20
    pseudo_refresh: str = "never"


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
    val_baseline_accuracy: Optional[float] = None
    val_best_accuracy: Optional[float] = None
    best_epoch: int = 0
    stopped_early: bool = False


@dataclass
class MetaBatch:
    X: Any
    y_train: Any
    y_query: Any
    train_size: int
    skip_reason: Optional[str] = None
    real_query_size: int = 0
    pseudo_query_size: int = 0


@dataclass
class PseudoLabelPool:
    X: Any
    y: Any
    test_indices: Any
    confidence: Any
    margin: Any
    class_hist: str
    threshold_mode: str
    source: str = "transductive"

    @property
    def selected_count(self) -> int:
        return int(len(self.y))


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
    pseudo_enabled_count: int
    total_pseudo_selected_ok: int
    avg_pseudo_selected_ratio_ok: Optional[float]
    avg_pseudo_conf_mean_ok: Optional[float]
    avg_pseudo_margin_mean_ok: Optional[float]
    avg_pseudo_precision_posthoc_ok: Optional[float]
    avg_accuracy_ok: Optional[float]
    avg_f1_ok: Optional[float]
    avg_roc_auc_ok: Optional[float]
    avg_fit_seconds_ok: Optional[float]
    avg_predict_seconds_ok: Optional[float]
    avg_dataset_seconds_ok: Optional[float]
    total_dataset_seconds_ok: float
    model_wall_seconds: float
    status: str
    error: Optional[str]
    failed_datasets: str


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


def compute_tabpfn_roc_auc(y_true: Any, y_proba: Any) -> Optional[float]:
    ensure_runtime_deps()
    try:
        from sklearn.metrics import roc_auc_score

        y_true_arr = np.asarray(y_true)
        proba_arr = np.asarray(y_proba)
        if proba_arr.ndim != 2 or proba_arr.shape[0] != y_true_arr.shape[0] or proba_arr.shape[1] < 2:
            return None
        if len(np.unique(y_true_arr)) < 2:
            return None
        if proba_arr.shape[1] == 2:
            return float(roc_auc_score(y_true_arr, proba_arr[:, 1]))
        return float(roc_auc_score(y_true_arr, proba_arr, multi_class="ovr"))
    except (ValueError, RuntimeError, AttributeError, TypeError):
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
            f"roc_auc={format_optional_float(row.roc_auc)} "
            f"fit={row.fit_seconds:.3f}s "
            f"predict={row.predict_seconds:.3f}s "
            f"ttt_applied={row.ttt_applied} "
            f"ttt_update={row.ttt_update_seconds:.3f}s "
            f"ttt_loss={format_optional_float(row.ttt_loss)} "
            f"ttt_steps={row.ttt_steps} "
            f"ttt_epochs={row.ttt_epochs} "
            f"ttt_mode={row.ttt_batch_mode} "
            f"ttt_val_best={format_optional_float(row.ttt_val_best_accuracy)} "
            f"ttt_best_epoch={row.ttt_best_epoch} "
            f"ttt_stopped_early={row.ttt_stopped_early} "
            f"ttt_oom_fallback={row.ttt_oom_fallback} "
            f"pseudo_enabled={row.pseudo_enabled} "
            f"pseudo_selected={row.pseudo_selected_count} "
            f"pseudo_mode={row.pseudo_threshold_mode}"
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
    if not 0.0 < float(args.ttt_validation_fraction) < 1.0:
        raise ValueError("--ttt-validation-fraction must be in (0, 1)")
    if int(args.ttt_validation_n_estimators) < 1:
        raise ValueError("--ttt-validation-n-estimators must be >= 1")
    if args.ttt_pseudo_labels not in {"off", "transductive"}:
        raise ValueError("--ttt-pseudo-labels must be one of: off, transductive")
    if args.ttt_pseudo_refresh != "never":
        raise ValueError("--ttt-pseudo-refresh currently supports only 'never'")
    if not 0.0 <= float(args.ttt_pseudo_target_val_precision) <= 1.0:
        raise ValueError("--ttt-pseudo-target-val-precision must be in [0, 1]")
    if not 0.0 <= float(args.ttt_pseudo_min_confidence) <= 1.0:
        raise ValueError("--ttt-pseudo-min-confidence must be in [0, 1]")
    if not 0.0 <= float(args.ttt_pseudo_min_margin) <= 1.0:
        raise ValueError("--ttt-pseudo-min-margin must be in [0, 1]")
    if not 0.0 <= float(args.ttt_pseudo_max_ratio) <= 1.0:
        raise ValueError("--ttt-pseudo-max-ratio must be in [0, 1]")
    if not 0.0 <= float(args.ttt_pseudo_query_ratio) <= 1.0:
        raise ValueError("--ttt-pseudo-query-ratio must be in [0, 1]")
    if float(args.ttt_pseudo_loss_weight) < 0.0:
        raise ValueError("--ttt-pseudo-loss-weight must be >= 0")

    return TTTConfig(
        enabled=bool(args.ttt_enabled),
        lr=float(args.ttt_lr),
        scheduler=str(args.ttt_scheduler),
        grad_clip=float(args.ttt_grad_clip),
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
        validation_fraction=float(args.ttt_validation_fraction),
        validation_n_estimators=int(args.ttt_validation_n_estimators),
        freeze_col=bool(args.ttt_freeze_col),
        freeze_row=bool(args.ttt_freeze_row),
        random_state=int(args.random_state),
        data_parallel=bool(args.ttt_data_parallel),
        save_ckpt=bool(args.ttt_save_ckpt),
        save_ckpt_every=int(args.ttt_save_ckpt_every),
        save_ckpt_start_step=(
            int(args.ttt_save_ckpt_start_step) if args.ttt_save_ckpt_start_step is not None else None
        ),
        ckpt_root=str((Path(args.out_dir).expanduser() / "ttt_ckpts").resolve()),
        pseudo_labels=str(args.ttt_pseudo_labels),
        pseudo_target_val_precision=float(args.ttt_pseudo_target_val_precision),
        pseudo_min_confidence=float(args.ttt_pseudo_min_confidence),
        pseudo_min_margin=float(args.ttt_pseudo_min_margin),
        pseudo_max_ratio=float(args.ttt_pseudo_max_ratio),
        pseudo_query_ratio=float(args.ttt_pseudo_query_ratio),
        pseudo_loss_weight=float(args.ttt_pseudo_loss_weight),
        pseudo_refresh=str(args.ttt_pseudo_refresh),
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


def _local_label_encode(y_values) -> tuple[Any, Dict[int, int]]:
    classes = np.unique(y_values.astype(int))
    mapping = {int(value): idx for idx, value in enumerate(classes.tolist())}
    encoded = np.asarray([mapping[int(value)] for value in y_values], dtype=np.int64)
    return encoded, mapping


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


def _configure_ttt_trainable_params(classifier, config: TTTConfig):
    model = _get_ttt_base_model(classifier)
    model.train()

    for param in model.parameters():
        param.requires_grad = False

    # Keep frozen blocks in train mode so TabICL uses the training forward path.
    # Their parameters remain frozen by requires_grad=False.
    model.col_embedder.train()
    if not config.freeze_col:
        _set_requires_grad(model.col_embedder, True)

    model.row_interactor.train()
    if not config.freeze_row:
        _set_requires_grad(model.row_interactor, True)

    model.icl_predictor.train()
    _set_requires_grad(model.icl_predictor, True)

    return [param for param in model.parameters() if param.requires_grad]


def _set_ttt_train_mode(classifier) -> None:
    model = _get_ttt_base_model(classifier)
    model.train()
    model.col_embedder.train()
    model.row_interactor.train()
    model.icl_predictor.train()


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


def _evaluate_ttt_validation_accuracy(
    classifier,
    X_context,
    y_context,
    X_val,
    y_val,
    config: TTTConfig,
) -> Optional[float]:
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
            y_pred = classifier.predict(X_val)
        return float(np.mean(np.asarray(y_pred) == np.asarray(y_val)))
    finally:
        if original_n_estimators is not None:
            classifier.n_estimators = original_n_estimators


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _label_to_python(value: Any) -> Any:
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def _label_key(value: Any) -> str:
    return str(_label_to_python(value))


def _empty_pseudo_pool(X_test, *, threshold_mode: str) -> PseudoLabelPool:
    ensure_runtime_deps()
    empty_idx = np.asarray([], dtype=int)
    return PseudoLabelPool(
        X=take_rows(X_test, empty_idx),
        y=np.asarray([], dtype=np.int64),
        test_indices=empty_idx,
        confidence=np.asarray([], dtype=np.float64),
        margin=np.asarray([], dtype=np.float64),
        class_hist="{}",
        threshold_mode=threshold_mode,
    )


def _top_confidence_and_margin(proba: Any) -> tuple[Any, Any, Any]:
    ensure_runtime_deps()
    proba_arr = np.asarray(proba, dtype=np.float64)
    if proba_arr.ndim != 2 or proba_arr.shape[0] == 0 or proba_arr.shape[1] == 0:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
        )
    top_idx = np.argmax(proba_arr, axis=1)
    top1 = proba_arr[np.arange(proba_arr.shape[0]), top_idx]
    if proba_arr.shape[1] == 1:
        top2 = np.zeros_like(top1)
    else:
        top2 = np.partition(proba_arr, -2, axis=1)[:, -2]
    return top_idx, top1, top1 - top2


def _calibrate_pseudo_thresholds(
    classifier,
    X_val,
    y_val,
    config: TTTConfig,
) -> tuple[Dict[str, Optional[float]], str]:
    ensure_runtime_deps()

    if X_val is None or y_val is None or len(y_val) == 0:
        return {}, "no_validation"

    import torch

    with torch.inference_mode():
        val_proba = np.asarray(classifier.predict_proba(X_val), dtype=np.float64)
    if val_proba.ndim != 2 or val_proba.shape[0] == 0:
        return {}, "no_validation"

    top_idx, top1, _margin = _top_confidence_and_margin(val_proba)
    pred_labels = classifier.y_encoder_.inverse_transform(top_idx)
    y_val_arr = np.asarray(y_val)

    thresholds: Dict[str, Optional[float]] = {}
    modes: set[str] = set()
    target_precision = float(config.pseudo_target_val_precision)
    min_confidence = float(config.pseudo_min_confidence)

    for class_label in np.asarray(classifier.y_encoder_.classes_):
        class_key = _label_key(class_label)
        pred_mask = np.asarray(pred_labels == class_label)
        pred_count = int(pred_mask.sum())
        true_count = int(np.asarray(y_val_arr == class_label).sum())
        if true_count < 3 or pred_count < 3:
            thresholds[class_key] = min_confidence
            modes.add("fallback")
            continue

        scores = top1[pred_mask]
        correct = np.asarray(y_val_arr[pred_mask] == class_label, dtype=bool)
        best_threshold = None
        best_selected = -1
        best_precision = -1.0
        for threshold in np.unique(scores)[::-1]:
            selected = scores >= float(threshold)
            selected_count = int(selected.sum())
            if selected_count <= 0:
                continue
            precision = float(correct[selected].mean())
            if precision + 1e-12 >= target_precision and (
                selected_count > best_selected
                or (selected_count == best_selected and precision > best_precision)
            ):
                best_threshold = float(threshold)
                best_selected = selected_count
                best_precision = precision

        if best_threshold is None:
            thresholds[class_key] = None
            modes.add("no_feasible")
        else:
            thresholds[class_key] = best_threshold
            modes.add("calibrated")

    if not modes:
        threshold_mode = "no_validation"
    elif modes == {"calibrated"}:
        threshold_mode = "calibrated"
    elif modes == {"fallback"}:
        threshold_mode = "fallback_only"
    elif modes <= {"no_feasible"}:
        threshold_mode = "no_feasible"
    else:
        threshold_mode = "mixed"
    return thresholds, threshold_mode


def build_transductive_pseudo_pool(
    classifier,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    config: TTTConfig,
) -> PseudoLabelPool:
    ensure_runtime_deps()

    if config.pseudo_labels == "off":
        return _empty_pseudo_pool(X_test, threshold_mode="off")
    if config.pseudo_labels != "transductive":
        raise ValueError("--ttt-pseudo-labels must be one of: off, transductive")
    if config.pseudo_refresh != "never":
        raise ValueError("--ttt-pseudo-refresh currently supports only 'never'")

    if X_val is None or y_val is None or len(y_val) == 0:
        return _empty_pseudo_pool(X_test, threshold_mode="no_validation")

    classifier.fit(X_train, y_train)
    if classifier.n_classes_ > classifier.model_.max_classes:
        return _empty_pseudo_pool(X_test, threshold_mode="no_feasible")

    thresholds, threshold_mode = _calibrate_pseudo_thresholds(classifier, X_val, y_val, config)
    if threshold_mode == "no_validation":
        return _empty_pseudo_pool(X_test, threshold_mode=threshold_mode)

    import torch

    with torch.inference_mode():
        test_proba = np.asarray(classifier.predict_proba(X_test), dtype=np.float64)
    top_idx, top1, margin = _top_confidence_and_margin(test_proba)
    if top_idx.size == 0:
        return _empty_pseudo_pool(X_test, threshold_mode="no_feasible")
    pred_labels = classifier.y_encoder_.inverse_transform(top_idx)

    train_labels = np.asarray(y_train)
    train_class_counts: Dict[str, int] = {}
    for label in train_labels:
        key = _label_key(label)
        train_class_counts[key] = train_class_counts.get(key, 0) + 1

    global_limit = int(np.floor(float(config.pseudo_max_ratio) * int(len(train_labels))))
    if global_limit <= 0:
        return _empty_pseudo_pool(X_test, threshold_mode=threshold_mode)

    accepted: list[int] = []
    for idx, label in enumerate(pred_labels):
        class_key = _label_key(label)
        threshold = thresholds.get(class_key)
        if threshold is None:
            continue
        threshold = max(float(threshold), float(config.pseudo_min_confidence))
        if float(top1[idx]) < threshold:
            continue
        if float(margin[idx]) < float(config.pseudo_min_margin):
            continue
        if int(np.floor(0.5 * train_class_counts.get(class_key, 0))) <= 0:
            continue
        accepted.append(idx)

    if not accepted:
        return _empty_pseudo_pool(
            X_test,
            threshold_mode="no_feasible" if threshold_mode != "no_validation" else threshold_mode,
        )

    accepted.sort(key=lambda item: (-float(top1[item]), -float(margin[item]), int(item)))
    selected: list[int] = []
    selected_by_class: Dict[str, int] = {}
    for idx in accepted:
        class_key = _label_key(pred_labels[idx])
        class_quota = int(np.floor(0.5 * train_class_counts.get(class_key, 0)))
        if selected_by_class.get(class_key, 0) >= class_quota:
            continue
        selected.append(idx)
        selected_by_class[class_key] = selected_by_class.get(class_key, 0) + 1
        if len(selected) >= global_limit:
            break

    if not selected:
        return _empty_pseudo_pool(
            X_test,
            threshold_mode="no_feasible" if threshold_mode != "no_validation" else threshold_mode,
        )

    selected_idx = np.asarray(selected, dtype=int)
    class_hist = {
        key: int(value)
        for key, value in sorted(selected_by_class.items(), key=lambda item: item[0])
    }
    return PseudoLabelPool(
        X=take_rows(X_test, selected_idx),
        y=np.asarray(pred_labels[selected_idx]),
        test_indices=selected_idx,
        confidence=np.asarray(top1[selected_idx], dtype=np.float64),
        margin=np.asarray(margin[selected_idx], dtype=np.float64),
        class_hist=_json_dumps(class_hist),
        threshold_mode=threshold_mode,
    )


def _build_classification_meta_batch(
    classifier,
    X_chunk,
    y_chunk,
    *,
    config: TTTConfig,
    query_size: int,
    epoch_seed: int,
    chunk_idx: int,
    X_pseudo_encoded=None,
    y_pseudo_encoded=None,
) -> MetaBatch:
    try:
        from tabicl._sklearn.preprocessing import EnsembleGenerator
    except ImportError:
        from tabicl.sklearn.preprocessing import EnsembleGenerator

    if len(y_chunk) < 2:
        return MetaBatch(None, None, None, 0, "chunk has fewer than two samples")

    split_seed = int(epoch_seed + chunk_idx * 7919)
    n_classes_in_chunk = int(len(np.unique(y_chunk)))
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

    y_local_all, label_mapping = _local_label_encode(np.asarray(y_chunk))
    y_ctx = y_local_all[ctx_idx]
    y_qry = np.asarray([label_mapping[int(value)] for value in y_qry_raw], dtype=np.int64)
    X_ctx = X_chunk[ctx_idx]
    X_qry = X_chunk[qry_idx]
    real_query_size = int(len(y_qry))

    pseudo_query_size = 0
    if (
        X_pseudo_encoded is not None
        and y_pseudo_encoded is not None
        and len(y_pseudo_encoded) > 0
        and config.pseudo_labels == "transductive"
        and config.pseudo_query_ratio > 0
        and config.pseudo_loss_weight > 0
    ):
        max_pseudo = min(
            int(np.floor(real_query_size * float(config.pseudo_query_ratio))),
            int(len(y_pseudo_encoded)),
        )
        if max_pseudo > 0:
            eligible_indices = [
                idx
                for idx, label in enumerate(np.asarray(y_pseudo_encoded).astype(int))
                if int(label) in label_mapping
            ]
            if eligible_indices:
                pseudo_rng = np.random.default_rng(split_seed + 104729)
                chosen = np.asarray(eligible_indices, dtype=int)
                pseudo_rng.shuffle(chosen)
                chosen = chosen[: min(max_pseudo, int(len(chosen)))]
                if chosen.size > 0:
                    X_pseudo_qry = np.asarray(X_pseudo_encoded)[chosen]
                    y_pseudo_qry = np.asarray(
                        [label_mapping[int(value)] for value in np.asarray(y_pseudo_encoded)[chosen]],
                        dtype=np.int64,
                    )
                    X_qry = np.concatenate([X_qry, X_pseudo_qry], axis=0)
                    y_qry = np.concatenate([y_qry, y_pseudo_qry], axis=0)
                    pseudo_query_size = int(len(y_pseudo_qry))

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
        class_shuffles = gen.class_shuffles_[norm_method]
        if len(class_shuffles) != X_variant.shape[0]:
            raise RuntimeError(
                f"Ensemble data/class shuffle mismatch for norm_method={norm_method!r}: "
                f"{X_variant.shape[0]} views vs {len(class_shuffles)} class shuffles"
            )
        for class_shuffle in class_shuffles:
            y_query_list.append(np.asarray(class_shuffle, dtype=np.int64)[y_qry])

    import torch

    return MetaBatch(
        X=torch.from_numpy(np.concatenate(X_list, axis=0)).float(),
        y_train=torch.from_numpy(np.concatenate(y_train_list, axis=0)).float(),
        y_query=torch.from_numpy(np.stack(y_query_list, axis=0)).long(),
        train_size=int(len(ctx_idx)),
        real_query_size=real_query_size,
        pseudo_query_size=pseudo_query_size,
        skip_reason=None,
    )


def iter_epoch_meta_batches(
    classifier,
    X_encoded,
    y_encoded,
    *,
    config: TTTConfig,
    epoch_seed: int,
    X_pseudo_encoded=None,
    y_pseudo_encoded=None,
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
            X_pseudo_encoded=X_pseudo_encoded,
            y_pseudo_encoded=y_pseudo_encoded,
        )


def move_meta_batch(batch: MetaBatch, device) -> MetaBatch:
    return MetaBatch(
        X=batch.X.to(device, non_blocking=True),
        y_train=batch.y_train.to(device, non_blocking=True),
        y_query=batch.y_query.to(device, non_blocking=True),
        train_size=batch.train_size,
        real_query_size=batch.real_query_size,
        pseudo_query_size=batch.pseudo_query_size,
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
    pseudo_pool: PseudoLabelPool | None = None,
) -> TTTUpdateResult:
    ensure_runtime_deps()

    if config.scheduler != "constant":
        raise ValueError("--ttt-scheduler currently supports only 'constant'")
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
    torch_dtype = _torch_dtype(config.dtype)
    trainable_params = _configure_ttt_trainable_params(classifier, config)
    if not trainable_params:
        return TTTUpdateResult(
            applied=False,
            loss=None,
            steps=0,
            update_seconds=time.time() - update_start,
            reason="No trainable parameters selected for TTT",
            epochs=0,
            chunks_per_epoch=0,
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    forward_model, data_parallel_world_size = _build_ttt_forward_model(classifier, config)
    effective_micro_batch_size = config.micro_batch_size * data_parallel_world_size

    X_encoded = classifier.X_encoder_.transform(X_train)
    y_encoded = classifier.y_encoder_.transform(y_train)
    X_pseudo_encoded = None
    y_pseudo_encoded = None
    if pseudo_pool is not None and pseudo_pool.selected_count > 0 and config.pseudo_labels == "transductive":
        X_pseudo_encoded = classifier.X_encoder_.transform(pseudo_pool.X)
        y_pseudo_encoded = classifier.y_encoder_.transform(pseudo_pool.y)
    chunks_per_epoch = count_ttt_chunks(
        int(len(y_encoded)),
        max_chunk_size=config.max_chunk_size,
        min_chunk_size=config.min_chunk_size,
    )

    device = classifier.device_
    device_type = getattr(device, "type", str(device).split(":")[0])
    use_autocast = torch_dtype != torch.float32 and device_type in {"cuda", "cpu"}

    from contextlib import nullcontext

    def forward_context():
        if use_autocast:
            return torch.autocast(device_type=device_type, dtype=torch_dtype)
        return nullcontext()

    last_loss = None
    update_steps = 0
    skipped_batches = 0
    skip_reasons: Dict[str, int] = {}
    baseline_metric = None
    best_metric = None
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
            baseline_metric = _evaluate_ttt_validation_accuracy(classifier, X_train, y_train, X_val, y_val, config)
            if baseline_metric is not None:
                best_metric = float(baseline_metric)
                best_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
                print(
                    f"[ttt-val] model={model_name} dataset={dataset_name} "
                    f"baseline_accuracy={best_metric:.6f}",
                    flush=True,
                )

        last_saved_step = 0
        for epoch_idx in range(config.epochs):
            _set_ttt_train_mode(classifier)
            epoch_seed = config.random_state + epoch_idx
            epoch_loss_sum = 0.0
            epoch_updates = 0
            for batch in iter_epoch_meta_batches(
                classifier,
                X_encoded,
                y_encoded,
                config=config,
                epoch_seed=epoch_seed,
                X_pseudo_encoded=X_pseudo_encoded,
                y_pseudo_encoded=y_pseudo_encoded,
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
                    with forward_context():
                        logits = forward_model(X_batch, y_train_batch.float())
                        n_classes = int(y_train_batch.max().item()) + 1
                        real_query_size = int(batch.real_query_size)
                        pseudo_query_size = int(batch.pseudo_query_size)
                        if real_query_size <= 0:
                            raise RuntimeError("TTT meta-batch has no real query samples")
                        logits_real = logits[:, :real_query_size, :n_classes].reshape(-1, n_classes)
                        targets_real = y_query_batch[:, :real_query_size].long().reshape(-1)
                        loss = F.cross_entropy(logits_real, targets_real)
                        if pseudo_query_size > 0 and config.pseudo_loss_weight > 0:
                            pseudo_start = real_query_size
                            pseudo_end = real_query_size + pseudo_query_size
                            logits_pseudo = logits[:, pseudo_start:pseudo_end, :n_classes].reshape(-1, n_classes)
                            targets_pseudo = y_query_batch[:, pseudo_start:pseudo_end].long().reshape(-1)
                            loss = loss + float(config.pseudo_loss_weight) * F.cross_entropy(
                                logits_pseudo,
                                targets_pseudo,
                            )
                        scaled_loss = loss * (X_batch.shape[0] / total_views)

                    scaled_loss.backward()
                    batch_loss += float(scaled_loss.detach().cpu())

                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip)
                optimizer.step()
                update_steps += 1
                epoch_updates += 1
                epoch_loss_sum += batch_loss
                last_loss = batch_loss

                if update_steps % 3 == 0:
                    print(
                        f"[ttt-loss] model={model_name} dataset={dataset_name} "
                        f"epoch={epoch_idx + 1}/{config.epochs} step={update_steps} "
                        f"loss={batch_loss:.6f}",
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
                    f"mean_loss={epoch_loss_sum / epoch_updates:.6f} updates={epoch_updates}",
                    flush=True,
                )

            if best_state is not None:
                val_metric = _evaluate_ttt_validation_accuracy(classifier, X_train, y_train, X_val, y_val, config)
                if val_metric is not None:
                    improved = val_metric > float(best_metric) + config.min_delta
                    if improved:
                        best_metric = float(val_metric)
                        best_epoch = epoch_idx + 1
                        patience_counter = 0
                        best_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
                    else:
                        patience_counter += 1
                    print(
                        f"[ttt-val] model={model_name} dataset={dataset_name} "
                        f"epoch={epoch_idx + 1}/{config.epochs} "
                        f"accuracy={val_metric:.6f} best={best_metric:.6f} "
                        f"patience={patience_counter}/{config.patience}",
                        flush=True,
                    )
                    if patience_counter >= config.patience:
                        stopped_early = True
                        print(
                            f"[ttt-early-stop] model={model_name} dataset={dataset_name} "
                            f"epoch={epoch_idx + 1} best_epoch={best_epoch} "
                            f"best_accuracy={best_metric:.6f}",
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
                val_baseline_accuracy=baseline_metric,
                val_best_accuracy=best_metric,
                best_epoch=best_epoch,
                stopped_early=stopped_early,
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
        val_baseline_accuracy=baseline_metric,
        val_best_accuracy=best_metric,
        best_epoch=best_epoch,
        stopped_early=stopped_early,
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
    pseudo_enabled_count = int(truthy_column_mask(ok_df, "pseudo_enabled").sum()) if len(ok_df) else 0
    total_pseudo_selected_ok = (
        int(pd.to_numeric(ok_df["pseudo_selected_count"], errors="coerce").fillna(0).sum())
        if len(ok_df) and "pseudo_selected_count" in ok_df.columns
        else 0
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
            pseudo_enabled_count=0,
            total_pseudo_selected_ok=0,
            avg_pseudo_selected_ratio_ok=None,
            avg_pseudo_conf_mean_ok=None,
            avg_pseudo_margin_mean_ok=None,
            avg_pseudo_precision_posthoc_ok=None,
            avg_accuracy_ok=None,
            avg_f1_ok=None,
            avg_roc_auc_ok=None,
            avg_fit_seconds_ok=None,
            avg_predict_seconds_ok=None,
            avg_dataset_seconds_ok=None,
            total_dataset_seconds_ok=0.0,
            model_wall_seconds=float(model_wall_seconds),
            status="fail",
            error=error,
            failed_datasets=",".join(path.name for path in dataset_dirs),
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
        pseudo_enabled_count=pseudo_enabled_count,
        total_pseudo_selected_ok=total_pseudo_selected_ok,
        avg_pseudo_selected_ratio_ok=(
            float(pd.to_numeric(ok_df["pseudo_selected_ratio"], errors="coerce").mean())
            if len(ok_df) and "pseudo_selected_ratio" in ok_df.columns
            else None
        ),
        avg_pseudo_conf_mean_ok=(
            float(pd.to_numeric(ok_df["pseudo_conf_mean"], errors="coerce").mean())
            if len(ok_df) and "pseudo_conf_mean" in ok_df.columns
            else None
        ),
        avg_pseudo_margin_mean_ok=(
            float(pd.to_numeric(ok_df["pseudo_margin_mean"], errors="coerce").mean())
            if len(ok_df) and "pseudo_margin_mean" in ok_df.columns
            else None
        ),
        avg_pseudo_precision_posthoc_ok=(
            float(pd.to_numeric(ok_df["pseudo_precision_posthoc"], errors="coerce").mean())
            if len(ok_df) and "pseudo_precision_posthoc" in ok_df.columns
            else None
        ),
        avg_accuracy_ok=(float(ok_df["accuracy"].mean()) if len(ok_df) else None),
        avg_f1_ok=(float(ok_df["f1"].mean()) if len(ok_df) else None),
        avg_roc_auc_ok=(
            float(ok_df["roc_auc"].mean())
            if len(ok_df) and ok_df["roc_auc"].notna().any()
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
                roc_auc=None,
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
        ttt_val_baseline_accuracy = None
        ttt_val_best_accuracy = None
        ttt_best_epoch = 0
        ttt_stopped_early = False
        ttt_oom_fallback = False
        ttt_fallback_reason = None
        pseudo_pool: PseudoLabelPool | None = None
        pseudo_enabled = bool(ttt_config.enabled and ttt_config.pseudo_labels == "transductive")
        pseudo_source = "transductive" if pseudo_enabled else None
        pseudo_selected_count = 0
        pseudo_selected_ratio = 0.0 if pseudo_enabled else None
        pseudo_conf_mean = None
        pseudo_margin_mean = None
        pseudo_loss_weight = ttt_config.pseudo_loss_weight if pseudo_enabled else None
        pseudo_class_hist = "{}"
        pseudo_threshold_mode = "off" if not pseudo_enabled else None
        pseudo_precision_posthoc = None
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
                    if pseudo_enabled:
                        pseudo_pool = build_transductive_pseudo_pool(
                            classifier,
                            X_ttt_train,
                            y_ttt_train,
                            X_ttt_val,
                            y_ttt_val,
                            X_test,
                            ttt_config,
                        )
                        pseudo_selected_count = int(pseudo_pool.selected_count)
                        pseudo_selected_ratio = (
                            float(pseudo_selected_count / len(y_test)) if len(y_test) else 0.0
                        )
                        pseudo_conf_mean = (
                            float(np.asarray(pseudo_pool.confidence, dtype=np.float64).mean())
                            if pseudo_selected_count > 0
                            else None
                        )
                        pseudo_margin_mean = (
                            float(np.asarray(pseudo_pool.margin, dtype=np.float64).mean())
                            if pseudo_selected_count > 0
                            else None
                        )
                        pseudo_class_hist = str(pseudo_pool.class_hist)
                        pseudo_threshold_mode = str(pseudo_pool.threshold_mode)
                        ttt_split_reason = append_ttt_reason(
                            ttt_split_reason,
                            f"pseudo={pseudo_pool.source}:{pseudo_selected_count}:mode={pseudo_threshold_mode}",
                        )
                    ttt_result = run_ttt_epoch_chunk_update(
                        classifier,
                        X_ttt_train,
                        y_ttt_train,
                        X_ttt_val,
                        y_ttt_val,
                        ttt_config,
                        model_name=model_name,
                        dataset_name=dataset_dir.name,
                        pseudo_pool=pseudo_pool,
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
                    ttt_val_baseline_accuracy = ttt_result.val_baseline_accuracy
                    ttt_val_best_accuracy = ttt_result.val_best_accuracy
                    ttt_best_epoch = ttt_result.best_epoch
                    ttt_stopped_early = ttt_result.stopped_early
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
        roc_auc = compute_tabpfn_roc_auc(y_test, y_proba)
        predict_seconds = time.time() - t1

        from sklearn.metrics import f1_score

        accuracy = float(np.mean(np.asarray(y_pred) == np.asarray(y_test)))
        f1 = float(f1_score(np.asarray(y_test), np.asarray(y_pred), average="weighted", zero_division=0))
        if pseudo_pool is not None and pseudo_pool.selected_count > 0:
            selected_true = np.asarray(y_test)[np.asarray(pseudo_pool.test_indices, dtype=int)]
            pseudo_precision_posthoc = float(np.mean(np.asarray(pseudo_pool.y) == selected_true))

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
            roc_auc=roc_auc,
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
            ttt_val_baseline_accuracy=ttt_val_baseline_accuracy,
            ttt_val_best_accuracy=ttt_val_best_accuracy,
            ttt_best_epoch=ttt_best_epoch,
            ttt_stopped_early=ttt_stopped_early,
            ttt_oom_fallback=ttt_oom_fallback,
            ttt_fallback_reason=ttt_fallback_reason,
            pseudo_enabled=pseudo_enabled,
            pseudo_source=pseudo_source,
            pseudo_selected_count=pseudo_selected_count,
            pseudo_selected_ratio=pseudo_selected_ratio,
            pseudo_conf_mean=pseudo_conf_mean,
            pseudo_margin_mean=pseudo_margin_mean,
            pseudo_loss_weight=pseudo_loss_weight,
            pseudo_class_hist=pseudo_class_hist,
            pseudo_threshold_mode=pseudo_threshold_mode,
            pseudo_precision_posthoc=pseudo_precision_posthoc,
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
            roc_auc=None,
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
                        roc_auc=None,
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

    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()
    skipped_df = result_df[result_df["status"] == "skip"].copy() if len(result_df) else pd.DataFrame()
    oom_fallback_df = (
        ok_df[truthy_column_mask(ok_df, "ttt_oom_fallback")].copy()
        if len(ok_df)
        else pd.DataFrame()
    )
    pseudo_enabled_count = int(truthy_column_mask(ok_df, "pseudo_enabled").sum()) if len(ok_df) else 0
    total_pseudo_selected_ok = (
        int(pd.to_numeric(ok_df["pseudo_selected_count"], errors="coerce").fillna(0).sum())
        if len(ok_df) and "pseudo_selected_count" in ok_df.columns
        else 0
    )
    avg_pseudo_selected_ratio_ok = (
        pd.to_numeric(ok_df["pseudo_selected_ratio"], errors="coerce").mean()
        if len(ok_df) and "pseudo_selected_ratio" in ok_df.columns
        else None
    )
    avg_pseudo_conf_mean_ok = (
        pd.to_numeric(ok_df["pseudo_conf_mean"], errors="coerce").mean()
        if len(ok_df) and "pseudo_conf_mean" in ok_df.columns
        else None
    )
    avg_pseudo_margin_mean_ok = (
        pd.to_numeric(ok_df["pseudo_margin_mean"], errors="coerce").mean()
        if len(ok_df) and "pseudo_margin_mean" in ok_df.columns
        else None
    )
    avg_pseudo_precision_posthoc_ok = (
        pd.to_numeric(ok_df["pseudo_precision_posthoc"], errors="coerce").mean()
        if len(ok_df) and "pseudo_precision_posthoc" in ok_df.columns
        else None
    )

    lines = [
        f"discovered_datasets: {len(dataset_dirs)}",
        f"processed_datasets: {len(result_df)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        f"skipped_count: {len(skipped_df)}",
        f"ttt_oom_fallback_count: {len(oom_fallback_df)}",
        f"pseudo_enabled_count: {pseudo_enabled_count}",
        f"total_pseudo_selected_ok: {total_pseudo_selected_ok}",
        f"avg_pseudo_selected_ratio_ok: {format_optional_float(avg_pseudo_selected_ratio_ok)}",
        f"avg_pseudo_conf_mean_ok: {format_optional_float(avg_pseudo_conf_mean_ok)}",
        f"avg_pseudo_margin_mean_ok: {format_optional_float(avg_pseudo_margin_mean_ok)}",
        f"avg_pseudo_precision_posthoc_ok: {format_optional_float(avg_pseudo_precision_posthoc_ok)}",
        (
            f"avg_accuracy_ok: {ok_df['accuracy'].mean():.6f}"
            if len(ok_df)
            else "avg_accuracy_ok: (none)"
        ),
        (
            f"avg_f1_ok: {ok_df['f1'].mean():.6f}"
            if len(ok_df)
            else "avg_f1_ok: (none)"
        ),
        (
            f"avg_roc_auc_ok: {ok_df['roc_auc'].mean():.6f}"
            if len(ok_df) and ok_df["roc_auc"].notna().any()
            else "avg_roc_auc_ok: (none)"
        ),
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
        "avg_roc_auc_ok",
        "avg_fit_seconds_ok",
        "avg_predict_seconds_ok",
        "avg_dataset_seconds_ok",
        "total_dataset_seconds_ok",
        "model_wall_seconds",
        "ttt_oom_fallback_count",
        "pseudo_enabled_count",
        "total_pseudo_selected_ok",
        "avg_pseudo_selected_ratio_ok",
        "avg_pseudo_conf_mean_ok",
        "avg_pseudo_margin_mean_ok",
        "avg_pseudo_precision_posthoc_ok",
    ):
        if column in summary_df.columns:
            summary_df[column] = pd.to_numeric(summary_df[column], errors="coerce")
    all_csv = out_dir / "all_models_summary.csv"
    summary_txt = out_dir / "summary.txt"
    summary_df.to_csv(all_csv, index=False)

    ok_df = summary_df[summary_df["status"] == "ok"].copy() if len(summary_df) else pd.DataFrame()
    failed_df = summary_df[summary_df["status"] == "fail"].copy() if len(summary_df) else pd.DataFrame()

    lines = [
        f"total_models: {len(summary_df)}",
        f"successful_models: {len(ok_df)}",
        f"failed_models: {len(failed_df)}",
        (
            f"average_avg_dataset_seconds_ok: {ok_df['avg_dataset_seconds_ok'].mean():.6f}"
            if len(ok_df)
            else "average_avg_dataset_seconds_ok: (none)"
        ),
        (
            f"average_avg_accuracy_ok: {ok_df['avg_accuracy_ok'].mean():.6f}"
            if len(ok_df)
            else "average_avg_accuracy_ok: (none)"
        ),
        (
            f"average_avg_f1_ok: {ok_df['avg_f1_ok'].mean():.6f}"
            if len(ok_df)
            else "average_avg_f1_ok: (none)"
        ),
        (
            f"average_avg_roc_auc_ok: {ok_df['avg_roc_auc_ok'].mean():.6f}"
            if len(ok_df) and ok_df["avg_roc_auc_ok"].notna().any()
            else "average_avg_roc_auc_ok: (none)"
        ),
        (
            f"total_pseudo_selected_ok: {int(ok_df['total_pseudo_selected_ok'].sum())}"
            if len(ok_df) and "total_pseudo_selected_ok" in ok_df.columns
            else "total_pseudo_selected_ok: 0"
        ),
        (
            f"average_avg_pseudo_selected_ratio_ok: {ok_df['avg_pseudo_selected_ratio_ok'].mean():.6f}"
            if len(ok_df) and ok_df["avg_pseudo_selected_ratio_ok"].notna().any()
            else "average_avg_pseudo_selected_ratio_ok: (none)"
        ),
        (
            f"average_avg_pseudo_conf_mean_ok: {ok_df['avg_pseudo_conf_mean_ok'].mean():.6f}"
            if len(ok_df) and ok_df["avg_pseudo_conf_mean_ok"].notna().any()
            else "average_avg_pseudo_conf_mean_ok: (none)"
        ),
        (
            f"average_avg_pseudo_margin_mean_ok: {ok_df['avg_pseudo_margin_mean_ok'].mean():.6f}"
            if len(ok_df) and ok_df["avg_pseudo_margin_mean_ok"].notna().any()
            else "average_avg_pseudo_margin_mean_ok: (none)"
        ),
        (
            f"average_avg_pseudo_precision_posthoc_ok: {ok_df['avg_pseudo_precision_posthoc_ok'].mean():.6f}"
            if len(ok_df) and ok_df["avg_pseudo_precision_posthoc_ok"].notna().any()
            else "average_avg_pseudo_precision_posthoc_ok: (none)"
        ),
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
            "epoch-shuffled chunk TTT, optional transductive pseudo-label "
            "query loss, and AMD/ROCm multi-GPU workers."
        )
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--checkpoint-version", default=DEFAULT_CHECKPOINT_VERSION)
    parser.add_argument("--out-dir", default="1b_result/1d_pesudo_ttt_epoch8_chunk")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--gpu-groups", default="1,2,3")
    parser.add_argument("--n-estimators", type=int, default=32)
    parser.add_argument("--batch-size", type=parse_optional_int, default=8)
    parser.add_argument("--kv-cache", type=parse_kv_cache, default=False)
    parser.add_argument("--use-amp", type=parse_auto_bool, default="auto")
    parser.add_argument("--use-fa3", type=parse_auto_bool, default="auto")
    parser.add_argument("--offload-mode", choices=["auto", "gpu", "cpu", "disk"], default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-datasets", type=int, default=None)
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
    parser.add_argument("--ttt-lr", type=float, default=5e-6)
    parser.add_argument("--ttt-scheduler", choices=["constant"], default="constant")
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
    parser.add_argument("--ttt-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--ttt-epochs",
        "--ttt-steps",
        dest="ttt_epochs",
        type=int,
        default=8,
        help="Number of epoch-shuffled chunk TTT passes. --ttt-steps is kept as a compatibility alias.",
    )
    parser.add_argument("--ttt-max-chunk-size", type=int, default=20_000)
    parser.add_argument("--ttt-min-chunk-size", type=int, default=50)
    parser.add_argument("--ttt-query-ratio", type=float, default=0.2)
    parser.add_argument("--ttt-n-estimators-finetune", type=int, default=32)
    parser.add_argument("--ttt-early-stopping", type=parse_bool, default=True)
    parser.add_argument("--ttt-patience", type=int, default=8)
    parser.add_argument("--ttt-min-delta", type=float, default=1e-4)
    parser.add_argument("--ttt-validation-fraction", type=float, default=0.1)
    parser.add_argument("--ttt-validation-n-estimators", type=int, default=8)
    parser.add_argument("--ttt-freeze-col", type=parse_bool, default=False)
    parser.add_argument("--ttt-freeze-row", type=parse_bool, default=False)
    parser.add_argument(
        "--ttt-pseudo-labels",
        choices=["off", "transductive"],
        default="off",
        help="Enable fixed teacher pseudo labels from X_test features for TTT query loss only.",
    )
    parser.add_argument("--ttt-pseudo-target-val-precision", type=float, default=0.90)
    parser.add_argument("--ttt-pseudo-min-confidence", type=float, default=0.90)
    parser.add_argument("--ttt-pseudo-min-margin", type=float, default=0.15)
    parser.add_argument("--ttt-pseudo-max-ratio", type=float, default=0.30)
    parser.add_argument("--ttt-pseudo-query-ratio", type=float, default=0.50)
    parser.add_argument("--ttt-pseudo-loss-weight", type=float, default=0.20)
    parser.add_argument(
        "--ttt-pseudo-refresh",
        choices=["never"],
        default="never",
        help="Pseudo labels are generated once by the initial teacher; iterative refresh is not implemented.",
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
