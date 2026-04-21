#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
import queue
import re
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_DATA_ROOT = Path("data178")
DEFAULT_MODEL_PATH = "tabicl-classifier-v1.1-20250506.ckpt"
DEFAULT_CHECKPOINT_VERSION = "tabicl-classifier-v1.1-20250506.ckpt"
CLASSIFICATION_TASKS = {"binclass", "multiclass"}
CATEGORICAL_MISSING_TOKEN = "__tabicl_missing__"
DEFAULT_PSEUDO_MAX_ERROR_RATE = 0.01
DEFAULT_PSEUDO_ROUNDS = 2
DEFAULT_CLASS_AWARE_PSEUDO_TOLERANCE = True
CLASS_AWARE_PSEUDO_ERROR_RATE_CAP = 0.08

np = None
pd = None


@dataclass
class ResultRow:
    dataset_name: str
    dataset_dir: str
    task_type: Optional[str]
    n_train: int
    n_train_initial: int
    n_train_final: int
    n_val: int
    n_test: int
    n_features: int
    n_classes: int
    accuracy: Optional[float]
    acc_round1: Optional[float]
    acc_round2: Optional[float]
    acc_delta: Optional[float]
    acc_improved: Optional[bool]
    train_ratio_before: Optional[float]
    train_ratio_after: Optional[float]
    pass1_acc: Optional[float]
    entropy_threshold: Optional[str]
    pseudo_selected: int
    pseudo_correct: int
    pseudo_wrong: int
    pseudo_precision: Optional[float]
    pseudo_error_rate: Optional[float]
    effective_pseudo_max_error_rate: Optional[float]
    pseudo_rounds: int
    entropy_file: str
    fit_seconds: float
    predict_seconds: float
    status: str
    error: Optional[str]


@dataclass
class ModelSummaryRow:
    model_name: str
    model_path: str
    gpu_id: int
    datasets_discovered: int
    ok_count: int
    failed_count: int
    skipped_count: int
    avg_accuracy_ok: Optional[float]
    avg_acc_round1_ok: Optional[float]
    avg_acc_round2_ok: Optional[float]
    avg_acc_delta_ok: Optional[float]
    avg_fit_seconds_ok: Optional[float]
    avg_predict_seconds_ok: Optional[float]
    avg_dataset_seconds_ok: Optional[float]
    avg_pass1_acc_ok: Optional[float]
    avg_train_ratio_before_ok: Optional[float]
    avg_train_ratio_after_ok: Optional[float]
    avg_pseudo_selected_ok: Optional[float]
    total_pseudo_selected_ok: int
    improved_dataset_count: int
    degraded_dataset_count: int
    unchanged_dataset_count: int
    avg_pseudo_precision_ok: Optional[float]
    avg_pseudo_error_rate_ok: Optional[float]
    total_dataset_seconds_ok: float
    model_wall_seconds: float
    status: str
    error: Optional[str]
    failed_datasets: str


def ensure_runtime_deps() -> None:
    global np
    global pd

    if np is None or pd is None:
        import numpy as _np
        import pandas as _pd

        np = _np
        pd = _pd


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


def parse_kv_cache(value: str) -> bool | str:
    lowered = value.strip().lower()
    if lowered in {"false", "0", "no", "off"}:
        return False
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"kv", "repr"}:
        return lowered
    raise argparse.ArgumentTypeError("kv_cache must be one of: false, true, kv, repr")


def resolve_data_root_path(data_root: str | os.PathLike[str]) -> Path:
    raw_path = Path(data_root).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    candidate_paths = [
        raw_path,
        Path.cwd() / raw_path,
        REPO_ROOT / raw_path,
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / raw_path).resolve()


def apply_worker_environment_updates(gpu_id: int) -> str:
    gpu_id_str = str(gpu_id)
    # Set visibility vars for both CUDA and ROCm stacks so each worker sees one GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str
    os.environ["ROCR_VISIBLE_DEVICES"] = gpu_id_str
    os.environ["HIP_VISIBLE_DEVICES"] = gpu_id_str
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    return "cuda:0"


def parse_gpu_id_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


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


def dataset_entropy_key(dataset_dir: Path) -> str:
    dataset_hash = hashlib.md5(str(dataset_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{dataset_dir.name}__{dataset_hash}"


def make_empty_result_row(
    dataset_dir: Path,
    *,
    task_type: Optional[str],
    status: str,
    error: Optional[str],
) -> ResultRow:
    return ResultRow(
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir.as_posix(),
        task_type=task_type,
        n_train=0,
        n_train_initial=0,
        n_train_final=0,
        n_val=0,
        n_test=0,
        n_features=0,
        n_classes=0,
        accuracy=None,
        acc_round1=None,
        acc_round2=None,
        acc_delta=None,
        acc_improved=None,
        train_ratio_before=None,
        train_ratio_after=None,
        pass1_acc=None,
        entropy_threshold=None,
        pseudo_selected=0,
        pseudo_correct=0,
        pseudo_wrong=0,
        pseudo_precision=None,
        pseudo_error_rate=None,
        effective_pseudo_max_error_rate=None,
        pseudo_rounds=0,
        entropy_file="",
        fit_seconds=0.0,
        predict_seconds=0.0,
        status=status,
        error=error,
    )


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
        accuracy = float(row.accuracy) if row.accuracy is not None else float("nan")
        pass1_acc = float(row.pass1_acc) if row.pass1_acc is not None else float("nan")
        return (
            f"{prefix} [ok] {row.dataset_name} accuracy={accuracy:.6f} "
            f"pass1_acc={pass1_acc:.6f} pseudo_selected={row.pseudo_selected}"
        )
    if row.status == "skip":
        return f"{prefix} [skip] {row.dataset_name} reason={row.error}"
    return f"{prefix} [fail] {row.dataset_name} error={row.error}"


def entropy_from_proba(proba, eps: float = 1e-12) -> np.ndarray:
    ensure_runtime_deps()

    probs = np.asarray(proba, dtype=np.float64)
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return (-(probs * np.log(probs)).sum(axis=1)).astype(np.float32)


def choose_entropy_threshold(
    entropy,
    correct_mask,
    max_error_rate: float,
) -> tuple[float, np.ndarray, int, int, float]:
    ensure_runtime_deps()

    entropy_arr = np.asarray(entropy, dtype=np.float64).reshape(-1)
    correct_arr = np.asarray(correct_mask, dtype=bool).reshape(-1)
    if entropy_arr.size == 0:
        return float("-inf"), np.zeros(0, dtype=bool), 0, 0, 0.0

    bounded_error_rate = float(np.clip(max_error_rate, 0.0, 1.0))
    uniq_entropy, inverse = np.unique(entropy_arr, return_inverse=True)
    bucket_selected = np.bincount(inverse, minlength=uniq_entropy.size).astype(np.int64)
    bucket_correct = np.bincount(
        inverse,
        weights=correct_arr.astype(np.int64),
        minlength=uniq_entropy.size,
    ).astype(np.int64)

    cum_selected = np.cumsum(bucket_selected)
    cum_correct = np.cumsum(bucket_correct)
    cum_wrong = cum_selected - cum_correct
    cum_error_rate = cum_wrong / np.maximum(cum_selected, 1)
    feasible = np.where(cum_error_rate <= (bounded_error_rate + 1e-12))[0]
    if feasible.size == 0:
        return float("-inf"), np.zeros_like(correct_arr), 0, 0, 0.0

    best_idx = int(feasible[0])
    best_correct = int(cum_correct[best_idx])
    best_selected = int(cum_selected[best_idx])
    for idx in feasible[1:]:
        correct_count = int(cum_correct[idx])
        selected_count = int(cum_selected[idx])
        if correct_count > best_correct or (
            correct_count == best_correct and selected_count > best_selected
        ):
            best_idx = int(idx)
            best_correct = correct_count
            best_selected = selected_count

    threshold = float(uniq_entropy[best_idx])
    selected_mask = entropy_arr <= threshold
    selected_count = int(selected_mask.sum())
    selected_correct = int(correct_arr[selected_mask].sum()) if selected_count > 0 else 0
    selected_precision = (
        float(selected_correct / selected_count)
        if selected_count > 0
        else 0.0
    )
    return threshold, selected_mask, selected_count, selected_correct, selected_precision


def stringify_class_label(label: Any) -> str:
    ensure_runtime_deps()

    if isinstance(label, np.generic):
        label = label.item()
    return str(label)


def serialize_threshold_by_class(threshold_by_class: dict[str, Optional[float]]) -> str:
    return json.dumps(threshold_by_class, ensure_ascii=False, sort_keys=True)


def resolve_effective_pseudo_error_rate(
    base_rate: float,
    n_classes: int,
    class_aware_enabled: bool,
) -> float:
    bounded_base_rate = min(max(float(base_rate), 0.0), 1.0)
    if not class_aware_enabled:
        return bounded_base_rate

    bounded_class_count = max(int(n_classes), 0)
    if bounded_class_count <= 2:
        multiplier = 1.0
    elif bounded_class_count <= 5:
        multiplier = 2.0
    elif bounded_class_count <= 10:
        multiplier = 3.0
    else:
        multiplier = 4.0

    return float(
        min(
            bounded_base_rate * multiplier,
            CLASS_AWARE_PSEUDO_ERROR_RATE_CAP,
        )
    )


def choose_entropy_thresholds_by_class(
    entropy,
    predicted_labels,
    correct_mask,
    max_error_rate: float,
) -> tuple[dict[str, Optional[float]], np.ndarray, int, int, float]:
    ensure_runtime_deps()

    entropy_arr = np.asarray(entropy, dtype=np.float64).reshape(-1)
    predicted_arr = np.asarray(predicted_labels)
    correct_arr = np.asarray(correct_mask, dtype=bool).reshape(-1)
    if entropy_arr.size == 0:
        return {}, np.zeros(0, dtype=bool), 0, 0, 0.0

    selected_mask_all = np.zeros(entropy_arr.shape[0], dtype=bool)
    threshold_by_class: dict[str, Optional[float]] = {}
    selected_total = 0
    correct_total = 0

    for class_label in pd.unique(pd.Series(predicted_arr)):
        class_mask = np.asarray(predicted_arr == class_label, dtype=bool)
        class_key = stringify_class_label(class_label)
        class_entropy = entropy_arr[class_mask]
        class_correct = correct_arr[class_mask]
        (
            threshold,
            class_selected_mask,
            class_selected_count,
            class_selected_correct,
            _class_selected_precision,
        ) = choose_entropy_threshold(
            entropy=class_entropy,
            correct_mask=class_correct,
            max_error_rate=max_error_rate,
        )

        threshold_by_class[class_key] = (
            float(threshold) if np.isfinite(threshold) else None
        )
        if class_selected_count > 0:
            class_indices = np.flatnonzero(class_mask)
            selected_mask_all[class_indices[class_selected_mask]] = True
            selected_total += int(class_selected_count)
            correct_total += int(class_selected_correct)

    selected_precision = (
        float(correct_total / selected_total)
        if selected_total > 0
        else 0.0
    )
    return threshold_by_class, selected_mask_all, selected_total, correct_total, selected_precision


def append_feature_rows(X_existing, X_new):
    ensure_runtime_deps()

    if isinstance(X_existing, pd.DataFrame):
        return pd.concat([X_existing, X_new], axis=0, ignore_index=True)
    return np.concatenate([np.asarray(X_existing), np.asarray(X_new)], axis=0)


def predict_labels_from_proba(classifier, proba: np.ndarray, X_test):
    ensure_runtime_deps()

    classes = getattr(classifier, "classes_", None)
    if classes is not None:
        class_values = np.asarray(classes)
        pred_idx = np.asarray(np.argmax(proba, axis=1), dtype=np.int64)
        return class_values[pred_idx]
    return np.asarray(classifier.predict(X_test))


def run_entropy_two_pass_pseudo(
    classifier,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    pseudo_max_error_rate: float,
    effective_pseudo_max_error_rate: float | None = None,
    pseudo_rounds: int,
    entropy_save_dir: Path | None = None,
    dataset_key: str | None = None,
) -> tuple[np.ndarray, float, float, dict[str, Any]]:
    ensure_runtime_deps()

    rounds = max(1, int(pseudo_rounds))
    effective_error_rate = (
        float(effective_pseudo_max_error_rate)
        if effective_pseudo_max_error_rate is not None
        else float(np.clip(pseudo_max_error_rate, 0.0, 1.0))
    )
    # Avoid eagerly duplicating large train blocks; we only materialize a new
    # object once pseudo-labeled rows are appended.
    cur_X_train = X_train
    cur_y_train = np.asarray(y_train)
    X_test_values = X_test
    y_test_values = np.asarray(y_test)
    added_mask = np.zeros(int(len(y_test_values)), dtype=bool)

    total_fit_time = 0.0
    total_pred_time = 0.0
    y_pred_last = None
    pass1_acc = float("nan")
    entropy_files: list[str] = []
    total_selected = 0
    total_correct = 0
    total_wrong = 0
    last_threshold_by_class: dict[str, Optional[float]] = {}
    last_threshold_json = ""
    last_precision = float("nan")
    last_error_rate = float("nan")

    for round_idx in range(1, rounds + 1):
        fit_started = time.perf_counter()
        classifier.fit(cur_X_train, cur_y_train)
        total_fit_time += time.perf_counter() - fit_started

        pred_started = time.perf_counter()
        proba = np.asarray(classifier.predict_proba(X_test_values))
        y_pred = np.asarray(predict_labels_from_proba(classifier, proba, X_test_values))
        total_pred_time += time.perf_counter() - pred_started
        y_pred_last = y_pred

        entropy = entropy_from_proba(proba)
        correct = np.asarray(y_pred == y_test_values)
        if round_idx == 1:
            pass1_acc = float(correct.mean()) if correct.size > 0 else float("nan")

        (
            threshold_by_class,
            selected_mask_all,
            selected_cnt_all,
            _selected_correct_all,
            selected_precision_all,
        ) = choose_entropy_thresholds_by_class(
            entropy=entropy,
            predicted_labels=y_pred,
            correct_mask=correct,
            max_error_rate=effective_error_rate,
        )
        selected_new_mask = selected_mask_all & (~added_mask)
        selected_cnt = int(selected_new_mask.sum())
        selected_correct = int(correct[selected_new_mask].sum()) if selected_cnt > 0 else 0
        selected_wrong = int(selected_cnt - selected_correct)
        selected_precision = float(selected_correct / selected_cnt) if selected_cnt > 0 else 0.0

        last_threshold_by_class = dict(threshold_by_class)
        last_threshold_json = serialize_threshold_by_class(last_threshold_by_class)
        last_precision = selected_precision_all
        last_error_rate = float(1.0 - selected_precision_all) if selected_cnt_all > 0 else 0.0

        if entropy_save_dir is not None and dataset_key is not None:
            entropy_save_dir.mkdir(parents=True, exist_ok=True)
            entropy_save_path = entropy_save_dir / f"{dataset_key}__round{round_idx}_entropy.npz"
            np.savez_compressed(
                entropy_save_path,
                round_idx=np.asarray(round_idx, dtype=np.int32),
                entropy=entropy.astype(np.float32),
                y_pred=np.asarray(y_pred),
                y_test=np.asarray(y_test_values),
                selected_mask=selected_mask_all.astype(np.uint8),
                selected_new_mask=selected_new_mask.astype(np.uint8),
                selected_correct=np.asarray(correct[selected_mask_all]).astype(np.uint8),
                threshold_by_class_json=np.asarray(last_threshold_json),
                selected_count=np.asarray(selected_cnt_all, dtype=np.int64),
                selected_count_new=np.asarray(selected_cnt, dtype=np.int64),
                selected_precision=np.asarray(selected_precision_all, dtype=np.float32),
                selected_error_rate=np.asarray(last_error_rate, dtype=np.float32),
                effective_pseudo_max_error_rate=np.asarray(effective_error_rate, dtype=np.float32),
            )
            entropy_files.append(str(entropy_save_path))

        if selected_cnt > 0:
            if isinstance(X_test_values, pd.DataFrame):
                selected_X = X_test_values.loc[selected_new_mask].copy()
            else:
                selected_X = np.asarray(X_test_values)[selected_new_mask]
            cur_X_train = append_feature_rows(cur_X_train, selected_X)
            cur_y_train = np.concatenate([cur_y_train, y_pred[selected_new_mask]], axis=0)
            added_mask[selected_new_mask] = True
            total_selected += selected_cnt
            total_correct += selected_correct
            total_wrong += selected_wrong

    if y_pred_last is None:
        raise RuntimeError("Pseudo-label inference produced no predictions")

    final_train_count = int(len(cur_y_train))
    test_count = int(len(y_test_values))
    total_count = final_train_count + test_count
    final_train_ratio = float(final_train_count / total_count) if total_count > 0 else float("nan")
    meta = {
        "threshold_by_class": dict(last_threshold_by_class),
        "threshold_by_class_json": last_threshold_json,
        "selected_cnt": total_selected,
        "selected_correct": total_correct,
        "selected_wrong": total_wrong,
        "selected_precision": float(total_correct / total_selected) if total_selected > 0 else 0.0,
        "selected_error_rate": float(total_wrong / total_selected) if total_selected > 0 else 0.0,
        "effective_pseudo_max_error_rate": effective_error_rate,
        "train_ratio_final": final_train_ratio,
        "pass1_acc": pass1_acc,
        "rounds": rounds,
        "last_round_precision": last_precision,
        "last_round_error_rate": last_error_rate,
        "entropy_files": entropy_files,
        "n_train_final": final_train_count,
    }
    return y_pred_last, total_fit_time, total_pred_time, meta


def load_dataset_info(dataset_dir: Path) -> dict | None:
    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        return None
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None


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
    release_classifier_dataset_state(classifier, keep_model=False)


def release_classifier_dataset_state(classifier: Any, *, keep_model: bool) -> None:
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

    attr_names = ["model_kv_cache_", "ensemble_generator_", "X_encoder_", "y_encoder_"]
    if not keep_model:
        attr_names.append("model_")

    for attr_name in attr_names:
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
    avg_acc_round1_ok = float(ok_df["acc_round1"].mean()) if len(ok_df) else None
    avg_acc_round2_ok = float(ok_df["acc_round2"].mean()) if len(ok_df) else None
    avg_acc_delta_ok = float(ok_df["acc_delta"].mean()) if len(ok_df) else None
    avg_pass1_acc_ok = float(ok_df["pass1_acc"].mean()) if len(ok_df) else None
    avg_train_ratio_before_ok = float(ok_df["train_ratio_before"].mean()) if len(ok_df) else None
    avg_train_ratio_after_ok = float(ok_df["train_ratio_after"].mean()) if len(ok_df) else None
    avg_pseudo_selected_ok = float(ok_df["pseudo_selected"].mean()) if len(ok_df) else None
    total_pseudo_selected_ok = int(ok_df["pseudo_selected"].sum()) if len(ok_df) else 0
    improved_dataset_count = int((ok_df["acc_delta"] > 0).sum()) if len(ok_df) else 0
    degraded_dataset_count = int((ok_df["acc_delta"] < 0).sum()) if len(ok_df) else 0
    unchanged_dataset_count = int((ok_df["acc_delta"] == 0).sum()) if len(ok_df) else 0
    avg_pseudo_precision_ok = float(ok_df["pseudo_precision"].mean()) if len(ok_df) else None
    avg_pseudo_error_rate_ok = compute_avg_pseudo_error_rate_ok(result_df)
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
            avg_accuracy_ok=None,
            avg_acc_round1_ok=None,
            avg_acc_round2_ok=None,
            avg_acc_delta_ok=None,
            avg_fit_seconds_ok=None,
            avg_predict_seconds_ok=None,
            avg_dataset_seconds_ok=None,
            avg_pass1_acc_ok=None,
            avg_train_ratio_before_ok=None,
            avg_train_ratio_after_ok=None,
            avg_pseudo_selected_ok=None,
            total_pseudo_selected_ok=0,
            improved_dataset_count=0,
            degraded_dataset_count=0,
            unchanged_dataset_count=0,
            avg_pseudo_precision_ok=None,
            avg_pseudo_error_rate_ok=None,
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
        avg_accuracy_ok=(float(ok_df["accuracy"].mean()) if len(ok_df) else None),
        avg_acc_round1_ok=avg_acc_round1_ok,
        avg_acc_round2_ok=avg_acc_round2_ok,
        avg_acc_delta_ok=avg_acc_delta_ok,
        avg_fit_seconds_ok=avg_fit_seconds_ok,
        avg_predict_seconds_ok=avg_predict_seconds_ok,
        avg_dataset_seconds_ok=avg_dataset_seconds_ok,
        avg_pass1_acc_ok=avg_pass1_acc_ok,
        avg_train_ratio_before_ok=avg_train_ratio_before_ok,
        avg_train_ratio_after_ok=avg_train_ratio_after_ok,
        avg_pseudo_selected_ok=avg_pseudo_selected_ok,
        total_pseudo_selected_ok=total_pseudo_selected_ok,
        improved_dataset_count=improved_dataset_count,
        degraded_dataset_count=degraded_dataset_count,
        unchanged_dataset_count=unchanged_dataset_count,
        avg_pseudo_precision_ok=avg_pseudo_precision_ok,
        avg_pseudo_error_rate_ok=avg_pseudo_error_rate_ok,
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
    *,
    pseudo_max_error_rate: float,
    class_aware_pseudo_tolerance: bool,
    pseudo_rounds: int,
    entropy_save_dir: Path | None = None,
) -> ResultRow:
    ensure_runtime_deps()

    task_type: Optional[str] = None
    try:
        info = load_dataset_info(dataset_dir)
        task_type = str(info.get("task_type", "")).lower() if info else None
        if task_type not in CLASSIFICATION_TASKS:
            return make_empty_result_row(
                dataset_dir,
                task_type=task_type,
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
        if val_split is not None:
            X_val, y_val = load_split(
                val_split[0],
                val_split[1],
                val_split[2],
                context=f"{dataset_dir.name}-val",
            )
            val_count = int(len(y_val))
            X_train = pd.concat([X_train, X_val], axis=0, ignore_index=True)
            y_train = np.concatenate([np.asarray(y_train), np.asarray(y_val)], axis=0)
        train_count = int(len(y_train))

        X_test, y_test = load_split(
            test_split[0],
            test_split[1],
            test_split[2],
            context=f"{dataset_dir.name}-test",
        )
        test_count = int(len(y_test))

        classes = pd.unique(pd.Series(np.concatenate([np.asarray(y_train), np.asarray(y_test)], axis=0)))
        n_classes = int(len(classes))
        effective_pseudo_max_error_rate = resolve_effective_pseudo_error_rate(
            pseudo_max_error_rate,
            n_classes=n_classes,
            class_aware_enabled=class_aware_pseudo_tolerance,
        )
        total_count_before = train_count + test_count
        train_ratio_before = (
            float(train_count / total_count_before)
            if total_count_before > 0
            else float("nan")
        )
        dataset_key = dataset_entropy_key(dataset_dir) if entropy_save_dir is not None else None

        y_pred, fit_seconds, predict_seconds, pseudo_meta = run_entropy_two_pass_pseudo(
            classifier,
            X_train,
            y_train,
            X_test,
            y_test,
            pseudo_max_error_rate=pseudo_max_error_rate,
            effective_pseudo_max_error_rate=effective_pseudo_max_error_rate,
            pseudo_rounds=pseudo_rounds,
            entropy_save_dir=entropy_save_dir,
            dataset_key=dataset_key,
        )

        accuracy = float(np.mean(np.asarray(y_pred) == np.asarray(y_test)))
        acc_round1 = float(pseudo_meta["pass1_acc"])
        acc_round2 = accuracy
        acc_delta = float(acc_round2 - acc_round1)
        acc_improved = bool(acc_delta > 0.0)
        entropy_files = pseudo_meta.get("entropy_files", [])
        entropy_file = "|".join(entropy_files) if entropy_files else ""

        return ResultRow(
            dataset_name=dataset_dir.name,
            dataset_dir=dataset_dir.as_posix(),
            task_type=task_type,
            n_train=int(pseudo_meta["n_train_final"]),
            n_train_initial=train_count,
            n_train_final=int(pseudo_meta["n_train_final"]),
            n_val=val_count,
            n_test=test_count,
            n_features=int(X_train.shape[1]),
            n_classes=n_classes,
            accuracy=accuracy,
            acc_round1=acc_round1,
            acc_round2=acc_round2,
            acc_delta=acc_delta,
            acc_improved=acc_improved,
            train_ratio_before=train_ratio_before,
            train_ratio_after=float(pseudo_meta["train_ratio_final"]),
            pass1_acc=acc_round1,
            entropy_threshold=str(pseudo_meta["threshold_by_class_json"]),
            pseudo_selected=int(pseudo_meta["selected_cnt"]),
            pseudo_correct=int(pseudo_meta["selected_correct"]),
            pseudo_wrong=int(pseudo_meta["selected_wrong"]),
            pseudo_precision=float(pseudo_meta["selected_precision"]),
            pseudo_error_rate=float(pseudo_meta["selected_error_rate"]),
            effective_pseudo_max_error_rate=float(pseudo_meta["effective_pseudo_max_error_rate"]),
            pseudo_rounds=int(pseudo_meta["rounds"]),
            entropy_file=entropy_file,
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            status="ok",
            error=None,
        )
    except Exception as exc:
        return make_empty_result_row(
            dataset_dir,
            task_type=task_type,
            status="fail",
            error=f"{type(exc).__name__}: {exc}",
        )


def worker_main(
    worker_id: int,
    gpu_id: int,
    assigned_dataset_dirs: List[str],
    ready_queue,
    start_event,
    worker_out_csv: str,
    model_kwargs: Dict,
    pseudo_max_error_rate: float,
    class_aware_pseudo_tolerance: bool,
    pseudo_rounds: int,
    entropy_save_dir: str,
    verbose: bool,
) -> None:
    try:
        ensure_runtime_deps()
        device_str = apply_worker_environment_updates(gpu_id)

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
        preload_model_once(classifier, f"worker {worker_id} | gpu {gpu_id}", verbose)

        ready_queue.put(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
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
                pseudo_max_error_rate=pseudo_max_error_rate,
                class_aware_pseudo_tolerance=class_aware_pseudo_tolerance,
                pseudo_rounds=pseudo_rounds,
                entropy_save_dir=Path(entropy_save_dir) if entropy_save_dir else None,
            )
            rows.append(row)
            print(
                format_dataset_result_log(
                    f"worker {worker_id} | gpu {gpu_id}",
                    row,
                ),
                flush=True,
            )

            release_classifier_dataset_state(classifier, keep_model=True)
            force_memory_cleanup(device_str)

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
                {
                    "dataset_name": f"__WORKER_CRASH__{worker_id}",
                    "dataset_dir": "__worker__",
                    "task_type": None,
                    "n_train": 0,
                    "n_train_initial": 0,
                    "n_train_final": 0,
                    "n_val": 0,
                    "n_test": 0,
                    "n_features": 0,
                    "n_classes": 0,
                    "accuracy": None,
                    "acc_round1": None,
                    "acc_round2": None,
                    "acc_delta": None,
                    "acc_improved": None,
                    "train_ratio_before": None,
                    "train_ratio_after": None,
                    "pass1_acc": None,
                    "entropy_threshold": None,
                    "pseudo_selected": 0,
                    "pseudo_correct": 0,
                    "pseudo_wrong": 0,
                    "pseudo_precision": None,
                    "pseudo_error_rate": None,
                    "effective_pseudo_max_error_rate": None,
                    "pseudo_rounds": 0,
                    "entropy_file": "",
                    "fit_seconds": 0.0,
                    "predict_seconds": 0.0,
                    "status": "fail",
                    "error": traceback.format_exc(),
                }
            ]
        )
        crash_row.to_csv(worker_out_csv, index=False)


def model_pool_worker_main(
    worker_id: int,
    gpu_id: int,
    dataset_dirs: List[str],
    ready_queue,
    task_queue,
    result_queue,
    base_model_kwargs: Dict,
    pseudo_max_error_rate: float,
    class_aware_pseudo_tolerance: bool,
    pseudo_rounds: int,
    verbose: bool,
) -> None:
    device_str = "cuda:0"
    worker_label = f"worker {worker_id} | gpu {gpu_id}"

    try:
        ensure_runtime_deps()
        device_str = apply_worker_environment_updates(gpu_id)

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
            entropy_save_dir = Path(task["entropy_save_dir"]) if task.get("entropy_save_dir") else None
            started_at = time.time()
            classifier = None

            try:
                worker_kwargs = dict(base_model_kwargs)
                worker_kwargs["device"] = device_str
                worker_kwargs["model_path"] = normalize_model_path(str(model_path))
                classifier = TabICLClassifier(**worker_kwargs)
                preload_model_once(classifier, worker_label, verbose)

                rows: List[ResultRow] = []
                for dataset_dir in resolved_dataset_dirs:
                    row = evaluate_one_dataset(
                        classifier,
                        dataset_dir,
                        pseudo_max_error_rate=pseudo_max_error_rate,
                        class_aware_pseudo_tolerance=class_aware_pseudo_tolerance,
                        pseudo_rounds=pseudo_rounds,
                        entropy_save_dir=entropy_save_dir,
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

                    release_classifier_dataset_state(classifier, keep_model=True)
                    force_memory_cleanup(device_str)

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
                        "rows": [asdict(row) for row in rows],
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
                        "rows": [],
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


def format_optional_metric(value: Any) -> str:
    if value is None:
        return "(none)"
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if numeric != numeric:
        return "nan"
    return f"{numeric:.6f}"


def filter_ok_rows_with_pseudo_selection(result_df: pd.DataFrame) -> pd.DataFrame:
    ensure_runtime_deps()

    if result_df.empty:
        return result_df.iloc[0:0].copy()

    pseudo_selected = pd.to_numeric(result_df["pseudo_selected"], errors="coerce")
    return result_df[
        (result_df["status"] == "ok")
        & (pseudo_selected > 0)
    ].copy()


def compute_avg_pseudo_error_rate_ok(result_df: pd.DataFrame) -> Optional[float]:
    ensure_runtime_deps()

    selected_ok_df = filter_ok_rows_with_pseudo_selection(result_df)
    if selected_ok_df.empty:
        return None

    pseudo_error_rate = pd.to_numeric(selected_ok_df["pseudo_error_rate"], errors="coerce")
    if pseudo_error_rate.isna().all():
        return None
    return float(pseudo_error_rate.mean())


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

    lines = [
        f"discovered_datasets: {len(dataset_dirs)}",
        f"processed_datasets: {len(result_df)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        f"skipped_count: {len(skipped_df)}",
        f"avg_accuracy_ok: {format_optional_metric(ok_df['accuracy'].mean() if len(ok_df) else None)}",
        f"avg_acc_round1_ok: {format_optional_metric(ok_df['acc_round1'].mean() if len(ok_df) else None)}",
        f"avg_acc_round2_ok: {format_optional_metric(ok_df['acc_round2'].mean() if len(ok_df) else None)}",
        f"avg_acc_delta_ok: {format_optional_metric(ok_df['acc_delta'].mean() if len(ok_df) else None)}",
        f"improved_dataset_count: {int((ok_df['acc_delta'] > 0).sum()) if len(ok_df) else 0}",
        f"degraded_dataset_count: {int((ok_df['acc_delta'] < 0).sum()) if len(ok_df) else 0}",
        f"unchanged_dataset_count: {int((ok_df['acc_delta'] == 0).sum()) if len(ok_df) else 0}",
        f"avg_pass1_acc_ok: {format_optional_metric(ok_df['pass1_acc'].mean() if len(ok_df) else None)}",
        f"avg_train_ratio_before_ok: {format_optional_metric(ok_df['train_ratio_before'].mean() if len(ok_df) else None)}",
        f"avg_train_ratio_after_ok: {format_optional_metric(ok_df['train_ratio_after'].mean() if len(ok_df) else None)}",
        f"avg_pseudo_selected_ok: {format_optional_metric(ok_df['pseudo_selected'].mean() if len(ok_df) else None)}",
        f"total_pseudo_selected_ok: {int(ok_df['pseudo_selected'].sum()) if len(ok_df) else 0}",
        f"avg_pseudo_precision_ok: {format_optional_metric(ok_df['pseudo_precision'].mean() if len(ok_df) else None)}",
        f"avg_pseudo_error_rate_ok: {format_optional_metric(compute_avg_pseudo_error_rate_ok(result_df))}",
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

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_outputs(
    out_dir: Path,
    result_df: pd.DataFrame,
    dataset_dirs: List[Path],
    wall_seconds: float,
) -> tuple[Path, Path]:
    ensure_runtime_deps()

    out_dir.mkdir(parents=True, exist_ok=True)
    all_csv = out_dir / "all_classification_results.csv"
    summary_txt = out_dir / "summary.txt"
    result_df.to_csv(all_csv, index=False)
    write_summary(summary_txt, result_df, dataset_dirs, wall_seconds)
    return all_csv, summary_txt


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
        "avg_acc_round1_ok",
        "avg_acc_round2_ok",
        "avg_acc_delta_ok",
        "avg_fit_seconds_ok",
        "avg_predict_seconds_ok",
        "avg_dataset_seconds_ok",
        "avg_pass1_acc_ok",
        "avg_train_ratio_before_ok",
        "avg_train_ratio_after_ok",
        "avg_pseudo_selected_ok",
        "avg_pseudo_precision_ok",
        "avg_pseudo_error_rate_ok",
        "total_dataset_seconds_ok",
        "total_pseudo_selected_ok",
        "improved_dataset_count",
        "degraded_dataset_count",
        "unchanged_dataset_count",
        "model_wall_seconds",
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
        f"average_avg_dataset_seconds_ok: {format_optional_metric(ok_df['avg_dataset_seconds_ok'].mean() if len(ok_df) else None)}",
        f"average_avg_accuracy_ok: {format_optional_metric(ok_df['avg_accuracy_ok'].mean() if len(ok_df) else None)}",
        f"average_avg_acc_round1_ok: {format_optional_metric(ok_df['avg_acc_round1_ok'].mean() if len(ok_df) else None)}",
        f"average_avg_acc_round2_ok: {format_optional_metric(ok_df['avg_acc_round2_ok'].mean() if len(ok_df) else None)}",
        f"average_avg_acc_delta_ok: {format_optional_metric(ok_df['avg_acc_delta_ok'].mean() if len(ok_df) else None)}",
        f"total_improved_dataset_count: {int(ok_df['improved_dataset_count'].sum()) if len(ok_df) else 0}",
        f"total_degraded_dataset_count: {int(ok_df['degraded_dataset_count'].sum()) if len(ok_df) else 0}",
        f"total_unchanged_dataset_count: {int(ok_df['unchanged_dataset_count'].sum()) if len(ok_df) else 0}",
        f"average_avg_pass1_acc_ok: {format_optional_metric(ok_df['avg_pass1_acc_ok'].mean() if len(ok_df) else None)}",
        f"average_avg_train_ratio_before_ok: {format_optional_metric(ok_df['avg_train_ratio_before_ok'].mean() if len(ok_df) else None)}",
        f"average_avg_train_ratio_after_ok: {format_optional_metric(ok_df['avg_train_ratio_after_ok'].mean() if len(ok_df) else None)}",
        f"average_avg_pseudo_selected_ok: {format_optional_metric(ok_df['avg_pseudo_selected_ok'].mean() if len(ok_df) else None)}",
        f"total_pseudo_selected_ok: {int(ok_df['total_pseudo_selected_ok'].sum()) if len(ok_df) else 0}",
        f"average_avg_pseudo_precision_ok: {format_optional_metric(ok_df['avg_pseudo_precision_ok'].mean() if len(ok_df) else None)}",
        f"average_avg_pseudo_error_rate_ok: {format_optional_metric(ok_df['avg_pseudo_error_rate_ok'].mean() if len(ok_df) else None)}",
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
            "Run TabICLv1.1 classification benchmarks on data178 with "
            "AMD/ROCm multi-GPU workers."
        )
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--checkpoint-version", default=DEFAULT_CHECKPOINT_VERSION)
    parser.add_argument("--out-dir", default="single1B_result/persudo_test_tabiclv1.1_0.01")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--batch-size", type=parse_optional_int, default=1)
    parser.add_argument("--kv-cache", type=parse_kv_cache, default=False)
    parser.add_argument("--use-amp", type=parse_auto_bool, default="auto")
    parser.add_argument("--use-fa3", type=parse_auto_bool, default="auto")
    parser.add_argument("--offload-mode", choices=["auto", "gpu", "cpu", "disk"], default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--pseudo-max-error-rate", type=float, default=DEFAULT_PSEUDO_MAX_ERROR_RATE)
    parser.add_argument(
        "--class-aware-pseudo-tolerance",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CLASS_AWARE_PSEUDO_TOLERANCE,
    )
    parser.add_argument("--pseudo-rounds", type=int, default=DEFAULT_PSEUDO_ROUNDS)
    parser.add_argument(
        "--save-entropy-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--prefetch-models", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    return parser


def resolve_gpu_ids(args: argparse.Namespace) -> List[int]:
    gpu_ids = parse_gpu_id_list(args.gpus) if args.gpus else detect_default_gpu_ids()
    if args.workers is None:
        args.workers = len(gpu_ids)
    if len(gpu_ids) != args.workers:
        raise ValueError(f"--gpus must contain exactly {args.workers} ids")
    return gpu_ids


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
    out_dir: Path,
) -> None:
    model_kwargs = build_common_model_kwargs(args)
    model_kwargs["model_path"] = normalize_model_path(args.model_path or DEFAULT_MODEL_PATH)
    entropy_save_dir = out_dir / "entropy_pass1" if args.save_entropy_artifacts else None

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
                assigned_dirs,
                ready_queue,
                start_event,
                str(worker_csv),
                dict(model_kwargs),
                float(np.clip(args.pseudo_max_error_rate, 0.0, 1.0)),
                bool(args.class_aware_pseudo_tolerance),
                max(1, int(args.pseudo_rounds)),
                str(entropy_save_dir) if entropy_save_dir is not None else "",
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
                    f"[worker {message['worker_id']} | gpu {message['gpu_id']}] "
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
    wall_seconds = time.time() - start_time
    all_csv, summary_txt = write_dataset_outputs(out_dir, all_df, dataset_dirs, wall_seconds)

    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {summary_txt}")
    print("model_kwargs:")
    print(json.dumps(model_kwargs, indent=2, ensure_ascii=False))


def run_multi_model_mode(
    args: argparse.Namespace,
    dataset_dirs: List[Path],
    gpu_ids: List[int],
    out_dir: Path,
) -> None:
    model_paths = discover_model_paths(Path(args.models_dir), max_models=args.max_models)
    if not model_paths:
        raise FileNotFoundError(f"No checkpoint files found under {args.models_dir}")

    worker_count = min(args.workers, len(model_paths))
    base_model_kwargs = build_common_model_kwargs(args)
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
                    [str(path.resolve()) for path in dataset_dirs],
                    ready_queue,
                    task_queue,
                    result_queue,
                    dict(base_model_kwargs),
                    float(np.clip(args.pseudo_max_error_rate, 0.0, 1.0)),
                    bool(args.class_aware_pseudo_tolerance),
                    max(1, int(args.pseudo_rounds)),
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
                        f"[worker {message['worker_id']} | gpu {message['gpu_id']}] "
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
            model_path = model_paths[next_model_idx]
            model_out_dir = out_dir / model_path.stem
            task_queues[worker_id].put(
                {
                    "model_path": str(model_path),
                    "entropy_save_dir": (
                        str(model_out_dir / "entropy_pass1")
                        if args.save_entropy_artifacts
                        else ""
                    ),
                }
            )
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
            rows = list(message.get("rows", []))
            collected_summaries.append(summary)
            completed_models += 1
            model_out_dir = out_dir / summary["model_name"]
            model_df = (
                pd.DataFrame(rows)
                if rows
                else pd.DataFrame(columns=ResultRow.__annotations__.keys())
            )
            write_dataset_outputs(
                model_out_dir,
                model_df,
                dataset_dirs,
                float(summary["model_wall_seconds"]),
            )

            if args.verbose:
                print(
                    f"[worker {worker_id} | gpu {summary['gpu_id']}] "
                    f"finished model={summary['model_name']} status={summary['status']} "
                    f"ok={summary['ok_count']} fail={summary['failed_count']} "
                    f"wall={summary['model_wall_seconds']:.2f}s",
                    flush=True,
                )

            if next_model_idx < len(model_paths):
                model_path = model_paths[next_model_idx]
                model_out_dir = out_dir / model_path.stem
                task_queues[worker_id].put(
                    {
                        "model_path": str(model_path),
                        "entropy_save_dir": (
                            str(model_out_dir / "entropy_pass1")
                            if args.save_entropy_artifacts
                            else ""
                        ),
                    }
                )
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
    args.pseudo_max_error_rate = float(np.clip(args.pseudo_max_error_rate, 0.0, 1.0))
    args.pseudo_rounds = max(1, int(args.pseudo_rounds))

    if args.models_dir is not None and args.model_path is not None:
        raise ValueError("--models-dir and --model-path are mutually exclusive")

    data_root = resolve_data_root_path(args.data_root)
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

    gpu_ids = resolve_gpu_ids(args)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if args.models_dir is not None:
        run_multi_model_mode(args, dataset_dirs, gpu_ids, out_dir)
    else:
        run_single_model_mode(args, dataset_dirs, gpu_ids, out_dir)


if __name__ == "__main__":
    main()
