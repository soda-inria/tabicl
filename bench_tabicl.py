#!/usr/bin/env python3
"""
Benchmark TabICLClassifier on TALENT classification datasets.

Usage examples:
    python bench_tabicl.py
    python bench_tabicl.py --data-root data181
    python bench_tabicl.py --data-root /path/to/TALENT/data --device cuda --device-id 1

Notes:
 - Only standard TALENT splits are supported: ``N/C/y_{train,val,test}.npy``.
 - Importing this module does not load TabICL checkpoints; model import is delayed.
 - GPU memory monitoring is best-effort and optional. Missing ``pynvml`` is tolerated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import inspect
import json
import logging
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CHECKPOINT_VERSION = "tabicl-classifier-v2-20260212.ckpt"
DEFAULT_DATA_ROOT = "data181"
CLASSIFICATION_TASKS = {"binclass", "multiclass", "unknown"}


SplitRef = tuple[Path | None, Path | None, Path]


@dataclass(frozen=True)
class DatasetFiles:
    train: SplitRef
    val: SplitRef | None
    test: SplitRef


@dataclass(frozen=True)
class DatasetResult:
    name: str
    accuracy: float
    fit_time_s: float
    predict_time_s: float
    total_time_s: float
    peak_vram_mib: float | None
    val_accuracy: float | None = None
    val_time_s: float | None = None


class GpuMemoryMonitor:
    """Best-effort NVML wrapper used to sample GPU memory."""

    def __init__(self, *, enabled: bool, device_id: int):
        self.enabled = enabled
        self.device_id = device_id
        self._pynvml = None
        self._initialized = False
        self._disabled_reason_logged = False
        self._sample_error_logged = False

    def _log_disabled_reason(self, message: str) -> None:
        if not self._disabled_reason_logged:
            logging.info(message)
            self._disabled_reason_logged = True

    def _ensure_initialized(self) -> bool:
        if not self.enabled:
            return False
        if self._initialized:
            return self._pynvml is not None

        self._initialized = True
        try:
            pynvml = importlib.import_module("pynvml")
        except ImportError:
            self._log_disabled_reason("未安装 pynvml，跳过 GPU 显存监控。")
            return False

        try:
            pynvml.nvmlInit()
        except Exception as exc:
            self._log_disabled_reason(f"pynvml 初始化失败，跳过 GPU 显存监控: {exc}")
            return False

        self._pynvml = pynvml
        logging.info("pynvml 初始化成功，将监控 GPU 显存。")
        return True

    def sample(self) -> float | None:
        if not self._ensure_initialized():
            return None

        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / (1024 * 1024)
        except Exception as exc:
            if not self._sample_error_logged:
                logging.warning("查询 GPU 显存失败 (device %d): %s", self.device_id, exc)
                self._sample_error_logged = True
            return None

    def shutdown(self) -> None:
        if self._pynvml is None:
            return
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass


def convert_features(X: np.ndarray, enabled: bool) -> np.ndarray:
    """Optionally coerce mixed-type arrays into numeric arrays."""
    X = np.asarray(X)
    if not enabled:
        return X

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    df = pd.DataFrame(X)
    encoded = pd.DataFrame(index=df.index)

    for col in df.columns:
        series = df.iloc[:, col]
        numeric_series = pd.to_numeric(series, errors="coerce")

        if series.isna().equals(numeric_series.isna()):
            encoded[col] = numeric_series
        else:
            string_series = series.astype("string")
            codes, uniques = pd.factorize(string_series, sort=True)
            codes = codes.astype(np.int32)
            if (codes == -1).any():
                codes[codes == -1] = len(uniques)
            encoded[col] = codes

    return encoded.fillna(0).values.astype(np.float32)


def handle_missing_entries(X: np.ndarray, y: np.ndarray, *, context: str) -> tuple[np.ndarray, np.ndarray]:
    """Fill numeric NaNs and drop rows with missing string features."""
    X = np.asarray(X)
    y = np.asarray(y)

    df = pd.DataFrame(X)
    y_series = pd.Series(y, index=df.index)
    drop_mask = pd.Series(False, index=df.index)

    for col in df.columns:
        series = df.iloc[:, col]
        numeric_series = pd.to_numeric(series, errors="coerce")

        if series.isna().equals(numeric_series.isna()):
            nan_mask = numeric_series.isna()
            if nan_mask.any():
                mean_value = float(numeric_series.mean(skipna=True))
                if np.isnan(mean_value):
                    mean_value = 0.0
                df.iloc[:, col] = numeric_series.fillna(mean_value)
                logging.info(
                    "%s: 数值列 %s 使用均值 %.6f 填充 %d 个 NaN",
                    context,
                    col,
                    mean_value,
                    int(nan_mask.sum()),
                )
        else:
            nan_mask = series.isna()
            if nan_mask.any():
                drop_mask |= nan_mask

    if drop_mask.any():
        drop_count = int(drop_mask.sum())
        df = df.loc[~drop_mask].copy()
        y_series = y_series.loc[df.index]
        logging.info("%s: 删除 %d 行包含字符串缺失值", context, drop_count)

    return df.values, y_series.values


def count_missing(values: np.ndarray | None) -> int:
    """Count NaN/None values in an array-like object."""
    if values is None:
        return 0

    arr = np.asarray(values)
    if arr.dtype.kind in {"f", "c"}:
        return int(np.isnan(arr).sum())

    mask = pd.isna(pd.DataFrame(arr))
    return int(mask.values.sum())


def log_nan_presence(
    context: str,
    values: np.ndarray,
    *,
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
) -> None:
    """Warn when missing values are detected and record the dataset id."""
    missing = count_missing(values)
    if missing:
        logging.warning("%s: 原始数据包含 %d 个 NaN/缺失值", context, missing)
        if dataset_id and missing_registry is not None:
            missing_registry.add(dataset_id)


def _normalize_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 3 and X.shape[1] == 1:
        X = X.squeeze(1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _normalize_target(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    elif y.ndim > 1 and y.shape[0] == 1:
        y = y.squeeze(0)
    return pd.Series(y).values


def load_array(file_path: Path) -> np.ndarray:
    """Load an array from npy/npz/csv/tsv/parquet."""
    suffix = file_path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        try:
            arr = np.load(file_path, allow_pickle=False)
        except ValueError:
            arr = np.load(file_path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        return np.asarray(arr)
    if suffix == ".parquet":
        return pd.read_parquet(file_path).values
    sep = "\t" if suffix == ".tsv" else None
    return pd.read_csv(file_path, sep=sep, header=None).values


def load_pair(
    X_path: Path,
    y_path: Path,
    *,
    context: str = "",
    coerce_numeric: bool = False,
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load separate X/y files."""
    X = _normalize_features(load_array(X_path))
    y = _normalize_target(load_array(y_path))

    ctx = context or X_path.stem
    log_nan_presence(f"{ctx}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{ctx}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

    X, y = handle_missing_entries(X, y, context=ctx)
    X = convert_features(X, coerce_numeric)
    return X, y


def load_split(
    num_path: Path | None,
    cat_path: Path | None,
    y_path: Path,
    *,
    context: str = "",
    coerce_numeric: bool = False,
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load TALENT numeric/categorical feature files and target."""
    features = []
    ctx_base = context or (num_path.stem if num_path else (cat_path.stem if cat_path else y_path.stem))

    if num_path:
        X_num = _normalize_features(load_array(num_path))
        log_nan_presence(f"{ctx_base}-num_raw", X_num, dataset_id=dataset_id, missing_registry=missing_registry)
        features.append(X_num)
    if cat_path:
        X_cat = _normalize_features(load_array(cat_path))
        log_nan_presence(f"{ctx_base}-cat_raw", X_cat, dataset_id=dataset_id, missing_registry=missing_registry)
        features.append(X_cat)

    if not features:
        raise ValueError("No numeric or categorical feature files found for split")

    n_samples = features[0].shape[0]
    for idx, feat in enumerate(features):
        if feat.shape[0] != n_samples:
            raise ValueError(f"Feature array #{idx} has mismatched sample count: {feat.shape[0]} vs {n_samples}")

    X = features[0] if len(features) == 1 else np.concatenate(features, axis=1)
    log_nan_presence(f"{ctx_base}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)

    y = _normalize_target(load_array(y_path))
    log_nan_presence(f"{ctx_base}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

    X, y = handle_missing_entries(X, y, context=ctx_base)
    X = convert_features(X, coerce_numeric)
    return X, y


def load_table(
    split_ref: tuple[Path | None, Path | None, Path] | tuple[Path, Path],
    *,
    context: str = "",
    coerce_numeric: bool = False,
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch loading based on split reference shape."""
    if len(split_ref) == 2:
        X_path, y_path = split_ref
        return load_pair(
            X_path,
            y_path,
            context=context,
            coerce_numeric=coerce_numeric,
            dataset_id=dataset_id,
            missing_registry=missing_registry,
        )

    if len(split_ref) == 3:
        num_path, cat_path, y_path = split_ref
        return load_split(
            num_path,
            cat_path,
            y_path,
            context=context,
            coerce_numeric=coerce_numeric,
            dataset_id=dataset_id,
            missing_registry=missing_registry,
        )

    raise ValueError(f"Unsupported split reference: {split_ref}")


def find_data_files(dataset_dir: Path) -> tuple[DatasetFiles | None, str | None]:
    """Find standard TALENT train/val/test split files."""
    files = [p for p in dataset_dir.iterdir() if p.is_file()]
    lower_names = {p.name.lower(): p for p in files}

    def find_by_suffix(key: str) -> Path | None:
        for name, path in lower_names.items():
            if name.endswith(key):
                return path
        return None

    n_train = find_by_suffix("n_train.npy")
    c_train = find_by_suffix("c_train.npy")
    y_train = find_by_suffix("y_train.npy")
    n_val = find_by_suffix("n_val.npy")
    c_val = find_by_suffix("c_val.npy")
    y_val = find_by_suffix("y_val.npy")
    n_test = find_by_suffix("n_test.npy")
    c_test = find_by_suffix("c_test.npy")
    y_test = find_by_suffix("y_test.npy")

    if y_train and y_test and (n_train or c_train) and (n_test or c_test):
        val_split = None
        if y_val and (n_val or c_val):
            val_split = (n_val, c_val, y_val)
        return DatasetFiles(
            train=(n_train, c_train, y_train),
            val=val_split,
            test=(n_test, c_test, y_test),
        ), None

    table_candidates = sorted(
        p.name for p in files if p.suffix.lower() in {".npy", ".npz", ".csv", ".tsv", ".parquet"}
    )
    if table_candidates:
        preview = ", ".join(table_candidates[:3])
        return None, f"仅支持标准 TALENT split，跳过非标准文件布局: {preview}"

    return None, "没有可识别的 TALENT train/test split"


def load_dataset_info(dataset_dir: Path) -> dict[str, Any] | None:
    """Load TALENT info.json if it exists."""
    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        return None
    try:
        with info_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logging.warning("读取 %s 失败: %s", info_path, exc)
        return None


def summarize_task_types(dirs: list[Path]) -> dict[str, int]:
    """Summarize TALENT task types from info.json."""
    counts = {"regression": 0, "binclass": 0, "multiclass": 0, "unknown": 0}

    for dataset_dir in dirs:
        info = load_dataset_info(dataset_dir)
        task_type = str(info.get("task_type", "")).lower() if info else ""
        if not task_type:
            counts["unknown"] += 1
        elif task_type in counts:
            counts[task_type] += 1
        else:
            counts["unknown"] += 1

    logging.info(
        "TALENT 数据集任务统计: regression=%d, binclass=%d, multiclass=%d, unknown=%d (总计 %d)",
        counts["regression"],
        counts["binclass"],
        counts["multiclass"],
        counts["unknown"],
        len(dirs),
    )
    return counts


def _parse_available_checkpoint_versions(error_message: str) -> list[str]:
    return re.findall(r"'([^']+\.ckpt)'", error_message or "")


def _resolve_checkpoint_version(requested: str, available: list[str]) -> str:
    if requested in available:
        return requested
    return available[0] if available else requested


def _format_optional_float(value: float | None, *, precision: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def _resolve_device_argument(device: str | None, device_id: int) -> str | None:
    if device is None:
        return None
    if device == "cuda":
        return f"cuda:{device_id}"
    return device


def _should_try_gpu_monitor(device: str | None) -> bool:
    return device is None or str(device).startswith("cuda")


def resolve_data_root_path(data_root: str | Path) -> Path:
    """Resolve a dataset directory from an absolute path or current-directory folder name."""
    candidate = Path(data_root).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def parse_optional_int(value: str) -> int | None:
    if value.lower() == "none":
        return None
    return int(value)


def parse_auto_bool(value: str) -> bool | str:
    normalized = value.lower()
    if normalized == "auto":
        return "auto"
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("Expected one of: auto, true, false")


def parse_kv_cache(value: str) -> bool | str:
    normalized = value.lower()
    if normalized == "false":
        return False
    if normalized == "true":
        return True
    if normalized in {"kv", "repr"}:
        return normalized
    raise argparse.ArgumentTypeError("Expected one of: false, true, kv, repr")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark TabICLClassifier on TALENT datasets")
    bool_action = argparse.BooleanOptionalAction

    parser.add_argument("--model-path", default=None, help="Local TabICL checkpoint path. Default: None")
    parser.add_argument(
        "--checkpoint-version",
        default=DEFAULT_CHECKPOINT_VERSION,
        help="Checkpoint version used when --model-path is None or missing",
    )
    parser.add_argument("--allow-auto-download", action=bool_action, default=True)
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=(
            "Dataset directory. Accepts an absolute path or any folder name under the current working directory. "
            f"Default: {DEFAULT_DATA_ROOT}"
        ),
    )
    parser.add_argument("--outdir", default="tabiclv2_talent_benchmark", help="Directory to save results")
    parser.add_argument("--max-datasets", type=int, default=None, help="Limit number of datasets to evaluate")
    parser.add_argument("--merge-val", action=bool_action, default=True, help="Merge validation split into training")
    parser.add_argument("--coerce-numeric", action=bool_action, default=True, help="Coerce mixed-type arrays")
    parser.add_argument("--device", default=None, help="Inference device: cuda, cpu, mps, cuda:1, etc.")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device index used with --device cuda")
    parser.add_argument("--debug", action="store_true", help="Enable benchmark debug logging")

    parser.add_argument("--n-estimators", type=int, default=32)
    parser.add_argument("--norm-methods", nargs="+", default=None, help="Normalization methods for the ensemble")
    parser.add_argument("--feat-shuffle-method", default="latin")
    parser.add_argument("--class-shuffle-method", default="shift")
    parser.add_argument("--outlier-threshold", type=float, default=4.0)
    parser.add_argument("--softmax-temperature", type=float, default=0.9)
    parser.add_argument("--average-logits", action=bool_action, default=True)
    parser.add_argument("--support-many-classes", action=bool_action, default=True)
    parser.add_argument("--batch-size", type=parse_optional_int, default=8, help="Integer or 'none'")
    parser.add_argument("--kv-cache", type=parse_kv_cache, default=False)
    parser.add_argument("--use-amp", type=parse_auto_bool, default="auto")
    parser.add_argument("--use-fa3", type=parse_auto_bool, default="auto")
    parser.add_argument("--offload-mode", choices=["auto", "gpu", "cpu", "disk"], default="auto")
    parser.add_argument("--disk-offload-dir", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=parse_optional_int, default=None, help="Integer or 'none'")
    parser.add_argument("--verbose", action="store_true", help="Enable TabICL model verbose output")

    return parser


def classifier_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_estimators": args.n_estimators,
        "norm_methods": args.norm_methods,
        "feat_shuffle_method": args.feat_shuffle_method,
        "class_shuffle_method": args.class_shuffle_method,
        "outlier_threshold": args.outlier_threshold,
        "softmax_temperature": args.softmax_temperature,
        "average_logits": args.average_logits,
        "support_many_classes": args.support_many_classes,
        "batch_size": args.batch_size,
        "kv_cache": args.kv_cache,
        "model_path": args.model_path,
        "allow_auto_download": args.allow_auto_download,
        "checkpoint_version": args.checkpoint_version,
        "device": _resolve_device_argument(args.device, args.device_id),
        "use_amp": args.use_amp,
        "use_fa3": args.use_fa3,
        "offload_mode": args.offload_mode,
        "disk_offload_dir": args.disk_offload_dir,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
        "verbose": args.verbose,
    }


def _filter_supported_classifier_kwargs(classifier_cls, classifier_kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop kwargs unsupported by the installed TabICLClassifier version."""
    try:
        signature = inspect.signature(classifier_cls)
    except (TypeError, ValueError):
        return dict(classifier_kwargs), []

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(classifier_kwargs), []

    supported_names = {name for name in parameters if name != "self"}
    filtered_kwargs = {name: value for name, value in classifier_kwargs.items() if name in supported_names}
    dropped_kwargs = sorted(name for name in classifier_kwargs if name not in supported_names)
    return filtered_kwargs, dropped_kwargs


def _build_classifier(classifier_cls, classifier_kwargs: dict[str, Any]):
    filtered_kwargs, dropped_kwargs = _filter_supported_classifier_kwargs(classifier_cls, classifier_kwargs)

    if dropped_kwargs:
        warned = getattr(_build_classifier, "_warned_dropped_kwargs", set())
        warning_key = (classifier_cls.__name__, tuple(dropped_kwargs))
        if warning_key not in warned:
            logging.warning(
                "当前安装的 %s 不支持以下参数，将自动忽略以兼容旧版本: %s",
                classifier_cls.__name__,
                ", ".join(dropped_kwargs),
            )
            warned.add(warning_key)
            setattr(_build_classifier, "_warned_dropped_kwargs", warned)

    return classifier_cls(**filtered_kwargs)


def _write_results(
    outdir: Path,
    results: list[DatasetResult],
    *,
    datasets_with_missing: set[str],
) -> None:
    detailed_path = outdir / "talent_detailed.txt"
    summary_path = outdir / "talent_summary.txt"

    with detailed_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "dataset\taccuracy\tfit_time_s\tpredict_time_s\ttotal_time_s\tpeak_vram_mib\tval_accuracy\tval_time_s\n"
        )
        for result in results:
            handle.write(
                f"{result.name}\t"
                f"{result.accuracy:.6f}\t"
                f"{result.fit_time_s:.3f}\t"
                f"{result.predict_time_s:.3f}\t"
                f"{result.total_time_s:.3f}\t"
                f"{_format_optional_float(result.peak_vram_mib, precision=2)}\t"
                f"{_format_optional_float(result.val_accuracy)}\t"
                f"{_format_optional_float(result.val_time_s, precision=3)}\n"
            )

    total_time = sum(result.total_time_s for result in results)
    avg_time = total_time / len(results) if results else 0.0
    avg_fit_time = sum(result.fit_time_s for result in results) / len(results) if results else 0.0
    avg_predict_time = sum(result.predict_time_s for result in results) / len(results) if results else 0.0
    avg_acc = sum(result.accuracy for result in results) / len(results) if results else 0.0

    valid_vram = [result.peak_vram_mib for result in results if result.peak_vram_mib is not None]
    avg_vram = sum(valid_vram) / len(valid_vram) if valid_vram else None
    max_vram = max(valid_vram) if valid_vram else None

    val_results = [result for result in results if result.val_accuracy is not None]
    avg_val_acc = (
        sum(result.val_accuracy for result in val_results if result.val_accuracy is not None) / len(val_results)
        if val_results
        else None
    )
    avg_val_time = (
        sum(result.val_time_s for result in val_results if result.val_time_s is not None) / len(val_results)
        if val_results
        else None
    )

    missing_results = [result for result in results if result.name in datasets_with_missing]
    avg_missing_acc = (
        sum(result.accuracy for result in missing_results) / len(missing_results) if missing_results else None
    )

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Total datasets: {len(results)}\n")
        handle.write(f"Average accuracy: {avg_acc:.6f}\n")
        handle.write(f"Total time s: {total_time:.3f}\n")
        handle.write(f"Average time s: {avg_time:.3f}\n")
        handle.write(f"Average fit time s: {avg_fit_time:.3f}\n")
        handle.write(f"Average predict time s: {avg_predict_time:.3f}\n")
        handle.write(f"Average Peak VRAM (MiB): {_format_optional_float(avg_vram, precision=2)}\n")
        handle.write(f"Overall Max Peak VRAM (MiB): {_format_optional_float(max_vram, precision=2)}\n")
        handle.write(f"Datasets with NaN values: {len(missing_results)}\n")
        if avg_missing_acc is not None:
            handle.write(f"Average accuracy (NaN datasets): {avg_missing_acc:.6f}\n")
            handle.write(f"List (NaN datasets): {', '.join(sorted(result.name for result in missing_results))}\n")
        if avg_val_acc is not None:
            handle.write(f"Average validation accuracy: {avg_val_acc:.6f}\n")
            handle.write(f"Average validation time s: {avg_val_time:.3f}\n")


def evaluate_datasets(
    *,
    classifier_kwargs: dict[str, Any],
    data_root: str | Path,
    outdir: str | Path,
    max_datasets: int | None = None,
    merge_val: bool = True,
    coerce_numeric: bool = True,
    device_id: int = 0,
) -> list[DatasetResult]:
    """Run the TALENT benchmark and write detailed + summary outputs."""
    from sklearn.utils.multiclass import type_of_target
    from tabicl.sklearn.classifier import TabICLClassifier

    data_root = resolve_data_root_path(data_root)
    outdir = Path(outdir)

    if not data_root.exists():
        available_dirs = sorted(path.name for path in Path.cwd().iterdir() if path.is_dir())
        preview = ", ".join(available_dirs[:10])
        extra = f" Current directory folders: {preview}" if preview else ""
        raise FileNotFoundError(f"Data root does not exist: {data_root}.{extra}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    outdir.mkdir(parents=True, exist_ok=True)

    active_classifier_kwargs = dict(classifier_kwargs)
    monitor = GpuMemoryMonitor(
        enabled=_should_try_gpu_monitor(active_classifier_kwargs.get("device")),
        device_id=device_id,
    )

    logging.info(
        "使用 TabICLClassifier: model_path=%s, checkpoint_version=%s, device=%s",
        active_classifier_kwargs.get("model_path"),
        active_classifier_kwargs.get("checkpoint_version"),
        active_classifier_kwargs.get("device"),
    )

    results: list[DatasetResult] = []
    datasets_with_missing: set[str] = set()

    dirs = [path for path in sorted(data_root.iterdir()) if path.is_dir()]
    if max_datasets is not None:
        dirs = dirs[:max_datasets]

    summarize_task_types(dirs)

    try:
        for dataset_dir in dirs:
            try:
                info = load_dataset_info(dataset_dir)
                task_type = str(info.get("task_type", "")).lower() if info else ""

                if task_type == "regression":
                    logging.info("跳过数据集 %s: task_type=regression", dataset_dir.name)
                    continue
                if task_type and task_type not in CLASSIFICATION_TASKS:
                    logging.info(
                        "跳过数据集 %s: 不支持的 task_type=%s (仅支持 %s)",
                        dataset_dir.name,
                        task_type,
                        sorted(CLASSIFICATION_TASKS),
                    )
                    continue

                dataset_files, skip_reason = find_data_files(dataset_dir)
                if dataset_files is None:
                    logging.info("跳过数据集 %s: %s", dataset_dir.name, skip_reason)
                    continue

                X_train, y_train = load_table(
                    dataset_files.train,
                    context=f"{dataset_dir.name}-train",
                    coerce_numeric=coerce_numeric,
                    dataset_id=dataset_dir.name,
                    missing_registry=datasets_with_missing,
                )
                X_test, y_test = load_table(
                    dataset_files.test,
                    context=f"{dataset_dir.name}-test",
                    coerce_numeric=coerce_numeric,
                    dataset_id=dataset_dir.name,
                    missing_registry=datasets_with_missing,
                )

                X_val = y_val = None
                if dataset_files.val is not None:
                    X_val, y_val = load_table(
                        dataset_files.val,
                        context=f"{dataset_dir.name}-val",
                        coerce_numeric=coerce_numeric,
                        dataset_id=dataset_dir.name,
                        missing_registry=datasets_with_missing,
                    )
                    if merge_val:
                        X_train = np.concatenate([X_train, X_val], axis=0)
                        y_train = np.concatenate([y_train, y_val], axis=0)
                        logging.info("%s: 已将 validation split 合并进训练集 (%d 条样本)", dataset_dir.name, X_train.shape[0])

                inferred_target_type = None
                try:
                    inferred_target_type = type_of_target(y_train)
                except Exception:
                    inferred_target_type = None

                if (not task_type or task_type == "unknown") and inferred_target_type and inferred_target_type.startswith(
                    "continuous"
                ):
                    logging.info("跳过数据集 %s: 标签看起来是连续值，非分类任务", dataset_dir.name)
                    continue

                dataset_start = time.perf_counter()
                clf = _build_classifier(TabICLClassifier, active_classifier_kwargs)

                fit_start = time.perf_counter()
                try:
                    clf.fit(X_train, y_train)
                except ValueError as exc:
                    message = str(exc)
                    if "Invalid checkpoint version" not in message:
                        raise

                    available_versions = _parse_available_checkpoint_versions(message)
                    resolved_version = _resolve_checkpoint_version(
                        active_classifier_kwargs["checkpoint_version"], available_versions
                    )
                    if resolved_version == active_classifier_kwargs["checkpoint_version"]:
                        raise

                    logging.warning(
                        "%s: checkpoint '%s' 不可用，自动切换为 '%s'（可用: %s）",
                        dataset_dir.name,
                        active_classifier_kwargs["checkpoint_version"],
                        resolved_version,
                        ", ".join(available_versions) if available_versions else "unknown",
                    )
                    active_classifier_kwargs["checkpoint_version"] = resolved_version
                    clf = _build_classifier(TabICLClassifier, active_classifier_kwargs)
                    fit_start = time.perf_counter()
                    clf.fit(X_train, y_train)

                fit_time_s = time.perf_counter() - fit_start
                mem_after_fit = monitor.sample()

                val_accuracy = None
                val_time_s = None
                mem_after_val = None
                if X_val is not None and not merge_val:
                    val_start = time.perf_counter()
                    y_val_pred = clf.predict(X_val)
                    val_time_s = time.perf_counter() - val_start
                    mem_after_val = monitor.sample()
                    val_accuracy = float(np.mean(y_val_pred == y_val))

                predict_start = time.perf_counter()
                y_pred = clf.predict(X_test)
                predict_time_s = time.perf_counter() - predict_start
                mem_after_predict = monitor.sample()

                accuracy = float(np.mean(y_pred == y_test))
                total_time_s = time.perf_counter() - dataset_start

                sampled_memory = [value for value in (mem_after_fit, mem_after_val, mem_after_predict) if value is not None]
                peak_vram_mib = max(sampled_memory) if sampled_memory else None

                result = DatasetResult(
                    name=dataset_dir.name,
                    accuracy=accuracy,
                    fit_time_s=fit_time_s,
                    predict_time_s=predict_time_s,
                    total_time_s=total_time_s,
                    peak_vram_mib=peak_vram_mib,
                    val_accuracy=val_accuracy,
                    val_time_s=val_time_s,
                )
                results.append(result)

                logging.info(
                    "%s: accuracy=%.4f fit=%.2fs predict=%.2fs total=%.2fs peak_vram=%s val_acc=%s",
                    dataset_dir.name,
                    accuracy,
                    fit_time_s,
                    predict_time_s,
                    total_time_s,
                    _format_optional_float(peak_vram_mib, precision=1),
                    _format_optional_float(val_accuracy, precision=4),
                )

            except Exception as exc:
                logging.exception("评测失败 %s: %s", dataset_dir.name, exc)
    finally:
        monitor.shutdown()

    if results:
        _write_results(outdir, results, datasets_with_missing=datasets_with_missing)

        total_time = sum(result.total_time_s for result in results)
        avg_acc = sum(result.accuracy for result in results) / len(results)
        avg_time = total_time / len(results)
        logging.info(
            "评测完成，共 %d 个数据集，平均准确率 %.4f，总耗时 %.2fs，平均每数据集 %.2fs",
            len(results),
            avg_acc,
            total_time,
            avg_time,
        )
    else:
        logging.info("没有成功的评测结果。")

    return results


def configure_logging(outdir: Path, *, debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root_logger.addHandler(console_handler)

    outdir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(outdir / "bench_talent.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    root_logger.addHandler(file_handler)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    configure_logging(outdir, debug=args.debug)

    evaluate_datasets(
        classifier_kwargs=classifier_kwargs_from_args(args),
        data_root=args.data_root,
        outdir=outdir,
        max_datasets=args.max_datasets,
        merge_val=args.merge_val,
        coerce_numeric=args.coerce_numeric,
        device_id=args.device_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
