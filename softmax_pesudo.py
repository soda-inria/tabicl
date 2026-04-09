#!/usr/bin/env python3
"""
批量在 TALENT 数据目录上并行评测 TabICLClassifier。

特性：
 - 支持标准 TALENT split，以及单文件表格数据的 80/20 自动切分。
 - 对齐 README 中的模型参数，并按模型版本自动选择推荐的 n_estimators：
   v1/v1.1 使用 32，v2 使用 8。
 - 通过运行时签名探测兼容不同 TabICLClassifier 接口差异，例如
   kv_cache/use_kv_cache、class_shuffle_method/class_shift。
 - 并行逻辑采用固定切分：预先把数据集按 round-robin 分配到不同 GPU。
 - 每个 GPU 对应一个独立 Python 子进程，通过清单文件领取任务并写回结果，
   避免 multiprocessing worker/Queue 在 CUDA 场景下的异常退出问题。
 - 伪标签筛选规则使用 softmax 最大概率阈值：max_prob >= alpha。
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
import gc
import inspect
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Optional, Union

src_path = str(Path(__file__).resolve().parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = "data181"
PSEUDO_ACC_DELTA_TOL = 1e-9
CLASSIFICATION_TASKS = {"binclass", "multiclass", "unknown"}
CATEGORICAL_MISSING_TOKEN = "__tabicl_missing__"
README_RECOMMENDED_N_ESTIMATORS = {
    "v1": 32,
    "v1.1": 32,
    "v2": 8,
    "unknown": 8,
}
CHECKPOINT_ALIASES = {
    "v1": (
        "tabicl-classifier-v1-20250208.ckpt",
        "tabicl-classifier-v1-0208.ckpt",
    ),
    "v1.1": (
        "tabicl-classifier-v1.1-20250506.ckpt",
        "tabicl-classifier-v1.1-0506.ckpt",
    ),
    "v2": (
        "tabicl-classifier-v2-20260212.ckpt",
    ),
}


# ------------------------------
# 参数与模型兼容工具
# ------------------------------


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
    raise argparse.ArgumentTypeError("必须是以下之一: auto, true, false")


def parse_kv_cache(value: str) -> bool | str:
    """将命令行中的 kv_cache 参数解析为通用配置。"""
    lowered = value.strip().lower()
    if lowered in {"false", "0", "no", "off"}:
        return False
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"kv", "repr"}:
        return lowered
    raise argparse.ArgumentTypeError("kv_cache 必须是以下之一: false, true, kv, repr")


def parse_gpu_ids(value: str) -> list[int]:
    ids = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        ids.append(int(item))
    if not ids:
        raise argparse.ArgumentTypeError("gpu_ids 不能为空，例如 0,1,2")
    return ids


def infer_model_family_from_text(value: str | None) -> str:
    if not value:
        return "unknown"

    raw_text = str(value).lower().replace("\\", "/")
    candidates = [raw_text.rsplit("/", 1)[-1]]
    if candidates[0] != raw_text:
        candidates.append(raw_text)

    for text in candidates:
        if "regressor" in text:
            return "regressor"
        if "v1.1" in text or "20250506" in text or "0506" in text:
            return "v1.1"
        if "v2" in text or "20260212" in text:
            return "v2"
        if re.search(r"(^|[^0-9])v1([^0-9.]|$)", text) or "20250208" in text or "0208" in text:
            return "v1"
    return "unknown"


def resolve_model_family(
    *,
    model_path: str | None,
    checkpoint_version: str | None,
    explicit_family: str,
) -> str:
    if explicit_family != "auto":
        return explicit_family

    for candidate in (checkpoint_version, model_path):
        family = infer_model_family_from_text(candidate)
        if family != "unknown":
            return family
    return "v2"


def resolve_recommended_n_estimators(
    *,
    model_path: str | None,
    checkpoint_version: str | None,
    explicit_family: str,
    requested_n_estimators: int | None,
) -> tuple[str, int]:
    family = resolve_model_family(
        model_path=model_path,
        checkpoint_version=checkpoint_version,
        explicit_family=explicit_family,
    )
    if requested_n_estimators is not None:
        return family, requested_n_estimators
    return family, README_RECOMMENDED_N_ESTIMATORS.get(family, 8)


def get_signature_metadata(callable_obj) -> tuple[dict[str, inspect.Parameter] | None, bool]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return None, True

    params = {name: param for name, param in signature.parameters.items() if name != "self"}
    has_var_keyword = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    return params, has_var_keyword


def coerce_value_for_parameter(param: inspect.Parameter | None, value: Any) -> Any:
    if param is None:
        return value
    if value == "auto" and isinstance(param.default, bool):
        return param.default
    return value


def build_classifier_call_plan(classifier_cls, classifier_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    init_params, init_has_var_keyword = get_signature_metadata(classifier_cls.__init__)
    fit_params, fit_has_var_keyword = get_signature_metadata(classifier_cls.fit)

    if init_params is None:
        init_kwargs = dict(classifier_config)
        fit_kwargs = {}
        return init_kwargs, fit_kwargs, []

    working = dict(classifier_config)
    fit_kwargs: dict[str, Any] = {}
    notes: list[str] = []

    kv_value = working.pop("kv_cache", False)
    if init_has_var_keyword or "kv_cache" in init_params:
        working["kv_cache"] = kv_value
    elif "use_kv_cache" in init_params:
        mapped_value = bool(kv_value)
        working["use_kv_cache"] = mapped_value
        if kv_value not in {False, True}:
            notes.append(
                f"当前接口仅支持布尔 use_kv_cache，已将 kv_cache={kv_value!r} 降级映射为 {mapped_value!r}"
            )
    elif fit_has_var_keyword or (fit_params is not None and "kv_cache" in fit_params):
        fit_kwargs["kv_cache"] = kv_value
    elif kv_value:
        notes.append(f"当前接口不支持 kv_cache={kv_value!r}，已忽略该设置")

    class_shuffle_value = working.pop("class_shuffle_method", "shift")
    if init_has_var_keyword or "class_shuffle_method" in init_params:
        working["class_shuffle_method"] = class_shuffle_value
    elif "class_shift" in init_params:
        mapped_value = class_shuffle_value != "none"
        working["class_shift"] = mapped_value
        if class_shuffle_value not in {"none", "shift"}:
            notes.append(
                f"当前接口没有 class_shuffle_method，已将 {class_shuffle_value!r} 近似映射为 class_shift={mapped_value!r}"
            )
    elif class_shuffle_value != "shift":
        notes.append(f"当前接口不支持 class_shuffle_method={class_shuffle_value!r}，已忽略")

    many_classes_value = working.pop("support_many_classes", True)
    if init_has_var_keyword or "support_many_classes" in init_params:
        working["support_many_classes"] = many_classes_value
    elif "use_hierarchical" in init_params:
        working["use_hierarchical"] = many_classes_value
    elif many_classes_value is not True:
        notes.append(f"当前接口不支持 support_many_classes={many_classes_value!r}，已忽略")

    init_kwargs: dict[str, Any] = {}
    dropped_init_keys: list[str] = []
    for name, value in working.items():
        if init_has_var_keyword or name in init_params:
            param = init_params.get(name) if init_params is not None else None
            init_kwargs[name] = coerce_value_for_parameter(param, value)
        else:
            dropped_init_keys.append(name)

    if dropped_init_keys:
        notes.append("当前接口不支持以下初始化参数，已自动忽略: " + ", ".join(sorted(dropped_init_keys)))

    if fit_params is not None and not fit_has_var_keyword:
        filtered_fit_kwargs: dict[str, Any] = {}
        dropped_fit_keys: list[str] = []
        for name, value in fit_kwargs.items():
            if name in fit_params:
                filtered_fit_kwargs[name] = coerce_value_for_parameter(fit_params.get(name), value)
            else:
                dropped_fit_keys.append(name)
        fit_kwargs = filtered_fit_kwargs
        if dropped_fit_keys:
            notes.append("当前接口不支持以下 fit 参数，已自动忽略: " + ", ".join(sorted(dropped_fit_keys)))

    return init_kwargs, fit_kwargs, notes


def parse_available_checkpoint_versions(error_message: str) -> list[str]:
    message = error_message or ""
    marker = "Available ones are:"
    if marker in message:
        message = message.split(marker, 1)[1]
    return re.findall(r"'([^']+\.ckpt)'", message)


def resolve_checkpoint_alias(requested: str, available: list[str]) -> str:
    if not requested or not available:
        return requested
    if requested in available:
        return requested

    requested_family = infer_model_family_from_text(requested)
    if requested_family == "unknown":
        return requested

    same_family = [item for item in available if infer_model_family_from_text(item) == requested_family]
    for candidate in CHECKPOINT_ALIASES.get(requested_family, ()):
        if candidate in same_family:
            return candidate
    if len(same_family) == 1:
        return same_family[0]
    return requested


def worker_log(prefix: str, message: str) -> None:
    print(f"{prefix} {message}", flush=True)


def instantiate_worker_classifier(classifier_cls, classifier_config: dict[str, Any], msg_prefix: str):
    classifier, fit_kwargs, notes = instantiate_classifier(classifier_cls, classifier_config)
    for note in notes:
        worker_log(msg_prefix, note)
    return classifier, fit_kwargs


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


def validate_shard_result_payload(payload: dict[str, Any]) -> tuple[int, int, list[dict[str, Any]], set[str]]:
    if not isinstance(payload, dict):
        raise ValueError("shard 结果文件必须是 JSON 对象")

    rank = payload.get("rank")
    device_id = payload.get("device_id")
    results = payload.get("results")
    datasets_with_missing = payload.get("datasets_with_missing")

    if not isinstance(rank, int):
        raise ValueError("shard 结果缺少合法的 rank")
    if not isinstance(device_id, int):
        raise ValueError("shard 结果缺少合法的 device_id")
    if not isinstance(results, list):
        raise ValueError("shard 结果缺少合法的 results 列表")
    if not isinstance(datasets_with_missing, list):
        raise ValueError("shard 结果缺少合法的 datasets_with_missing 列表")
    if not all(isinstance(item, dict) for item in results):
        raise ValueError("shard results 内元素必须都是对象")

    return rank, device_id, results, set(str(item) for item in datasets_with_missing)


# ------------------------------
# 数据预处理工具
# ------------------------------


def normalize_categorical_series(series: pd.Series) -> pd.Series:
    """将类别列统一成纯字符串，并把缺失值替换为稳定哨兵值。"""
    string_series = series.astype("string")
    string_series = string_series.fillna(CATEGORICAL_MISSING_TOKEN)
    return string_series.astype(str)


def normalize_inferred_frame(df: pd.DataFrame) -> pd.DataFrame:
    """推断数值列与类别列，确保类别列不会混入 pandas.NA。"""
    normalized = pd.DataFrame(index=df.index)

    for column in df.columns:
        series = df[column]
        numeric_series = pd.to_numeric(series, errors="coerce")
        if pd.isna(numeric_series).equals(pd.isna(series)):
            normalized[column] = numeric_series
        else:
            normalized[column] = normalize_categorical_series(series)

    return normalized


def make_feature_frame(
    values: pd.DataFrame | np.ndarray,
    *,
    kind: str = "infer",
    prefix: str = "x",
) -> pd.DataFrame:
    """构造一个 DataFrame，并尽量保留特征语义，交给 TabICL 自身做预处理。"""
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
    elif kind == "infer":
        df = normalize_inferred_frame(df)
    else:
        raise ValueError(f"不支持的特征类型: {kind}")

    df.columns = [f"{prefix}_{i}" for i in range(df.shape[1])]
    return df


def make_target_array(values: np.ndarray) -> np.ndarray:
    """将标签整理为一维 NumPy 数组，同时保留原始缺失值。"""
    y = np.asarray(values)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    elif y.ndim > 1 and y.shape[0] == 1:
        y = y.squeeze(0)
    return pd.Series(y).values


def split_single_file_dataset(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    *,
    dataset_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray, np.ndarray, np.ndarray]:
    """将单表数据集切分为 train/test；如果条件允许则优先分层抽样。"""
    from sklearn.model_selection import train_test_split

    y = np.asarray(y)
    if len(y) < 2:
        raise ValueError(f"{dataset_name}: 行数不足，无法进行 train/test 切分")

    stratify = None
    y_series = pd.Series(y)
    class_counts = y_series.value_counts(dropna=False)
    if y_series.nunique(dropna=False) > 1 and not class_counts.empty and int(class_counts.min()) >= 2:
        stratify = y

    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    except ValueError:
        if stratify is None:
            raise
        logging.warning("%s: 分层 80/20 切分失败，回退到随机切分", dataset_name)
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)


def slice_rows(values: pd.DataFrame | np.ndarray, mask: np.ndarray) -> pd.DataFrame | np.ndarray:
    """按布尔掩码选取样本行，并为 DataFrame 重置索引。"""
    mask = np.asarray(mask, dtype=bool)
    if isinstance(values, pd.DataFrame):
        return values.loc[mask].reset_index(drop=True)
    return np.asarray(values)[mask]


def concat_feature_blocks(blocks: list[pd.DataFrame | np.ndarray]) -> pd.DataFrame | np.ndarray:
    """拼接多个特征块，兼容 DataFrame 与 ndarray。"""
    valid_blocks = [block for block in blocks if block is not None and len(block) > 0]
    if not valid_blocks:
        raise ValueError("没有可拼接的特征块")
    if all(isinstance(block, pd.DataFrame) for block in valid_blocks):
        return pd.concat(valid_blocks, axis=0, ignore_index=True)
    return np.concatenate([np.asarray(block) for block in valid_blocks], axis=0)


def concat_target_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    """拼接多个标签块。"""
    valid_blocks = [np.asarray(block) for block in blocks if block is not None and len(block) > 0]
    if not valid_blocks:
        raise ValueError("没有可拼接的标签块")
    return np.concatenate(valid_blocks, axis=0)


def compute_train_ratio(y_train: np.ndarray, y_test: np.ndarray) -> float:
    train_count = int(len(np.asarray(y_train)))
    test_count = int(len(np.asarray(y_test)))
    total_count = train_count + test_count
    return float(train_count / total_count) if total_count > 0 else float("nan")


def count_missing(values: np.ndarray | pd.DataFrame | pd.Series | None) -> int:
    """统计数组中的 NaN/None 数量。"""
    if values is None:
        return 0

    arr = np.asarray(values)
    if arr.dtype.kind in {"f", "c"}:
        return int(np.isnan(arr).sum())

    mask = pd.isna(pd.DataFrame(arr))
    return int(mask.values.sum())


def log_nan_presence(
    context: str,
    values: np.ndarray | pd.DataFrame | pd.Series,
    *,
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
) -> None:
    """如果存在缺失值就登记到缺失集合。"""
    missing = count_missing(values)
    if missing and dataset_id and missing_registry is not None:
        missing_registry.add(dataset_id)


def stable_feature_prefix(context: str, fallback: str) -> str:
    """为 train/val/test 生成一致的列名前缀，避免 sklearn 特征名校验失败。"""
    stem = Path(context or fallback).stem
    for suffix in ("-train", "-test", "-val", "-single"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or fallback


# ------------------------------
# 数据集文件查找与加载
# ------------------------------


def find_data_files(dataset_dir: Path):
    files = [p for p in dataset_dir.iterdir() if p.is_file()]
    lower_names = {p.name.lower(): p for p in files}

    def find_by_suffix(key: str):
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
        val_pair = None
        if y_val and (n_val or c_val):
            val_pair = (n_val, c_val, y_val)
        return (n_train, c_train, y_train), val_pair, (n_test, c_test, y_test)

    table_candidates = [p for p in files if p.suffix.lower() in {".npy", ".npz", ".csv", ".tsv", ".parquet"}]
    if len(table_candidates) == 1:
        return table_candidates[0], None, None

    return None, None, None


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
    if suffix == ".parquet":
        return pd.read_parquet(file_path).values
    sep = "\t" if suffix == ".tsv" else None
    return pd.read_csv(file_path, sep=sep, header=None).values


def load_frame(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        return make_feature_frame(load_array(file_path), kind="infer", prefix=file_path.stem)
    if suffix == ".parquet":
        return make_feature_frame(pd.read_parquet(file_path), kind="infer", prefix=file_path.stem)
    sep = "\t" if suffix == ".tsv" else None
    return make_feature_frame(pd.read_csv(file_path, sep=sep, header=None), kind="infer", prefix=file_path.stem)


def load_table(
    file_path: Union[Path, tuple],
    *,
    context: str = "",
    coerce_numeric: bool = False,
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if isinstance(file_path, (tuple, list)):
        if len(file_path) == 2:
            x_path, y_path = Path(file_path[0]), Path(file_path[1])
            return load_pair(
                x_path,
                y_path,
                context=context,
                coerce_numeric=coerce_numeric,
                dataset_id=dataset_id,
                missing_registry=missing_registry,
            )
        if len(file_path) == 3:
            num_path, cat_path, y_path = file_path
            return load_split(
                Path(num_path) if num_path else None,
                Path(cat_path) if cat_path else None,
                Path(y_path),
                context=context,
                coerce_numeric=coerce_numeric,
                dataset_id=dataset_id,
                missing_registry=missing_registry,
            )
        raise ValueError(f"load_table 不支持该元组格式: {file_path}")

    data = load_frame(file_path)
    if data.ndim == 1:
        raise ValueError(f"{file_path} 中的数据为 1D，当前不支持")

    log_target = context or str(file_path)
    log_nan_presence(f"{log_target}-raw", data, dataset_id=dataset_id, missing_registry=missing_registry)

    col0 = data.iloc[:, 0]
    try:
        uniques0 = np.unique(col0)
    except Exception:
        uniques0 = np.array([])

    if 0 < uniques0.size < max(2, data.shape[0] // 2):
        y = col0
        X = data.iloc[:, 1:].copy()
        heuristic_column = "first"
    else:
        y = data.iloc[:, -1]
        X = data.iloc[:, :-1].copy()
        heuristic_column = "last"

    log_nan_presence(f"{log_target}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{log_target}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

    feature_prefix = stable_feature_prefix(log_target, Path(file_path).stem)
    X = make_feature_frame(X, kind="infer", prefix=f"{feature_prefix}_x")
    y = make_target_array(y)

    logging.info("%s: 使用单文件启发式拆分标签 (取 %s 列)", log_target, heuristic_column)
    return X, y


def load_pair(
    x_path: Path,
    y_path: Path,
    *,
    context: str = "",
    coerce_numeric: bool = False,
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    del coerce_numeric

    X = load_array(x_path)
    y = load_array(y_path)

    ctx = context or x_path.stem
    log_nan_presence(f"{ctx}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{ctx}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

    feature_prefix = stable_feature_prefix(ctx, x_path.stem)
    X = make_feature_frame(X, kind="infer", prefix=f"{feature_prefix}_x")
    y = make_target_array(y)
    return X, y


def load_split(
    num_path: Optional[Path],
    cat_path: Optional[Path],
    y_path: Path,
    *,
    context: str = "",
    coerce_numeric: bool = False,
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    del coerce_numeric

    features: list[pd.DataFrame] = []
    ctx_base = context or (num_path.stem if num_path else (cat_path.stem if cat_path else y_path.stem))
    feature_prefix = stable_feature_prefix(ctx_base, y_path.stem)

    if num_path:
        x_num = load_array(num_path)
        log_nan_presence(f"{ctx_base}-num_raw", x_num, dataset_id=dataset_id, missing_registry=missing_registry)
        features.append(make_feature_frame(x_num, kind="numeric", prefix=f"{feature_prefix}_n"))
    if cat_path:
        x_cat = load_array(cat_path)
        log_nan_presence(f"{ctx_base}-cat_raw", x_cat, dataset_id=dataset_id, missing_registry=missing_registry)
        features.append(make_feature_frame(x_cat, kind="categorical", prefix=f"{feature_prefix}_c"))

    if not features:
        raise ValueError("split 数据中未找到数值特征文件或类别特征文件")

    n_samples = features[0].shape[0]
    for idx, feat in enumerate(features):
        if feat.shape[0] != n_samples:
            raise ValueError(f"特征数组 #{idx} 的样本数不一致: {feat.shape[0]} vs {n_samples}")

    X = features[0] if len(features) == 1 else pd.concat(features, axis=1)
    log_nan_presence(f"{ctx_base}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)

    y = load_array(y_path)
    log_nan_presence(f"{ctx_base}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)
    y = make_target_array(y)
    return X, y


def load_dataset_info(dataset_dir: Path) -> Optional[dict]:
    """读取 info.json（任务类型、元信息）。"""
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
        "任务统计: regression=%d, binclass=%d, multiclass=%d, unknown=%d (总计 %d)",
        counts["regression"],
        counts["binclass"],
        counts["multiclass"],
        counts["unknown"],
        len(dirs),
    )
    return counts


# ------------------------------
# 核心评测逻辑（Worker）
# ------------------------------


def instantiate_classifier(classifier_cls, classifier_config: dict[str, Any]):
    init_kwargs, fit_kwargs, notes = build_classifier_call_plan(classifier_cls, classifier_config)
    classifier = classifier_cls(**init_kwargs)
    return classifier, fit_kwargs, notes


def maybe_retry_with_checkpoint_alias(
    *,
    classifier_cls,
    classifier_config: dict[str, Any],
    fit_kwargs: dict[str, Any],
    X_train,
    y_train,
    error: Exception,
    msg_prefix: str,
):
    message = str(error)
    if "Invalid checkpoint version" not in message:
        raise error

    available_versions = parse_available_checkpoint_versions(message)
    resolved_version = resolve_checkpoint_alias(classifier_config.get("checkpoint_version"), available_versions)
    if not resolved_version or resolved_version == classifier_config.get("checkpoint_version"):
        raise error

    worker_log(
        msg_prefix,
        (
            f"checkpoint_version={classifier_config.get('checkpoint_version')!r} 不可用，"
            f"已切换为 {resolved_version!r}"
        ),
    )
    classifier_config["checkpoint_version"] = resolved_version
    classifier, fit_kwargs, notes = instantiate_classifier(classifier_cls, classifier_config)
    for note in notes:
        worker_log(msg_prefix, note)
    classifier.fit(X_train, y_train, **fit_kwargs)
    return classifier, fit_kwargs


def run_pseudo_label_resplit_inference(
    *,
    fit_classifier_fn: Callable[[pd.DataFrame | np.ndarray, np.ndarray], tuple[Any, float]],
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: np.ndarray,
    dataset_name: str,
    alpha: float,
) -> dict[str, Any]:
    """
    两阶段伪标签推理：
    1. 原始 train 上训练，并在原始 test 上计算 predict_proba。
    2. 用 softmax 最大概率 max_prob >= alpha 选出高置信样本加入伪标签集合。
    3. 将这些样本直接并入训练集，剩余样本作为新 test，再次 fit/predict。

    评测阶段始终使用真实标签；训练阶段仅使用预测标签形式的伪标签。
    最终返回的 y_pred/y_eval 始终对齐原始整个测试集，因此准确率表示
    “第一轮已确定的样本 + 第二轮剩余样本”的总准确率。
    """
    if not np.isfinite(float(alpha)) or not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"{dataset_name}: alpha 必须是有限且位于 [0, 1] 的数值，当前为 {alpha}")

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    train_ratio_before = compute_train_ratio(y_train, y_test)

    classifier, first_fit_time = fit_classifier_fn(X_train, y_train)

    predict_start = time.perf_counter()
    probs = np.asarray(classifier.predict_proba(X_test))
    first_predict_time = time.perf_counter() - predict_start
    if probs.ndim != 2:
        raise ValueError(f"{dataset_name}: predict_proba 结果维度异常，应为 2D，当前为 {probs.shape}")

    pred_indices = np.argmax(probs, axis=1)
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        raise ValueError(f"{dataset_name}: 分类器缺少 classes_，无法还原预测标签")
    classes = np.asarray(classes)
    baseline_pred = classes[pred_indices]
    baseline_accuracy = float(np.mean(np.asarray(baseline_pred) == y_test))

    max_probs = np.max(probs.astype(float), axis=1)
    confident_mask = np.asarray(max_probs >= float(alpha))
    pseudo_added = int(confident_mask.sum())

    result: dict[str, Any] = {
        "baseline_accuracy": baseline_accuracy,
        "fit_time": first_fit_time,
        "predict_time": first_predict_time,
        "pseudo_added": pseudo_added,
        "pseudo_applied": False,
        "pseudo_alpha": float(alpha),
        "train_ratio_before": train_ratio_before,
        "train_ratio_after": train_ratio_before,
        "y_pred": np.asarray(baseline_pred),
        "y_eval": y_test,
    }

    if pseudo_added == 0:
        return result

    X_confident = slice_rows(X_test, confident_mask)
    X_remaining = slice_rows(X_test, ~confident_mask)
    y_pseudo = np.asarray(baseline_pred)[confident_mask]
    y_remaining_true = y_test[~confident_mask]

    X_labeled = concat_feature_blocks([X_train, X_confident])
    y_labeled_train = concat_target_blocks([y_train, y_pseudo])

    if len(y_remaining_true) == 0:
        return result

    classifier, second_fit_time = fit_classifier_fn(X_labeled, y_labeled_train)
    predict_start = time.perf_counter()
    second_pred = np.asarray(classifier.predict(X_remaining))
    second_predict_time = time.perf_counter() - predict_start
    final_pred = np.asarray(baseline_pred).copy()
    final_pred[~confident_mask] = second_pred

    result.update(
        {
            "fit_time": first_fit_time + second_fit_time,
            "predict_time": first_predict_time + second_predict_time,
            "pseudo_applied": True,
            "train_ratio_after": compute_train_ratio(y_labeled_train, y_remaining_true),
            "y_pred": final_pred,
            "y_eval": y_test,
        }
    )
    return result


def evaluate_datasets_worker(
    rank: int,                  # 助教的编号（比如 0号 worker, 1号 worker）
    device_id: int,             # 分配给这个助教的 GPU 编号（如 0 代表第一张显卡，负数代表用 CPU）
    assigned_datasets: list[str], # 分配给该助教的数据集路径列表（静态切分）
    classifier_config: dict[str, Any], # 模型的配置文件（包含模型怎么设置）
    *,
    result_queue=None,          # 兼容旧的 multiprocessing 方式；当前默认不使用
    merge_val: bool = True,     # 是否把“模拟考（验证集）”也并入“平时作业（训练集）”里一起学
    coerce_numeric: bool = True,# 是否强制把数据转成数字
    pseudo_label_alpha: float | None = None,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], set[str]]:
    # ================= 1. 准备工作：配置环境 =================
    import sys
    from pathlib import Path

    # 把项目的 src 目录加到系统路径里，确保等会儿能找到我们自定义的模型代码
    local_src_path = str(Path(__file__).resolve().parent / "src")
    if local_src_path not in sys.path:
        sys.path.insert(0, local_src_path)

    # 给这个助教起个专属的日志名字，方便查错，比如 "[Worker 0][GPU 0]"
    msg_prefix = f"[Worker {rank}][GPU {device_id}]" if device_id >= 0 else f"[Worker {rank}][CPU]"
    # 确定计算设备，是使用显卡 'cuda:0' 还是 'cpu'
    device_str = f"cuda:{device_id}" if device_id >= 0 else "cpu"

    # 小本本：用来记录每个数据集的 [名字, 准确率, 总耗时]
    results: list[dict[str, Any]] = []
    # 记录哪些数据集里面有缺失值（残缺的数据）
    datasets_with_missing: set[str] = set()
    # 统计这个助教总共干了多少活
    processed_count = 0

    # 用 try...except 包住最外层，防止初始化直接炸掉
    try:
        # 尝试导入我们自己写的模型分类器
        try:
            from tabicl import TabICLClassifier
            from sklearn.utils.multiclass import type_of_target
        except ImportError as exc:
            worker_log(msg_prefix, f"导入失败: {exc}")
            # 连工具都找不到，直接辞职不干了
            return results, datasets_with_missing

        # 把配置里的设备改成当前分配给这个助教的专属设备
        active_config = dict(classifier_config)
        active_config["device"] = device_str

        worker_log(msg_prefix, f"启动成功，设备={device_str}")
        
        # ================= 2. 核心工作循环（遍历分配给自己的数据集） =================
        for dataset_dir in assigned_datasets:

            processed_count += 1
            dataset_dir = Path(dataset_dir)
            clf = None
            fit_kwargs: dict[str, Any] = {}
            X_train = X_test = X_val = X_all = None
            y_train = y_test = y_val = y_all = None
            y_pred = y_eval = pseudo_result = None
            baseline_acc: float | None = None
            pseudo_added = 0
            pseudo_applied = False
            train_ratio_before = float("nan")
            train_ratio_after = float("nan")
            fit_time = predict_time = total_time = 0.0

            # 用 try...except 包住单个数据集的处理流程，这样即使这个数据集报错，也不会导致整个进程崩溃
            try:
                # --- A. 检查数据集身份信息 ---
                info = load_dataset_info(dataset_dir)
                task_type = str(info.get("task_type", "")).lower() if info else ""
                # 这个脚本只测“分类(classification)”。如果是“回归(regression)”，直接跳过拿下一个。
                if task_type == "regression":
                    worker_log(msg_prefix, f"跳过 {dataset_dir.name}: task_type=regression")
                    continue
                if task_type and task_type not in CLASSIFICATION_TASKS:
                    worker_log(msg_prefix, f"跳过 {dataset_dir.name}: 未知 task_type={task_type}")
                    continue

                # --- B. 寻找并加载数据文件 ---
                train_path, val_path, test_path = find_data_files(dataset_dir)
                if train_path is None and test_path is None:
                    worker_log(msg_prefix, f"跳过 {dataset_dir.name}: 无可识别数据文件")
                    continue

                if train_path and test_path:
                    # 如果数据集本来就分好了“训练卷(train)”和“考试卷(test)”，直接分别加载
                    X_train, y_train = load_table(
                        train_path,
                        context=f"{dataset_dir.name}-train",
                        coerce_numeric=coerce_numeric,
                        dataset_id=dataset_dir.name,
                        missing_registry=datasets_with_missing,
                    )
                    X_test, y_test = load_table(
                        test_path,
                        context=f"{dataset_dir.name}-test",
                        coerce_numeric=coerce_numeric,
                        dataset_id=dataset_dir.name,
                        missing_registry=datasets_with_missing,
                    )
                else:
                    # 如果只有一个总文件，就自动帮它切分：80%做训练题，20%做考试题
                    X_all, y_all = load_table(
                        train_path,
                        context=f"{dataset_dir.name}-single",
                        coerce_numeric=coerce_numeric,
                        dataset_id=dataset_dir.name,
                        missing_registry=datasets_with_missing,
                    )
                    X_train, X_test, y_train, y_test = split_single_file_dataset(
                        X_all,
                        y_all,
                        dataset_name=dataset_dir.name,
                    )
                    val_path = None
                    worker_log(msg_prefix, f"{dataset_dir.name}: 单文件数据按 80/20 自动切分")

                # 如果有模拟考卷（验证集）且允许合并，就拼接到训练题库里
                if val_path:
                    X_val, y_val = load_table(
                        val_path,
                        context=f"{dataset_dir.name}-val",
                        coerce_numeric=coerce_numeric,
                        dataset_id=dataset_dir.name,
                        missing_registry=datasets_with_missing,
                    )
                    y_val = np.asarray(y_val)
                    if y_val.ndim > 1 and y_val.shape[-1] == 1:
                        y_val = y_val.reshape(-1)
                    if merge_val:
                        X_train = pd.concat([X_train, X_val], axis=0, ignore_index=True)
                        y_train = np.concatenate([y_train, y_val], axis=0)

                # 再次确认标签到底是不是分类，如果是连续数字（说明是回归），跳过不做了
                try:
                    target_type = type_of_target(y_train)
                except Exception:
                    target_type = None

                if (not task_type or task_type == "unknown") and target_type and target_type.startswith("continuous"):
                    worker_log(msg_prefix, f"跳过 {dataset_dir.name}: 检测到连续标签")
                    continue

                # ================= 4. 开始让模型学习和考试 =================
                clf, fit_kwargs = instantiate_worker_classifier(TabICLClassifier, active_config, msg_prefix)

                def fit_with_retry(train_X, train_y):
                    nonlocal clf, fit_kwargs
                    fit_start = time.perf_counter()
                    try:
                        clf.fit(train_X, train_y, **fit_kwargs)
                    except ValueError as exc:
                        clf, fit_kwargs = maybe_retry_with_checkpoint_alias(
                            classifier_cls=TabICLClassifier,
                            classifier_config=active_config,
                            fit_kwargs=fit_kwargs,
                            X_train=train_X,
                            y_train=train_y,
                            error=exc,
                            msg_prefix=msg_prefix,
                        )
                    return clf, time.perf_counter() - fit_start

                train_ratio_before = compute_train_ratio(y_train, y_test)
                train_ratio_after = train_ratio_before

                if pseudo_label_alpha is not None:
                    pseudo_result = run_pseudo_label_resplit_inference(
                        fit_classifier_fn=fit_with_retry,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        dataset_name=dataset_dir.name,
                        alpha=pseudo_label_alpha,
                    )
                    y_pred = pseudo_result["y_pred"]
                    y_eval = pseudo_result["y_eval"]
                    fit_time = float(pseudo_result["fit_time"])
                    predict_time = float(pseudo_result["predict_time"])
                    total_time = fit_time + predict_time
                    baseline_acc = float(pseudo_result["baseline_accuracy"])
                    pseudo_added = int(pseudo_result["pseudo_added"])
                    pseudo_applied = bool(pseudo_result["pseudo_applied"])
                    train_ratio_before = float(pseudo_result["train_ratio_before"])
                    train_ratio_after = float(pseudo_result["train_ratio_after"])
                else:
                    clf, fit_time = fit_with_retry(X_train, y_train)
                    predict_start = time.perf_counter()
                    y_pred = clf.predict(X_test)
                    predict_time = time.perf_counter() - predict_start
                    total_time = fit_time + predict_time
                    y_eval = y_test

                # ================= 5. 批改算分并上报 =================
                acc = float(np.mean(np.asarray(y_pred) == np.asarray(y_eval)))
                if pseudo_label_alpha is not None:
                    worker_log(
                        msg_prefix,
                        (
                            f"{dataset_dir.name}: accuracy={acc:.4f}"
                            + (f" baseline={baseline_acc:.4f}" if baseline_acc is not None else "")
                            + f" fit={fit_time:.2f}s predict={predict_time:.2f}s total={total_time:.2f}s"
                            + f" pseudo_added={pseudo_added} pseudo_applied={pseudo_applied}"
                            + f" train_ratio={train_ratio_before:.4f}->{train_ratio_after:.4f}"
                        ),
                    )
                else:
                    worker_log(
                        msg_prefix,
                        (
                            f"{dataset_dir.name}: accuracy={acc:.4f} "
                            f"fit={fit_time:.2f}s predict={predict_time:.2f}s total={total_time:.2f}s"
                        ),
                    )
                results.append(
                    {
                        "dataset": dataset_dir.name,
                        "accuracy": acc,
                        "time_s": total_time,
                        "fit_time_s": fit_time,
                        "predict_time_s": predict_time,
                        "baseline_accuracy": baseline_acc,
                        "pseudo_added": pseudo_added,
                        "pseudo_applied": pseudo_applied,
                        "pseudo_alpha": pseudo_label_alpha,
                        "train_ratio_before": train_ratio_before,
                        "train_ratio_after": train_ratio_after,
                    }
                )

            except Exception as exc:
                # 如果中途因为内存溢出(OOM)等爆炸了，打印一下错误，继续拿下一个数据集，不影响其他！
                worker_log(msg_prefix, f"评测失败 {dataset_dir.name}: {exc}")
                traceback.print_exc()
            finally:
                release_classifier_resources(clf)
                clf = None
                fit_kwargs = {}
                X_train = X_test = X_val = X_all = None
                y_train = y_test = y_val = y_all = None
                y_pred = y_eval = pseudo_result = None
                force_memory_cleanup(device_str)
                worker_log(msg_prefix, f"{dataset_dir.name}: cleanup completed")

    except BaseException as exc:
        worker_log(msg_prefix, f"worker 异常退出: {exc.__class__.__name__}: {exc}")
        traceback.print_exc()
    finally:
        # ================= 6. 循环结束，收工交卷 =================
        worker_log(msg_prefix, f"处理结束，成功数据集数: {len(results)} / 已领取任务数: {processed_count}")
        if result_queue is not None:
            try:
                # 兼容旧的 multiprocessing 方式
                result_queue.put((rank, results, datasets_with_missing))
            except Exception:
                pass
    return results, datasets_with_missing


# ------------------------------
# CLI 与主流程
# ------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在 TALENT 数据集上并行评测 TabICLClassifier")
    bool_action = argparse.BooleanOptionalAction

    parser.add_argument("--model-path", default='tabicl-classifier-v1.1-20250506.ckpt', help="本地 checkpoint 路径；为 None 时按 checkpoint_version 下载或查找")
    parser.add_argument(
        "--checkpoint-version",
        default='tabicl-classifier-v1.1-20250506.ckpt',
        help="模型版本名；与 README 保持一致，默认使用 TabICLv2 分类模型",
    )
    parser.add_argument(
        "--model-family",
        choices=["auto", "v1", "v1.1", "v2"],
        default="auto",
        help="用于推断 README 推荐的 n_estimators；默认自动根据 model-path/checkpoint-version 判断",
    )
    parser.add_argument("--allow-auto-download", action=bool_action, default=True)

    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="TALENT 数据目录根路径")
    parser.add_argument("--outdir", default="tabiclv1.1_pesudo_test_softmax0.8", help="结果输出目录")
    parser.add_argument("--max-datasets", type=int, default=None, help="限制最多处理的数据集数量")
    parser.add_argument("--merge-val", action=bool_action, default=True, help="是否把 val 合并到 train")
    parser.add_argument(
        "--coerce-numeric",
        action=bool_action,
        default=True,
        help="保留旧参数；当前脚本默认把特征交给 TabICL 内部处理",
    )
    parser.add_argument("--debug", action="store_true", help="启用脚本级 debug 日志")
    parser.add_argument(
        "--pseudo-label-alpha",
        type=float,
        default=0.8,
        help=(
            "启用二阶段伪标签推理。"
            "第一轮按 softmax 最大概率 max_prob >= alpha 筛选高置信样本打伪标签，"
            "然后直接并入训练集，并在剩余测试集上再次推理。"
        ),
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="若不指定，则按 README 自动使用 v1/v1.1=32、v2=8",
    )
    parser.add_argument("--norm-methods", nargs="+", default=None, help="归一化方法列表")
    parser.add_argument("--feat-shuffle-method", default="latin")
    parser.add_argument("--class-shuffle-method", default="shift")
    parser.add_argument("--outlier-threshold", type=float, default=4.0)
    parser.add_argument("--softmax-temperature", type=float, default=0.9)
    parser.add_argument("--average-logits", action=bool_action, default=True)
    parser.add_argument("--support-many-classes", action=bool_action, default=True)
    parser.add_argument("--batch-size", type=parse_optional_int, default=8, help="整数或 none")
    parser.add_argument("--kv-cache", type=parse_kv_cache, default=False)
    parser.add_argument("--use-amp", type=parse_auto_bool, default="auto")
    parser.add_argument("--use-fa3", type=parse_auto_bool, default="auto")
    parser.add_argument("--offload-mode", choices=["auto", "gpu", "cpu", "disk"], default="auto")
    parser.add_argument("--disk-offload-dir", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=parse_optional_int, default=None, help="整数或 none")
    parser.add_argument("--verbose", action="store_true", help="启用 TabICL 模型自身的 verbose 输出")

    parser.add_argument("--num-gpus", type=int, default=3, help="使用的 GPU 数量；默认使用 0..num_gpus-1")
    parser.add_argument("--gpu-ids", type=parse_gpu_ids, default="0,1,2", help="显式指定 GPU 编号，例如 0,2,3")

    parser.add_argument("--run-shard", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shard-rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shard-device-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dataset-manifest", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shard-result-file", default=None, help=argparse.SUPPRESS)
    return parser


def classifier_config_from_args(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    model_family, resolved_n_estimators = resolve_recommended_n_estimators(
        model_path=args.model_path,
        checkpoint_version=args.checkpoint_version,
        explicit_family=args.model_family,
        requested_n_estimators=args.n_estimators,
    )
    config = {
        "n_estimators": resolved_n_estimators,
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
        "use_amp": args.use_amp,
        "use_fa3": args.use_fa3,
        "offload_mode": args.offload_mode,
        "disk_offload_dir": args.disk_offload_dir,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
        "verbose": args.verbose,
    }
    return model_family, config


def resolve_device_ids(args: argparse.Namespace) -> list[int]:
    if args.gpu_ids is not None:
        return args.gpu_ids
    if args.num_gpus < 1:
        raise ValueError("--num-gpus 必须 >= 1")
    return list(range(args.num_gpus))


def split_datasets_round_robin(dirs: list[Path], device_ids: list[int]) -> list[tuple[int, int, list[str]]]:
    assignments: list[tuple[int, int, list[str]]] = []
    for rank, device_id in enumerate(device_ids):
        assigned_datasets = [str(path.resolve()) for path in dirs[rank::len(device_ids)]]
        assignments.append((rank, device_id, assigned_datasets))
    return assignments


def write_json_file(file_path: Path, payload: dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    tmp_path.replace(file_path)


def read_json_file(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_dataset_manifest(file_path: Path, dataset_paths: list[str]) -> None:
    write_json_file(file_path, {"datasets": dataset_paths})


def read_dataset_manifest(file_path: Path) -> list[str]:
    payload = read_json_file(file_path)
    dataset_paths = payload.get("datasets")
    if not isinstance(dataset_paths, list):
        raise ValueError(f"{file_path} 中缺少 datasets 列表")
    return [str(item) for item in dataset_paths]


def write_shard_result_file(
    file_path: Path,
    *,
    rank: int,
    device_id: int,
    results: list[dict[str, Any]],
    datasets_with_missing: set[str],
) -> None:
    write_json_file(
        file_path,
        {
            "rank": int(rank),
            "device_id": int(device_id),
            "results": results,
            "datasets_with_missing": sorted(datasets_with_missing),
        },
    )


def run_single_shard(args: argparse.Namespace, classifier_config: dict[str, Any]) -> int:
    if args.dataset_manifest is None:
        raise ValueError("--run-shard 模式下必须提供 --dataset-manifest")
    if args.shard_result_file is None:
        raise ValueError("--run-shard 模式下必须提供 --shard-result-file")
    if args.shard_rank is None:
        raise ValueError("--run-shard 模式下必须提供 --shard-rank")
    if args.shard_device_id is None:
        raise ValueError("--run-shard 模式下必须提供 --shard-device-id")

    assigned_datasets = read_dataset_manifest(Path(args.dataset_manifest))
    logging.info(
        "shard 模式: rank=%d, gpu=%s, datasets=%d",
        args.shard_rank,
        args.shard_device_id,
        len(assigned_datasets),
    )
    results, datasets_with_missing = evaluate_datasets_worker(
        rank=args.shard_rank,
        device_id=args.shard_device_id,
        assigned_datasets=assigned_datasets,
        classifier_config=classifier_config,
        result_queue=None,
        merge_val=args.merge_val,
        coerce_numeric=args.coerce_numeric,
        pseudo_label_alpha=args.pseudo_label_alpha,
        random_state=args.random_state,
    )
    write_shard_result_file(
        Path(args.shard_result_file),
        rank=args.shard_rank,
        device_id=args.shard_device_id,
        results=results,
        datasets_with_missing=datasets_with_missing,
    )
    return 0


def write_results(
    outdir: Path,
    results: list[dict[str, Any]],
    *,
    datasets_with_missing: set[str],
    script_duration: float,
    model_family: str,
    classifier_config: dict[str, Any],
) -> None:
    if not results:
        logging.info("没有获得成功的评测结果。")
        return

    results.sort(key=lambda item: str(item["dataset"]))

    total_time = sum(float(item["time_s"]) for item in results)
    avg_time = total_time / len(results)
    avg_acc = sum(float(item["accuracy"]) for item in results) / len(results)
    pseudo_enabled = any(item.get("pseudo_alpha") is not None for item in results)
    baseline_values = [
        float(item["baseline_accuracy"])
        for item in results
        if item.get("baseline_accuracy") is not None
    ]
    avg_baseline_acc = (
        sum(baseline_values) / len(baseline_values)
        if baseline_values
        else None
    )
    avg_train_ratio_before = (
        sum(float(item.get("train_ratio_before", float("nan"))) for item in results) / len(results)
    )
    avg_train_ratio_after = (
        sum(float(item.get("train_ratio_after", float("nan"))) for item in results) / len(results)
    )
    avg_pseudo_added = (
        sum(int(item.get("pseudo_added", 0)) for item in results) / len(results)
        if pseudo_enabled
        else None
    )
    pseudo_alpha_value = next(
        (item.get("pseudo_alpha") for item in results if item.get("pseudo_alpha") is not None),
        None,
    )
    pseudo_applied_count = sum(1 for item in results if item.get("pseudo_applied"))
    delta_values = [
        float(item["accuracy"]) - float(item["baseline_accuracy"])
        for item in results
        if item.get("baseline_accuracy") is not None
    ]
    avg_pseudo_acc_gain = (
        sum(delta_values) / len(delta_values)
        if delta_values
        else None
    )
    improved_count = sum(1 for delta in delta_values if delta > PSEUDO_ACC_DELTA_TOL)
    degraded_count = sum(1 for delta in delta_values if delta < -PSEUDO_ACC_DELTA_TOL)
    unchanged_count = sum(1 for delta in delta_values if abs(delta) <= PSEUDO_ACC_DELTA_TOL)

    csv_path = outdir / "talent_detailed.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write(
                "dataset,accuracy,time_s,fit_time_s,predict_time_s,baseline_accuracy,"
                "pseudo_added,pseudo_applied,pseudo_alpha,train_ratio_before,train_ratio_after\n"
            )
        for item in results:
            baseline_accuracy = item.get("baseline_accuracy")
            pseudo_alpha = item.get("pseudo_alpha")
            handle.write(
                "{dataset},{accuracy:.6f},{time_s:.3f},{fit_time_s:.3f},{predict_time_s:.3f},{baseline_accuracy},{pseudo_added},{pseudo_applied},{pseudo_alpha},{train_ratio_before:.6f},{train_ratio_after:.6f}\n".format(
                    dataset=item["dataset"],
                    accuracy=float(item["accuracy"]),
                    time_s=float(item["time_s"]),
                    fit_time_s=float(item["fit_time_s"]),
                    predict_time_s=float(item["predict_time_s"]),
                    baseline_accuracy=(
                        f"{float(baseline_accuracy):.6f}" if baseline_accuracy is not None else ""
                    ),
                    pseudo_added=int(item.get("pseudo_added", 0)),
                    pseudo_applied=str(bool(item.get("pseudo_applied", False))).lower(),
                    pseudo_alpha=("" if pseudo_alpha is None else f"{float(pseudo_alpha):.6f}"),
                    train_ratio_before=float(item.get("train_ratio_before", float("nan"))),
                    train_ratio_after=float(item.get("train_ratio_after", float("nan"))),
                )
            )
        handle.write(
            "Average,{avg_acc:.6f},{avg_time:.3f},,,{avg_baseline},,,,{avg_train_before:.6f},{avg_train_after:.6f}\n".format(
                avg_acc=avg_acc,
                avg_time=avg_time,
                avg_baseline=("" if avg_baseline_acc is None else f"{avg_baseline_acc:.6f}"),
                avg_train_before=avg_train_ratio_before,
                avg_train_after=avg_train_ratio_after,
            )
        )

    missing_results = [
        (str(item["dataset"]), float(item["accuracy"]))
        for item in results
        if str(item["dataset"]) in datasets_with_missing
    ]
    missing_names = sorted(name for name, _ in missing_results)
    avg_missing_acc = sum(acc for _, acc in missing_results) / len(missing_results) if missing_results else None

    summary_path = outdir / "talent_summary.txt"
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n--- Run at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        handle.write(f"Model family: {model_family}\n")
        handle.write(f"Checkpoint version: {classifier_config.get('checkpoint_version')}\n")
        handle.write(f"Model path: {classifier_config.get('model_path')}\n")
        handle.write(f"n_estimators: {classifier_config.get('n_estimators')}\n")
        handle.write(f"Total datasets: {len(results)}\n")
        handle.write(f"Average accuracy: {avg_acc:.6f}\n")
        if avg_baseline_acc is not None:
            handle.write(f"Average baseline accuracy: {avg_baseline_acc:.6f}\n")
        if avg_pseudo_acc_gain is not None:
            handle.write(f"Average pseudo ACC gain: {avg_pseudo_acc_gain:.6f}\n")
            handle.write(f"Datasets improved by pseudo: {improved_count}\n")
            handle.write(f"Datasets degraded by pseudo: {degraded_count}\n")
            handle.write(f"Datasets unchanged by pseudo: {unchanged_count}\n")
        handle.write(f"Total inference time s: {total_time:.3f}\n")
        handle.write(f"Average inference time s: {avg_time:.3f}\n")
        handle.write(f"Script total execution time s: {script_duration:.3f}\n")
        if pseudo_enabled:
            handle.write(f"Pseudo label alpha: {pseudo_alpha_value}\n")
            handle.write(f"Average pseudo-added samples: {avg_pseudo_added:.6f}\n")
            handle.write(f"Datasets with pseudo update applied: {pseudo_applied_count}\n")
            handle.write(
                f"Average train_ratio before/after pseudo update: {avg_train_ratio_before:.6f} / {avg_train_ratio_after:.6f}\n"
            )
        if missing_results:
            handle.write(f"Datasets with NaN values: {len(missing_names)}\n")
            handle.write(f"Average accuracy (NaN datasets): {avg_missing_acc:.6f}\n")
            handle.write(f"List (NaN datasets): {', '.join(missing_names)}\n")
        else:
            handle.write("Datasets with NaN values: 0\n")

    logging.info(
        "评测完成，共 %d 个数据集，平均准确率 %.4f，平均耗时 %.2fs，脚本总耗时 %.2fs",
        len(results),
        avg_acc,
        avg_time,
        script_duration,
    )


def main(argv=None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    args = parser.parse_args(raw_argv)

    if args.pseudo_label_alpha is not None and (
        not np.isfinite(args.pseudo_label_alpha) or not (0.0 <= args.pseudo_label_alpha <= 1.0)
    ):
        parser.error("--pseudo-label-alpha 必须是有限且位于 [0, 1] 的数值")

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    model_family, classifier_config = classifier_config_from_args(args)
    if model_family == "regressor":
        raise ValueError("当前 bench 脚本只支持分类模型，请改用 classifier checkpoint")

    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"数据目录不是文件夹: {data_root}")

    if args.run_shard:
        return run_single_shard(args, classifier_config)

    dirs = [path for path in sorted(data_root.iterdir()) if path.is_dir()]
    if args.max_datasets:
        dirs = dirs[:args.max_datasets]
    if not dirs:
        logging.info("没有可处理的数据集目录。")
        return 0

    summarize_task_types(dirs)

    device_ids = resolve_device_ids(args)
    worker_device_ids = device_ids[: min(len(device_ids), len(dirs))]
    script_start_time = time.perf_counter()

    logging.info(
        "模型版本族=%s, checkpoint=%s, model_path=%s, n_estimators=%s",
        model_family,
        classifier_config.get("checkpoint_version"),
        classifier_config.get("model_path"),
        classifier_config.get("n_estimators"),
    )
    logging.info(
        "待处理数据集总数: %d，worker 数量: %d，GPU 列表: %s（固定切分）",
        len(dirs),
        len(worker_device_ids),
        worker_device_ids,
    )

    shard_dir = outdir / "_shards" / time.strftime("%Y%m%d_%H%M%S")
    script_path = Path(__file__).resolve()
    child_env = dict(os.environ)
    child_env["PYTHONUNBUFFERED"] = "1"

    processes: list[tuple[int, int, int, Path, subprocess.Popen[str]]] = []
    for rank, device_id, assigned_datasets in split_datasets_round_robin(dirs, worker_device_ids):
        assigned_count = len(assigned_datasets)
        logging.info(
            "worker rank=%d -> gpu=%s: 固定分配 %d 个数据集，result=%s",
            rank,
            device_id,
            assigned_count,
            shard_dir / f"results_rank{rank}_gpu{device_id}.json",
        )
        manifest_path = shard_dir / f"datasets_rank{rank}_gpu{device_id}.json"
        result_path = shard_dir / f"results_rank{rank}_gpu{device_id}.json"
        write_dataset_manifest(manifest_path, assigned_datasets)
        command = [
            sys.executable,
            str(script_path),
            *raw_argv,
            "--run-shard",
            "--shard-rank",
            str(rank),
            "--shard-device-id",
            str(device_id),
            "--dataset-manifest",
            str(manifest_path),
            "--shard-result-file",
            str(result_path),
        ]
        process = subprocess.Popen(
            command,
            env=child_env,
            text=True,
        )
        processes.append((rank, device_id, assigned_count, result_path, process))

    results_by_worker: list[tuple[int, list[dict[str, Any]], set[str]]] = []
    for rank, device_id, assigned_count, result_path, process in processes:
        exit_code = process.wait()
        if not result_path.exists():
            logging.error(
                "shard rank=%d gpu=%s datasets=%d exitcode=%s result=%s status=missing-result",
                rank,
                device_id,
                assigned_count,
                exit_code,
                result_path,
            )
            results_by_worker.append((rank, [], set()))
            continue
        try:
            payload = read_json_file(result_path)
            payload_rank, payload_device_id, result_items, missing_items = validate_shard_result_payload(payload)
            if payload_rank != rank or payload_device_id != device_id:
                raise ValueError(
                    f"结果元信息不匹配: payload(rank={payload_rank}, device_id={payload_device_id})"
                )
        except Exception as exc:
            logging.error(
                "shard rank=%d gpu=%s datasets=%d exitcode=%s result=%s status=invalid-result error=%s",
                rank,
                device_id,
                assigned_count,
                exit_code,
                result_path,
                exc,
            )
            results_by_worker.append((rank, [], set()))
            continue

        log_fn = logging.info if exit_code == 0 else logging.warning
        log_fn(
            "shard rank=%d gpu=%s datasets=%d exitcode=%s result=%s loaded=%d",
            rank,
            device_id,
            assigned_count,
            exit_code,
            result_path,
            len(result_items),
        )
        results_by_worker.append((rank, result_items, missing_items))

    all_results: list[dict[str, Any]] = []
    all_missing_registry: set[str] = set()
    for _, result_items, missing_items in sorted(results_by_worker, key=lambda item: item[0]):
        all_results.extend(result_items)
        all_missing_registry.update(missing_items)

    script_duration = time.perf_counter() - script_start_time
    write_results(
        outdir,
        all_results,
        datasets_with_missing=all_missing_registry,
        script_duration=script_duration,
        model_family=model_family,
        classifier_config=classifier_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
