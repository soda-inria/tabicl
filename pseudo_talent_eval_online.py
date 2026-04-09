# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# TALENT batch evaluation with online checkpoint watching.

# Key behavior:
# - If `--models_dir` is provided, keep polling this directory.
# - Whenever a new checkpoint appears and becomes stable, evaluate it once.
# - Each model evaluation shards datasets across up to 8 GPUs in parallel.
# - Write per-model outputs:
#   - <outdir>/<model_tag>/talent_detailed.txt
#   - <outdir>/<model_tag>/talent_summary.txt
# - Append global summary:
#   - <outdir>/all_models_summary.tsv
# """

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Allow running the script directly from repo root without editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

# ===== Fixed defaults =====
DEFAULT_MODEL_PATH = "/vast/users/guangyi.chen/causal_group/zijian.li/LDM/Orion-MSP/stage1/checkpoint/dir2/step-27250d.ckpt"
DEFAULT_DATA_ROOT = "/vast/users/guangyi.chen/causal_group/zijian.li/LDM/data"
# DEFAULT_OUTDIR = "evaluation_results_fulltrain_stage_infonceall_0.05"
DEFAULT_OUTDIR = "evaluation_results_fulltrain_stage2_1024"
FIXED_GPUS = 8
COERCE_NUMERIC = True
MERGE_VAL = True
SKIP_REGRESSION = True


def parse_bool_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def run_pseudo_self_training_rounds(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pseudo_repeat_times: int,
    confidence_threshold: float = 0.90,
) -> tuple[np.ndarray, float, float, int, int, float]:
    """
    真实伪标签自训练循环 (Confidence-based Pseudo Self-training)
    - 第0轮：用原始训练集训练模型。
    - 后续轮次：模型去预测测试集，输出概率。挑选出“预测概率大于门槛值（即交叉熵极低）”
      的样本，认为模型极其自信，把它们及其预测标签当做新数据加进去重训。
    """
    repeat_times = max(0, int(pseudo_repeat_times))
    total_rounds = 1 + repeat_times

    cur_X_train = X_train
    cur_y_train = y_train
    
    already_added = np.zeros(int(X_test.shape[0]), dtype=bool)
    total_added = 0  # 记录总共成功拉了多少个测试集样本当“壮丁”

    total_fit_time = 0.0
    total_pred_time = 0.0
    y_pred = None

    # 4. 开始一轮一轮地循环训练
    for round_idx in range(total_rounds):
        # --- 步骤 A：模型学习（训练） ---
        # 记录开始时间，让模型在当前的训练集上学习
        t_fit = time.perf_counter()
        clf.fit(cur_X_train, cur_y_train)
        total_fit_time += time.perf_counter() - t_fit

        # --- 步骤 B：模型考试（输出概率） ---
        t_pred = time.perf_counter()
        
        # 获取概率分布，计算最大概率(自信度)和对应的预测标签
        probs = clf.predict_proba(X_test)
        max_probs = np.max(probs, axis=1)
        # 将概率最大的索引映射回实际类别标签
        pred_classes = clf.classes_[np.argmax(probs, axis=1)]
        
        total_pred_time += time.perf_counter() - t_pred

        # 如果这已经是最后一轮了，那考完试就可以放学了，不用再挑题进题库了
        if round_idx == total_rounds - 1:
            y_pred = pred_classes
            break

        # --- 步骤 C：利用交叉熵门槛(高概率)挑出自信的题 ---
        # 只要 max_prob >= confidence_threshold，就说明该伪标签对应极低的交叉熵
        confident_mask = np.asarray(max_probs >= confidence_threshold)
        
        # 挑出那些“做对了” 并且 “之前没被加进过题库” 的新题目
        new_mask = confident_mask & (~already_added)
        
        # --- 步骤 D：扩充题库（壮大训练集） ---
        if np.any(new_mask):
            # 把这些测试题的题目（特征 X）接在现有的训练题后面
            cur_X_train = np.concatenate([cur_X_train, X_test[new_mask]], axis=0)
            # 把这些题的预测答案（伪标签 y）接在现有答案后面
            cur_y_train = np.concatenate([cur_y_train, pred_classes[new_mask]], axis=0)
            
            # 在小本本上把这些题打上勾，表示已经加进去了，下次别加了
            already_added[new_mask] = True
            # 顺便统计一下这一轮加了多少题
            total_added += int(new_mask.sum())

    assert y_pred is not None
    
    # 5. 循环结束，算一下最终的账（统计最终数据比例）
    final_train_count = int(cur_X_train.shape[0])
    test_count = int(X_test.shape[0])
    # 算一下“最终训练题量”占“总题量（训练+测试）”的百分比
    final_train_ratio = float(final_train_count / (final_train_count + test_count)) if (final_train_count + test_count) > 0 else float("nan")

    # 把最后一次的预测结果、总训练耗时、总预测耗时、加了多少题等信息汇报上去
    return y_pred, total_fit_time, total_pred_time, total_added, total_rounds, final_train_ratio


def convert_features(X: np.ndarray, enabled: bool) -> np.ndarray:
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


def count_missing(values: np.ndarray) -> int:
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
    missing = count_missing(values)
    if missing:
        logging.warning("%s: 原始数据包含 %d 个 NaN/缺失值", context, missing)
        if dataset_id and missing_registry is not None:
            missing_registry.add(dataset_id)


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


def find_data_files(dataset_dir: Path):
    files = [p for p in dataset_dir.iterdir() if p.is_file()]
    lower = {p.name.lower(): p for p in files}

    def by_suffix(key: str):
        for name, p in lower.items():
            if name.endswith(key):
                return p
        return None

    n_train = by_suffix("n_train.npy")
    c_train = by_suffix("c_train.npy")
    y_train = by_suffix("y_train.npy")
    n_val = by_suffix("n_val.npy")
    c_val = by_suffix("c_val.npy")
    y_val = by_suffix("y_val.npy")
    n_test = by_suffix("n_test.npy")
    c_test = by_suffix("c_test.npy")
    y_test = by_suffix("y_test.npy")

    if y_train and y_test and (n_train or c_train) and (n_test or c_test):
        val_pair = None
        if y_val and (n_val or c_val):
            val_pair = (n_val, c_val, y_val)
        return (n_train, c_train, y_train), val_pair, (n_test, c_test, y_test)

    table_candidates = [p for p in files if p.suffix.lower() in {".npy", ".npz", ".csv", ".tsv", ".parquet"}]
    if len(table_candidates) == 1:
        return table_candidates[0], None, None
    return None, None, None


def load_pair(
    X_path: Path,
    y_path: Path,
    context: str = "",
    coerce_numeric: bool = False,
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
):
    X = load_array(X_path)
    y = load_array(y_path)
    log_nan_presence(f"{context or X_path.stem}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{context or X_path.stem}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim > 1:
        if y.shape[1] == 1:
            y = y.squeeze(1)
        elif y.shape[0] == 1:
            y = y.squeeze(0)
    X, y = handle_missing_entries(X, y, context=context or X_path.stem)
    X = convert_features(X, coerce_numeric)
    return X, y


def load_split(
    num_path: Optional[Path],
    cat_path: Optional[Path],
    y_path: Path,
    context: str = "",
    coerce_numeric: bool = False,
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
):
    feats = []
    base = context or (num_path.stem if num_path else (cat_path.stem if cat_path else y_path.stem))

    if num_path:
        Xn = np.asarray(load_array(num_path))
        if Xn.ndim == 1:
            Xn = Xn.reshape(-1, 1)
        log_nan_presence(f"{base}-num_raw", Xn, dataset_id=dataset_id, missing_registry=missing_registry)
        feats.append(Xn)
    if cat_path:
        Xc = np.asarray(load_array(cat_path))
        if Xc.ndim == 1:
            Xc = Xc.reshape(-1, 1)
        log_nan_presence(f"{base}-cat_raw", Xc, dataset_id=dataset_id, missing_registry=missing_registry)
        feats.append(Xc)
    if not feats:
        raise ValueError("缺少数值/类别特征文件")

    n = feats[0].shape[0]
    for i, f in enumerate(feats):
        if f.shape[0] != n:
            raise ValueError(f"特征数量不一致: #{i} 有 {f.shape[0]} vs {n}")

    X = feats[0] if len(feats) == 1 else np.concatenate(feats, axis=1)
    log_nan_presence(f"{base}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    y = np.asarray(load_array(y_path))
    log_nan_presence(f"{base}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)
    if y.ndim > 1:
        if y.shape[1] == 1:
            y = y.squeeze(1)
        elif y.shape[0] == 1:
            y = y.squeeze(0)
    X, y = handle_missing_entries(X, y, context=base)
    X = convert_features(X, coerce_numeric)
    return X, y


def load_table(
    file_path: Union[Path, Tuple],
    context: str = "",
    coerce_numeric: bool = False,
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(file_path, (tuple, list)):
        if len(file_path) == 2:
            Xp, yp = Path(file_path[0]), Path(file_path[1])
            return load_pair(
                Xp,
                yp,
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
        raise ValueError(f"Unsupported tuple for load_table: {file_path}")

    path: Path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        try:
            arr = np.load(path, allow_pickle=False)
        except ValueError:
            arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        data = np.asarray(arr)
    elif suffix == ".parquet":
        data = pd.read_parquet(path).values
    else:
        sep = "\t" if suffix == ".tsv" else None
        data = pd.read_csv(path, sep=sep, header=None).values

    if data.ndim == 1:
        raise ValueError(f"Unsupported 1D data in {path}")

    log_target = context or str(path)
    log_nan_presence(f"{log_target}-raw", data, dataset_id=dataset_id, missing_registry=missing_registry)

    col0 = data[:, 0]
    try:
        uniques0 = np.unique(col0)
    except Exception:
        uniques0 = np.array([])

    if 0 < uniques0.size < max(2, data.shape[0] // 2):
        y = data[:, 0]
        X = data[:, 1:]
        which = "first"
    else:
        y = data[:, -1]
        X = data[:, :-1]
        which = "last"

    log_nan_presence(f"{log_target}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{log_target}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

    X = np.asarray(pd.DataFrame(X).values)
    y = pd.Series(y).values
    X, y = handle_missing_entries(X, y, context=log_target)
    X = convert_features(X, coerce_numeric)
    logging.info("%s: 使用单文件启发式拆分标签 (取 %s 列)", log_target, which)
    return X, y


def load_dataset_info(dataset_dir: Path) -> Optional[dict]:
    p = dataset_dir / "info.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logging.warning("读取 %s 失败: %s", p, exc)
        return None


def summarize_task_types(dirs: List[Path]) -> None:
    counts = {"regression": 0, "binclass": 0, "multiclass": 0, "unknown": 0}
    for d in dirs:
        info = load_dataset_info(d)
        t = str(info.get("task_type", "")).lower() if info else ""
        if not t:
            counts["unknown"] += 1
        elif t in counts:
            counts["regression" if t == "regression" else t] += 1
        else:
            counts["unknown"] += 1
    logging.info(
        "任务统计: regression=%d, binclass=%d, multiclass=%d, unknown=%d, 总计=%d",
        counts["regression"],
        counts["binclass"],
        counts["multiclass"],
        counts["unknown"],
        len(dirs),
    )


def resolve_gpu_devices(max_gpus: int = FIXED_GPUS) -> List[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        devices = [x.strip() for x in env.split(",") if x.strip()]
    else:
        try:
            import torch

            n = torch.cuda.device_count()
        except Exception:
            n = 0
        devices = [str(i) for i in range(n)] if n > 0 else [str(i) for i in range(max_gpus)]

    if len(devices) > max_gpus:
        devices = devices[:max_gpus]
    return devices


def build_classifier(
    model_path: str,
    n_estimators: int,
    batch_size: Optional[int],
    n_jobs: Optional[int],
    use_torch_compile: bool,
    torch_compile_mode: Optional[str],
    torch_compile_backend: Optional[str],
    torch_compile_fullgraph: bool,
    torch_compile_dynamic: Optional[bool],
):
    """Build classifier with backward-compatible kwargs across tabicl versions."""
    from tabicl.sklearn.classifier import TabICLClassifier

    kwargs = {
        "verbose": False,
        "model_path": model_path,
        "device": "cuda",
        "use_amp": True,
        "n_estimators": n_estimators,
    }
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if n_jobs is not None:
        kwargs["n_jobs"] = n_jobs
    try:
        sig = inspect.signature(TabICLClassifier.__init__)
        params = sig.parameters
        if "use_kv_cache" in params:
            kwargs["use_kv_cache"] = True
        if use_torch_compile and "use_torch_compile" in params:
            kwargs["use_torch_compile"] = True
            if "torch_compile_mode" in params:
                kwargs["torch_compile_mode"] = torch_compile_mode
            if "torch_compile_backend" in params:
                kwargs["torch_compile_backend"] = torch_compile_backend
            if "torch_compile_fullgraph" in params:
                kwargs["torch_compile_fullgraph"] = torch_compile_fullgraph
            if "torch_compile_dynamic" in params:
                kwargs["torch_compile_dynamic"] = torch_compile_dynamic
    except Exception:
        pass
    return TabICLClassifier(**kwargs)


def preload_model_once(clf, gpu_device: str) -> None:
    """Force one-time checkpoint load and prevent per-dataset reload in legacy classifier."""
    load_fn = getattr(clf, "_load_model", None)
    if not callable(load_fn):
        return

    t0 = time.perf_counter()
    load_fn()

    # Newer classifier variants use _loaded_model_key in _ensure_model_loaded.
    if hasattr(clf, "_get_model_load_key"):
        try:
            clf._loaded_model_key = clf._get_model_load_key()
        except Exception:
            pass

    # Legacy classifier calls _load_model() in every fit(); make it a no-op after preload.
    def _skip_reloading():
        return None

    try:
        clf._load_model = _skip_reloading
    except Exception:
        pass

    logging.info("[GPU %s] 模型预加载完成: %.2fs", gpu_device, time.perf_counter() - t0)


def _set_cpu_thread_limits(cpu_threads: int) -> None:
    cpu_threads = max(1, int(cpu_threads))
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_threads)

    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=cpu_threads)
    except Exception:
        pass


def _dataset_cache_file(dataset_dir: Path, cache_root: Path) -> Path:
    dataset_key = hashlib.md5(str(dataset_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return cache_root / f"{dataset_dir.name}__{dataset_key}.npz"


def _load_cached_dataset(cache_file: Path):
    try:
        with np.load(cache_file, allow_pickle=True) as z:
            X_train = z["X_train"]
            y_train = z["y_train"]
            X_test = z["X_test"]
            y_test = z["y_test"]
            train_ratio = float(z["train_ratio"])
        return X_train, y_train, X_test, y_test, train_ratio
    except Exception:
        return None


def _prepare_dataset_payload(dataset_dir: Path, cache_root: Path):
    info = load_dataset_info(dataset_dir)
    ttype = str(info.get("task_type", "")).lower() if info else None
    if SKIP_REGRESSION and ttype == "regression":
        return None, "回归任务", False

    cache_file = _dataset_cache_file(dataset_dir, cache_root)
    if cache_file.exists():
        cached = _load_cached_dataset(cache_file)
        if cached is not None:
            X_train, y_train, X_test, y_test, train_ratio = cached
            return (
                {
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                    "y_test": y_test,
                    "train_ratio": train_ratio,
                },
                None,
                True,
            )
        logging.warning("%s: 缓存文件损坏，重新构建: %s", dataset_dir.name, cache_file)

    missing_datasets: set[str] = set()
    train_path, val_path, test_path = find_data_files(dataset_dir)
    if train_path is None and test_path is None:
        return None, "未识别数据文件", False
    if not (train_path and test_path):
        return None, "只有单文件，当前策略跳过", False

    X_train, y_train = load_table(
        train_path,
        context=f"{dataset_dir.name}-train",
        coerce_numeric=COERCE_NUMERIC,
        dataset_id=dataset_dir.name,
        missing_registry=missing_datasets,
    )
    X_test, y_test = load_table(
        test_path,
        context=f"{dataset_dir.name}-test",
        coerce_numeric=COERCE_NUMERIC,
        dataset_id=dataset_dir.name,
        missing_registry=missing_datasets,
    )

    if dataset_dir.name in missing_datasets:
        logging.info("%s: 原始数据包含缺失值", dataset_dir.name)

    if val_path is not None and MERGE_VAL:
        try:
            X_val, y_val = load_table(
                val_path,
                context=f"{dataset_dir.name}-val",
                coerce_numeric=COERCE_NUMERIC,
                dataset_id=dataset_dir.name,
                missing_registry=missing_datasets,
            )
            if X_val.ndim == 3 and X_val.shape[1] == 1:
                X_val = X_val.squeeze(1)
            X_val = X_val.astype(np.float32, copy=False)
            X_train = np.concatenate([X_train, X_val], axis=0)
            y_train = np.concatenate([y_train, y_val], axis=0)
            logging.info("%s: 已将 val(%d) 并入 train", dataset_dir.name, int(X_val.shape[0]))
        except Exception as exc:
            logging.warning("%s: 合并 val 失败：%s", dataset_dir.name, exc)

    if X_train.ndim == 3 and X_train.shape[1] == 1:
        X_train = X_train.squeeze(1)
    if X_test.ndim == 3 and X_test.shape[1] == 1:
        X_test = X_test.squeeze(1)
    X_train = X_train.astype(np.float32, copy=False)
    X_test = X_test.astype(np.float32, copy=False)

    train_count = int(X_train.shape[0])
    test_count = int(X_test.shape[0])
    total_count = train_count + test_count
    train_ratio = float(train_count / total_count) if total_count > 0 else float("nan")

    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        np.savez(
            cache_file,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            train_ratio=np.array(train_ratio, dtype=np.float32),
        )
    except Exception as exc:
        logging.warning("%s: 写缓存失败 (%s): %s", dataset_dir.name, cache_file, exc)

    return (
        {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "train_ratio": train_ratio,
        },
        None,
        False,
    )


def _worker_loop(
    task_queue,
    result_queue,
    gpu_device: str,
    cache_root: str,
    clf_n_estimators: int,
    clf_batch_size: Optional[int],
    clf_n_jobs: Optional[int],
    cpu_threads: int,
    use_torch_compile: bool,
    torch_compile_mode: Optional[str],
    torch_compile_backend: Optional[str],
    torch_compile_fullgraph: bool,
    torch_compile_dynamic: Optional[bool],
    torchinductor_cache_dir: Optional[str],
    pseudo_repeat_times: int,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    if torchinductor_cache_dir:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = torchinductor_cache_dir
    _set_cpu_thread_limits(cpu_threads)
    try:
        import torch

        torch.cuda.set_device(0)
        torch.set_num_threads(max(1, int(cpu_threads)))
    except Exception:
        pass

    logging.info("[GPU %s] 启动，常驻worker模式", gpu_device)

    clf = None
    current_model_path = None
    cache_root_path = Path(cache_root)

    while True:
        task = task_queue.get()
        if task is None:
            break

        eval_id = int(task["eval_id"])
        model_path = str(task["model_path"])
        d = Path(task["dataset_dir"])

        if clf is None or current_model_path != model_path:
            clf = build_classifier(
                model_path=model_path,
                n_estimators=clf_n_estimators,
                batch_size=clf_batch_size,
                n_jobs=clf_n_jobs,
                use_torch_compile=use_torch_compile,
                torch_compile_mode=torch_compile_mode,
                torch_compile_backend=torch_compile_backend,
                torch_compile_fullgraph=torch_compile_fullgraph,
                torch_compile_dynamic=torch_compile_dynamic,
            )
            preload_model_once(clf, gpu_device)
            current_model_path = model_path

        ds_t0 = time.perf_counter()
        try:
            payload, skip_reason, cache_hit = _prepare_dataset_payload(d, cache_root_path)
            if payload is None:
                result_queue.put(
                    {
                        "eval_id": eval_id,
                        "status": "skip",
                        "dataset": d.name,
                        "reason": skip_reason,
                    }
                )
                continue

            X_train = payload["X_train"]
            y_train = payload["y_train"]
            X_test = payload["X_test"]
            y_test = payload["y_test"]
            train_ratio = float(payload["train_ratio"])
            this_repeat_times = int(task.get("pseudo_repeat_times", pseudo_repeat_times))

            prep_time = time.perf_counter() - ds_t0

            y_pred, fit_time, pred_time, pseudo_added, pseudo_rounds, final_train_ratio = run_pseudo_self_training_rounds(
                clf=clf,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                pseudo_repeat_times=this_repeat_times,
            )

            acc = float(np.mean(y_pred == y_test))
            infer_time = fit_time + pred_time
            total_e2e = time.perf_counter() - ds_t0
            logging.info(
                "[GPU %s] %s: acc=%.4f, infer=%.2fs (fit=%.2fs, pred=%.2fs), prep=%.2fs, e2e=%.2fs, train_ratio=%.4f -> %.4f, pseudo_rounds=%d, pseudo_added=%d",
                gpu_device,
                d.name,
                acc,
                infer_time,
                fit_time,
                pred_time,
                prep_time,
                total_e2e,
                train_ratio,
                final_train_ratio,
                pseudo_rounds,
                pseudo_added,
            )
            result_queue.put(
                {
                    "eval_id": eval_id,
                    "status": "ok",
                    "dataset": d.name,
                    "acc": acc,
                    "infer_time": infer_time,
                    "train_ratio": train_ratio,
                    "prep_time": prep_time,
                    "fit_time": fit_time,
                    "pred_time": pred_time,
                    "total_e2e": total_e2e,
                    "cache_hit": cache_hit,
                    "pseudo_rounds": pseudo_rounds,
                    "pseudo_added": pseudo_added,
                    "final_train_ratio": final_train_ratio,
                }
            )

        except Exception as exc:
            logging.exception("[GPU %s] 评测失败 %s: %s", gpu_device, d.name, exc)
            result_queue.put(
                {
                    "eval_id": eval_id,
                    "status": "error",
                    "dataset": d.name,
                    "error": str(exc),
                }
            )


class PersistentEvaluatorPool:
    def __init__(
        self,
        gpu_devices: List[str],
        cache_root: Path,
        clf_n_estimators: int,
        clf_batch_size: Optional[int],
        clf_n_jobs: Optional[int],
        cpu_threads: int,
        use_torch_compile: bool,
        torch_compile_mode: Optional[str],
        torch_compile_backend: Optional[str],
        torch_compile_fullgraph: bool,
        torch_compile_dynamic: Optional[bool],
        torchinductor_cache_dir: Optional[str],
        pseudo_repeat_times: int,
    ) -> None:
        if not gpu_devices:
            raise RuntimeError("未检测到可用GPU设备，无法执行并行评测。")

        self.ctx = mp.get_context("spawn")
        self.task_queue = self.ctx.Queue(maxsize=max(64, len(gpu_devices) * 8))
        self.result_queue = self.ctx.Queue()
        self.processes = []
        self.eval_id = 0
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)

        for gpu_dev in gpu_devices:
            p = self.ctx.Process(
                target=_worker_loop,
                args=(
                    self.task_queue,
                    self.result_queue,
                    gpu_dev,
                    str(self.cache_root),
                    clf_n_estimators,
                    clf_batch_size,
                    clf_n_jobs,
                    cpu_threads,
                    use_torch_compile,
                    torch_compile_mode,
                    torch_compile_backend,
                    torch_compile_fullgraph,
                    torch_compile_dynamic,
                    torchinductor_cache_dir,
                    int(pseudo_repeat_times),
                ),
                daemon=False,
            )
            p.start()
            self.processes.append(p)

        logging.info("常驻评测池已启动: %d workers, cache=%s", len(self.processes), self.cache_root)

    def close(self) -> None:
        for _ in self.processes:
            self.task_queue.put(None)
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

    def evaluate_model(self, model_path: str, dirs: List[Path]):
        eval_id = self.eval_id
        self.eval_id += 1
        total_tasks = len(dirs)

        for d in dirs:
            self.task_queue.put({"eval_id": eval_id, "model_path": model_path, "dataset_dir": str(d)})

        results = []
        done = 0
        skip_count = 0
        err_count = 0
        cache_hits = 0
        while done < total_tasks:
            msg = self.result_queue.get()
            if int(msg.get("eval_id", -1)) != eval_id:
                continue
            done += 1
            status = msg.get("status")
            if status == "ok":
                results.append(
                    (
                        msg["dataset"],
                        float(msg["acc"]),
                        float(msg["infer_time"]),
                        float(msg["train_ratio"]),
                        float(msg["prep_time"]),
                        float(msg["fit_time"]),
                        float(msg["pred_time"]),
                        float(msg["total_e2e"]),
                    )
                )
                if msg.get("cache_hit"):
                    cache_hits += 1
            elif status == "skip":
                skip_count += 1
            else:
                err_count += 1
                logging.warning("数据集失败: %s (%s)", msg.get("dataset"), msg.get("error"))

        logging.info(
            "模型 %s 评测完成: ok=%d, skip=%d, error=%d, cache_hit=%d/%d",
            Path(model_path).stem,
            len(results),
            skip_count,
            err_count,
            cache_hits,
            total_tasks,
        )
        return results


def evaluate_model(
    model_path: str,
    outdir_root: Path,
    evaluator_pool: PersistentEvaluatorPool,
    dirs: List[Path],
) -> Tuple[str, int, float, float, float, float]:
    """
    Returns:
    (model_tag, total_datasets, avg_acc, total_time, avg_time, avg_train_ratio)
    """
    model_tag = Path(model_path).stem
    outdir = outdir_root / model_tag
    outdir.mkdir(parents=True, exist_ok=True)

    results = evaluator_pool.evaluate_model(model_path=model_path, dirs=dirs)
    results.sort(key=lambda x: x[0])

    detailed_path = outdir / "talent_detailed.txt"
    summary_path = outdir / "talent_summary.txt"

    if results:
        with open(detailed_path, "w") as f:
            f.write("dataset\taccuracy\ttime_s\ttrain_ratio\tprep_s\tfit_s\tpredict_s\ttotal_e2e_s\n")
            for name, acc, dur, tr, prep_s, fit_s, pred_s, e2e_s in results:
                tr_str = f"{tr:.6f}" if tr == tr else "nan"
                f.write(f"{name}\t{acc:.6f}\t{dur:.3f}\t{tr_str}\t{prep_s:.3f}\t{fit_s:.3f}\t{pred_s:.3f}\t{e2e_s:.3f}\n")

        total_time = sum(dur for _, _, dur, _, _, _, _, _ in results)
        avg_time = total_time / len(results)
        avg_acc = sum(acc for _, acc, _, _, _, _, _, _ in results) / len(results)
        tr_values = [tr for _, _, _, tr, _, _, _, _ in results if tr == tr]
        avg_prep_time = sum(prep for _, _, _, _, prep, _, _, _ in results) / len(results)
        avg_fit_time = sum(fit for _, _, _, _, _, fit, _, _ in results) / len(results)
        avg_pred_time = sum(pred for _, _, _, _, _, _, pred, _ in results) / len(results)
        avg_e2e_time = sum(e2e for _, _, _, _, _, _, _, e2e in results) / len(results)
        avg_train_ratio = (sum(tr_values) / len(tr_values)) if tr_values else float("nan")

        with open(summary_path, "w") as f:
            f.write(f"Model: {model_tag}\n")
            f.write(f"Total datasets: {len(results)}\n")
            f.write(f"Average accuracy: {avg_acc:.6f}\n")
            f.write(f"Total time s: {total_time:.3f}\n")
            f.write(f"Average time s: {avg_time:.3f}\n")
            f.write(f"Average prep time s: {avg_prep_time:.3f}\n")
            f.write(f"Average fit time s: {avg_fit_time:.3f}\n")
            f.write(f"Average predict time s: {avg_pred_time:.3f}\n")
            f.write(f"Average end-to-end time s: {avg_e2e_time:.3f}\n")
            f.write(f"Average train_ratio: {avg_train_ratio:.6f}\n")

        logging.info("[%s] 汇总完成：%s / %s", model_tag, detailed_path, summary_path)
        return model_tag, len(results), avg_acc, total_time, avg_time, avg_train_ratio

    logging.info("[%s] 没有成功的评测结果。", model_tag)
    return model_tag, 0, float("nan"), 0.0, float("nan"), float("nan")


def _extract_last_int(stem: str) -> Optional[int]:
    nums = re.findall(r"\d+", stem)
    if not nums:
        return None
    return int(nums[-1])


def discover_ckpts(models_dir: Path, step_mod: int, reverse_order: bool = False) -> List[Path]:
    files = [p for p in models_dir.iterdir() if p.is_file() and p.suffix.lower() in {".ckpt", ".pt", ".pth"}]

    def sort_key(p: Path):
        step = _extract_last_int(p.stem)
        if step is not None:
            return (0, step, p.stem)
        return (1, int(p.stat().st_mtime), p.stem)

    ordered = sorted(files, key=sort_key, reverse=bool(reverse_order))
    if step_mod <= 1:
        return ordered

    filtered = []
    for p in ordered:
        step = _extract_last_int(p.stem)
        if step is None or step % step_mod == 0:
            filtered.append(p)
    return filtered


def is_file_stable(p: Path, last_sizes: dict[str, int], stable_sec: float) -> bool:
    key = str(p)
    try:
        st = p.stat()
    except Exception:
        return False

    age = time.time() - st.st_mtime
    if age < stable_sec:
        return False

    size = st.st_size
    prev = last_sizes.get(key)
    last_sizes[key] = size
    return prev is not None and prev == size


def ensure_master_header(master_path: Path) -> None:
    if not master_path.exists():
        master_path.parent.mkdir(parents=True, exist_ok=True)
        with open(master_path, "w") as f:
            f.write("model_name\ttotal_datasets\taverage_accuracy\ttotal_time_s\taverage_time_s\taverage_train_ratio\n")


def append_master(master_path: Path, model_tag: str, total: int, avg_acc: float, total_t: float, avg_t: float, avg_tr: float):
    avg_acc_str = f"{avg_acc:.6f}" if avg_acc == avg_acc else "nan"
    avg_t_str = f"{avg_t:.3f}" if avg_t == avg_t else "nan"
    avg_tr_str = f"{avg_tr:.6f}" if avg_tr == avg_tr else "nan"
    with open(master_path, "a") as f:
        f.write(f"{model_tag}\t{total}\t{avg_acc_str}\t{total_t:.3f}\t{avg_t_str}\t{avg_tr_str}\n")


def load_tested(tested_log: Path) -> set[str]:
    if not tested_log.exists():
        return set()
    with open(tested_log, "r", encoding="utf-8") as f:
        return {str(Path(line.strip()).resolve()) for line in f if line.strip()}


def append_tested(tested_log: Path, ckpt_path: str) -> None:
    tested_log.parent.mkdir(parents=True, exist_ok=True)
    with open(tested_log, "a", encoding="utf-8") as f:
        f.write(str(Path(ckpt_path).resolve()) + "\n")


def loop_eval_new_ckpts(
    models_dir: Path,
    outdir_root: Path,
    poll_sec: float,
    stable_sec: float,
    idle_exit_sec: Optional[float],
    step_mod: int,
    reverse_ckpt_order: bool,
    tested_log: Path,
    evaluator_pool: PersistentEvaluatorPool,
    dirs: List[Path],
) -> Path:
    master_path = outdir_root / "all_models_summary.tsv"
    ensure_master_header(master_path)
    tested = load_tested(tested_log)
    last_sizes: dict[str, int] = {}
    last_new_ts = time.time()

    logging.info(
        "进入在线评测模式: models_dir=%s, poll_sec=%.1f, stable_sec=%.1f, step_mod=%d, reverse_ckpt_order=%s, idle_exit_sec=%s",
        str(models_dir),
        poll_sec,
        stable_sec,
        step_mod,
        str(reverse_ckpt_order),
        str(idle_exit_sec),
    )
    logging.info("已加载历史已评测ckpt数: %d", len(tested))

    while True:
        ckpts = discover_ckpts(models_dir, step_mod=step_mod, reverse_order=reverse_ckpt_order)
        candidates = []
        for p in ckpts:
            sp = str(p.resolve())
            if sp in tested:
                continue
            if is_file_stable(p, last_sizes, stable_sec=stable_sec):
                candidates.append(p)

        if candidates:
            logging.info("发现 %d 个新模型待评测: %s", len(candidates), " -> ".join([c.stem for c in candidates]))
            for p in candidates:
                sp = str(p.resolve())
                t0 = time.perf_counter()
                model_tag, total, avg_acc, total_t, avg_t, avg_tr = evaluate_model(
                    model_path=sp,
                    outdir_root=outdir_root,
                    evaluator_pool=evaluator_pool,
                    dirs=dirs,
                )
                append_master(master_path, model_tag, total, avg_acc, total_t, avg_t, avg_tr)
                tested.add(sp)
                append_tested(tested_log, sp)
                logging.info("[%s] Done in %.2fs", model_tag, time.perf_counter() - t0)
            last_new_ts = time.time()
        else:
            if idle_exit_sec is not None and (time.time() - last_new_ts) > idle_exit_sec:
                logging.info("超过 %.0fs 没有新checkpoint，退出。", idle_exit_sec)
                break
            time.sleep(poll_sec)

    return master_path


def evaluate_once(
    model_path: str,
    outdir_root: Path,
    evaluator_pool: PersistentEvaluatorPool,
    dirs: List[Path],
) -> Path:
    master_path = outdir_root / "all_models_summary.tsv"
    ensure_master_header(master_path)
    t0 = time.perf_counter()
    model_tag, total, avg_acc, total_t, avg_t, avg_tr = evaluate_model(
        model_path=model_path,
        outdir_root=outdir_root,
        evaluator_pool=evaluator_pool,
        dirs=dirs,
    )
    append_master(master_path, model_tag, total, avg_acc, total_t, avg_t, avg_tr)
    logging.info("[%s] Done in %.2fs", model_tag, time.perf_counter() - t0)
    return master_path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=None, help="单个模型路径（与 --models_dir 互斥）")
    ap.add_argument("--models_dir", type=str, default=None, help="checkpoint目录，开启在线轮询评测")
    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    ap.add_argument("--poll_sec", type=float, default=30.0, help="轮询间隔秒")
    ap.add_argument("--stable_sec", type=float, default=10.0, help="checkpoint最小稳定时长秒")
    ap.add_argument("--idle_exit_sec", type=float, default=None, help="超过该秒数无新ckpt则退出；默认一直运行")
    ap.add_argument("--step_mod", type=int, default=1, help="仅评测step%%step_mod==0的checkpoint；1表示不过滤")
    ap.add_argument(
        "--reverse_ckpt_order",
        type=parse_bool_flag,
        default=True,
        help="models_dir 模式下是否按 checkpoint 步数倒序评测（默认True）",
    )
    ap.add_argument(
        "--pseudo_repeat_times",
        type=int,
        default=2,
        help="伪标签迭代重复次数。总推理轮数=1+repeat_times，默认2（即总共3轮）。",
    )
    ap.add_argument("--clf_n_estimators", type=int, default=8, help="TabICLClassifier n_estimators")
    ap.add_argument(
        "--clf_batch_size",
        type=int,
        default=1,
        help="TabICLClassifier batch_size；设为-1表示None（单次处理所有ensemble）",
    )
    ap.add_argument("--clf_n_jobs", type=int, default=1, help="TabICLClassifier n_jobs（推荐1避免8进程CPU争抢）")
    ap.add_argument("--cpu_threads", type=int, default=1, help="每个GPU进程的CPU线程上限")
    ap.add_argument("--use_torch_compile", action="store_true", help="启用 torch.compile（默认关闭）")
    ap.add_argument(
        "--torch_compile_mode",
        type=str,
        default="reduce-overhead",
        help="torch.compile mode，设为none表示使用PyTorch默认",
    )
    ap.add_argument("--torch_compile_backend", type=str, default=None, help="torch.compile backend，默认PyTorch自动")
    ap.add_argument("--torch_compile_fullgraph", action="store_true", help="torch.compile fullgraph=True")
    ap.add_argument("--torch_compile_dynamic", action="store_true", help="torch.compile dynamic=True")
    ap.add_argument(
        "--torchinductor_cache_dir",
        type=str,
        default=None,
        help="TORCHINDUCTOR_CACHE_DIR，默认 <outdir>/_torchinductor_cache",
    )
    ap.add_argument(
        "--cache_root",
        type=str,
        default=None,
        help="预处理数据缓存目录，默认 <outdir>/_dataset_cache",
    )
    ap.add_argument(
        "--tested_log",
        type=str,
        default=None,
        help="已评测checkpoint记录文件，默认 <outdir>/tested_ckpts.txt",
    )
    return ap.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    data_root = Path(args.data_root)
    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)

    tested_log = Path(args.tested_log) if args.tested_log else (outdir_root / "tested_ckpts.txt")
    clf_batch_size = None if int(args.clf_batch_size) <= 0 else int(args.clf_batch_size)
    clf_n_jobs = None if int(args.clf_n_jobs) <= 0 else int(args.clf_n_jobs)
    cpu_threads = max(1, int(args.cpu_threads))
    cache_root = Path(args.cache_root) if args.cache_root else (outdir_root / "_dataset_cache")
    use_torch_compile = bool(args.use_torch_compile)
    torch_compile_mode = args.torch_compile_mode
    if torch_compile_mode is not None and torch_compile_mode.strip().lower() in {"", "none"}:
        torch_compile_mode = None
    torch_compile_backend = args.torch_compile_backend
    if torch_compile_backend is not None and torch_compile_backend.strip().lower() in {"", "none"}:
        torch_compile_backend = None
    torch_compile_fullgraph = bool(args.torch_compile_fullgraph)
    torch_compile_dynamic = True if args.torch_compile_dynamic else None
    pseudo_repeat_times = max(0, int(args.pseudo_repeat_times))
    reverse_ckpt_order = bool(args.reverse_ckpt_order)
    torchinductor_cache_dir = (
        str(Path(args.torchinductor_cache_dir))
        if args.torchinductor_cache_dir
        else str(outdir_root / "_torchinductor_cache")
    )

    dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
    summarize_task_types(dirs)

    gpu_devices = resolve_gpu_devices(FIXED_GPUS)
    if not gpu_devices:
        raise RuntimeError("未检测到可用GPU设备，无法执行并行评测。")
    logging.info("常驻评测池使用 %d 张GPU: %s", len(gpu_devices), ",".join(gpu_devices))
    if use_torch_compile:
        logging.info(
            "torch.compile 已开启: mode=%s, backend=%s, fullgraph=%s, dynamic=%s, cache=%s",
            str(torch_compile_mode),
            str(torch_compile_backend),
            str(torch_compile_fullgraph),
            str(torch_compile_dynamic),
            torchinductor_cache_dir,
        )
    logging.info("pseudo self-training: repeat_times=%d (总轮数=%d)", pseudo_repeat_times, 1 + pseudo_repeat_times)
    logging.info("models_dir checkpoint顺序: %s", "倒序" if reverse_ckpt_order else "正序")

    evaluator_pool = PersistentEvaluatorPool(
        gpu_devices=gpu_devices,
        cache_root=cache_root,
        clf_n_estimators=int(args.clf_n_estimators),
        clf_batch_size=clf_batch_size,
        clf_n_jobs=clf_n_jobs,
        cpu_threads=cpu_threads,
        use_torch_compile=use_torch_compile,
        torch_compile_mode=torch_compile_mode,
        torch_compile_backend=torch_compile_backend,
        torch_compile_fullgraph=torch_compile_fullgraph,
        torch_compile_dynamic=torch_compile_dynamic,
        torchinductor_cache_dir=torchinductor_cache_dir,
        pseudo_repeat_times=pseudo_repeat_times,
    )

    try:
        if args.models_dir:
            models_dir = Path(args.models_dir)
            if not models_dir.exists():
                raise FileNotFoundError(f"--models_dir 不存在: {models_dir}")
            master_path = loop_eval_new_ckpts(
                models_dir=models_dir,
                outdir_root=outdir_root,
                poll_sec=float(args.poll_sec),
                stable_sec=float(args.stable_sec),
                idle_exit_sec=None if args.idle_exit_sec is None else float(args.idle_exit_sec),
                step_mod=int(args.step_mod),
                reverse_ckpt_order=reverse_ckpt_order,
                tested_log=tested_log,
                evaluator_pool=evaluator_pool,
                dirs=dirs,
            )
            print("\n汇总总表：", master_path)
            return

        model_path = args.model_path or DEFAULT_MODEL_PATH
        master_path = evaluate_once(
            model_path=model_path,
            outdir_root=outdir_root,
            evaluator_pool=evaluator_pool,
            dirs=dirs,
        )
        print("\n汇总总表：", master_path)
    finally:
        evaluator_pool.close()


if __name__ == "__main__":
    main()

