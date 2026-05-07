#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data178"
CLASSIFICATION_TASKS = {"binclass", "multiclass"}
CATEGORICAL_MISSING_TOKEN = "__pipeline_compare_missing__"

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
    fit_seconds: float
    predict_seconds: float
    status: str
    error: Optional[str]


@dataclass
class DatasetBundle:
    dataset_name: str
    dataset_dir: Path
    task_type: str
    X_train: Any
    y_train: Any
    X_val: Any | None
    y_val: Any | None
    X_test: Any
    y_test: Any
    categorical_feature_indices: list[int]
    original_train_count: int
    val_count: int


class BenchmarkPredictor(Protocol):
    def fit_predict(self, bundle: DatasetBundle) -> Any:
        ...


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


def load_dataset_info(dataset_dir: Path) -> dict | None:
    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        return None
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_dataset_dirs(data_root: Path) -> list[Path]:
    return [path for path in sorted(data_root.iterdir()) if path.is_dir()]


def stable_feature_prefix(context: str, fallback: str) -> str:
    stem = Path(context or fallback).stem
    for suffix in ("-train", "-test", "-val", "-single"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or fallback


def normalize_categorical_series(series) -> Any:
    ensure_runtime_deps()
    string_series = series.astype("string")
    string_series = string_series.fillna(CATEGORICAL_MISSING_TOKEN)
    return string_series.astype(str)


def make_feature_frame(values: Any, *, kind: str, prefix: str) -> Any:
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


def make_target_array(values: Any) -> Any:
    ensure_runtime_deps()
    y = np.asarray(values)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    elif y.ndim > 1 and y.shape[0] == 1:
        y = y.squeeze(0)
    return pd.Series(y).values


def load_array(file_path: Path) -> Any:
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
    val_split = (
        (n_val, c_val, y_val)
        if y_val is not None and (n_val is not None or c_val is not None)
        else None
    )
    test_split = (n_test, c_test, y_test)
    return train_split, val_split, test_split


def load_split(
    num_path: Path | None,
    cat_path: Path | None,
    y_path: Path,
    *,
    context: str,
) -> tuple[Any, Any, list[int]]:
    ensure_runtime_deps()
    features: list[Any] = []
    cat_indices: list[int] = []
    feature_prefix = stable_feature_prefix(context, y_path.stem)

    if num_path is not None:
        x_num = load_array(num_path)
        features.append(make_feature_frame(x_num, kind="numeric", prefix=f"{feature_prefix}_n"))
    if cat_path is not None:
        x_cat = load_array(cat_path)
        cat_df = make_feature_frame(x_cat, kind="categorical", prefix=f"{feature_prefix}_c")
        offset = sum(feature_df.shape[1] for feature_df in features)
        cat_indices.extend(range(offset, offset + cat_df.shape[1]))
        features.append(cat_df)

    if not features:
        raise ValueError("No feature files found for split")

    n_samples = features[0].shape[0]
    for idx, feature_df in enumerate(features):
        if feature_df.shape[0] != n_samples:
            raise ValueError(
                f"Inconsistent rows across feature blocks: block {idx} has "
                f"{feature_df.shape[0]} rows but expected {n_samples}"
            )

    X = features[0] if len(features) == 1 else pd.concat(features, axis=1)
    y = make_target_array(load_array(y_path))
    if len(X) != len(y):
        raise ValueError(f"Feature/target row mismatch: X has {len(X)} rows while y has {len(y)}")
    return X, y, cat_indices


def load_dataset_bundle(dataset_dir: Path, *, merge_val_into_train: bool) -> DatasetBundle | ResultRow:
    ensure_runtime_deps()
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
            fit_seconds=0.0,
            predict_seconds=0.0,
            status="skip",
            error=f"Skipped due to task_type={task_type!r}",
        )

    train_split, val_split, test_split = find_split_files(dataset_dir)
    X_train, y_train, cat_indices = load_split(
        train_split[0],
        train_split[1],
        train_split[2],
        context=f"{dataset_dir.name}-train",
    )
    original_train_count = int(len(y_train))

    X_val = None
    y_val = None
    val_count = 0
    if val_split is not None:
        X_val, y_val, val_cat_indices = load_split(
            val_split[0],
            val_split[1],
            val_split[2],
            context=f"{dataset_dir.name}-val",
        )
        if val_cat_indices != cat_indices:
            raise ValueError("Categorical feature layout differs between train and val")
        val_count = int(len(y_val))
        if merge_val_into_train:
            X_train = pd.concat([X_train, X_val], axis=0, ignore_index=True)
            y_train = np.concatenate([np.asarray(y_train), np.asarray(y_val)], axis=0)

    X_test, y_test, test_cat_indices = load_split(
        test_split[0],
        test_split[1],
        test_split[2],
        context=f"{dataset_dir.name}-test",
    )
    if test_cat_indices != cat_indices:
        raise ValueError("Categorical feature layout differs between train and test")

    return DatasetBundle(
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir,
        task_type=task_type,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        categorical_feature_indices=cat_indices,
        original_train_count=original_train_count,
        val_count=val_count,
    )


def evaluate_one_dataset(
    predictor: BenchmarkPredictor,
    dataset_dir: Path,
    *,
    merge_val_into_train: bool,
) -> ResultRow:
    ensure_runtime_deps()
    task_type: Optional[str] = None
    try:
        bundle_or_row = load_dataset_bundle(dataset_dir, merge_val_into_train=merge_val_into_train)
        if isinstance(bundle_or_row, ResultRow):
            return bundle_or_row
        bundle = bundle_or_row
        task_type = bundle.task_type
        classes = pd.unique(
            pd.Series(np.concatenate([np.asarray(bundle.y_train), np.asarray(bundle.y_test)], axis=0))
        )

        t0 = time.time()
        prediction_result = predictor.fit_predict(bundle)
        elapsed = time.time() - t0
        if (
            isinstance(prediction_result, tuple)
            and len(prediction_result) == 3
        ):
            y_pred, fit_seconds, predict_seconds = prediction_result
        else:
            y_pred = prediction_result
            fit_seconds = elapsed
            predict_seconds = 0.0

        accuracy = float(np.mean(np.asarray(y_pred) == np.asarray(bundle.y_test)))
        return ResultRow(
            dataset_name=dataset_dir.name,
            dataset_dir=dataset_dir.as_posix(),
            task_type=task_type,
            n_train=int(len(bundle.y_train)),
            n_val=int(bundle.val_count),
            n_test=int(len(bundle.y_test)),
            n_features=int(bundle.X_train.shape[1]),
            n_classes=int(len(classes)),
            accuracy=accuracy,
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            status="ok",
            error=None,
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
            fit_seconds=0.0,
            predict_seconds=0.0,
            status="fail",
            error=f"{type(exc).__name__}: {exc}",
        )


def write_summary(summary_path: Path, result_df: Any, dataset_dirs: list[Path], wall_seconds: float) -> None:
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
        f"avg_accuracy_ok: {ok_df['accuracy'].mean():.6f}" if len(ok_df) else "avg_accuracy_ok: (none)",
        f"wall_seconds: {wall_seconds:.3f}",
    ]
    lines.append(
        "failed_datasets: " + ", ".join(failed_df["dataset_name"].astype(str).tolist())
        if len(failed_df)
        else "failed_datasets: (none)"
    )
    lines.append(
        "skipped_datasets: " + ", ".join(skipped_df["dataset_name"].astype(str).tolist())
        if len(skipped_df)
        else "skipped_datasets: (none)"
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bind_worker_to_gpu(gpu_id: int) -> None:
    os.environ["ROCR_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def force_memory_cleanup() -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            ipc_collect()


def worker_main(
    worker_id: int,
    gpu_id: int,
    assigned_dataset_dirs: list[str],
    worker_out_csv: str,
    args_dict: dict[str, Any],
    model_factory: Callable[[dict[str, Any], int, int], BenchmarkPredictor],
    merge_val_into_train: bool,
    verbose: bool,
) -> None:
    try:
        ensure_runtime_deps()
        bind_worker_to_gpu(gpu_id)
        predictor = model_factory(args_dict, worker_id, gpu_id)
        rows: list[ResultRow] = []
        for dataset_dir in assigned_dataset_dirs:
            row = evaluate_one_dataset(
                predictor,
                Path(dataset_dir),
                merge_val_into_train=merge_val_into_train,
            )
            rows.append(row)
            if verbose:
                prefix = f"[worker {worker_id} | gpu {gpu_id}]"
                if row.status == "ok":
                    print(f"{prefix} [ok] {row.dataset_name} accuracy={row.accuracy:.6f}", flush=True)
                elif row.status == "skip":
                    print(f"{prefix} [skip] {row.dataset_name} reason={row.error}", flush=True)
                else:
                    print(f"{prefix} [fail] {row.dataset_name} error={row.error}", flush=True)
            force_memory_cleanup()

        pd.DataFrame([asdict(row) for row in rows]).to_csv(worker_out_csv, index=False)
    except Exception:
        ensure_runtime_deps()
        crash_row = pd.DataFrame(
            [
                {
                    "dataset_name": f"__WORKER_CRASH__{worker_id}",
                    "dataset_dir": "__worker__",
                    "task_type": None,
                    "n_train": 0,
                    "n_val": 0,
                    "n_test": 0,
                    "n_features": 0,
                    "n_classes": 0,
                    "accuracy": None,
                    "fit_seconds": 0.0,
                    "predict_seconds": 0.0,
                    "status": "fail",
                    "error": traceback.format_exc(),
                }
            ]
        )
        crash_row.to_csv(worker_out_csv, index=False)


def add_common_args(parser: argparse.ArgumentParser, *, default_out_dir: str) -> None:
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default=default_out_dir)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")


def resolve_gpu_ids(args: argparse.Namespace) -> list[int]:
    gpu_ids = [int(x.strip()) for x in str(args.gpus).split(",") if x.strip()]
    if len(gpu_ids) != int(args.workers):
        raise ValueError(f"--gpus must contain exactly {args.workers} ids")
    return gpu_ids


def run_benchmark(
    args: argparse.Namespace,
    *,
    model_factory: Callable[[dict[str, Any], int, int], BenchmarkPredictor],
    merge_val_into_train: bool,
) -> None:
    ensure_runtime_deps()
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

    gpu_ids = resolve_gpu_ids(args)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    start_time = time.time()
    worker_csv_paths: list[Path] = []
    processes: list[mp.Process] = []
    args_dict = vars(args).copy()
    for worker_id in range(args.workers):
        assigned_dirs = [str(path.resolve()) for path in dataset_dirs[worker_id:: args.workers]]
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        worker_csv_paths.append(worker_csv)
        proc = mp.Process(
            target=worker_main,
            args=(
                worker_id,
                gpu_ids[worker_id],
                assigned_dirs,
                str(worker_csv),
                args_dict,
                model_factory,
                merge_val_into_train,
                args.verbose,
            ),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    dfs = [pd.read_csv(path) for path in worker_csv_paths if path.exists()]
    all_df = (
        pd.concat(dfs, ignore_index=True)
        if dfs
        else pd.DataFrame(columns=ResultRow.__annotations__.keys())
    )
    all_csv = out_dir / "all_classification_results.csv"
    summary_txt = out_dir / "summary.txt"
    all_df.to_csv(all_csv, index=False)
    write_summary(summary_txt, all_df, dataset_dirs, time.time() - start_time)

    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {summary_txt}")


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "dataset"
