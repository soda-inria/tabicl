#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

BASELINE_COMPARE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_COMPARE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benchmark as tabicl_benchmark

CLASSIFICATION_TASKS = {"binclass", "multiclass"}


@dataclass
class LoadedDataset:
    dataset_name: str
    dataset_dir: Path
    task_type: Optional[str]
    X_train: Any
    y_train: Any
    X_val: Any | None
    y_val: Any | None
    X_test: Any
    y_test: Any
    categorical_feature_indices: list[int]

    @property
    def n_val(self) -> int:
        return 0 if self.y_val is None else int(len(self.y_val))

    @property
    def X_train_merged(self):
        pd = tabicl_benchmark.pd
        if self.X_val is None:
            return self.X_train
        return pd.concat([self.X_train, self.X_val], axis=0, ignore_index=True)

    @property
    def y_train_merged(self):
        np = tabicl_benchmark.np
        if self.y_val is None:
            return self.y_train
        return np.concatenate([np.asarray(self.y_train), np.asarray(self.y_val)], axis=0)

    @property
    def n_train_report(self) -> int:
        return int(len(self.y_train_merged))

    @property
    def n_classes(self) -> int:
        np = tabicl_benchmark.np
        pd = tabicl_benchmark.pd
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


RESULT_COLUMNS = list(ResultRow.__annotations__.keys())


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_out_dir: str,
) -> argparse.ArgumentParser:
    parser.add_argument("--data-root", default="data178")
    parser.add_argument("--out-dir", default=default_out_dir)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--gpus",
        default="auto",
        help="Comma-separated physical GPU ids, or 'auto' to use detected GPUs.",
    )
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_gpu_id_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def resolve_workers_and_gpu_ids(args: argparse.Namespace) -> tuple[int, list[int]]:
    if args.gpus is None or str(args.gpus).strip().lower() == "auto":
        gpu_ids = detect_env_gpu_ids() or detect_nvidia_gpu_ids() or detect_torch_gpu_ids()
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


def categorical_indices_from_split(train_split) -> list[int]:
    np = tabicl_benchmark.np
    num_path, cat_path, _ = train_split
    n_num = 0
    if num_path is not None:
        arr = tabicl_benchmark.load_array(Path(num_path))
        n_num = int(arr.reshape(arr.shape[0], -1).shape[1]) if arr.ndim > 1 else 1
    if cat_path is None:
        return []
    arr = tabicl_benchmark.load_array(Path(cat_path))
    n_cat = int(arr.reshape(arr.shape[0], -1).shape[1]) if arr.ndim > 1 else 1
    return list(range(n_num, n_num + n_cat))


def load_classification_dataset(dataset_dir: Path) -> LoadedDataset:
    tabicl_benchmark.ensure_runtime_deps()
    info = tabicl_benchmark.load_dataset_info(dataset_dir)
    task_type = str(info.get("task_type", "")).lower() if info else None
    if task_type not in CLASSIFICATION_TASKS:
        raise SkipDataset(f"Skipped due to task_type={task_type!r}")

    train_split, val_split, test_split = tabicl_benchmark.find_split_files(dataset_dir)
    categorical_feature_indices = categorical_indices_from_split(train_split)

    X_train, y_train = tabicl_benchmark.load_split(
        train_split[0],
        train_split[1],
        train_split[2],
        context=f"{dataset_dir.name}-train",
    )

    X_val = None
    y_val = None
    if val_split is not None:
        X_val, y_val = tabicl_benchmark.load_split(
            val_split[0],
            val_split[1],
            val_split[2],
            context=f"{dataset_dir.name}-val",
        )

    X_test, y_test = tabicl_benchmark.load_split(
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


class SkipDataset(Exception):
    pass


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


def evaluate_one_dataset(adapter: Any, dataset_dir: Path) -> ResultRow:
    np = tabicl_benchmark.np
    task_type: Optional[str] = None
    try:
        loaded = load_classification_dataset(dataset_dir)
        task_type = loaded.task_type
        result = adapter.fit_predict(loaded)
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


def rows_to_frame(rows: Iterable[ResultRow]):
    pd = tabicl_benchmark.pd
    frame = pd.DataFrame([asdict(row) for row in rows])
    if frame.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    for column in RESULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[RESULT_COLUMNS]


def write_summary(summary_path: Path, result_df: Any, dataset_dirs: list[Path], wall_seconds: float) -> None:
    pd = tabicl_benchmark.pd
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
        "failed_datasets: "
        + (", ".join(failed_df["dataset_name"].astype(str).tolist()) if len(failed_df) else "(none)"),
        "skipped_datasets: "
        + (", ".join(skipped_df["dataset_name"].astype(str).tolist()) if len(skipped_df) else "(none)"),
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bind_worker_gpu(gpu_id: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["ROCR_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def load_adapter_class(script_path: str, class_name: str):
    path = Path(script_path).resolve()
    module_name = f"_tfm_benchmark_{path.parent.name}_{path.stem}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load adapter script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def worker_main(
    adapter_script: str,
    adapter_class_name: str,
    worker_id: int,
    gpu_id: int,
    dataset_dirs: list[str],
    worker_csv: str,
    args_dict: dict[str, Any],
) -> None:
    tabicl_benchmark.ensure_runtime_deps()
    bind_worker_gpu(gpu_id)
    args = argparse.Namespace(**args_dict)
    rows: list[ResultRow] = []
    try:
        adapter_cls = load_adapter_class(adapter_script, adapter_class_name)
        adapter = adapter_cls(args, device="cuda:0")
        for dataset_dir in dataset_dirs:
            row = evaluate_one_dataset(adapter, Path(dataset_dir))
            rows.append(row)
            if args.verbose:
                if row.status == "ok":
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] [ok] "
                        f"{row.dataset_name} accuracy={row.accuracy:.6f}",
                        flush=True,
                    )
                else:
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] [{row.status}] "
                        f"{row.dataset_name} error={row.error}",
                        flush=True,
                    )
    except Exception:
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


def run_benchmark(
    args: argparse.Namespace,
    *,
    adapter_script: str,
    adapter_class_name: str,
) -> None:
    tabicl_benchmark.ensure_runtime_deps()
    data_root = resolve_repo_path(args.data_root)
    out_dir = resolve_repo_path(args.out_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = tabicl_benchmark.find_dataset_dirs(data_root)
    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: args.max_datasets]
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset directories found under {data_root}")

    args.data_root = str(data_root)
    args.out_dir = str(out_dir)
    args.workers, gpu_ids = resolve_workers_and_gpu_ids(args)
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
            args=(
                str(Path(adapter_script).resolve()),
                adapter_class_name,
                worker_id,
                gpu_ids[worker_id],
                assigned,
                str(worker_csv),
                args_dict,
            ),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    pd = tabicl_benchmark.pd
    frames = [pd.read_csv(path) for path in worker_csvs if path.exists()]
    result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RESULT_COLUMNS)
    result_df = result_df[RESULT_COLUMNS]
    all_csv = out_dir / "all_classification_results.csv"
    summary_txt = out_dir / "summary.txt"
    result_df.to_csv(all_csv, index=False)
    write_summary(summary_txt, result_df, dataset_dirs, time.time() - started)
    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {summary_txt}")
