#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA_DIRS = [
    Path("dataset/ctr23"),
    Path("dataset/tabarena/reg"),
    Path("dataset/talent_reg"),
]

TARGET_CANDIDATES = ["target", "label", "y", "TARGET", "Label", "Y"]

np = None
pd = None
mean_absolute_error = None
mean_squared_error = None
r2_score = None
train_test_split = None


@dataclass
class ResultRow:
    dataset_group: str
    dataset_dir: str
    dataset_name: str
    n_train: int
    n_test: int
    n_features: int
    r2: Optional[float]
    rmse: Optional[float]
    mae: Optional[float]
    fit_seconds: float
    predict_seconds: float
    status: str
    error: Optional[str]


def ensure_runtime_deps() -> None:
    global np
    global pd
    global mean_absolute_error
    global mean_squared_error
    global r2_score
    global train_test_split

    if np is None or pd is None:
        import numpy as _np
        import pandas as _pd
        from sklearn.metrics import (
            mean_absolute_error as _mean_absolute_error,
            mean_squared_error as _mean_squared_error,
            r2_score as _r2_score,
        )
        from sklearn.model_selection import train_test_split as _train_test_split

        np = _np
        pd = _pd
        mean_absolute_error = _mean_absolute_error
        mean_squared_error = _mean_squared_error
        r2_score = _r2_score
        train_test_split = _train_test_split


def infer_target_column(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    return df.columns[-1]


def dataset_group_name(csv_path: Path) -> str:
    path_str = csv_path.as_posix()
    if "dataset/tabarena/reg" in path_str:
        return "tabarena_reg"
    if "dataset/talent_reg" in path_str:
        return "talent_reg"
    if "dataset/ctr23" in path_str:
        return "ctr23"
    return csv_path.parent.name


def find_csv_files(data_dirs: List[Path]) -> List[Path]:
    csv_files: List[Path] = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        csv_files.extend(sorted(data_dir.glob("*.csv")))
    return csv_files


def normalize_local_ckpt_path(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    try:
        path = path.resolve()
    except Exception:
        pass
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


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


def evaluate_one_dataset(
    regressor,
    csv_path: Path,
    *,
    test_size: float,
    random_state: int,
) -> ResultRow:
    ensure_runtime_deps()
    try:
        df = pd.read_csv(csv_path)
        target_col = infer_target_column(df)
        df = df.dropna(subset=[target_col])

        X = df.drop(columns=[target_col])
        y = pd.to_numeric(df[target_col], errors="coerce")
        valid_mask = y.notna()
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]

        if len(X) < 2:
            raise ValueError(
                "Not enough valid rows after converting target column to numeric."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        t0 = time.time()
        regressor.fit(X_train, y_train)
        fit_seconds = time.time() - t0

        t1 = time.time()
        y_pred = regressor.predict(X_test)
        predict_seconds = time.time() - t1

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))

        return ResultRow(
            dataset_group=dataset_group_name(csv_path),
            dataset_dir=csv_path.parent.as_posix(),
            dataset_name=csv_path.name,
            n_train=int(len(X_train)),
            n_test=int(len(X_test)),
            n_features=int(X_train.shape[1]),
            r2=r2,
            rmse=rmse,
            mae=mae,
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            status="ok",
            error=None,
        )
    except Exception as exc:
        return ResultRow(
            dataset_group=dataset_group_name(csv_path),
            dataset_dir=csv_path.parent.as_posix(),
            dataset_name=csv_path.name,
            n_train=0,
            n_test=0,
            n_features=0,
            r2=None,
            rmse=None,
            mae=None,
            fit_seconds=0.0,
            predict_seconds=0.0,
            status="fail",
            error=f"{type(exc).__name__}: {exc}",
        )


def worker_main(
    worker_id: int,
    gpu_id: int,
    assigned_csv_paths: List[str],
    ready_queue,
    start_event,
    worker_out_csv: str,
    model_kwargs: Dict,
    test_size: float,
    random_state: int,
    verbose: bool,
) -> None:
    try:
        ensure_runtime_deps()
        gpu_id_str = str(gpu_id)
        os.environ["ROCR_VISIBLE_DEVICES"] = gpu_id_str
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        import torch
        from tabpfn import TabPFNRegressor

        torch_diag = collect_torch_diagnostics()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU backend is not available in this worker. "
                f"Diagnostics: {json.dumps(torch_diag, ensure_ascii=False)}"
            )

        worker_kwargs = dict(model_kwargs)
        worker_kwargs["device"] = "cuda:0"
        regressor = TabPFNRegressor(**worker_kwargs)
        ready_queue.put(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "status": "ready",
                "assigned_count": len(assigned_csv_paths),
            }
        )
        start_event.wait()

        rows: List[ResultRow] = []
        for item in assigned_csv_paths:
            csv_path = Path(item)
            row = evaluate_one_dataset(
                regressor,
                csv_path,
                test_size=test_size,
                random_state=random_state,
            )
            rows.append(row)

            if verbose:
                if row.status == "ok":
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] "
                        f"[ok] {row.dataset_name} r2={row.r2:.6f} rmse={row.rmse:.6f}"
                    )
                else:
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] "
                        f"[fail] {row.dataset_name} error={row.error}"
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
        crash_row = pd.DataFrame(
            [
                {
                    "dataset_group": "__worker__",
                    "dataset_dir": "__worker__",
                    "dataset_name": f"__WORKER_CRASH__{worker_id}",
                    "n_train": 0,
                    "n_test": 0,
                    "n_features": 0,
                    "r2": None,
                    "rmse": None,
                    "mae": None,
                    "fit_seconds": 0.0,
                    "predict_seconds": 0.0,
                    "status": "fail",
                    "error": traceback.format_exc(),
                }
            ]
        )
        crash_row.to_csv(worker_out_csv, index=False)


def write_summary(
    summary_path: Path,
    result_df: pd.DataFrame,
    csv_files: List[Path],
    wall_seconds: float,
) -> None:
    ensure_runtime_deps()
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()

    lines = [
        f"discovered_datasets: {len(csv_files)}",
        f"processed_datasets: {len(result_df)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        f"avg_r2_ok: {ok_df['r2'].mean():.6f}" if len(ok_df) else "avg_r2_ok: (none)",
        f"avg_rmse_ok: {ok_df['rmse'].mean():.6f}" if len(ok_df) else "avg_rmse_ok: (none)",
        f"avg_mae_ok: {ok_df['mae'].mean():.6f}" if len(ok_df) else "avg_mae_ok: (none)",
        f"wall_seconds: {wall_seconds:.3f}",
    ]

    if len(failed_df):
        failed_names = ", ".join(failed_df["dataset_name"].astype(str).tolist())
        lines.append(f"failed_datasets: {failed_names}")
    else:
        lines.append("failed_datasets: (none)")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_group_outputs(
    out_dir: Path,
    all_df: pd.DataFrame,
    csv_files: List[Path],
    wall_seconds: float,
) -> None:
    ensure_runtime_deps()
    grouped_csv_files: Dict[str, List[Path]] = {}
    for csv_path in csv_files:
        grouped_csv_files.setdefault(dataset_group_name(csv_path), []).append(csv_path)

    for group_name, group_files in grouped_csv_files.items():
        group_dir = out_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        if len(all_df):
            group_df = all_df[all_df["dataset_group"] == group_name].copy()
        else:
            group_df = pd.DataFrame(columns=ResultRow.__annotations__.keys())

        group_csv = group_dir / "all_regression_results.csv"
        group_summary = group_dir / "summary.txt"
        group_df.to_csv(group_csv, index=False)
        write_summary(group_summary, group_df, group_files, wall_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official TabPFN 2.5 regression checkpoint on the 3 regression "
            "benchmark folders with AMD/ROCm multi-GPU."
        )
    )
    parser.add_argument(
        "--model-path",
        default="ckpt/TabPFN-2.5/tabpfn-v2.5-regressor-v2.5_default.ckpt",
    )
    parser.add_argument("--out-dir", default="result/TabPFN_2_5_official_regression_8gpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument(
        "--fit-mode",
        default="fit_preprocessors",
        choices=["low_memory", "fit_preprocessors", "fit_with_cache", "batched"],
    )
    parser.add_argument(
        "--inference-precision",
        default="auto",
        choices=["auto", "autocast"],
    )
    parser.add_argument("--memory-saving-mode", default="auto")
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--ignore-pretraining-limits",
        dest="ignore_pretraining_limits",
        action="store_true",
        help="Ignore TabPFN's pretraining sample/feature limits. Enabled by default.",
    )
    limit_group.add_argument(
        "--enforce-pretraining-limits",
        dest="ignore_pretraining_limits",
        action="store_false",
        help="Re-enable TabPFN's pretraining sample/feature limit checks.",
    )
    parser.set_defaults(ignore_pretraining_limits=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ensure_runtime_deps()
    model_path = normalize_local_ckpt_path(args.model_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = find_csv_files(DATA_DIRS)
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found under dataset/ctr23, dataset/tabarena/reg, "
            "dataset/talent_reg"
        )

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if len(gpu_ids) != args.workers:
        raise ValueError(f"--gpus must contain exactly {args.workers} ids")

    model_kwargs: Dict[str, object] = {
        "model_path": str(model_path),
        "n_estimators": args.n_estimators,
        "fit_mode": args.fit_mode,
        "inference_precision": args.inference_precision,
        "memory_saving_mode": args.memory_saving_mode,
        "n_preprocessing_jobs": 1,
        "random_state": args.random_state,
    }
    model_kwargs["ignore_pretraining_limits"] = args.ignore_pretraining_limits

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    start_time = time.time()
    ready_queue: mp.Queue = mp.Queue()
    start_event = mp.Event()

    worker_csv_paths: List[Path] = []
    processes: List[mp.Process] = []
    for worker_id in range(args.workers):
        assigned_csv_paths = [str(path) for path in csv_files[worker_id::args.workers]]
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        worker_csv_paths.append(worker_csv)
        proc = mp.Process(
            target=worker_main,
            args=(
                worker_id,
                gpu_ids[worker_id],
                assigned_csv_paths,
                ready_queue,
                start_event,
                str(worker_csv),
                dict(model_kwargs),
                args.test_size,
                args.random_state,
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
    all_csv = out_dir / "all_regression_results.csv"
    summary_txt = out_dir / "summary.txt"
    all_df.to_csv(all_csv, index=False)

    wall_seconds = time.time() - start_time
    write_summary(summary_txt, all_df, csv_files, wall_seconds)
    write_group_outputs(out_dir, all_df, csv_files, wall_seconds)

    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {summary_txt}")
    print("saved_group_summaries:")
    for group_name in sorted({dataset_group_name(csv_path) for csv_path in csv_files}):
        print(f"  {group_name}: {out_dir / group_name / 'summary.txt'}")
    print("model_kwargs:")
    print(json.dumps(model_kwargs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
