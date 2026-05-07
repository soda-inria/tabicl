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
from typing import Any, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import TTT_1B_ensmble as ens


DEFAULT_DATA_ROOT = Path("data178")
DEFAULT_CKPT_ROOT = Path("1b_result/ensmble_ttt_step15_lr5e-6/ttt_ckpts")
DEFAULT_OUT_DIR = Path("1b_result/ensmble_ttt_step15_lr5e-6/ckpt_test")
DEFAULT_CHECKPOINT_VERSION = ens.DEFAULT_CHECKPOINT_VERSION


@dataclass
class EnsembleTestResultRow:
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
    ckpt_path: Optional[str] = None
    ckpt_step: int = 20
    ckpt_layout: Optional[str] = None


def format_dataset_result_log(worker_label: str, row: EnsembleTestResultRow) -> str:
    prefix = f"[{worker_label}]"
    if row.status == "ok":
        return (
            f"{prefix} [ok] {row.dataset_name} "
            f"accuracy={ens.format_optional_float(row.accuracy)} "
            f"fit={row.fit_seconds:.3f}s "
            f"predict={row.predict_seconds:.3f}s "
            f"ckpt_step={row.ckpt_step} "
            f"ckpt_layout={row.ckpt_layout}"
        )
    if row.status == "skip":
        return f"{prefix} [skip] {row.dataset_name} reason={row.error}"
    return f"{prefix} [fail] {row.dataset_name} error={row.error}"


def _step_filename(step: int) -> str:
    return f"step_{int(step)}.ckpt"


def resolve_dataset_ckpt(
    ckpt_root: Path,
    dataset_name: str,
    step: int,
    ckpt_model_name: str | None = None,
) -> tuple[Path, str]:
    ckpt_root = ckpt_root.expanduser()
    sanitized_dataset = ens._sanitize_path_component(dataset_name)
    filename = _step_filename(step)

    direct = ckpt_root / sanitized_dataset / filename
    if direct.exists():
        return direct.resolve(), "dataset"

    if ckpt_model_name:
        nested = (
            ckpt_root
            / ens._sanitize_path_component(ckpt_model_name)
            / sanitized_dataset
            / filename
        )
        if nested.exists():
            return nested.resolve(), "model_dataset"
        raise FileNotFoundError(
            f"Missing checkpoint for dataset={dataset_name!r} step={step}: "
            f"checked {direct} and {nested}"
        )

    candidates = sorted(
        path
        for path in ckpt_root.glob(f"*/{sanitized_dataset}/{filename}")
        if path.is_file()
    )
    if len(candidates) == 1:
        return candidates[0].resolve(), "model_dataset"
    if len(candidates) > 1:
        candidate_text = ", ".join(str(path) for path in candidates[:10])
        raise RuntimeError(
            f"Multiple checkpoints found for dataset={dataset_name!r} step={step}. "
            f"Pass --ckpt-model-name to disambiguate. Candidates: {candidate_text}"
        )

    raise FileNotFoundError(
        f"Missing checkpoint for dataset={dataset_name!r} step={step}: "
        f"checked {direct} and {ckpt_root}/*/{sanitized_dataset}/{filename}"
    )


def count_available_ckpts(
    dataset_dirs: List[Path],
    ckpt_root: Path,
    ckpt_step: int,
    ckpt_model_name: str | None,
) -> dict[str, Any]:
    found = 0
    missing: list[str] = []
    ambiguous: list[str] = []
    for dataset_dir in dataset_dirs:
        try:
            resolve_dataset_ckpt(ckpt_root, dataset_dir.name, ckpt_step, ckpt_model_name)
            found += 1
        except RuntimeError:
            ambiguous.append(dataset_dir.name)
        except Exception:
            missing.append(dataset_dir.name)
    return {
        "datasets_discovered": len(dataset_dirs),
        "ckpt_step": int(ckpt_step),
        "step_ckpts_found": found,
        "missing_ckpts": len(missing),
        "ambiguous_ckpts": len(ambiguous),
        "missing_ckpt_datasets": missing,
        "ambiguous_ckpt_datasets": ambiguous,
    }


def build_model_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "n_estimators": args.n_estimators,
        "batch_size": args.batch_size,
        "kv_cache": args.kv_cache,
        "allow_auto_download": False,
        "checkpoint_version": args.checkpoint_version,
        "use_amp": args.use_amp,
        "use_fa3": args.use_fa3,
        "offload_mode": args.offload_mode,
        "random_state": args.random_state,
    }


def _empty_row(
    dataset_dir: Path,
    *,
    task_type: Optional[str],
    status: str,
    error: str,
    ckpt_step: int,
    ckpt_path: str | None = None,
    ckpt_layout: str | None = None,
) -> EnsembleTestResultRow:
    return EnsembleTestResultRow(
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
        ckpt_path=ckpt_path,
        ckpt_step=int(ckpt_step),
        ckpt_layout=ckpt_layout,
    )


def evaluate_one_dataset(
    dataset_dir: Path,
    *,
    ckpt_root: Path,
    ckpt_step: int,
    ckpt_model_name: str | None,
    model_kwargs: dict[str, object],
    device_str: str,
) -> EnsembleTestResultRow:
    ens.ensure_runtime_deps()
    task_type: Optional[str] = None
    classifier = None
    ckpt_path: Path | None = None
    ckpt_layout: str | None = None
    n_train = 0
    n_val = 0
    n_test = 0
    n_features = 0
    n_classes = 0

    try:
        info = ens.load_dataset_info(dataset_dir)
        task_type = str(info.get("task_type", "")).lower() if info else None
        if task_type not in ens.CLASSIFICATION_TASKS:
            return _empty_row(
                dataset_dir,
                task_type=task_type,
                status="skip",
                error=f"Skipped due to task_type={task_type!r}",
                ckpt_step=ckpt_step,
            )

        train_split, val_split, test_split = ens.find_split_files(dataset_dir)

        X_train, y_train = ens.load_split(
            train_split[0],
            train_split[1],
            train_split[2],
            context=f"{dataset_dir.name}-train",
        )
        val_count = 0
        if val_split is not None:
            X_val, y_val = ens.load_split(
                val_split[0],
                val_split[1],
                val_split[2],
                context=f"{dataset_dir.name}-val",
            )
            val_count = int(len(y_val))
            X_train = ens.pd.concat([X_train, X_val], axis=0, ignore_index=True)
            y_train = ens.np.concatenate([ens.np.asarray(y_train), ens.np.asarray(y_val)], axis=0)

        X_test, y_test = ens.load_split(
            test_split[0],
            test_split[1],
            test_split[2],
            context=f"{dataset_dir.name}-test",
        )
        classes = ens.pd.unique(
            ens.pd.Series(
                ens.np.concatenate([ens.np.asarray(y_train), ens.np.asarray(y_test)], axis=0)
            )
        )
        n_train = int(len(y_train))
        n_val = val_count
        n_test = int(len(y_test))
        n_features = int(X_train.shape[1])
        n_classes = int(len(classes))

        ckpt_path, ckpt_layout = resolve_dataset_ckpt(
            ckpt_root,
            dataset_dir.name,
            ckpt_step,
            ckpt_model_name,
        )

        from tabicl import TabICLClassifier

        worker_kwargs = dict(model_kwargs)
        worker_kwargs["device"] = device_str
        worker_kwargs["model_path"] = str(ckpt_path)
        classifier = TabICLClassifier(**worker_kwargs)

        t0 = time.time()
        classifier.fit(X_train, y_train)
        fit_seconds = time.time() - t0

        t1 = time.time()
        y_pred = classifier.predict(X_test)
        predict_seconds = time.time() - t1

        accuracy = float(ens.np.mean(ens.np.asarray(y_pred) == ens.np.asarray(y_test)))

        return EnsembleTestResultRow(
            dataset_name=dataset_dir.name,
            dataset_dir=dataset_dir.as_posix(),
            task_type=task_type,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            n_features=n_features,
            n_classes=n_classes,
            accuracy=accuracy,
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            status="ok",
            error=None,
            n_train_a=n_train,
            n_test_d=n_test,
            ttt_split_reason="Loaded per-dataset TTT checkpoint; no TTT update run.",
            ckpt_path=ckpt_path.as_posix(),
            ckpt_step=int(ckpt_step),
            ckpt_layout=ckpt_layout,
        )
    except Exception as exc:
        return EnsembleTestResultRow(
            dataset_name=dataset_dir.name,
            dataset_dir=dataset_dir.as_posix(),
            task_type=task_type,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            n_features=n_features,
            n_classes=n_classes,
            accuracy=None,
            fit_seconds=0.0,
            predict_seconds=0.0,
            status="fail",
            error=f"{type(exc).__name__}: {exc}",
            n_train_a=n_train,
            n_test_d=n_test,
            ckpt_path=ckpt_path.as_posix() if ckpt_path is not None else None,
            ckpt_step=int(ckpt_step),
            ckpt_layout=ckpt_layout,
        )
    finally:
        ens.release_classifier_resources(classifier)
        ens.force_memory_cleanup(device_str)


def worker_main(
    worker_id: int,
    gpu_id: int,
    gpu_group: str,
    assigned_dataset_dirs: List[str],
    ready_queue,
    start_event,
    worker_out_csv: str,
    model_kwargs: dict[str, object],
    ckpt_root: str,
    ckpt_step: int,
    ckpt_model_name: str | None,
) -> None:
    try:
        ens.ensure_runtime_deps()
        device_str = ens.apply_worker_environment_updates(gpu_group)
        worker_label = f"worker {worker_id} | gpu {gpu_group}"

        import torch

        torch_diag = ens.collect_torch_diagnostics()
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
                "assigned_count": len(assigned_dataset_dirs),
            }
        )
        start_event.wait()

        rows: list[EnsembleTestResultRow] = []
        for dataset_dir in assigned_dataset_dirs:
            row = evaluate_one_dataset(
                Path(dataset_dir),
                ckpt_root=Path(ckpt_root),
                ckpt_step=ckpt_step,
                ckpt_model_name=ckpt_model_name,
                model_kwargs=model_kwargs,
                device_str=device_str,
            )
            rows.append(row)
            print(format_dataset_result_log(worker_label, row), flush=True)

        worker_df = (
            ens.pd.DataFrame([asdict(row) for row in rows])
            if rows
            else ens.pd.DataFrame(columns=EnsembleTestResultRow.__annotations__.keys())
        )
        worker_df.to_csv(worker_out_csv, index=False)
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

        ens.ensure_runtime_deps()
        crash_row = ens.pd.DataFrame(
            [
                asdict(
                    EnsembleTestResultRow(
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
                        ckpt_step=int(ckpt_step),
                    )
                )
            ]
        )
        crash_row.to_csv(worker_out_csv, index=False)


def write_summary(
    summary_path: Path,
    result_df,
    dataset_dirs: List[Path],
    wall_seconds: float,
    coverage: dict[str, Any],
) -> None:
    ens.ensure_runtime_deps()

    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else ens.pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else ens.pd.DataFrame()
    skipped_df = result_df[result_df["status"] == "skip"].copy() if len(result_df) else ens.pd.DataFrame()

    lines = [
        f"discovered_datasets: {len(dataset_dirs)}",
        f"processed_datasets: {len(result_df)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        f"skipped_count: {len(skipped_df)}",
        (
            f"avg_accuracy_ok: {ok_df['accuracy'].mean():.6f}"
            if len(ok_df)
            else "avg_accuracy_ok: (none)"
        ),
        f"ckpt_step: {coverage.get('ckpt_step')}",
        f"step_ckpts_found: {coverage.get('step_ckpts_found')}",
        f"missing_ckpts: {coverage.get('missing_ckpts')}",
        f"ambiguous_ckpts: {coverage.get('ambiguous_ckpts')}",
        f"wall_seconds: {wall_seconds:.3f}",
    ]

    failed_names = ", ".join(failed_df["dataset_name"].astype(str).tolist()) if len(failed_df) else "(none)"
    skipped_names = ", ".join(skipped_df["dataset_name"].astype(str).tolist()) if len(skipped_df) else "(none)"
    missing_names = ", ".join(coverage.get("missing_ckpt_datasets", [])) or "(none)"
    ambiguous_names = ", ".join(coverage.get("ambiguous_ckpt_datasets", [])) or "(none)"
    lines.extend(
        [
            f"failed_datasets: {failed_names}",
            f"skipped_datasets: {skipped_names}",
            f"missing_ckpt_datasets: {missing_names}",
            f"ambiguous_ckpt_datasets: {ambiguous_names}",
        ]
    )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_dataset_names(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    names: list[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                names.append(item)
    return names or None


def filter_dataset_dirs(dataset_dirs: List[Path], requested_names: list[str] | None) -> List[Path]:
    if not requested_names:
        return dataset_dirs

    by_name = {path.name: path for path in dataset_dirs}
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        raise FileNotFoundError(
            "Requested datasets not found under data root: " + ", ".join(missing)
        )
    requested_set = set(requested_names)
    return [path for path in dataset_dirs if path.name in requested_set]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run per-dataset TabICL ensemble inference from saved TTT step checkpoints."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--ckpt-root", default=str(DEFAULT_CKPT_ROOT))
    parser.add_argument("--ckpt-step", type=int, default=15)
    parser.add_argument("--ckpt-model-name", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--gpu-groups", default=None)
    parser.add_argument("--n-estimators", type=int, default=32)
    parser.add_argument("--batch-size", type=ens.parse_optional_int, default=8)
    parser.add_argument("--kv-cache", type=ens.parse_kv_cache, default=False)
    parser.add_argument("--use-amp", type=ens.parse_auto_bool, default="auto")
    parser.add_argument("--use-fa3", type=ens.parse_auto_bool, default="auto")
    parser.add_argument("--offload-mode", choices=["auto", "gpu", "cpu", "disk"], default="auto")
    parser.add_argument("--checkpoint-version", default=DEFAULT_CHECKPOINT_VERSION)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument(
        "--dataset-names",
        nargs="*",
        default=None,
        help="Optional dataset names to run. Supports space-separated names or comma-separated groups.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def resolve_gpu_ids(args: argparse.Namespace) -> List[int]:
    gpu_ids = ens.parse_gpu_id_list(args.gpus) if args.gpus else ens.detect_default_gpu_ids()
    if args.workers is None:
        args.workers = len(gpu_ids)
    if len(gpu_ids) != args.workers:
        raise ValueError(f"--gpus must contain exactly {args.workers} ids")
    return gpu_ids


def resolve_gpu_assignments(args: argparse.Namespace) -> tuple[List[int], List[str]]:
    if args.gpu_groups:
        if args.gpus:
            raise ValueError("Use either --gpu-groups or --gpus, not both")
        gpu_groups = ens.parse_gpu_group_list(args.gpu_groups)
        if not gpu_groups:
            raise ValueError("--gpu-groups must contain at least one group")
        if args.workers is None:
            args.workers = len(gpu_groups)
        if len(gpu_groups) != args.workers:
            raise ValueError(f"--gpu-groups must contain exactly {args.workers} groups")
        gpu_ids = [ens.first_gpu_id_from_group(gpu_group) for gpu_group in gpu_groups]
        return gpu_ids, gpu_groups

    gpu_ids = resolve_gpu_ids(args)
    return gpu_ids, [str(gpu_id) for gpu_id in gpu_ids]


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    ens.ensure_runtime_deps()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    ckpt_root = Path(args.ckpt_root)
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {ckpt_root}")
    if not ckpt_root.is_dir():
        raise NotADirectoryError(f"Checkpoint root is not a directory: {ckpt_root}")

    dataset_dirs = ens.find_dataset_dirs(data_root)
    dataset_dirs = filter_dataset_dirs(dataset_dirs, parse_dataset_names(args.dataset_names))
    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: args.max_datasets]
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset directories found under {data_root}")

    coverage = count_available_ckpts(
        dataset_dirs,
        ckpt_root,
        args.ckpt_step,
        args.ckpt_model_name,
    )
    print("checkpoint_coverage:")
    print(json.dumps(coverage, indent=2, ensure_ascii=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids, gpu_groups = resolve_gpu_assignments(args)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    model_kwargs = build_model_kwargs(args)
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
                str(ckpt_root),
                int(args.ckpt_step),
                args.ckpt_model_name,
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
                    f"ready assigned={message.get('assigned_count', '?')}",
                    flush=True,
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

    dfs = []
    for worker_csv in worker_csv_paths:
        if worker_csv.exists():
            try:
                dfs.append(ens.pd.read_csv(worker_csv))
            except ens.pd.errors.EmptyDataError:
                dfs.append(ens.pd.DataFrame(columns=EnsembleTestResultRow.__annotations__.keys()))

    all_df = (
        ens.pd.concat(dfs, ignore_index=True)
        if dfs
        else ens.pd.DataFrame(columns=EnsembleTestResultRow.__annotations__.keys())
    )
    all_csv = out_dir / "all_classification_results.csv"
    summary_txt = out_dir / "summary.txt"
    all_df.to_csv(all_csv, index=False)

    wall_seconds = time.time() - start_time
    write_summary(summary_txt, all_df, dataset_dirs, wall_seconds, coverage)

    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {summary_txt}")
    print("model_kwargs:")
    print(json.dumps(model_kwargs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
