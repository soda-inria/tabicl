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
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import TTT_1A_ICL as ttt1a

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_CONTEXT_LENGTH = 10
DEFAULT_OUT_DIR = "context_eva_result/context100_step40_lr5e-6"
METHOD_BASELINE = "baseline"
METHOD_TTT = "ttt"


@dataclass
class ZeroContextResultRow:
    dataset_name: str
    dataset_dir: str
    method: str
    context_length: int
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
    n_train_b: int = 0
    n_holdout_c: int = 0
    ttt_loss: Optional[float] = None
    ttt_steps: int = 0
    ttt_lr: Optional[float] = None
    ttt_applied: bool = False
    ttt_update_seconds: float = 0.0
    ttt_split_strategy: Optional[str] = None
    ttt_split_reason: Optional[str] = None
    ttt_step_losses: Optional[str] = None


def ensure_runtime_deps() -> None:
    ttt1a.ensure_runtime_deps()


def _np():
    ensure_runtime_deps()
    return ttt1a.np


def _pd():
    ensure_runtime_deps()
    return ttt1a.pd


def load_train_val_test(dataset_dir: Path):
    train_split, val_split, test_split = ttt1a.find_split_files(dataset_dir)

    X_train, y_train = ttt1a.load_split(
        train_split[0],
        train_split[1],
        train_split[2],
        context=f"{dataset_dir.name}-train",
    )
    val_count = 0
    if val_split is not None:
        X_val, y_val = ttt1a.load_split(
            val_split[0],
            val_split[1],
            val_split[2],
            context=f"{dataset_dir.name}-val",
        )
        val_count = int(len(y_val))
        X_train = _pd().concat([X_train, X_val], axis=0, ignore_index=True)
        y_train = _np().concatenate([_np().asarray(y_train), _np().asarray(y_val)], axis=0)

    X_test, y_test = ttt1a.load_split(
        test_split[0],
        test_split[1],
        test_split[2],
        context=f"{dataset_dir.name}-test",
    )
    return X_train, y_train, val_count, X_test, y_test


def select_context_indices(y, context_length: int, random_state: int):
    ensure_runtime_deps()

    y_array = _np().asarray(y)
    if context_length < 1:
        raise ValueError("--context-length must be >= 1 for supported TabICL ICL evaluation")
    if len(y_array) < context_length:
        raise ValueError(
            f"Need at least context_length={context_length} training samples, got n_train={len(y_array)}"
        )

    classes = _pd().unique(_pd().Series(y_array))
    if len(classes) > context_length:
        raise ValueError(
            f"Cannot build a class-covering context because n_classes={len(classes)} "
            f"exceeds context_length={context_length}"
        )

    rng = _np().random.default_rng(int(random_state))
    selected: list[int] = []
    for class_value in classes:
        candidates = _np().flatnonzero(y_array == class_value)
        if len(candidates) == 0:
            continue
        selected.append(int(rng.choice(candidates)))

    selected_array = _np().asarray(selected, dtype=int)
    remaining = _np().setdiff1d(_np().arange(len(y_array)), selected_array, assume_unique=False)
    needed = context_length - len(selected)
    if needed > 0:
        selected.extend(rng.choice(remaining, size=needed, replace=False).astype(int).tolist())

    return _np().asarray(selected, dtype=int)


def _derive_model_name_from_classifier(classifier) -> str:
    return ttt1a._derive_model_name(
        str(getattr(classifier, "model_path", None)) if getattr(classifier, "model_path", None) is not None else None,
        str(getattr(classifier, "checkpoint_version", None))
        if getattr(classifier, "checkpoint_version", None) is not None
        else None,
    )


def _skip_row(
    dataset_dir: Path,
    method: str,
    task_type: Optional[str],
    error: str,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    n_train: int = 0,
    n_val: int = 0,
    n_test: int = 0,
    n_features: int = 0,
    n_classes: int = 0,
) -> ZeroContextResultRow:
    return ZeroContextResultRow(
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir.as_posix(),
        method=method,
        context_length=context_length,
        task_type=task_type,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        n_features=n_features,
        n_classes=n_classes,
        accuracy=None,
        fit_seconds=0.0,
        predict_seconds=0.0,
        status="skip",
        error=error,
    )


def _fail_row(
    dataset_dir: Path,
    method: str,
    task_type: Optional[str],
    exc: BaseException,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
) -> ZeroContextResultRow:
    return ZeroContextResultRow(
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir.as_posix(),
        method=method,
        context_length=context_length,
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


def evaluate_baseline_context(
    classifier_cls,
    model_kwargs: Dict[str, object],
    dataset_dir: Path,
    task_type: Optional[str],
    X_train,
    y_train,
    val_count: int,
    X_test,
    y_test,
    context_indices,
    context_length: int,
) -> tuple[ZeroContextResultRow, Optional[int]]:
    classifier = classifier_cls(**dict(model_kwargs))
    X_context = ttt1a.take_rows(X_train, context_indices)
    y_context = _np().asarray(y_train)[context_indices]
    fit_start = time.time()
    classifier.fit(X_context, y_context)
    fit_seconds = time.time() - fit_start

    n_classes = int(len(_pd().unique(_pd().Series(_np().asarray(y_train)))))
    n_context_classes = int(len(classifier.classes_))
    max_classes = int(getattr(classifier.model_, "max_classes", n_classes))
    common = {
        "n_train": int(len(y_train)),
        "n_val": int(val_count),
        "n_test": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
        "n_classes": n_classes,
    }
    if n_classes > max_classes:
        return (
            _skip_row(
                dataset_dir,
                METHOD_BASELINE,
                task_type,
                f"Skipped context evaluation because n_classes={n_classes} exceeds model max_classes={max_classes}",
                context_length=context_length,
                **common,
            ),
            max_classes,
        )
    if n_context_classes != n_classes:
        return (
            _skip_row(
                dataset_dir,
                METHOD_BASELINE,
                task_type,
                (
                    "Skipped context evaluation because selected context does not cover all classes: "
                    f"context_classes={n_context_classes}, n_classes={n_classes}"
                ),
                context_length=context_length,
                **common,
            ),
            max_classes,
        )

    pred_start = time.time()
    y_pred = classifier.predict(X_test)
    predict_seconds = time.time() - pred_start
    accuracy = float(_np().mean(_np().asarray(y_pred) == _np().asarray(y_test)))
    return (
        ZeroContextResultRow(
            dataset_name=dataset_dir.name,
            dataset_dir=dataset_dir.as_posix(),
            method=METHOD_BASELINE,
            context_length=context_length,
            task_type=task_type,
            accuracy=accuracy,
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            status="ok",
            error=None,
            **common,
        ),
        max_classes,
    )


def evaluate_ttt_context(
    classifier_cls,
    model_kwargs: Dict[str, object],
    dataset_dir: Path,
    info: Optional[dict],
    task_type: Optional[str],
    X_train,
    y_train,
    val_count: int,
    X_test,
    y_test,
    ttt_config: ttt1a.TTTConfig,
    max_classes: Optional[int],
    context_indices,
    context_length: int,
) -> ZeroContextResultRow:
    n_classes = int(len(_pd().unique(_pd().Series(_np().asarray(y_train)))))
    common = {
        "n_train": int(len(y_train)),
        "n_val": int(val_count),
        "n_test": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
        "n_classes": n_classes,
    }
    if max_classes is not None and n_classes > int(max_classes):
        return _skip_row(
            dataset_dir,
            METHOD_TTT,
            task_type,
            f"Skipped context evaluation because n_classes={n_classes} exceeds model max_classes={max_classes}",
            context_length=context_length,
            **common,
        )

    classifier = classifier_cls(**dict(model_kwargs))
    X_context = ttt1a.take_rows(X_train, context_indices)
    y_context = _np().asarray(y_train)[context_indices]
    ttt_loss = None
    ttt_steps = 0
    ttt_lr = ttt_config.lr if ttt_config.enabled else None
    ttt_applied = False
    ttt_update_seconds = 0.0
    ttt_split_strategy = None
    ttt_split_reason = None
    ttt_step_losses = None
    n_train_b = 0
    n_holdout_c = 0

    fit_start = time.time()
    if ttt_config.enabled:
        if ttt1a.should_skip_ttt_for_dataset(dataset_dir, info):
            ttt_split_reason = "TTT skipped for dataset=volkert to avoid OOM"
            classifier.fit(X_context, y_context)
        else:
            split = ttt1a.split_ttt_holdout(y_train, ttt_config)
            ttt_split_strategy = split.strategy
            ttt_split_reason = split.reason
            n_train_b = int(len(split.b_indices))
            n_holdout_c = int(len(split.c_indices))
            if n_train_b > 0 and n_holdout_c > 0:
                X_b = ttt1a.take_rows(X_train, split.b_indices)
                y_b = _np().asarray(y_train)[split.b_indices]
                X_c = ttt1a.take_rows(X_train, split.c_indices)
                y_c = _np().asarray(y_train)[split.c_indices]
                ttt_result = ttt1a.run_ttt_holdout_update(
                    classifier,
                    X_b,
                    y_b,
                    X_c,
                    y_c,
                    ttt_config,
                    model_name=_derive_model_name_from_classifier(classifier),
                    dataset_name=dataset_dir.name,
                )
            else:
                ttt_result = ttt1a.TTTUpdateResult(
                    applied=False,
                    loss=None,
                    steps=0,
                    update_seconds=0.0,
                    reason="B/C split produced an empty side; skipped TTT",
                    step_losses=[],
                )

            ttt_loss = ttt_result.loss
            ttt_steps = ttt_result.steps
            ttt_applied = ttt_result.applied
            ttt_update_seconds = float(ttt_result.update_seconds)
            if ttt_result.step_losses is not None:
                ttt_step_losses = json.dumps(ttt_result.step_losses, ensure_ascii=False)
            if ttt_result.reason:
                ttt_split_reason = f"{ttt_split_reason} | {ttt_result.reason}"

            if ttt_applied:
                ttt1a._fit_preserving_model_weights(classifier, X_context, y_context)
            else:
                classifier.fit(X_context, y_context)
    else:
        classifier.fit(X_context, y_context)
    fit_seconds = time.time() - fit_start

    fitted_classes = int(len(classifier.classes_))
    fitted_max_classes = int(getattr(classifier.model_, "max_classes", fitted_classes))
    if fitted_classes > fitted_max_classes:
        return _skip_row(
            dataset_dir,
            METHOD_TTT,
            task_type,
            (
                "Skipped context evaluation because "
                f"n_classes={fitted_classes} exceeds model max_classes={fitted_max_classes}"
            ),
            context_length=context_length,
            **{**common, "n_classes": fitted_classes},
        )

    pred_start = time.time()
    y_pred = classifier.predict(X_test)
    predict_seconds = time.time() - pred_start
    accuracy = float(_np().mean(_np().asarray(y_pred) == _np().asarray(y_test)))
    return ZeroContextResultRow(
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir.as_posix(),
        method=METHOD_TTT,
        context_length=context_length,
        task_type=task_type,
        accuracy=accuracy,
        fit_seconds=float(fit_seconds),
        predict_seconds=float(predict_seconds),
        status="ok",
        error=None,
        n_train_b=n_train_b,
        n_holdout_c=n_holdout_c,
        ttt_loss=ttt_loss,
        ttt_steps=ttt_steps,
        ttt_lr=ttt_lr,
        ttt_applied=ttt_applied,
        ttt_update_seconds=ttt_update_seconds,
        ttt_split_strategy=ttt_split_strategy,
        ttt_split_reason=ttt_split_reason,
        ttt_step_losses=ttt_step_losses,
        **common,
    )


def evaluate_one_dataset_context(
    classifier_cls,
    model_kwargs: Dict[str, object],
    dataset_dir: Path,
    ttt_config: ttt1a.TTTConfig,
    context_length: int,
    random_state: int,
) -> List[ZeroContextResultRow]:
    task_type: Optional[str] = None
    try:
        info = ttt1a.load_dataset_info(dataset_dir)
        task_type = str(info.get("task_type", "")).lower() if info else None
        if task_type not in ttt1a.CLASSIFICATION_TASKS:
            reason = f"Skipped due to task_type={task_type!r}"
            return [
                _skip_row(dataset_dir, METHOD_BASELINE, task_type, reason, context_length=context_length),
                _skip_row(dataset_dir, METHOD_TTT, task_type, reason, context_length=context_length),
            ]

        X_train, y_train, val_count, X_test, y_test = load_train_val_test(dataset_dir)
        common = {
            "n_train": int(len(y_train)),
            "n_val": int(val_count),
            "n_test": int(len(y_test)),
            "n_features": int(X_train.shape[1]),
            "n_classes": int(len(_pd().unique(_pd().Series(_np().asarray(y_train))))),
        }
        try:
            context_indices = select_context_indices(y_train, context_length, random_state)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            return [
                _skip_row(
                    dataset_dir,
                    METHOD_BASELINE,
                    task_type,
                    reason,
                    context_length=context_length,
                    **common,
                ),
                _skip_row(
                    dataset_dir,
                    METHOD_TTT,
                    task_type,
                    reason,
                    context_length=context_length,
                    **common,
                ),
            ]

        try:
            baseline_row, max_classes = evaluate_baseline_context(
                classifier_cls,
                model_kwargs,
                dataset_dir,
                task_type,
                X_train,
                y_train,
                val_count,
                X_test,
                y_test,
                context_indices,
                context_length,
            )
        except Exception as exc:
            return [
                _fail_row(dataset_dir, METHOD_BASELINE, task_type, exc, context_length=context_length),
                _fail_row(dataset_dir, METHOD_TTT, task_type, exc, context_length=context_length),
            ]

        if baseline_row.status == "skip":
            return [
                baseline_row,
                _skip_row(
                    dataset_dir,
                    METHOD_TTT,
                    task_type,
                    baseline_row.error or "Skipped because baseline context row was skipped",
                    context_length=context_length,
                    n_train=baseline_row.n_train,
                    n_val=baseline_row.n_val,
                    n_test=baseline_row.n_test,
                    n_features=baseline_row.n_features,
                    n_classes=baseline_row.n_classes,
                ),
            ]

        try:
            ttt_row = evaluate_ttt_context(
                classifier_cls,
                model_kwargs,
                dataset_dir,
                info,
                task_type,
                X_train,
                y_train,
                val_count,
                X_test,
                y_test,
                ttt_config,
                max_classes,
                context_indices,
                context_length,
            )
        except Exception as exc:
            ttt_row = _fail_row(dataset_dir, METHOD_TTT, task_type, exc, context_length=context_length)
        return [baseline_row, ttt_row]
    except Exception as exc:
        return [
            _fail_row(dataset_dir, METHOD_BASELINE, task_type, exc, context_length=context_length),
            _fail_row(dataset_dir, METHOD_TTT, task_type, exc, context_length=context_length),
        ]


def format_context_log(worker_label: str, row: ZeroContextResultRow) -> str:
    if row.status == "ok":
        return (
            f"[{worker_label}] [ok] {row.dataset_name} method={row.method} "
            f"context={row.context_length} accuracy={ttt1a.format_optional_float(row.accuracy)} "
            f"fit={row.fit_seconds:.3f}s predict={row.predict_seconds:.3f}s "
            f"ttt_applied={row.ttt_applied} ttt_loss={ttt1a.format_optional_float(row.ttt_loss)}"
        )
    return f"[{worker_label}] [{row.status}] {row.dataset_name} method={row.method} reason={row.error}"


def worker_main(
    worker_id: int,
    gpu_id: int,
    gpu_group: str,
    assigned_dataset_dirs: List[str],
    ready_queue,
    start_event,
    worker_out_csv: str,
    model_kwargs: Dict[str, object],
    ttt_config: ttt1a.TTTConfig,
    context_length: int,
    random_state: int,
    verbose: bool,
) -> None:
    try:
        ensure_runtime_deps()
        device_str = ttt1a.apply_worker_environment_updates(gpu_group)
        worker_label = f"worker {worker_id} | gpu {gpu_group}"
        worker_ttt_config = replace(ttt_config, gpu_group=gpu_group)

        import torch
        from tabicl import TabICLClassifier

        torch_diag = ttt1a.collect_torch_diagnostics()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU backend is not available in this worker. "
                f"Diagnostics: {json.dumps(torch_diag, ensure_ascii=False)}"
            )

        worker_kwargs = dict(model_kwargs)
        worker_kwargs["device"] = device_str

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

        rows: List[ZeroContextResultRow] = []
        for dataset_dir in assigned_dataset_dirs:
            dataset_rows = evaluate_one_dataset_context(
                TabICLClassifier,
                worker_kwargs,
                Path(dataset_dir),
                worker_ttt_config,
                context_length,
                random_state,
            )
            rows.extend(dataset_rows)
            for row in dataset_rows:
                print(format_context_log(worker_label, row), flush=True)
            ttt1a.force_memory_cleanup(device_str)

        _pd().DataFrame([asdict(row) for row in rows]).to_csv(worker_out_csv, index=False)
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
        crash_rows = [
            ZeroContextResultRow(
                dataset_name=f"__WORKER_CRASH__{worker_id}",
                dataset_dir="__worker__",
                method=method,
                context_length=context_length,
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
            for method in (METHOD_BASELINE, METHOD_TTT)
        ]
        _pd().DataFrame([asdict(row) for row in crash_rows]).to_csv(worker_out_csv, index=False)


def write_context_plot(summary_df, plot_path: Path, context_length: int) -> None:
    ok_df = summary_df[summary_df["status"] == "ok"].copy() if len(summary_df) else _pd().DataFrame()
    means = ok_df.groupby("method")["accuracy"].mean() if len(ok_df) else {}
    baseline_acc = float(means.get(METHOD_BASELINE, float("nan")))
    ttt_acc = float(means.get(METHOD_TTT, float("nan")))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    labels = ["TabICL", "TabICL + TTT"]
    values = [baseline_acc, ttt_acc]
    bars = ax.bar(labels, values, color=["#1f77b4", "#ff7f0e"], width=0.55)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("n (Context length)")
    ax.set_title(f"ICL accuracy (n={context_length})")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.35)
    for bar, value in zip(bars, values):
        if value == value:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def write_summary(
    out_dir: Path,
    result_df,
    dataset_dirs: List[Path],
    wall_seconds: float,
    context_length: int,
) -> None:
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else _pd().DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else _pd().DataFrame()
    skipped_df = result_df[result_df["status"] == "skip"].copy() if len(result_df) else _pd().DataFrame()

    empty_method_df = _pd().DataFrame(columns=["dataset_name", "accuracy"])
    baseline_ok = ok_df[ok_df["method"] == METHOD_BASELINE].copy() if len(ok_df) else empty_method_df.copy()
    ttt_ok = ok_df[ok_df["method"] == METHOD_TTT].copy() if len(ok_df) else empty_method_df.copy()
    shared = baseline_ok[["dataset_name", "accuracy"]].merge(
        ttt_ok[["dataset_name", "accuracy"]],
        on="dataset_name",
        suffixes=("_baseline", "_ttt"),
    )

    lines = [
        f"discovered_datasets: {len(dataset_dirs)}",
        f"processed_rows: {len(result_df)}",
        f"ok_rows: {len(ok_df)}",
        f"failed_rows: {len(failed_df)}",
        f"skipped_rows: {len(skipped_df)}",
        (
            f"baseline_avg_accuracy_ok: {baseline_ok['accuracy'].mean():.6f}"
            if len(baseline_ok)
            else "baseline_avg_accuracy_ok: (none)"
        ),
        (
            f"ttt_avg_accuracy_ok: {ttt_ok['accuracy'].mean():.6f}"
            if len(ttt_ok)
            else "ttt_avg_accuracy_ok: (none)"
        ),
        f"shared_ok_datasets: {len(shared)}",
        (
            f"shared_delta_ttt_minus_baseline: "
            f"{(shared['accuracy_ttt'] - shared['accuracy_baseline']).mean():.6f}"
            if len(shared)
            else "shared_delta_ttt_minus_baseline: (none)"
        ),
        f"wall_seconds: {wall_seconds:.3f}",
    ]

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_path = out_dir / "context_accuracy.png"
    try:
        write_context_plot(result_df, plot_path, context_length)
    except Exception as exc:
        lines.append(f"plot_error: {type(exc).__name__}: {exc}")
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-context TabICL vs TabICL+TTT classification accuracy."
    )
    parser.add_argument("--data-root", default=str(ttt1a.DEFAULT_DATA_ROOT))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--checkpoint-version", default=ttt1a.DEFAULT_CHECKPOINT_VERSION)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--gpu-groups", default=None)
    parser.add_argument("--context-length", type=int, default=100)
    parser.add_argument("--n-estimators", type=int, default=32)
    parser.add_argument("--batch-size", type=ttt1a.parse_optional_int, default=8)
    parser.add_argument("--kv-cache", type=ttt1a.parse_kv_cache, default=False)
    parser.add_argument("--use-amp", type=ttt1a.parse_auto_bool, default="auto")
    parser.add_argument("--use-fa3", type=ttt1a.parse_auto_bool, default="auto")
    parser.add_argument("--offload-mode", choices=["auto", "gpu", "cpu", "disk"], default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--ttt-holdout", type=ttt1a.parse_bool, nargs="?", const="true", default=True)
    parser.add_argument("--no-ttt-holdout", dest="ttt_holdout", action="store_false")
    parser.add_argument("--ttt-lr", type=float, default=2e-6)
    parser.add_argument("--ttt-scheduler", choices=["constant"], default="constant")
    parser.add_argument("--ttt-grad-clip", type=float, default=1.0)
    parser.add_argument("--ttt-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--ttt-micro-batch-size", type=int, default=1)
    parser.add_argument("--ttt-weight-decay", type=float, default=0.0)
    parser.add_argument("--ttt-steps", type=int, default=40)
    parser.add_argument("--ttt-freeze-col", type=ttt1a.parse_bool, default=True)
    parser.add_argument("--ttt-freeze-row", type=ttt1a.parse_bool, default=True)
    parser.add_argument("--ttt-save-ckpt", type=ttt1a.parse_bool, default=False)
    parser.add_argument("--ttt-save-ckpt-every", type=int, default=40)
    parser.add_argument("--ttt-data-parallel", default=True, action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def build_common_model_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "n_estimators": args.n_estimators,
        "batch_size": args.batch_size,
        "kv_cache": args.kv_cache,
        "allow_auto_download": True,
        "checkpoint_version": args.checkpoint_version,
        "model_path": ttt1a.normalize_model_path(args.model_path or ttt1a.DEFAULT_MODEL_PATH),
        "use_amp": args.use_amp,
        "use_fa3": args.use_fa3,
        "offload_mode": args.offload_mode,
        "random_state": args.random_state,
    }


def run_context_eval(
    args: argparse.Namespace,
    dataset_dirs: List[Path],
    gpu_ids: List[int],
    gpu_groups: List[str],
    out_dir: Path,
) -> None:
    model_kwargs = build_common_model_kwargs(args)
    ttt_config = ttt1a.build_ttt_config(args)
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
                int(args.context_length),
                int(args.random_state),
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
                    "Some workers exited before initialization completed: " + ", ".join(dead_workers)
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

    dfs: List[Any] = []
    for worker_csv in worker_csv_paths:
        if worker_csv.exists():
            dfs.append(_pd().read_csv(worker_csv))

    all_df = (
        _pd().concat(dfs, ignore_index=True)
        if dfs
        else _pd().DataFrame(columns=ZeroContextResultRow.__annotations__.keys())
    )
    all_csv = out_dir / "context_results.csv"
    all_df.to_csv(all_csv, index=False)
    write_summary(out_dir, all_df, dataset_dirs, time.time() - start_time, int(args.context_length))

    print(f"saved_results_csv: {all_csv}")
    print(f"saved_summary: {out_dir / 'summary.txt'}")
    plot_path = out_dir / "context_accuracy.png"
    if plot_path.exists():
        print(f"saved_plot: {plot_path}")
    else:
        print(f"plot_skipped: {plot_path}")
    print("model_kwargs:")
    print(json.dumps(model_kwargs, indent=2, ensure_ascii=False))
    if ttt_config.enabled:
        print("ttt_config:")
        print(json.dumps(asdict(ttt_config), indent=2, ensure_ascii=False))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if int(args.context_length) < 1:
        raise ValueError("TTT_context_eva.py requires --context-length >= 1")

    ensure_runtime_deps()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = ttt1a.find_dataset_dirs(data_root)
    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: args.max_datasets]
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset directories found under {data_root}")

    gpu_ids, gpu_groups = ttt1a.resolve_gpu_assignments(args)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    run_context_eval(args, dataset_dirs, gpu_ids, gpu_groups, out_dir)


if __name__ == "__main__":
    main()
