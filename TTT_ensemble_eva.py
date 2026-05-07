#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import struct
import sys
import time
import traceback
import zlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import TTT_1B_ensmble as ens


METHOD_BASELINE = "baseline"
METHOD_TTT = "ttt"
# DEFAULT_CONTEXT_LENGTHS = "1,2,4,5,8,10,20,25,40,50,100,125,200,250,500,1000"
DEFAULT_CONTEXT_LENGTHS = "20,25,40,50,100,125,200,250,500,1000"
DEFAULT_OUT_DIR = "result/ensemble_context_eva/k1000_ttt10"


@dataclass
class EnsembleContextResultRow:
    dataset_name: str
    dataset_dir: str
    method: str
    context_length: int
    group_count: int
    ttt_iters_per_group: int
    k_train: int
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


def ensure_runtime_deps() -> None:
    ens.ensure_runtime_deps()


def _np():
    ensure_runtime_deps()
    return ens.np


def _pd():
    ensure_runtime_deps()
    return ens.pd


def parse_context_lengths(value: str) -> List[int]:
    lengths: list[int] = []
    for raw_item in str(value).split(","):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        context_length = int(raw_item)
        if context_length < 1:
            raise argparse.ArgumentTypeError("--context-lengths values must be >= 1")
        lengths.append(context_length)
    lengths = sorted(set(lengths))
    if not lengths:
        raise argparse.ArgumentTypeError("--context-lengths must contain at least one value")
    return lengths


def load_train_val_test(dataset_dir: Path):
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
        X_train = _pd().concat([X_train, X_val], axis=0, ignore_index=True)
        y_train = _np().concatenate([_np().asarray(y_train), _np().asarray(y_val)], axis=0)

    X_test, y_test = ens.load_split(
        test_split[0],
        test_split[1],
        test_split[2],
        context=f"{dataset_dir.name}-test",
    )
    return X_train, y_train, val_count, X_test, y_test


def cap_features(X_train, X_test, max_features: Optional[int], random_state: int):
    if max_features is None or int(max_features) <= 0:
        return X_train, X_test
    max_features = int(max_features)
    n_features = int(X_train.shape[1])
    if n_features <= max_features:
        return X_train, X_test

    rng = _np().random.default_rng(int(random_state))
    selected_cols = sorted(rng.choice(_np().arange(n_features), size=max_features, replace=False).astype(int).tolist())
    if hasattr(X_train, "iloc"):
        return (
            X_train.iloc[:, selected_cols].copy(),
            X_test.iloc[:, selected_cols].copy(),
        )
    return _np().asarray(X_train)[:, selected_cols], _np().asarray(X_test)[:, selected_cols]


def filter_top_classes(
    X_train,
    y_train,
    X_test,
    y_test,
    max_classes: int = 10,
):
    y_train_array = _np().asarray(y_train)
    y_test_array = _np().asarray(y_test)
    if int(max_classes) < 1:
        raise ValueError("max_classes must be >= 1")

    class_counts: dict[object, int] = {}
    first_seen: dict[object, int] = {}
    for idx, value in enumerate(y_train_array.tolist()):
        if value not in class_counts:
            class_counts[value] = 0
            first_seen[value] = idx
        class_counts[value] += 1

    if len(class_counts) < int(max_classes):
        return (
            X_train,
            y_train_array,
            X_test,
            y_test_array,
            [],
            (
                "Skipped because train+val has fewer than "
                f"{int(max_classes)} classes after top-class filtering: "
                f"n_classes={len(class_counts)}"
            ),
        )

    top_classes = [
        class_value
        for class_value, _ in sorted(
            class_counts.items(),
            key=lambda item: (-item[1], first_seen[item[0]]),
        )[: int(max_classes)]
    ]
    top_class_set = set(top_classes)
    train_mask = _np().asarray([value in top_class_set for value in y_train_array], dtype=bool)
    test_mask = _np().asarray([value in top_class_set for value in y_test_array], dtype=bool)

    if int(train_mask.sum()) == 0:
        return (
            X_train,
            y_train_array,
            X_test,
            y_test_array,
            top_classes,
            "Skipped because top-class filtering removed all train+val samples",
        )
    if int(test_mask.sum()) == 0:
        return (
            X_train,
            y_train_array,
            X_test,
            y_test_array,
            top_classes,
            "Skipped because top-class filtering removed all test samples",
        )

    return (
        ens.take_rows(X_train, _np().flatnonzero(train_mask)),
        y_train_array[train_mask],
        ens.take_rows(X_test, _np().flatnonzero(test_mask)),
        y_test_array[test_mask],
        top_classes,
        None,
    )


def class_covering_indices(
    y,
    sample_size: int,
    random_state: int,
    *,
    candidate_indices=None,
):
    y_array = _np().asarray(y)
    if candidate_indices is None:
        candidates = _np().arange(len(y_array), dtype=int)
    else:
        candidates = _np().asarray(candidate_indices, dtype=int)

    if sample_size < 1:
        raise ValueError("sample_size must be >= 1")
    if len(candidates) < sample_size:
        raise ValueError(f"Need sample_size={sample_size}, got only {len(candidates)} candidate samples")

    candidate_y = y_array[candidates]
    classes = _pd().unique(_pd().Series(candidate_y))
    if len(classes) > sample_size:
        raise ValueError(
            f"Cannot select class-covering sample because n_classes={len(classes)} exceeds sample_size={sample_size}"
        )

    rng = _np().random.default_rng(int(random_state))
    selected: list[int] = []
    for class_value in classes:
        class_candidates = candidates[candidate_y == class_value]
        if len(class_candidates) == 0:
            continue
        selected.append(int(rng.choice(class_candidates)))

    selected_array = _np().asarray(selected, dtype=int)
    remaining = _np().setdiff1d(candidates, selected_array, assume_unique=False)
    needed = sample_size - len(selected)
    if needed > 0:
        selected.extend(rng.choice(remaining, size=needed, replace=False).astype(int).tolist())

    return _np().asarray(selected, dtype=int)


def _empty_row(
    dataset_dir: Path,
    *,
    method: str,
    context_length: int,
    group_count: int,
    ttt_iters_per_group: int,
    k_train: int,
    task_type: Optional[str],
    status: str,
    error: str,
    n_train: int = 0,
    n_val: int = 0,
    n_test: int = 0,
    n_features: int = 0,
    n_classes: int = 0,
) -> EnsembleContextResultRow:
    return EnsembleContextResultRow(
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir.as_posix(),
        method=method,
        context_length=int(context_length),
        group_count=int(group_count),
        ttt_iters_per_group=int(ttt_iters_per_group),
        k_train=int(k_train),
        task_type=task_type,
        n_train=int(n_train),
        n_val=int(n_val),
        n_test=int(n_test),
        n_features=int(n_features),
        n_classes=int(n_classes),
        accuracy=None,
        fit_seconds=0.0,
        predict_seconds=0.0,
        status=status,
        error=error,
    )


def _fail_pair(dataset_dir: Path, exc: BaseException, context_lengths: List[int], args_snapshot: Dict[str, Any]):
    rows: list[EnsembleContextResultRow] = []
    for context_length in context_lengths:
        group_count = int(math.ceil(float(args_snapshot["k_train"]) / float(context_length)))
        for method in (METHOD_BASELINE, METHOD_TTT):
            rows.append(
                _empty_row(
                    dataset_dir,
                    method=method,
                    context_length=context_length,
                    group_count=group_count,
                    ttt_iters_per_group=int(args_snapshot["ttt_iters_per_group"]),
                    k_train=int(args_snapshot["k_train"]),
                    task_type=None,
                    status="fail",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return rows


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
    group_count: int,
    ttt_iters_per_group: int,
    k_train: int,
) -> tuple[EnsembleContextResultRow, Optional[int]]:
    classifier = None
    try:
        classifier = classifier_cls(**dict(model_kwargs))
        X_context = ens.take_rows(X_train, context_indices)
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
                _empty_row(
                    dataset_dir,
                    method=METHOD_BASELINE,
                    context_length=context_length,
                    group_count=group_count,
                    ttt_iters_per_group=ttt_iters_per_group,
                    k_train=k_train,
                    task_type=task_type,
                    status="skip",
                    error=f"Skipped because n_classes={n_classes} exceeds model max_classes={max_classes}",
                    **common,
                ),
                max_classes,
            )
        if n_context_classes != n_classes:
            return (
                _empty_row(
                    dataset_dir,
                    method=METHOD_BASELINE,
                    context_length=context_length,
                    group_count=group_count,
                    ttt_iters_per_group=ttt_iters_per_group,
                    k_train=k_train,
                    task_type=task_type,
                    status="skip",
                    error=(
                        "Skipped because selected context does not cover all classes: "
                        f"context_classes={n_context_classes}, n_classes={n_classes}"
                    ),
                    **common,
                ),
                max_classes,
            )

        pred_start = time.time()
        y_pred = classifier.predict(X_test)
        predict_seconds = time.time() - pred_start
        accuracy = float(_np().mean(_np().asarray(y_pred) == _np().asarray(y_test)))

        return (
            EnsembleContextResultRow(
                dataset_name=dataset_dir.name,
                dataset_dir=dataset_dir.as_posix(),
                method=METHOD_BASELINE,
                context_length=int(context_length),
                group_count=int(group_count),
                ttt_iters_per_group=int(ttt_iters_per_group),
                k_train=int(k_train),
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
    finally:
        ens.release_classifier_resources(classifier)


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
    adapt_indices,
    context_indices,
    base_ttt_config: ens.TTTConfig,
    max_classes: Optional[int],
    model_name: str,
    context_length: int,
    group_count: int,
    ttt_iters_per_group: int,
    k_train: int,
) -> EnsembleContextResultRow:
    classifier = None
    n_classes = int(len(_pd().unique(_pd().Series(_np().asarray(y_train)))))
    common = {
        "n_train": int(len(y_train)),
        "n_val": int(val_count),
        "n_test": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
        "n_classes": n_classes,
    }
    if max_classes is not None and n_classes > int(max_classes):
        return _empty_row(
            dataset_dir,
            method=METHOD_TTT,
            context_length=context_length,
            group_count=group_count,
            ttt_iters_per_group=ttt_iters_per_group,
            k_train=k_train,
            task_type=task_type,
            status="skip",
            error=f"Skipped because n_classes={n_classes} exceeds model max_classes={max_classes}",
            **common,
        )

    try:
        classifier = classifier_cls(**dict(model_kwargs))
        X_context = ens.take_rows(X_train, context_indices)
        y_context = _np().asarray(y_train)[context_indices]
        ttt_steps = int(group_count) * int(ttt_iters_per_group)
        ttt_config = replace(base_ttt_config, steps=ttt_steps)

        ttt_loss = None
        ttt_applied = False
        ttt_update_seconds = 0.0
        ttt_split_strategy = None
        ttt_split_reason = None
        n_train_b = 0
        n_holdout_c = 0

        fit_start = time.time()
        if ttt_config.enabled:
            if ens.should_skip_ttt_for_dataset(dataset_dir, info):
                ttt_split_reason = "TTT skipped for dataset=volkert to avoid OOM"
                classifier.fit(X_context, y_context)
                ttt_steps = 0
            else:
                X_adapt = ens.take_rows(X_train, adapt_indices)
                y_adapt = _np().asarray(y_train)[adapt_indices]
                split = ens.split_ttt_holdout(y_adapt, ttt_config)
                ttt_split_strategy = split.strategy
                ttt_split_reason = split.reason
                n_train_b = int(len(split.b_indices))
                n_holdout_c = int(len(split.c_indices))

                if n_train_b > 0 and n_holdout_c > 0:
                    X_b = ens.take_rows(X_adapt, split.b_indices)
                    y_b = _np().asarray(y_adapt)[split.b_indices]
                    X_c = ens.take_rows(X_adapt, split.c_indices)
                    y_c = _np().asarray(y_adapt)[split.c_indices]
                    ttt_result = ens.run_ttt_holdout_update(
                        classifier,
                        X_b,
                        y_b,
                        X_c,
                        y_c,
                        ttt_config,
                        model_name=model_name,
                        dataset_name=f"{dataset_dir.name}_n{context_length}",
                    )
                else:
                    ttt_result = ens.TTTUpdateResult(
                        applied=False,
                        loss=None,
                        steps=0,
                        update_seconds=0.0,
                        reason="B/C split produced an empty side; skipped TTT",
                    )

                ttt_loss = ttt_result.loss
                ttt_steps = ttt_result.steps
                ttt_applied = ttt_result.applied
                ttt_update_seconds = float(ttt_result.update_seconds)
                if ttt_result.reason:
                    ttt_split_reason = f"{ttt_split_reason} | {ttt_result.reason}"

                if ttt_applied:
                    ens._fit_preserving_model_weights(classifier, X_context, y_context)
                else:
                    classifier.fit(X_context, y_context)
        else:
            classifier.fit(X_context, y_context)
            ttt_steps = 0
        fit_seconds = time.time() - fit_start

        fitted_classes = int(len(classifier.classes_))
        fitted_max_classes = int(getattr(classifier.model_, "max_classes", fitted_classes))
        if fitted_classes > fitted_max_classes:
            return _empty_row(
                dataset_dir,
                method=METHOD_TTT,
                context_length=context_length,
                group_count=group_count,
                ttt_iters_per_group=ttt_iters_per_group,
                k_train=k_train,
                task_type=task_type,
                status="skip",
                error=(
                    "Skipped because fitted context classes exceed model max_classes: "
                    f"n_classes={fitted_classes}, max_classes={fitted_max_classes}"
                ),
                **{**common, "n_classes": fitted_classes},
            )

        pred_start = time.time()
        y_pred = classifier.predict(X_test)
        predict_seconds = time.time() - pred_start
        accuracy = float(_np().mean(_np().asarray(y_pred) == _np().asarray(y_test)))

        return EnsembleContextResultRow(
            dataset_name=dataset_dir.name,
            dataset_dir=dataset_dir.as_posix(),
            method=METHOD_TTT,
            context_length=int(context_length),
            group_count=int(group_count),
            ttt_iters_per_group=int(ttt_iters_per_group),
            k_train=int(k_train),
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
            ttt_lr=ttt_config.lr if ttt_config.enabled else None,
            ttt_applied=ttt_applied,
            ttt_update_seconds=ttt_update_seconds,
            ttt_split_strategy=ttt_split_strategy,
            ttt_split_reason=ttt_split_reason,
            **common,
        )
    finally:
        ens.release_classifier_resources(classifier)


def evaluate_one_dataset(
    classifier_cls,
    model_kwargs: Dict[str, object],
    dataset_dir: Path,
    base_ttt_config: ens.TTTConfig,
    context_lengths: List[int],
    random_state: int,
    k_train: int,
    min_total_samples: int,
    max_features: Optional[int],
    ttt_iters_per_group: int,
) -> List[EnsembleContextResultRow]:
    task_type: Optional[str] = None
    try:
        info = ens.load_dataset_info(dataset_dir)
        task_type = str(info.get("task_type", "")).lower() if info else None
        if task_type not in ens.CLASSIFICATION_TASKS:
            reason = f"Skipped due to task_type={task_type!r}"
            return [
                _empty_row(
                    dataset_dir,
                    method=method,
                    context_length=context_length,
                    group_count=int(math.ceil(float(k_train) / float(context_length))),
                    ttt_iters_per_group=ttt_iters_per_group,
                    k_train=k_train,
                    task_type=task_type,
                    status="skip",
                    error=reason,
                )
                for context_length in context_lengths
                for method in (METHOD_BASELINE, METHOD_TTT)
            ]

        X_train, y_train, val_count, X_test, y_test = load_train_val_test(dataset_dir)
        X_train, y_train, X_test, y_test, _top_classes, top_class_skip_reason = filter_top_classes(
            X_train,
            y_train,
            X_test,
            y_test,
            max_classes=10,
        )
        X_train, X_test = cap_features(
            X_train,
            X_test,
            max_features=max_features,
            random_state=int(random_state) + 17,
        )

        n_total = int(len(y_train) + len(y_test))
        n_classes = int(len(_pd().unique(_pd().Series(_np().asarray(y_train)))))
        common = {
            "n_train": int(len(y_train)),
            "n_val": int(val_count),
            "n_test": int(len(y_test)),
            "n_features": int(X_train.shape[1]),
            "n_classes": n_classes,
        }
        if top_class_skip_reason is not None:
            return [
                _empty_row(
                    dataset_dir,
                    method=method,
                    context_length=context_length,
                    group_count=int(math.ceil(float(k_train) / float(context_length))),
                    ttt_iters_per_group=ttt_iters_per_group,
                    k_train=k_train,
                    task_type=task_type,
                    status="skip",
                    error=top_class_skip_reason,
                    **common,
                )
                for context_length in context_lengths
                for method in (METHOD_BASELINE, METHOD_TTT)
            ]
        if n_total < int(min_total_samples):
            reason = f"Skipped because total samples={n_total} < min_total_samples={min_total_samples}"
            return [
                _empty_row(
                    dataset_dir,
                    method=method,
                    context_length=context_length,
                    group_count=int(math.ceil(float(k_train) / float(context_length))),
                    ttt_iters_per_group=ttt_iters_per_group,
                    k_train=k_train,
                    task_type=task_type,
                    status="skip",
                    error=reason,
                    **common,
                )
                for context_length in context_lengths
                for method in (METHOD_BASELINE, METHOD_TTT)
            ]
        if len(y_train) < int(k_train):
            reason = f"Skipped because train+val samples={len(y_train)} < k_train={k_train}"
            return [
                _empty_row(
                    dataset_dir,
                    method=method,
                    context_length=context_length,
                    group_count=int(math.ceil(float(k_train) / float(context_length))),
                    ttt_iters_per_group=ttt_iters_per_group,
                    k_train=k_train,
                    task_type=task_type,
                    status="skip",
                    error=reason,
                    **common,
                )
                for context_length in context_lengths
                for method in (METHOD_BASELINE, METHOD_TTT)
            ]

        adapt_indices = class_covering_indices(
            y_train,
            int(k_train),
            int(random_state),
        )
        model_name = ens._derive_model_name(
            str(model_kwargs.get("model_path")) if model_kwargs.get("model_path") is not None else None,
            str(model_kwargs.get("checkpoint_version"))
            if model_kwargs.get("checkpoint_version") is not None
            else None,
        )

        rows: list[EnsembleContextResultRow] = []
        for context_length in context_lengths:
            group_count = int(math.ceil(float(k_train) / float(context_length)))
            try:
                context_indices = class_covering_indices(
                    y_train,
                    int(context_length),
                    int(random_state) + int(context_length),
                    candidate_indices=adapt_indices,
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                rows.extend(
                    [
                        _empty_row(
                            dataset_dir,
                            method=method,
                            context_length=context_length,
                            group_count=group_count,
                            ttt_iters_per_group=ttt_iters_per_group,
                            k_train=k_train,
                            task_type=task_type,
                            status="skip",
                            error=reason,
                            **common,
                        )
                        for method in (METHOD_BASELINE, METHOD_TTT)
                    ]
                )
                continue

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
                    group_count,
                    ttt_iters_per_group,
                    k_train,
                )
            except Exception as exc:
                rows.extend(
                    [
                        _empty_row(
                            dataset_dir,
                            method=method,
                            context_length=context_length,
                            group_count=group_count,
                            ttt_iters_per_group=ttt_iters_per_group,
                            k_train=k_train,
                            task_type=task_type,
                            status="fail",
                            error=f"{type(exc).__name__}: {exc}",
                            **common,
                        )
                        for method in (METHOD_BASELINE, METHOD_TTT)
                    ]
                )
                continue

            rows.append(baseline_row)
            if baseline_row.status != "ok":
                rows.append(
                    _empty_row(
                        dataset_dir,
                        method=METHOD_TTT,
                        context_length=context_length,
                        group_count=group_count,
                        ttt_iters_per_group=ttt_iters_per_group,
                        k_train=k_train,
                        task_type=task_type,
                        status="skip",
                        error=baseline_row.error or "Skipped because baseline row was skipped",
                        **common,
                    )
                )
                continue

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
                    adapt_indices,
                    context_indices,
                    base_ttt_config,
                    max_classes,
                    model_name,
                    context_length,
                    group_count,
                    ttt_iters_per_group,
                    k_train,
                )
            except Exception as exc:
                ttt_row = _empty_row(
                    dataset_dir,
                    method=METHOD_TTT,
                    context_length=context_length,
                    group_count=group_count,
                    ttt_iters_per_group=ttt_iters_per_group,
                    k_train=k_train,
                    task_type=task_type,
                    status="fail",
                    error=f"{type(exc).__name__}: {exc}",
                    **common,
                )
            rows.append(ttt_row)
        return rows
    except Exception as exc:
        return _fail_pair(
            dataset_dir,
            exc,
            context_lengths,
            {
                "k_train": k_train,
                "ttt_iters_per_group": ttt_iters_per_group,
            },
        )


def format_context_log(worker_label: str, row: EnsembleContextResultRow) -> str:
    if row.status == "ok":
        return (
            f"[{worker_label}] [ok] {row.dataset_name} method={row.method} "
            f"n={row.context_length} acc={ens.format_optional_float(row.accuracy)} "
            f"fit={row.fit_seconds:.3f}s pred={row.predict_seconds:.3f}s "
            f"groups={row.group_count} ttt_steps={row.ttt_steps} "
            f"ttt_applied={row.ttt_applied} ttt_loss={ens.format_optional_float(row.ttt_loss)}"
        )
    return (
        f"[{worker_label}] [{row.status}] {row.dataset_name} method={row.method} "
        f"n={row.context_length} reason={row.error}"
    )


def worker_main(
    worker_id: int,
    gpu_id: int,
    gpu_group: str,
    assigned_dataset_dirs: List[str],
    ready_queue,
    start_event,
    worker_out_csv: str,
    model_kwargs: Dict[str, object],
    ttt_config: ens.TTTConfig,
    context_lengths: List[int],
    random_state: int,
    k_train: int,
    min_total_samples: int,
    max_features: Optional[int],
    ttt_iters_per_group: int,
    verbose: bool,
) -> None:
    try:
        ensure_runtime_deps()
        device_str = ens.apply_worker_environment_updates(gpu_group)
        worker_label = f"worker {worker_id} | gpu {gpu_group}"
        worker_ttt_config = replace(ttt_config, gpu_group=gpu_group)

        import torch
        from tabicl import TabICLClassifier

        torch_diag = ens.collect_torch_diagnostics()
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

        rows: list[EnsembleContextResultRow] = []
        for dataset_dir in assigned_dataset_dirs:
            dataset_rows = evaluate_one_dataset(
                TabICLClassifier,
                worker_kwargs,
                Path(dataset_dir),
                worker_ttt_config,
                context_lengths,
                random_state,
                k_train,
                min_total_samples,
                max_features,
                ttt_iters_per_group,
            )
            rows.extend(dataset_rows)
            for row in dataset_rows:
                print(format_context_log(worker_label, row), flush=True)
            ens.force_memory_cleanup(device_str)

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
        crash_rows = []
        for context_length in context_lengths:
            group_count = int(math.ceil(float(k_train) / float(context_length)))
            for method in (METHOD_BASELINE, METHOD_TTT):
                crash_rows.append(
                    EnsembleContextResultRow(
                        dataset_name=f"__WORKER_CRASH__{worker_id}",
                        dataset_dir="__worker__",
                        method=method,
                        context_length=int(context_length),
                        group_count=group_count,
                        ttt_iters_per_group=int(ttt_iters_per_group),
                        k_train=int(k_train),
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
        _pd().DataFrame([asdict(row) for row in crash_rows]).to_csv(worker_out_csv, index=False)


def build_paired_delta(result_df):
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else _pd().DataFrame()
    baseline_ok = ok_df[ok_df["method"] == METHOD_BASELINE].copy() if len(ok_df) else _pd().DataFrame()
    ttt_ok = ok_df[ok_df["method"] == METHOD_TTT].copy() if len(ok_df) else _pd().DataFrame()
    if not len(baseline_ok) or not len(ttt_ok):
        return _pd().DataFrame(
            columns=[
                "dataset_name",
                "context_length",
                "task_type",
                "accuracy_baseline",
                "accuracy_ttt",
                "delta_ttt_minus_baseline",
                "delta_pp",
            ]
        )

    paired = baseline_ok[
        ["dataset_name", "context_length", "task_type", "accuracy", "fit_seconds", "predict_seconds"]
    ].merge(
        ttt_ok[["dataset_name", "context_length", "accuracy", "fit_seconds", "predict_seconds", "ttt_update_seconds"]],
        on=["dataset_name", "context_length"],
        suffixes=("_baseline", "_ttt"),
    )
    paired["delta_ttt_minus_baseline"] = paired["accuracy_ttt"] - paired["accuracy_baseline"]
    paired["delta_pp"] = paired["delta_ttt_minus_baseline"] * 100.0
    return paired.sort_values(["context_length", "delta_ttt_minus_baseline", "dataset_name"], ascending=[True, False, True])


def build_context_summary(paired_df):
    rows: list[dict[str, Any]] = []
    if not len(paired_df):
        return _pd().DataFrame(
            columns=[
                "context_length",
                "method",
                "mean_accuracy",
                "n_common_datasets",
                "mean_delta_ttt_minus_baseline",
                "median_delta_ttt_minus_baseline",
                "ttt_win",
                "baseline_win",
                "tie",
            ]
        )

    for context_length, group in paired_df.groupby("context_length", sort=True):
        mean_delta = float(group["delta_ttt_minus_baseline"].mean())
        median_delta = float(group["delta_ttt_minus_baseline"].median())
        ttt_win = int((group["delta_ttt_minus_baseline"] > 1e-12).sum())
        baseline_win = int((group["delta_ttt_minus_baseline"] < -1e-12).sum())
        tie = int((group["delta_ttt_minus_baseline"].abs() <= 1e-12).sum())
        common = int(len(group))
        rows.append(
            {
                "context_length": int(context_length),
                "method": METHOD_BASELINE,
                "mean_accuracy": float(group["accuracy_baseline"].mean()),
                "n_common_datasets": common,
                "mean_delta_ttt_minus_baseline": mean_delta,
                "median_delta_ttt_minus_baseline": median_delta,
                "ttt_win": ttt_win,
                "baseline_win": baseline_win,
                "tie": tie,
            }
        )
        rows.append(
            {
                "context_length": int(context_length),
                "method": METHOD_TTT,
                "mean_accuracy": float(group["accuracy_ttt"].mean()),
                "n_common_datasets": common,
                "mean_delta_ttt_minus_baseline": mean_delta,
                "median_delta_ttt_minus_baseline": median_delta,
                "ttt_win": ttt_win,
                "baseline_win": baseline_win,
                "tie": tie,
            }
        )
    return _pd().DataFrame(rows)


def _png_pack(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack("!I", len(data))
        + tag
        + data
        + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _write_rgb_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * stride : (y + 1) * stride])
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_pack(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_pack(b"IDAT", zlib.compress(bytes(raw), level=6))
        + _png_pack(b"IEND", b"")
    )
    path.write_bytes(png)


def _set_pixel(pixels: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    idx = (y * width + x) * 3
    pixels[idx : idx + 3] = bytes(color)


def _draw_line(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    radius = max(0, thickness // 2)
    while True:
        for ox in range(-radius, radius + 1):
            for oy in range(-radius, radius + 1):
                _set_pixel(pixels, width, height, x + ox, y + oy, color)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _draw_filled_circle(
    pixels: bytearray,
    width: int,
    height: int,
    cx: int,
    cy: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    rr = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= rr:
                _set_pixel(pixels, width, height, x, y, color)


def _fallback_accuracy_plot(summary_df, plot_path: Path, *, log_x: bool) -> None:
    width, height = 960, 620
    left, right, top, bottom = 90, 40, 45, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    pixels = bytearray([255, 255, 255] * width * height)

    axis = (35, 35, 35)
    grid = (220, 220, 220)
    baseline_color = (31, 119, 180)
    ttt_color = (255, 127, 14)

    x_values = sorted(set(int(x) for x in summary_df["context_length"].dropna().tolist()))
    if not x_values:
        _write_rgb_png(plot_path, width, height, pixels)
        return

    min_x = min(x_values)
    max_x = max(x_values)
    if log_x:
        min_x_plot = math.log10(max(1, min_x))
        max_x_plot = math.log10(max(1, max_x))
    else:
        min_x_plot = float(min_x)
        max_x_plot = float(max_x)
    if max_x_plot == min_x_plot:
        max_x_plot += 1.0

    def sx(value: float) -> int:
        value = math.log10(max(1.0, value)) if log_x else value
        return int(left + ((value - min_x_plot) / (max_x_plot - min_x_plot)) * plot_w)

    def sy(value: float) -> int:
        value = max(0.0, min(1.0, float(value)))
        return int(top + (1.0 - value) * plot_h)

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = int(top + (1.0 - frac) * plot_h)
        _draw_line(pixels, width, height, left, y, width - right, y, grid, 1)
    for x in x_values:
        px = sx(float(x))
        _draw_line(pixels, width, height, px, top, px, height - bottom, grid, 1)

    _draw_line(pixels, width, height, left, top, left, height - bottom, axis, 2)
    _draw_line(pixels, width, height, left, height - bottom, width - right, height - bottom, axis, 2)

    for method, color in ((METHOD_BASELINE, baseline_color), (METHOD_TTT, ttt_color)):
        method_df = summary_df[summary_df["method"] == method].sort_values("context_length")
        points = [
            (sx(float(row["context_length"])), sy(float(row["mean_accuracy"])))
            for _, row in method_df.iterrows()
            if row["mean_accuracy"] == row["mean_accuracy"]
        ]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            _draw_line(pixels, width, height, x0, y0, x1, y1, color, 3)
        for x, y in points:
            _draw_filled_circle(pixels, width, height, x, y, 5, color)

    # Minimal legend without text, used only when matplotlib is unavailable.
    for idx, color in enumerate((baseline_color, ttt_color)):
        y = top + 18 + idx * 18
        _draw_line(pixels, width, height, width - right - 100, y, width - right - 50, y, color, 4)
        _draw_filled_circle(pixels, width, height, width - right - 75, y, 5, color)

    _write_rgb_png(plot_path, width, height, pixels)


def write_accuracy_plot(summary_df, plot_path: Path, *, log_x: bool) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _fallback_accuracy_plot(summary_df, plot_path, log_x=log_x)
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    label_map = {METHOD_BASELINE: "TabICL", METHOD_TTT: "TabICL + TTT"}
    color_map = {METHOD_BASELINE: "#1f77b4", METHOD_TTT: "#ff7f0e"}

    for method in (METHOD_BASELINE, METHOD_TTT):
        method_df = summary_df[summary_df["method"] == method].sort_values("context_length")
        if not len(method_df):
            continue
        ax.plot(
            method_df["context_length"],
            method_df["mean_accuracy"],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=label_map[method],
            color=color_map[method],
        )

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("n (Context length)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.35)
    ax.legend()
    ax.set_title("TabICL context-length evaluation on data178")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)


def write_outputs(
    out_dir: Path,
    result_df,
    dataset_dirs: List[Path],
    wall_seconds: float,
    args: argparse.Namespace,
    command: str,
) -> None:
    all_csv = out_dir / "all_context_results.csv"
    delta_csv = out_dir / "per_dataset_delta.csv"
    summary_csv = out_dir / "context_summary.csv"
    summary_txt = out_dir / "summary.txt"

    result_df.to_csv(all_csv, index=False)
    paired_df = build_paired_delta(result_df)
    paired_df.to_csv(delta_csv, index=False)
    context_summary = build_context_summary(paired_df)
    context_summary.to_csv(summary_csv, index=False)

    plot_errors: list[str] = []
    if len(context_summary):
        try:
            write_accuracy_plot(context_summary, out_dir / "accuracy_vs_context.png", log_x=False)
        except Exception as exc:
            plot_errors.append(f"accuracy_vs_context.png: {type(exc).__name__}: {exc}")
        try:
            write_accuracy_plot(context_summary, out_dir / "accuracy_vs_context_logx.png", log_x=True)
        except Exception as exc:
            plot_errors.append(f"accuracy_vs_context_logx.png: {type(exc).__name__}: {exc}")

    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else _pd().DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else _pd().DataFrame()
    skipped_df = result_df[result_df["status"] == "skip"].copy() if len(result_df) else _pd().DataFrame()
    baseline_ok = ok_df[ok_df["method"] == METHOD_BASELINE].copy() if len(ok_df) else _pd().DataFrame()
    ttt_ok = ok_df[ok_df["method"] == METHOD_TTT].copy() if len(ok_df) else _pd().DataFrame()

    lines = [
        "experiment: paper-inspired TabICL/TTT context-length evaluation on data178",
        f"command: {command}",
        f"discovered_datasets: {len(dataset_dirs)}",
        f"processed_rows: {len(result_df)}",
        f"ok_rows: {len(ok_df)}",
        f"failed_rows: {len(failed_df)}",
        f"skipped_rows: {len(skipped_df)}",
        f"context_lengths: {','.join(str(x) for x in parse_context_lengths(args.context_lengths))}",
        f"k_train: {int(args.k_train)}",
        f"min_total_samples: {int(args.min_total_samples)}",
        f"max_features: {args.max_features}",
        f"ttt_iters_per_group: {int(args.ttt_iters_per_group)}",
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
        f"shared_ok_dataset_context_pairs: {len(paired_df)}",
        (
            f"shared_delta_ttt_minus_baseline: {paired_df['delta_ttt_minus_baseline'].mean():.6f}"
            if len(paired_df)
            else "shared_delta_ttt_minus_baseline: (none)"
        ),
        f"wall_seconds: {wall_seconds:.3f}",
        f"saved_all_context_results: {all_csv}",
        f"saved_context_summary: {summary_csv}",
        f"saved_per_dataset_delta: {delta_csv}",
    ]
    if (out_dir / "accuracy_vs_context.png").exists():
        lines.append(f"saved_accuracy_plot: {out_dir / 'accuracy_vs_context.png'}")
    if (out_dir / "accuracy_vs_context_logx.png").exists():
        lines.append(f"saved_accuracy_logx_plot: {out_dir / 'accuracy_vs_context_logx.png'}")
    if plot_errors:
        lines.append("plot_errors: " + " | ".join(plot_errors))
    if len(failed_df):
        failed_names = sorted(set(failed_df["dataset_name"].astype(str).tolist()))
        lines.append("failed_datasets: " + ", ".join(failed_names))
    else:
        lines.append("failed_datasets: (none)")

    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate TabICL baseline vs compatible B/C TTT across context lengths."
    )
    parser.add_argument("--data-root", default=str(ens.DEFAULT_DATA_ROOT))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--checkpoint-version", default=ens.DEFAULT_CHECKPOINT_VERSION)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--gpu-groups", default=None)
    parser.add_argument("--context-lengths", default=DEFAULT_CONTEXT_LENGTHS)
    parser.add_argument("--k-train", type=int, default=1000)
    parser.add_argument("--min-total-samples", type=int, default=1250)
    parser.add_argument("--max-features", type=ens.parse_optional_int, default=100)
    parser.add_argument("--ttt-iters-per-group", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=32)
    parser.add_argument("--batch-size", type=ens.parse_optional_int, default=16)
    parser.add_argument("--kv-cache", type=ens.parse_kv_cache, default=False)
    parser.add_argument("--use-amp", type=ens.parse_auto_bool, default="auto")
    parser.add_argument("--use-fa3", type=ens.parse_auto_bool, default="auto")
    parser.add_argument("--offload-mode", choices=["auto", "gpu", "cpu", "disk"], default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--ttt-holdout", type=ens.parse_bool, nargs="?", const="true", default=True)
    parser.add_argument("--no-ttt-holdout", dest="ttt_holdout", action="store_false")
    parser.add_argument("--ttt-lr", type=float, default=5e-6)
    parser.add_argument("--ttt-scheduler", choices=["constant"], default="constant")
    parser.add_argument("--ttt-grad-clip", type=float, default=1.0)
    parser.add_argument("--ttt-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--ttt-micro-batch-size", type=int, default=16)
    parser.add_argument("--ttt-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--ttt-steps",
        type=int,
        default=1,
        help=(
            "Base value required by the shared TTT config builder. The script "
            "overrides it per context length as ceil(k_train / n) * ttt_iters_per_group."
        ),
    )
    parser.add_argument("--ttt-freeze-col", type=ens.parse_bool, default=True)
    parser.add_argument("--ttt-freeze-row", type=ens.parse_bool, default=True)
    parser.add_argument("--ttt-save-ckpt", type=ens.parse_bool, default=False)
    parser.add_argument("--ttt-save-ckpt-every", type=int, default=50)
    parser.add_argument("--ttt-data-parallel", dest="ttt_data_parallel", action="store_true")
    parser.add_argument("--no-ttt-data-parallel", dest="ttt_data_parallel", action="store_false")
    parser.set_defaults(ttt_data_parallel=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def build_common_model_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "n_estimators": args.n_estimators,
        "batch_size": args.batch_size,
        "kv_cache": args.kv_cache,
        "allow_auto_download": True,
        "checkpoint_version": args.checkpoint_version,
        "model_path": ens.normalize_model_path(args.model_path or ens.DEFAULT_MODEL_PATH),
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
    context_lengths: List[int],
) -> None:
    model_kwargs = build_common_model_kwargs(args)
    ttt_config = ens.build_ttt_config(args)
    start_time = time.time()
    ready_queue: mp.Queue = mp.Queue()
    start_event = mp.Event()

    worker_csv_paths: list[Path] = []
    processes: list[mp.Process] = []
    for worker_id in range(args.workers):
        assigned_dirs = [str(path.resolve()) for path in dataset_dirs[worker_id:: args.workers]]
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
                list(context_lengths),
                int(args.random_state),
                int(args.k_train),
                int(args.min_total_samples),
                args.max_features,
                int(args.ttt_iters_per_group),
                bool(args.verbose),
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

    dfs: list[Any] = []
    for worker_csv in worker_csv_paths:
        if worker_csv.exists():
            dfs.append(_pd().read_csv(worker_csv))

    all_df = (
        _pd().concat(dfs, ignore_index=True)
        if dfs
        else _pd().DataFrame(columns=EnsembleContextResultRow.__annotations__.keys())
    )
    write_outputs(
        out_dir,
        all_df,
        dataset_dirs,
        time.time() - start_time,
        args,
        " ".join([sys.executable, *sys.argv]),
    )

    print(f"saved_all_context_results: {out_dir / 'all_context_results.csv'}")
    print(f"saved_context_summary: {out_dir / 'context_summary.csv'}")
    print(f"saved_per_dataset_delta: {out_dir / 'per_dataset_delta.csv'}")
    print(f"saved_summary: {out_dir / 'summary.txt'}")
    for plot_name in ("accuracy_vs_context.png", "accuracy_vs_context_logx.png"):
        plot_path = out_dir / plot_name
        if plot_path.exists():
            print(f"saved_plot: {plot_path}")
    print("model_kwargs:")
    print(json.dumps(model_kwargs, indent=2, ensure_ascii=False))
    if ttt_config.enabled:
        print("base_ttt_config:")
        print(json.dumps(asdict(ttt_config), indent=2, ensure_ascii=False))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    context_lengths = parse_context_lengths(args.context_lengths)
    if int(args.k_train) < 1:
        raise ValueError("--k-train must be >= 1")
    if int(args.min_total_samples) < 1:
        raise ValueError("--min-total-samples must be >= 1")
    if int(args.ttt_iters_per_group) < 1:
        raise ValueError("--ttt-iters-per-group must be >= 1")
    if int(args.ttt_save_ckpt_every) < 1:
        raise ValueError("--ttt-save-ckpt-every must be >= 1")

    ensure_runtime_deps()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = ens.find_dataset_dirs(data_root)
    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: args.max_datasets]
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset directories found under {data_root}")

    gpu_ids, gpu_groups = ens.resolve_gpu_assignments(args)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    run_context_eval(args, dataset_dirs, gpu_ids, gpu_groups, out_dir, context_lengths)


if __name__ == "__main__":
    main()
