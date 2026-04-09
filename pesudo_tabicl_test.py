#!/usr/bin/env python3
"""
批量在 TALENT 数据目录上并行评测 TabICLClassifier。

该版本采用 correctness-only 伪标签逻辑：
 - 第一轮在原始 train 上训练，并直接预测原始 test。
 - 仅将预测正确的测试样本及其预测标签并入训练集。
 - 第二轮仅在剩余预测错误的测试样本上重新训练和评测。
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing
import queue
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd

import ce_pesudo as base


DEFAULT_DATA_ROOT = base.DEFAULT_DATA_ROOT
PSEUDO_ACC_DELTA_TOL = base.PSEUDO_ACC_DELTA_TOL
CLASSIFICATION_TASKS = base.CLASSIFICATION_TASKS
PSEUDO_RULE = "correct_prediction_only"


def run_correct_prediction_resplit_inference(
    *,
    fit_classifier_fn,
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: np.ndarray,
    dataset_name: str,
) -> dict[str, Any]:
    """
    两阶段 correctness-only 伪标签推理。

    1. 原始 train 上训练，并在原始 test 上直接 predict。
    2. 仅把预测正确的测试样本加入伪标签集合。
    3. 将这些样本并入训练集，剩余预测错误的样本作为新 test，再次 fit/predict。

    若第一轮没有正确样本，或第一轮已把测试集全部预测正确，则直接返回第一轮结果。
    最终返回的 y_pred/y_eval 始终对齐原始整个测试集，因此准确率表示
    “第一轮已确定正确的样本 + 第二轮剩余样本”的总准确率。
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    train_ratio_before = base.compute_train_ratio(y_train, y_test)

    classifier, first_fit_time = fit_classifier_fn(X_train, y_train)

    predict_start = time.perf_counter()
    baseline_pred = np.asarray(classifier.predict(X_test))
    first_predict_time = time.perf_counter() - predict_start
    baseline_accuracy = float(np.mean(baseline_pred == y_test))

    correct_mask = np.asarray(baseline_pred == y_test)
    pseudo_added = int(correct_mask.sum())

    result: dict[str, Any] = {
        "baseline_accuracy": baseline_accuracy,
        "fit_time": first_fit_time,
        "predict_time": first_predict_time,
        "pseudo_added": pseudo_added,
        "pseudo_applied": False,
        "pseudo_rule": PSEUDO_RULE,
        "train_ratio_before": train_ratio_before,
        "train_ratio_after": train_ratio_before,
        "y_pred": baseline_pred,
        "y_eval": y_test,
    }

    if pseudo_added == 0 or pseudo_added == len(y_test):
        return result

    X_correct = base.slice_rows(X_test, correct_mask)
    X_remaining = base.slice_rows(X_test, ~correct_mask)
    y_pseudo = baseline_pred[correct_mask]
    y_remaining_true = y_test[~correct_mask]

    X_labeled = base.concat_feature_blocks([X_train, X_correct])
    y_labeled_train = base.concat_target_blocks([y_train, y_pseudo])

    classifier, second_fit_time = fit_classifier_fn(X_labeled, y_labeled_train)
    predict_start = time.perf_counter()
    second_pred = np.asarray(classifier.predict(X_remaining))
    second_predict_time = time.perf_counter() - predict_start
    final_pred = baseline_pred.copy()
    final_pred[~correct_mask] = second_pred

    result.update(
        {
            "fit_time": first_fit_time + second_fit_time,
            "predict_time": first_predict_time + second_predict_time,
            "pseudo_applied": True,
            "train_ratio_after": base.compute_train_ratio(y_labeled_train, y_remaining_true),
            "y_pred": final_pred,
            "y_eval": y_test,
        }
    )
    return result


def evaluate_datasets_worker(
    rank: int,
    device_id: int,
    assigned_datasets: list[str],
    result_queue,
    classifier_config: dict[str, Any],
    *,
    merge_val: bool = True,
    coerce_numeric: bool = True,
    random_state: int = 42,
) -> None:
    results: list[dict[str, Any]] = []
    datasets_with_missing: set[str] = set()
    processed_count = 0

    msg_prefix = f"[Worker {rank}][GPU {device_id}]" if device_id >= 0 else f"[Worker {rank}][CPU]"
    device_str = f"cuda:{device_id}" if device_id >= 0 else "cpu"

    try:
        try:
            from tabicl import TabICLClassifier
            from sklearn.utils.multiclass import type_of_target
        except ImportError as exc:
            base.worker_log(msg_prefix, f"导入失败: {exc}")
            return

        active_config = dict(classifier_config)
        active_config["device"] = device_str

        base.worker_log(msg_prefix, f"启动成功，设备={device_str}")

        try:
            clf, fit_kwargs, notes = base.instantiate_classifier(TabICLClassifier, active_config)
            for note in notes:
                base.worker_log(msg_prefix, note)
        except Exception as exc:
            base.worker_log(msg_prefix, f"模型初始化失败: {exc}")
            return

        for dataset_dir in assigned_datasets:
            processed_count += 1
            dataset_dir = Path(dataset_dir)

            try:
                info = base.load_dataset_info(dataset_dir)
                task_type = str(info.get("task_type", "")).lower() if info else ""
                if task_type == "regression":
                    base.worker_log(msg_prefix, f"跳过 {dataset_dir.name}: task_type=regression")
                    continue
                if task_type and task_type not in CLASSIFICATION_TASKS:
                    base.worker_log(msg_prefix, f"跳过 {dataset_dir.name}: 未知 task_type={task_type}")
                    continue

                train_path, val_path, test_path = base.find_data_files(dataset_dir)
                if train_path is None and test_path is None:
                    base.worker_log(msg_prefix, f"跳过 {dataset_dir.name}: 无可识别数据文件")
                    continue

                if train_path and test_path:
                    X_train, y_train = base.load_table(
                        train_path,
                        context=f"{dataset_dir.name}-train",
                        coerce_numeric=coerce_numeric,
                        dataset_id=dataset_dir.name,
                        missing_registry=datasets_with_missing,
                    )
                    X_test, y_test = base.load_table(
                        test_path,
                        context=f"{dataset_dir.name}-test",
                        coerce_numeric=coerce_numeric,
                        dataset_id=dataset_dir.name,
                        missing_registry=datasets_with_missing,
                    )
                else:
                    X_all, y_all = base.load_table(
                        train_path,
                        context=f"{dataset_dir.name}-single",
                        coerce_numeric=coerce_numeric,
                        dataset_id=dataset_dir.name,
                        missing_registry=datasets_with_missing,
                    )
                    X_train, X_test, y_train, y_test = base.split_single_file_dataset(
                        X_all,
                        y_all,
                        dataset_name=dataset_dir.name,
                        random_state=random_state,
                    )
                    val_path = None
                    base.worker_log(msg_prefix, f"{dataset_dir.name}: 单文件数据按 80/20 自动切分")

                if val_path:
                    X_val, y_val = base.load_table(
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

                try:
                    target_type = type_of_target(y_train)
                except Exception:
                    target_type = None

                if (not task_type or task_type == "unknown") and target_type and target_type.startswith("continuous"):
                    base.worker_log(msg_prefix, f"跳过 {dataset_dir.name}: 检测到连续标签")
                    continue

                def fit_with_retry(train_X, train_y):
                    nonlocal clf, fit_kwargs
                    fit_start = time.perf_counter()
                    try:
                        clf.fit(train_X, train_y, **fit_kwargs)
                    except ValueError as exc:
                        clf, fit_kwargs = base.maybe_retry_with_checkpoint_alias(
                            classifier_cls=TabICLClassifier,
                            classifier_config=active_config,
                            fit_kwargs=fit_kwargs,
                            X_train=train_X,
                            y_train=train_y,
                            error=exc,
                            msg_prefix=msg_prefix,
                        )
                    return clf, time.perf_counter() - fit_start

                pseudo_result = run_correct_prediction_resplit_inference(
                    fit_classifier_fn=fit_with_retry,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    dataset_name=dataset_dir.name,
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

                acc = float(np.mean(np.asarray(y_pred) == np.asarray(y_eval)))
                base.worker_log(
                    msg_prefix,
                    (
                        f"{dataset_dir.name}: accuracy={acc:.4f}"
                        + f" baseline={baseline_acc:.4f}"
                        + f" fit={fit_time:.2f}s predict={predict_time:.2f}s total={total_time:.2f}s"
                        + f" pseudo_added={pseudo_added} pseudo_applied={pseudo_applied}"
                        + f" train_ratio={train_ratio_before:.4f}->{train_ratio_after:.4f}"
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
                        "pseudo_rule": PSEUDO_RULE,
                        "train_ratio_before": train_ratio_before,
                        "train_ratio_after": train_ratio_after,
                    }
                )

            except Exception as exc:
                base.worker_log(msg_prefix, f"评测失败 {dataset_dir.name}: {exc}")
                traceback.print_exc()

    except BaseException as exc:
        base.worker_log(msg_prefix, f"worker 异常退出: {exc.__class__.__name__}: {exc}")
        traceback.print_exc()
    finally:
        base.worker_log(msg_prefix, f"处理结束，成功数据集数: {len(results)} / 已领取任务数: {processed_count}")
        try:
            result_queue.put((rank, results, datasets_with_missing))
        except Exception:
            pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在 TALENT 数据集上并行评测 TabICLClassifier（correctness-only 伪标签版）")
    bool_action = argparse.BooleanOptionalAction

    parser.add_argument(
        "--model-path",
        default="tabicl-classifier-v1.1-20250506.ckpt",
        help="本地 checkpoint 路径；为 None 时按 checkpoint_version 下载或查找",
    )
    parser.add_argument(
        "--checkpoint-version",
        default="tabicl-classifier-v1.1-20250506.ckpt",
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
    parser.add_argument("--outdir", default="tabiclv1.1_pesudo_test_correct_only", help="结果输出目录")
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
        "--n-estimators",
        type=base.parse_optional_int,
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
    parser.add_argument("--batch-size", type=base.parse_optional_int, default=8, help="整数或 none")
    parser.add_argument("--kv-cache", type=base.parse_kv_cache, default=False)
    parser.add_argument("--use-amp", type=base.parse_auto_bool, default="auto")
    parser.add_argument("--use-fa3", type=base.parse_auto_bool, default="auto")
    parser.add_argument("--offload-mode", choices=["auto", "gpu", "cpu", "disk"], default="auto")
    parser.add_argument("--disk-offload-dir", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=base.parse_optional_int, default=None, help="整数或 none")
    parser.add_argument("--verbose", action="store_true", help="启用 TabICL 模型自身的 verbose 输出")

    parser.add_argument("--num-gpus", type=int, default=3, help="使用的 GPU 数量；默认使用 0..num_gpus-1")
    parser.add_argument("--gpu-ids", type=base.parse_gpu_ids, default="0,1,2", help="显式指定 GPU 编号，例如 0,2,3")
    return parser


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
    baseline_values = [float(item["baseline_accuracy"]) for item in results if item.get("baseline_accuracy") is not None]
    avg_baseline_acc = sum(baseline_values) / len(baseline_values) if baseline_values else None
    avg_train_ratio_before = sum(float(item.get("train_ratio_before", float("nan"))) for item in results) / len(results)
    avg_train_ratio_after = sum(float(item.get("train_ratio_after", float("nan"))) for item in results) / len(results)
    avg_pseudo_added = sum(int(item.get("pseudo_added", 0)) for item in results) / len(results)
    pseudo_applied_count = sum(1 for item in results if item.get("pseudo_applied"))
    pseudo_rule_value = next((str(item.get("pseudo_rule")) for item in results if item.get("pseudo_rule")), PSEUDO_RULE)
    delta_values = [
        float(item["accuracy"]) - float(item["baseline_accuracy"])
        for item in results
        if item.get("baseline_accuracy") is not None
    ]
    avg_pseudo_acc_gain = sum(delta_values) / len(delta_values) if delta_values else None
    improved_count = sum(1 for delta in delta_values if delta > PSEUDO_ACC_DELTA_TOL)
    degraded_count = sum(1 for delta in delta_values if delta < -PSEUDO_ACC_DELTA_TOL)
    unchanged_count = sum(1 for delta in delta_values if abs(delta) <= PSEUDO_ACC_DELTA_TOL)

    csv_path = outdir / "talent_detailed.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write(
                "dataset,accuracy,time_s,fit_time_s,predict_time_s,baseline_accuracy,"
                "pseudo_added,pseudo_applied,pseudo_rule,train_ratio_before,train_ratio_after\n"
            )
        for item in results:
            baseline_accuracy = item.get("baseline_accuracy")
            handle.write(
                "{dataset},{accuracy:.6f},{time_s:.3f},{fit_time_s:.3f},{predict_time_s:.3f},{baseline_accuracy},{pseudo_added},{pseudo_applied},{pseudo_rule},{train_ratio_before:.6f},{train_ratio_after:.6f}\n".format(
                    dataset=item["dataset"],
                    accuracy=float(item["accuracy"]),
                    time_s=float(item["time_s"]),
                    fit_time_s=float(item["fit_time_s"]),
                    predict_time_s=float(item["predict_time_s"]),
                    baseline_accuracy=(f"{float(baseline_accuracy):.6f}" if baseline_accuracy is not None else ""),
                    pseudo_added=int(item.get("pseudo_added", 0)),
                    pseudo_applied=str(bool(item.get("pseudo_applied", False))).lower(),
                    pseudo_rule=str(item.get("pseudo_rule", PSEUDO_RULE)),
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
        handle.write(f"Pseudo label rule: {pseudo_rule_value}\n")
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
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    model_family, classifier_config = base.classifier_config_from_args(args)
    if model_family == "regressor":
        raise ValueError("当前 bench 脚本只支持分类模型，请改用 classifier checkpoint")

    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"数据目录不是文件夹: {data_root}")

    dirs = [path for path in sorted(data_root.iterdir()) if path.is_dir()]
    if args.max_datasets:
        dirs = dirs[:args.max_datasets]
    if not dirs:
        logging.info("没有可处理的数据集目录。")
        return 0

    base.summarize_task_types(dirs)

    device_ids = base.resolve_device_ids(args)
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

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for rank, device_id in enumerate(worker_device_ids):
        assigned_datasets = [str(d) for d in dirs[rank::len(worker_device_ids)]]
        logging.info(
            "worker rank=%d -> gpu=%s: 固定分配 %d 个数据集",
            rank,
            device_id,
            len(assigned_datasets),
        )
        process = ctx.Process(
            target=evaluate_datasets_worker,
            args=(rank, device_id, assigned_datasets, result_queue, classifier_config),
            kwargs={
                "merge_val": args.merge_val,
                "coerce_numeric": args.coerce_numeric,
                "random_state": args.random_state,
            },
        )
        process.start()
        processes.append(process)

    results_by_worker: list[tuple[int, list[dict[str, Any]], set[str]]] = []
    pending_ranks = {rank for rank, _ in enumerate(worker_device_ids)}
    processes_by_rank = {rank: process for rank, process in enumerate(processes)}

    while pending_ranks:
        try:
            rank, result_items, missing_items = result_queue.get(timeout=5)
            if rank in pending_ranks:
                results_by_worker.append((rank, result_items, missing_items))
                pending_ranks.remove(rank)
        except queue.Empty:
            finished_ranks = {
                rank
                for rank in list(pending_ranks)
                if not processes_by_rank[rank].is_alive() and processes_by_rank[rank].exitcode is not None
            }
            pending_ranks -= finished_ranks

    for process in processes:
        process.join()

    results: list[dict[str, Any]] = []
    datasets_with_missing: set[str] = set()
    for _, result_items, missing_items in sorted(results_by_worker, key=lambda item: item[0]):
        results.extend(result_items)
        datasets_with_missing.update(missing_items)

    script_duration = time.perf_counter() - script_start_time
    write_results(
        outdir,
        results,
        datasets_with_missing=datasets_with_missing,
        script_duration=script_duration,
        model_family=model_family,
        classifier_config=classifier_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
