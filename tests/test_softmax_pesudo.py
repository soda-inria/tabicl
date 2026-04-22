from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import queue
import threading
import time

import numpy as np
import pandas as pd

import softmax_pesudo


class MappingPseudoClassifier:
    def __init__(self, proba_by_value: dict[int, list[float]], classes: np.ndarray | None = None):
        self.proba_by_value = {
            int(key): np.asarray(value, dtype=float)
            for key, value in proba_by_value.items()
        }
        self.classes_ = np.asarray(classes if classes is not None else [0, 1])
        self.fit_labels: list[np.ndarray] = []
        self.fit_feature_values: list[np.ndarray] = []

    def _extract_values(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            values = X.iloc[:, 0].to_numpy()
        else:
            values = np.asarray(X)[:, 0]
        return np.asarray(values, dtype=int)

    def fit(self, X, y):
        values = self._extract_values(X)
        self.fit_feature_values.append(values.copy())
        self.fit_labels.append(np.asarray(y).copy())
        return self

    def predict_proba(self, X):
        values = self._extract_values(X)
        return np.vstack([self.proba_by_value[int(value)] for value in values])

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def default_pseudo_kwargs(**overrides):
    base = {
        "pseudo_rounds": 3,
        "pseudo_base_confidence": 0.90,
        "pseudo_min_margin": 0.10,
        "pseudo_density_floor": 0.60,
        "pseudo_round_budget_ratio": 0.80,
        "pseudo_max_added_per_round": None,
        "pseudo_class_balance_power": 0.5,
        "pseudo_min_class_support": 2,
        "pseudo_stop_min_added": 1,
        "pseudo_artifact_dir": None,
    }
    base.update(overrides)
    return base


def test_run_hybrid_pseudo_label_inference_appends_incrementally_and_aligns_full_test():
    clf = MappingPseudoClassifier(
        {
            0: [0.99, 0.01],
            1: [0.01, 0.99],
            10: [0.94, 0.06],
            11: [0.06, 0.94],
            12: [0.85, 0.15],
        }
    )

    X_train = pd.DataFrame({"x": [0, 1]})
    y_train = np.array([0, 1])
    X_test = pd.DataFrame({"x": [10, 11, 12]})
    y_test = np.array([0, 1, 1])

    def fit_fn(X, y):
        clf.fit(X, y)
        return clf, 0.1

    result = softmax_pesudo.run_hybrid_pseudo_label_inference(
        fit_classifier_fn=fit_fn,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_name="demo",
        **default_pseudo_kwargs(),
    )

    assert result["pseudo_applied"] is True
    assert result["pseudo_selected"] == 2
    assert result["pseudo_rounds"] == 2
    np.testing.assert_array_equal(result["y_eval"], y_test)
    np.testing.assert_array_equal(result["y_pred"], np.array([0, 1, 0]))
    assert abs(result["acc_round1"] - (2 / 3)) < 1e-9
    assert abs(result["acc_round_final"] - (2 / 3)) < 1e-9
    assert len(clf.fit_labels) == 2
    np.testing.assert_array_equal(clf.fit_labels[0], np.array([0, 1]))
    np.testing.assert_array_equal(clf.fit_labels[1], np.array([0, 1, 0, 1]))

    round_meta = json.loads(result["pseudo_selected_by_round_json"])
    assert round_meta[0]["selected_count"] == 2
    assert round_meta[1]["stop_reason"] == "no_eligible_samples"


def test_compute_density_scores_falls_back_for_low_support_class():
    train_proba = np.array([[0.90, 0.10], [0.20, 0.80], [0.10, 0.90]])
    train_labels = np.array([0, 1, 1])
    candidate_proba = np.array([[0.85, 0.15], [0.15, 0.85]])
    candidate_pred_labels = np.array([0, 1])

    density = softmax_pesudo.compute_density_scores(
        train_proba=train_proba,
        train_labels=train_labels,
        candidate_proba=candidate_proba,
        candidate_pred_labels=candidate_pred_labels,
        min_class_support=2,
    )

    assert density[0] == 1.0
    assert 0.0 < density[1] <= 1.0


def test_allocate_classwise_quotas_reserves_budget_for_minority_class():
    quotas = softmax_pesudo.allocate_classwise_quotas(
        eligible_pred_labels=np.array([0, 0, 1]),
        eligible_scores=np.array([0.99, 0.98, 0.80]),
        current_train_labels=np.array([0] * 9 + [1]),
        round_budget=2,
        class_balance_power=1.0,
    )

    assert quotas == {0: 1, 1: 1}


def test_run_hybrid_pseudo_label_inference_rejects_high_confidence_sample_when_margin_too_small():
    clf = MappingPseudoClassifier(
        {
            0: [0.99, 0.01],
            1: [0.01, 0.99],
            10: [0.91, 0.09],
        }
    )

    X_train = pd.DataFrame({"x": [0, 1]})
    y_train = np.array([0, 1])
    X_test = pd.DataFrame({"x": [10]})
    y_test = np.array([0])

    def fit_fn(X, y):
        clf.fit(X, y)
        return clf, 0.05

    result = softmax_pesudo.run_hybrid_pseudo_label_inference(
        fit_classifier_fn=fit_fn,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_name="margin_gate",
        **default_pseudo_kwargs(pseudo_min_margin=0.95),
    )

    assert result["pseudo_applied"] is False
    assert result["pseudo_selected"] == 0
    assert result["pseudo_precision"] is None
    assert result["pseudo_error_rate"] is None


def test_run_hybrid_pseudo_label_inference_reports_precision_and_error_without_using_y_test_for_selection():
    clf = MappingPseudoClassifier(
        {
            0: [0.99, 0.01],
            1: [0.01, 0.99],
            10: [0.94, 0.06],
            11: [0.05, 0.95],
        }
    )

    X_train = pd.DataFrame({"x": [0, 1]})
    y_train = np.array([0, 1])
    X_test = pd.DataFrame({"x": [10, 11]})
    y_test = np.array([1, 1])

    def fit_fn(X, y):
        clf.fit(X, y)
        return clf, 0.1

    result = softmax_pesudo.run_hybrid_pseudo_label_inference(
        fit_classifier_fn=fit_fn,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_name="precision_demo",
        **default_pseudo_kwargs(pseudo_rounds=1),
    )

    assert result["pseudo_selected"] == 2
    assert result["pseudo_applied"] is True
    assert abs(result["pseudo_precision"] - 0.5) < 1e-9
    assert abs(result["pseudo_error_rate"] - 0.5) < 1e-9
    assert len(clf.fit_labels) == 2


def test_write_results_uses_new_fields_and_drops_legacy_alpha(tmp_path):
    outdir = tmp_path / "result"
    outdir.mkdir()
    results = [
        {
            "dataset": "alpha",
            "accuracy": 0.8,
            "time_s": 1.2,
            "fit_time_s": 0.7,
            "predict_time_s": 0.5,
            "baseline_accuracy": 0.6,
            "acc_round1": 0.6,
            "acc_round_final": 0.8,
            "acc_delta": 0.2,
            "pseudo_selected": 3,
            "pseudo_added": 3,
            "pseudo_applied": True,
            "pseudo_rounds": 2,
            "pseudo_selected_by_round_json": '[{"round":1,"selected_count":3}]',
            "pseudo_selected_by_class_json": '{"0":1,"1":2}',
            "pseudo_threshold_by_class_json": '{"1":{"0":0.91,"1":0.93}}',
            "pseudo_avg_max_prob": 0.94,
            "pseudo_avg_margin": 0.89,
            "pseudo_avg_density_score": 0.97,
            "pseudo_precision": 0.67,
            "pseudo_error_rate": 0.33,
            "pseudo_artifact_files_json": '["/tmp/alpha__round1.npz"]',
            "train_ratio_before": 0.67,
            "train_ratio_after": 0.8,
        }
    ]

    softmax_pesudo.write_results(
        outdir,
        results,
        datasets_with_missing=set(),
        script_duration=2.0,
        model_family="v1.1",
        classifier_config={"checkpoint_version": "demo.ckpt", "model_path": "demo.ckpt", "n_estimators": 32},
    )

    csv_text = (outdir / "talent_detailed.csv").read_text(encoding="utf-8")
    summary_text = (outdir / "talent_summary.txt").read_text(encoding="utf-8")
    assert "pseudo_selected_by_round_json" in csv_text
    assert "pseudo_threshold_by_class_json" in csv_text
    assert "pseudo_alpha" not in csv_text
    assert "Average acc_round1: 0.600000" in summary_text
    assert "Average acc_round_final: 0.800000" in summary_text
    assert "Average acc_delta: 0.200000" in summary_text
    assert "Average pseudo density score: 0.970000" in summary_text


def test_run_hybrid_pseudo_label_inference_writes_round_artifacts(tmp_path):
    clf = MappingPseudoClassifier(
        {
            0: [0.99, 0.01],
            1: [0.01, 0.99],
            10: [0.94, 0.06],
        }
    )

    X_train = pd.DataFrame({"x": [0, 1]})
    y_train = np.array([0, 1])
    X_test = pd.DataFrame({"x": [10]})
    y_test = np.array([0])

    def fit_fn(X, y):
        clf.fit(X, y)
        return clf, 0.1

    result = softmax_pesudo.run_hybrid_pseudo_label_inference(
        fit_classifier_fn=fit_fn,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_name="artifact_demo",
        **default_pseudo_kwargs(pseudo_rounds=1, pseudo_artifact_dir=tmp_path),
    )

    artifact_paths = json.loads(result["artifact_files_json"])
    assert len(artifact_paths) == 1
    artifact = np.load(artifact_paths[0])
    assert set(artifact.files) == {
        "round_idx",
        "max_prob",
        "margin",
        "density_score",
        "hybrid_score",
        "pred_label",
        "selected_mask",
        "selected_new_mask",
        "selected_threshold_by_class_json",
    }


def test_build_arg_parser_defaults_to_dynamic_scheduler():
    parser = softmax_pesudo.build_arg_parser()
    args = parser.parse_args([])
    assert args.scheduler == "dynamic"
    assert args.pseudo_method == "hybrid_bc"


def test_worker_environment_updates_sets_single_visible_gpu():
    updates = softmax_pesudo.worker_environment_updates(2)
    assert updates["CUDA_VISIBLE_DEVICES"] == "2"
    assert updates["OMP_NUM_THREADS"] == "1"
    assert updates["MKL_NUM_THREADS"] == "1"


def test_order_dataset_dirs_by_estimated_cost_desc(tmp_path):
    small = tmp_path / "small"
    large = tmp_path / "large"
    for path in (small, large):
        path.mkdir()

    (small / "table.npy").write_bytes(b"1" * 8)
    (large / "table.npy").write_bytes(b"1" * 64)

    ordered = softmax_pesudo.order_dataset_dirs_by_estimated_cost([small, large])
    assert ordered == [large, small]


def test_dynamic_queue_preserves_result_set_and_keeps_multiple_workers_busy(tmp_path):
    task_queue: queue.Queue = queue.Queue()
    ordered_names = ["slow", "fast_a", "fast_b", "fast_c", "fast_d"]
    for name in ordered_names:
        task_queue.put({"dataset_path": name})
    worker_count = 3
    for _ in range(worker_count):
        task_queue.put(None)

    durations = {
        "slow": 0.05,
        "fast_a": 0.01,
        "fast_b": 0.01,
        "fast_c": 0.01,
        "fast_d": 0.01,
    }

    assignments: list[tuple[str, str]] = []
    results: list[dict[str, object]] = []
    lock = threading.Lock()

    def make_worker(worker_name: str):
        def process_dataset(dataset_path: str):
            time.sleep(durations[dataset_path])
            return {"dataset": dataset_path, "accuracy": float(len(dataset_path))}, False

        def on_complete(dataset_path: str, result_item: dict[str, object] | None, has_missing: bool):
            del has_missing
            with lock:
                assignments.append((worker_name, dataset_path))
                if result_item is not None:
                    results.append(result_item)

        return threading.Thread(
            target=softmax_pesudo.drain_dataset_task_queue,
            args=(task_queue, process_dataset),
            kwargs={"on_dataset_complete": on_complete},
        )

    threads = [make_worker(f"worker-{idx}") for idx in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    worker_counts = Counter(worker_name for worker_name, _ in assignments)
    static_assignments = []
    fake_dirs = [tmp_path / name for name in ordered_names]
    for path in fake_dirs:
        path.mkdir()
    for _, _, assigned in softmax_pesudo.split_datasets_round_robin(fake_dirs, [0, 1, 2]):
        static_assignments.extend(Path(item).name for item in assigned)

    assert sorted(item["dataset"] for item in results) == sorted(ordered_names)
    assert sorted(static_assignments) == sorted(ordered_names)
    assert len(worker_counts) >= 2
    assert max(worker_counts.values()) < len(ordered_names)
