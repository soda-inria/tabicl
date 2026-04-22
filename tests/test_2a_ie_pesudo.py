from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "2a_ie_pesudo.py"


def load_module():
    module_name = "module_2a_ie_pesudo"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


bench = load_module()


class FakePseudoClassifier:
    def __init__(self, proba: np.ndarray, classes: np.ndarray | None = None):
        self.proba = np.asarray(proba, dtype=float)
        self.classes_ = np.asarray(classes if classes is not None else [0, 1])
        self.fit_labels: list[np.ndarray] = []
        self.fit_feature_values: list[np.ndarray] = []

    def fit(self, X, y):
        self.fit_labels.append(np.asarray(y).copy())
        if isinstance(X, pd.DataFrame):
            values = X.iloc[:, 0].to_numpy()
        else:
            values = np.asarray(X)[:, 0]
        self.fit_feature_values.append(np.asarray(values).copy())
        return self

    def predict_proba(self, X):
        assert len(X) == len(self.proba)
        return self.proba


def test_choose_pmax_quota_by_class_balances_total_coverage_and_orders_by_confidence():
    pmax = np.array([0.80, 0.90, 0.70, 0.60, 0.99, 0.50, 0.40, 0.95, 0.85, 0.65])
    predicted = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])

    quota, selected_mask, selected_total = bench.choose_pmax_quota_by_class(
        pmax,
        predicted,
        target_test_coverage=0.7,
    )

    assert selected_total == 7
    np.testing.assert_array_equal(
        np.flatnonzero(selected_mask),
        np.array([0, 1, 2, 4, 5, 7, 8]),
    )
    assert quota["0"]["quota"] == 3
    assert quota["1"]["quota"] == 2
    assert quota["2"]["quota"] == 2
    assert quota["0"]["min_pmax"] == 0.70
    assert quota["0"]["target_total"] == 7
    assert quota["0"]["target_test_coverage"] == 0.7


def test_choose_pmax_quota_by_class_tie_breaks_by_original_test_index():
    pmax = np.array([0.90, 0.90, 0.80])
    predicted = np.array([1, 1, 1])

    _, selected_mask, selected_total = bench.choose_pmax_quota_by_class(
        pmax,
        predicted,
        target_test_coverage=0.67,
    )

    assert selected_total == 2
    np.testing.assert_array_equal(np.flatnonzero(selected_mask), np.array([0, 1]))


def test_run_top1_quota_pseudo_uses_classes_mapping_and_writes_pmax_artifacts(tmp_path):
    clf = FakePseudoClassifier(
        np.array(
            [
                [0.99, 0.01],
                [0.95, 0.05],
                [0.90, 0.10],
                [0.80, 0.20],
                [0.70, 0.30],
                [0.02, 0.98],
                [0.06, 0.94],
                [0.11, 0.89],
                [0.40, 0.60],
                [0.45, 0.55],
            ]
        ),
        classes=np.array(["no", "yes"]),
    )

    X_train = pd.DataFrame({"x": [100, 101]})
    y_train = np.array(["no", "yes"])
    X_test = pd.DataFrame({"x": np.arange(10)})
    y_test = np.array(["yes", "no", "no", "no", "no", "no", "yes", "yes", "yes", "yes"])

    y_pred, fit_seconds, pred_seconds, meta = bench.run_entropy_two_pass_pseudo(
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        pseudo_max_error_rate=0.0,
        pseudo_rounds=2,
        target_test_coverage=0.7,
        entropy_save_dir=tmp_path,
        dataset_key="demo",
    )

    np.testing.assert_array_equal(
        y_pred,
        np.array(["no", "no", "no", "no", "no", "yes", "yes", "yes", "yes", "yes"]),
    )
    assert fit_seconds >= 0.0
    assert pred_seconds >= 0.0
    assert meta["selected_cnt"] == 7
    assert meta["selected_correct"] == 5
    assert meta["selected_wrong"] == 2
    assert meta["selected_precision"] == 5 / 7
    assert meta["selected_error_rate"] == 2 / 7
    assert meta["target_test_coverage"] == 0.7
    assert meta["rounds"] == 2
    assert meta["round_accs"] == [0.8, 0.8]
    assert len(clf.fit_labels) == 2
    np.testing.assert_array_equal(
        clf.fit_labels[1],
        np.array(["no", "yes", "no", "no", "no", "no", "yes", "yes", "yes"]),
    )
    np.testing.assert_array_equal(
        clf.fit_feature_values[1],
        np.array([100, 101, 0, 1, 2, 3, 5, 6, 7]),
    )

    quota = meta["threshold_by_class"]
    assert quota["no"]["selected"] == 4
    assert quota["yes"]["selected"] == 3
    assert quota["no"]["min_pmax"] == 0.80
    assert quota["yes"]["min_pmax"] == 0.89
    assert meta["threshold_by_class_json"] == json.dumps(
        quota,
        ensure_ascii=False,
        sort_keys=True,
    )

    saved = np.load(tmp_path / "demo__round1_entropy.npz")
    assert "pmax" in saved.files
    assert "entropy" not in saved.files
    np.testing.assert_allclose(saved["pmax"], np.array([0.99, 0.95, 0.90, 0.80, 0.70, 0.98, 0.94, 0.89, 0.60, 0.55]))
    np.testing.assert_array_equal(saved["selected_mask"], np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 0], dtype=np.uint8))
    assert saved["threshold_by_class_json"].tolist() == meta["threshold_by_class_json"]


def test_build_arg_parser_defaults_target_test_coverage_to_seventy_percent():
    parser = bench.build_arg_parser()

    assert parser.parse_args([]).target_test_coverage == 0.7
    assert parser.parse_args(["--target-test-coverage", "0.25"]).target_test_coverage == 0.25
