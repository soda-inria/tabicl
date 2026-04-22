from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "ie_pesudo_single_tabicl.py"


def load_module():
    module_name = "module_ie_pesudo_single_tabicl"
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


class SequentialPseudoClassifier(FakePseudoClassifier):
    def __init__(self, probas: list[np.ndarray], classes: np.ndarray | None = None):
        super().__init__(probas[0], classes=classes)
        self.probas = [np.asarray(proba, dtype=float) for proba in probas]
        self.predict_calls = 0

    def predict_proba(self, X):
        proba = self.probas[min(self.predict_calls, len(self.probas) - 1)]
        self.predict_calls += 1
        assert len(X) == len(proba)
        return proba


def make_result_row(**overrides):
    base = {
        "dataset_name": "alpha",
        "dataset_dir": "/tmp/alpha",
        "task_type": "binclass",
        "n_train": 6,
        "n_train_initial": 4,
        "n_train_final": 6,
        "n_val": 1,
        "n_test": 2,
        "n_features": 3,
        "n_classes": 2,
        "accuracy": 0.75,
        "acc_rounds_json": "[0.5, 0.625, 0.75]",
        "acc_round1": 0.5,
        "acc_round2": 0.75,
        "acc_delta": 0.25,
        "acc_improved": True,
        "train_ratio_before": 0.666667,
        "train_ratio_after": 0.75,
        "pass1_acc": 0.5,
        "entropy_threshold": '{"0": 0.123456}',
        "pseudo_selected": 2,
        "pseudo_correct": 2,
        "pseudo_wrong": 0,
        "pseudo_precision": 1.0,
        "pseudo_error_rate": 0.0,
        "pseudo_rounds": 3,
        "entropy_file": "/tmp/alpha_round1.npz|/tmp/alpha_round2.npz",
        "fit_seconds": 0.4,
        "predict_seconds": 0.2,
        "status": "ok",
        "error": None,
    }
    base.update(overrides)
    return base


def test_run_entropy_two_pass_pseudo_appends_predicted_labels_and_writes_entropy(tmp_path):
    clf = FakePseudoClassifier(
        np.array(
            [
                [0.95, 0.05],
                [0.55, 0.45],
                [0.02, 0.98],
                [0.45, 0.55],
            ]
        )
    )

    X_train = pd.DataFrame({"x": [0, 1]})
    y_train = np.array([0, 1])
    X_test = pd.DataFrame({"x": [10, 11, 12, 13]})
    y_test = np.array([0, 1, 1, 0])

    result = bench.run_entropy_two_pass_pseudo(
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        pseudo_max_error_rate=0.05,
        pseudo_rounds=2,
        entropy_save_dir=tmp_path,
        dataset_key="demo",
    )

    y_pred, fit_seconds, pred_seconds, meta = result
    np.testing.assert_array_equal(y_pred, np.array([0, 0, 1, 1]))
    assert fit_seconds >= 0.0
    assert pred_seconds >= 0.0
    assert meta["selected_cnt"] == 2
    assert meta["selected_correct"] == 2
    assert meta["selected_wrong"] == 0
    assert meta["rounds"] == 2
    assert meta["round_accs"] == [0.5, 0.5]
    assert meta["round_accs_json"] == "[0.5, 0.5]"
    expected_thresholds = {
        "0": bench.entropy_from_proba(np.array([[0.95, 0.05]])).item(),
        "1": bench.entropy_from_proba(np.array([[0.02, 0.98]])).item(),
    }
    assert meta["threshold_by_class"] == expected_thresholds
    assert meta["threshold_by_class_json"] == json.dumps(
        expected_thresholds,
        ensure_ascii=False,
        sort_keys=True,
    )
    assert len(clf.fit_labels) == 2
    np.testing.assert_array_equal(clf.fit_labels[0], np.array([0, 1]))
    np.testing.assert_array_equal(clf.fit_labels[1], np.array([0, 1, 0, 1]))
    np.testing.assert_array_equal(clf.fit_feature_values[1], np.array([0, 1, 10, 12]))
    assert len(meta["entropy_files"]) == 2
    assert (tmp_path / "demo__round1_entropy.npz").exists()
    assert (tmp_path / "demo__round2_entropy.npz").exists()
    saved = np.load(tmp_path / "demo__round1_entropy.npz")
    assert saved["threshold_by_class_json"].tolist() == meta["threshold_by_class_json"]


def test_run_entropy_two_pass_pseudo_sets_null_threshold_for_class_without_feasible_selection():
    clf = FakePseudoClassifier(
        np.array(
            [
                [0.90, 0.10],
                [0.80, 0.20],
                [0.10, 0.90],
            ]
        )
    )

    X_train = pd.DataFrame({"x": [0, 1]})
    y_train = np.array([0, 1])
    X_test = pd.DataFrame({"x": [20, 21, 22]})
    y_test = np.array([1, 1, 1])

    y_pred, _, _, meta = bench.run_entropy_two_pass_pseudo(
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        pseudo_max_error_rate=0.0,
        pseudo_rounds=1,
        entropy_save_dir=None,
        dataset_key=None,
    )

    np.testing.assert_array_equal(y_pred, np.array([0, 0, 1]))
    assert meta["selected_cnt"] == 1
    assert meta["selected_correct"] == 1
    assert meta["selected_wrong"] == 0
    assert meta["round_accs"] == [1 / 3]
    assert meta["threshold_by_class"]["0"] is None
    expected_thresholds = {
        "0": None,
        "1": bench.entropy_from_proba(np.array([[0.10, 0.90]])).item(),
    }
    assert meta["threshold_by_class"] == expected_thresholds
    assert meta["threshold_by_class_json"] == json.dumps(
        expected_thresholds,
        ensure_ascii=False,
        sort_keys=True,
    )


def test_run_entropy_two_pass_pseudo_falls_back_when_no_sample_selected():
    clf = FakePseudoClassifier(np.array([[0.60, 0.40], [0.55, 0.45]]))

    X_train = pd.DataFrame({"x": [0, 1]})
    y_train = np.array([0, 1])
    X_test = pd.DataFrame({"x": [0, 0]})
    y_test = np.array([1, 1])

    y_pred, _, _, meta = bench.run_entropy_two_pass_pseudo(
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        pseudo_max_error_rate=-3.0,
        pseudo_rounds=0,
        entropy_save_dir=None,
        dataset_key=None,
    )

    np.testing.assert_array_equal(y_pred, np.array([0, 0]))
    assert len(clf.fit_labels) == 1
    assert meta["selected_cnt"] == 0
    assert meta["rounds"] == 1
    assert meta["round_accs"] == [0.0]
    assert meta["threshold_by_class"] == {"0": None}
    assert meta["threshold_by_class_json"] == '{"0": null}'


def test_run_entropy_two_pass_pseudo_records_three_round_accuracies():
    clf = SequentialPseudoClassifier(
        [
            np.array([[0.90, 0.10], [0.80, 0.20], [0.70, 0.30], [0.60, 0.40]]),
            np.array([[0.90, 0.10], [0.20, 0.80], [0.70, 0.30], [0.60, 0.40]]),
            np.array([[0.90, 0.10], [0.20, 0.80], [0.30, 0.70], [0.60, 0.40]]),
        ]
    )

    X_train = pd.DataFrame({"x": [0, 1]})
    y_train = np.array([0, 1])
    X_test = pd.DataFrame({"x": [20, 21, 22, 23]})
    y_test = np.array([0, 1, 1, 1])

    y_pred, _, _, meta = bench.run_entropy_two_pass_pseudo(
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        pseudo_max_error_rate=0.0,
        pseudo_rounds=3,
        entropy_save_dir=None,
        dataset_key=None,
    )

    np.testing.assert_array_equal(y_pred, np.array([0, 1, 1, 0]))
    assert meta["rounds"] == 3
    assert meta["round_accs"] == [0.25, 0.5, 0.75]
    assert meta["round_accs_json"] == "[0.25, 0.5, 0.75]"
    assert meta["pass1_acc"] == 0.25


def test_format_dataset_result_log_for_single_model_ok():
    row = bench.ResultRow(**make_result_row())

    text = bench.format_dataset_result_log("worker 0 | gpu 1", row)

    assert text == (
        "[worker 0 | gpu 1] [ok] alpha accuracy=0.750000 "
        "pass1_acc=0.500000 pseudo_selected=2"
    )


def test_format_dataset_result_log_for_multi_model_skip_and_fail():
    skip_row = bench.ResultRow(
        **make_result_row(
            dataset_name="beta",
            status="skip",
            error="Skipped due to task_type='regression'",
        )
    )
    fail_row = bench.ResultRow(
        **make_result_row(
            dataset_name="gamma",
            status="fail",
            error="RuntimeError: boom",
        )
    )

    skip_text = bench.format_dataset_result_log(
        "worker 2 | gpu 3",
        skip_row,
        model_name="model_v1",
    )
    fail_text = bench.format_dataset_result_log(
        "worker 2 | gpu 3",
        fail_row,
        model_name="model_v1",
    )

    assert skip_text == (
        "[worker 2 | gpu 3] [model_v1] [skip] beta "
        "reason=Skipped due to task_type='regression'"
    )
    assert fail_text == (
        "[worker 2 | gpu 3] [model_v1] [fail] gamma error=RuntimeError: boom"
    )


def test_build_arg_parser_enables_entropy_artifacts_by_default_and_supports_opt_out():
    parser = bench.build_arg_parser()
    assert parser.parse_args([]).save_entropy_artifacts is True
    assert parser.parse_args(["--no-save-entropy-artifacts"]).save_entropy_artifacts is False


def test_resolve_data_root_uses_current_directory_subfolder(tmp_path, monkeypatch):
    data_dir = tmp_path / "data178"
    data_dir.mkdir()

    monkeypatch.chdir(tmp_path)

    resolved = bench.resolve_data_root_path("data178")
    assert resolved == data_dir.resolve()


def test_resolve_data_root_falls_back_to_repo_root(tmp_path, monkeypatch):
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    monkeypatch.chdir(outside_dir)

    data_dir = bench.REPO_ROOT / "data178"
    resolved = bench.resolve_data_root_path("data178")
    assert resolved == data_dir.resolve()


def test_write_dataset_outputs_persists_pseudo_columns_and_summary(tmp_path):
    result_df = pd.DataFrame([make_result_row()])
    all_csv, summary_path = bench.write_dataset_outputs(
        tmp_path,
        result_df,
        [tmp_path / "alpha"],
        wall_seconds=1.5,
    )

    csv_text = all_csv.read_text(encoding="utf-8")
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "n_train_initial" in csv_text
    assert "acc_rounds_json" in csv_text
    assert "acc_round1" in csv_text
    assert "acc_round2" in csv_text
    assert "acc_delta" in csv_text
    assert "acc_improved" in csv_text
    assert "pseudo_selected" in csv_text
    assert "entropy_file" in csv_text
    assert '"{""0"": 0.123456}"' in csv_text
    assert "avg_acc_round1_ok: 0.500000" in summary_text
    assert "avg_acc_round2_ok: 0.625000" in summary_text
    assert "avg_acc_round3_ok: 0.750000" in summary_text
    assert "avg_acc_delta_ok: 0.250000" in summary_text
    assert "improved_dataset_count: 1" in summary_text
    assert "degraded_dataset_count: 0" in summary_text
    assert "unchanged_dataset_count: 0" in summary_text
    assert "avg_pass1_acc_ok: 0.500000" in summary_text
    assert "total_pseudo_selected_ok: 2" in summary_text
    assert "avg_pseudo_precision_ok: 1.000000" in summary_text
    assert "avg_pseudo_error_rate_ok: 0.000000" in summary_text


def test_build_model_summary_row_tracks_acc_comparison_counts():
    rows = [
        bench.ResultRow(**make_result_row(dataset_name="alpha", acc_delta=0.25, acc_improved=True)),
        bench.ResultRow(
            **make_result_row(
                dataset_name="beta",
                accuracy=0.4,
                acc_round1=0.5,
                acc_round2=0.4,
                acc_delta=-0.1,
                acc_improved=False,
                pass1_acc=0.5,
            )
        ),
        bench.ResultRow(
            **make_result_row(
                dataset_name="gamma",
                accuracy=0.5,
                acc_round1=0.5,
                acc_round2=0.5,
                acc_delta=0.0,
                acc_improved=False,
                pass1_acc=0.5,
            )
        ),
    ]

    summary = bench.build_model_summary_row(
        Path("/tmp/model.ckpt"),
        gpu_id=0,
        dataset_dirs=[Path("/tmp/alpha"), Path("/tmp/beta"), Path("/tmp/gamma")],
        rows=rows,
        model_wall_seconds=1.2,
    )

    assert summary.avg_acc_round1_ok == 0.5
    assert abs(summary.avg_acc_round2_ok - 0.55) < 1e-9
    assert abs(summary.avg_acc_delta_ok - 0.05) < 1e-9
    assert summary.improved_dataset_count == 1
    assert summary.degraded_dataset_count == 1
    assert summary.unchanged_dataset_count == 1


def test_build_model_summary_row_ignores_zero_selection_datasets_for_avg_pseudo_error_rate_ok():
    rows = [
        bench.ResultRow(**make_result_row(dataset_name="alpha", pseudo_selected=3, pseudo_error_rate=0.2)),
        bench.ResultRow(**make_result_row(dataset_name="beta", pseudo_selected=0, pseudo_error_rate=0.0)),
        bench.ResultRow(**make_result_row(dataset_name="gamma", pseudo_selected=5, pseudo_error_rate=0.4)),
    ]

    summary = bench.build_model_summary_row(
        Path("/tmp/model.ckpt"),
        gpu_id=0,
        dataset_dirs=[Path("/tmp/alpha"), Path("/tmp/beta"), Path("/tmp/gamma")],
        rows=rows,
        model_wall_seconds=1.2,
    )

    assert abs(summary.avg_pseudo_error_rate_ok - 0.3) < 1e-9


def test_write_dataset_outputs_reports_none_when_all_ok_datasets_select_no_pseudo_labels(tmp_path):
    result_df = pd.DataFrame(
        [
            make_result_row(dataset_name="alpha", pseudo_selected=0, pseudo_error_rate=0.0),
            make_result_row(dataset_name="beta", pseudo_selected=0, pseudo_error_rate=0.0),
        ]
    )

    _, summary_path = bench.write_dataset_outputs(
        tmp_path,
        result_df,
        [tmp_path / "alpha", tmp_path / "beta"],
        wall_seconds=1.5,
    )

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "avg_pseudo_error_rate_ok: (none)" in summary_text

    summary = bench.build_model_summary_row(
        Path("/tmp/model.ckpt"),
        gpu_id=0,
        dataset_dirs=[Path("/tmp/alpha"), Path("/tmp/beta")],
        rows=[bench.ResultRow(**row) for row in result_df.to_dict(orient="records")],
        model_wall_seconds=1.2,
    )
    assert summary.avg_pseudo_error_rate_ok is None


def test_write_model_pool_outputs_includes_pseudo_aggregates(tmp_path):
    bench.write_model_pool_outputs(
        tmp_path,
        [
            {
                "model_name": "m1",
                "model_path": "/tmp/m1.ckpt",
                "gpu_id": 0,
                "datasets_discovered": 3,
                "ok_count": 2,
                "failed_count": 1,
                "skipped_count": 0,
                "avg_accuracy_ok": 0.7,
                "avg_acc_rounds_json": "[0.5, 0.7]",
                "avg_acc_round1_ok": 0.5,
                "avg_acc_round2_ok": 0.7,
                "avg_acc_delta_ok": 0.2,
                "avg_fit_seconds_ok": 0.4,
                "avg_predict_seconds_ok": 0.2,
                "avg_dataset_seconds_ok": 0.6,
                "avg_pass1_acc_ok": 0.5,
                "avg_train_ratio_before_ok": 0.6,
                "avg_train_ratio_after_ok": 0.7,
                "avg_pseudo_selected_ok": 1.5,
                "total_pseudo_selected_ok": 3,
                "improved_dataset_count": 1,
                "degraded_dataset_count": 0,
                "unchanged_dataset_count": 1,
                "avg_pseudo_precision_ok": 0.9,
                "avg_pseudo_error_rate_ok": 0.1,
                "total_dataset_seconds_ok": 1.2,
                "model_wall_seconds": 2.0,
                "status": "ok",
                "error": None,
                "failed_datasets": "gamma",
            }
        ],
        wall_seconds=2.5,
    )

    csv_text = (tmp_path / "all_models_summary.csv").read_text(encoding="utf-8")
    summary_text = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    assert "avg_acc_round1_ok" in csv_text
    assert "avg_acc_round2_ok" in csv_text
    assert "avg_acc_delta_ok" in csv_text
    assert "improved_dataset_count" in csv_text
    assert "avg_pass1_acc_ok" in csv_text
    assert "total_pseudo_selected_ok" in csv_text
    assert "average_avg_acc_round1_ok: 0.500000" in summary_text
    assert "average_avg_acc_round2_ok: 0.700000" in summary_text
    assert "average_avg_acc_delta_ok: 0.200000" in summary_text
    assert "total_improved_dataset_count: 1" in summary_text
    assert "total_degraded_dataset_count: 0" in summary_text
    assert "total_unchanged_dataset_count: 1" in summary_text
    assert "average_avg_pass1_acc_ok: 0.500000" in summary_text
    assert "total_pseudo_selected_ok: 3" in summary_text
    assert "average_avg_pseudo_precision_ok: 0.900000" in summary_text


def test_write_model_pool_outputs_skips_none_avg_pseudo_error_rate_ok(tmp_path):
    bench.write_model_pool_outputs(
        tmp_path,
        [
            {
                "model_name": "m1",
                "model_path": "/tmp/m1.ckpt",
                "gpu_id": 0,
                "datasets_discovered": 2,
                "ok_count": 2,
                "failed_count": 0,
                "skipped_count": 0,
                "avg_accuracy_ok": 0.7,
                "avg_acc_rounds_json": "[0.5, 0.7]",
                "avg_acc_round1_ok": 0.5,
                "avg_acc_round2_ok": 0.7,
                "avg_acc_delta_ok": 0.2,
                "avg_fit_seconds_ok": 0.4,
                "avg_predict_seconds_ok": 0.2,
                "avg_dataset_seconds_ok": 0.6,
                "avg_pass1_acc_ok": 0.5,
                "avg_train_ratio_before_ok": 0.6,
                "avg_train_ratio_after_ok": 0.7,
                "avg_pseudo_selected_ok": 1.5,
                "total_pseudo_selected_ok": 3,
                "improved_dataset_count": 1,
                "degraded_dataset_count": 0,
                "unchanged_dataset_count": 1,
                "avg_pseudo_precision_ok": 0.9,
                "avg_pseudo_error_rate_ok": None,
                "total_dataset_seconds_ok": 1.2,
                "model_wall_seconds": 2.0,
                "status": "ok",
                "error": None,
                "failed_datasets": "",
            },
            {
                "model_name": "m2",
                "model_path": "/tmp/m2.ckpt",
                "gpu_id": 1,
                "datasets_discovered": 2,
                "ok_count": 2,
                "failed_count": 0,
                "skipped_count": 0,
                "avg_accuracy_ok": 0.8,
                "avg_acc_rounds_json": "[0.6, 0.8]",
                "avg_acc_round1_ok": 0.6,
                "avg_acc_round2_ok": 0.8,
                "avg_acc_delta_ok": 0.2,
                "avg_fit_seconds_ok": 0.5,
                "avg_predict_seconds_ok": 0.3,
                "avg_dataset_seconds_ok": 0.8,
                "avg_pass1_acc_ok": 0.6,
                "avg_train_ratio_before_ok": 0.5,
                "avg_train_ratio_after_ok": 0.7,
                "avg_pseudo_selected_ok": 2.0,
                "total_pseudo_selected_ok": 4,
                "improved_dataset_count": 2,
                "degraded_dataset_count": 0,
                "unchanged_dataset_count": 0,
                "avg_pseudo_precision_ok": 0.8,
                "avg_pseudo_error_rate_ok": 0.25,
                "total_dataset_seconds_ok": 1.6,
                "model_wall_seconds": 2.5,
                "status": "ok",
                "error": None,
                "failed_datasets": "",
            },
        ],
        wall_seconds=3.0,
    )

    summary_text = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    assert "average_avg_pseudo_error_rate_ok: 0.250000" in summary_text
