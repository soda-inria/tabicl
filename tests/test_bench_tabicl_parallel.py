from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ce_pesudo


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

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            values = X.iloc[:, 0].to_numpy()
        else:
            values = np.asarray(X)[:, 0]
        return values.astype(self.classes_.dtype, copy=False)


def test_run_pseudo_label_resplit_inference_refits_with_low_supervised_ce_samples():
    clf = FakePseudoClassifier(
        np.array(
            [
                [0.01, 0.99],
                [0.55, 0.45],
            ]
        )
    )

    X_train = pd.DataFrame({"x": [0, 1]})
    y_train = np.array([0, 1])
    X_test = pd.DataFrame({"x": [1, 0]})
    y_test = np.array([1, 1])

    def fit_fn(X, y):
        clf.fit(X, y)
        return clf, 0.1

    result = ce_pesudo.run_pseudo_label_resplit_inference(
        fit_classifier_fn=fit_fn,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_name="demo",
        ce_alpha=0.22,
    )

    assert result["pseudo_added"] == 1
    assert result["pseudo_applied"] is True
    assert result["baseline_accuracy"] == 0.5
    assert result["pseudo_ce_alpha"] == 0.22
    assert len(clf.fit_labels) == 2
    np.testing.assert_array_equal(clf.fit_labels[1], np.array([0, 1, 1]))
    np.testing.assert_array_equal(clf.fit_feature_values[1], np.array([0, 1, 1]))
    np.testing.assert_array_equal(result["y_eval"], y_test)
    assert np.mean(result["y_pred"] == result["y_eval"]) == 0.5


def test_run_pseudo_label_resplit_inference_falls_back_when_no_low_ce_sample():
    clf = FakePseudoClassifier(
        np.array(
            [
                [0.60, 0.40],
                [0.55, 0.45],
            ]
        )
    )

    X_train = pd.DataFrame({"x": [0, 1]})
    y_train = np.array([0, 1])
    X_test = pd.DataFrame({"x": [0, 0]})
    y_test = np.array([0, 0])

    def fit_fn(X, y):
        clf.fit(X, y)
        return clf, 0.2

    result = ce_pesudo.run_pseudo_label_resplit_inference(
        fit_classifier_fn=fit_fn,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_name="demo",
        ce_alpha=0.22,
    )

    assert result["pseudo_added"] == 0
    assert result["pseudo_applied"] is False
    assert len(clf.fit_labels) == 1
    np.testing.assert_array_equal(result["y_eval"], y_test)
    np.testing.assert_array_equal(result["y_pred"], np.array([0, 0]))


def test_run_pseudo_label_resplit_inference_uses_predicted_labels_after_ce_selection():
    clf = FakePseudoClassifier(
        np.array(
            [
                [0.60, 0.30, 0.10],
                [0.10, 0.80, 0.10],
            ]
        ),
        classes=np.array([0, 1, 2]),
    )

    X_train = pd.DataFrame({"x": [2, 2]})
    y_train = np.array([2, 2])
    X_test = pd.DataFrame({"x": [0, 1]})
    y_test = np.array([1, 0])

    def fit_fn(X, y):
        clf.fit(X, y)
        return clf, 0.1

    result = ce_pesudo.run_pseudo_label_resplit_inference(
        fit_classifier_fn=fit_fn,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_name="demo",
        ce_alpha=1.3,
    )

    assert result["pseudo_added"] == 1
    assert result["pseudo_applied"] is True
    assert result["baseline_accuracy"] == 0.0
    assert len(clf.fit_labels) == 2
    assert 0 in clf.fit_labels[1]
    assert 1 not in clf.fit_labels[1]


def test_run_pseudo_label_resplit_inference_raises_when_true_label_missing_from_classes():
    clf = FakePseudoClassifier(np.array([[0.90, 0.10]]), classes=np.array([0, 1]))

    X_train = pd.DataFrame({"x": [0, 1]})
    y_train = np.array([0, 1])
    X_test = pd.DataFrame({"x": [1]})
    y_test = np.array([2])

    def fit_fn(X, y):
        clf.fit(X, y)
        return clf, 0.1

    with pytest.raises(ValueError, match="未出现在 classifier\\.classes_"):
        ce_pesudo.run_pseudo_label_resplit_inference(
            fit_classifier_fn=fit_fn,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            dataset_name="demo",
            ce_alpha=0.22,
        )


def test_write_results_adds_pseudo_gain_stats_to_summary(tmp_path):
    results = [
        {
            "dataset": "alpha",
            "accuracy": 0.75,
            "time_s": 1.0,
            "fit_time_s": 0.7,
            "predict_time_s": 0.3,
            "baseline_accuracy": 0.50,
            "pseudo_added": 10,
            "pseudo_applied": True,
            "pseudo_ce_alpha": 0.22,
            "train_ratio_before": 0.8,
            "train_ratio_after": 0.8,
        },
        {
            "dataset": "beta",
            "accuracy": 0.50,
            "time_s": 2.0,
            "fit_time_s": 1.2,
            "predict_time_s": 0.8,
            "baseline_accuracy": 0.75,
            "pseudo_added": 12,
            "pseudo_applied": True,
            "pseudo_ce_alpha": 0.22,
            "train_ratio_before": 0.8,
            "train_ratio_after": 0.8,
        },
        {
            "dataset": "gamma",
            "accuracy": 0.50,
            "time_s": 3.0,
            "fit_time_s": 2.0,
            "predict_time_s": 1.0,
            "baseline_accuracy": 0.50,
            "pseudo_added": 8,
            "pseudo_applied": False,
            "pseudo_ce_alpha": 0.22,
            "train_ratio_before": 0.8,
            "train_ratio_after": 0.8,
        },
    ]

    ce_pesudo.write_results(
        tmp_path,
        results,
        datasets_with_missing=set(),
        script_duration=4.0,
        model_family="v1.1",
        classifier_config={
            "checkpoint_version": "ckpt",
            "model_path": "model.ckpt",
            "n_estimators": 32,
        },
    )

    summary_text = (tmp_path / "talent_summary.txt").read_text(encoding="utf-8")
    assert "Average baseline accuracy: 0.583333" in summary_text
    assert "Average pseudo ACC gain: 0.000000" in summary_text
    assert "Datasets improved by pseudo: 1" in summary_text
    assert "Datasets degraded by pseudo: 1" in summary_text
    assert "Datasets unchanged by pseudo: 1" in summary_text
    assert "Pseudo label CE alpha: 0.22" in summary_text
    assert "Datasets with pseudo update applied: 2" in summary_text


def test_build_arg_parser_sets_default_pseudo_label_ce_alpha():
    parser = ce_pesudo.build_arg_parser()
    args = parser.parse_args([])
    assert args.pseudo_label_ce_alpha == 0.22


def test_main_rejects_invalid_pseudo_label_ce_alpha(tmp_path):
    tmp_path.mkdir(exist_ok=True)

    with pytest.raises(SystemExit):
        ce_pesudo.main(
            [
                "--data-root",
                str(tmp_path),
                "--pseudo-label-ce-alpha",
                "-1",
            ]
        )

    with pytest.raises(SystemExit):
        ce_pesudo.main(
            [
                "--data-root",
                str(tmp_path),
                "--pseudo-label-ce-alpha",
                "nan",
            ]
        )
