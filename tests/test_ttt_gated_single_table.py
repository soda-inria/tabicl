from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import TTT_2A_gated as gated


class ScriptedClassifier:
    def __init__(self):
        self.fit_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fit(self, X, y):
        self.fit_calls.append((np.asarray(X).copy(), np.asarray(y).copy()))
        return self

    def predict(self, X):
        return np.asarray(["test_ok"] * len(X), dtype=object)


def install_dataset_mocks(monkeypatch, train_y, test_y=None):
    train_y = np.asarray(train_y, dtype=object)
    test_y = np.asarray(test_y if test_y is not None else ["test_ok", "test_ok"], dtype=object)
    train_x = pd.DataFrame({"f1": np.arange(len(train_y), dtype=float)})
    test_x = pd.DataFrame({"f1": np.arange(len(test_y), dtype=float) + 100.0})

    monkeypatch.setattr(gated, "load_dataset_info", lambda dataset_dir: {"task_type": "binclass"})
    monkeypatch.setattr(
        gated,
        "find_split_files",
        lambda dataset_dir: (("n_train", "c_train", "y_train"), None, ("n_test", "c_test", "y_test")),
    )
    monkeypatch.setattr(gated, "should_skip_ttt_for_dataset", lambda dataset_dir, info: False)

    def fake_load_split(n_file, c_file, y_file, *, context):
        if context.endswith("-train"):
            return train_x.copy(), train_y.copy()
        if context.endswith("-test"):
            return test_x.copy(), test_y.copy()
        raise AssertionError(f"Unexpected split context: {context}")

    monkeypatch.setattr(gated, "load_split", fake_load_split)
    return train_x, train_y, test_x, test_y


def run_gated_case(
    monkeypatch,
    *,
    train_y,
    split,
    baseline_metrics,
    adapted_metrics,
    ttt_result=None,
):
    train_x, train_y, _, _ = install_dataset_mocks(monkeypatch, train_y)
    monkeypatch.setattr(gated, "split_ttt_gated_holdout", lambda y, config: split)

    metrics_iter = iter([baseline_metrics, adapted_metrics])
    monkeypatch.setattr(gated, "evaluate_holdout_metrics", lambda classifier, X, y: next(metrics_iter))

    if ttt_result is None:
        ttt_result = gated.TTTUpdateResult(
            applied=True,
            loss=0.25,
            steps=1,
            update_seconds=0.05,
            reason=None,
        )

    def fake_run_ttt_holdout_update(classifier, X_b, y_b, X_c, y_c, config):
        return ttt_result

    preserve_calls = []

    def fake_fit_preserving_model_weights(classifier, X, y):
        preserve_calls.append((np.asarray(X).copy(), np.asarray(y).copy()))

    monkeypatch.setattr(gated, "run_ttt_holdout_update", fake_run_ttt_holdout_update)
    monkeypatch.setattr(gated, "_fit_preserving_model_weights", fake_fit_preserving_model_weights)

    classifier = ScriptedClassifier()
    row = gated.evaluate_one_dataset(classifier, Path("/tmp/demo"), gated.TTTConfig(enabled=True))
    return row, classifier, preserve_calls, train_x, train_y


def test_split_ttt_gated_holdout_uses_expected_631_ratio():
    y = np.asarray([0, 1] * 10, dtype=np.int64)

    split = gated.split_ttt_gated_holdout(y, gated.TTTConfig(enabled=True))

    assert split.strategy == "stratified"
    assert len(split.b_indices) == 12
    assert len(split.c_indices) == 6
    assert len(split.d_indices) == 2
    np.testing.assert_array_equal(
        np.sort(np.concatenate([split.b_indices, split.c_indices, split.d_indices])),
        np.arange(len(y)),
    )
    assert set(y[split.b_indices].tolist()) == {0, 1}
    assert set(y[split.c_indices].tolist()) == {0, 1}
    assert set(y[split.d_indices].tolist()) == {0, 1}


def test_split_ttt_gated_holdout_falls_back_to_random_when_stratify_unavailable():
    y = np.asarray([0, 0, 0, 1], dtype=np.int64)

    split = gated.split_ttt_gated_holdout(y, gated.TTTConfig(enabled=True))

    assert split.strategy == "random"
    assert "stratified split unavailable" in split.reason
    np.testing.assert_array_equal(
        np.sort(np.concatenate([split.b_indices, split.c_indices, split.d_indices])),
        np.arange(len(y)),
    )


def test_gate_accepts_only_when_adapted_accuracy_is_strictly_better(monkeypatch):
    train_y = ["a", "a", "a", "b", "b", "b", "a", "b", "a", "b"]
    split = gated.TTTSplit(
        b_indices=np.asarray([0, 1, 2, 3, 4, 5]),
        c_indices=np.asarray([6, 7, 8]),
        d_indices=np.asarray([9]),
        strategy="stratified",
        reason="test split",
    )

    row, classifier, preserve_calls, _, train_y_array = run_gated_case(
        monkeypatch,
        train_y=train_y,
        split=split,
        baseline_metrics=gated.GateEvalResult(accuracy=0.0, loss=0.80),
        adapted_metrics=gated.GateEvalResult(accuracy=1.0, loss=0.60),
    )

    assert row.ttt_update_ran is True
    assert row.ttt_gate_passed is True
    assert row.ttt_applied is True
    assert row.gate_accuracy_delta == 1.0
    assert np.isclose(row.gate_loss_delta, -0.20)
    assert len(classifier.fit_calls) == 1
    assert len(preserve_calls) == 1
    np.testing.assert_array_equal(preserve_calls[0][1], train_y_array)


def test_gate_rejects_when_accuracy_is_tied_even_if_loss_is_better(monkeypatch):
    train_y = ["a", "a", "a", "b", "b", "b", "a", "b", "a", "b"]
    split = gated.TTTSplit(
        b_indices=np.asarray([0, 1, 2, 3, 4, 5]),
        c_indices=np.asarray([6, 7, 8]),
        d_indices=np.asarray([9]),
        strategy="stratified",
        reason="test split",
    )

    row, classifier, preserve_calls, train_x, train_y_array = run_gated_case(
        monkeypatch,
        train_y=train_y,
        split=split,
        baseline_metrics=gated.GateEvalResult(accuracy=1.0, loss=0.80),
        adapted_metrics=gated.GateEvalResult(accuracy=1.0, loss=0.10),
    )

    assert row.ttt_update_ran is True
    assert row.ttt_gate_passed is False
    assert row.ttt_applied is False
    assert row.gate_accuracy_delta == 0.0
    assert np.isclose(row.gate_loss_delta, -0.70)
    assert "Gate rejected" in row.ttt_gate_reason
    assert len(preserve_calls) == 0
    assert len(classifier.fit_calls) == 2
    np.testing.assert_array_equal(classifier.fit_calls[1][0], train_x.to_numpy())
    np.testing.assert_array_equal(classifier.fit_calls[1][1], train_y_array)


def test_gate_rejects_when_accuracy_gets_worse(monkeypatch):
    train_y = ["a", "a", "a", "b", "b", "b", "a", "b", "a", "b"]
    split = gated.TTTSplit(
        b_indices=np.asarray([0, 1, 2, 3, 4, 5]),
        c_indices=np.asarray([6, 7, 8]),
        d_indices=np.asarray([9]),
        strategy="stratified",
        reason="test split",
    )

    row, classifier, preserve_calls, _, _ = run_gated_case(
        monkeypatch,
        train_y=train_y,
        split=split,
        baseline_metrics=gated.GateEvalResult(accuracy=1.0, loss=0.20),
        adapted_metrics=gated.GateEvalResult(accuracy=0.0, loss=0.90),
    )

    assert row.ttt_update_ran is True
    assert row.ttt_gate_passed is False
    assert row.ttt_applied is False
    assert row.gate_accuracy_delta == -1.0
    assert np.isclose(row.gate_loss_delta, 0.70)
    assert len(preserve_calls) == 0
    assert len(classifier.fit_calls) == 2


def test_ttt_is_skipped_when_c_contains_unseen_labels_from_b(monkeypatch):
    train_y = ["a", "a", "a", "a", "a", "a", "c", "a", "a", "a"]
    install_dataset_mocks(monkeypatch, train_y)
    split = gated.TTTSplit(
        b_indices=np.asarray([0, 1, 2, 3, 4, 5]),
        c_indices=np.asarray([6, 7, 8]),
        d_indices=np.asarray([9]),
        strategy="stratified",
        reason="test split",
    )
    monkeypatch.setattr(gated, "split_ttt_gated_holdout", lambda y, config: split)
    monkeypatch.setattr(
        gated,
        "run_ttt_holdout_update",
        lambda classifier, X_b, y_b, X_c, y_c, config: (_ for _ in ()).throw(AssertionError("TTT should be skipped")),
    )

    classifier = ScriptedClassifier()
    row = gated.evaluate_one_dataset(classifier, Path("/tmp/demo"), gated.TTTConfig(enabled=True))

    assert row.ttt_update_ran is False
    assert row.ttt_gate_passed is False
    assert "C labels absent from B context" in row.ttt_gate_reason
    assert len(classifier.fit_calls) == 1
    assert len(classifier.fit_calls[0][1]) == len(train_y)


def test_ttt_is_skipped_when_d_contains_unseen_labels_from_b(monkeypatch):
    train_y = ["a", "a", "a", "a", "a", "a", "a", "a", "a", "c"]
    install_dataset_mocks(monkeypatch, train_y)
    split = gated.TTTSplit(
        b_indices=np.asarray([0, 1, 2, 3, 4, 5]),
        c_indices=np.asarray([6, 7, 8]),
        d_indices=np.asarray([9]),
        strategy="stratified",
        reason="test split",
    )
    monkeypatch.setattr(gated, "split_ttt_gated_holdout", lambda y, config: split)
    monkeypatch.setattr(
        gated,
        "run_ttt_holdout_update",
        lambda classifier, X_b, y_b, X_c, y_c, config: (_ for _ in ()).throw(AssertionError("TTT should be skipped")),
    )

    classifier = ScriptedClassifier()
    row = gated.evaluate_one_dataset(classifier, Path("/tmp/demo"), gated.TTTConfig(enabled=True))

    assert row.ttt_update_ran is False
    assert row.ttt_gate_passed is False
    assert "D labels absent from B context" in row.ttt_gate_reason
    assert len(classifier.fit_calls) == 1
    assert len(classifier.fit_calls[0][1]) == len(train_y)


def test_evaluate_one_dataset_records_gate_metrics_and_split_sizes(monkeypatch):
    train_y = ["a", "a", "a", "b", "b", "b", "a", "b", "a", "b"]
    split = gated.TTTSplit(
        b_indices=np.asarray([0, 1, 2, 3, 4, 5]),
        c_indices=np.asarray([6, 7, 8]),
        d_indices=np.asarray([9]),
        strategy="stratified",
        reason="test split",
    )

    row, _, preserve_calls, _, train_y_array = run_gated_case(
        monkeypatch,
        train_y=train_y,
        split=split,
        baseline_metrics=gated.GateEvalResult(accuracy=0.25, loss=0.90),
        adapted_metrics=gated.GateEvalResult(accuracy=0.75, loss=0.30),
        ttt_result=gated.TTTUpdateResult(
            applied=True,
            loss=0.123,
            steps=2,
            update_seconds=0.5,
            reason=None,
        ),
    )

    assert row.status == "ok"
    assert row.accuracy == 1.0
    assert row.n_train_a == len(train_y_array)
    assert row.n_train_b == 6
    assert row.n_holdout_c == 3
    assert row.n_gate_d == 1
    assert row.n_test_e == 2
    assert row.ttt_loss == 0.123
    assert row.ttt_steps == 2
    assert row.ttt_update_ran is True
    assert row.ttt_gate_passed is True
    assert row.ttt_applied is True
    assert row.gate_baseline_accuracy == 0.25
    assert row.gate_adapted_accuracy == 0.75
    assert row.gate_accuracy_delta == 0.5
    assert row.gate_baseline_loss == 0.90
    assert row.gate_adapted_loss == 0.30
    assert np.isclose(row.gate_loss_delta, -0.60)
    assert row.ttt_gate_threshold == 0.0
    assert row.ttt_split_strategy == "stratified"
    assert row.ttt_split_reason == "test split"
    assert len(preserve_calls) == 1


def test_write_summary_includes_gate_statistics(tmp_path):
    df = pd.DataFrame(
        [
            {
                "dataset_name": "alpha",
                "status": "ok",
                "accuracy": 0.8,
                "ttt_gate_threshold": 0.0,
                "ttt_update_ran": True,
                "ttt_gate_passed": True,
                "gate_accuracy_delta": 0.1,
                "gate_loss_delta": -0.2,
            },
            {
                "dataset_name": "beta",
                "status": "ok",
                "accuracy": 0.7,
                "ttt_gate_threshold": 0.0,
                "ttt_update_ran": True,
                "ttt_gate_passed": False,
                "gate_accuracy_delta": 0.0,
                "gate_loss_delta": -0.1,
            },
            {
                "dataset_name": "gamma",
                "status": "ok",
                "accuracy": 0.6,
                "ttt_gate_threshold": 0.0,
                "ttt_update_ran": False,
                "ttt_gate_passed": False,
                "gate_accuracy_delta": None,
                "gate_loss_delta": None,
            },
        ]
    )

    summary_path = tmp_path / "summary.txt"
    gated.write_summary(summary_path, df, [Path("alpha"), Path("beta"), Path("gamma")], wall_seconds=1.23)
    text = summary_path.read_text(encoding="utf-8")

    assert "gate_pass_count: 1" in text
    assert "gate_reject_count: 1" in text
    assert "ttt_skipped_count: 1" in text
    assert "avg_gate_accuracy_delta_ok: 0.050000" in text
    assert "avg_gate_loss_delta_ok: -0.150000" in text
