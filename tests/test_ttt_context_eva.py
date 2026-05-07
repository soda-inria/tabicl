from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import TTT_1A_ICL as ttt1a
import TTT_context_eva as context_eva


class RecordingContextModel(torch.nn.Module):
    def __init__(self, max_classes: int = 10):
        super().__init__()
        self.max_classes = max_classes
        self.adapted = False


class FakeContextClassifier:
    instances: list["FakeContextClassifier"] = []
    max_classes = 10

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.predict_calls: list[np.ndarray] = []
        self.model_ = RecordingContextModel(max_classes=self.max_classes)
        self.device_ = torch.device("cpu")
        self.classes_ = np.asarray([])
        self.n_classes_ = 0
        type(self).instances.append(self)

    def fit(self, X, y):
        self.fit_calls.append((np.asarray(X).copy(), np.asarray(y).copy()))
        self.classes_ = np.asarray(list(dict.fromkeys(np.asarray(y).tolist())))
        self.n_classes_ = len(self.classes_)
        return self

    def predict(self, X):
        X = np.asarray(X)
        self.predict_calls.append(X.copy())
        if self.model_.adapted and len(self.classes_) > 1:
            return np.full(X.shape[0], self.classes_[1])
        return np.full(X.shape[0], self.classes_[0])

    @classmethod
    def reset(cls, max_classes: int = 10):
        cls.instances.clear()
        cls.max_classes = max_classes


def write_dataset(dataset_dir: Path, *, y_train, y_test, task_type: str = "binclass") -> None:
    dataset_dir.mkdir()
    (dataset_dir / "info.json").write_text(f'{{"task_type": "{task_type}"}}', encoding="utf-8")
    np.save(dataset_dir / "N_train.npy", np.arange(len(y_train) * 2, dtype=float).reshape(len(y_train), 2))
    np.save(dataset_dir / "y_train.npy", np.asarray(y_train))
    np.save(dataset_dir / "N_test.npy", np.arange(len(y_test) * 2, dtype=float).reshape(len(y_test), 2))
    np.save(dataset_dir / "y_test.npy", np.asarray(y_test))


def test_select_context_indices_uses_requested_size_and_covers_classes():
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 0, 1, 2, 1])

    indices = context_eva.select_context_indices(y, context_length=10, random_state=7)

    assert indices.shape == (10,)
    assert len(set(indices.tolist())) == 10
    assert set(y[indices].tolist()) == {0, 1, 2}


def test_evaluate_context_skips_when_classes_exceed_context_length(tmp_path):
    FakeContextClassifier.reset(max_classes=10)
    dataset_dir = tmp_path / "eleven_class"
    write_dataset(dataset_dir, y_train=np.arange(11), y_test=np.array([0, 1]))

    rows = context_eva.evaluate_one_dataset_context(
        FakeContextClassifier,
        {},
        dataset_dir,
        ttt1a.TTTConfig(enabled=True, save_ckpt=False),
        context_length=10,
        random_state=42,
    )

    assert [row.method for row in rows] == [context_eva.METHOD_BASELINE, context_eva.METHOD_TTT]
    assert [row.status for row in rows] == ["skip", "skip"]
    assert all("exceeds context_length=10" in (row.error or "") for row in rows)
    assert len(FakeContextClassifier.instances) == 0


def test_evaluate_context_runs_ttt_then_uses_same_ten_sample_context(tmp_path, monkeypatch):
    FakeContextClassifier.reset(max_classes=10)
    dataset_dir = tmp_path / "binary"
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    write_dataset(dataset_dir, y_train=y_train, y_test=np.array([0, 1]))
    update_calls: list[dict[str, object]] = []

    def fake_ttt_update(classifier, X_b, y_b, X_c, y_c, config, *, model_name, dataset_name):
        update_calls.append(
            {
                "classifier": classifier,
                "n_b": len(y_b),
                "n_c": len(y_c),
                "model_name": model_name,
                "dataset_name": dataset_name,
            }
        )
        classifier.model_.adapted = True
        return ttt1a.TTTUpdateResult(
            applied=True,
            loss=0.25,
            steps=1,
            update_seconds=0.5,
            step_losses=[0.25],
        )

    monkeypatch.setattr(context_eva.ttt1a, "run_ttt_holdout_update", fake_ttt_update)

    rows = context_eva.evaluate_one_dataset_context(
        FakeContextClassifier,
        {},
        dataset_dir,
        ttt1a.TTTConfig(enabled=True, steps=1, save_ckpt=False),
        context_length=10,
        random_state=42,
    )

    baseline_row, ttt_row = rows
    assert baseline_row.status == "ok"
    assert ttt_row.status == "ok"
    assert baseline_row.context_length == 10
    assert ttt_row.context_length == 10
    assert ttt_row.ttt_applied is True
    assert ttt_row.ttt_loss == 0.25
    assert ttt_row.ttt_step_losses == "[0.25]"
    assert len(update_calls) == 1
    assert update_calls[0]["dataset_name"] == "binary"

    baseline_classifier, ttt_classifier = FakeContextClassifier.instances
    assert len(baseline_classifier.fit_calls) == 1
    assert len(ttt_classifier.fit_calls) == 1
    assert baseline_classifier.fit_calls[0][0].shape == (10, 2)
    assert ttt_classifier.fit_calls[0][0].shape == (10, 2)
    assert set(baseline_classifier.fit_calls[0][1].tolist()) == {0, 1}
    np.testing.assert_array_equal(baseline_classifier.fit_calls[0][0], ttt_classifier.fit_calls[0][0])
    np.testing.assert_array_equal(baseline_classifier.fit_calls[0][1], ttt_classifier.fit_calls[0][1])
    assert len(baseline_classifier.predict_calls) == 1
    assert len(ttt_classifier.predict_calls) == 1
    assert ttt_classifier.model_.adapted is True
