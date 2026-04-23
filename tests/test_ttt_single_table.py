from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

import TTT_1A_ICL as ttt
import TTT_1B_ensmble as ttt_ensmble


class RecordingXEncoder:
    def __init__(self):
        self.calls: list[np.ndarray] = []

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        self.calls.append(arr.copy())
        return arr + 0.5


class SimpleYEncoder:
    def fit(self, y):
        self.classes_ = list(dict.fromkeys(np.asarray(y).tolist()))
        self.mapping_ = {value: idx for idx, value in enumerate(self.classes_)}
        return self

    def transform(self, y):
        return np.asarray([self.mapping_[value] for value in np.asarray(y).tolist()], dtype=np.int64)


class FailingEnsembleGenerator:
    def transform(self, *args, **kwargs):
        raise AssertionError("TTT must not use ensemble_generator_.transform")


class RecordingTTTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_classes = 2
        self.col_embedder = torch.nn.Linear(1, 1, bias=False)
        self.row_interactor = torch.nn.Linear(1, 1, bias=False)
        self.icl_predictor = torch.nn.Linear(1, self.max_classes, bias=False)
        self.forward_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.cache_cleared = False

    def forward(self, X, y_train, d):
        self.forward_calls.append((X.detach().cpu(), y_train.detach().cpu(), d.detach().cpu()))
        test_size = X.shape[1] - y_train.shape[1]
        features = torch.ones((X.shape[0], test_size, 1), device=X.device, dtype=X.dtype)
        return self.icl_predictor(features)

    def clear_cache(self):
        self.cache_cleared = True


class FakeClassifier:
    def __init__(self):
        self.fit_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.model_ = RecordingTTTModel()
        self.device_ = torch.device("cpu")
        self.softmax_temperature = 0.9
        self.inference_config_ = object()
        self.model_kv_cache_ = object()

    def fit(self, X, y):
        self.fit_calls.append((np.asarray(X).copy(), np.asarray(y).copy()))
        self.X_encoder_ = RecordingXEncoder()
        self.y_encoder_ = SimpleYEncoder().fit(y)
        self.classes_ = np.asarray(self.y_encoder_.classes_)
        self.n_classes_ = len(self.classes_)
        self.ensemble_generator_ = FailingEnsembleGenerator()
        return self


def test_run_ttt_holdout_update_uses_single_raw_encoded_table_without_ensemble():
    classifier = FakeClassifier()
    X_b = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])
    y_b = np.array(["a", "b", "a"], dtype=object)
    X_c = np.array([[3.0, 13.0], [4.0, 14.0]])
    y_c = np.array(["b", "a"], dtype=object)

    result = ttt.run_ttt_holdout_update(
        classifier,
        X_b,
        y_b,
        X_c,
        y_c,
        ttt.TTTConfig(enabled=True, steps=1, lr=1e-3, grad_clip=0.0),
    )

    assert result.applied is True
    assert result.steps == 1
    assert result.loss is not None
    assert len(classifier.fit_calls) == 1
    np.testing.assert_array_equal(classifier.fit_calls[0][0], X_b)
    np.testing.assert_array_equal(classifier.fit_calls[0][1], y_b)

    assert len(classifier.X_encoder_.calls) == 2
    np.testing.assert_array_equal(classifier.X_encoder_.calls[0], X_b)
    np.testing.assert_array_equal(classifier.X_encoder_.calls[1], X_c)

    assert len(classifier.model_.forward_calls) == 1
    X_forward, y_train_forward, d_forward = classifier.model_.forward_calls[0]
    expected_X = np.concatenate([X_b + 0.5, X_c + 0.5], axis=0)
    np.testing.assert_allclose(X_forward.numpy(), expected_X[None, :, :])
    np.testing.assert_allclose(y_train_forward.numpy(), np.array([[0.0, 1.0, 0.0]]))
    np.testing.assert_array_equal(d_forward.numpy(), np.array([2]))
    assert classifier.model_kv_cache_ is None
    assert classifier.model_.cache_cleared is True
    assert classifier.model_.training is False


class RecordingEnsembleGenerator:
    def __init__(self):
        self.transform_calls: list[tuple[np.ndarray, str]] = []
        self.class_shuffles_ = OrderedDict(
            [
                (
                    "none",
                    [
                        np.array([0, 1], dtype=np.int64),
                        np.array([1, 0], dtype=np.int64),
                    ],
                )
            ]
        )

    def transform(self, X, mode="both"):
        self.transform_calls.append((np.asarray(X).copy(), mode))
        Xs = np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [-2.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0], [-2.0, 0.0]],
            ],
            dtype=np.float32,
        )
        ys = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
            ],
            dtype=np.int64,
        )
        return OrderedDict([("none", (Xs, ys))])


class RecordingEnsembleTTTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_classes = 2
        self.col_embedder = torch.nn.Linear(1, 1, bias=False)
        self.row_interactor = torch.nn.Linear(1, 1, bias=False)
        self.icl_predictor = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.icl_predictor.weight.fill_(1.0)
        self.forward_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.cache_cleared = False

    def forward(self, X, y_train, d):
        self.forward_calls.append((X.detach().cpu(), y_train.detach().cpu(), d.detach().cpu()))
        train_size = y_train.shape[1]
        z = X[:, train_size:, 0] * self.icl_predictor.weight.squeeze()
        return torch.stack([z, -z], dim=-1)

    def clear_cache(self):
        self.cache_cleared = True


class FakeEnsembleClassifier:
    def __init__(self):
        self.fit_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.model_ = RecordingEnsembleTTTModel()
        self.device_ = torch.device("cpu")
        self.softmax_temperature = 0.9
        self.inference_config_ = object()
        self.model_kv_cache_ = object()
        self.ensemble_generator_: RecordingEnsembleGenerator | None = None

    def fit(self, X, y):
        self.fit_calls.append((np.asarray(X).copy(), np.asarray(y).copy()))
        self.X_encoder_ = RecordingXEncoder()
        self.y_encoder_ = SimpleYEncoder().fit(y)
        self.classes_ = np.asarray(self.y_encoder_.classes_)
        self.n_classes_ = len(self.classes_)
        self.ensemble_generator_ = RecordingEnsembleGenerator()
        return self


def test_ensemble_ttt_uses_ensemble_views_and_remaps_holdout_labels():
    classifier = FakeEnsembleClassifier()
    X_b = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])
    y_b = np.array(["a", "b", "a"], dtype=object)
    X_c = np.array([[3.0, 13.0], [4.0, 14.0]])
    y_c = np.array(["b", "a"], dtype=object)

    result = ttt_ensmble.run_ttt_holdout_update(
        classifier,
        X_b,
        y_b,
        X_c,
        y_c,
        ttt_ensmble.TTTConfig(
            enabled=True,
            steps=1,
            lr=1e-3,
            grad_clip=0.0,
            micro_batch_size=1,
        ),
    )

    assert result.applied is True
    assert result.steps == 1
    assert result.loss is not None
    assert classifier.ensemble_generator_ is not None
    assert len(classifier.ensemble_generator_.transform_calls) == 1
    X_transform, mode = classifier.ensemble_generator_.transform_calls[0]
    assert mode == "both"
    np.testing.assert_array_equal(X_transform, X_c + 0.5)

    assert len(classifier.model_.forward_calls) == 2
    _, y_train_view0, d_view0 = classifier.model_.forward_calls[0]
    _, y_train_view1, d_view1 = classifier.model_.forward_calls[1]
    np.testing.assert_array_equal(y_train_view0.numpy(), np.array([[0.0, 1.0, 0.0]]))
    np.testing.assert_array_equal(y_train_view1.numpy(), np.array([[1.0, 0.0, 1.0]]))
    np.testing.assert_array_equal(d_view0.numpy(), np.array([2]))
    np.testing.assert_array_equal(d_view1.numpy(), np.array([2]))

    logits = torch.tensor(
        [
            [[2.0, -2.0], [-2.0, 2.0]],
            [[2.0, -2.0], [-2.0, 2.0]],
        ]
    )
    remapped_targets = torch.tensor([[1, 0], [0, 1]])
    expected_loss = (
        F.cross_entropy(logits[0], remapped_targets[0])
        + F.cross_entropy(logits[1], remapped_targets[1])
    ) / 2
    assert np.isclose(result.loss, float(expected_loss), atol=1e-6)
    assert classifier.model_kv_cache_ is None
    assert classifier.model_.cache_cleared is True
    assert classifier.model_.training is False
