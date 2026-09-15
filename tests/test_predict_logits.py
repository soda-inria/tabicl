"""Public classifier logits, including probability ensembles and cached inference."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import softmax
from sklearn.exceptions import NotFittedError

from tabicl import TabICLClassifier
from tabicl._model.tabicl import TabICL


@pytest.fixture(scope="module")
def small_checkpoint(tmp_path_factory):
    """Exercise real inference without downloading pretrained weights."""
    config = {
        "max_classes": 3,
        "embed_dim": 16,
        "col_num_blocks": 1,
        "col_nhead": 2,
        "col_num_inds": 4,
        "row_num_blocks": 1,
        "row_nhead": 2,
        "row_num_cls": 2,
        "icl_num_blocks": 1,
        "icl_nhead": 2,
        "zero_init": False,
    }
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = TabICL(**config)
    path = tmp_path_factory.mktemp("logits") / "model.ckpt"
    torch.save({"config": config, "state_dict": model.state_dict()}, path)
    return path


def make_classifier(small_checkpoint, **kwargs):
    return TabICLClassifier(
        model_path=small_checkpoint,
        allow_auto_download=False,
        n_estimators=4,
        batch_size=1,
        device="cpu",
        use_amp=False,
        n_jobs=1,
        random_state=0,
        **kwargs,
    )


@pytest.mark.parametrize("average_logits", [True, False])
@pytest.mark.parametrize("temperature", [0.5, 1.7])
@pytest.mark.parametrize("kv_cache", [False, "kv", "repr"])
def test_predict_logits_matches_probabilities(small_checkpoint, average_logits, temperature, kv_cache):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 4)), columns=list("abcd"))
    X.iloc[::5, 1] = np.nan
    y = np.array(["zebra", "ant", "mouse"] * 10)
    clf = make_classifier(
        small_checkpoint,
        average_logits=average_logits,
        softmax_temperature=temperature,
        kv_cache=kv_cache,
    ).fit(X.iloc[:24], y[:24])
    X_test = X.iloc[24:]
    original = X_test.copy()
    probabilities = clf.predict_proba(X_test)

    logits = clf.predict_logits(X_test)

    assert logits.shape == (6, 3)
    assert np.isfinite(logits).all()
    np.testing.assert_allclose(softmax(logits / temperature, axis=-1), probabilities, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(clf.classes_[logits.argmax(axis=-1)], clf.predict(X_test))
    np.testing.assert_array_equal(clf.predict_proba(X_test), probabilities)
    pd.testing.assert_frame_equal(X_test, original)


@pytest.mark.parametrize("average_logits", [True, False])
def test_predict_logits_many_classes(small_checkpoint, average_logits):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(36, 4))
    y = np.tile([20, 5, 90, 40], 9)
    clf = make_classifier(small_checkpoint, average_logits=average_logits, softmax_temperature=0.7).fit(X[:28], y[:28])

    assert clf.n_classes_ > clf.model_.max_classes
    logits = clf.predict_logits(X[28:])

    assert logits.shape == (8, 4)
    assert np.isfinite(logits).all()
    np.testing.assert_allclose(
        softmax(logits / clf.softmax_temperature, axis=-1),
        clf.predict_proba(X[28:]),
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize("average_logits", [True, False])
def test_predict_logits_preserves_ensemble_semantics(monkeypatch, average_logits):
    """Class shuffles and the two averaging orders must produce the right scores."""
    temperature = 0.5
    logits_by_member = np.array([[[2.0, -1.0, 4.0]], [[5.0, 3.0, -2.0]]], dtype=np.float32)
    permutations = [np.array([2, 0, 1]), np.array([1, 2, 0])]
    outputs = logits_by_member if average_logits else softmax(logits_by_member / temperature, axis=-1)
    clf = TabICLClassifier(average_logits=average_logits, softmax_temperature=temperature)
    clf.classes_ = np.array(["a", "b", "c"])
    clf.X_encoder_ = SimpleNamespace(transform=lambda X: X)
    clf.ensemble_generator_ = SimpleNamespace(
        X_=np.zeros((1, 2)),
        class_shuffles_={"none": permutations},
        feature_shuffles_={"none": None},
        transform=lambda X, mode: {"none": (None, None)},
    )
    monkeypatch.setattr(clf, "_batch_forward", lambda *args: outputs)

    result = clf.predict_logits(np.zeros((1, 2)))

    aligned = [logits_by_member[i, :, p].T for i, p in enumerate(permutations)]
    if average_logits:
        # Returning log(p) instead of the original averaged logits would fail this.
        np.testing.assert_allclose(result, np.mean(aligned, axis=0))
    else:
        probabilities = np.mean([softmax(x / temperature, axis=-1) for x in aligned], axis=0)
        np.testing.assert_allclose(result, temperature * np.log(probabilities), rtol=1e-6)


def test_predict_logits_zero_probabilities_are_finite(monkeypatch):
    clf = TabICLClassifier(average_logits=False, softmax_temperature=0.7)
    outputs = np.array([[0.0, 0.25, 0.75]], dtype=np.float32)
    monkeypatch.setattr(clf, "_predict_ensemble", lambda X: outputs)

    with np.errstate(divide="raise", invalid="raise"):
        logits = clf.predict_logits(np.zeros((1, 2)))

    assert np.isfinite(logits).all()
    np.testing.assert_allclose(softmax(logits / 0.7, axis=-1), outputs, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(clf.predict_proba(np.zeros((1, 2))), outputs)


def test_predict_logits_requires_fit():
    with pytest.raises(NotFittedError):
        TabICLClassifier().predict_logits(np.zeros((2, 3)))


def test_predict_logits_rejects_1d_input(small_checkpoint):
    clf = make_classifier(small_checkpoint).fit(np.arange(24).reshape(8, 3), [0, 1] * 4)
    with pytest.raises(ValueError, match="one-dimensional"):
        clf.predict_logits(np.zeros(3))


@pytest.mark.parametrize("average_logits", [True, False])
def test_shap_explains_logits(small_checkpoint, average_logits):
    pytest.importorskip("shap")
    from tabicl.shap import get_shap_explainer

    X = np.random.default_rng(0).normal(size=(12, 2))
    clf = make_classifier(small_checkpoint, average_logits=average_logits).fit(X[:10], [0, 1] * 5)
    explainer = get_shap_explainer(clf, X[:10], predict_fn="predict_logits", algorithm="exact")

    explanation = explainer(X[10:11])

    np.testing.assert_allclose(
        explanation.base_values + explanation.values.sum(axis=1),
        clf.predict_logits(X[10:11]),
        rtol=1e-5,
        atol=1e-7,
    )


def test_shapiq_selects_logits(small_checkpoint):
    pytest.importorskip("shapiq")
    from shapiq.explainer.utils import get_predict_function_and_model_type

    X = np.random.default_rng(0).normal(size=(12, 2))
    clf = make_classifier(small_checkpoint).fit(X[:10], [0, 1] * 5)
    predict, _ = get_predict_function_and_model_type(clf, class_index=1)

    np.testing.assert_allclose(predict(clf, X[10:]), clf.predict_logits(X[10:])[:, 1])
