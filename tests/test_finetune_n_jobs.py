"""Tests for the ``n_jobs`` parameter of the fine-tuning estimators (GitHub issue #164)."""

import numpy as np
import pytest
import torch
import transformers  # noqa: F401
from sklearn.base import clone

from tabicl import FinetunedTabICLClassifier, FinetunedTabICLRegressor

from conftest import model_path


@pytest.mark.parametrize("cls", [FinetunedTabICLClassifier, FinetunedTabICLRegressor])
@pytest.mark.parametrize("n_jobs", [None, 1, -1])
def test_n_jobs_is_forwarded_to_inner_estimator(cls, n_jobs):
    est = cls(n_jobs=n_jobs)
    assert est.get_params()["n_jobs"] == n_jobs
    assert clone(est).n_jobs == n_jobs

    inner = est._create_inner_estimator(n_estimators=1, device=torch.device("cpu"))
    assert inner.n_jobs == n_jobs


def test_n_jobs_fit_predict_restores_threads():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = rng.choice([0, 1], size=60)

    est = FinetunedTabICLClassifier(
        epochs=1,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_inference=1,
        early_stopping=False,
        amp=False,
        device="cpu",
        n_jobs=1,
        **model_path("classifier"),
    )
    est.fit(X, y)
    assert est._final_estimator_.n_jobs == 1

    n_threads_before = torch.get_num_threads()
    preds = est.predict(X)
    assert preds.shape == (60,)
    assert torch.get_num_threads() == n_threads_before
