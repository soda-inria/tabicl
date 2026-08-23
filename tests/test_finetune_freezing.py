"""Tests for finetuning with partial module freezing (GitHub issue #128)."""

import numpy as np
import pytest

pytest.importorskip("transformers", reason="finetune extra not installed")
from tabicl import FinetunedTabICLClassifier, FinetunedTabICLRegressor

from conftest import model_path


@pytest.mark.parametrize(
    "freeze_col,freeze_row,freeze_icl",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ],
)
def test_classifier_partial_freezing(freeze_col, freeze_row, freeze_icl):
    """Partial freezing should not raise autograd errors (issue #128)."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 5))
    y = rng.choice([0, 1], size=60)

    est = FinetunedTabICLClassifier(
        epochs=1,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_inference=1,
        early_stopping=False,
        amp=False,
        freeze_col=freeze_col,
        freeze_row=freeze_row,
        freeze_icl=freeze_icl,
        **model_path("classifier"),
    )
    est.fit(X, y)
    preds = est.predict(X)
    assert preds.shape == (60,)


@pytest.mark.parametrize(
    "freeze_col,freeze_row,freeze_icl",
    [
        (True, True, False),
        (True, False, True),
    ],
)
def test_regressor_partial_freezing(freeze_col, freeze_row, freeze_icl):
    """Partial freezing should not raise autograd errors for regressor."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 5))
    y = rng.normal(size=60)

    est = FinetunedTabICLRegressor(
        epochs=1,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_inference=1,
        early_stopping=False,
        amp=False,
        freeze_col=freeze_col,
        freeze_row=freeze_row,
        freeze_icl=freeze_icl,
        **model_path("regressor"),
    )
    est.fit(X, y)
    preds = est.predict(X)
    assert preds.shape == (60,)
