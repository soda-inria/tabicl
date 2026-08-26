"""Tests for fine-tuning early stopping behavior with NaN validation metrics."""

import numpy as np
import pytest
import transformers  # noqa: F401

from tabicl import TabICLClassifier, FinetunedTabICLClassifier, FinetunedTabICLRegressor

from conftest import model_path


def test_early_stopping_false_with_nan_validation_classifier():
    """Verify early_stopping=False keeps final weights when validation metric is NaN."""
    rng = np.random.default_rng(42)

    # Create normal multi-class training data
    X_train = rng.normal(size=(60, 5))
    y_train = rng.choice([0, 1, 2], size=60)

    # Single-class validation set (ROC-AUC will be NaN)
    X_val = rng.normal(size=(20, 5))
    y_val = np.zeros(20, dtype=int)  # All zeros - single class

    # Test data for comparing predictions
    X_test = rng.normal(size=(30, 5))

    # First, get predictions from a non-finetuned classifier (pretrained weights)
    clf_pretrained = TabICLClassifier(
        n_estimators=1,
        **model_path("classifier"),
    )
    clf_pretrained.fit(X_train, y_train)
    pred_pretrained = clf_pretrained.predict_proba(X_test)

    # Now fit with fine-tuning and NaN validation
    clf_finetuned = FinetunedTabICLClassifier(
        early_stopping=False,
        eval_metric="roc_auc",
        epochs=2,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_inference=1,
        amp=False,
        verbose=True,
        **model_path("classifier"),
    )

    # Should not raise despite NaN validation metric
    clf_finetuned.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # Verify _best_metric_ is NaN
    assert np.isnan(clf_finetuned._best_metric_)

    # Get predictions from fine-tuned model
    pred_finetuned = clf_finetuned.predict_proba(X_test)

    # Verify predictions are different (proving fine-tuning actually changed weights)
    # If they were identical, it would mean the model reverted to pretrained weights
    assert not np.allclose(pred_pretrained, pred_finetuned, rtol=1e-5, atol=1e-5), (
        "Fine-tuned model makes identical predictions to pretrained model, "
        "indicating it reverted to pretrained weights despite early_stopping=False"
    )


def test_early_stopping_true_with_baseline_nan_raises():
    """Verify early_stopping=True raises error when baseline validation metric is NaN.

    Note: If a metric becomes NaN mid-training (after some valid epochs), the
    training loop stops gracefully with a warning instead of raising. This
    handles training instability (e.g., high learning rate, exploding gradients).
    Baseline NaN indicates a validation setup issue that must be fixed.
    """
    rng = np.random.default_rng(42)

    # Create normal multi-class training data
    X_train = rng.normal(size=(60, 5))
    y_train = rng.choice([0, 1, 2], size=60)

    # Single-class validation set (ROC-AUC will be NaN)
    X_val = rng.normal(size=(20, 5))
    y_val = np.zeros(20, dtype=int)  # All zeros - single class

    clf = FinetunedTabICLClassifier(
        early_stopping=True,
        eval_metric="roc_auc",
        epochs=2,
        patience=1,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_inference=1,
        amp=False,
        **model_path("classifier"),
    )

    # Should raise ValueError immediately when baseline metric is NaN
    with pytest.raises(ValueError, match="Early stopping enabled but validation metric.*is NaN"):
        clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)


def test_early_stopping_false_with_valid_metrics_works():
    """Verify early_stopping=False works normally when validation metrics are valid."""
    rng = np.random.default_rng(42)

    # Create normal multi-class training and validation data
    X_train = rng.normal(size=(60, 5))
    y_train = rng.choice([0, 1, 2], size=60)
    X_val = rng.normal(size=(20, 5))
    y_val = rng.choice([0, 1, 2], size=20)  # Multi-class validation

    clf = FinetunedTabICLClassifier(
        early_stopping=False,
        eval_metric="roc_auc",
        epochs=2,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_inference=1,
        amp=False,
        **model_path("classifier"),
    )

    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # Verify _best_metric_ is not NaN
    assert not np.isnan(clf._best_metric_)

    # Verify predictions work
    predictions = clf.predict(X_val)
    assert predictions.shape == (20,)
