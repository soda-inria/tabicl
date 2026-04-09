from __future__ import annotations

import numpy as np
import pandas as pd

import softmax_pesudo


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


def test_run_pseudo_label_resplit_inference_reports_accuracy_on_full_test_set():
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

    result = softmax_pesudo.run_pseudo_label_resplit_inference(
        fit_classifier_fn=fit_fn,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_name="demo",
        ce_alpha=0.22,
    )

    assert result["pseudo_applied"] is True
    np.testing.assert_array_equal(result["y_eval"], y_test)
    assert np.mean(result["y_pred"] == result["y_eval"]) == 0.5
