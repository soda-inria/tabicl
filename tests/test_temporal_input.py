"""Datetime and timedelta columns of a DataFrame are converted to seconds and reach the model."""

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.base import is_classifier

from tabicl import TabICLClassifier, TabICLRegressor
from tabicl._sklearn.preprocessing import TransformToNumerical


def test_temporal_columns_become_seconds():
    X = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "naive": pd.to_datetime(["1970-01-02", None, "1970-01-04"]),
            "coarse": pd.to_datetime(["1970-01-02", "1970-01-03", "1970-01-04"]).astype("datetime64[s]"),
            # 01:00 in Rome in January is midnight UTC
            "aware": pd.to_datetime(["2020-01-01 01:00", "2020-01-01 02:00", None]).tz_localize("Europe/Rome"),
            "duration": pd.to_timedelta([1, 2, None], unit="h"),
        }
    )
    day, midnight_2020 = 86400.0, 1577836800.0
    expected = np.array(
        [
            # missing values are imputed with the column mean, as for numeric columns
            [1.0, 1 * day, 1 * day, midnight_2020, 3600.0],
            [2.0, 2 * day, 2 * day, midnight_2020 + 3600, 7200.0],
            [3.0, 3 * day, 3 * day, midnight_2020 + 1800, 5400.0],
        ]
    )

    np.testing.assert_array_equal(TransformToNumerical().fit_transform(X), expected)


@pytest.mark.parametrize("Estimator", [TabICLClassifier, TabICLRegressor])
@pytest.mark.parametrize("kind", ["datetime", "timedelta"])
def test_estimator_sees_temporal_column(Estimator, kind):
    """Predictions match those for the same column passed as seconds."""
    rng = np.random.RandomState(0)
    n = 120
    if kind == "datetime":
        values = pd.Series(pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.randint(0, 1500, n), unit="D"))
        seconds = (values - pd.Timestamp("1970-01-01")).dt.total_seconds()
    else:
        values = pd.Series(pd.to_timedelta(rng.randint(0, 1500, n), unit="h"))
        seconds = values.dt.total_seconds()
    X = pd.DataFrame({"noise": rng.randn(n), "t": values})
    X_seconds = X.assign(t=seconds)
    classifier = is_classifier(Estimator())
    # the target depends on the temporal column only
    y = (seconds > seconds.median()).astype(int).to_numpy() if classifier else seconds.to_numpy() / 86400

    def fit_predict(X):
        est = Estimator(n_estimators=2, random_state=0, device="cpu").fit(X[:80], y[:80])
        return est.predict_proba(X[80:]) if classifier else est.predict(X[80:])

    np.testing.assert_allclose(fit_predict(X), fit_predict(X_seconds))


def test_temporal_columns_with_pandas_output():
    """With sklearn's pandas output, rows stay aligned across column types (test rows are not
    indexed from 0)."""
    rng = np.random.RandomState(0)
    n = 100
    X = pd.DataFrame(
        {
            "noise": rng.randn(n),
            "c": rng.choice(["a", "b"], n),
            "t": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.randint(0, 900, n), unit="D"),
        }
    )
    y = (X["t"].dt.year >= 2021).astype(int).to_numpy()

    def fit_predict():
        return TabICLClassifier(n_estimators=1, random_state=0, device="cpu").fit(X[:70], y[:70]).predict_proba(X[70:])

    expected = fit_predict()
    with sklearn.config_context(transform_output="pandas"):
        np.testing.assert_allclose(fit_predict(), expected)
