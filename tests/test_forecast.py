import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("gluonts")
pytest.importorskip("statsmodels")

from tabicl import TabICLForecaster
from tabicl.forecast import TimeSeriesDataFrame
from tabicl.forecast._engine import ForecastEngine


def test_time_series_forecasting(monkeypatch):
    timestamps = pd.date_range("2024-01-01", periods=12, freq="h")
    df = pd.DataFrame(
        {
            "item_id": np.repeat(["meter_0", "meter_1", "meter_2"], len(timestamps)),
            "timestamp": np.tile(timestamps, 3),
            "target": np.concatenate(
                [
                    np.arange(len(timestamps), dtype=float),
                    np.arange(len(timestamps), dtype=float) + 100,
                    np.arange(len(timestamps), dtype=float) + 200,
                ]
            ),
        }
    )

    data = TimeSeriesDataFrame.from_data_frame(df)

    prediction_length = 3
    selected_items = data.item_ids[:2]
    train_data, test_data = data.train_test_split(prediction_length)

    context_df = train_data.reset_index()
    context_df = context_df[context_df["item_id"].isin(selected_items)]
    test_df = test_data.reset_index()
    test_df = test_df[test_df["item_id"].isin(selected_items)]
    test_df = test_df.groupby("item_id").tail(prediction_length)

    def fake_predict(self, train_tsdf, test_tsdf, quantiles):
        assert train_tsdf.item_ids.equals(selected_items)
        assert test_tsdf.item_ids.equals(selected_items)
        assert quantiles == [0.1, 0.5, 0.9]

        pred = pd.DataFrame(index=test_tsdf.index)
        pred["target"] = np.arange(len(pred), dtype=float)
        for quantile in quantiles:
            pred[quantile] = pred["target"] + quantile
        return TimeSeriesDataFrame(pred)

    monkeypatch.setattr(ForecastEngine, "predict", fake_predict)

    forecaster = TabICLForecaster(max_context_length=10240, temporal_features=["index"])
    pred_df = forecaster.predict_df(
        context_df,
        prediction_length=prediction_length,
        quantiles=[0.1, 0.5, 0.9],
    )

    assert len(test_df) == len(selected_items) * prediction_length
    assert pred_df.index.names == ["item_id", "timestamp"]
    assert pred_df.index.get_level_values("item_id").unique().equals(selected_items)
    assert list(pred_df.columns) == ["target", 0.1, 0.5, 0.9]
    assert len(pred_df) == len(selected_items) * prediction_length

    for item_id in selected_items:
        last_context_timestamp = context_df.loc[context_df["item_id"] == item_id, "timestamp"].max()
        expected_timestamps = pd.date_range(
            last_context_timestamp + pd.Timedelta(hours=1),
            periods=prediction_length,
            freq="h",
            name="timestamp",
        )
        actual_timestamps = pred_df.loc[item_id].index
        pd.testing.assert_index_equal(actual_timestamps, expected_timestamps)
