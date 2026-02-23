# Copied from https://github.com/PriorLabs/tabpfn-time-series
from __future__ import annotations

import pandas as pd
import numpy as np

from tabicl.forecast.ts_dataframe import TimeSeriesDataFrame


def generate_test_X(train_tsdf: TimeSeriesDataFrame, prediction_length: int) -> TimeSeriesDataFrame:
    """Generate test features for time series forecasting.

    Creates a ``TimeSeriesDataFrame`` with future timestamps extending
    beyond the training data, with NaN targets to be filled by predictions.

    Parameters
    ----------
    train_tsdf : TimeSeriesDataFrame
        Training time series data used to determine the frequency and
        last timestamps.

    prediction_length : int
        Number of future time steps to generate for each time series.

    Returns
    -------
    TimeSeriesDataFrame
        Test data with NaN targets and timestamps extending from the end
        of each training series.
    """

    freq = train_tsdf.freq
    item_ids = train_tsdf.item_ids
    n_items = len(item_ids)

    # Get last timestamp for each item_id in one groupby operation (instead of xs per item)
    last_timestamps = (
        train_tsdf.reset_index()
        .groupby("item_id", sort=False)["timestamp"]
        .max()
        .loc[item_ids]  # Ensure same order as item_ids
        .values
    )

    # Compute the frequency offset
    freq_offset = pd.tseries.frequencies.to_offset(freq)

    # Pre-allocate timestamp array
    all_timestamps = np.empty(n_items * prediction_length, dtype="datetime64[ns]")

    for i, last_ts in enumerate(last_timestamps):
        start_ts = last_ts + freq_offset
        timestamps = pd.date_range(start=start_ts, periods=prediction_length, freq=freq)
        all_timestamps[i * prediction_length : (i + 1) * prediction_length] = timestamps.values

    # Build single DataFrame with vectorized arrays
    test_df = pd.DataFrame(
        {
            "target": np.full(n_items * prediction_length, np.nan),
            "timestamp": all_timestamps,
            "item_id": np.repeat(item_ids.values, prediction_length),
        }
    )
    test_tsdf = TimeSeriesDataFrame.from_data_frame(test_df)
    assert test_tsdf.item_ids.equals(train_tsdf.item_ids)

    return test_tsdf


def split_time_series_to_X_y(df: pd.DataFrame, target_col: str = "target") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series DataFrame into features and target.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features and target.

    target_col : str, default="target"
        Name of the target column.

    Returns
    -------
    X : pd.DataFrame
        Features (all columns except ``target_col``).

    y : pd.DataFrame
        Target values (single-column DataFrame).

    """
    X = pd.DataFrame(df.drop(columns=[target_col]))
    y = pd.DataFrame(df[target_col])

    return X, y
