from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from tabicl import TabICLRegressor
from tabicl.forecast.ts_dataframe import TimeSeriesDataFrame
from tabicl.forecast.dispatch import SeriesDispatcher


logger = logging.getLogger(__name__)


def _to_numpy(obj: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
    """Convert a DataFrame or Series to a numpy array, passing arrays through."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.values
    return obj


class ForecastEngine:
    """Engine that generates forecasts for multiple time series in parallel.

    Uses ``TabICLRegressor`` for zero-shot time series forecasting with
    quantile predictions, distributed across CPUs or GPUs.

    Parameters
    ----------
    tabicl_config : dict | None, default=None
        Configuration for ``TabICLRegressor`` initialization. If ``None``,
        defaults to empty dict.

    point_estimate : {"mean", "median"}, default="mean"
        Which output to use as the point prediction.
    """

    def __init__(
        self,
        tabicl_config: dict | None = None,
        point_estimate: str = "mean",
    ):
        if point_estimate not in ("mean", "median"):
            raise ValueError(f"point_estimate must be 'mean' or 'median', got {point_estimate}")

        self._tabicl_config = tabicl_config or {}
        self._point_estimate = point_estimate

        self._dispatcher = SeriesDispatcher(inference_fn=self._inference_routine)

    def _inference_routine(
        self,
        train_X: np.ndarray | pd.DataFrame,
        train_y: np.ndarray | pd.Series,
        test_X: np.ndarray | pd.DataFrame,
        quantiles: list[float] | None = None,
    ) -> dict[str | float, np.ndarray]:
        """Train a TabICLRegressor and make predictions."""
        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        train_X_np, train_y_np, test_X_np = map(_to_numpy, (train_X, train_y, test_X))

        model = TabICLRegressor(**self._tabicl_config)
        model.fit(train_X_np, train_y_np)
        raw = model.predict(test_X_np, output_type=[self._point_estimate, "quantiles"], alphas=quantiles)

        result = {"target": raw[self._point_estimate]}
        result.update({q: raw["quantiles"][:, i] for i, q in enumerate(quantiles)})

        return result

    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        quantiles: list[float] | None = None,
    ) -> TimeSeriesDataFrame:
        """Generate predictions for each time series.

        Parameters
        ----------
        train_tsdf : TimeSeriesDataFrame
            Training data for each time series.

        test_tsdf : TimeSeriesDataFrame
            Forecasting horizon for each time series.

        quantiles : list[float] | None, default=None
            Quantiles to compute for probabilistic prediction.
            Defaults to ``[0.1, 0.2, ..., 0.9]``.

        Returns
        -------
        TimeSeriesDataFrame
            Predictions containing point forecasts and quantile columns.
        """
        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        self._validate_quantiles(quantiles)
        return self._dispatcher.run(train_tsdf=train_tsdf, test_tsdf=test_tsdf, quantiles=quantiles)

    @staticmethod
    def _validate_quantiles(quantiles: list[float]):
        """Validate the quantiles."""
        if not isinstance(quantiles, list):
            raise ValueError("Quantiles must be a list")
        if not all(isinstance(q, float) for q in quantiles):
            raise ValueError("Quantiles must be a list of floats")
        if not all(0 <= q <= 1 for q in quantiles):
            raise ValueError("Quantiles must be between 0 and 1")
