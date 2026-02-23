from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pandas as pd
import torch

from tabicl import TabICLRegressor
from tabicl.forecast.ts_dataframe import TimeSeriesDataFrame
from tabicl.forecast.worker import CPUParallelWorker, GPUParallelWorker


logger = logging.getLogger(__name__)


class TabICLTimeSeriesPredictor:
    """Predictor that generates forecasts for multiple time series in parallel.

    Uses ``TabICLRegressor`` for zero-shot time series forecasting with
    quantile predictions, distributed across CPUs or GPUs.

    Parameters
    ----------
    tabicl_config : dict, default={}
        Configuration for ``TabICLRegressor`` initialization.

    output_selection : {"mean", "median"}, default="mean"
        Which output to use as the point prediction.
    """

    def __init__(self, tabicl_config: dict = {}, output_selection: str = "mean"):
        if output_selection not in ("mean", "median"):
            raise ValueError(f"output_selection must be 'mean' or 'median', got {output_selection}")

        self._tabicl_config = tabicl_config
        self._output_selection = output_selection

        worker_class = GPUParallelWorker if torch.cuda.is_available() else CPUParallelWorker
        self._worker = worker_class(inference_routine=self._inference_routine)

    def _inference_routine(
        self,
        train_X: Union[np.ndarray, pd.DataFrame],
        train_y: Union[np.ndarray, pd.Series],
        test_X: Union[np.ndarray, pd.DataFrame],
        quantiles: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ) -> dict[str | float, np.ndarray]:
        """Train a TabICLRegressor and make predictions."""
        model = TabICLRegressor(**self._tabicl_config)

        # Convert DataFrames to numpy
        if isinstance(train_X, pd.DataFrame):
            train_X = train_X.values
        if isinstance(train_y, pd.Series):
            train_y = train_y.values
        if isinstance(test_X, pd.DataFrame):
            test_X = test_X.values

        # Fit and predict
        model.fit(train_X, train_y)
        raw = model.predict(test_X, output_type=[self._output_selection, "quantiles"], alphas=quantiles)

        # Postprocess
        result = {"target": raw[self._output_selection]}
        for i, q in enumerate(quantiles):
            result[q] = raw["quantiles"][:, i]

        return result

    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        quantiles: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ) -> TimeSeriesDataFrame:
        """Generate predictions for each time series.

        Parameters
        ----------
        train_tsdf : TimeSeriesDataFrame
            Training data for each time series.

        test_tsdf : TimeSeriesDataFrame
            Forecasting horizon for each time series.

        quantiles : list of float, default=[0.1, 0.2, ..., 0.9]
            Quantiles to compute for probabilistic prediction.

        Returns
        -------
        TimeSeriesDataFrame
            Predictions containing point forecasts and quantile columns.
        """
        self._validate_quantiles(quantiles)
        return self._worker.predict(train_tsdf=train_tsdf, test_tsdf=test_tsdf, quantiles=quantiles)

    def _validate_quantiles(self, quantiles: list[float]):
        """Validate the quantiles."""
        if not isinstance(quantiles, list):
            raise ValueError("Quantiles must be a list")
        if not all(isinstance(q, float) for q in quantiles):
            raise ValueError("Quantiles must be a list of floats")
        if not all(0 <= q <= 1 for q in quantiles):
            raise ValueError("Quantiles must be between 0 and 1")
