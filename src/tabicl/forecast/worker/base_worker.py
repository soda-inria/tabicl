# Copied from https://github.com/PriorLabs/tabpfn-time-series
"""Base parallel worker for time series prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Callable

from tabicl.forecast.ts_dataframe import TimeSeriesDataFrame
from tabicl.forecast.data_preparation import split_time_series_to_X_y


class ParallelWorker(ABC):
    """Abstract base class for parallel prediction workers.

    Provides a common interface for distributing time series predictions
    across multiple workers (CPU cores or GPUs).

    Parameters
    ----------
    inference_routine : callable
        Function that performs inference on a single time series. Expected
        signature: ``(train_X, train_y, test_X, **kwargs) -> dict``.
    """

    def __init__(self, inference_routine: Callable):
        self.inference_routine = inference_routine

    @abstractmethod
    def predict(self, train_tsdf: TimeSeriesDataFrame, test_tsdf: TimeSeriesDataFrame, **kwargs):
        """Predict on multiple time series.

        Parameters
        ----------
        train_tsdf : TimeSeriesDataFrame
            Training time series data.

        test_tsdf : TimeSeriesDataFrame
            Test time series data.

        **kwargs
            Additional arguments passed to the inference routine.

        Returns
        -------
        TimeSeriesDataFrame
            Predictions for all time series.
        """
        pass

    def _prediction_routine(
        self,
        item_id: str,
        single_train_tsdf: TimeSeriesDataFrame,
        single_test_tsdf: TimeSeriesDataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """Run the prediction routine for a single time series.

        Parameters
        ----------
        item_id : str
            Identifier for the time series.

        single_train_tsdf : TimeSeriesDataFrame
            Training data for a single time series.

        single_test_tsdf : TimeSeriesDataFrame
            Test data for a single time series.

        **kwargs
            Additional arguments passed to the inference routine.

        Returns
        -------
        pd.DataFrame
            Predictions for the single time series.
        """

        test_index = single_test_tsdf.index
        train_X, train_y = split_time_series_to_X_y(single_train_tsdf.copy())
        test_X, _ = split_time_series_to_X_y(single_test_tsdf.copy())
        train_y = train_y.squeeze()

        # TODO: solve the issue of constant target

        results = self.inference_routine(train_X, train_y, test_X, **kwargs)
        self._assert_valid_inference_output(results)
        result = pd.DataFrame(results, index=test_index)
        result["item_id"] = item_id
        result.set_index(["item_id", result.index], inplace=True)

        return result

    def _assert_valid_inference_output(self, inference_output: dict[str, np.ndarray]):
        """Validate the structure of inference output.

        Parameters
        ----------
        inference_output : dict of str to np.ndarray
            Dictionary containing predictions.

        Raises
        ------
        ValueError
            If the output is not a dictionary, is missing the ``"target"``
            key, or contains arrays with mismatched shapes.
        """

        if not isinstance(inference_output, dict):
            raise ValueError("Inference output must be a dictionary")

        if "target" not in inference_output:
            raise ValueError("Inference output must contain a 'target' key")

        if not isinstance(inference_output["target"], np.ndarray):
            raise ValueError("Inference output 'target' must be a numpy array")

        for q, q_pred in inference_output.items():
            if q != "target":
                if not isinstance(q_pred, np.ndarray):
                    raise ValueError(f"Inference output '{q}' must be a numpy array")
                if q_pred.shape != inference_output["target"].shape:
                    raise ValueError(f"Inference output '{q}' must have the same shape as the target")
