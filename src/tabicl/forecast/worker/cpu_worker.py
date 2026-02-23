# Copied from https://github.com/PriorLabs/tabpfn-time-series
"""CPU-based parallel worker for time series prediction."""
from __future__ import annotations

import pandas as pd
from typing import Callable
from joblib import Parallel, delayed
from tqdm import tqdm

from tabicl.forecast.ts_dataframe import TimeSeriesDataFrame
from tabicl.forecast.worker.base_worker import ParallelWorker


class CPUParallelWorker(ParallelWorker):
    """Parallel worker that distributes time series predictions across CPU cores.

    Uses ``joblib`` with the ``loky`` backend for parallel execution.

    Parameters
    ----------
    inference_routine : callable
        Function that performs inference on a single time series.

    num_workers : int, default=8
        Number of parallel workers to use.
    """

    _DEFAULT_NUM_WORKERS = 8

    def __init__(self, inference_routine: Callable, num_workers: int = _DEFAULT_NUM_WORKERS):
        super().__init__(inference_routine)
        self.num_workers = num_workers

    def predict(self, train_tsdf: TimeSeriesDataFrame, test_tsdf: TimeSeriesDataFrame, **kwargs):
        """Predict on multiple time series in parallel using CPU cores.

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

        predictions = Parallel(n_jobs=min(self.num_workers, len(train_tsdf.item_ids)), backend="loky")(
            delayed(self._prediction_routine)(
                item_id,
                train_tsdf.loc[item_id],
                test_tsdf.loc[item_id],
                **kwargs,
            )
            for item_id in tqdm(train_tsdf.item_ids, desc="Predicting time series")
        )
        predictions = pd.concat(predictions)

        # Sort predictions according to original item_ids order
        predictions = predictions.loc[train_tsdf.item_ids]

        return TimeSeriesDataFrame(predictions)
