# Copied from https://github.com/PriorLabs/tabpfn-time-series
"""GPU-based parallel worker for time series prediction."""
from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from typing import Callable
from joblib import Parallel, delayed
from tqdm import tqdm

from tabicl.forecast.ts_dataframe import TimeSeriesDataFrame
from tabicl.forecast.worker.base_worker import ParallelWorker


class GPUParallelWorker(ParallelWorker):
    """Parallel worker that distributes time series predictions across GPUs.

    Splits time series into chunks and processes them in parallel across
    available GPUs using ``joblib`` with the ``loky`` backend.

    Parameters
    ----------
    inference_routine : callable
        Function that performs inference on a single time series.

    num_gpus : int or None, default=None
        Number of GPUs to use. If ``None``, uses all available GPUs.

    num_workers_per_gpu : int, default=1
        Number of workers per GPU.

    Raises
    ------
    ValueError
        If CUDA is not available.
    """

    def __init__(self, inference_routine: Callable, num_gpus: int = None, num_workers_per_gpu: int = 1):
        super().__init__(inference_routine)

        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.num_workers_per_gpu = num_workers_per_gpu
        self.total_num_workers = self.num_gpus * self.num_workers_per_gpu

        if not torch.cuda.is_available():
            raise ValueError("GPU is required for GPU parallel inference")

    def predict(self, train_tsdf: TimeSeriesDataFrame, test_tsdf: TimeSeriesDataFrame, **kwargs):
        """Predict on multiple time series in parallel using GPUs.

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

        if self.total_num_workers == 1 or len(train_tsdf.item_ids) < self.total_num_workers:
            predictions = self._prediction_routine_per_gpu(train_tsdf, test_tsdf, gpu_id=0, **kwargs)
            return TimeSeriesDataFrame(predictions)

        # Split data into chunks for parallel inference on each GPU
        #   since the time series are of different lengths, we shuffle
        #   the item_ids s.t. the workload is distributed evenly across GPUs
        # Also, using 'min' since num_workers could be larger than the number of time series
        np.random.seed(0)
        item_ids_chunks = np.array_split(
            np.random.permutation(train_tsdf.item_ids),
            min(self.total_num_workers, len(train_tsdf.item_ids)),
        )

        # Run predictions in parallel
        predictions = Parallel(n_jobs=len(item_ids_chunks), backend="loky")(
            delayed(self._prediction_routine_per_gpu)(
                train_tsdf.loc[chunk],
                test_tsdf.loc[chunk],
                gpu_id=i % self.num_gpus,  # Alternate between available GPUs
            )
            for i, chunk in enumerate(item_ids_chunks)
        )
        predictions = pd.concat(predictions)

        # Sort predictions according to original item_ids order
        predictions = predictions.loc[train_tsdf.item_ids]

        return TimeSeriesDataFrame(predictions)

    def _prediction_routine_per_gpu(
        self, train_tsdf: TimeSeriesDataFrame, test_tsdf: TimeSeriesDataFrame, gpu_id: int, **kwargs
    ):
        """Run predictions on a specific GPU.

        Parameters
        ----------
        train_tsdf : TimeSeriesDataFrame
            Training time series data subset.

        test_tsdf : TimeSeriesDataFrame
            Test time series data subset.

        gpu_id : int
            ID of the GPU to use.

        **kwargs
            Additional arguments passed to the inference routine.

        Returns
        -------
        pd.DataFrame
            Concatenated predictions for all time series in the subset.
        """

        # Set GPU
        torch.cuda.set_device(gpu_id)
        all_pred = []
        for item_id in tqdm(train_tsdf.item_ids, desc=f"GPU {gpu_id}:"):
            predictions = self._prediction_routine(item_id, train_tsdf.loc[item_id], test_tsdf.loc[item_id], **kwargs)
            all_pred.append(predictions)

        # Clear GPU cache
        torch.cuda.empty_cache()

        return pd.concat(all_pred)
