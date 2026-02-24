"""Unified series dispatcher for parallel time series prediction."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Callable

from tabicl.forecast.ts_dataframe import TimeSeriesDataFrame
from tabicl.forecast.preprocessing import separate_target


class SeriesDispatcher:
    """Dispatches per-series inference across CPU cores or GPUs.

    Automatically selects CPU or GPU execution based on hardware
    availability. For CPU, uses ``joblib`` with the ``loky`` backend.
    For GPU, processes sequentially on a single GPU or distributes
    across multiple GPUs via ``joblib``.

    Parameters
    ----------
    inference_fn : callable
        Function that performs inference on a single time series. Expected
        signature: ``(train_X, train_y, test_X, **kwargs) -> dict``.

    num_workers : int | None, default=None
        Number of parallel workers. If ``None``, auto-detects:
        ``min(cpu_count, 8)`` for CPU or ``gpu_count`` for GPU.
    """

    def __init__(self, inference_fn: Callable, num_workers: int | None = None):
        self.inference_fn = inference_fn
        self._use_gpu = torch.cuda.is_available()

        if num_workers is not None:
            self.num_workers = num_workers
        elif self._use_gpu:
            self.num_workers = torch.cuda.device_count()
        else:
            self.num_workers = min(os.cpu_count() or 1, 8)

    def run(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        """Run inference on all time series in the dataset.

        Parameters
        ----------
        train_tsdf : TimeSeriesDataFrame
            Training time series data.

        test_tsdf : TimeSeriesDataFrame
            Test time series data.

        **kwargs
            Additional arguments passed to the inference function.

        Returns
        -------
        TimeSeriesDataFrame
            Predictions for all time series.
        """
        if self._use_gpu:
            predictions = self._dispatch_gpu(train_tsdf, test_tsdf, **kwargs)
        else:
            predictions = self._dispatch_cpu(train_tsdf, test_tsdf, **kwargs)

        predictions = predictions.loc[train_tsdf.item_ids]
        return TimeSeriesDataFrame(predictions)

    def _dispatch_cpu(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """Distribute inference across CPU cores using joblib."""
        n_jobs = min(self.num_workers, len(train_tsdf.item_ids))
        predictions = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._process_single)(
                item_id,
                train_tsdf.loc[item_id],
                test_tsdf.loc[item_id],
                **kwargs,
            )
            for item_id in tqdm(train_tsdf.item_ids, desc="Predicting time series")
        )
        return pd.concat(predictions)

    def _dispatch_gpu(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """Distribute inference across GPUs.

        Single GPU or few items: sequential processing.
        Multiple GPUs: joblib-based parallel chunks.
        """
        total_workers = self.num_workers
        n_items = len(train_tsdf.item_ids)

        if total_workers == 1 or n_items < total_workers:
            return self._run_gpu_batch(train_tsdf, test_tsdf, gpu_id=0, **kwargs)

        rng = np.random.default_rng(0)
        shuffled_ids = rng.permutation(train_tsdf.item_ids)
        chunks = np.array_split(shuffled_ids, min(total_workers, n_items))

        predictions = Parallel(n_jobs=len(chunks), backend="loky")(
            delayed(self._run_gpu_batch)(
                train_tsdf.loc[chunk],
                test_tsdf.loc[chunk],
                gpu_id=i % self.num_workers,
                **kwargs,
            )
            for i, chunk in enumerate(chunks)
        )
        return pd.concat(predictions)

    def _run_gpu_batch(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        gpu_id: int,
        **kwargs,
    ) -> pd.DataFrame:
        """Process a batch of series sequentially on a single GPU."""
        torch.cuda.set_device(gpu_id)
        all_pred = []
        for item_id in tqdm(train_tsdf.item_ids, desc=f"GPU {gpu_id}:"):
            result = self._process_single(
                item_id,
                train_tsdf.loc[item_id],
                test_tsdf.loc[item_id],
                **kwargs,
            )
            all_pred.append(result)

        torch.cuda.empty_cache()
        return pd.concat(all_pred)

    def _process_single(
        self,
        item_id: str,
        single_train: TimeSeriesDataFrame,
        single_test: TimeSeriesDataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """Run inference for a single time series.

        Parameters
        ----------
        item_id : str
            Identifier for the time series.

        single_train : TimeSeriesDataFrame
            Training data for a single time series.

        single_test : TimeSeriesDataFrame
            Test data for a single time series.

        **kwargs
            Additional arguments passed to the inference function.

        Returns
        -------
        pd.DataFrame
            Predictions for the single time series.
        """
        test_index = single_test.index
        train_X, train_y = separate_target(single_train.copy())
        test_X, _ = separate_target(single_test.copy())

        results = self.inference_fn(train_X, train_y, test_X, **kwargs)
        self._validate_output(results)

        result = pd.DataFrame(results, index=test_index)
        result["item_id"] = item_id
        result = result.set_index(["item_id", result.index])

        return result

    @staticmethod
    def _validate_output(inference_output: dict[str, np.ndarray]):
        """Validate the structure of inference output.

        Parameters
        ----------
        inference_output : dict[str, np.ndarray]
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
