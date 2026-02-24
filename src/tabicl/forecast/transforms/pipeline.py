from __future__ import annotations

import pandas as pd

from tabicl.forecast.ts_dataframe import TimeSeriesDataFrame
from tabicl.forecast.transforms.base import TimeTransform


class TimeTransformChain:
    """Orchestrates feature generation for time series data.

    Applies a sequence of ``TimeTransform`` instances to both training
    and test data, ensuring consistent feature columns across splits.

    Parameters
    ----------
    transforms : list[TimeTransform]
        Transforms to apply sequentially.
    """

    def __init__(self, transforms: list[TimeTransform]):
        self.transforms = transforms

    def transform(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        target_column: str = "target",
    ) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Transform both training and test data with the configured transforms.

        Parameters
        ----------
        train_tsdf : TimeSeriesDataFrame
            Training time series data.

        test_tsdf : TimeSeriesDataFrame
            Test time series data.

        target_column : str, default="target"
            Name of the target column.

        Returns
        -------
        tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]
            Transformed ``(train_tsdf, test_tsdf)`` with generated features.

        Raises
        ------
        ValueError
            If ``target_column`` is not found in training data or if test
            data contains non-NaN target values.
        """
        self._validate_input(train_tsdf, test_tsdf, target_column)
        static_features = train_tsdf.static_features

        original_train_index = train_tsdf.index
        tsdf = pd.concat([train_tsdf, test_tsdf])

        for t in self.transforms:
            tsdf = tsdf.groupby(level="item_id", group_keys=False).apply(t)

        # Split using the original index to avoid positional drift
        train_slice = tsdf.loc[original_train_index]
        test_slice = tsdf.drop(index=original_train_index)

        train_tsdf = TimeSeriesDataFrame(train_slice, static_features=static_features)
        test_tsdf = TimeSeriesDataFrame(test_slice, static_features=static_features)

        if train_tsdf[target_column].isna().any():
            raise ValueError("All target values in train_tsdf should be non-NaN")
        if not test_tsdf[target_column].isna().all():
            raise ValueError("All target values in test_tsdf should be NaN")

        return train_tsdf, test_tsdf

    @staticmethod
    def _validate_input(
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        target_column: str,
    ):
        """Validate inputs before transformation.

        Parameters
        ----------
        train_tsdf : TimeSeriesDataFrame
            Training time series data.

        test_tsdf : TimeSeriesDataFrame
            Test time series data.

        target_column : str
            Name of the target column.

        Raises
        ------
        ValueError
            If ``target_column`` is not in training data or test data
            contains non-NaN target values.
        """
        if target_column not in train_tsdf.columns:
            raise ValueError(f"Target column '{target_column}' not found in training data")

        if not test_tsdf[target_column].isna().all():
            raise ValueError("Test data should not contain target values")
