# Copied from https://github.com/PriorLabs/tabpfn-time-series
from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from tabicl.forecast.ts_dataframe import TimeSeriesDataFrame
from tabicl.forecast.features.feature_generator_base import FeatureGenerator


class FeatureTransformer:
    """Orchestrates feature generation for time series data.

    Applies a sequence of ``FeatureGenerator`` instances to both training
    and test data, ensuring consistent feature columns across splits.

    Parameters
    ----------
    feature_generators : list of FeatureGenerator
        Feature generators to apply sequentially.
    """

    def __init__(self, feature_generators: List[FeatureGenerator]):
        self.feature_generators = feature_generators

    def transform(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        target_column: str = "target",
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Transform both training and test data with the configured feature generators.

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
        tuple of (TimeSeriesDataFrame, TimeSeriesDataFrame)
            Transformed ``(train_tsdf, test_tsdf)`` with generated features.

        Raises
        ------
        ValueError
            If ``target_column`` is not found in training data or if test
            data contains non-NaN target values.
        """

        self._validate_input(train_tsdf, test_tsdf, target_column)
        static_features = train_tsdf.static_features
        tsdf = pd.concat([train_tsdf, test_tsdf])

        # Apply all feature generators
        for generator in self.feature_generators:
            tsdf = tsdf.groupby(level="item_id", group_keys=False).apply(generator)

        # Split train and test tsdf
        train_slice = tsdf.iloc[: len(train_tsdf)]
        test_slice = tsdf.iloc[len(train_tsdf) :]

        # Convert back to TimeSeriesDataFrame and re-attach static features
        # This ensures the metadata remains intact even if generators returned a standard DF
        train_tsdf = TimeSeriesDataFrame(train_slice, static_features=static_features)
        test_tsdf = TimeSeriesDataFrame(test_slice, static_features=static_features)

        assert not train_tsdf[target_column].isna().any(), "All target values in train_tsdf should be non-NaN"
        assert test_tsdf[target_column].isna().all()

        return train_tsdf, test_tsdf

    @staticmethod
    def _validate_input(train_tsdf: TimeSeriesDataFrame, test_tsdf: TimeSeriesDataFrame, target_column: str):
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
