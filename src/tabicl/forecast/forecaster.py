from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd

from tabicl.forecast.ts_dataframe import TimeSeriesDataFrame
from tabicl.forecast.preprocessing import build_horizon
from tabicl.forecast.transforms.base import TimeTransform
from tabicl.forecast.transforms import AutoPeriodicEncoder, DatetimeEncoder, IndexEncoder, TimeTransformChain
from tabicl.forecast.engine import ForecastEngine


_DEFAULT_TRANSFORMS: tuple[TimeTransform, ...] = (
    IndexEncoder(),
    DatetimeEncoder(),
    AutoPeriodicEncoder(),
)


logger = logging.getLogger(__name__)


class TabICLForecaster:
    """TabICL-based time series forecasting pipeline.

    This pipeline uses TabICL for zero-shot time series forecasting.
    It handles the entire prediction workflow:

    1. Data preprocessing (missing value handling, context slicing)
    2. Feature engineering (calendar features, seasonal detection)
    3. Prediction with probabilistic quantile estimates

    Parameters
    ----------
    max_context_length : int, default=4096
        Maximum number of historical timesteps to use as context. The pipeline
        automatically slices to the last ``max_context_length`` timesteps if the
        historical data is longer.

    temporal_features : list[TimeTransform] | None, default=None
        Feature transforms to apply to the time series. If ``None``, uses
        ``(IndexEncoder(), DatetimeEncoder(), AutoPeriodicEncoder())``.

    point_estimate : {"mean", "median"}, default="mean"
        Method to select the point prediction from TabICL output.

    tabicl_config : dict | None, default=None
        Configuration for ``TabICLRegressor`` initialization. If None, defaults to
        empty dict (uses default settings).

    Notes
    -----
    - For time series with irregular timestamps, consider opting out of
      ``AutoPeriodicEncoder``.
    """

    def __init__(
        self,
        max_context_length: int = 4096,
        temporal_features: list[TimeTransform] | None = None,
        point_estimate: Literal["mean", "median"] = "mean",
        tabicl_config: dict | None = None,
    ):
        if tabicl_config is None:
            tabicl_config = {}
        if temporal_features is None:
            temporal_features = list(_DEFAULT_TRANSFORMS)

        self.max_context_length = max_context_length
        self.predictor = ForecastEngine(tabicl_config=tabicl_config, point_estimate=point_estimate)
        self.feature_transformer = TimeTransformChain(temporal_features)

    @staticmethod
    def _impute_missing_targets(tsdf: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Handle missing values in a TimeSeriesDataFrame.

        Strategy:

        - If a series has <= 1 valid values: fill NaNs with 0
        - Otherwise: drop rows with NaN targets

        Parameters
        ----------
        tsdf : TimeSeriesDataFrame
            Time series data with potential NaN values in the ``target`` column.

        Returns
        -------
        TimeSeriesDataFrame
            Data with NaNs handled.
        """
        valid_counts = tsdf["target"].notna().groupby(level="item_id").sum()
        invalid_items = valid_counts[valid_counts <= 1].index

        result = tsdf.copy()

        if len(invalid_items) > 0:
            mask_to_fill = result.index.get_level_values("item_id").isin(invalid_items) & result["target"].isna()
            result.loc[mask_to_fill, "target"] = 0

        result = result[result["target"].notna()]
        return result

    def _prepare_context(self, context_tsdf: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Impute missing targets and slice to max context length."""
        context_tsdf = self._impute_missing_targets(context_tsdf)
        if context_tsdf.target.isnull().any():
            raise ValueError("Context still contains NaN targets after imputation")
        return context_tsdf.slice_by_timestep(-self.max_context_length, None)

    @staticmethod
    def _prepare_future(future_tsdf: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Ensure future data has an all-NaN target column."""
        future_tsdf = future_tsdf.copy()
        if "target" in future_tsdf.columns and future_tsdf["target"].notna().any():
            raise ValueError(
                "future_tsdf: All entries in 'target' must be NaN for the prediction horizon. "
                "Got at least one non-NaN in 'target'."
            )
        future_tsdf["target"] = np.nan
        return future_tsdf

    @staticmethod
    def _align_covariates(
        context_tsdf: TimeSeriesDataFrame,
        future_tsdf: TimeSeriesDataFrame,
    ) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Keep only covariates present in both context and future data."""
        valid_covariates = context_tsdf.columns.intersection(future_tsdf.columns).drop("target").tolist()
        logger.info(f"Valid covariates: {valid_covariates}")

        if not context_tsdf.target.notna().all():
            raise ValueError("All target values in context_tsdf must be present (no missing values).")
        context_tsdf = context_tsdf.fill_missing_values(method="constant", value=np.nan)

        if future_tsdf[valid_covariates].isnull().any().any():
            warnings.warn(
                "Some covariate values in future_tsdf are missing (NaN). This may affect prediction quality.",
                UserWarning,
                stacklevel=2,
            )

        return context_tsdf, future_tsdf

    @staticmethod
    def _ensure_item_id(df: pd.DataFrame, item_id_column: str) -> pd.DataFrame:
        """Add a dummy ``item_id`` column for single time series.

        When users provide a DataFrame without an ``item_id`` column, this adds
        a dummy column with value 0 to enable uniform processing.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame without an ``item_id`` column.

        item_id_column : str
            Name of the ``item_id`` column to add.

        Returns
        -------
        pd.DataFrame
            DataFrame with the added ``item_id`` column.
        """
        df = df.copy()
        df[item_id_column] = 0
        return df

    def predict(
        self,
        context_tsdf: TimeSeriesDataFrame,
        future_tsdf: TimeSeriesDataFrame,
        quantiles: list[float] | None = None,
    ) -> TimeSeriesDataFrame:
        """Generate forecasts using TimeSeriesDataFrame objects.

        This is the core prediction method. For a simpler pandas DataFrame
        interface, see ``predict_df``.

        Parameters
        ----------
        context_tsdf : TimeSeriesDataFrame
            Historical time series data used as context for prediction. Must
            contain a ``target`` column with historical values. May contain
            additional covariate columns.

        future_tsdf : TimeSeriesDataFrame
            Future timestamps for which to generate predictions. Should contain
            the same covariate columns as ``context_tsdf``. The ``target``
            column should be NaN (will be filled with predictions).

        quantiles : list[float] | None, default=None
            Quantiles to predict for probabilistic forecasting.
            Defaults to ``[0.1, 0.2, ..., 0.9]``.

        Returns
        -------
        TimeSeriesDataFrame
            Predictions containing:

            - ``target``: Point predictions (mean by default)
            - One column per quantile (e.g., ``0.1``, ``0.9``) for prediction
              intervals

        Notes
        -----
        - Only covariates present in both ``context_tsdf`` and ``future_tsdf``
          will be used.
        - Context is automatically sliced to ``max_context_length`` if longer.
        - Missing values in context are handled automatically.
        """
        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Preprocess
        context_tsdf = self._prepare_context(context_tsdf)
        future_tsdf = self._prepare_future(future_tsdf)
        context_tsdf, future_tsdf = self._align_covariates(context_tsdf, future_tsdf)

        # Featurization
        context_tsdf, future_tsdf = self.feature_transformer.transform(context_tsdf, future_tsdf)

        # Prediction
        return self.predictor.predict(train_tsdf=context_tsdf, test_tsdf=future_tsdf, quantiles=quantiles)

    def predict_df(
        self,
        context_df: pd.DataFrame,
        future_df: pd.DataFrame | None = None,
        prediction_length: int | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts from pandas DataFrames.

        This is the recommended user-facing API. It accepts standard pandas
        DataFrames and returns predictions as a DataFrame.

        Parameters
        ----------
        context_df : pd.DataFrame
            Historical time series data. Required columns:

            - ``timestamp``: Timestamps for each observation (datetime).
            - ``target``: Historical values to forecast from (numeric).
            - ``item_id`` (optional): Identifier for multiple time series.
              If omitted, assumes a single time series.
            - Additional columns are treated as known covariates.

        future_df : pd.DataFrame or None, default=None
            Future timestamps for prediction. Required columns:

            - ``timestamp``: Future timestamps to forecast (datetime).
            - ``item_id`` (optional): Must match item_ids in ``context_df``.
            - Covariate columns matching those in ``context_df``.

            Mutually exclusive with ``prediction_length``. Use this when you
            have known future covariate values or irregular timestamps.

        prediction_length : int or None, default=None
            Number of time steps to forecast into the future. Mutually exclusive
            with ``future_df``. Use this for simple forecasting when you don't
            have future covariates.

        quantiles : list[float] | None, default=None
            Quantiles to predict for uncertainty estimation.
            Defaults to ``[0.1, 0.2, ..., 0.9]``.

        Returns
        -------
        pd.DataFrame
            Forecasts indexed by (item_id, timestamp) containing:

            - ``target``: Point predictions (mean by default)
            - One column per quantile for prediction intervals

        Raises
        ------
        ValueError
            If both or neither of ``future_df`` and ``prediction_length`` are
            provided.
        """
        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        if (future_df is None) == (prediction_length is None):
            raise ValueError("Provide exactly one of future_df or prediction_length")

        # Handle single-series case (no item_id column)
        if "item_id" not in context_df.columns:
            context_df = self._ensure_item_id(context_df, "item_id")

        context_tsdf = TimeSeriesDataFrame.from_data_frame(context_df)

        if prediction_length is not None:
            future_tsdf = build_horizon(context_tsdf, prediction_length=prediction_length)
        else:
            if "item_id" not in future_df.columns:
                future_df = self._ensure_item_id(future_df, "item_id")
            future_tsdf = TimeSeriesDataFrame.from_data_frame(future_df)

        pred = self.predict(context_tsdf, future_tsdf, quantiles=quantiles)
        result = pred.to_data_frame()

        return result
