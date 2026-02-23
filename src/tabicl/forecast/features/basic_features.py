# Copied from https://github.com/PriorLabs/tabpfn-time-series
from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import gluonts.time_feature

from tabicl.forecast.features.feature_generator_base import FeatureGenerator


class RunningIndexFeature(FeatureGenerator):
    """Feature generator that adds a ``running_index`` column.

    Assigns a sequential integer index (0, 1, 2, ...) to each row.
    """

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate a running index feature.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with an added ``running_index`` column.
        """

        df = df.copy()
        df["running_index"] = range(len(df))

        return df


class CalendarFeature(FeatureGenerator):
    """Feature generator that creates calendar-based temporal features.

    Extracts calendar components (e.g., year) and encodes seasonal
    patterns (e.g., hour of day, day of week) as sin/cosine pairs using
    ``gluonts.time_feature``.

    Parameters
    ----------
    components : list of str or None, default=None
        Calendar components to extract (e.g., ``["year"]``). If ``None``,
        defaults to ``["year"]``.

    seasonal_features : dict of str to list of float or None, default=None
        Mapping of seasonal feature names to their natural periods. Each
        feature is encoded as sin/cosine pairs. If ``None``, uses default
        temporal features (second, minute, hour, day, week, month).
    """

    def __init__(
        self,
        components: Optional[List[str]] = None,
        seasonal_features: Optional[Dict[str, List[float]]] = None,
    ):
        self.components = components or ["year"]
        self.seasonal_features = seasonal_features or {
            # (feature, natural seasonality)
            "second_of_minute": [60],
            "minute_of_hour": [60],
            "hour_of_day": [24],
            "day_of_week": [7],
            "day_of_month": [30.5],
            "day_of_year": [365],
            "week_of_year": [52],
            "month_of_year": [12],
        }

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate calendar features from timestamps.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with a ``timestamp`` level in the index.

        Returns
        -------
        pd.DataFrame
            DataFrame with added calendar component and sin/cosine columns.
        """

        df = df.copy()
        timestamps = df.index.get_level_values("timestamp")

        # Add basic calendar components
        for component in self.components:
            df[component] = getattr(timestamps, component)

        # Add seasonal features
        for feature_name, periods in self.seasonal_features.items():
            feature_func = getattr(gluonts.time_feature, f"{feature_name}_index")
            feature = feature_func(timestamps).astype(np.int32)

            if periods is not None:
                for period in periods:
                    period = period - 1  # Adjust for 0-based indexing
                    df[f"{feature_name}_sin"] = np.sin(2 * np.pi * feature / period)
                    df[f"{feature_name}_cos"] = np.cos(2 * np.pi * feature / period)
            else:
                df[feature_name] = feature

        return df


class AdditionalCalendarFeature(CalendarFeature):
    """Extended calendar feature generator with additional seasonal features.

    Inherits from ``CalendarFeature`` and merges additional seasonal
    features with the defaults.

    Parameters
    ----------
    components : list of str or None, default=None
        Calendar components to extract.

    additional_seasonal_features : dict of str to list of float or None, default=None
        Additional seasonal features to merge with the defaults.
    """

    def __init__(
        self,
        components: Optional[List[str]] = None,
        additional_seasonal_features: Optional[Dict[str, List[float]]] = None,
    ):
        super().__init__(components=components)
        self.seasonal_features = {**additional_seasonal_features, **self.seasonal_features}


class PeriodicSinCosineFeature(FeatureGenerator):
    """Feature generator that creates sin/cosine features for given periods.

    For each period, adds ``sin_{period}`` and ``cos_{period}`` columns
    based on the row index position.

    Parameters
    ----------
    periods : list of float
        Periods for which to generate sin/cosine features.

    name_suffix : str or None, default=None
        If provided, column names use ``sin_{suffix}_{i}`` instead of
        ``sin_{period}``.
    """

    def __init__(self, periods: List[float], name_suffix: str = None):
        self.periods = periods
        self.name_suffix = name_suffix

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate sin/cosine features for the configured periods.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with added sin/cosine columns for each period.
        """

        df = df.copy()
        for i, period in enumerate(self.periods):
            name_suffix = f"{self.name_suffix}_{i}" if self.name_suffix else f"{period}"
            df[f"sin_{name_suffix}"] = np.sin(2 * np.pi * np.arange(len(df)) / period)
            df[f"cos_{name_suffix}"] = np.cos(2 * np.pi * np.arange(len(df)) / period)

        return df
