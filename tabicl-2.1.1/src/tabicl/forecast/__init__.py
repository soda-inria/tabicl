"""TabICL time series forecasting pipeline."""

try:
    from tabicl.forecast._forecaster import TabICLForecaster
    from tabicl.forecast._ts_dataframe import TimeSeriesDataFrame
    from tabicl.forecast.transforms import TimeTransformChain
    from tabicl.forecast._plotting import plot_forecast
except ImportError:
    raise ImportError(
        "tabicl.forecast requires extra dependencies. Install with: pip install tabicl[forecast]"
    ) from None

__all__ = ["TabICLForecaster", "TimeSeriesDataFrame", "TimeTransformChain", "plot_forecast"]
