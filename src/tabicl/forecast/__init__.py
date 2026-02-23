try:
    from tabicl.forecast.forecaster import TabICLForecaster
    from tabicl.forecast.ts_dataframe import TimeSeriesDataFrame
    from tabicl.forecast.features import FeatureTransformer
    from tabicl.forecast.plot import plot_forecast
except ImportError:
    raise ImportError(
        "tabicl.forecast requires extra dependencies. Install with: pip install tabicl[forecast]"
    ) from None
