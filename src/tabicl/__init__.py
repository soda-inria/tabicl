from .model.tabicl import TabICL
from .model.kv_cache import TabICLCache
from .model.inference_config import InferenceConfig

from .sklearn.classifier import TabICLClassifier
from .sklearn.regressor import TabICLRegressor


def __getattr__(name):
    if name == "TabICLForecaster":
        try:
            from .forecast.forecaster import TabICLForecaster

            return TabICLForecaster
        except ImportError:
            raise ImportError(
                "TabICLForecaster requires extra dependencies. " "Install with: pip install tabicl[forecast]"
            ) from None
    raise AttributeError(f"module 'tabicl' has no attribute {name}")
