from .model.kv_cache import TabICLCache
from .model.inference_config import InferenceConfig

from .sklearn.classifier import TabICLClassifier
from .sklearn.regressor import TabICLRegressor

__all__ = [
    "TabICLClassifier",
    "TabICLRegressor",
    "TabICLForecaster",
    "TabICLUnsupervised",
    "InferenceConfig",
    "TabICLCache",
]


def __getattr__(name):
    if name == "TabICLForecaster":
        try:
            from .forecast.forecaster import TabICLForecaster

            return TabICLForecaster
        except ImportError:
            raise ImportError(
                "TabICLForecaster requires extra dependencies. Install with: pip install tabicl[forecast]"
            ) from None

    if name == "TabICLUnsupervised":
        from .unsupervised import TabICLUnsupervised

        return TabICLUnsupervised

    raise AttributeError(f"module 'tabicl' has no attribute {name}")
