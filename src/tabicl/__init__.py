from ._model import InferenceConfig
from ._sklearn import TabICLClassifier, TabICLRegressor

__all__ = [
    "TabICLClassifier",
    "TabICLRegressor",
    "TabICLForecaster",
    "TabICLUnsupervised",
    "InferenceConfig",
]


def __getattr__(name):
    if name == "TabICLForecaster":
        try:
            from .forecast import TabICLForecaster

            return TabICLForecaster
        except ImportError:
            raise ImportError(
                "TabICLForecaster requires extra dependencies. Install with: pip install tabicl[forecast]"
            ) from None

    if name == "TabICLUnsupervised":
        from ._unsupervised import TabICLUnsupervised

        return TabICLUnsupervised

    raise AttributeError(f"module 'tabicl' has no attribute {name}")
