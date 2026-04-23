from ._model import InferenceConfig
from ._sklearn import TabICLClassifier, TabICLRegressor

__all__ = [
    "TabICLClassifier",
    "TabICLRegressor",
    "TabICLForecaster",
    "TabICLUnsupervised",
    "FinetunedTabICLClassifier",
    "FinetunedTabICLRegressor",
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

    if name in {"FinetunedTabICLClassifier", "FinetunedTabICLRegressor"}:
        from . import _finetune

        return getattr(_finetune, name)

    raise AttributeError(f"module 'tabicl' has no attribute {name}")
