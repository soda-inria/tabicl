"""TabICL unsupervised learning: imputation, outlier detection, and synthetic data generation."""

__all__ = ["TabICLUnsupervised"]

_LAZY_IMPORTS = {"TabICLUnsupervised": "_unsupervised"}


def __getattr__(name: str):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is not None:
        import importlib

        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
