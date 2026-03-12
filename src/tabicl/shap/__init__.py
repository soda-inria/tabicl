"""TabICL interpretability: SHAP and ShapIQ"""

__all__ = [
    "get_shap_explainer",
    "get_shap_values",
    "get_shapiq_explainer",
    "plot_shap",
    "plot_shap_feature",
]

_LAZY_IMPORTS = {
    "get_shap_explainer": "_shap",
    "get_shap_values": "_shap",
    "plot_shap": "_shap",
    "plot_shap_feature": "_shap",
    "get_shapiq_explainer": "_shapiq",
}


def __getattr__(name: str):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is not None:
        try:
            import importlib

            module = importlib.import_module(f".{module_name}", __name__)
        except ImportError as e:
            raise ImportError(f"Cannot import '{name}'. Install the shap extra: pip install 'tabicl[shap]'") from e
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
