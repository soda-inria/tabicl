"""TabICL interpretability: SHAP and ShapIQ."""

try:
    from tabicl.shap._shap import (
        get_shap_explainer,
        get_shap_values,
        plot_shap,
        plot_shap_feature,
    )
    from tabicl.shap._shapiq import get_shapiq_explainer
except ImportError:
    raise ImportError(
        "tabicl.shap requires extra dependencies. Install with: pip install tabicl[shap]"
    ) from None

__all__ = [
    "get_shap_explainer",
    "get_shap_values",
    "get_shapiq_explainer",
    "plot_shap",
    "plot_shap_feature",
]
