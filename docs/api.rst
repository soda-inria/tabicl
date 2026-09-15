.. _api_ref:

API
===

Estimators
----------

.. autoclass:: tabicl.TabICLClassifier
   :members:

.. autoclass:: tabicl.TabICLRegressor
   :members:

.. autoclass:: tabicl.FinetunedTabICLClassifier
   :members:

.. autoclass:: tabicl.FinetunedTabICLRegressor
   :members:

.. autoclass:: tabicl.TabICLForecaster
   :members:

.. autoclass:: tabicl.TabICLUnsupervised
   :members:

Inference configuration
-----------------------

.. autoclass:: tabicl.InferenceConfig
   :members:

Forecasting utilities
---------------------

.. autoclass:: tabicl.forecast.TimeSeriesDataFrame
   :members:

.. autoclass:: tabicl.forecast.TimeTransformChain
   :members:

.. autofunction:: tabicl.forecast.plot_forecast

Time feature transforms
-----------------------

.. autoclass:: tabicl.forecast.transforms.TimeTransform
   :members:

.. autoclass:: tabicl.forecast.transforms.IndexEncoder
.. autoclass:: tabicl.forecast.transforms.DatetimeEncoder
.. autoclass:: tabicl.forecast.transforms.ExtendedDatetimeEncoder
.. autoclass:: tabicl.forecast.transforms.FourierEncoder
.. autoclass:: tabicl.forecast.transforms.AutoPeriodicEncoder
.. autoclass:: tabicl.forecast.transforms.PeriodicDetectionConfig

Pre-training data
-----------------

.. autoclass:: tabicl.prior.PriorDataset
   :members:

SHAP interpretability
---------------------

By default, SHAP explanations use a single all-NaN row as their reference.
To explain predictions relative to a representative dataset, pass a numeric
background with the same feature order and encoding as the training data:

.. code-block:: python

   from tabicl.shap import get_shap_values

   shap_values = get_shap_values(estimator, X_test, X_background=X_train[:20])

The background changes the reference prediction and feature attributions.
Choose samples representative of the population you want to compare against.

.. autofunction:: tabicl.shap.get_shap_explainer
.. autofunction:: tabicl.shap.get_shap_values
.. autofunction:: tabicl.shap.get_shapiq_explainer
.. autofunction:: tabicl.shap.plot_shap
.. autofunction:: tabicl.shap.plot_shap_feature
