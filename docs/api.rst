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

For classifiers, :meth:`tabicl.TabICLClassifier.predict_logits` exposes scores
before the final softmax. To explain these scores with SHAP, select the prediction
method explicitly::

    explainer = get_shap_explainer(clf, X_train, predict_fn="predict_logits")
    explanations = explainer(X_test)

ShapIQ automatically prefers ``predict_logits`` when it is available, so its
classifier explanations are now in logit space. The SHAP helper's default
``predict_proba`` behavior is unchanged. See ``predict_logits`` for the temperature
and ensemble-averaging conventions.

.. autofunction:: tabicl.shap.get_shap_explainer
.. autofunction:: tabicl.shap.get_shap_values
.. autofunction:: tabicl.shap.get_shapiq_explainer
.. autofunction:: tabicl.shap.plot_shap
.. autofunction:: tabicl.shap.plot_shap_feature
