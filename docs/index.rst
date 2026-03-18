.. title:: TabICL: An Open Tabular Foundation Model

.. raw:: html

    <div class="container-fluid sk-landing-bg">
    <div class="container sk-landing-container">
        <div class="row">
        <div class="col-md-4 d-flex align-items-center">
            <h1 class="sk-landing-header text-white text-monospace">TabICLv2</h1>
        </div>
        <div class="col-md-8">
            <ul class="sk-landing-header-body">
            <li>Fully open</li>
            <li>Scikit-learn compatible · pip-installable</li>
            <li>Easy-to-use and powerful</li>
            </ul>
        </div>
        </div>
    </div>
    </div>

    <div class="container-fluid sk-landing-container">
    <div class="row row-padding-main-container">
        <h1 class="hero-title">Open Tabular Foundation Model</h1>
    </div>
    </div>



.. toctree::
   :maxdepth: 2
   :hidden:

   tutorials/index
   api


|test| |PyPI version| |Downloads|

If you usually rely on models like XGBoost or LightGBM, TabICL offers a
different approach: **no dataset-specific training, no hyperparameter search,
strong performance out of the box**. You simply provide your data, and TabICL
makes predictions using a pre-trained model.

**TabICL** is an open tabular foundation model, providing strong performance
for regression and classification tasks out-of-the-box. It can also be extended
to many other tabular tasks, including time-series forecasting
(see :doc:`tutorials <tutorials/index>`).  You can use it
with the familiar scikit-learn ``fit`` / ``predict`` API. Under the hood,
predictions are produced by a pre-trained Transformer in a single forward pass.
Calling `fit` just stores the data, learning happens during `predict`.

**Who is TabICL for?**

- Applied researchers, data scientits and ML teams who need high
  tabular performance without expansive model training and selection.
- AI researchers interested in tabular learning, who want to build on top of a
  strong open foundation model.
- Curious minds who want to explore the capabilities of tabular foundation models!

The `TabICL repository <https://github.com/soda-inria/tabicl>`__ contains the
official implementation of `TabICLv2 <https://arxiv.org/abs/2602.11139>`__
(current default) and the original
`TabICL <https://arxiv.org/abs/2502.05564>`__ (ICML 2025).


What TabICL can do
------------------

**State-of-the-art accuracy — zero tuning required.** TabICLv2 is competitive
with heavily tuned XGBoost, CatBoost, and LightGBM, and even outperforms them
on ~80% of `TabArena <https://tabarena.ai>`__ datasets. It is state-of-the-art
on the `TabArena <https://tabarena.ai>`__ and
`TALENT <https://arxiv.org/abs/2407.00956>`__ benchmarks, among other
hyperparameter-tuned gradient boosted trees, as well as concurrent tabular
foundation models.

**Easy to use:** TabICL is easy to install with `pip`. It is also fully
scikit-learn compliant, by notably giving access to the classical `fit` and
`predict` methods of the scikit-learn API. It is also **open source**
(including pre-training for v1, and soon for v2), with a
permissive license.

**Scalability:** TabICL shows excellent performance on benchmarks with
300 to 100,000 training samples and up to 2,000 features. It can scale
to even larger datasets (e.g., 500K samples) through CPU and disk
offloading, though its accuracy may degrade at some point.

**Handles real-world data.** Pass pandas DataFrames or NumPy arrays.
Missing values, categorical columns, and outliers
are handled automatically. For richer string preprocessing, which is desirable
when strings in a column are diverse and semantically meaningful, TabICL
integrates seamlessly with `skrub <https://skrub-data.org>`__.


Quick start
-----------

.. code:: bash

   pip install tabicl

On Intel Macs, installing PyTorch via ``pip`` may fail. In that case,
install it first with:

.. code:: bash

   conda install pytorch -c pytorch

Then install ``tabicl`` as above.

TabICL follows the standard scikit-learn ``fit`` / ``predict`` API:

.. code:: python

   from tabicl import TabICLClassifier, TabICLRegressor

   # Classification
   clf = TabICLClassifier()
   clf.fit(X_train, y_train)   # checkpoint downloaded once on first run
   clf.predict(X_test)         # in-context learning happens here

   # Regression
   reg = TabICLRegressor()
   reg.fit(X_train, y_train)
   reg.predict(X_test)

It also plugs directly into scikit-learn pipelines:

.. code:: python

   from sklearn.pipeline import make_pipeline
   from skrub import TableVectorizer
   from tabicl import TabICLClassifier

   pipeline = make_pipeline(TableVectorizer(), TabICLClassifier())
   pipeline.fit(X_train, y_train)
   pipeline.predict(X_test)

For **time-series forecasting**, install the extra dependencies and use
:class:`tabicl.TabICLForecaster`:

.. code:: bash

   pip install tabicl[forecast]

.. code:: python

   from tabicl import TabICLForecaster

   forecaster = TabICLForecaster()
   pred_df = forecaster.predict_df(context_df, prediction_length=96)


Explore the docs
----------------

:doc:`Tutorials <tutorials/index>` provides runnable examples on a
laptop. Learn by task:

- :doc:`Classification tutorial <tutorials/classification_2D_proba>`,
  with probability estimates.
- :doc:`Regression tutorial <tutorials/regression_heteroscedastic_1D>`,
  with quantile estimates.
- :doc:`Time-series forecasting tutorial <tutorials/time_series_forecasting>`.

See the :doc:`API reference <api>` for estimator and parameter details.


Available models
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Model
     - Classification checkpoint
     - Regression checkpoint
   * - | **TabICLv2**
       | (`arXiv <https://arxiv.org/abs/2602.11139>`__)
     - ``tabicl-classifier-v2-20260212.ckpt`` *(default)*
     - ``tabicl-regressor-v2-20260212.ckpt`` *(default)*
   * - | **TabICLv1.1**
       | (May 2025)
     - ``tabicl-classifier-v1.1-20250506.ckpt``
     - —
   * - | **TabICLv1**
       | (`ICML 2025 <https://arxiv.org/abs/2502.05564>`__)
     - ``tabicl-classifier-v1-20250208.ckpt``
     - —

- **TabICLv2** — state-of-the-art, supports classification and
  regression. Improved accuracy through better synthetic pre-training
  data, architectural improvements, and better pre-training, with
  comparable runtime to v1.
- **TabICLv1.1** — TabICLv1 post-trained on an early v2 prior.
  Classification only.
- **TabICLv1** — original model. Classification only.

FAQ
---

**How does TabICL work?**
TabICL is a tabular foundation model. It uses in-context
learning (ICL) to learn from new data in a single forward pass through
a Transformer model. In practice, you provide training data in ``fit``
and get predictions in ``predict``. Internally, nothing happens when calling
``fit``: the training data is just stored for later, and model weights are
loaded if necessary. Everything happens in ``predict``, which under the hood
performs a forward pass using ``X_train``, ``y_train``, and
``X_test``, i.e., ``y_pred = model(X_train, y_train, X_test)``. TabICL's
learning capabilities come from pre-training on millions of synthetic datasets.


**What is the architecture of TabICL?**
TabICL is based on a standard Transformer architecture, with some modifications
to better handle tabular data. It includes column-wise attention,
row-wise attention, and a final Transformer performing attention over the
samples. Detailed architecture diagrams and descriptions can be found in our
`TabICLv1 <https://arxiv.org/abs/2502.05564>`__ and
`TabICLv2 <https://arxiv.org/abs/2602.11139>`__ papers. If code is more your
thing, `NanoTabICL <https://github.com/soda-inria/nanotabicl>`__ provides a
minimal implementation of the TabICLv2 architecture for educational and
experimental purposes.

**Do I need GPUs to use TabICL?**
TabICL works on both CPU and GPU, and can run on a laptop for many
practical datasets. A GPU is recommended when datasets get larger or
when you need faster predictions. On modern GPUs, TabICL can be much
faster than CPU and can scale to very large datasets (a million samples) in a
few minutes. On an H100 GPU, TabICLv2 handles a dataset with
50,000 samples and 100 features in under 10 seconds. Through KV caching, TabICL
supports faster repeated inference on the same training data.

**What dataset sizes work well?**
TabICLv2 is pre-trained on datasets between 300 and 48K training
samples. However, it can generalize to larger datasets: we have seen good
performance on some datasets with 600K samples. We have not tested whether
TabICL generalizes to datasets smaller than 300 samples.

**What about the number of columns?**
TabICLv2 is pre-trained on datasets between 2 and 100 columns. In
practice, it seems to generalize well to more columns, but the exact limit is
currently unknown.


Citation
--------

If you use TabICL for research purposes, please cite our papers:

.. code:: bibtex

   @inproceedings{qu2025tabicl,
     title={Tab{ICL}: {A} Tabular Foundation Model for In-Context Learning on Large Data},
     author={Qu, Jingang and Holzm{\"u}ller, David and Varoquaux, Ga{\"e}l and Le Morvan, Marine},
     booktitle={International Conference on Machine Learning},
     year={2025}
   }

   @article{qu2026tabiclv2,
     title={{TabICLv2}: {A} better, faster, scalable, and open tabular foundation model},
     author={Qu, Jingang and Holzm{\"u}ller, David and Varoquaux, Ga{\"e}l and Le Morvan, Marine},
     journal={arXiv preprint arXiv:2602.11139},
     year={2026}
   }


.. |test| image:: https://github.com/soda-inria/tabicl/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/soda-inria/tabicl/actions/workflows/testing.yml
.. |PyPI version| image:: https://badge.fury.io/py/tabicl.svg
   :target: https://badge.fury.io/py/tabicl
.. |Downloads| image:: https://img.shields.io/pypi/dm/tabicl
   :target: https://pypistats.org/packages/tabicl
