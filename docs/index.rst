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

TabICL is a transformer-based tabular foundation model that relies on in-context learning
to learn the underlying mapping between features and target from a given training set, and
then predict on the test set, all in a single forward pass. In practice, you provide training
data in ``fit`` and get predictions in ``predict_proba`` or ``predict``. During ``fit``, we preprocess
the training data, create multiple transformed dataset views (e.g., by shuffling features),
load the pre-trained TabICL model, and optionally pre-compute KV caches for the training data
to speed up inference (controlled by the ``kv_cache`` init parameter). During ``predict_proba`` or 
``predict``, we process the test data, forward each dataset view through the TabICL model via 
in-context learning, and average predictions across all ensemble members. TabICL's learning 
capabilities come from pre-training on millions of synthetic datasets.

**What is the architecture of TabICL?**

TabICL is based on the Transformer architecture, with several improvements to better handle
tabular data. It processes input through three stages: a column-wise Transformer that embeds
each feature, a row-wise Transformer that aggregates features into row representations, and
a dataset-wise Transformer that performs in-context learning over training and test samples
to produce predictions. Detailed architecture diagrams and descriptions can be found in our
`TabICLv1 <https://arxiv.org/abs/2502.05564>`__ and `TabICLv2 <https://arxiv.org/abs/2602.11139>`__
papers. If code is more your thing, `NanoTabICL <https://github.com/soda-inria/nanotabicl>`__
provides a minimal implementation of the TabICLv2 architecture for educational and experimental purposes.

**Do I need GPUs to use TabICL?**

TabICL works on both CPU and GPU, and performs well on a GPU-free laptop for medium-sized
datasets within a reasonable time. However, a GPU is recommended when datasets get larger
or when you need faster predictions. Thanks to architectural
efficiency and engineering efforts, TabICL can scale to very large datasets of up
to a million samples, though you need to enable CPU/disk offloading to ease GPU memory
requirements. On an H100 GPU, TabICLv2 handles a dataset with 50K samples and 100 features
within 10 seconds.

**What dataset sizes work well?**

TabICLv2 is pre-trained on datasets with 300 to 60K training samples.
However, it can generalize beyond this range and we have observed good
performance on datasets with 600K samples. Generalization to datasets
smaller than 300 samples has not yet been tested.

**What about the number of columns?**

TabICLv2 is pre-trained on datasets with 2 to 100 columns. In practice,
it generalizes well beyond this range, though the exact upper limit remains unknown.

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
