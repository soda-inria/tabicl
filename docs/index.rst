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
            <li>Excellent scikit-learn compatiblity</li>
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



TabICL is a tabular foundation model.
It uses in-context learning (ICL) to learn from new data in a single
forward pass through a Transformer model: ``predict(X_test)`` calls ``y_pred = model(X_train, y_train, X_test)``.
It has acquired strong learning capabilities through
pre-training on millions of synthetic datasets.


|test| |PyPI version| |Downloads|

Installation
------------

.. code:: bash

   pip install tabicl

Optional dependencies can be installed as needed:

.. code:: bash

   pip install tabicl[forecast]   # time series forecasting
   pip install tabicl[pretrain]   # pre-training
   pip install tabicl[all]        # everything

On Intel Macs, installing PyTorch via ``pip`` may fail. In that case,
install it first with:

.. code:: bash

   conda install pytorch -c pytorch

Then install ``tabicl`` as above.

Tutorials and functionality
---------------------------

- **Basic usage**: :doc:`tutorial <tutorials/getting_started>`.
- **Model parameters**: See :class:`tabicl.TabICLClassifier` and :class:`tabicl.TabICLRegressor`.
- **Probabilistic classification**: :doc:`tutorial <tutorials/classification_2D_proba>`.
- **Quantile regression**: :doc:`tutorial <tutorials/regression_heteroscedastic_1D>`.
- **Preprocessing**: TabICL will automatically use simple preprocessing to handle missing values and categorical features.
  To handle string and date columns, see the :doc:`tutorial on using skrub <tutorials/string_handling>`.
- **Time-series forecasting**: See our
  :doc:`tutorial on time-series forecasting with TabICL <tutorials/time_series_forecasting>`.
- **Minimal architecture implementation**: `NanoTabICL <https://github.com/soda-inria/nanotabicl>`__
  provides a minimal implementation of the TabICLv2 architecture for educational and experimental purposes.

Available models
----------------


.. list-table::
   :header-rows: 1

   * - Model
     - Classification checkpoint
     - Regression checkpoint
   * - **TabICLv2** (`arXiv <https://arxiv.org/abs/2602.11139>`__)
     - ``tabicl-classifier-v2-20260212.ckpt``
     - ``tabicl-regressor-v2-20260212.ckpt``
   * - **TabICL v1.1** (May 2025, no paper)
     - ``tabicl-classifier-v1.1-20250506.ckpt``
     - —
   * - **TabICLv1** (`ICML 2025 <https://arxiv.org/abs/2502.05564>`__)
     - ``tabicl-classifier-v1-20250208.ckpt``
     - —


- **TabICLv2**: Our state-of-the-art model, supporting both
  classification and regression. Strongly improved accuracy over v1
  through better synthetic pre-training data, architectural
  improvements, and better pre-training, with comparable runtime.
- **TabICLv1.1**: TabICLv1 post-trained on an early version of the v2
  prior. Classification only.
- **TabICLv1**: Original model. Classification only. TabICLv1 and v1.1
  originally used ``n_estimators=32``; we reduced the default to 8
  afterwards.

Pre-training
------------

Pre-training code (including synthetic data generation) is currently
available for the v1 model. The scripts folder provides the commands for
`stage 1 <https://github.com/soda-inria/tabicl/blob/main/scripts/train_stage1.sh>`__,
`stage 2 <https://github.com/soda-inria/tabicl/blob/main/scripts/train_stage2.sh>`__,
and `stage 3 <https://github.com/soda-inria/tabicl/blob/main/scripts/train_stage3.sh>`__
of curriculum learning. Pre-training code for v2 will be released upon publication.

FAQ
---

**How fast is TabICL?** On datasets with :math:`n` training rows and
:math:`m` columns, the runtime complexity of TabICL (v1 and v2) is
:math:`O(n^2 + nm^2)`. On datasets with many rows and columns, it can be
10x faster than TabPFN-2.5. On modern GPUs, TabICL can handle a million
samples in a few minutes without RAM overflow thanks to CPU and disk
offloading.

.. image:: ./figures/runtime_tabpfnv25_tabiclv2.png
   :width: 70%
   :alt: Runtimes for different hardware and sample sizes
   :align: center

**What dataset sizes work well?** TabICLv2 is pre-trained on datasets
between 300 and 48K training samples. However, it can generalize to
larger datasets to some extent, and we see good results even on some
datasets with 600K samples. We have not tested if TabICL generalizes to
datasets smaller than 300 samples.

.. image:: ./figures/tabiclv2_perf_vs_n_samples.png
   :width: 50%
   :alt: Average rank vs. number of samples
   :align: center

**What about the number of columns?** TabICLv2 is pre-trained on
datasets between 2 and 100 columns. We see good generalization to more
columns and don’t know where the limit is.

.. image:: ./figures/tabiclv2_perf_vs_n_features.png
   :width: 50%
   :alt: Average rank vs. number of features
   :align: center

Results from state-of-the-art research
----------------------------------------

This repository is the official implementation of **TabICLv2**
(`arXiv <https://arxiv.org/abs/2602.11139>`__) and **TabICL** (`ICML
2025 <https://arxiv.org/abs/2502.05564>`__).

**State-of-the-art accuracy even without hyperparameter tuning:**
TabICLv2 is the new state-of-the-art model for tabular classification
and regression on the `TabArena <https://tabarena.ai>`__ and
`TALENT <https://arxiv.org/abs/2407.00956>`__ benchmarks. It does not
require hyperparameter tuning and still outperforms heavily tuned
XGBoost, CatBoost, or LightGBM on TabArena on ~80% of datasets.

**Easy to use:** TabICL is pip-installable and scikit-learn compliant.
It is also **open source** (including `pre-training <#pre-training>`__
for v1), with a permissive license.

**Speed:** TabICL performs ``fit`` and ``predict`` jointly via a single
forward pass through a pre-trained transformer model. For larger
datasets, we recommend a GPU. On an H100 GPU, TabIClv2 can ``fit`` and
``predict`` a dataset with 50,000 samples and 100 features in under 10
seconds, which is 10x faster than TabPFN-2.5. Through KV caching, TabICL
supports faster repeated inference on the same training data.

**Scalability:** TabICL shows excellent performance on benchmarks with
300 to 100,000 training samples and up to 2,000 features. It can scale
to even larger datasets (e.g., 500K samples) through CPU and disk
offloading, though its accuracy may degrade at some point.

.. image:: ./figures/pareto_front_improvability_tabarena.png
   :width: 50%
   :alt: Model comparison on TabArena
   :align: center


Citation
--------

If you use TabICL for research purposes, please cite our papers for
`TabICL <https://arxiv.org/abs/2502.05564>`__ and
`TabICLv2 <https://arxiv.org/abs/2602.11139>`__:

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

Contributors
------------

- `Jingang Qu <https://github.com/jingangQu>`__
- `David Holzmüller <https://github.com/dholzmueller>`__
- `Marine Le Morvan <https://github.com/marineLM>`__

Star history
------------

|Star History Chart|

.. |test| image:: https://github.com/soda-inria/tabicl/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/soda-inria/tabicl/actions/workflows/testing.yml
.. |PyPI version| image:: https://badge.fury.io/py/tabicl.svg
   :target: https://badge.fury.io/py/tabicl
.. |Downloads| image:: https://img.shields.io/pypi/dm/tabicl
   :target: https://pypistats.org/packages/tabicl
.. |Star History Chart| image:: https://api.star-history.com/svg?repos=soda-inria/tabicl&type=date&legend=top-left
   :target: https://www.star-history.com/#soda-inria/tabicl&type=date&legend=top-left

.. toctree::
   :maxdepth: 2
   :hidden:

   api

.. toctree::
   :maxdepth: 2
   :hidden:

   tutorials/index
