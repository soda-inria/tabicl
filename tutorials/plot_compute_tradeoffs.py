"""
Compute time and memory footprint trade-offs
============================================

This gallery explores how TabICL's computational parameters affect
compute time, covering the use of the KV cache.

.. note::
   The figure in the gallery is rendered when the documentation is built,
   from a snapshot of benchmark results committed alongside the docs in
   ``docs/_data/tabicl_benchmarks_data.csv``. The runs themselves are produced
   by the `tabicl-benchmarks <https://github.com/probabl/tabicl-benchmarks>`_
   project repository. The version of ``tabicl`` displayed in figure footnotes
   may not match the latest ``tabicl`` release.

.. raw:: html

   <style>.sphx-glr-timing{display:none !important}</style>
"""

# %%
# Introducing kv-caching
# ----------------------
#
# At the core of ``tabicl``'s architecture are transformer-like components, and
# much like other transformer-based systems, it can benefit from temporarily
# storing intermediate objects computed on the way to the final output, so
# they can be reused later to speed up predictions on similar inputs, or even,
# in some cases, to reduce the memory footprint.
#
# These objects are called, in transformer terminology, *keys* and *values*,
# hence the name of the memoization system: the *kv*-cache.
#
# Tabular foundational models encode the *entire* training dataset along with
# the training labels. In this case, enabling the key-value cache means that
# all the operations on the training data that do not depend on the target
# dataset are stored and available for reuse to speed up prediction on
# previously unseen target datasets.
#
# By default, the :meth:`.fit <tabicl.TabICLClassifier.fit>` method of
# ``tabicl`` estimators does not run any compute-heavy operations. It merely
# downloads (if not already present) and loads the model weights, and stores the
# training data after applying some light pre-processing. At predict time,
# both the training data and the input dataset are jointly forwarded to the
# actual compute machinery.
#
# With kv-cache enabled, :meth:`.fit <tabicl.TabICLClassifier.fit>` additionally
# computes some of the intermediate outputs that only depend on the training
# data, and stores them in host memory if ``device='cpu'``, otherwise on device
# memory.
#
# The ``tabicl`` API exposes several caching levels:
#
# - ``kv_cache=False`` (the default) to disable caching
# - ``kv_cache=True`` or ``kv_cache='kv'`` for aggressive caching
# - ``kv_cache='repr'`` for a moderate amount of caching, tuned down in the
#   later layers
#
# The :doc:`Getting Started <getting_started>` gallery already showcases usage
# of this parameter. The present gallery provides an in-depth analysis of its
# effects on a range of workloads.

# %%
# Effects of ``kv_cache`` on device compute time
# -----------------------------------------------
#
# The figure plots predict time and total fit+predict time against three
# dataset-size axes: number of training rows, number of features, and size of
# test data, with and without kv-caching.
#
# .. raw:: html
#    :file: ../_static/compute_tradeoffs/fig1.html
#
# As expected, enabling kv-caching considerably speeds up the subsequent
# :meth:`.predict <tabicl.TabICLClassifier.predict>` calls. When it is
# enabled, :meth:`.fit <tabicl.TabICLClassifier.fit>` absorbs most of the
# operations related to the training data, so that
# :meth:`.predict <tabicl.TabICLClassifier.predict>` compute time no longer
# depends on the number of training rows (see left plot). Naturally, since
# kv-caching only affects operations on training data, the effect on total
# :meth:`.predict <tabicl.TabICLClassifier.predict>` time is less noticeable
# with increased test-data size, and the speed-up eventually vanishes for very
# large test data (see right plot).
#
# The fit+predict curves, where a :meth:`.fit <tabicl.TabICLClassifier.fit>`
# call is followed by a single subsequent
# :meth:`.predict <tabicl.TabICLClassifier.predict>` call, are nearly
# identical between the two modes. This reveals that enabling kv-caching
# moves some of the time budget from
# :meth:`.predict <tabicl.TabICLClassifier.predict>` to
# :meth:`.fit <tabicl.TabICLClassifier.fit>`, without any noticeable overhead.
# The effect on :meth:`.predict <tabicl.TabICLClassifier.predict>` speed is
# even more beneficial for large training data (see left plot).
#
# Furthermore, the predict and fit+predict curves converge for large training
# data (without kv-caching) or for large test data. This trend highlights the
# computational complexity of ``tabicl`` in different regimes. The complexity
# of loading the model weights and preprocessing the data scales linearly with
# the amount of data. On the other hand, computation of the intermediate hidden
# activations within the transformer scales quadratically. For large amounts
# of data, the former becomes negligible compared to the latter, but it remains
# a significant offset for smaller data.
#
# Finally, the middle plot shows that the effect of the cache on
# :meth:`.predict <tabicl.TabICLClassifier.predict>` time grows super-linearly
# as the number of features increases. This means kv-caching is especially
# beneficial for wide tables.
#
# In a nutshell, enabling ``kv-caching`` has **no downside** on compute time;
# the effect is negligible at worst.

# %%
# Practical summary
# -----------------
#
# In short, a good rule overall is to *always activate the kv-cache* unless the
# available memory is too limited to build the cache.

# %%
# Other limitations
# -----------------
#
# One important limitation is that caching is not implemented for
# classification problems with more than 10 classes. In this case, the
# :meth:`.fit <tabicl.TabICLClassifier.fit>` call raises an exception instead.

# %%
# WIP
# ---
#
# .. raw:: html
#    :file: ../_static/compute_tradeoffs/fig2.html
#
# .. raw:: html
#    :file: ../_static/compute_tradeoffs/fig3.html
#
# .. raw:: html
#    :file: ../_static/compute_tradeoffs/fig4.html
