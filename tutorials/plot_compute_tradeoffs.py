"""
Compute time and memory footprint trade-offs
============================================

This gallery explores how TabICL's computational parameters affect
compute time and memory footprints (host and device memory), covering
device choice and use of the KV cache.

.. note::
   The figures in the gallery are generated in the
   `tabicl-benchmarks <https://github.com/probabl/tabicl-benchmarks>`_
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
# the effect is negligible at worst. The trade-off is a higher memory
# footprint, which the next section quantifies.

# %%
# Effects of ``kv_cache`` on memory
# ---------------------------------
#
# The previous figure displays the compute-time side of the KV-cache trade-off;
# this one displays the memory side. Peak memory usage is recorded separately
# for the fit and predict phases. The peak memory allocated during :meth:`.fit
# <tabicl.TabICLClassifier.fit>` when kv-caching is disabled is always
# negligible, and is not displayed. When kv-caching is enabled, the peak memory
# reported during :meth:`.predict <tabicl.TabICLClassifier.predict>` includes
# the cache overhead.
#
# .. raw:: html
#    :file: ../_static/compute_tradeoffs/fig2.html
# 
# Somewhat surprisingly, the memory footprint of
# :meth:`.predict <tabicl.TabICLClassifier.predict>` is uniformly lower
# when the key-value cache is enabled, so caching also appears to have **no
# downside** with respect to memory usage. In fact, the
# :meth:`.predict <tabicl.TabICLClassifier.predict>` memory curves are
# remarkably similar in shape to the compute-time plots, and the same
# conclusions apply. The reason is that the overhead added by the cache is 
# more than offset by the memory saved by skipping the computation of intermediate 
# hidden activations.
#
# The left and right plots, however, suggest that for very large training data
# (or very small test data), the peak memory needed to build the key-value
# cache (during :meth:`.fit <tabicl.TabICLClassifier.fit>`) surpasses the peak
# memory of :meth:`.predict <tabicl.TabICLClassifier.predict>` without the
# cache. In those cases, there may be a genuine trade-off: caching is
# worthwhile only if enough memory is available to build the cache.
#
# The particularly high peak memory observed during
# :meth:`.fit <tabicl.TabICLClassifier.fit>` is the sum of the peak memory
# needed to compute the intermediate activations in the later layers, and the
# memory held by the objects from earlier layers that are retained in the cache
# being built.

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
