"""
TabICL compute time: a case study
=================================

This case study examines how the compute time of a TabICL classifier depends on
the shape of the data, the device it runs on, available memory, and
parameterization of caching mechanisms.

.. note::
   The figures in this gallery are rendered from a database of benchmark
   results committed alongside the documentation in
   ``docs/_data/tabicl_benchmarks_data.csv``. The results themselves are
   produced independently by the `tabicl-benchmarks
   <https://github.com/probabl/tabicl-benchmarks>`_ project repository. As a
   consequence, the version of ``tabicl`` displayed in figure footnotes may not
   match the latest ``tabicl`` release.

.. raw:: html

   <style>.sphx-glr-timing{display:none !important}</style>
"""

# %%
# .. _compute_tradeoffs_caching:
#
# Understanding caching in ``tabicl``
# -----------------------------------
#
# At :meth:`.fit <tabicl.TabICLClassifier.fit>` time, ``tabicl`` estimators
# store a copy of the training data in memory. At prediction time, that copy
# is forwarded, jointly with the prediction dataset, to the underlying
# transformer components, which return the probabilities. The compute cost is
# concentrated in those components, while the time spent in fit, which merely
# registers the training data and applies some mild pre-processing, is
# negligible.
#
# But this distribution of compute can almost be reversed, depending on the
# value of the ``kv_cache`` parameter. Some of the operations within the
# transformer components are computed on the training data, independently of
# the prediction dataset. The corresponding intermediate compute artifacts can
# be computed once, ahead of time, stored in memory and reused to make later
# prediction calls faster. In transformer terminology, those objects are called
# *keys* and *values*, and this memory system is accordingly named the
# *kv*-cache. When it is enabled, part of the computational cost moves to the
# fit step that prepares the cache.
#
# The :doc:`Getting Started <getting_started>` gallery shows how to set this
# parameter. Three levels trade prediction speed against the time spent in fit:
# ``kv_cache=False`` (default) disables the cache, ``kv_cache='repr'`` caches a
# reduced set of intermediate outputs (tuned down in the later layers), and
# ``kv_cache=True`` (or ``kv_cache='kv'``) caches every reusable intermediate
# output.
#
# Benchmarks in this case study report compute time with different cache
# settings.
#
# Note that caching also has a memory overhead, but that overhead is
# counterbalanced by ``tabicl``'s internal auto-batching policy. Auto-batching
# is especially robust on discrete device backends, for which we assume caching
# can be enabled without a second thought. On CPU and unified memory devices,
# auto-batching is less robust, because memory is also occupied by other
# processes and by the operating system. If you experience slowdowns or memory
# overflows, especially when the training data saturates memory, using
# ``kv_cache='repr'`` or ``kv_cache=False`` might help. Read more in the section
# :ref:`Effects of device memory attrition <compute_tradeoffs_attrition>` of
# this case study.
#
# .. note::
#
#    Ongoing efforts aim to ensure consistent behavior across all devices.

# %%
# Compute times on a discrete (GPU) device
# ----------------------------------------
#
# The figure plots predict time and total ``.fit`` + ``.predict`` time for each
# of the three cache levels, sweeping three dataset-size axes: the number of
# rows, the number of features, and the size of the test set. All other
# parameters are set to their default values, and the measurements are taken on
# a discrete device (NVIDIA L4).
#
# .. note::
#
#    The figures in this case study are interactive. Hover the cursor to reveal
#    additional information and select areas to zoom in.
#
# .. raw:: html
#    :file: ../_static/compute_tradeoffs/fig1.html
#
# Let's review a few key takeaways about ``tabicl`` compute times from those
# numbers.
#
# *Orders of magnitude* — compute times range from sub-second for small
# datasets to minute-long runs for medium-sized datasets.
#
# *Effects of caching* — caching displays no compute time overhead. It really
# only consists of different allocation strategies of the compute time budget
# over the ``.fit`` step and the
# :meth:`.predict <tabicl.TabICLClassifier.predict>` step, while the overall
# budget for a ``.fit`` followed by a single ``.predict`` step remains
# invariant. Use the zoom tool to reveal the three ``.fit`` + ``.predict``
# curves that are barely distinguishable from one another. Besides, the cache
# significantly improves prediction compute time, even more so when the
# testing set is small compared to the number of training rows.
#
# *Scaling trends* — with the chosen dataset dimensions, compute times grow
# steadily as any one of the dimensions grows. The theory predicts a linear
# increase with respect to the size of the test set, and a quadratic increase
# with respect to the number of training rows or the number of features. Keep
# in mind that the overall complexity of ``tabicl`` is superlinear
# (``O(nb_features**2 * nb_total_rows + nb_training_rows**2)``).

# %%
# Compute times on CPU
# --------------------
#
# Here's an apples-to-apples comparison of compute times observed on CPU and on
# the same discrete GPU device, with the kv cache enabled, all other parameters
# set to their default values, sweeping the dataset-size axes from extra-small
# to small datasets.
#
# .. raw:: html
#    :file: ../_static/compute_tradeoffs/fig2.html
#
# Compute times on those tiny and small datasets on CPU increase up to several
# seconds, while remaining tractable. On the other hand, on-device compute
# times are all far below one second and do not show much variation with
# respect to the dimension increases. In those regimes, the discrete device
# throughput is not saturated.
#
# This follow-up figure displays compute times on CPU in regimes that leverage
# the full power of the discrete GPU device.
#
# .. raw:: html
#    :file: ../_static/compute_tradeoffs/fig3.html
#
# Where the discrete device showcases compute times below 5 seconds, CPU
# requires tens of seconds to more than a minute, with about 30 times slower
# throughput.

# %%
# .. _compute_tradeoffs_attrition:
#
# Effects of device memory attrition
# ----------------------------------
#
# The section :ref:`Understanding caching in tabicl
# <compute_tradeoffs_caching>` of this case study mentions the risks of
# slowdowns or memory overflows; this follow-up figure highlights the effects
# of memory attrition on compute time, for a given medium-small-sized dataset,
# with all ``tabicl`` parameters set to their default values.
#
# .. raw:: html
#    :file: ../_static/compute_tradeoffs/fig4.html
#
# Thanks to ``tabicl``'s internal auto-batching policy, it does not run out of
# memory and compute times do not display significant variations either, even
# after the available device memory has been shrunk below 1 GB. Beyond this
# threshold, avoiding memory overflows requires falling back to unit-sized
# batches that do not saturate device throughput, and we notice a significant
# slowdown; shrinking the available memory further will eventually trigger
# overflows.

# %%
# Practical summary
# -----------------
#
# Key takeaways:
#
# - Compute times on specialized discrete devices range from sub-second for
#   small datasets, to tens of seconds on medium-sized datasets. Compute on CPU
#   is viable for tiny datasets, but is tens of times slower than discrete
#   devices otherwise.
#
# - The kv-cache is great at reducing compute time of consecutive predictions
#   from a given saved set of training data. In environments where device
#   memory or host memory is scarce, its memory overhead might exacerbate
#   instability, so use it cautiously.

# %%
# Other limitations
# -----------------
#
# Caching is available for classification problems with up to 10 classes. For a
# problem with more classes, the :meth:`.fit <tabicl.TabICLClassifier.fit>`
# call raises an exception.
