"""
Compute time and memory footprint trade-offs
============================================

This gallery explores how TabICL's computational parameters affect
**compute time** and **memory footprints** (RAM, VRAM, disk) — *not*
predictive performance.

The figures below are generated from benchmarks run on a single NVIDIA L4
GPU (24 GB VRAM, 16 GB host RAM). They cover three axes:

1. the **KV cache** (pre-computed attention keys/values that speed up repeated
   prediction),
2. **VRAM usage** during fit and predict,
3. **CPU/disk offload** modes that trade compute time for lower VRAM.

.. raw:: html

   <style>.sphx-glr-timing{display:none !important}</style>
"""

# %%
# KV cache: faster repeated prediction at the cost of VRAM
# -------------------------------------------------------
#
# TabICL can pre-compute a KV cache for the training set during ``fit``.
# The cache stores the attention keys and values for the training rows so
# that subsequent ``predict`` calls reuse them instead of recomputing them,
# which speeds up inference. The trade-off is VRAM: the cache is resident
# on the GPU, so it increases the memory footprint for as long as the
# estimator is alive.
#
# The figure plots predict time (dashed) and total fit+predict time (solid)
# against three dataset-size axes — number of training rows, number of
# features, and test-set size — with the KV cache OFF (green) and ON
# (vermillion). Two things to read from it:
#
# * **Predict speed-up.** The dashed vermillion line (cache ON, predict
#   only) sits below the dashed green line (cache OFF, predict only): the
#   cache makes prediction faster. The gap widens with the number of
#   training rows, because more training context means more attention work
#   is saved by the cache. The cache is per-estimator and does not carry
#   over across the ensemble members, so the benefit scales linearly with
#   ``n_estimators`` without amortization.
# * **No free fit.** The solid lines (fit+predict) include the cache build,
#   which happens during ``fit``. The solid vermillion line is therefore
#   above the solid green one: enabling the cache costs fit time upfront
#   to save predict time later. Whether this pays off depends on how many
#   times ``predict`` is called on the same fitted estimator.
#
# .. raw:: html
#    :file: ../_static/compute_tradeoffs/fig1.html

# %%
# VRAM footprint: where the memory goes
# -------------------------------------
#
# The previous figure showed the compute-time side of the KV-cache
# trade-off; this one shows the memory side. Peak VRAM is recorded
# separately for the fit and predict phases. With the cache ON, the
# predict footprint (dashed vermillion) rises well above the fit peak
# (solid vermillion), because the resident cache stays on the GPU during
# prediction. With the cache OFF, the fit footprint is a near-constant
# ~110 MB (the estimator's own weights) and is omitted from the plot;
# only the predict footprint matters.
#
# The cache's VRAM cost scales with the number of training rows — more
# training samples mean a larger cache — and, to a lesser extent, with the
# number of features. On the 24 GB L4 used here the cache always fits, but
# on smaller GPUs or with much larger training sets it can become the
# binding memory constraint, which is exactly what the offload modes in
# the next figure are designed to relieve.
#
# .. raw:: html
#    :file: ../_static/compute_tradeoffs/fig2.html

# %%
# Offload modes: trading compute time for VRAM headroom
# -----------------------------------------------------
#
# When the output tensors are large (many test samples), the offload mode
# becomes the main lever for VRAM usage. ``offload=gpu`` keeps the outputs
# on the GPU — the fastest option, but the most VRAM-hungry.
# ``offload=cpu`` and ``offload=disk`` proactively move the outputs to host
# RAM or to disk, saving gigabytes of VRAM at a substantial compute-time
# cost (around an order of magnitude on this hardware).
#
# The left panel compares GPU and CPU at a small test size (``n_test=200``):
# the GPU is much faster, and on the CPU the offload mode is a no-op
# (the offload logic is bypassed for ``device='cpu'``). The right panel
# isolates the GPU offload modes at a large test size (``n_test=10 000``),
# where the output tensors are large enough that ``offload=cpu`` and
# ``offload=disk`` save roughly 2.5 GB of VRAM. The bars show predict time
# (black), peak RAM (blue), and peak VRAM (vermillion) on a log y-axis, so
# the VRAM savings and the compute-time penalty are both visible. Use
# these modes when the model would otherwise not fit in VRAM; the
# compute-time penalty is the price of admission.
#
# .. raw:: html
#    :file: ../_static/compute_tradeoffs/fig3.html
