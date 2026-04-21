"""Synthetic prior data generation for TabICL pre-training.

Only :class:`PriorDataset` is public. All other symbols in this subpackage
are internal pre-training utilities and may change without notice. A CLI
entry point is provided via ``python -m tabicl.prior``.
"""

from ._dataset import PriorDataset

__all__ = ["PriorDataset"]
