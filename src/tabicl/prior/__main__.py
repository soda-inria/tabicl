"""CLI entry point: python -m tabicl.prior

Generates synthetic prior datasets for TabICL training.
"""

import runpy

runpy.run_module("tabicl.prior._genload", run_name="__main__", alter_sys=True)
