"""CLI entry point: python -m tabicl.prior

Generates synthetic prior datasets for TabICL training.
Equivalent to: python -m tabicl.prior.genload
"""
import runpy

runpy.run_module("tabicl.prior.genload", run_name="__main__", alter_sys=True)
