"""Shared test configuration and helpers."""

import os
import sys

import pytest

# Make the tests/ directory importable so test modules can do `from conftest import ...`
sys.path.insert(0, os.path.dirname(__file__))

_CKPT_DIR = os.environ.get("TABICL_CHECKPOINT_DIR")

_FILENAMES = {
    "classifier": "tabicl-classifier-v2-20260212.ckpt",
    "regressor": "tabicl-regressor-v2-20260212.ckpt",
}


def model_path(kind: str) -> dict:
    """Return a model_path kwarg dict for the given kind ('classifier'/'regressor').

    When TABICL_CHECKPOINT_DIR is unset, returns an empty dict so the estimator
    uses default auto-download (works on GitHub Actions). When set, points at a
    local checkpoint file.
    """
    if _CKPT_DIR is None:
        return {}
    return {"model_path": os.path.join(_CKPT_DIR, _FILENAMES[kind])}
