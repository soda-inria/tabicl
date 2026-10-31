"""Tests for resolving ``temporal_features`` in the forecasting pipeline."""

import pytest

from tabicl.forecast._forecaster import _resolve_transforms
from tabicl.forecast.transforms import DatetimeEncoder, FourierEncoder, IndexEncoder


def test_resolve_string_names():
    resolved = _resolve_transforms(["index", "datetime"])
    assert isinstance(resolved[0], IndexEncoder)
    assert isinstance(resolved[1], DatetimeEncoder)


def test_fourier_string_raises_helpful_error():
    """``FourierEncoder`` requires ``periods``, so it cannot be built from a string (GitHub issue #172)."""
    with pytest.raises(ValueError, match=r"FourierEncoder\(periods="):
        _resolve_transforms(["fourier"])


def test_fourier_instance_is_accepted():
    encoder = FourierEncoder(periods=[7])
    resolved = _resolve_transforms(["index", encoder])
    assert isinstance(resolved[0], IndexEncoder)
    assert resolved[1] is encoder


def test_unknown_string_raises():
    with pytest.raises(ValueError, match="Unknown temporal feature"):
        _resolve_transforms(["does_not_exist"])
