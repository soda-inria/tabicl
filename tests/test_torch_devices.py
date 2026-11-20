"""Tests for shared torch device helpers."""

from types import SimpleNamespace

import pytest
import torch

from tabicl._torch_devices import device_uses_unified_host_memory
from tests.torch_devices_helpers import skip_if_device_unusable


@pytest.mark.parametrize(("device", "expected"), [("cpu", False), ("mps", True)])
def test_unified_detection_by_device_type(device, expected):
    """CPU is never unified; MPS always is, decided from the type alone so it works without Metal."""
    assert device_uses_unified_host_memory(device) is expected


def test_missing_flag_is_unknown(monkeypatch):
    """Device properties without an integrated-GPU flag (XPU before torch 2.13) give None and a warning."""
    monkeypatch.setattr(torch, "xpu", SimpleNamespace(get_device_properties=lambda _: SimpleNamespace()), raising=False)
    with pytest.warns(RuntimeWarning, match="shares memory with the host"):
        assert device_uses_unified_host_memory("xpu") is None


@pytest.mark.parametrize(("device_backend", "flag"), [("cuda", "is_integrated"), ("xpu", "is_integrated_gpu")])
def test_unified_detection_matches_device_properties(device_backend, flag):
    """On a real XPU/CUDA device the classification follows the backend's integrated-GPU flag."""
    skip_if_device_unusable(device_backend)
    props = getattr(torch, device_backend).get_device_properties(0)
    if not hasattr(props, flag):
        pytest.skip(f"{device_backend} device properties do not expose {flag}")
    assert device_uses_unified_host_memory(device_backend) is bool(getattr(props, flag))
