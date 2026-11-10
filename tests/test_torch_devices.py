"""Tests for shared torch device helpers."""

from types import SimpleNamespace

import pytest
import torch

from tabicl._torch_devices import _integrated_flag, device_uses_unified_host_memory
from tests.torch_devices_helpers import skip_if_device_unusable


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cpu", False),
        (torch.device("cpu"), False),
        ("mps", True),
        (torch.device("mps"), True),
        (torch.device("mps:0"), True),
    ],
)
def test_unified_detection_by_device_type(device, expected):
    """CPU is never unified; MPS always is, decided from the type alone.

    Deciding MPS by type keeps it classifiable without Metal (device
    properties are unavailable there), so Linux CI can exercise the policy.
    """
    assert device_uses_unified_host_memory(device) is expected


def test_missing_flag_raises(monkeypatch):
    """Device properties without an integrated-GPU flag raise instead of guessing."""
    monkeypatch.setattr(
        "tabicl._torch_devices._accelerator_device_properties",
        lambda _: SimpleNamespace(total_memory=32 * 1024**3),
    )
    with pytest.raises(RuntimeError, match="shares memory with the host"):
        device_uses_unified_host_memory("xpu")


@pytest.mark.parametrize("device_backend", ["xpu", "cuda"])
def test_unified_detection_matches_device_properties(device_backend):
    """On a real XPU/CUDA device the classification follows the backend's integrated-GPU flag."""
    skip_if_device_unusable(device_backend)
    flagged = _integrated_flag(getattr(torch, device_backend).get_device_properties(0))
    if flagged is None:
        pytest.skip(f"{device_backend} device properties do not expose an integrated-GPU flag")
    assert device_uses_unified_host_memory(device_backend) is flagged
