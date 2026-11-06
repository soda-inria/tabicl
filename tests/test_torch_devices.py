"""Tests for shared torch device helpers."""

from types import SimpleNamespace

import pytest
import torch

from tabicl._torch_devices import _integrated_flag, device_uses_unified_host_memory
from tests.torch_devices_helpers import skip_if_device_unusable


def test_cpu_is_not_unified_host_memory():
    assert device_uses_unified_host_memory("cpu") is False
    assert device_uses_unified_host_memory(torch.device("cpu")) is False


def test_mps_is_always_unified_host_memory():
    # Classified by backend type so Linux CI can exercise the unified-memory
    # policy without Metal, and so MPS stays True even without device properties.
    assert device_uses_unified_host_memory("mps") is True
    assert device_uses_unified_host_memory(torch.device("mps")) is True
    assert device_uses_unified_host_memory(torch.device("mps:0")) is True


@pytest.mark.parametrize(
    ("pool_ratio", "expected"),
    [
        pytest.param(0.95, True, id="host-sized-pool"),
        pytest.param(0.75, False, id="partial-pool"),
    ],
)
def test_missing_flag_falls_back_to_host_ram_ratio(monkeypatch, pool_ratio, expected):
    """Without an integrated-GPU flag, a device pool of >= 90% host RAM counts as unified.

    Real CUDA and XPU builds expose the flag, so this last-resort path can only
    be exercised with faked device properties.
    """
    host_total = 32 * 1024**3
    monkeypatch.setattr("tabicl._torch_devices._host_total_memory_bytes", lambda: host_total)
    monkeypatch.setattr(
        "tabicl._torch_devices._accelerator_device_properties",
        lambda _: SimpleNamespace(total_memory=int(pool_ratio * host_total)),
    )
    assert device_uses_unified_host_memory("xpu") is expected


@pytest.mark.parametrize("device_backend", ["xpu", "cuda"])
def test_unified_detection_matches_device_properties(device_backend):
    """On a real XPU/CUDA device the classification follows the backend's integrated-GPU flag."""
    skip_if_device_unusable(device_backend)
    flagged = _integrated_flag(getattr(torch, device_backend).get_device_properties(0))
    if flagged is None:
        pytest.skip(f"{device_backend} device properties do not expose an integrated-GPU flag")
    assert device_uses_unified_host_memory(device_backend) is flagged
