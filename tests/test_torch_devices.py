"""Tests for shared torch device helpers."""

from types import SimpleNamespace

import pytest
import torch

from tabicl._torch_devices import _integrated_flag, device_uses_unified_host_memory
from tests.torch_devices_helpers import skip_if_device_unusable

GiB = 1024**3
HOST_TOTAL = 32 * GiB


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
    ("device", "props", "expected"),
    [
        pytest.param("xpu", SimpleNamespace(is_integrated_gpu=True, total_memory=8 * GiB), True, id="xpu-integrated"),
        # An explicit discrete flag wins over a host-sized memory pool.
        pytest.param(
            "xpu",
            SimpleNamespace(is_integrated_gpu=False, total_memory=int(0.95 * HOST_TOTAL)),
            False,
            id="xpu-discrete",
        ),
        pytest.param("cuda", SimpleNamespace(is_integrated=True, total_memory=8 * GiB), True, id="cuda-integrated"),
        pytest.param("cuda", SimpleNamespace(is_integrated=False, total_memory=24 * GiB), False, id="cuda-discrete"),
        # Without a flag, fall back to the device-pool / host-RAM ratio.
        pytest.param("xpu", SimpleNamespace(total_memory=int(0.95 * HOST_TOTAL)), True, id="no-flag-host-sized-pool"),
        pytest.param("xpu", SimpleNamespace(total_memory=int(0.75 * HOST_TOTAL)), False, id="no-flag-small-pool"),
        pytest.param("xpu", None, False, id="xpu-no-properties"),
        pytest.param("cuda", None, False, id="cuda-no-properties"),
    ],
)
def test_unified_detection_from_device_properties(monkeypatch, device, props, expected):
    monkeypatch.setattr("tabicl._torch_devices._host_total_memory_bytes", lambda: HOST_TOTAL)
    monkeypatch.setattr("tabicl._torch_devices._accelerator_device_properties", lambda _: props)
    assert device_uses_unified_host_memory(device) is expected


@pytest.mark.parametrize("device_backend", ["xpu", "cuda"])
def test_live_unified_detection_matches_device_properties(device_backend):
    skip_if_device_unusable(device_backend)
    flagged = _integrated_flag(getattr(torch, device_backend).get_device_properties(0))
    if flagged is None:
        pytest.skip(f"{device_backend} device properties do not expose an integrated-GPU flag")
    assert device_uses_unified_host_memory(device_backend) is flagged
