"""Tests for shared torch device helpers."""

from types import SimpleNamespace

import pytest
import torch

from tabicl._torch_devices import device_uses_unified_host_memory
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


def test_xpu_integrated_flag(monkeypatch):
    monkeypatch.setattr(
        "tabicl._torch_devices._accelerator_device_properties",
        lambda device: SimpleNamespace(is_integrated_gpu=True, total_memory=8 * 1024**3),
    )
    assert device_uses_unified_host_memory("xpu") is True


def test_xpu_discrete_flag(monkeypatch):
    host_total = 32 * 1024**3
    monkeypatch.setattr("tabicl._torch_devices._host_total_memory_bytes", lambda: host_total)
    monkeypatch.setattr(
        "tabicl._torch_devices._accelerator_device_properties",
        lambda device: SimpleNamespace(
            is_integrated_gpu=False,
            total_memory=int(0.95 * host_total),
        ),
    )
    # Explicit discrete flag wins over a host-sized memory pool.
    assert device_uses_unified_host_memory("xpu") is False


def test_cuda_integrated_flag(monkeypatch):
    monkeypatch.setattr(
        "tabicl._torch_devices._accelerator_device_properties",
        lambda device: SimpleNamespace(integrated=True, total_memory=8 * 1024**3),
    )
    assert device_uses_unified_host_memory("cuda") is True


def test_cuda_discrete_flag(monkeypatch):
    monkeypatch.setattr(
        "tabicl._torch_devices._accelerator_device_properties",
        lambda device: SimpleNamespace(integrated=False, total_memory=24 * 1024**3),
    )
    assert device_uses_unified_host_memory("cuda") is False


def test_missing_flag_falls_back_to_host_ram_ratio(monkeypatch):
    host_total = 32 * 1024**3
    monkeypatch.setattr("tabicl._torch_devices._host_total_memory_bytes", lambda: host_total)
    monkeypatch.setattr(
        "tabicl._torch_devices._accelerator_device_properties",
        lambda device: SimpleNamespace(total_memory=int(0.95 * host_total)),
    )
    assert device_uses_unified_host_memory("xpu") is True

    monkeypatch.setattr(
        "tabicl._torch_devices._accelerator_device_properties",
        lambda device: SimpleNamespace(total_memory=int(0.75 * host_total)),
    )
    assert device_uses_unified_host_memory("xpu") is False


def test_missing_properties_is_not_unified(monkeypatch):
    monkeypatch.setattr("tabicl._torch_devices._accelerator_device_properties", lambda device: None)
    assert device_uses_unified_host_memory("xpu") is False
    assert device_uses_unified_host_memory("cuda") is False


@pytest.mark.parametrize("device_backend", ["xpu", "cuda"])
def test_live_unified_detection_matches_device_properties(device_backend):
    skip_if_device_unusable(device_backend)
    props = torch.device(device_backend)
    backend_api = getattr(torch, device_backend)
    device_props = backend_api.get_device_properties(0)
    flagged = None
    for name in ("is_integrated_gpu", "integrated", "is_integrated"):
        if hasattr(device_props, name):
            flagged = bool(getattr(device_props, name))
            break
    if flagged is None:
        pytest.skip(f"{device_backend} device properties do not expose an integrated-GPU flag")
    assert device_uses_unified_host_memory(props) is flagged
