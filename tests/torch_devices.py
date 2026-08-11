"""Shared torch device availability helpers for tests."""

from __future__ import annotations

import functools
import subprocess
import sys

import pytest
import torch


def sysctl(name: str) -> str | None:
    """Return a sysctl string value, or None if unavailable."""
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def device_available(device: str | torch.device | None) -> bool:
    """Return whether a torch device backend is available on this host.

    Uses torch's backend naming convention where a device type maps to
    ``torch.<device_type>`` exposing an ``is_available()`` function.
    """
    if device is None:
        # The default device is always available.
        return True
    try:
        device_type = torch.device(device).type
    except (TypeError, RuntimeError, ValueError):
        return False

    if device_type == "cpu":
        return True

    backend_api = getattr(torch, device_type, None)
    is_available = getattr(backend_api, "is_available", None)
    if not callable(is_available):
        return False
    return bool(is_available())


@functools.lru_cache(maxsize=1)
def mps_numerically_reliable() -> bool:
    """Return whether MPS is expected to produce correct TabICL results.

    GitHub Actions macOS arm64 runners are VirtualMac guests (``hw.model`` like
    ``VirtualMac2,1``, CPU brand like ``Apple M1 (Virtual)``). On those hosts
    ``torch.nn.functional.linear`` on 3D inputs silently returns corrupted
    results with bias (PyTorch issue
    https://github.com/pytorch/pytorch/issues/192934). Real Apple Silicon
    hardware (non-virtual ``sysctl`` identity) is fine.
    """
    if not device_available("mps"):
        return False
    if sys.platform != "darwin":
        return False

    brand = sysctl("machdep.cpu.brand_string") or ""
    model = sysctl("hw.model") or ""
    # Match GHA VirtualMac guests; keep parity tests on physical Macs.
    # See https://github.com/pytorch/pytorch/issues/192934
    if "Virtual" in brand or model.startswith("VirtualMac"):
        return False
    return True


def default_device_type() -> str:
    """Return the device type ``device=None`` would resolve to."""
    if device_available("cuda"):
        return "cuda"
    if device_available("xpu"):
        return "xpu"
    if device_available("mps"):
        return "mps"
    return "cpu"


def skip_if_device_unusable(device: str | None) -> None:
    """Skip the current test when ``device`` cannot be used reliably."""
    if device is None:
        resolved = default_device_type()
        if resolved == "mps" and not mps_numerically_reliable():
            pytest.skip(
                "device=None would select unreliable MPS "
                "(https://github.com/pytorch/pytorch/issues/192934)"
            )
        return

    if not device_available(device):
        pytest.skip(f"{device} device is not available on this host")
    if device == "mps" and not mps_numerically_reliable():
        pytest.skip(
            "MPS skipped on virtualized Apple Silicon "
            "(https://github.com/pytorch/pytorch/issues/192934)"
        )
