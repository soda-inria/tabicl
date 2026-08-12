"""Shared torch device resolution helpers."""

from __future__ import annotations

import functools
import subprocess
import sys
import warnings
from typing import Optional, Union

import torch

# Preference order when ``device=None``: CUDA → XPU → MPS → CPU.
DEFAULT_DEVICE_PREFERENCE = ("cuda", "xpu", "mps", "cpu")

# Virtualized Apple Silicon can silently corrupt MPS ``F.linear`` on 3D inputs.
MPS_NUMERICS_ISSUE_URL = "https://github.com/pytorch/pytorch/issues/192934"


def _sysctl(name: str) -> str | None:
    """Return a sysctl string value, or None if unavailable."""
    try:
        return subprocess.check_output(
            ["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@functools.lru_cache(maxsize=1)
def mps_possibly_faulty() -> bool:
    """Return whether this macOS host may have broken MPS numerics.

    GitHub Actions macOS arm64 runners are VirtualMac guests (``hw.model`` like
    ``VirtualMac2,1``, CPU brand like ``Apple M1 (Virtual)``). On those hosts
    MPS can silently return incorrect results (PyTorch issue
    https://github.com/pytorch/pytorch/issues/192934). Real Apple Silicon is fine.

    Always runs the hardware identity check on Darwin; returns ``False`` on other
    platforms.
    """
    if sys.platform != "darwin":
        return False

    brand = _sysctl("machdep.cpu.brand_string") or ""
    model = _sysctl("hw.model") or ""
    return "Virtual" in brand or model.startswith("VirtualMac")


def backend_is_available(device_type: str) -> bool:
    """Return whether ``torch.<device_type>`` reports itself available.

    Uses the usual backend convention where accelerators expose
    ``torch.<backend>.is_available()``. CPU is always available.
    """
    if device_type == "cpu":
        return True

    backend_api = getattr(torch, device_type, None)
    is_available = getattr(backend_api, "is_available", None)
    if not callable(is_available):
        return False
    return bool(is_available())


def resolve_default_device() -> torch.device:
    """Return the default device: CUDA → XPU → MPS → CPU.

    On virtualized macOS hosts with known-bad MPS numerics, MPS is skipped and
    CPU is used instead (with a warning).
    """
    for device_type in DEFAULT_DEVICE_PREFERENCE:
        if not backend_is_available(device_type):
            continue
        if device_type == "mps" and mps_possibly_faulty():
            warnings.warn(
                "MPS appears to run on virtualized Apple Silicon where PyTorch "
                f"can return incorrect results ({MPS_NUMERICS_ISSUE_URL}). "
                "Falling back to CPU because device=None. Pass device='mps' to "
                "force MPS anyway, or device='cpu' to choose CPU silently.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        return torch.device(device_type)
    return torch.device("cpu")


def resolve_torch_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve ``None``, a device string, or a ``torch.device`` to a concrete device.

    ``None`` selects :func:`resolve_default_device`. Explicit ``mps`` on a
    possibly faulty virtualized Mac keeps MPS but warns and recommends CPU.
    """
    if device is None:
        return resolve_default_device()

    resolved = torch.device(device) if isinstance(device, str) else device
    if resolved.type == "mps" and mps_possibly_faulty():
        warnings.warn(
            "device='mps' was requested on virtualized Apple Silicon where "
            f"PyTorch can return incorrect results ({MPS_NUMERICS_ISSUE_URL}). "
            "Consider passing device='cpu' instead.",
            RuntimeWarning,
            stacklevel=2,
        )
    return resolved
