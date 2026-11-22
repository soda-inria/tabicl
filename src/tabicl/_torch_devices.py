"""Shared torch device resolution helpers."""

from __future__ import annotations

import functools
import subprocess
import sys
import warnings
from typing import Any, Optional, Union

import psutil
import torch

# Preference order when ``device=None``: CUDA → XPU → MPS → CPU.
DEFAULT_DEVICE_PREFERENCE = ("cuda", "xpu", "mps", "cpu")

# Backend-specific names for "this GPU shares host DRAM".
# CUDA uses ``integrated``; Intel XPU uses ``is_integrated_gpu``.
_INTEGRATED_GPU_ATTRS = ("is_integrated_gpu", "integrated", "is_integrated")

# Fallback when the backend does not expose a flag: iGPUs typically advertise
# a "device" pool that is nearly all of host RAM. A discrete 24 GB card on a
# 32 GB machine is 0.75, so 0.9 is high enough to avoid that false positive.
_UNIFIED_MEMORY_TOTAL_RATIO = 0.9

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


def _accelerator_device_properties(device: torch.device) -> Any | None:
    """Return ``torch.<backend>.get_device_properties`` for ``device``, or None."""
    backend_api = getattr(torch, device.type, None)
    getter = getattr(backend_api, "get_device_properties", None)
    if not callable(getter):
        return None
    index = 0 if device.index is None else device.index
    try:
        return getter(index)
    except TypeError:
        try:
            return getter(device)
        except Exception:
            return None
    except Exception:
        return None


def _host_total_memory_bytes() -> int:
    return int(psutil.virtual_memory().total)


def _integrated_flag(props: Any) -> bool | None:
    """Return the backend integrated-GPU flag, or None if the backend has none."""
    for name in _INTEGRATED_GPU_ATTRS:
        if hasattr(props, name):
            return bool(getattr(props, name))
    return None


def device_uses_unified_host_memory(device: Union[str, torch.device]) -> bool:
    """Return whether accelerator memory is the same physical pool as host RAM.

    CPU is not an accelerator and returns ``False`` (it already has its own
    budgeted path). MPS on Apple Silicon always shares DRAM with the process,
    so it returns ``True`` without querying device properties (those APIs are
    missing on MPS, and ``device='mps'`` must stay classifiable on machines
    that cannot run Metal).

    CUDA and XPU use the backend's integrated-GPU property when present
    (``integrated`` / ``is_integrated_gpu``). If that flag is absent, a device
    whose reported ``total_memory`` is at least 90% of host RAM is treated as
    unified — the usual iGPU accounting where the whole DRAM pool is exposed
    as device memory.
    """
    resolved = torch.device(device) if isinstance(device, str) else device
    if resolved.type == "cpu":
        return False
    if resolved.type == "mps":
        return True

    props = _accelerator_device_properties(resolved)
    if props is None:
        return False

    flagged = _integrated_flag(props)
    if flagged is not None:
        return flagged

    total_memory = getattr(props, "total_memory", None)
    if not isinstance(total_memory, (int, float)) or total_memory <= 0:
        return False
    host_total = _host_total_memory_bytes()
    if host_total <= 0:
        return False
    return (float(total_memory) / float(host_total)) >= _UNIFIED_MEMORY_TOTAL_RATIO
