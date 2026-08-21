"""Shared torch device availability helpers for tests."""

from __future__ import annotations

import pytest
import torch

from tabicl._torch_devices import (
    backend_is_available,
    mps_possibly_faulty,
)


def device_available(device: str | torch.device | None) -> bool:
    """Return whether a torch device backend is available on this host.

    Uses the shared :func:`tabicl._torch_devices.backend_is_available` helper so
    test skips stay aligned with library default-device selection.
    """
    if device is None:
        # The default device is always available (CPU fallback if needed).
        return True
    try:
        device_type = torch.device(device).type
    except (TypeError, RuntimeError, ValueError):
        return False

    return backend_is_available(device_type)


def skip_if_device_unusable(device: str | None) -> None:
    """Skip the current test when ``device`` cannot be used reliably.

    ``device=None`` is always usable: on virtualized Apple Silicon the library
    falls back to CPU. Explicit ``mps`` is skipped when MPS is missing or the
    host is a known-bad VirtualMac guest.
    """
    if device is None:
        return

    if not device_available(device):
        pytest.skip(f"{device} device is not available on this host")
    if device == "mps" and mps_possibly_faulty():
        pytest.skip(
            "MPS skipped on virtualized Apple Silicon "
            "(https://github.com/pytorch/pytorch/issues/192934)"
        )
