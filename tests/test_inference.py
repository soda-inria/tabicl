import warnings

import pytest
import torch

from tabicl._model.inference import AsyncCopyManager, InferenceManager


@pytest.mark.parametrize("device_backend", ["cuda", "xpu"])
def test_get_available_gpu_memory_non_zero_for_available_backends(device_backend):
    backend_api = getattr(torch, device_backend, None)
    if backend_api is None or not callable(getattr(backend_api, "is_available", None)) or not backend_api.is_available():
        pytest.skip(f"{device_backend} backend is not available on this host")

    mgr = InferenceManager(enc_name="tf_col", out_dim=16)
    mgr.configure(device=device_backend)
    available_mb = mgr.get_available_gpu_memory()

    assert available_mb > 0.0, (
        f"Expected non-zero available memory for backend '{device_backend}', "
        f"got {available_mb} MB"
    )


class _FakeNoAsyncBackend:
    @staticmethod
    def is_available():
        return True


@pytest.mark.parametrize("device_backend", ["cuda", "xpu"])
def test_async_copy_manager_uses_backend_async_primitives(device_backend):
    backend_api = getattr(torch, device_backend, None)
    if backend_api is None or not callable(getattr(backend_api, "is_available", None)) or not backend_api.is_available():
        pytest.skip(f"{device_backend} backend is not available on this host")

    required_primitives = ("Stream", "Event", "current_stream", "stream")
    if any(not callable(getattr(backend_api, primitive, None)) for primitive in required_primitives):
        pytest.skip(f"{device_backend} backend does not expose full async stream/event primitives")

    device = torch.device(device_backend)
    manager = AsyncCopyManager(device=device, max_pending=2)
    if manager._copy_stream is None:
        pytest.skip(f"{device_backend} async stream setup is not supported in this runtime")

    src = torch.ones((4,), dtype=torch.float32, device=device)
    dst = torch.zeros((4,), dtype=torch.float32)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        manager.submit_copy(src, dst, (slice(None),))

    # With async-capable backend, copy is pending before drain.
    assert len(manager._pending) == 1
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught_warnings)
    manager.drain_all()

    assert len(manager._pending) == 0
    assert torch.equal(dst, src.cpu())
    assert manager.get_bytes_written() > 0.0


def test_async_copy_manager_falls_back_to_sync_without_async_primitives(monkeypatch):
    backend_name = "fakenoasync"
    monkeypatch.setattr(torch, backend_name, _FakeNoAsyncBackend, raising=False)

    device = type("Device", (), {"type": backend_name})()
    manager = AsyncCopyManager(device=device)

    src = torch.ones((4,), dtype=torch.float32)
    dst = torch.zeros((4,), dtype=torch.float32)

    with pytest.warns(RuntimeWarning, match="falling back to synchronous copy"):
        manager.submit_copy(src, dst, (slice(None),))

    # No async primitives: direct sync copy, no pending events.
    assert len(manager._pending) == 0
    assert torch.equal(dst, src)
    assert manager.get_bytes_written() > 0.0
