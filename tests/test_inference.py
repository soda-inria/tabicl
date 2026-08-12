import warnings
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest
import torch

from tabicl._model.inference import AsyncCopyManager, InferenceManager, devices_match


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ("cpu", "cpu", True),
        ("cuda", "cuda:0", True),
        ("cuda:0", "cuda", True),
        ("cuda:0", "cuda:0", True),
        ("cuda:0", "cuda:1", False),
        ("mps", "mps:0", True),
        ("mps:0", "mps", True),
        ("mps:0", "mps:0", True),
        ("xpu", "xpu:0", True),
        ("xpu:0", "xpu", True),
        ("xpu:0", "xpu:1", False),
        ("cuda", "cpu", False),
        ("cuda", "mps", False),
        ("mps", "xpu", False),
    ],
)
def test_devices_match(left, right, expected):
    assert devices_match(torch.device(left), torch.device(right)) is expected


@pytest.mark.parametrize("device_backend", ["cpu", "cuda", "xpu", "mps"])
def test_is_tensor_on_exe_device_accepts_unindexed_and_indexed(device_backend):
    """``device='cuda'`` must match tensors that land on ``cuda:0`` (same for MPS/XPU)."""
    backend_api = (
        getattr(torch, device_backend, None) if device_backend != "cpu" else None
    )
    if device_backend != "cpu" and (
        backend_api is None
        or not callable(getattr(backend_api, "is_available", None))
        or not backend_api.is_available()
    ):
        pytest.skip(f"{device_backend} backend is not available on this host")

    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    mgr.configure(device=device_backend, use_amp=False, use_fa3=False, use_async=False)

    # configure() stores an unindexed device (e.g. ``cuda``); tensors usually get index 0.
    assert mgr.exe_device.index is None or device_backend == "cpu"

    on_device = torch.zeros(1, device=device_backend)
    on_cpu = torch.zeros(1, device="cpu")
    assert mgr._is_tensor_on_exe_device(on_device) is True
    assert mgr._is_tensor_on_exe_device(on_cpu) is (device_backend == "cpu")
    # Avoid a redundant host→device copy when the tensor is already placed correctly.
    moved = mgr._to_exe_device(on_device)
    assert moved is on_device


@pytest.mark.parametrize("device_backend", ["cuda", "xpu", "mps"])
def test_get_available_gpu_memory_non_zero_for_available_backends(device_backend):
    backend_api = getattr(torch, device_backend, None)
    if (
        backend_api is None
        or not callable(getattr(backend_api, "is_available", None))
        or not backend_api.is_available()
    ):
        pytest.skip(f"{device_backend} backend is not available on this host")

    if device_backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS backend is not available on this host")

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
    if (
        backend_api is None
        or not callable(getattr(backend_api, "is_available", None))
        or not backend_api.is_available()
    ):
        pytest.skip(f"{device_backend} backend is not available on this host")

    required_primitives = ("Stream", "Event", "current_stream", "stream")
    if any(
        not callable(getattr(backend_api, primitive, None))
        for primitive in required_primitives
    ):
        pytest.skip(
            f"{device_backend} backend does not expose full async stream/event primitives"
        )

    device = torch.device(device_backend)
    manager = AsyncCopyManager(device=device, max_pending=2)
    if manager._copy_stream is None:
        pytest.skip(
            f"{device_backend} async stream setup is not supported in this runtime"
        )

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


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_async_copy_manager_mps_sync_fallback_is_silent():
    """MPS has no D2H streams; sync fallback must not warn."""
    manager = AsyncCopyManager(device=torch.device("mps"))
    src = torch.ones((4,), dtype=torch.float32, device="mps")
    dst = torch.zeros((4,), dtype=torch.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        manager.submit_copy(src, dst, (slice(None),))

    assert len(manager._pending) == 0
    assert torch.equal(dst, src.cpu())
    assert not any(
        issubclass(w.category, RuntimeWarning) and "AsyncCopyManager" in str(w.message)
        for w in caught
    )


def test_cpu_path_uses_run_forward():
    """CPU still skips auto-batching but must go through ``_run_forward``."""
    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    mgr.configure(device="cpu", use_amp=False, use_fa3=False, use_async=False)

    called = {"run_forward": 0}

    original = mgr._run_forward

    def _spy(forward_fn, inputs):
        called["run_forward"] += 1
        return original(forward_fn, inputs)

    mgr._run_forward = _spy

    features = torch.randn(2, 3, 5, 1)

    def forward_fn(features):
        return torch.zeros(*features.shape[:-1], 4, dtype=features.dtype)

    out = mgr(forward_fn, OrderedDict([("features", features)]))
    assert called["run_forward"] == 1
    assert out.shape == (2, 3, 5, 4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_path_uses_run_forward_and_amp():
    """MPS must not take the old CPU-like early return; AMP should engage."""
    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    mgr.configure(
        device="mps", use_amp=True, use_fa3=False, use_async=False, offload="gpu"
    )

    called = {"run_forward": 0, "saw_autocast": False}
    original = mgr._run_forward

    def _spy(forward_fn, inputs):
        called["run_forward"] += 1
        return original(forward_fn, inputs)

    mgr._run_forward = _spy

    features = torch.randn(2, 3, 8, 1, device="mps")

    def forward_fn(features):
        # Device-typed query: bare is_autocast_enabled() is CUDA-centric.
        called["saw_autocast"] = bool(torch.is_autocast_enabled("mps"))
        return torch.zeros(
            *features.shape[:-1], 4, device=features.device, dtype=features.dtype
        )

    out = mgr(forward_fn, OrderedDict([("features", features)]))
    assert called["run_forward"] >= 1
    assert called["saw_autocast"] is True
    assert out.device.type == "mps"
    assert out.shape == (2, 3, 8, 4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_auto_batch_uses_memory_estimate(monkeypatch):
    """With an approximate free-memory query, MPS should enter the batching path."""
    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    mgr.configure(
        device="mps",
        use_amp=False,
        use_fa3=False,
        use_async=False,
        offload="gpu",
        min_batch_size=1,
    )

    # Force a tiny safe batch so the manager must split.
    monkeypatch.setattr(mgr, "get_available_gpu_memory", lambda: 1.0)
    monkeypatch.setattr(
        mgr,
        "estimate_safe_batch_size",
        lambda seq_len, include_inputs=True, in_dim=None, max_bs=50000: (1.0, 1),
    )

    features = torch.randn(4, 2, 8, 1, device="mps")
    n_calls = {"n": 0}

    def forward_fn(features):
        n_calls["n"] += 1
        return torch.zeros(
            *features.shape[:-1], 4, device=features.device, dtype=features.dtype
        )

    out = mgr(forward_fn, OrderedDict([("features", features)]))
    assert (
        n_calls["n"] >= 2
    )  # batch size 1 over leading dim product 8 -> multiple calls
    assert out.shape == (4, 2, 8, 4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_memory_estimate_uses_recommended_minus_current(monkeypatch):
    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    mgr.configure(device="mps")

    fake_api = MagicMock()
    fake_api.is_available.return_value = True
    fake_api.mem_get_info = None
    fake_api.recommended_max_memory.return_value = 8 * 1024 * 1024 * 1024
    fake_api.current_allocated_memory.return_value = 2 * 1024 * 1024 * 1024
    fake_api.synchronize = MagicMock()
    fake_api.empty_cache = MagicMock()

    monkeypatch.setattr(mgr, "_get_device_backend_api", lambda: fake_api)
    available_mb = mgr.get_available_gpu_memory()
    assert available_mb == pytest.approx(6 * 1024.0)
