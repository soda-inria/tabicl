"""Device-agnostic inference plumbing: placement, AMP, auto-batching, async copies.

CUDA, XPU, and MPS share one inference path (CPU is the only early-return
exception).

CUDA and XPU expose async copy primitives, while MPS does not.

Other devices backends should still work, but will fall back to synchronous
copies with a warning if they don't expose all the required primitives.
"""

import warnings
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest
import torch

from tabicl._model.inference import (
    AsyncCopyManager,
    InferenceManager,
    MemoryEstimator,
    _TOTAL_DEVICE_MEM_MB,
    _device_cache_key,
    devices_match,
)
from tabicl._model.inference_config import InferenceConfig, MgrConfig


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
    """Unindexed devices match index 0; different indices or backends do not."""
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
    # Avoid a redundant host -> device copy when the tensor is already placed correctly.
    moved = mgr._to_exe_device(on_device)
    assert moved is on_device


@pytest.mark.parametrize("device_backend", ["cuda", "xpu", "mps"])
def test_total_device_memory_positive_for_available_backends(device_backend):
    """The backend's total-memory API must be readable on real accelerators."""
    backend_api = getattr(torch, device_backend, None)
    if (
        backend_api is None
        or not callable(getattr(backend_api, "is_available", None))
        or not backend_api.is_available()
    ):
        pytest.skip(f"{device_backend} backend is not available on this host")

    mgr = InferenceManager(enc_name="tf_col", out_dim=16)
    mgr.configure(device=device_backend)
    total_mb = mgr.get_total_device_memory()

    assert total_mb > 0.0, (
        f"Expected non-zero total memory for backend '{device_backend}', "
        f"got {total_mb} MB"
    )


class _FakeMemBackend:
    """Stand-in for ``torch.<backend>`` exposing only the total-memory API.

    ``mem_get_info`` returns a deliberately tiny free figure so that a test using
    it fails loudly if batch sizing ever reads the free half again.
    """

    def __init__(self, total_mb: float, free_mb: float = 1.0, has_mem_get_info: bool = True):
        self.total_bytes = int(total_mb * 1024**2)
        self.free_bytes = int(free_mb * 1024**2)
        self.calls = []
        if has_mem_get_info:
            self.mem_get_info = self._mem_get_info
        else:
            self.recommended_max_memory = self._recommended_max_memory

    def _mem_get_info(self, *args):
        self.calls.append(("mem_get_info", args))
        return self.free_bytes, self.total_bytes

    def _recommended_max_memory(self):
        self.calls.append(("recommended_max_memory", ()))
        return self.total_bytes


class _RaisingMemBackend:
    """Backend whose every memory operation raises: proves none of them is used."""

    def __init__(self):
        self.calls = []

    def mem_get_info(self, *args):
        self.calls.append("mem_get_info")
        raise AssertionError("batch sizing must not query free memory")

    def recommended_max_memory(self):
        self.calls.append("recommended_max_memory")
        raise AssertionError("batch sizing must not query free memory")

    def synchronize(self):
        self.calls.append("synchronize")
        raise AssertionError("batch sizing must not synchronize the device")

    def empty_cache(self):
        self.calls.append("empty_cache")
        raise AssertionError("batch sizing must not flush the allocator cache")


def _cpu_manager(**kwargs) -> InferenceManager:
    """An InferenceManager configured on CPU (no accelerator needed in CI)."""
    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    mgr.configure(device="cpu", use_amp=False, use_fa3=False, use_async=False, **kwargs)
    return mgr


def test_batch_budget_is_1024mb_plus_fraction_of_total_device_memory(monkeypatch):
    """The budget is ``1024 + memory_fraction * total`` and the free half is ignored."""
    total_mb = 24 * 1024.0
    mgr = _cpu_manager(memory_fraction=0.10)
    fake = _FakeMemBackend(total_mb=total_mb, free_mb=8.0)
    monkeypatch.setattr(mgr, "_get_device_backend_api", lambda: fake)

    budget_mb, batch_size = mgr.estimate_safe_batch_size(seq_len=64, include_inputs=True, in_dim=8)

    target_mb = 1024.0 + 0.10 * total_mb
    assert budget_mb == pytest.approx(target_mb)
    assert batch_size == MemoryEstimator.estimate_batch_size(64, target_mb, "tf_col", True, 8)

    # A nearly exhausted device does not change the budget: only the total was read.
    fake.free_bytes = 1
    budget_again, _ = mgr.estimate_safe_batch_size(seq_len=64, include_inputs=True, in_dim=8)
    assert budget_again == pytest.approx(target_mb)

    # Total memory is a static property: queried once, never per prediction.
    assert fake.calls == [("mem_get_info", (mgr.exe_device,))]


def test_batch_budget_falls_back_to_the_base_when_no_total_memory_api(monkeypatch):
    """Backends exposing neither ``mem_get_info`` nor ``recommended_max_memory``."""
    mgr = _cpu_manager(memory_fraction=0.10)
    monkeypatch.setattr(mgr, "_get_device_backend_api", lambda: None)

    assert mgr.get_total_device_memory() == 0.0
    assert mgr.get_batch_memory_budget() == pytest.approx(1024.0)


def test_mps_style_backend_uses_recommended_max_memory_as_total(monkeypatch):
    """``recommended_max_memory`` is the total when ``mem_get_info`` is absent (MPS)."""
    total_mb = 8 * 1024.0
    mgr = _cpu_manager(memory_fraction=0.10)
    fake = _FakeMemBackend(total_mb=total_mb, has_mem_get_info=False)
    monkeypatch.setattr(mgr, "_get_device_backend_api", lambda: fake)

    assert mgr.get_total_device_memory() == pytest.approx(total_mb)
    assert mgr.get_batch_memory_budget() == pytest.approx(1024.0 + 0.10 * total_mb)


def test_batch_sizing_never_synchronizes_flushes_or_queries_free_memory(monkeypatch):
    """Batch sizing must not synchronize, empty the cache, or read free memory.

    Every memory operation of the backend raises: the sizing path still completes
    and degrades to the fixed 1024 MB base instead of calling any of them.
    """
    mgr = _cpu_manager(memory_fraction=0.10)
    fake = _RaisingMemBackend()
    monkeypatch.setattr(mgr, "_get_device_backend_api", lambda: fake)

    budget_mb, batch_size = mgr.estimate_safe_batch_size(seq_len=64, include_inputs=True, in_dim=8)

    assert budget_mb == pytest.approx(1024.0)
    assert batch_size >= 1
    # mem_get_info may be probed for its *total* half; sync/cache never run.
    assert "synchronize" not in fake.calls
    assert "empty_cache" not in fake.calls


def _batching_stage(monkeypatch, enc_name: str, backend) -> InferenceManager:
    """One inference stage routed through the accelerator batching path on CPU.

    ``exe_device`` is set to ``mps`` so ``__call__`` takes the auto-batching branch,
    while every tensor stays on CPU and the forward pass is stubbed: no accelerator is
    needed, yet the real ``estimate_safe_batch_size`` -> budget path still runs.
    """
    mgr = InferenceManager(enc_name=enc_name, out_dim=4)
    mgr.configure(
        device="cpu",
        use_amp=False,
        use_fa3=False,
        use_async=False,
        offload="gpu",
        max_pinned_memory_mb=0.0,  # pinned memory needs a CUDA driver, absent here
    )
    mgr.exe_device = torch.device("mps")
    monkeypatch.setattr(mgr, "_get_device_backend_api", lambda: backend)
    monkeypatch.setattr(mgr, "_to_exe_device", lambda tensor: tensor)
    monkeypatch.setattr(
        mgr,
        "_run_forward",
        lambda forward_fn, inputs: torch.zeros(inputs["features"].shape[0], 4),
    )
    return mgr


def test_total_device_memory_is_queried_once_per_device_for_the_process(monkeypatch):
    """A full prediction -- three stages, three managers -- queries each device once."""
    fake = _FakeMemBackend(total_mb=24 * 1024.0)
    stages = [
        _batching_stage(monkeypatch, enc_name, fake)
        for enc_name in ("tf_col", "tf_row", "tf_icl")
    ]
    features = torch.randn(8, 4, 2)

    for mgr in stages:
        out = mgr(lambda x: x, OrderedDict([("features", features)]))
        assert out.shape[0] == 8, mgr.enc_name

    # Three managers, three configure() calls, one backend query for the process.
    assert fake.calls == [("mem_get_info", (torch.device("mps"),))]

    # configure() no longer invalidates the entry: a second configuration on the same
    # device does not re-query the backend.
    stages[0].configure(
        device="cpu",
        use_amp=False,
        use_fa3=False,
        use_async=False,
        offload="gpu",
        max_pinned_memory_mb=0.0,
    )
    stages[0].exe_device = torch.device("mps")
    monkeypatch.setattr(stages[0], "_get_device_backend_api", lambda: fake)
    stages[0](lambda x: x, OrderedDict([("features", features)]))
    assert fake.calls == [("mem_get_info", (torch.device("mps"),))]


def test_each_execution_device_gets_its_own_total_memory_entry(monkeypatch):
    """A different execution device is a distinct cache key and a fresh query."""
    fake = _FakeMemBackend(total_mb=24 * 1024.0)
    mgr = _batching_stage(monkeypatch, "tf_col", fake)

    assert mgr.get_total_device_memory() == pytest.approx(24 * 1024.0)
    assert len(fake.calls) == 1

    mgr.exe_device = torch.device("xpu:1")
    assert mgr.get_total_device_memory() == pytest.approx(24 * 1024.0)
    assert fake.calls[1] == ("mem_get_info", (torch.device("xpu:1"),))
    assert len(fake.calls) == 2

    # Each device keeps its own cached figure, queried once.
    assert mgr.get_total_device_memory() == pytest.approx(24 * 1024.0)
    mgr.exe_device = torch.device("mps")
    assert mgr.get_total_device_memory() == pytest.approx(24 * 1024.0)
    assert len(fake.calls) == 2
    assert set(_TOTAL_DEVICE_MEM_MB) == {("mps", 0), ("xpu", 1)}


def test_total_device_memory_cache_key_follows_the_devices_match_convention():
    """``cuda`` and ``cuda:0`` share one entry; a different index or backend does not."""
    assert _device_cache_key(torch.device("cuda")) == _device_cache_key(torch.device("cuda:0"))
    assert _device_cache_key(torch.device("cuda:1")) != _device_cache_key(torch.device("cuda:0"))
    assert _device_cache_key(torch.device("cpu")) != _device_cache_key(torch.device("cuda"))


def test_memory_fraction_defaults_to_a_tenth_of_total_device_memory():
    """``memory_fraction`` defaults to 0.10 for every manager configuration."""
    assert "memory_fraction" in MgrConfig._ALLOWED_KEYS
    assert InferenceConfig().COL_CONFIG.memory_fraction == pytest.approx(0.10)

    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    mgr.configure(device="cpu", use_amp=False, use_fa3=False, use_async=False)
    assert mgr.memory_fraction == pytest.approx(0.10)


def test_floor_case_warns_when_one_element_exceeds_the_budget(monkeypatch):
    """A sequence that cannot fit warns instead of silently overshooting."""
    mgr = _cpu_manager(memory_fraction=0.0)  # budget == 1024 MB
    monkeypatch.setattr(mgr, "_get_device_backend_api", lambda: None)

    seq_len = 400_000
    with pytest.warns(RuntimeWarning, match="exceed the target budget"):
        budget_mb, batch_size = mgr.estimate_safe_batch_size(seq_len=seq_len, include_inputs=True, in_dim=8)

    assert budget_mb == pytest.approx(1024.0)
    assert batch_size == 1
    assert MemoryEstimator.estimate_peak_mem(1, seq_len, "tf_col", True, 8) > budget_mb


def _cpu_batching_manager(monkeypatch) -> InferenceManager:
    """A CPU-bound manager routed through the accelerator batching path.

    ``exe_device`` is set to a non-CPU type so ``__call__`` takes the multi-batch
    branch, while every tensor stays on CPU: no accelerator is needed.
    """
    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    mgr.configure(
        device="cpu",
        use_amp=False,
        use_fa3=False,
        use_async=False,
        offload="cpu",
        max_pinned_memory_mb=0.0,  # pinned memory needs a CUDA driver, absent here
    )
    mgr.exe_device = torch.device("mps")
    monkeypatch.setattr(mgr, "_get_device_backend_api", lambda: None)
    monkeypatch.setattr(mgr, "_to_exe_device", lambda tensor: tensor)
    monkeypatch.setattr(mgr, "estimate_safe_batch_size", lambda *args, **kwargs: (1024.0, 4))
    return mgr


def test_oom_retry_halves_the_batch_and_succeeds(monkeypatch):
    """A batch that runs out of memory is retried at half its size."""
    mgr = _cpu_batching_manager(monkeypatch)
    tried = []

    def _run_forward(forward_fn, inputs):
        n = inputs["features"].shape[0]
        tried.append(n)
        if n > 1:
            raise torch.OutOfMemoryError("simulated out of memory")
        return torch.zeros(n, 3, 4)

    monkeypatch.setattr(mgr, "_run_forward", _run_forward)
    features = torch.randn(8, 3, 2)

    out = mgr(lambda x: x, OrderedDict([("features", features)]))

    assert tried[:3] == [4, 2, 1]
    assert set(tried[2:]) == {1}
    assert out.shape == (8, 3, 4)


def test_oom_retry_raises_when_one_element_still_fails(monkeypatch):
    """The retry stops at one element and raises, since nothing smaller can be tried."""
    mgr = _cpu_batching_manager(monkeypatch)
    tried = []

    def _run_forward(forward_fn, inputs):
        tried.append(inputs["features"].shape[0])
        raise torch.OutOfMemoryError("simulated out of memory")

    monkeypatch.setattr(mgr, "_run_forward", _run_forward)
    features = torch.randn(8, 3, 2)

    with pytest.raises(RuntimeError, match="batch size of one element"):
        mgr(lambda x: x, OrderedDict([("features", features)]))

    assert tried == [4, 2, 1]


class _FakeNoAsyncBackend:
    @staticmethod
    def is_available():
        return True


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_async_copy_manager_uses_backend_async_primitives(device_type):
    """Check that the async copy manager uses CUDA/XPU Stream/Event primitives.

    CUDA and XPU device backends are expected to always expose Stream/Event primitives.
    """
    device = torch.device(device_type)
    manager = AsyncCopyManager(device=device, max_pending=2)
    if manager._copy_stream is None:
        pytest.skip(
            f"{device_type} async stream setup is not supported in this runtime"
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
    assert manager.get_bytes_written() == pytest.approx(
        src.numel() * src.element_size() / (1024 * 1024)
    )


def test_async_copy_manager_falls_back_to_sync_without_async_primitives(monkeypatch):
    """Other unknown backends without Stream/Event copy synchronously and warn."""
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
    assert manager.get_bytes_written() == pytest.approx(
        src.numel() * src.element_size() / (1024 * 1024)
    )


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
    )

    # Force a tiny safe batch so the manager must split.
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


def test_mps_memory_estimate_uses_recommended_minus_current(monkeypatch):
    """The MPS memory inspection API is specific compared to CUDA/XPU.

    In particular, it does not expose mem_get_info, and the available memory is
    therefore estimated as recommended_max_memory minus
    current_allocated_memory. On unified memory the estimate is also capped by host
    RAM times ``unified_memory_safety_factor``.
    """
    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    mgr.configure(device="mps", unified_memory_safety_factor=0.5)

    fake_api = MagicMock()
    fake_api.mem_get_info = None
    fake_api.recommended_max_memory.return_value = 8 * 1024 * 1024 * 1024
    fake_api.current_allocated_memory.return_value = 2 * 1024 * 1024 * 1024
    monkeypatch.setattr(mgr, "_get_device_backend_api", lambda: fake_api)

    # Host RAM larger than the recommended-minus-current estimate.
    monkeypatch.setattr(mgr, "get_available_cpu_memory", lambda: 16 * 1024.0)
    available_mb = mgr.get_available_gpu_memory()
    assert available_mb == pytest.approx(6 * 1024.0)

    # Host RAM smaller than the estimate: the unified-memory cap wins.
    monkeypatch.setattr(mgr, "get_available_cpu_memory", lambda: 2 * 1024.0)
    available_mb = mgr.get_available_gpu_memory()
    assert available_mb == pytest.approx(1024.0)


@pytest.mark.parametrize(("previous", "expected"), [(1.0, 7 / 16), (0.25, 0.25)])
def test_allocator_cap_on_integrated_gpu(monkeypatch, previous, expected):
    """On an integrated GPU, inference caps the allocator at the usable host RAM.

    With 1 GiB already reserved, 8 GiB of host RAM available and a
    ``unified_memory_safety_factor`` of 0.75, a 16 GiB device may use (1 + 8 * 0.75) / 16 of
    its memory, unless a stricter cap was already set. The previous cap is restored
    afterwards. The backend is faked so this runs without the device.
    """
    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    mgr.configure(device="cuda", use_amp=False, use_fa3=False, unified_memory_safety_factor=0.75)
    fake_api = MagicMock()
    fake_api.get_per_process_memory_fraction.return_value = previous
    fake_api.memory_reserved.return_value = 1024**3
    fake_api.get_device_properties.return_value.total_memory = 16 * 1024**3
    monkeypatch.setattr(mgr, "_get_device_backend_api", lambda: fake_api)
    monkeypatch.setattr(mgr, "_unified_host", True)
    monkeypatch.setattr(mgr, "get_available_cpu_memory", lambda: 8 * 1024.0)

    with mgr._cap_allocator_to_host_memory():
        fake_api.set_per_process_memory_fraction.assert_called_once_with(expected, mgr.exe_device)
    fake_api.set_per_process_memory_fraction.assert_called_with(previous, mgr.exe_device)


def test_allocator_cap_not_set_on_mps(monkeypatch):
    """MPS has no way to read its allocator cap back, so inference leaves it untouched."""
    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    mgr.configure(device="mps", use_amp=False, use_fa3=False)
    fake_api = MagicMock()
    monkeypatch.setattr(mgr, "_get_device_backend_api", lambda: fake_api)

    with mgr._cap_allocator_to_host_memory():
        pass
    fake_api.set_per_process_memory_fraction.assert_not_called()


def test_allocator_cap_warns_without_torch_support(monkeypatch):
    """Torch versions without get/set_per_process_memory_fraction (CUDA < 2.6, XPU < 2.10) only warn."""
    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    mgr.configure(device="cuda", use_amp=False, use_fa3=False)
    monkeypatch.setattr(mgr, "_get_device_backend_api", lambda: MagicMock(spec=[]))
    monkeypatch.setattr(mgr, "_unified_host", True)

    with pytest.warns(RuntimeWarning, match="cannot limit"):
        with mgr._cap_allocator_to_host_memory():
            pass


def test_call_before_configure_raises():
    """Running inference on a manager that was never configured raises a clear error."""
    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    with pytest.raises(RuntimeError, match="must be configured"):
        mgr(forward_fn=None, inputs=None)


def test_call_runs_forward_under_allocator_cap(monkeypatch):
    """The forward pass runs while the allocator cap is active."""
    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    mgr.configure(device="cpu")
    cap = MagicMock()
    monkeypatch.setattr(mgr, "_cap_allocator_to_host_memory", cap)

    def forward_fn(x):
        cap.return_value.__enter__.assert_called_once()
        return x

    mgr(forward_fn, OrderedDict(x=torch.zeros(1, 1, 1)))

