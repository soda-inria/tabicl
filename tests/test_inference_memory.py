"""Available-CPU-memory detection must respect cgroup limits.

``psutil.virtual_memory().available`` reports the host's free memory and is blind to
cgroup limits, so inside a memory-limited container it overstates what the process may
allocate. That figure feeds ``_resolve_offload_mode``, so overestimating it means
``offload="auto"`` never escalates from CPU to disk -- and the resulting cgroup OOM kill
arrives as SIGKILL, with no Python traceback and no CUDA error.
"""

from __future__ import annotations

import pathlib

import psutil
import pytest

from tabicl._model.inference import InferenceManager, _cgroup_memory_headroom


def test_headroom_is_none_when_there_is_no_cgroup(monkeypatch):
    def missing(self, *a, **k):
        raise FileNotFoundError(str(self))

    monkeypatch.setattr(pathlib.Path, "read_text", missing)
    assert _cgroup_memory_headroom() is None


def test_headroom_is_limit_minus_usage(monkeypatch):
    # Values deliberately below any plausible host RAM: a limit at or above physical
    # memory is treated as "no real constraint", which is how cgroup v1's unlimited
    # sentinel is caught, so a larger fixture would be discarded and assert nothing.
    def fake(self, *a, **k):
        name = str(self)
        if name.endswith(("memory.max", "memory.limit_in_bytes")):
            return "8000000000"
        if name.endswith(("memory.current", "memory.usage_in_bytes")):
            return "1000000000"
        raise FileNotFoundError(name)

    monkeypatch.setattr(pathlib.Path, "read_text", fake)
    assert _cgroup_memory_headroom() == 7_000_000_000


def test_cgroup_v2_max_means_unlimited(monkeypatch):
    monkeypatch.setattr(pathlib.Path, "read_text", lambda self, *a, **k: "max")
    assert _cgroup_memory_headroom() is None


def test_cgroup_v1_unlimited_sentinel_is_not_a_limit(monkeypatch):
    # v1 signals "unlimited" with a value near 2**63. Treating it as a real limit would
    # make every uncontained run believe it had exabytes of headroom.
    monkeypatch.setattr(pathlib.Path, "read_text",
                        lambda self, *a, **k: "9223372036854771712")
    assert _cgroup_memory_headroom() is None


def test_detection_never_raises(monkeypatch):
    # A memory estimate is not worth failing an inference call over; every error path
    # must fall back to the previous behaviour.
    def boom(self, *a, **k):
        raise PermissionError("denied")

    monkeypatch.setattr(pathlib.Path, "read_text", boom)
    assert _cgroup_memory_headroom() is None


def test_available_memory_is_the_minimum_of_psutil_and_cgroup(monkeypatch):
    from tabicl._model import inference

    mgr = InferenceManager.__new__(InferenceManager)

    # Under a cgroup cap, the cap wins.
    monkeypatch.setattr(inference, "_cgroup_memory_headroom", lambda: 1024 ** 3)
    assert InferenceManager.get_available_cpu_memory(mgr) == pytest.approx(1024.0)

    # Without one, behaviour is exactly as before -- this change can only shrink the
    # estimate, never grow it.
    monkeypatch.setattr(inference, "_cgroup_memory_headroom", lambda: None)
    assert InferenceManager.get_available_cpu_memory(mgr) == pytest.approx(
        psutil.virtual_memory().available / 1024 ** 2, rel=0.2)
