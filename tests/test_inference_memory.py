"""Available-CPU-memory detection must respect cgroup limits.

``psutil.virtual_memory().available`` reports the host's free memory and is blind to
cgroup limits, so inside a memory-limited container it overstates what the process may
allocate. That figure feeds ``_resolve_offload_mode``, so overestimating it means
``offload="auto"`` never escalates from CPU to disk -- and the resulting cgroup OOM kill
arrives as SIGKILL, with no Python traceback and no CUDA error.

The process cgroup (from ``/proc/self/cgroup``) is often nested, not
``/sys/fs/cgroup/memory.max``. Walk ancestors and take the tightest ``limit - usage``.
"""

from __future__ import annotations

import pathlib

import psutil
import pytest

from tabicl._model._cgroup_memory import _cgroup_memory_headroom
from tabicl._model.inference import InferenceManager, _cgroup_memory_headroom as _exported_headroom

PHYSICAL = 32 * 10**9
LIMIT = 8_000_000_000
USAGE = 1_000_000_000
HEADROOM = LIMIT - USAGE


def _write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _headroom(root: pathlib.Path, rel: str, *, v2: bool = True) -> int | None:
    proc = root / "proc_cgroup"
    if v2:
        _write(proc, f"0::{rel}\n")
        return _cgroup_memory_headroom(
            proc_cgroup=proc, v2_root=root / "cg", physical_bytes=PHYSICAL
        )
    _write(proc, f"4:memory:{rel}\n")
    return _cgroup_memory_headroom(
        proc_cgroup=proc, v1_root=root / "memory", physical_bytes=PHYSICAL
    )


def test_headroom_is_none_when_there_is_no_cgroup(tmp_path):
    proc = tmp_path / "cgroup"
    _write(proc, "")
    assert (
        _cgroup_memory_headroom(
            proc_cgroup=proc, v2_root=tmp_path / "missing", v1_root=tmp_path / "missing"
        )
        is None
    )


def test_headroom_is_limit_minus_usage(tmp_path):
    # Values below PHYSICAL so the limit is not discarded as "unlimited".
    cg = tmp_path / "cg"
    _write(cg / "memory.max", str(LIMIT))
    _write(cg / "memory.current", str(USAGE))
    assert _headroom(tmp_path, "/") == HEADROOM


def test_cgroup_v2_max_means_unlimited(tmp_path):
    cg = tmp_path / "cg"
    _write(cg / "memory.max", "max")
    _write(cg / "memory.current", str(USAGE))
    assert _headroom(tmp_path, "/") is None


def test_cgroup_v1_unlimited_sentinel_is_not_a_limit(tmp_path):
    mem = tmp_path / "memory"
    _write(mem / "memory.limit_in_bytes", "9223372036854771712")
    _write(mem / "memory.usage_in_bytes", str(USAGE))
    assert _headroom(tmp_path, "/", v2=False) is None


def test_detection_never_raises(monkeypatch):
    def boom(self, *a, **k):
        raise PermissionError("denied")

    monkeypatch.setattr(pathlib.Path, "read_text", boom)
    assert _cgroup_memory_headroom() is None


def test_available_memory_is_the_minimum_of_psutil_and_cgroup(monkeypatch):
    from tabicl._model import inference

    mgr = InferenceManager.__new__(InferenceManager)
    monkeypatch.setattr(inference, "_cgroup_memory_headroom", lambda: 1024 ** 3)
    assert InferenceManager.get_available_cpu_memory(mgr) == pytest.approx(1024.0)
    monkeypatch.setattr(inference, "_cgroup_memory_headroom", lambda: None)
    assert InferenceManager.get_available_cpu_memory(mgr) == pytest.approx(
        psutil.virtual_memory().available / 1024 ** 2, rel=0.2
    )


def test_nested_cgroup_path_is_used(tmp_path):
    """The limit lives on the process cgroup, not ``/sys/fs/cgroup/memory.max``.

    ``docker run --cgroupns=host`` (and Kubernetes) nest the container under
    ``docker/<id>`` while the cgroup root stays ``max``. Reading only the root
    file would report unlimited host RAM.
    """
    cg = tmp_path / "cg"
    nested = cg / "docker" / "abc123"
    _write(cg / "memory.max", "max")
    _write(cg / "memory.current", "0")
    _write(nested / "memory.max", str(LIMIT))
    _write(nested / "memory.current", str(USAGE))
    assert _headroom(tmp_path, "/docker/abc123") == HEADROOM


def test_leaf_unlimited_parent_limit_wins(tmp_path):
    """A child cgroup with ``memory.max = max`` still inherits a parent's cap.

    systemd scopes and in-container helpers often have no limit of their own;
    the pod or slice above them does.
    """
    cg = tmp_path / "cg"
    leaf = cg / "pod" / "ctr"
    _write(cg / "memory.max", str(LIMIT))
    _write(cg / "memory.current", str(USAGE))
    _write(leaf / "memory.max", "max")
    _write(leaf / "memory.current", "100")
    assert _headroom(tmp_path, "/pod/ctr") == HEADROOM


def test_tightest_ancestor_headroom_wins(tmp_path):
    """When several ancestors publish a limit, remaining budget is the minimum.

    A container may look like it has 7 GB free while the pod, shared with
    sidecars, only has 500 MB left. Using the leaf alone would overstate
    headroom and skip disk offload.
    """
    cg = tmp_path / "cg"
    leaf = cg / "pod" / "ctr"
    parent_limit, parent_usage = 4_000_000_000, 3_500_000_000
    _write(cg / "memory.max", "max")
    _write(cg / "memory.current", "0")
    _write(cg / "pod" / "memory.max", str(parent_limit))
    _write(cg / "pod" / "memory.current", str(parent_usage))
    _write(leaf / "memory.max", str(LIMIT))
    _write(leaf / "memory.current", str(USAGE))
    assert _headroom(tmp_path, "/pod/ctr") == parent_limit - parent_usage


def test_missing_leaf_files_walk_to_parent(tmp_path):
    """If the leaf has no memory files (controller not delegated), use the parent."""
    cg = tmp_path / "cg"
    (cg / "docker" / "abc").mkdir(parents=True)
    _write(cg / "docker" / "memory.max", str(LIMIT))
    _write(cg / "docker" / "memory.current", str(USAGE))
    assert _headroom(tmp_path, "/docker/abc") == HEADROOM


def test_v1_nested_memory_controller(tmp_path):
    """cgroup v1 uses ``memory:/rel`` in ``/proc/self/cgroup`` and files under
    ``/sys/fs/cgroup/memory/<rel>``, not the v2 root.
    """
    mem = tmp_path / "memory"
    nested = mem / "docker" / "abc123"
    _write(mem / "memory.limit_in_bytes", "9223372036854771712")
    _write(mem / "memory.usage_in_bytes", "0")
    _write(nested / "memory.limit_in_bytes", str(LIMIT))
    _write(nested / "memory.usage_in_bytes", str(USAGE))
    assert _headroom(tmp_path, "/docker/abc123", v2=False) == HEADROOM


def test_dotdot_in_cgroup_path_is_ignored(tmp_path):
    """A ``..`` segment in the cgroup path must not escape the controller root."""
    cg = tmp_path / "cg"
    _write(cg / "memory.max", "max")
    _write(cg / "memory.current", "0")
    _write(tmp_path / "outside" / "memory.max", str(LIMIT))
    _write(tmp_path / "outside" / "memory.current", str(USAGE))
    assert _headroom(tmp_path, "/../../outside") is None


def test_inference_reexports_the_helper():
    """``InferenceManager`` must keep using the same helper tests import."""
    assert _exported_headroom is _cgroup_memory_headroom
