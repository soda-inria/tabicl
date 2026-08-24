"""Available-CPU-memory detection must respect cgroup limits.

``psutil.virtual_memory().available`` reports the host's free memory and is blind to
cgroup limits, so inside a memory-limited container it overstates what the process may
allocate. That figure feeds ``_resolve_offload_mode``, so overestimating it means
``offload="auto"`` never escalates from CPU to disk -- and the resulting cgroup OOM kill
arrives as SIGKILL, with no Python traceback and no CUDA error.

Limits are often not at ``/sys/fs/cgroup/memory.max``: they live on the process cgroup
or an ancestor (``docker run --cgroupns=host``, Kubernetes, systemd). Detection must
walk that hierarchy and take the tightest ``limit - usage``.
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


def _v2_mountinfo(mount: pathlib.Path) -> str:
    return (
        f"36 35 0:27 / {mount} rw,nosuid,nodev,noexec,relatime - "
        "cgroup2 cgroup2 rw,nsdelegate\n"
    )


def _v1_mountinfo(mount: pathlib.Path) -> str:
    return (
        f"36 35 0:27 / {mount} rw,nosuid,nodev,noexec,relatime - "
        "cgroup cgroup rw,memory\n"
    )


def _headroom(proc_cgroup: pathlib.Path, mountinfo: pathlib.Path) -> int | None:
    return _cgroup_memory_headroom(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=mountinfo,
        physical_bytes=PHYSICAL,
    )


def test_headroom_is_none_when_there_is_no_cgroup(tmp_path):
    proc = tmp_path / "cgroup"
    mountinfo = tmp_path / "mountinfo"
    _write(proc, "")
    _write(mountinfo, "1 0 8:1 / / rw - ext4 /dev/sda1 rw\n")
    assert _headroom(proc, mountinfo) is None


def test_headroom_is_limit_minus_usage(tmp_path):
    # Values deliberately below PHYSICAL: a limit at or above physical memory is
    # treated as "no real constraint", which is how cgroup v1's unlimited
    # sentinel is caught, so a larger fixture would be discarded and assert nothing.
    mount = tmp_path / "cg"
    _write(mount / "memory.max", str(LIMIT))
    _write(mount / "memory.current", str(USAGE))
    proc = tmp_path / "cgroup"
    mountinfo = tmp_path / "mountinfo"
    _write(proc, "0::/\n")
    _write(mountinfo, _v2_mountinfo(mount))
    assert _headroom(proc, mountinfo) == HEADROOM


def test_cgroup_v2_max_means_unlimited(tmp_path):
    mount = tmp_path / "cg"
    _write(mount / "memory.max", "max")
    _write(mount / "memory.current", str(USAGE))
    proc = tmp_path / "cgroup"
    mountinfo = tmp_path / "mountinfo"
    _write(proc, "0::/\n")
    _write(mountinfo, _v2_mountinfo(mount))
    assert _headroom(proc, mountinfo) is None


def test_cgroup_v1_unlimited_sentinel_is_not_a_limit(tmp_path):
    # v1 signals "unlimited" with a value near 2**63. Treating it as a real limit would
    # make every uncontained run believe it had exabytes of headroom.
    mount = tmp_path / "memory"
    _write(mount / "memory.limit_in_bytes", "9223372036854771712")
    _write(mount / "memory.usage_in_bytes", str(USAGE))
    proc = tmp_path / "cgroup"
    mountinfo = tmp_path / "mountinfo"
    _write(proc, "4:memory:/\n")
    _write(mountinfo, _v1_mountinfo(mount))
    assert _headroom(proc, mountinfo) is None


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
        psutil.virtual_memory().available / 1024 ** 2, rel=0.2
    )


def test_nested_cgroup_path_is_used(tmp_path):
    """``docker run --cgroupns=host`` stores the limit under /sys/fs/cgroup/docker/<id>."""
    mount = tmp_path / "cg"
    nested = mount / "docker" / "abc123"
    _write(mount / "memory.max", "max")
    _write(mount / "memory.current", "0")
    _write(nested / "memory.max", str(LIMIT))
    _write(nested / "memory.current", str(USAGE))
    proc = tmp_path / "cgroup"
    mountinfo = tmp_path / "mountinfo"
    _write(proc, "0::/docker/abc123\n")
    _write(mountinfo, _v2_mountinfo(mount))
    assert _headroom(proc, mountinfo) == HEADROOM


def test_leaf_unlimited_parent_limit_wins(tmp_path):
    mount = tmp_path / "cg"
    leaf = mount / "pod" / "ctr"
    _write(mount / "memory.max", str(LIMIT))
    _write(mount / "memory.current", str(USAGE))
    _write(leaf / "memory.max", "max")
    _write(leaf / "memory.current", "100")
    proc = tmp_path / "cgroup"
    mountinfo = tmp_path / "mountinfo"
    _write(proc, "0::/pod/ctr\n")
    _write(mountinfo, _v2_mountinfo(mount))
    assert _headroom(proc, mountinfo) == HEADROOM


def test_tightest_ancestor_headroom_wins(tmp_path):
    # Parent has a smaller remaining budget because siblings already consumed it.
    mount = tmp_path / "cg"
    leaf = mount / "pod" / "ctr"
    parent_limit = 4_000_000_000
    parent_usage = 3_500_000_000  # 500 MiB-ish left at the pod
    _write(mount / "memory.max", "max")
    _write(mount / "memory.current", "0")
    _write(mount / "pod" / "memory.max", str(parent_limit))
    _write(mount / "pod" / "memory.current", str(parent_usage))
    _write(leaf / "memory.max", str(LIMIT))
    _write(leaf / "memory.current", str(USAGE))
    proc = tmp_path / "cgroup"
    mountinfo = tmp_path / "mountinfo"
    _write(proc, "0::/pod/ctr\n")
    _write(mountinfo, _v2_mountinfo(mount))
    assert _headroom(proc, mountinfo) == parent_limit - parent_usage


def test_missing_leaf_files_walk_to_parent(tmp_path):
    mount = tmp_path / "cg"
    leaf = mount / "docker" / "abc"
    _write(mount / "docker" / "memory.max", str(LIMIT))
    _write(mount / "docker" / "memory.current", str(USAGE))
    leaf.mkdir(parents=True, exist_ok=True)
    proc = tmp_path / "cgroup"
    mountinfo = tmp_path / "mountinfo"
    _write(proc, "0::/docker/abc\n")
    _write(mountinfo, _v2_mountinfo(mount))
    assert _headroom(proc, mountinfo) == HEADROOM


def test_v1_nested_memory_controller(tmp_path):
    mount = tmp_path / "memory"
    nested = mount / "docker" / "abc123"
    _write(mount / "memory.limit_in_bytes", "9223372036854771712")
    _write(mount / "memory.usage_in_bytes", "0")
    _write(nested / "memory.limit_in_bytes", str(LIMIT))
    _write(nested / "memory.usage_in_bytes", str(USAGE))
    proc = tmp_path / "cgroup"
    mountinfo = tmp_path / "mountinfo"
    _write(proc, "4:memory:/docker/abc123\n")
    _write(mountinfo, _v1_mountinfo(mount))
    assert _headroom(proc, mountinfo) == HEADROOM


def test_path_escape_is_ignored(tmp_path):
    mount = tmp_path / "cg"
    _write(mount / "memory.max", "max")
    _write(mount / "memory.current", "0")
    outside = tmp_path / "outside"
    _write(outside / "memory.max", str(LIMIT))
    _write(outside / "memory.current", str(USAGE))
    proc = tmp_path / "cgroup"
    mountinfo = tmp_path / "mountinfo"
    _write(proc, "0::/../../outside\n")
    _write(mountinfo, _v2_mountinfo(mount))
    assert _headroom(proc, mountinfo) is None


def test_unreadable_proc_files_do_not_raise(tmp_path):
    proc = tmp_path / "cgroup"
    mountinfo = tmp_path / "mountinfo"
    _write(proc, "0::/\n")
    _write(mountinfo, "not a mountinfo line\n")
    proc.chmod(0)
    mountinfo.chmod(0)
    try:
        assert _headroom(proc, mountinfo) is None
    finally:
        proc.chmod(0o644)
        mountinfo.chmod(0o644)


def test_inference_reexports_the_helper():
    assert _exported_headroom is _cgroup_memory_headroom
