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
from tabicl._model.inference import InferenceManager

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


def test_hybrid_v2_line_without_memory_controller_uses_v1(tmp_path):
    """A ``0::`` line does not mean the memory controller lives on v2.

    Hybrid hosts list both a v2 hierarchy and a v1 ``memory:`` controller. If
    v2 has no ``memory.max`` (controller still on v1), headroom must come from
    the v1 files rather than falling back to host-wide psutil.
    """
    v2 = tmp_path / "unified"
    v1 = tmp_path / "memory"
    (v2 / "user.slice").mkdir(parents=True)
    nested = v1 / "docker" / "abc"
    _write(nested / "memory.limit_in_bytes", str(LIMIT))
    _write(nested / "memory.usage_in_bytes", str(USAGE))
    proc = tmp_path / "proc_cgroup"
    _write(proc, "0::/user.slice\n4:memory:/docker/abc\n")
    assert (
        _cgroup_memory_headroom(
            proc_cgroup=proc, v2_root=v2, v1_root=v1, physical_bytes=PHYSICAL
        )
        == HEADROOM
    )


def test_missing_usage_does_not_assume_empty_cgroup(tmp_path):
    """A readable limit with unreadable usage must not look like a free cgroup."""
    cg = tmp_path / "cg"
    _write(cg / "memory.max", str(LIMIT))
    assert _headroom(tmp_path, "/") == 0


def test_v2_mount_is_read_from_mountinfo(tmp_path):
    """cgroup v2 is not always at ``/sys/fs/cgroup`` (systemd hybrid uses unified/)."""
    unified = tmp_path / "sys" / "fs" / "cgroup" / "unified"
    _write(unified / "memory.max", str(LIMIT))
    _write(unified / "memory.current", str(USAGE))
    proc = tmp_path / "cgroup"
    _write(proc, "0::/\n")
    mountinfo = tmp_path / "mountinfo"
    _write(mountinfo, f"36 35 0:0 / {unified} rw - cgroup2 cgroup2 rw,nsdelegate\n")
    assert (
        _cgroup_memory_headroom(
            proc_cgroup=proc, proc_mountinfo=mountinfo, physical_bytes=PHYSICAL
        )
        == HEADROOM
    )


def test_v1_memory_mount_is_read_from_mountinfo(tmp_path):
    """v1 memory may be co-mounted with cpu, not at ``/sys/fs/cgroup/memory``."""
    comount = tmp_path / "sys" / "fs" / "cgroup" / "cpu,memory"
    nested = comount / "docker" / "abc"
    _write(nested / "memory.limit_in_bytes", str(LIMIT))
    _write(nested / "memory.usage_in_bytes", str(USAGE))
    proc = tmp_path / "cgroup"
    _write(proc, "4:cpu,memory:/docker/abc\n")
    mountinfo = tmp_path / "mountinfo"
    _write(mountinfo, f"32 26 0:0 / {comount} rw - cgroup cgroup rw,cpu,memory\n")
    assert (
        _cgroup_memory_headroom(
            proc_cgroup=proc, proc_mountinfo=mountinfo, physical_bytes=PHYSICAL
        )
        == HEADROOM
    )


def test_mountinfo_bind_mount_strips_hierarchy_root(tmp_path):
    """A subtree bind-mount's field 4 is not ``/``; strip that prefix from the path."""
    mount = tmp_path / "sys" / "fs" / "cgroup" / "system.slice"
    nested = mount / "docker-abc.scope"
    _write(nested / "memory.max", str(LIMIT))
    _write(nested / "memory.current", str(USAGE))
    proc = tmp_path / "cgroup"
    _write(proc, "0::/system.slice/docker-abc.scope\n")
    mountinfo = tmp_path / "mountinfo"
    _write(
        mountinfo,
        f"36 35 0:0 /system.slice {mount} rw - cgroup2 cgroup2 rw,nsdelegate\n",
    )
    assert (
        _cgroup_memory_headroom(
            proc_cgroup=proc, proc_mountinfo=mountinfo, physical_bytes=PHYSICAL
        )
        == HEADROOM
    )


def test_mountinfo_octal_escapes_in_mountpoint(tmp_path):
    """Kernel mountinfo encodes spaces as ``\\040``; decode before joining."""
    mount = tmp_path / "cgroup dir"
    _write(mount / "memory.max", str(LIMIT))
    _write(mount / "memory.current", str(USAGE))
    proc = tmp_path / "cgroup"
    _write(proc, "0::/\n")
    escaped = str(mount).replace(" ", "\\040")
    mountinfo = tmp_path / "mountinfo"
    _write(mountinfo, f"36 35 0:0 / {escaped} rw - cgroup2 cgroup2 rw\n")
    assert (
        _cgroup_memory_headroom(
            proc_cgroup=proc, proc_mountinfo=mountinfo, physical_bytes=PHYSICAL
        )
        == HEADROOM
    )
