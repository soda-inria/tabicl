"""Cgroup-aware CPU memory headroom for inference offload decisions.

``psutil.virtual_memory().available`` is the host's free RAM. The real cap for a
container is ``limit - usage`` on this process's cgroup, which is often not
``/sys/fs/cgroup/memory.max`` (Kubernetes, ``docker run --cgroupns=host``).

Look up the path in ``/proc/self/cgroup``, walk from there to the cgroup root,
and return the smallest ``limit - usage`` among ancestors that have a real limit.

Never raises: a memory estimate is not worth crashing inference over.
"""

from __future__ import annotations

from pathlib import Path

import psutil

_PROC_CGROUP = Path("/proc/self/cgroup")
_V2_ROOT = Path("/sys/fs/cgroup")
_V1_ROOT = Path("/sys/fs/cgroup/memory")


def _parse_bytes(raw: str) -> int | None:
    token = raw.split()[0] if raw.split() else ""
    if token == "max":
        return None
    try:
        return int(token)
    except ValueError:
        return None


def _read_bytes(path: Path) -> int | None:
    try:
        return _parse_bytes(path.read_text())
    except OSError:
        return None


def _headroom_at(directory: Path, limit_name: str, usage_name: str, physical: int) -> int | None:
    limit = _read_bytes(directory / limit_name)
    if limit is None or limit <= 0 or limit >= physical:
        return None
    usage = _read_bytes(directory / usage_name) or 0
    return max(0, limit - usage)


def _join(root: Path, rel: str) -> Path:
    parts = [p for p in rel.split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        return root
    return root.joinpath(*parts) if parts else root


def _walk(start: Path, stop: Path, limit_name: str, usage_name: str, physical: int) -> int | None:
    best: int | None = None
    path = start
    for _ in range(64):
        try:
            path.relative_to(stop)
        except ValueError:
            break
        headroom = _headroom_at(path, limit_name, usage_name, physical)
        if headroom is not None:
            best = headroom if best is None else min(best, headroom)
        if path == stop or path.parent == path:
            break
        path = path.parent
    return best


def _relpath(proc_cgroup: Path, *, v2: bool) -> str | None:
    try:
        text = proc_cgroup.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if v2:
            if line.startswith("0::"):
                return line[3:]
            continue
        fields = line.split(":", 2)
        if len(fields) == 3 and "memory" in fields[1].split(","):
            return fields[2]
    return None


def _cgroup_memory_headroom(
    *,
    proc_cgroup: Path = _PROC_CGROUP,
    v2_root: Path = _V2_ROOT,
    v1_root: Path = _V1_ROOT,
    physical_bytes: int | None = None,
) -> int | None:
    """Bytes this process may still allocate under its cgroup, or None if unlimited.

    v2: ``memory.max`` / ``memory.current``. v1: ``memory.limit_in_bytes`` /
    ``memory.usage_in_bytes``. ``"max"``, v1's ~2**63 sentinel, and any limit at
    or above physical RAM are treated as unlimited.

    Optional path arguments are for tests (a fake sysfs tree).
    """
    try:
        physical = physical_bytes if physical_bytes is not None else psutil.virtual_memory().total
        rel = _relpath(proc_cgroup, v2=True)
        if rel is not None:
            return _walk(_join(v2_root, rel), v2_root, "memory.max", "memory.current", physical)
        rel = _relpath(proc_cgroup, v2=False)
        if rel is not None:
            return _walk(
                _join(v1_root, rel), v1_root, "memory.limit_in_bytes", "memory.usage_in_bytes", physical
            )
        # /proc/self/cgroup missing: last try, Docker private-namespace root files.
        for root, limit_name, usage_name in (
            (v2_root, "memory.max", "memory.current"),
            (v1_root, "memory.limit_in_bytes", "memory.usage_in_bytes"),
        ):
            headroom = _headroom_at(root, limit_name, usage_name, physical)
            if headroom is not None:
                return headroom
        return None
    except Exception:
        return None
