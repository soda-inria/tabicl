"""Cgroup-aware CPU memory headroom for inference offload decisions.

``psutil.virtual_memory().available`` reports the host's free memory and is blind
to cgroup limits. This module finds the current process's cgroup (v2 or v1), walks
ancestors, and returns the tightest ``limit - usage`` headroom.

Hardcoded ``/sys/fs/cgroup/memory.max`` is only correct when the runtime remounts
the container cgroup at the cgroup root (Docker's default private cgroup
namespace). Nested layouts -- ``docker run --cgroupns=host``, Kubernetes,
systemd slices -- store the limit on an ancestor. Usage is always read on the
same directory as the limit so sibling processes sharing a parent budget are
not ignored.

Never raises: a memory estimate is not worth crashing inference over.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import psutil

_PROC_CGROUP = pathlib.Path("/proc/self/cgroup")
_PROC_MOUNTINFO = pathlib.Path("/proc/self/mountinfo")
_DEFAULT_V2_MOUNT = pathlib.Path("/sys/fs/cgroup")
_DEFAULT_V1_MOUNT = pathlib.Path("/sys/fs/cgroup/memory")

_V2_LIMIT = "memory.max"
_V2_USAGE = "memory.current"
_V1_LIMIT = "memory.limit_in_bytes"
_V1_USAGE = "memory.usage_in_bytes"


def _unescape_mountinfo(value: str) -> str:
    """Decode octal escapes used in ``/proc/self/mountinfo`` mount points."""
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _parse_cgroup_bytes(raw: str) -> int | None:
    token = raw.split()[0] if raw.split() else ""
    if token == "max":
        return None
    try:
        return int(token)
    except ValueError:
        return None


def _read_cgroup_bytes(path: pathlib.Path) -> int | None:
    try:
        return _parse_cgroup_bytes(path.read_text())
    except (OSError, IndexError):
        return None


def _headroom_at(
    directory: pathlib.Path,
    limit_name: str,
    usage_name: str,
    physical_bytes: int,
) -> int | None:
    limit = _read_cgroup_bytes(directory / limit_name)
    if limit is None or limit <= 0 or limit >= physical_bytes:
        return None
    usage = _read_cgroup_bytes(directory / usage_name)
    if usage is None:
        usage = 0
    return max(0, limit - usage)


def _safe_cgroup_dir(mount: pathlib.Path, rel: str) -> pathlib.Path:
    """Join ``rel`` onto ``mount``, refusing paths that escape the mount."""
    rel = rel.strip() or "/"
    if rel == "/":
        return mount
    candidate = mount / rel.lstrip("/")
    try:
        resolved = candidate.resolve()
        mount_resolved = mount.resolve()
    except OSError:
        return mount
    if resolved == mount_resolved or mount_resolved in resolved.parents:
        return resolved
    return mount


def _min_headroom_along_ancestors(
    start: pathlib.Path,
    mount: pathlib.Path,
    limit_name: str,
    usage_name: str,
    physical_bytes: int,
) -> int | None:
    try:
        start = start.resolve()
        mount = mount.resolve()
    except OSError:
        return None
    best: int | None = None
    path = start
    while True:
        headroom = _headroom_at(path, limit_name, usage_name, physical_bytes)
        if headroom is not None:
            best = headroom if best is None else min(best, headroom)
        if path == mount:
            break
        parent = path.parent
        if parent == path:
            break
        if mount not in parent.parents and parent != mount:
            break
        path = parent
    return best


def _cgroup2_mounts(mountinfo_text: str) -> list[pathlib.Path]:
    mounts: list[pathlib.Path] = []
    for line in mountinfo_text.splitlines():
        parts = line.split()
        try:
            hyphen = parts.index("-")
        except ValueError:
            continue
        if hyphen + 1 >= len(parts) or parts[hyphen + 1] != "cgroup2":
            continue
        if len(parts) < 5:
            continue
        mounts.append(pathlib.Path(_unescape_mountinfo(parts[4])))
    return mounts


def _cgroup1_memory_mounts(mountinfo_text: str) -> list[pathlib.Path]:
    mounts: list[pathlib.Path] = []
    for line in mountinfo_text.splitlines():
        parts = line.split()
        try:
            hyphen = parts.index("-")
        except ValueError:
            continue
        if hyphen + 1 >= len(parts) or parts[hyphen + 1] != "cgroup":
            continue
        super_opts = parts[hyphen + 3] if hyphen + 3 < len(parts) else ""
        if "memory" not in super_opts.split(","):
            continue
        if len(parts) < 5:
            continue
        mounts.append(pathlib.Path(_unescape_mountinfo(parts[4])))
    return mounts


def _cgroup_v2_relpath(cgroup_text: str) -> str | None:
    for line in cgroup_text.splitlines():
        if line.startswith("0::"):
            return line[3:] or "/"
    return None


def _cgroup_v1_memory_relpath(cgroup_text: str) -> str | None:
    for line in cgroup_text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        controllers = fields[1].split(",")
        if "memory" in controllers:
            return fields[2] or "/"
    return None


def _read_text(path: pathlib.Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def _walk_v2(
    cgroup_text: str | None,
    mountinfo_text: str | None,
    physical_bytes: int,
) -> tuple[bool, int | None]:
    """Return ``(v2_detected, headroom)``. Headroom is None if unlimited or unread."""
    mounts = _cgroup2_mounts(mountinfo_text) if mountinfo_text is not None else []
    rel = _cgroup_v2_relpath(cgroup_text) if cgroup_text is not None else None
    if not mounts and rel is None:
        return False, None
    mount = mounts[0] if mounts else _DEFAULT_V2_MOUNT
    start = _safe_cgroup_dir(mount, rel or "/")
    return True, _min_headroom_along_ancestors(
        start, mount, _V2_LIMIT, _V2_USAGE, physical_bytes
    )


def _walk_v1(
    cgroup_text: str | None,
    mountinfo_text: str | None,
    physical_bytes: int,
) -> tuple[bool, int | None]:
    """Return ``(v1_detected, headroom)``. Headroom is None if unlimited or unread."""
    mounts = _cgroup1_memory_mounts(mountinfo_text) if mountinfo_text is not None else []
    rel = _cgroup_v1_memory_relpath(cgroup_text) if cgroup_text is not None else None
    if not mounts and rel is None:
        return False, None
    if not mounts:
        mounts = [_DEFAULT_V1_MOUNT]
    best: int | None = None
    for mount in mounts:
        start = _safe_cgroup_dir(mount, rel or "/")
        headroom = _min_headroom_along_ancestors(
            start, mount, _V1_LIMIT, _V1_USAGE, physical_bytes
        )
        if headroom is not None:
            best = headroom if best is None else min(best, headroom)
    return True, best


def _fallback_root_headroom(physical_bytes: int) -> int | None:
    """Last resort: the root files Docker's private cgroup namespace exposes."""
    v2 = _headroom_at(_DEFAULT_V2_MOUNT, _V2_LIMIT, _V2_USAGE, physical_bytes)
    if v2 is not None:
        return v2
    return _headroom_at(_DEFAULT_V1_MOUNT, _V1_LIMIT, _V1_USAGE, physical_bytes)


def _cgroup_memory_headroom(
    *,
    proc_cgroup: Optional[pathlib.Path] = None,
    proc_mountinfo: Optional[pathlib.Path] = None,
    physical_bytes: Optional[int] = None,
) -> int | None:
    """Bytes this process may still allocate under its cgroup, or None if unlimited.

    Handles cgroup v2 (``memory.max`` / ``memory.current``) and v1
    (``memory.limit_in_bytes`` / ``memory.usage_in_bytes``). Walks from the
    process cgroup to the controller mount and returns the minimum headroom
    among ancestors that publish a real limit.

    Returns None when there is no cgroup, when every limit is literally
    ``"max"``, or when a limit is the sentinel huge value v1 uses to mean
    unlimited -- in all of which cases psutil's figure stands.

    ``proc_cgroup``, ``proc_mountinfo``, and ``physical_bytes`` override
    ``/proc/self/cgroup``, ``/proc/self/mountinfo``, and host RAM; tests use
    them to inject a fake hierarchy.

    Never raises: a memory estimate is not worth crashing an inference call
    over, and every failure path here simply falls back to the previous
    behaviour.
    """
    try:
        physical = (
            int(physical_bytes)
            if physical_bytes is not None
            else psutil.virtual_memory().total
        )
        cgroup_text = _read_text(proc_cgroup or _PROC_CGROUP)
        mountinfo_text = _read_text(proc_mountinfo or _PROC_MOUNTINFO)
        injected = proc_cgroup is not None or proc_mountinfo is not None
        v2_detected, v2_headroom = _walk_v2(cgroup_text, mountinfo_text, physical)
        if v2_detected:
            return v2_headroom
        v1_detected, v1_headroom = _walk_v1(cgroup_text, mountinfo_text, physical)
        if v1_detected:
            return v1_headroom
        if injected:
            return None
        return _fallback_root_headroom(physical)
    except Exception:
        return None
