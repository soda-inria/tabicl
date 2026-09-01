"""Cgroup-aware CPU memory headroom for inference offload decisions.

``psutil.virtual_memory().available`` is the host's free RAM. The real cap for a
container is ``limit - usage`` on this process's cgroup, which is often not
``/sys/fs/cgroup/memory.max`` (Kubernetes, ``docker run --cgroupns=host``).

Look up the path in ``/proc/self/cgroup``, resolve the controller mount from
``/proc/self/mountinfo``, walk from there to the mount root, and return the
smallest ``limit - usage`` among ancestors that have a real limit.

On hybrid hosts the ``0::`` v2 line can coexist with a v1 ``memory:`` line.
If the v2 walk finds no cap (the memory controller still lives on v1), the
v1 files are used next.

Never raises: a memory estimate is not worth crashing inference over.
"""

from __future__ import annotations

from pathlib import Path

import psutil

_PROC_CGROUP = Path("/proc/self/cgroup")
_PROC_MOUNTINFO = Path("/proc/self/mountinfo")
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
    usage = _read_bytes(directory / usage_name)
    if usage is None:
        # Finite cap, unknown usage: do not pretend the cgroup is empty.
        return 0
    return max(0, limit - usage)


def _min_headroom_along_ancestors(
    start: Path, stop: Path, limit_name: str, usage_name: str, physical: int
) -> int | None:
    """Smallest ``limit - usage`` on ``start`` and its parents up to ``stop``.

    Levels with no file, ``"max"``, a non-positive limit, or a limit at or
    above ``physical`` are skipped. Returns None if no ancestor has a real cap.
    The walk stays under ``stop`` and is bounded so a malformed tree cannot
    loop.
    """
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


def _process_cgroup_relpath(proc_cgroup: Path, *, v2: bool) -> str | None:
    """Relative cgroup path for this process, from a ``cgroup`` proc file.

    For v2, that is the suffix of the ``0::/foo`` line. For v1, the path on
    the ``memory:`` controller line (``4:memory:/foo``). Returns None if the
    file is unreadable or has no matching line.
    """
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


def _unescape_mountinfo(value: str) -> str:
    """Decode octal escapes used in mountinfo path fields (space is ``\\040``)."""
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _parse_mountinfo(
    text: str, *, fstype: str, controller: str | None = None
) -> list[tuple[Path, str]]:
    """Return ``(mountpoint, hierarchy_root)`` pairs from a mountinfo dump.

    ``hierarchy_root`` is field 4: the path inside the cgroup filesystem that
    is mounted. Hierarchy-root mounts (field 4 ``/``) come first so a
    ``/proc/self/cgroup`` path can be joined onto them directly. For cgroup
    v1, ``controller`` must appear in the super-block options (``memory`` in
    ``rw,memory`` or ``rw,cpu,memory``).
    """
    found: list[tuple[Path, str]] = []
    for line in text.splitlines():
        parts = line.split()
        try:
            hyphen = parts.index("-")
        except ValueError:
            continue
        if len(parts) < 5 or hyphen + 1 >= len(parts) or parts[hyphen + 1] != fstype:
            continue
        if controller is not None:
            super_opts = parts[hyphen + 3] if hyphen + 3 < len(parts) else ""
            if controller not in super_opts.split(","):
                continue
        found.append((Path(_unescape_mountinfo(parts[4])), _unescape_mountinfo(parts[3])))
    found.sort(key=lambda item: item[1] != "/")
    return found


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _under_mount(mountpoint: Path, mount_root: str, rel: str) -> Path | None:
    """Directory for ``rel`` on this mount, or None if ``rel`` is outside it.

    ``mount_root`` is mountinfo field 4. A bind-mount of ``/system.slice`` at
    ``mountpoint`` maps ``/system.slice/foo`` to ``mountpoint/foo``.
    """
    rel_parts = [p for p in rel.split("/") if p and p != "."]
    root_parts = [p for p in mount_root.split("/") if p and p != "."]
    # A ".." segment must not walk above the mount. Stay at the mount root.
    if any(p == ".." for p in rel_parts + root_parts):
        return mountpoint
    # Bind-mount of /system.slice cannot serve a process in /user.slice.
    if rel_parts[: len(root_parts)] != root_parts:
        return None
    suffix = rel_parts[len(root_parts) :]
    return mountpoint.joinpath(*suffix) if suffix else mountpoint


def _controller_mounts(
    explicit: Path | None,
    mountinfo_text: str,
    *,
    fstype: str,
    default: Path,
    controller: str | None = None,
) -> list[tuple[Path, str]]:
    """Mounts for a controller: explicit test root, else mountinfo, else ``default``."""
    if explicit is not None:
        return [(explicit, "/")]
    return _parse_mountinfo(mountinfo_text, fstype=fstype, controller=controller) or [
        (default, "/")
    ]


def _headroom_on_mounts(
    rel: str,
    mounts: list[tuple[Path, str]],
    limit_name: str,
    usage_name: str,
    physical: int,
) -> int | None:
    """Tightest ``limit - usage`` among mounts that contain ``rel``."""
    min_headroom: int | None = None
    for mountpoint, mount_root in mounts:
        start = _under_mount(mountpoint, mount_root, rel)
        if start is None:
            continue
        headroom = _min_headroom_along_ancestors(
            start, mountpoint, limit_name, usage_name, physical
        )
        if headroom is not None:
            min_headroom = headroom if min_headroom is None else min(min_headroom, headroom)
    return min_headroom


def _cgroup_memory_headroom(
    *,
    proc_cgroup: Path = _PROC_CGROUP,
    proc_mountinfo: Path = _PROC_MOUNTINFO,
    v2_root: Path | None = None,
    v1_root: Path | None = None,
    physical_bytes: int | None = None,
) -> int | None:
    """Bytes this process may still allocate under its cgroup, or None if unlimited.

    Reads ``proc_cgroup`` (``/proc/self/cgroup`` by default) and joins the
    relative path onto the controller mount from ``proc_mountinfo`` (or
    ``v2_root`` / ``v1_root`` when tests inject a fake sysfs tree). v2 uses
    ``memory.max`` / ``memory.current``; v1 uses ``memory.limit_in_bytes`` /
    ``memory.usage_in_bytes``. Parents are walked to the mount and the
    smallest ``limit - usage`` is returned.

    A ``0::`` v2 line is not proof that the memory controller lives on v2.
    Hybrid hosts list both hierarchies; if the v2 walk finds no cap, the v1
    ``memory:`` entry is inspected next.

    ``"max"``, a v1 ~2**63 unlimited sentinel, and any limit at or above
    physical RAM are treated as no cap at that level. If ``proc_cgroup`` is
    missing, the v2/v1 root files are tried (Docker's private cgroup namespace).

    Never raises: any error falls through to None so callers can keep using
    psutil's host figure.
    """
    try:
        physical = physical_bytes if physical_bytes is not None else psutil.virtual_memory().total
        mountinfo_text = _read_text(proc_mountinfo)
        rel_v2 = _process_cgroup_relpath(proc_cgroup, v2=True)
        rel_v1 = _process_cgroup_relpath(proc_cgroup, v2=False)

        if rel_v2 is not None:
            headroom = _headroom_on_mounts(
                rel_v2,
                _controller_mounts(
                    v2_root, mountinfo_text, fstype="cgroup2", default=_V2_ROOT
                ),
                "memory.max",
                "memory.current",
                physical,
            )
            if headroom is not None:
                return headroom
        if rel_v1 is not None:
            headroom = _headroom_on_mounts(
                rel_v1,
                _controller_mounts(
                    v1_root,
                    mountinfo_text,
                    fstype="cgroup",
                    default=_V1_ROOT,
                    controller="memory",
                ),
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
                physical,
            )
            if headroom is not None:
                return headroom
        if rel_v2 is not None or rel_v1 is not None:
            # A hierarchy line existed but published no cap ("max", v1 sentinel).
            return None
        # /proc/self/cgroup missing: last try, Docker private-namespace root files.
        v2 = _controller_mounts(v2_root, mountinfo_text, fstype="cgroup2", default=_V2_ROOT)[0][0]
        v1 = _controller_mounts(
            v1_root, mountinfo_text, fstype="cgroup", default=_V1_ROOT, controller="memory"
        )[0][0]
        for root, limit_name, usage_name in (
            (v2, "memory.max", "memory.current"),
            (v1, "memory.limit_in_bytes", "memory.usage_in_bytes"),
        ):
            headroom = _headroom_at(root, limit_name, usage_name, physical)
            if headroom is not None:
                return headroom
        return None
    except Exception:
        return None
