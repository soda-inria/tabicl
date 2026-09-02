#!/usr/bin/env python3
"""Print JSON cgroup-headroom probe results. Used inside Docker by integration tests.

Loads ``_cgroup_memory.py`` by file path so the container does not need torch.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import psutil


def _load_cgroup_module():
    # Repo is mounted at /src (see test_cgroup_memory_docker.py).
    candidates = [
        pathlib.Path("/src/src/tabicl/_model/_cgroup_memory.py"),
        pathlib.Path(__file__).resolve().parents[1] / "src/tabicl/_model/_cgroup_memory.py",
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("tabicl_cgroup_memory", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("could not locate src/tabicl/_model/_cgroup_memory.py")


def main() -> None:
    mod = _load_cgroup_module()
    vm = psutil.virtual_memory()
    headroom = mod._cgroup_memory_headroom()
    available = vm.available
    if headroom is not None:
        available = min(available, headroom)
    payload = {
        "headroom_bytes": headroom,
        "psutil_available_bytes": vm.available,
        "psutil_total_bytes": vm.total,
        "effective_available_bytes": available,
        "proc_self_cgroup": pathlib.Path("/proc/self/cgroup").read_text().strip(),
        "root_memory_max": None,
        "self_memory_max": None,
    }
    root_max = pathlib.Path("/sys/fs/cgroup/memory.max")
    if root_max.is_file():
        payload["root_memory_max"] = root_max.read_text().strip()
    rel = payload["proc_self_cgroup"].split(":")[-1]
    self_max = pathlib.Path("/sys/fs/cgroup") / rel.lstrip("/") / "memory.max"
    if self_max.is_file():
        payload["self_memory_max"] = self_max.read_text().strip()
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
