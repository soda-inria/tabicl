"""Live Docker checks for cgroup memory detection.

Skipped when ``docker`` is missing or ``docker info`` fails. These cover the
layouts unit tests can only fake: Docker's default private cgroup namespace
(limit at ``/sys/fs/cgroup/memory.max``) and ``--cgroupns=host`` (limit on a
nested path).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = "/src/tests/_cgroup_docker_probe.py"
IMAGE = "tabicl-cgroup-probe:local"
MEMORY_BYTES = 256 * 1024 * 1024


def docker_is_usable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not docker_is_usable(), reason="docker command is not available on this host"
)


@pytest.fixture(scope="module")
def probe_image() -> str:
    dockerfile = "FROM python:3.12-slim\nRUN pip install --no-cache-dir psutil\n"
    with tempfile.TemporaryDirectory() as tmp:
        build = subprocess.run(
            ["docker", "build", "-t", IMAGE, "-f", "-", tmp],
            input=dockerfile,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    if build.returncode != 0:
        pytest.skip(f"docker build failed: {build.stderr or build.stdout}")
    return IMAGE


def _run_container(image: str, extra_args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "-v",
            f"{REPO_ROOT}:/src:ro",
            *extra_args,
            image,
            "python",
            PROBE,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _parse_probe(result: subprocess.CompletedProcess[str]) -> dict:
    if result.returncode != 0:
        pytest.fail(
            f"docker run failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError as exc:
        pytest.fail(f"probe did not print JSON: {result.stdout!r} ({exc})")


def _assert_capped_at(payload: dict, cap_bytes: int) -> None:
    headroom = payload["headroom_bytes"]
    assert headroom is not None, payload
    assert 0 < headroom <= cap_bytes, payload
    # Leave room for the Python runtime inside the cap.
    assert headroom >= cap_bytes * 0.25, payload
    psutil_avail = payload["psutil_available_bytes"]
    # The bug: psutil reports host RAM, far above the container cap.
    assert psutil_avail > cap_bytes * 2, payload
    assert payload["effective_available_bytes"] == min(psutil_avail, headroom), payload


def test_docker_private_cgroupns_uses_memory_max(probe_image: str):
    payload = _parse_probe(
        _run_container(
            probe_image,
            [
                "--memory",
                "256m",
                "--memory-swap",
                "256m",
                "--cgroupns",
                "private",
            ],
        )
    )
    _assert_capped_at(payload, MEMORY_BYTES)
    cgroup = payload["proc_self_cgroup"]
    assert cgroup.endswith(":/") or cgroup.endswith("::/"), payload
    root = payload["root_memory_max"]
    assert root not in (None, "max"), payload
    assert int(root.split()[0]) == MEMORY_BYTES, payload


def test_docker_host_cgroupns_uses_nested_path(probe_image: str):
    result = _run_container(
        probe_image,
        [
            "--memory",
            "256m",
            "--memory-swap",
            "256m",
            "--cgroupns",
            "host",
        ],
    )
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode != 0 and "cgroupns" in combined:
        pytest.skip("docker rejected --cgroupns=host")
    payload = _parse_probe(result)
    _assert_capped_at(payload, MEMORY_BYTES)
    rel = payload["proc_self_cgroup"].split(":")[-1]
    assert rel not in ("", "/"), payload
    root = payload["root_memory_max"]
    assert root in (None, "max") or (
        root is not None and int(root.split()[0]) > MEMORY_BYTES * 2
    ), payload
    self_max = payload["self_memory_max"]
    assert self_max not in (None, "max"), payload
    assert int(self_max.split()[0]) == MEMORY_BYTES, payload


def test_docker_unlimited_is_not_a_cgroup_cap(probe_image: str):
    payload = _parse_probe(_run_container(probe_image, []))
    if payload["headroom_bytes"] is None:
        assert payload["effective_available_bytes"] == payload["psutil_available_bytes"], payload
        return
    # Some nested hosts still cap the docker daemon. Accept that only if it is
    # clearly in the host-RAM ballpark, not a 256MiB test limit.
    assert payload["headroom_bytes"] > MEMORY_BYTES * 2, payload
