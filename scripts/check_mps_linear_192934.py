#!/usr/bin/env python
"""Check pytorch/pytorch#192934: F.linear+bias on 3D input vs CPU on MPS.

Reproducer from https://github.com/pytorch/pytorch/issues/192934 (VirtualMac /
Apple M1 on GitHub Actions). Related to the large-batch fix in #189495 / #189496:
this script asks whether a post-fix nightly also clears the small-shape
VirtualMac failure.

Exit codes:
  0  MPS unavailable (skipped) or all checks within tolerance
  1  at least one check exceeded tolerance
"""

from __future__ import annotations

import platform
import subprocess
import sys


def _sysctl(name: str) -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except Exception:  # noqa: BLE001
        return "<unavailable>"


def main() -> int:
    import torch
    import torch.nn.functional as F

    print(f"python:  {sys.version.split()[0]}")
    print(f"platform:{platform.platform()}")
    print(f"torch:   {torch.__version__}")
    if sys.platform == "darwin":
        print(f"cpu:     {_sysctl('machdep.cpu.brand_string')}")
        print(f"model:   {_sysctl('hw.model')}")
    print(f"mps.built:{torch.backends.mps.is_built()}")
    print(f"mps.avail:{torch.backends.mps.is_available()}")

    if not torch.backends.mps.is_available():
        print("SKIP: MPS not available on this runner")
        return 0

    torch.manual_seed(0)
    x = torch.randn(2, 64, 128)
    w = torch.randn(384, 128)
    b = torch.randn(384)

    y_cpu = F.linear(x, w, b)
    y_mps = F.linear(x.to("mps"), w.to("mps"), b.to("mps")).cpu()
    y_mm = (x.to("mps") @ w.to("mps").T + b.to("mps")).cpu()
    y_nb = F.linear(x.to("mps"), w.to("mps"), None).cpu()
    torch.mps.synchronize()

    # Same seed tensors for the bias-free CPU reference.
    y_nb_cpu = x @ w.T

    checks = [
        ("F.linear+bias 3D (issue #192934)", (y_cpu - y_mps).abs().max().item(), 1e-2),
        ("matmul+bias 3D (control)", (y_cpu - y_mm).abs().max().item(), 1e-2),
        ("F.linear bias=None 3D (control)", (y_nb_cpu - y_nb).abs().max().item(), 1e-2),
    ]

    failed = False
    print()
    for name, diff, tol in checks:
        status = "OK" if diff < tol else "FAIL"
        if status == "FAIL":
            failed = True
        print(f"{status:4}  {name}: max_abs={diff:.6g} (tol={tol:g})")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
