#!/usr/bin/env python
"""CI/local helper to diagnose PyTorch MPS vs CPU behaviour.

Collects host/hardware info and runs numerical sanity checks that compare
CPU and MPS for kernels relevant to TabICL (matmul, SDPA, optional small
TabICLRegressor fit/predict). Intended to be run in CI on macOS runners
when MPS device-parity tests fail.

Exit code is 0 even when MPS checks fail numerically — the goal is to
surface diagnostics in the log. Pass ``--strict`` to fail the process on
mismatches.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    sys.stdout.flush()


def _run_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=30)
        return out.strip()
    except Exception as exc:  # noqa: BLE001 - diagnostics must not crash
        return f"<failed: {type(exc).__name__}: {exc}>"


def inspect_hardware() -> None:
    _section("Hardware / OS")
    print(f"python:           {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"executable:       {sys.executable}")
    print(f"platform:         {platform.platform()}")
    print(f"machine:          {platform.machine()}")
    print(f"processor:        {platform.processor()!r}")
    print(f"cpu_count:        {os.cpu_count()}")

    if sys.platform == "darwin":
        print(f"sysctl.cpu:       {_run_cmd(['sysctl', '-n', 'machdep.cpu.brand_string'])}")
        print(f"sysctl.memsize:   {_run_cmd(['sysctl', '-n', 'hw.memsize'])}")
        print(f"sysctl.ncpu:     {_run_cmd(['sysctl', '-n', 'hw.ncpu'])}")
        print(f"sysctl.perflevel0: {_run_cmd(['sysctl', '-n', 'hw.perflevel0.logicalcpu'])}")
        print(f"sysctl.perflevel1: {_run_cmd(['sysctl', '-n', 'hw.perflevel1.logicalcpu'])}")
        # Compact system profiler snippet (chip / memory).
        sp = _run_cmd(["system_profiler", "SPHardwareDataType", "-detailLevel", "mini"])
        print("system_profiler SPHardwareDataType:")
        for line in sp.splitlines():
            print(f"  {line}")

    try:
        import psutil

        vm = psutil.virtual_memory()
        print(
            f"psutil.memory:    total={vm.total / 1024**3:.2f}GB "
            f"available={vm.available / 1024**3:.2f}GB "
            f"used={vm.used / 1024**3:.2f}GB "
            f"percent={vm.percent}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"psutil.memory:    <unavailable: {exc}>")


def inspect_torch() -> bool:
    _section("PyTorch / MPS backend")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"torch import failed: {exc}")
        return False

    print(f"torch.__version__:     {torch.__version__}")
    print(f"torch.version.cuda:    {getattr(torch.version, 'cuda', None)}")
    print(f"torch file:            {torch.__file__}")
    print(f"cuda.is_available:     {torch.cuda.is_available()}")
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and callable(getattr(xpu, "is_available", None)):
        print(f"xpu.is_available:      {xpu.is_available()}")
    print(f"mps.is_built:          {torch.backends.mps.is_built()}")
    print(f"mps.is_available:      {torch.backends.mps.is_available()}")

    if torch.backends.mps.is_available():
        try:
            print(f"mps.recommended_max:   {torch.mps.recommended_max_memory() / 1024**3:.2f}GB")
            print(f"mps.current_allocated: {torch.mps.current_allocated_memory() / 1024**3:.4f}GB")
            print(f"mps.driver_allocated:  {torch.mps.driver_allocated_memory() / 1024**3:.4f}GB")
        except Exception as exc:  # noqa: BLE001
            print(f"mps memory APIs:       <failed: {exc}>")

    env_keys = [
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    print("relevant env:")
    for key in env_keys:
        print(f"  {key}={os.environ.get(key)!r}")

    return torch.backends.mps.is_available()


@dataclass
class CompareResult:
    name: str
    ok: bool
    detail: str


def _sync(device: str):
    import torch

    if device == "mps":
        torch.mps.synchronize()


def compare_matmul(dtype_name: str = "float32") -> CompareResult:
    import torch

    dtype = getattr(torch, dtype_name)
    gen = torch.Generator(device="cpu").manual_seed(0)
    a_cpu = torch.randn(256, 256, generator=gen, dtype=dtype)
    b_cpu = torch.randn(256, 256, generator=gen, dtype=dtype)
    out_cpu = a_cpu @ b_cpu

    a_mps = a_cpu.to("mps")
    b_mps = b_cpu.to("mps")
    _sync("mps")
    out_mps = (a_mps @ b_mps).cpu()
    _sync("mps")

    max_abs = (out_cpu - out_mps).abs().max().item()
    max_rel = ((out_cpu - out_mps).abs() / out_cpu.abs().clamp_min(1e-6)).max().item()
    # float16 needs looser tol; float32 should be very close
    tol = 5e-2 if dtype == torch.float16 else 1e-4
    ok = max_abs < tol
    return CompareResult(
        name=f"matmul[{dtype_name}]",
        ok=ok,
        detail=f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} tol={tol:.1e}",
    )


def compare_sdpa(dtype_name: str = "float32") -> CompareResult:
    import torch
    import torch.nn.functional as F

    dtype = getattr(torch, dtype_name)
    gen = torch.Generator(device="cpu").manual_seed(0)
    # Small attention: B=2, H=4, S=64, D=16
    q = torch.randn(2, 4, 64, 16, generator=gen, dtype=dtype)
    k = torch.randn(2, 4, 64, 16, generator=gen, dtype=dtype)
    v = torch.randn(2, 4, 64, 16, generator=gen, dtype=dtype)
    out_cpu = F.scaled_dot_product_attention(q, k, v)

    q_m, k_m, v_m = q.to("mps"), k.to("mps"), v.to("mps")
    _sync("mps")
    out_mps = F.scaled_dot_product_attention(q_m, k_m, v_m).cpu()
    _sync("mps")

    max_abs = (out_cpu - out_mps).abs().max().item()
    tol = 5e-2 if dtype == torch.float16 else 1e-3
    ok = max_abs < tol
    return CompareResult(
        name=f"sdpa[{dtype_name}]",
        ok=ok,
        detail=f"max_abs={max_abs:.3e} tol={tol:.1e}",
    )


def compare_autocast_matmul() -> CompareResult:
    import torch

    gen = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randn(128, 128, generator=gen, dtype=torch.float32)
    b = torch.randn(128, 128, generator=gen, dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        # CPU autocast may no-op or use bf16 depending on build; still useful signal.
        out_cpu = (a @ b).float()

    a_m, b_m = a.to("mps"), b.to("mps")
    _sync("mps")
    with torch.autocast(device_type="mps", dtype=torch.float16):
        out_mps = (a_m @ b_m).float().cpu()
    _sync("mps")

    max_abs = (out_cpu - out_mps).abs().max().item()
    # Autocast paths differ by design; only flag catastrophic divergence.
    ok = max_abs < 5.0
    return CompareResult(
        name="autocast_matmul[cpu_bf16? vs mps_fp16]",
        ok=ok,
        detail=f"max_abs={max_abs:.3e} (loose check; dtype paths differ)",
    )


def compare_tabicl_tiny() -> CompareResult:
    """Minimal TabICLRegressor CPU vs MPS parity (mirrors CI failure mode)."""
    try:
        from sklearn.datasets import make_friedman1
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split

        from tabicl import TabICLRegressor
    except Exception as exc:  # noqa: BLE001
        return CompareResult("tabicl_tiny", False, f"import failed: {exc}")

    X, y = make_friedman1(n_samples=120, n_features=16, noise=1.0, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    scores = {}
    preds = {}
    for device in ("cpu", "mps"):
        reg = TabICLRegressor(
            n_estimators=2,
            device=device,
            use_amp=False,
            use_fa3=False,
            random_state=0,
            verbose=False,
        )
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)
        preds[device] = pred
        scores[device] = float(r2_score(y_test, pred))

    import numpy as np

    d_r2 = abs(scores["cpu"] - scores["mps"])
    d_pred = float(np.max(np.abs(preds["cpu"] - preds["mps"])))
    ok = d_r2 < 1e-3 and d_pred < 1e-2
    return CompareResult(
        name="tabicl_regressor[amp=off,n_estimators=2]",
        ok=ok,
        detail=(
            f"R2_cpu={scores['cpu']:.4f} R2_mps={scores['mps']:.4f} "
            f"|dR2|={d_r2:.3e} max|pred|={d_pred:.3e}"
        ),
    )


def run_mps_checks(*, include_tabicl: bool) -> list[CompareResult]:
    _section("MPS numerical sanity checks (vs CPU)")
    results: list[CompareResult] = []
    checks = [
        ("matmul float32", lambda: compare_matmul("float32")),
        ("matmul float16", lambda: compare_matmul("float16")),
        ("sdpa float32", lambda: compare_sdpa("float32")),
        ("sdpa float16", lambda: compare_sdpa("float16")),
        ("autocast matmul", compare_autocast_matmul),
    ]
    if include_tabicl:
        checks.append(("tabicl tiny", compare_tabicl_tiny))

    for label, fn in checks:
        print(f"\n-- {label} --")
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001
            result = CompareResult(label, False, f"EXCEPTION {type(exc).__name__}: {exc}")
            traceback.print_exc()
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
        results.append(result)
        sys.stdout.flush()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any MPS numerical check fails.",
    )
    parser.add_argument(
        "--skip-tabicl",
        action="store_true",
        help="Skip the small TabICLRegressor CPU/MPS parity probe.",
    )
    args = parser.parse_args(argv)

    inspect_hardware()
    mps_ok = inspect_torch()

    results: list[CompareResult] = []
    if not mps_ok:
        _section("MPS numerical sanity checks")
        print("MPS is not available — skipping CPU/MPS comparisons.")
    else:
        results = run_mps_checks(include_tabicl=not args.skip_tabicl)

    _section("Summary")
    if not mps_ok:
        print("MPS unavailable on this host.")
        return 0

    n_fail = sum(not r.ok for r in results)
    for r in results:
        print(f"{'OK  ' if r.ok else 'FAIL'}  {r.name}: {r.detail}")
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed, {n_fail} failed.")

    if args.strict and n_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
