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


def _mps_mem(tag: str) -> None:
    import torch

    try:
        print(
            f"  mps_mem[{tag}]: "
            f"alloc={torch.mps.current_allocated_memory() / 1024**3:.3f}GB "
            f"driver={torch.mps.driver_allocated_memory() / 1024**3:.3f}GB "
            f"recommended_max={torch.mps.recommended_max_memory() / 1024**3:.2f}GB"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  mps_mem[{tag}]: <failed: {exc}>")


def _pred_stats(name: str, pred) -> str:
    import numpy as np

    p = np.asarray(pred, dtype=np.float64)
    return (
        f"{name}: shape={p.shape} mean={p.mean():.4f} std={p.std():.4f} "
        f"min={p.min():.4f} max={p.max():.4f} "
        f"nan={np.isnan(p).sum()} inf={np.isinf(p).sum()}"
    )


def compare_layernorm() -> CompareResult:
    import torch
    import torch.nn as nn

    gen = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(8, 128, 256, generator=gen, dtype=torch.float32)
    ln = nn.LayerNorm(256)
    out_cpu = ln(x)
    ln_m = ln.to("mps")
    _sync("mps")
    out_mps = ln_m(x.to("mps")).cpu()
    _sync("mps")
    max_abs = (out_cpu - out_mps).abs().max().item()
    ok = max_abs < 1e-4
    return CompareResult("layernorm[float32]", ok, f"max_abs={max_abs:.3e}")


def compare_sdpa_large() -> CompareResult:
    """SDPA with shapes closer to TabICL attention (longer sequence)."""
    import torch
    import torch.nn.functional as F

    gen = torch.Generator(device="cpu").manual_seed(0)
    # B=1, H=8, S=512, D=64 — stress virtualized MPS memory a bit more
    q = torch.randn(1, 8, 512, 64, generator=gen, dtype=torch.float32)
    k = torch.randn(1, 8, 512, 64, generator=gen, dtype=torch.float32)
    v = torch.randn(1, 8, 512, 64, generator=gen, dtype=torch.float32)
    out_cpu = F.scaled_dot_product_attention(q, k, v)
    _sync("mps")
    out_mps = F.scaled_dot_product_attention(q.to("mps"), k.to("mps"), v.to("mps")).cpu()
    _sync("mps")
    max_abs = (out_cpu - out_mps).abs().max().item()
    ok = max_abs < 1e-3
    return CompareResult("sdpa_large[1,8,512,64]", ok, f"max_abs={max_abs:.3e}")


def compare_weight_transfer() -> CompareResult:
    """Load TabICL regressor checkpoint on CPU, move to MPS, compare params."""
    try:
        from tabicl import TabICLRegressor
    except Exception as exc:  # noqa: BLE001
        return CompareResult("weight_transfer", False, f"import failed: {exc}")

    import torch

    reg = TabICLRegressor(device="cpu", n_estimators=1, verbose=False)
    reg._load_model()  # noqa: SLF001 - intentional diagnostic access
    cpu_sd = {k: v.detach().cpu().clone() for k, v in reg.model_.state_dict().items()}
    reg.model_.to("mps")
    _sync("mps")
    mps_sd = {k: v.detach().cpu() for k, v in reg.model_.state_dict().items()}
    max_abs = 0.0
    worst = ""
    for k in cpu_sd:
        d = (cpu_sd[k].float() - mps_sd[k].float()).abs().max().item()
        if d >= max_abs:
            max_abs = d
            worst = k
    ok = max_abs == 0.0
    n_params = sum(v.numel() for v in cpu_sd.values())
    return CompareResult(
        "weight_transfer[cpu->mps->cpu]",
        ok,
        f"max_abs={max_abs:.3e} worst={worst} n_params={n_params}",
    )


def compare_tabicl_model_forward() -> list[CompareResult]:
    """Isolate neural net + InferenceManager from sklearn ensemble preprocessing."""
    try:
        from tabicl import TabICLRegressor
        from tabicl._model.inference_config import InferenceConfig
    except Exception as exc:  # noqa: BLE001
        return [CompareResult("tabicl_model_forward", False, f"import failed: {exc}")]

    import torch

    results: list[CompareResult] = []
    reg = TabICLRegressor(device="cpu", n_estimators=1, use_amp=False, use_fa3=False, verbose=False)
    reg._load_model()  # noqa: SLF001
    model = reg.model_
    model.eval()

    gen = torch.Generator(device="cpu").manual_seed(0)
    B, T, H = 2, 64, 16
    train_size = 40
    X = torch.randn(B, T, H, generator=gen, dtype=torch.float32)
    y = torch.randn(B, train_size, generator=gen, dtype=torch.float32)

    def _cfg(device: str, *, offload: str, verbose: bool) -> InferenceConfig:
        cfg = InferenceConfig()
        cfg.update_from_dict(
            {
                "COL_CONFIG": {
                    "device": device,
                    "use_amp": False,
                    "use_fa3": False,
                    "verbose": verbose,
                    "offload": offload,
                },
                "ROW_CONFIG": {
                    "device": device,
                    "use_amp": False,
                    "use_fa3": False,
                    "verbose": verbose,
                    "offload": offload,
                },
                "ICL_CONFIG": {
                    "device": device,
                    "use_amp": False,
                    "use_fa3": False,
                    "verbose": verbose,
                    "offload": offload,
                },
            }
        )
        return cfg

    # CPU baseline (raw quantiles + mean).
    with torch.no_grad():
        raw_cpu = model.predict_stats(
            X, y, output_type=["mean", "raw_quantiles"], inference_config=_cfg("cpu", offload="gpu", verbose=False)
        )
    mean_cpu = raw_cpu["mean"].detach().cpu()
    q_cpu = raw_cpu["raw_quantiles"].detach().cpu()
    print(
        f"  raw_cpu: mean_stats mean={mean_cpu.mean():.4f} std={mean_cpu.std():.4f} "
        f"q_std={q_cpu.std():.4f} q_shape={tuple(q_cpu.shape)}"
    )

    # MPS with forced GPU offload + verbose (surfaces InferenceManager decisions on CI).
    model_mps = model.to("mps")
    _sync("mps")
    print("  --- MPS predict_stats verbose (offload=gpu) ---")
    with torch.no_grad():
        raw_mps = model_mps.predict_stats(
            X.to("mps"),
            y.to("mps"),
            output_type=["mean", "raw_quantiles"],
            inference_config=_cfg("mps", offload="gpu", verbose=True),
        )
    _sync("mps")
    mean_mps = raw_mps["mean"].detach().cpu()
    q_mps = raw_mps["raw_quantiles"].detach().cpu()
    print(
        f"  raw_mps[offload=gpu]: mean_stats mean={mean_mps.mean():.4f} std={mean_mps.std():.4f} "
        f"q_std={q_mps.std():.4f}"
    )
    d_mean = (mean_cpu - mean_mps).abs().max().item()
    d_q = (q_cpu - q_mps).abs().max().item()
    results.append(
        CompareResult(
            "model_predict_stats[offload=gpu]",
            d_mean < 1e-2 and d_q < 1e-1,
            f"max|dmean|={d_mean:.3e} max|dq|={d_q:.3e} "
            f"mean_std_cpu={mean_cpu.std():.4f} mean_std_mps={mean_mps.std():.4f}",
        )
    )

    # MPS with offload=cpu (exercises D2H path used by auto mode under pressure).
    print("  --- MPS predict_stats verbose (offload=cpu) ---")
    with torch.no_grad():
        raw_mps_cpu = model_mps.predict_stats(
            X.to("mps"),
            y.to("mps"),
            output_type=["mean", "raw_quantiles"],
            inference_config=_cfg("mps", offload="cpu", verbose=True),
        )
    _sync("mps")
    mean_mps_cpu = raw_mps_cpu["mean"].detach().cpu()
    q_mps_cpu = raw_mps_cpu["raw_quantiles"].detach().cpu()
    print(
        f"  raw_mps[offload=cpu]: mean_stats mean={mean_mps_cpu.mean():.4f} std={mean_mps_cpu.std():.4f} "
        f"q_std={q_mps_cpu.std():.4f}"
    )
    d_mean2 = (mean_cpu - mean_mps_cpu).abs().max().item()
    d_q2 = (q_cpu - q_mps_cpu).abs().max().item()
    results.append(
        CompareResult(
            "model_predict_stats[offload=cpu]",
            d_mean2 < 1e-2 and d_q2 < 1e-1,
            f"max|dmean|={d_mean2:.3e} max|dq|={d_q2:.3e} "
            f"mean_std_cpu={mean_cpu.std():.4f} mean_std_mps={mean_mps_cpu.std():.4f}",
        )
    )

    # Move model back to CPU for subsequent probes that reload/fit.
    model.to("cpu")
    return results


def compare_tabicl_tiny() -> list[CompareResult]:
    """TabICLRegressor parity probes (mirrors CI failure + isolation variants)."""
    try:
        from sklearn.datasets import make_friedman1
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split

        from tabicl import TabICLRegressor
    except Exception as exc:  # noqa: BLE001
        return [CompareResult("tabicl_tiny", False, f"import failed: {exc}")]

    import numpy as np
    import torch

    X, y = make_friedman1(n_samples=120, n_features=16, noise=1.0, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    results: list[CompareResult] = []

    # Probe A: fit+predict on each device (CI failure mode).
    _mps_mem("before_fit_predict")
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
        print(f"  {_pred_stats(f'pred[{device}]', pred)}")
        print(f"  R2[{device}]={scores[device]:.4f}")
        if device == "mps":
            _mps_mem("after_mps_fit_predict")
    d_r2 = abs(scores["cpu"] - scores["mps"])
    d_pred = float(np.max(np.abs(preds["cpu"] - preds["mps"])))
    results.append(
        CompareResult(
            "tabicl_fit_predict[amp=off,n_estimators=2]",
            d_r2 < 1e-3 and d_pred < 1e-2,
            f"R2_cpu={scores['cpu']:.4f} R2_mps={scores['mps']:.4f} "
            f"|dR2|={d_r2:.3e} max|dpred|={d_pred:.3e}",
        )
    )

    # Probe B: fit on CPU, move model to MPS, predict (isolates device move).
    reg_cpu = TabICLRegressor(
        n_estimators=2,
        device="cpu",
        use_amp=False,
        use_fa3=False,
        random_state=0,
        verbose=False,
    )
    reg_cpu.fit(X_train, y_train)
    pred_cpu = reg_cpu.predict(X_test)
    reg_cpu.device_ = torch.device("mps")
    reg_cpu.model_.to("mps")
    reg_cpu._build_inference_config()  # noqa: SLF001 - rebuild MgrConfig devices
    if hasattr(reg_cpu, "model_kv_cache_"):
        reg_cpu._move_cache_to_device()  # noqa: SLF001
    _sync("mps")
    pred_moved = reg_cpu.predict(X_test)
    print(f"  {_pred_stats('pred[cpu_fit]', pred_cpu)}")
    print(f"  {_pred_stats('pred[moved_mps]', pred_moved)}")
    d_moved = float(np.max(np.abs(pred_cpu - pred_moved)))
    r2_moved = float(r2_score(y_test, pred_moved))
    results.append(
        CompareResult(
            "tabicl_fit_cpu_then_predict_mps",
            d_moved < 1e-2,
            f"R2_cpu={float(r2_score(y_test, pred_cpu)):.4f} R2_moved={r2_moved:.4f} "
            f"max|dpred|={d_moved:.3e}",
        )
    )
    _mps_mem("after_moved_predict")

    # Probe C: n_estimators=1 (smaller / simpler).
    scores1 = {}
    preds1 = {}
    for device in ("cpu", "mps"):
        reg = TabICLRegressor(
            n_estimators=1,
            device=device,
            use_amp=False,
            use_fa3=False,
            random_state=0,
            verbose=False,
        )
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)
        preds1[device] = pred
        scores1[device] = float(r2_score(y_test, pred))
        print(f"  {_pred_stats(f'pred_n1[{device}]', pred)}")
    d_r2_1 = abs(scores1["cpu"] - scores1["mps"])
    d_pred_1 = float(np.max(np.abs(preds1["cpu"] - preds1["mps"])))
    results.append(
        CompareResult(
            "tabicl_fit_predict[n_estimators=1]",
            d_r2_1 < 1e-3 and d_pred_1 < 1e-2,
            f"R2_cpu={scores1['cpu']:.4f} R2_mps={scores1['mps']:.4f} "
            f"|dR2|={d_r2_1:.3e} max|dpred|={d_pred_1:.3e}",
        )
    )

    # Probe D: sklearn path with forced offload=gpu (no auto offload).
    scores_g = {}
    preds_g = {}
    for device in ("cpu", "mps"):
        reg = TabICLRegressor(
            n_estimators=1,
            device=device,
            use_amp=False,
            use_fa3=False,
            random_state=0,
            verbose=device == "mps",
            offload_mode="gpu",
        )
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)
        preds_g[device] = pred
        scores_g[device] = float(r2_score(y_test, pred))
        print(f"  {_pred_stats(f'pred_offload_gpu[{device}]', pred)}")
    d_r2_g = abs(scores_g["cpu"] - scores_g["mps"])
    d_pred_g = float(np.max(np.abs(preds_g["cpu"] - preds_g["mps"])))
    results.append(
        CompareResult(
            "tabicl_fit_predict[offload=gpu,n_estimators=1]",
            d_r2_g < 1e-3 and d_pred_g < 1e-2,
            f"R2_cpu={scores_g['cpu']:.4f} R2_mps={scores_g['mps']:.4f} "
            f"|dR2|={d_r2_g:.3e} max|dpred|={d_pred_g:.3e}",
        )
    )
    return results


def run_mps_checks(*, include_tabicl: bool) -> list[CompareResult]:
    _section("MPS numerical sanity checks (vs CPU)")
    results: list[CompareResult] = []
    checks = [
        ("matmul float32", lambda: compare_matmul("float32")),
        ("matmul float16", lambda: compare_matmul("float16")),
        ("sdpa float32", lambda: compare_sdpa("float32")),
        ("sdpa float16", lambda: compare_sdpa("float16")),
        ("sdpa large", compare_sdpa_large),
        ("layernorm", compare_layernorm),
        ("autocast matmul", compare_autocast_matmul),
    ]
    if include_tabicl:
        checks.append(("weight transfer", compare_weight_transfer))
        checks.append(("tabicl model forward", compare_tabicl_model_forward))
        checks.append(("tabicl probes", compare_tabicl_tiny))

    for label, fn in checks:
        print(f"\n-- {label} --")
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001
            result = CompareResult(label, False, f"EXCEPTION {type(exc).__name__}: {exc}")
            traceback.print_exc()
        batch = result if isinstance(result, list) else [result]
        for item in batch:
            status = "OK" if item.ok else "FAIL"
            print(f"[{status}] {item.name}: {item.detail}")
            results.append(item)
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
