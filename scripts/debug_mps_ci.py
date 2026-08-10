#!/usr/bin/env python
"""Craft a minimal CPU-vs-MPS numerical reproducer for virtualized Apple Silicon CI.

GitHub Actions macOS arm64 runners report ``Apple M* (Virtual)`` / ``VirtualMac*``.
There, isolated kernels can match CPU while larger attention-style graphs silently
diverge. This script does **not** import TabICL; it escalates pure-PyTorch probes
from known-good SDPA toward TabICL-like MultiheadAttention patterns so CI logs
can identify the smallest failing case.

Exit code is 0 even when cases fail (diagnostics). Pass ``--strict`` to fail on
mismatches. Use ``--only NAME`` / ``--from NAME`` while iterating.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import textwrap
import traceback
from dataclasses import dataclass
from typing import Callable


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    sys.stdout.flush()


def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=30).strip()
    except Exception as exc:  # noqa: BLE001
        return f"<failed: {type(exc).__name__}: {exc}>"


def inspect_host() -> None:
    _section("Host / PyTorch")
    print(f"python:     {sys.version.split()[0]}")
    print(f"executable: {sys.executable}")
    print(f"platform:   {platform.platform()}")
    if sys.platform == "darwin":
        print(f"sysctl.cpu: {_run_cmd(['sysctl', '-n', 'machdep.cpu.brand_string'])}")
        print(f"sysctl.model:{_run_cmd(['sysctl', '-n', 'hw.model'])}")
        print(f"sysctl.mem: {_run_cmd(['sysctl', '-n', 'hw.memsize'])}")
        sp = _run_cmd(["system_profiler", "SPHardwareDataType", "-detailLevel", "mini"])
        for line in sp.splitlines():
            if any(k in line for k in ("Chip", "Model", "Memory", "Cores")):
                print(f"  {line.strip()}")

    import torch

    print(f"torch:      {torch.__version__}")
    print(f"mps.built:  {torch.backends.mps.is_built()}")
    print(f"mps.avail:  {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"mps.rec_max:{torch.mps.recommended_max_memory() / 1024**3:.2f}GB")
    for key in (
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "PYTORCH_MPS_PREFER_METAL",
        "PYTORCH_MPS_FAST_MATH",
    ):
        print(f"  {key}={os.environ.get(key)!r}")


@dataclass
class CaseResult:
    name: str
    ok: bool
    detail: str
    snippet: str = ""


def _sync() -> None:
    import torch

    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _max_abs(a, b) -> float:
    return (a.float() - b.float()).abs().max().item()


def _compare(name: str, cpu_fn: Callable, mps_fn: Callable, *, tol: float, snippet: str) -> CaseResult:
    import torch

    try:
        with torch.no_grad():
            out_cpu = cpu_fn()
            out_mps = mps_fn()
            _sync()
            if not isinstance(out_cpu, torch.Tensor):
                out_cpu = out_cpu[0]
            if not isinstance(out_mps, torch.Tensor):
                out_mps = out_mps[0]
            out_mps = out_mps.detach().cpu()
            out_cpu = out_cpu.detach().cpu()
            diff = _max_abs(out_cpu, out_mps)
            ok = diff < tol
            detail = (
                f"max_abs={diff:.3e} tol={tol:.1e} "
                f"std_cpu={out_cpu.float().std():.4f} std_mps={out_mps.float().std():.4f} "
                f"shape={tuple(out_cpu.shape)}"
            )
            return CaseResult(name, ok, detail, snippet=snippet)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return CaseResult(name, False, f"EXCEPTION {type(exc).__name__}: {exc}", snippet=snippet)


def _seeded(seed: int = 0):
    import torch

    return torch.Generator(device="cpu").manual_seed(seed)


# ---------------------------------------------------------------------------
# Escalating probes (pure torch). Keep snippets copy-pasteable.
# ---------------------------------------------------------------------------


def case_sdpa_contiguous() -> CaseResult:
    import torch
    import torch.nn.functional as F

    snippet = textwrap.dedent(
        """\
        # contiguous SDPA baseline
        q = torch.randn(2, 8, 64, 16)
        o_cpu = F.scaled_dot_product_attention(q, q, q)
        o_mps = F.scaled_dot_product_attention(q.to("mps"), q.to("mps"), q.to("mps")).cpu()
        """
    )
    g = _seeded()
    q = torch.randn(2, 8, 64, 16, generator=g)

    def cpu():
        return F.scaled_dot_product_attention(q, q, q)

    def mps():
        qm = q.to("mps")
        return F.scaled_dot_product_attention(qm, qm, qm)

    return _compare("01_sdpa_contiguous", cpu, mps, tol=1e-3, snippet=snippet)


def case_sdpa_permute_noncontig() -> CaseResult:
    """Known Torch MPS footgun: permute-produced non-contiguous QKV (pytorch#181133)."""
    import torch
    import torch.nn.functional as F

    snippet = textwrap.dedent(
        """\
        # non-contiguous SDPA via permute(0,2,1,3); contiguous=False often diverges on MPS
        x = torch.randn(2, 4, 8, 64)  # B, S, H, D with D=64, S in [2,8]
        q = x.permute(0, 2, 1, 3)     # B, H, S, D — non-contiguous
        o = F.scaled_dot_product_attention(q, q, q)
        """
    )
    g = _seeded()
    # Shape band from pytorch#181133: head_dim>=64, seq_len in [2,8], batch>=2
    x = torch.randn(2, 4, 8, 64, generator=g)

    def _run(device: str, make_contiguous: bool):
        t = x.to(device)
        q = t.permute(0, 2, 1, 3)
        if make_contiguous:
            q = q.contiguous()
        return F.scaled_dot_product_attention(q, q, q)

    def cpu():
        return _run("cpu", False)

    def mps():
        return _run("mps", False)

    return _compare("02_sdpa_permute_noncontig", cpu, mps, tol=1e-2, snippet=snippet)


def case_sdpa_permute_then_contiguous() -> CaseResult:
    import torch
    import torch.nn.functional as F

    snippet = textwrap.dedent(
        """\
        q = x.permute(0, 2, 1, 3).contiguous()
        o = F.scaled_dot_product_attention(q, q, q)
        """
    )
    g = _seeded()
    x = torch.randn(2, 4, 8, 64, generator=g)

    def _run(device: str):
        q = x.to(device).permute(0, 2, 1, 3).contiguous()
        return F.scaled_dot_product_attention(q, q, q)

    return _compare(
        "03_sdpa_permute_then_contiguous",
        lambda: _run("cpu"),
        lambda: _run("mps"),
        tol=1e-2,
        snippet=snippet,
    )


def case_mha_view_transpose_sdpa() -> CaseResult:
    """TabICL-like: packed in_proj → view → transpose(-3,-2) → SDPA (often non-contig)."""
    import torch
    import torch.nn.functional as F

    snippet = textwrap.dedent(
        """\
        # Mimic torch MHA reshape used by TabICL attention.py
        B, S, E, H = 2, 64, 128, 8
        D = E // H
        x = torch.randn(B, S, E)
        w = torch.randn(3 * E, E)
        qkv = F.linear(x, w).view(B, S, 3, H, D).unbind(2)
        q, k, v = [t.transpose(1, 2) for t in qkv]  # (B, H, S, D), often non-contiguous
        o = F.scaled_dot_product_attention(q, k, v)
        """
    )
    g = _seeded(1)
    B, S, E, H = 2, 64, 128, 8
    D = E // H
    x = torch.randn(B, S, E, generator=g)
    w = torch.randn(3 * E, E, generator=g)
    b = torch.randn(3 * E, generator=g)

    def _run(device: str, contiguous_qkv: bool):
        xd, wd, bd = x.to(device), w.to(device), b.to(device)
        qkv = F.linear(xd, wd, bd)
        q, k, v = qkv.view(B, S, 3, H, D).unbind(-3)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if contiguous_qkv:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        return F.scaled_dot_product_attention(q, k, v)

    return _compare(
        "04_mha_view_transpose_sdpa",
        lambda: _run("cpu", False),
        lambda: _run("mps", False),
        tol=1e-2,
        snippet=snippet,
    )


def case_mha_view_transpose_sdpa_contig() -> CaseResult:
    import torch
    import torch.nn.functional as F

    snippet = "same as 04 but q,k,v = map(Tensor.contiguous, (q,k,v)) before SDPA"
    g = _seeded(1)
    B, S, E, H = 2, 64, 128, 8
    D = E // H
    x = torch.randn(B, S, E, generator=g)
    w = torch.randn(3 * E, E, generator=g)
    b = torch.randn(3 * E, generator=g)

    def _run(device: str):
        xd, wd, bd = x.to(device), w.to(device), b.to(device)
        q, k, v = F.linear(xd, wd, bd).view(B, S, 3, H, D).unbind(-3)
        q, k, v = [t.transpose(1, 2).contiguous() for t in (q, k, v)]
        return F.scaled_dot_product_attention(q, k, v)

    return _compare(
        "05_mha_view_transpose_sdpa_contig",
        lambda: _run("cpu"),
        lambda: _run("mps"),
        tol=1e-2,
        snippet=snippet,
    )


def case_multidim_batch_flatten_sdpa() -> CaseResult:
    """TabICL sdpa_with_flattened_batch: (B, F, H, S, D) → reshape(-1, H, S, D)."""
    import torch
    import torch.nn.functional as F

    snippet = textwrap.dedent(
        """\
        # Col-embed-like batching: features as an extra batch dim
        q = torch.randn(2, 20, 8, 64, 16)  # B, F, H, S, D
        qf = q.reshape(-1, 8, 64, 16)
        o = F.scaled_dot_product_attention(qf, qf, qf).view(q.shape)
        """
    )
    g = _seeded(2)
    q = torch.randn(2, 20, 8, 64, 16, generator=g)

    def _run(device: str):
        qd = q.to(device)
        qf = qd.reshape(-1, *qd.shape[-3:])
        out = F.scaled_dot_product_attention(qf, qf, qf)
        return out.view(qd.shape)

    return _compare(
        "06_multidim_batch_flatten_sdpa",
        lambda: _run("cpu"),
        lambda: _run("mps"),
        tol=1e-2,
        snippet=snippet,
    )


def case_sdpa_with_additive_mask() -> CaseResult:
    import torch
    import torch.nn.functional as F

    snippet = textwrap.dedent(
        """\
        q = torch.randn(2, 8, 64, 16)
        mask = torch.zeros(2, 8, 64, 64)
        mask[..., :, 32:] = float("-inf")  # key padding style
        o = F.scaled_dot_product_attention(q, q, q, attn_mask=mask)
        """
    )
    g = _seeded(3)
    q = torch.randn(2, 8, 64, 16, generator=g)
    mask = torch.zeros(2, 8, 64, 64)
    mask[..., :, 32:] = float("-inf")

    def _run(device: str):
        qd = q.to(device)
        md = mask.to(device)
        return F.scaled_dot_product_attention(qd, qd, qd, attn_mask=md)

    return _compare(
        "07_sdpa_additive_mask",
        lambda: _run("cpu"),
        lambda: _run("mps"),
        tol=1e-2,
        snippet=snippet,
    )


def case_ssmax_scaled_queries_sdpa() -> CaseResult:
    """SSMax-like: scale Q by per-head (and elementwise) factors before SDPA."""
    import torch
    import torch.nn.functional as F

    snippet = textwrap.dedent(
        """\
        q = torch.randn(2, 8, 64, 16)
        scales = torch.randn(1, 8, 1, 16)  # elementwise qassmax-like
        o = F.scaled_dot_product_attention(q * scales, q, q)
        """
    )
    g = _seeded(4)
    q = torch.randn(2, 8, 64, 16, generator=g)
    scales = torch.randn(1, 8, 1, 16, generator=g)

    def _run(device: str):
        qd, sd = q.to(device), scales.to(device)
        return F.scaled_dot_product_attention(qd * sd, qd, qd)

    return _compare(
        "08_ssmax_scaled_queries_sdpa",
        lambda: _run("cpu"),
        lambda: _run("mps"),
        tol=1e-2,
        snippet=snippet,
    )


def case_rope_interleaved_then_sdpa() -> CaseResult:
    """Minimal interleaved RoPE then SDPA (TabICL default rope_interleaved=True)."""
    import torch
    import torch.nn.functional as F

    snippet = textwrap.dedent(
        """\
        def rotate_half(x):
            x = x.unflatten(-1, (-1, 2))
            x1, x2 = x.unbind(-1)
            return torch.stack((-x2, x1), dim=-1).flatten(-2)

        q = torch.randn(2, 8, 64, 16)
        # freqs broadcast on seq dim
        theta = torch.randn(64, 16)
        q_rot = q * theta.cos() + rotate_half(q) * theta.sin()
        o = F.scaled_dot_product_attention(q_rot, q_rot, q)
        """
    )
    g = _seeded(5)
    q = torch.randn(2, 8, 64, 16, generator=g)
    theta = torch.randn(64, 16, generator=g)

    def rotate_half(x: "torch.Tensor") -> "torch.Tensor":
        x = x.unflatten(-1, (-1, 2))
        x1, x2 = x.unbind(-1)
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def _run(device: str):
        qd, th = q.to(device), theta.to(device)
        q_rot = qd * th.cos() + rotate_half(qd) * th.sin()
        return F.scaled_dot_product_attention(q_rot, q_rot, qd)

    return _compare(
        "09_rope_interleaved_then_sdpa",
        lambda: _run("cpu"),
        lambda: _run("mps"),
        tol=1e-2,
        snippet=snippet,
    )


def case_nn_multihead_attention() -> CaseResult:
    import torch
    import torch.nn as nn

    snippet = textwrap.dedent(
        """\
        mha = nn.MultiheadAttention(128, 8, batch_first=True, dropout=0.0)
        x = torch.randn(2, 64, 128)
        o, _ = mha(x, x, x, need_weights=False)
        """
    )
    g = _seeded(6)
    x = torch.randn(2, 64, 128, generator=g)
    mha = nn.MultiheadAttention(128, 8, batch_first=True, dropout=0.0)
    mha.eval()

    def _run(device: str):
        m = mha.to(device)
        xd = x.to(device)
        with torch.no_grad():
            out, _ = m(xd, xd, xd, need_weights=False)
        return out

    return _compare(
        "10_nn_multihead_attention",
        lambda: _run("cpu"),
        lambda: _run("mps"),
        tol=1e-2,
        snippet=snippet,
    )


def case_transformer_encoder_stack() -> CaseResult:
    """Deeper stack closer to TabICL col/row encoders."""
    import torch
    import torch.nn as nn

    snippet = textwrap.dedent(
        """\
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=512,
            dropout=0.0, batch_first=True, norm_first=True, activation="gelu",
        )
        enc = nn.TransformerEncoder(layer, num_layers=4)
        x = torch.randn(2, 64, 128)
        o = enc(x)
        """
    )
    g = _seeded(7)
    x = torch.randn(2, 64, 128, generator=g)
    layer = nn.TransformerEncoderLayer(
        d_model=128,
        nhead=8,
        dim_feedforward=512,
        dropout=0.0,
        batch_first=True,
        norm_first=True,
        activation="gelu",
    )
    enc = nn.TransformerEncoder(layer, num_layers=4)
    enc.eval()

    def _run(device: str):
        m = enc.to(device)
        with torch.no_grad():
            return m(x.to(device))

    return _compare(
        "11_transformer_encoder_4layer",
        lambda: _run("cpu"),
        lambda: _run("mps"),
        tol=5e-2,
        snippet=snippet,
    )


def case_wide_feature_batch_encoder() -> CaseResult:
    """Match col-embed style: many parallel sequences (features as batch)."""
    import torch
    import torch.nn as nn

    snippet = textwrap.dedent(
        """\
        # B*F flattened sequences through a small encoder (col-embed-ish)
        x = torch.randn(2 * 20, 64, 128)
        layer = nn.TransformerEncoderLayer(128, 8, 512, dropout=0.0, batch_first=True, norm_first=True)
        o = nn.TransformerEncoder(layer, num_layers=2)(x)
        """
    )
    g = _seeded(8)
    x = torch.randn(40, 64, 128, generator=g)
    layer = nn.TransformerEncoderLayer(
        128, 8, 512, dropout=0.0, batch_first=True, norm_first=True, activation="gelu"
    )
    enc = nn.TransformerEncoder(layer, num_layers=2)
    enc.eval()

    def _run(device: str):
        m = enc.to(device)
        with torch.no_grad():
            return m(x.to(device))

    return _compare(
        "12_wide_feature_batch_encoder",
        lambda: _run("cpu"),
        lambda: _run("mps"),
        tol=5e-2,
        snippet=snippet,
    )


def case_gelu_layernorm_chain() -> CaseResult:
    import torch
    import torch.nn as nn

    snippet = "LN→Linear→GELU→Linear×8 residual-ish chain without attention"
    g = _seeded(9)
    x = torch.randn(2, 64, 128, generator=g)
    mods = nn.ModuleList()
    for _ in range(8):
        mods.append(
            nn.Sequential(
                nn.LayerNorm(128),
                nn.Linear(128, 512),
                nn.GELU(),
                nn.Linear(512, 128),
            )
        )

    def _run(device: str):
        xd = x.to(device)
        ms = mods.to(device)
        with torch.no_grad():
            for m in ms:
                xd = xd + m(xd)
        return xd

    return _compare(
        "13_gelu_layernorm_mlp_chain",
        lambda: _run("cpu"),
        lambda: _run("mps"),
        tol=1e-2,
        snippet=snippet,
    )


CASES: list[tuple[str, Callable[[], CaseResult]]] = [
    ("01_sdpa_contiguous", case_sdpa_contiguous),
    ("02_sdpa_permute_noncontig", case_sdpa_permute_noncontig),
    ("03_sdpa_permute_then_contiguous", case_sdpa_permute_then_contiguous),
    ("04_mha_view_transpose_sdpa", case_mha_view_transpose_sdpa),
    ("05_mha_view_transpose_sdpa_contig", case_mha_view_transpose_sdpa_contig),
    ("06_multidim_batch_flatten_sdpa", case_multidim_batch_flatten_sdpa),
    ("07_sdpa_additive_mask", case_sdpa_with_additive_mask),
    ("08_ssmax_scaled_queries_sdpa", case_ssmax_scaled_queries_sdpa),
    ("09_rope_interleaved_then_sdpa", case_rope_interleaved_then_sdpa),
    ("10_nn_multihead_attention", case_nn_multihead_attention),
    ("11_transformer_encoder_4layer", case_transformer_encoder_stack),
    ("12_wide_feature_batch_encoder", case_wide_feature_batch_encoder),
    ("13_gelu_layernorm_mlp_chain", case_gelu_layernorm_chain),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any case fails.")
    parser.add_argument("--only", action="append", default=[], help="Run only these case names (repeatable).")
    parser.add_argument("--from", dest="from_case", default=None, help="Start from this case name.")
    args = parser.parse_args(argv)

    inspect_host()

    import torch

    if not torch.backends.mps.is_available():
        _section("Cases")
        print("MPS unavailable — nothing to compare.")
        return 0

    names = [n for n, _ in CASES]
    selected = CASES
    if args.only:
        wanted = set(args.only)
        selected = [(n, fn) for n, fn in CASES if n in wanted]
        missing = wanted - {n for n, _ in selected}
        if missing:
            print(f"Unknown --only cases: {sorted(missing)}; known={names}", file=sys.stderr)
            return 2
    if args.from_case:
        if args.from_case not in names:
            print(f"Unknown --from case {args.from_case!r}; known={names}", file=sys.stderr)
            return 2
        idx = names.index(args.from_case)
        selected = CASES[idx:]

    _section("CPU vs MPS cases (escalating toward TabICL-like attention)")
    results: list[CaseResult] = []
    first_fail: CaseResult | None = None
    for name, fn in selected:
        print(f"\n-- {name} --")
        result = fn()
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
        results.append(result)
        if not result.ok and first_fail is None:
            first_fail = result
        sys.stdout.flush()

    _section("Summary")
    for r in results:
        print(f"{'OK  ' if r.ok else 'FAIL'}  {r.name}: {r.detail}")
    n_fail = sum(not r.ok for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} passed, {n_fail} failed.")

    if first_fail is not None:
        _section(f"Smallest failing case: {first_fail.name}")
        print(first_fail.detail)
        if first_fail.snippet.strip():
            print("\nCandidate minimal snippet:\n")
            print(first_fail.snippet.rstrip())
            print()
    else:
        _section("No failing case")
        print("All selected probes matched within tolerance on this host.")
        print("Next iteration: add a narrower probe between the last OK and TabICL.")

    if args.strict and n_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
