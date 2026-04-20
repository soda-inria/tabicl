"""Phase 2: attribution-head unit tests.

Covers the three head modules in isolation (no TabICL trunk dependency here —
trunk wiring is covered in ``test_return_column_embeddings.py``).

PLAN gates exercised:
  - Head A is permutation-equivariant over the column axis by construction.
  - Heads A and I share architecture but keep independent weights.
  - Heads A, I, C start near zero (small-init on the final layer) so the
    Phase 4 freeze-trunk / train-heads-only warmup starts from a neutral
    point rather than a random offset.
  - Head C sum-pools over e_S; an all-False mask yields an e_S of zero and
    reduces Head C to a query-only function.
  - All heads are differentiable end-to-end (loss.backward() populates grads).
"""
from __future__ import annotations

import pytest
import torch

from tabicl.model.heads import (
    ConditionalHead,
    InterventionalHead,
    ObservationalHead,
)


def _embeddings(B: int = 2, H: int = 5, E: int = 16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, H, E, generator=g)


# ---------------------------------------------------------------------------
# Head A — ObservationalHead
# ---------------------------------------------------------------------------


def test_observational_head_shape():
    head = ObservationalHead(embed_dim=16)
    e = _embeddings(B=3, H=7, E=16)
    out = head(e)
    assert out.shape == (3, 7)
    assert out.dtype == e.dtype


def test_observational_head_permutation_equivariance():
    """Per-column parameter sharing -> permuting columns permutes outputs."""
    torch.manual_seed(0)
    head = ObservationalHead(embed_dim=16)
    e = _embeddings(B=2, H=5, E=16)

    perm = torch.tensor([3, 0, 4, 1, 2])
    out_original = head(e)
    out_permuted = head(e[:, perm, :])

    assert torch.allclose(out_permuted, out_original[:, perm], atol=1e-6)


def test_observational_head_small_init_outputs_near_zero():
    """Small-init final layer -> untrained head produces near-zero scores."""
    torch.manual_seed(0)
    head = ObservationalHead(embed_dim=32)
    e = _embeddings(B=4, H=10, E=32)
    out = head(e)
    # With std=0.01 on the final 1-dim linear, outputs should be well under 1.
    assert out.abs().max().item() < 0.1, out.abs().max().item()


def test_observational_head_backprop():
    head = ObservationalHead(embed_dim=16)
    e = _embeddings()
    target = torch.zeros(e.shape[:2])
    loss = torch.nn.functional.huber_loss(head(e), target)
    loss.backward()
    assert all(p.grad is not None for p in head.parameters())


# ---------------------------------------------------------------------------
# Head I — InterventionalHead
# ---------------------------------------------------------------------------


def test_interventional_head_shape_and_equivariance():
    torch.manual_seed(1)
    head = InterventionalHead(embed_dim=16)
    e = _embeddings(B=2, H=5, E=16)
    out = head(e)
    assert out.shape == (2, 5)

    perm = torch.tensor([4, 1, 3, 0, 2])
    assert torch.allclose(head(e[:, perm, :]), out[:, perm], atol=1e-6)


def test_heads_a_and_i_have_independent_weights():
    """Same architecture, distinct weight tensors."""
    torch.manual_seed(0)
    a = ObservationalHead(embed_dim=16)
    i = InterventionalHead(embed_dim=16)

    # Both have .mlp.fc1 and .mlp.fc2; weights should not be aliased.
    assert a.mlp.fc1.weight.data_ptr() != i.mlp.fc1.weight.data_ptr()
    assert a.mlp.fc2.weight.data_ptr() != i.mlp.fc2.weight.data_ptr()

    # And training A does not move I.
    e = _embeddings()
    before_fc2_i = i.mlp.fc2.weight.detach().clone()
    loss = a(e).sum()
    loss.backward()
    # Apply a fake step to A's params only.
    with torch.no_grad():
        for p in a.parameters():
            if p.grad is not None:
                p.sub_(0.1 * p.grad)
    assert torch.equal(before_fc2_i, i.mlp.fc2.weight)


# ---------------------------------------------------------------------------
# Head C — ConditionalHead
# ---------------------------------------------------------------------------


def test_conditional_head_shape():
    head = ConditionalHead(embed_dim=16)
    e = _embeddings(B=2, H=5, E=16)
    cond = torch.tensor([[True, False, False, True, False],
                         [False, True, False, False, True]])
    out = head(e, cond)
    assert out.shape == (2, 5)


def test_conditional_head_empty_conditioning_set_zeroes_e_S():
    """All-False cond_mask -> e_S is the zero vector; head reduces to f(e_i)."""
    torch.manual_seed(0)
    head = ConditionalHead(embed_dim=16)
    e = _embeddings(B=2, H=5, E=16)
    cond_empty = torch.zeros(2, 5, dtype=torch.bool)

    # Manually evaluate the head with a forced e_S = 0: just run the fusion MLP
    # on [e_i ; 0]. This must match head(e, cond_empty).
    zero_eS = torch.zeros_like(e)
    fused = torch.cat([e, zero_eS], dim=-1)
    manual = head.fc2(head.act(head.fc1(fused))).squeeze(-1)
    out = head(e, cond_empty)
    assert torch.allclose(out, manual, atol=1e-6)


def test_conditional_head_sum_pool_semantics():
    """Doubling an in-S column's embedding doubles its contribution to e_S."""
    torch.manual_seed(0)
    head = ConditionalHead(embed_dim=16)
    e1 = _embeddings(B=1, H=3, E=16).clone()
    e2 = e1.clone()
    e2[:, 0, :] = e1[:, 0, :] * 2.0

    cond = torch.tensor([[True, False, True]])  # S = {0, 2}
    # e_S for e2 = 2 * e1[:,0] + e1[:,2]; for e1 = e1[:,0] + e1[:,2].
    # So outputs should differ where the fusion MLP is non-degenerate.
    out1 = head(e1, cond)
    out2 = head(e2, cond)
    assert not torch.allclose(out1, out2, atol=1e-4)


def test_conditional_head_small_init_outputs_near_zero():
    torch.manual_seed(0)
    head = ConditionalHead(embed_dim=32)
    e = _embeddings(B=4, H=10, E=32)
    cond = torch.zeros(4, 10, dtype=torch.bool)
    cond[:, :3] = True
    out = head(e, cond)
    assert out.abs().max().item() < 0.1


def test_conditional_head_rejects_non_bool_mask():
    head = ConditionalHead(embed_dim=16)
    e = _embeddings()
    bad = torch.ones(e.shape[:2], dtype=torch.float32)
    with pytest.raises(TypeError):
        head(e, bad)


def test_conditional_head_rejects_mismatched_mask_shape():
    head = ConditionalHead(embed_dim=16)
    e = _embeddings(B=2, H=5, E=16)
    bad = torch.zeros(2, 4, dtype=torch.bool)
    with pytest.raises(ValueError):
        head(e, bad)


def test_conditional_head_backprop():
    head = ConditionalHead(embed_dim=16)
    e = _embeddings().requires_grad_(True)
    cond = torch.zeros(e.shape[:2], dtype=torch.bool)
    cond[:, 0] = True
    out = head(e, cond)
    out.sum().backward()
    assert e.grad is not None
    assert all(p.grad is not None for p in head.parameters())


# ---------------------------------------------------------------------------
# Integration with Phase-1 trunk output
# ---------------------------------------------------------------------------


def test_heads_consume_tabicl_column_embeddings():
    """Smoke test: plug TabICL's column_embeddings straight into the heads."""
    from tabicl.model.tabicl import TabICL

    torch.manual_seed(0)
    model = TabICL(
        max_classes=3,
        embed_dim=32,
        col_num_blocks=2,
        col_nhead=4,
        col_num_inds=8,
        icl_num_blocks=2,
        icl_nhead=4,
        row_num_blocks=2,
        row_nhead=4,
        row_num_cls=4,
        ff_factor=2,
    )
    model.train()

    B, H, train, test = 2, 5, 12, 4
    X = torch.randn(B, train + test, H)
    y_train = torch.randint(0, 3, (B, train))
    _, col_emb = model(X, y_train, return_column_embeddings=True)
    assert col_emb.shape == (B, H, model.embed_dim)

    head_a = ObservationalHead(embed_dim=model.embed_dim)
    head_i = InterventionalHead(embed_dim=model.embed_dim)
    head_c = ConditionalHead(embed_dim=model.embed_dim)

    a = head_a(col_emb)
    i = head_i(col_emb)
    cond = torch.zeros(B, H, dtype=torch.bool)
    cond[:, 0] = True
    c = head_c(col_emb, cond)

    assert a.shape == (B, H)
    assert i.shape == (B, H)
    assert c.shape == (B, H)
