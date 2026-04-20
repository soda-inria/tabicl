"""Phase 1: TabICL exposes per-column embeddings without disturbing predictions.

Checks:
  - ``return_column_embeddings=True`` returns a ``(logits, column_embeddings)``
    tuple with ``column_embeddings.shape == (B, n_features, embed_dim)``.
  - Logits from the default path (flag off) and the new path (flag on) are
    bit-for-bit identical in training mode on CPU — the PLAN.md Phase 1 gate.
  - A second forward pass with the flag off returns the exact same logits as
    the first (regression guard that the default path is untouched).
"""
from __future__ import annotations

import torch

from tabicl.model.tabicl import TabICL


def _build_small_model(seed: int = 0) -> TabICL:
    torch.manual_seed(seed)
    return TabICL(
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


def _toy_batch(B: int = 2, train: int = 12, test: int = 4, H: int = 5):
    torch.manual_seed(123)
    T = train + test
    X = torch.randn(B, T, H)
    y_train = torch.randint(0, 3, (B, train))
    return X, y_train


def test_column_embeddings_shape_training_mode():
    model = _build_small_model()
    model.train()
    X, y_train = _toy_batch(H=5)

    logits, col_emb = model(X, y_train, return_column_embeddings=True)

    assert logits.shape == (X.shape[0], X.shape[1] - y_train.shape[1], model.max_classes)
    assert col_emb.shape == (X.shape[0], X.shape[1] - (X.shape[1] - 5), model.embed_dim)  # noqa
    # Simpler + explicit:
    assert col_emb.shape == (2, 5, model.embed_dim)


def test_default_path_unchanged():
    """Calling the default path twice gives bit-exact identical logits."""
    model = _build_small_model()
    model.train()
    X, y_train = _toy_batch()

    with torch.no_grad():
        logits_a = model(X, y_train)
        logits_b = model(X, y_train)

    assert torch.equal(logits_a, logits_b)


def test_logits_bit_exact_between_flag_on_and_off():
    """PLAN.md Phase 1 gate: flag-on logits are bit-for-bit identical to flag-off."""
    model = _build_small_model()
    model.train()
    X, y_train = _toy_batch()

    with torch.no_grad():
        logits_off = model(X, y_train)
        logits_on, col_emb = model(X, y_train, return_column_embeddings=True)

    assert logits_on.shape == logits_off.shape
    assert col_emb.shape == (X.shape[0], X.shape[2], model.embed_dim)
    assert torch.equal(logits_off, logits_on), (
        f"Max abs diff: {(logits_off - logits_on).abs().max().item():.3e}"
    )


def test_column_embeddings_shape_inference_mode():
    model = _build_small_model()
    model.eval()
    X, y_train = _toy_batch(H=7)

    with torch.no_grad():
        logits, col_emb = model(X, y_train, return_column_embeddings=True)

    assert col_emb.shape == (X.shape[0], 7, model.embed_dim)
    assert logits.shape[0] == X.shape[0]
    assert logits.shape[1] == X.shape[1] - y_train.shape[1]
