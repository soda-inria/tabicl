"""Phase 2 — attribution head modules.

Three heads consume per-column embeddings ``e`` of shape ``(B, H, E)`` as
returned by ``TabICL.forward(..., return_column_embeddings=True)``:

- ``ObservationalHead``  (Head A): observational information score per feature,
  parameter-shared MLP applied independently to each column so the mapping is
  permutation-equivariant by construction.
- ``InterventionalHead`` (Head I): interventional-effect magnitude per feature.
  Architecturally identical to Head A with separate weights. The PLAN's
  identifiability scoping (train only on identifiable priors) is enforced at
  the loss via an ``is_identifiable`` sample mask, not inside this module.
- ``ConditionalHead``    (Head C): conditional contribution c_{i|S}. Consumes
  the per-column embeddings and a boolean conditioning-set mask, emits a
  scalar per feature. Conditioning-set representation is a sum-pool over e_S
  (PLAN §Phase 2); attention-pool is scheduled as an ablation in §Phase 6d.

The trunk stays frozen for the first 1000 steps of training (PLAN §Phase 4)
while these heads stabilise. To make that warmup useful, each head's final
linear layer is initialised with small weights so untrained heads produce
near-zero outputs rather than random offsets.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor


def _build_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name!r} (supported: 'gelu', 'relu')")


def _small_init_final_layer(linear: nn.Linear, std: float = 0.01) -> None:
    """Small-weight init for the output projection of an attribution head."""
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class _PerColumnScalarMLP(nn.Module):
    """MLP ``E -> hidden -> 1`` applied independently to every column.

    Shared building block for Heads A and I. The same weights are applied to
    every column embedding along the H axis, which makes the resulting score
    tensor permutation-equivariant over columns.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        hidden_dim = hidden_dim if hidden_dim is not None else max(1, embed_dim // 2)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = _build_activation(activation)
        self.fc2 = nn.Linear(hidden_dim, 1)

        _small_init_final_layer(self.fc2)

    def forward(self, column_embeddings: Tensor) -> Tensor:
        if column_embeddings.dim() != 3:
            raise ValueError(
                f"Expected column_embeddings of shape (B, H, E), got {tuple(column_embeddings.shape)}"
            )
        x = self.fc1(column_embeddings)       # (B, H, hidden)
        x = self.act(x)
        x = self.fc2(x)                       # (B, H, 1)
        return x.squeeze(-1)                  # (B, H)


class ObservationalHead(nn.Module):
    """Head A — observational information score per feature.

    Parameters
    ----------
    embed_dim : int
        Dimensionality ``E`` of the per-column embeddings emitted by the
        trunk. Must match ``TabICL.embed_dim``.
    hidden_dim : Optional[int], default=None
        Hidden-layer dimension. Defaults to ``embed_dim // 2``.
    activation : str, default="gelu"
        Activation between the two linear layers. ``"gelu"`` or ``"relu"``.

    Notes
    -----
    The PLAN target o*_i is on the scale of ``Var(y)``. Do not apply a
    squashing nonlinearity to the output — the loss (Huber, delta=1.0) is
    expected to operate on the natural-scale score.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = _PerColumnScalarMLP(embed_dim, hidden_dim=hidden_dim, activation=activation)

    def forward(self, column_embeddings: Tensor) -> Tensor:
        """Compute ``(B, H)`` scores from per-column embeddings ``(B, H, E)``."""
        return self.mlp(column_embeddings)


class InterventionalHead(nn.Module):
    """Head I — interventional effect magnitude per feature.

    Same architecture as ``ObservationalHead`` with independent weights. The
    identifiability restriction (train only on identifiable prior families) is
    applied at the training objective via a per-sample ``is_identifiable``
    mask; this module is architecturally unconstrained.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = _PerColumnScalarMLP(embed_dim, hidden_dim=hidden_dim, activation=activation)

    def forward(self, column_embeddings: Tensor) -> Tensor:
        """Compute ``(B, H)`` scores from per-column embeddings ``(B, H, E)``."""
        return self.mlp(column_embeddings)


class ConditionalHead(nn.Module):
    """Head C — conditional contribution c_{i|S} per feature.

    Consumes the per-column embeddings together with a boolean mask marking
    which features belong to the conditioning set ``S``, and emits a scalar
    per feature: the conditional contribution of revealing ``X_i`` given
    ``X_S``.

    The conditioning-set representation ``e_S`` is a sum-pool over the columns
    selected by the mask, as committed in PLAN §Phase 2. An attention-pool
    alternative is scheduled as an ablation in §Phase 6d.

    Parameters
    ----------
    embed_dim : int
        Dimensionality ``E`` of per-column embeddings.
    hidden_dim : Optional[int], default=None
        Hidden-layer dimension for the fusion MLP. Defaults to ``embed_dim // 2``.
    activation : str, default="gelu"
        Activation between the two linear layers.

    Notes
    -----
    The head computes a score for *every* feature in one forward pass — the
    natural substrate for the Phase 5 sklearn API call
    ``marginal_conditional_contributions(S=[...])``. Callers typically use
    only the scores at positions outside ``S``; positions inside ``S`` are
    still computed (the architecture is position-symmetric) and should be
    masked out downstream.

    Training schedule (PLAN §Phase 3): ``k = min(H, 16)`` random ``(i, S)``
    triples per dataset per step. Sampling is the training loop's job, not
    this module's.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        self.embed_dim = embed_dim
        hidden_dim = hidden_dim if hidden_dim is not None else max(1, embed_dim // 2)

        # Fusion MLP consumes concat(e_i, e_S) of dimension 2 * embed_dim.
        self.fc1 = nn.Linear(2 * embed_dim, hidden_dim)
        self.act = _build_activation(activation)
        self.fc2 = nn.Linear(hidden_dim, 1)

        _small_init_final_layer(self.fc2)

    def forward(self, column_embeddings: Tensor, cond_mask: Tensor) -> Tensor:
        """Compute a conditional-contribution score for every feature.

        Parameters
        ----------
        column_embeddings : Tensor
            Per-column embeddings of shape ``(B, H, E)``.
        cond_mask : Tensor
            Boolean mask of shape ``(B, H)``. ``True`` marks features in the
            conditioning set ``S``; ``False`` marks features not in ``S``.
            An all-``False`` row corresponds to ``S = ∅`` — ``e_S`` is then
            the zero vector and the head reduces to a function of ``e_i``
            alone.

        Returns
        -------
        Tensor
            Per-feature scores of shape ``(B, H)``.
        """
        if column_embeddings.dim() != 3:
            raise ValueError(
                f"Expected column_embeddings of shape (B, H, E), got {tuple(column_embeddings.shape)}"
            )
        if cond_mask.dtype != torch.bool:
            raise TypeError(f"cond_mask must be torch.bool, got {cond_mask.dtype}")
        if cond_mask.shape != column_embeddings.shape[:2]:
            raise ValueError(
                f"cond_mask shape {tuple(cond_mask.shape)} must match "
                f"column_embeddings.shape[:2] {tuple(column_embeddings.shape[:2])}"
            )

        mask = cond_mask.to(column_embeddings.dtype).unsqueeze(-1)   # (B, H, 1)
        e_S = (column_embeddings * mask).sum(dim=1)                  # (B, E)

        H = column_embeddings.shape[1]
        e_S_expanded = e_S.unsqueeze(1).expand(-1, H, -1)            # (B, H, E)
        fused = torch.cat([column_embeddings, e_S_expanded], dim=-1) # (B, H, 2E)

        x = self.act(self.fc1(fused))                                # (B, H, hidden)
        return self.fc2(x).squeeze(-1)                               # (B, H)
