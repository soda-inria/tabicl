from __future__ import annotations

import torch
from torch import nn, Tensor
from .embedding import ColEmbedding
from .interaction import RowInteraction


class TabCompressor(nn.Module):
    """
    Column-wise + Row-wise part of TabICL, **without** ICLearning.
    Forward returns (B, T, d_enc) — a per-row compressed representation.
    """
    def __init__(
        self,
        *,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        col_nhead: int = 4,
        col_num_inds: int = 128,
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100_000,
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()

        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=col_num_blocks,
            nhead=col_nhead,
            num_inds=col_num_inds,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            reserve_cls_tokens=row_num_cls,
        )
        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=row_num_blocks,
            nhead=row_nhead,
            num_cls=row_num_cls,
            rope_base=row_rope_base,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

        self.d_enc = embed_dim * row_num_cls


    def forward(self, X: Tensor) -> Tensor:
        reps = self.row_interactor(self.col_embedder(X, train_size=None))
        return reps


class CompressorProjector(nn.Module):
    """A simple projector for the compressor.

    This is used to project the input to a lower-dimensional space.
    It is used in the PerFeatureTransformer to compress the input before
    passing it to the transformer.
    """

    def __init__(
            self,
            input_dim: int = 128,
            output_dim: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)
