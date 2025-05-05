from __future__ import annotations

from typing import Optional
from torch import nn, Tensor

from .layers import MultiheadAttentionBlock, InducedSelfAttentionBlock
from .rope import RotaryEmbedding


class Encoder(nn.Module):
    """Stack of multihead attention blocks.

    Parameters
    ----------
    num_blocks : int
        Number of multihead attention blocks in the stack

    d_model : int
        Model dimension

    nhead : int
        Number of attention heads and should be a divisor of d_model

    dim_feedforward : int
        Dimension of the feedforward network in each block

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)

    use_rope : bool, default=False
        Whether to use rotary positional encoding

    rope_base : int, default=100000
        A base scaling factor for rotary position encoding
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        use_rope: bool = False,
        rope_base: int = 100000,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.blocks = nn.ModuleList(
            [
                MultiheadAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                )
                for _ in range(num_blocks)
            ]
        )

        self.rope = RotaryEmbedding(dim=d_model // nhead, theta=rope_base) if use_rope else None

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor | int] = None,
    ) -> Tensor:
        """Process input through the stacked blocks.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model)

        key_padding_mask : Optional[Tensor], default=None
            Mask of shape (..., src_len) that identifies padding elements
            in the key sequence to be ignored:
              - For binary masks: True values indicate positions to ignore
              - For float masks: Values are directly added to attention scores

        attn_mask : Optional[Tensor | int], default=None
            Controls attention pattern in two possible ways:
            1. When provided as Tensor: Traditional mask preventing attention to certain positions
              - Shape: (tgt_len, src_len) or (..., num_heads, tgt_len, src_len)
            2. When provided as integer: Creates a split attention pattern where:
              - The first `attn_mask` tokens perform self-attention only (attend to themselves)
              - The remaining tokens attend only to the first `attn_mask` tokens

        Returns
        -------
        Tensor
            Output tensor of shape (..., seq_len, d_model)
        """
        out = src
        for block in self.blocks:
            out = block(q=out, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rope=self.rope)

        return out


class SetTransformer(nn.Module):
    """Stack of induced self-attention blocks.

    A set transformer uses induced self-attention mechanism to efficiently
    process variable-sized sets while maintaining permutation invariance.

    Parameters
    ----------
    num_blocks : int
        Number of induced self-attention blocks in the stack

    d_model : int
        Model dimension

    nhead : int
        Number of attention heads and should be a divisor of d_model

    dim_feedforward : int
        Dimension of the feedforward network in each block

    num_inds : int, default=16
        Number of inducing points used in self-attention blocks

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)

    References
    ----------
    .. [1] Lee et al. "Set Transformer: A Framework for Attention-based
           Permutation-Invariant Neural Networks", ICML 2019
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int = 16,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.blocks = nn.ModuleList(
            [
                InducedSelfAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    num_inds=num_inds,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        """Process input through the stacked blocks.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model)

        train_size : Optional[int], default=None
            Position to split the input into training and test data. When provided,
            inducing points will only attend to training data in the first attention
            stage of induced self-attention blocks to prevent information leakage.

        Returns
        -------
        Tensor
            Output tensor of shape (..., seq_len, d_model)
        """
        out = src
        for block in self.blocks:
            out = block(out, train_size)

        return out
