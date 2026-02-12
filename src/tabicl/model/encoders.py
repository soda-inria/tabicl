from __future__ import annotations

from typing import Optional, Union
from functools import partial

from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from .rope import RotaryEmbedding
from .layers import MultiheadAttentionBlock, InducedSelfAttentionBlock
from .kv_cache import KVCacheEntry, KVCache


class Encoder(nn.Module):
    """Stack of multihead attention blocks.

    Parameters
    ----------
    num_blocks : int
        Number of multihead attention blocks in the stack.

    d_model : int
        Model dimension.

    nhead : int
        Number of attention heads and should be a divisor of ``d_model``.

    dim_feedforward : int
        Dimension of the feedforward network in each block.

    dropout : float, default=0.0
        Dropout probability.

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable.

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward).

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers.

    use_rope : bool, default=False
        Whether to use rotary positional encoding.

    rope_base : int, default=100000
        A base scaling factor for rotary position encoding.

    rope_interleaved : bool, default=True
        If True, uses interleaved rotation where dimension pairs are (0,1), (2,3), etc.
        If False, uses non-interleaved rotation where the embedding is split into
        first half [0:d//2] and second half [d//2:d].

    ssmax : bool or str, default=False
        Type of scalable softmax to use.
        If True, equivalent to "qassmax-mlp-elementwise".
        If False, equivalent to "none".
        If a string, uses the specified scalable softmax type.
        Options include:

        - "none": No scaling applied.
        - "ssmax": :math:`q_{\\text{scaled}} = q \\cdot (s \\cdot \\log n)` where
          :math:`s` is a learnable per-head parameter.
        - "ssmax-mlp": Uses MLP to compute scaling factors based on sequence length.
        - "ssmax-mlp-elementwise": Elementwise scaling per head dimension using MLP.
        - "qassmax-mlp": Query-aware scaling:
          :math:`\\text{scale} = \\text{base\\_mlp}(\\log n) \\cdot (1 + \\tanh(\\text{query\\_mlp}(q)))`.
        - "qassmax-mlp-elementwise": Elementwise query-aware scaling.

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of
        additional computation.
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
        bias_free_ln: bool = False,
        use_rope: bool = False,
        rope_base: int = 100000,
        rope_interleaved: bool = True,
        ssmax: Union[bool, str] = False,
        recompute: bool = False,
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
                    bias_free_ln=bias_free_ln,
                    ssmax=ssmax,
                )
                for _ in range(num_blocks)
            ]
        )

        self.rope = (
            RotaryEmbedding(dim=d_model // nhead, theta=rope_base, interleaved=rope_interleaved) if use_rope else None
        )
        self.recompute = recompute

    def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        """Process input through the stacked blocks.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model).

        train_size : Optional[int], default=None
            Positive integer indicating the number of training samples.
            When provided, queries attend only to the first ``train_size``
            positions. Useful in the ICL transformer where only training
            samples serve as context.

        Returns
        -------
        Tensor
            Output tensor with same shape as ``src``.
        """
        out = src
        for block in self.blocks:
            if self.recompute:
                kwargs = {"train_size": train_size, "rope": self.rope}
                out = checkpoint(partial(block, **kwargs), out, use_reentrant=False)
            else:
                out = block(q=out, train_size=train_size, rope=self.rope)

        return out

    def forward_with_cache(
        self,
        src: Tensor,
        icl_cache: KVCache,
        train_size: Optional[int] = None,
        use_cache: bool = False,
        store_cache: bool = True,
    ) -> Tensor:
        """Process input through the stacked blocks with KV caching support.

        1. If ``store_cache=True``, this method processes the full sequence and
           stores K/V projections from training data (positions
           ``[0:train_size]``) at each layer.

        2. If ``use_cache=True``, this method assumes ``src`` only contains test
           data and uses cached K/V from training data for attention at each layer.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model).

        icl_cache : KVCache
            Cache object for storing/retrieving K/V projections per layer.

        train_size : Optional[int], default=None
            Positive integer indicating the number of training samples.
            When provided, queries attend only to the first ``train_size``
            positions.

        use_cache : bool, default=False
            Whether to use cached values to avoid redundant computation.

        store_cache : bool, default=True
            Whether to store computed values in cache.

        Returns
        -------
        Tensor
            Output tensor with same shape as ``src``.
        """

        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")

        if store_cache and train_size is None:
            raise ValueError("train_size must be provided when store_cache=True")

        out = src
        for layer_idx, block in enumerate(self.blocks):
            if use_cache:
                # When using cache, src is already test-only data; no train_size needed
                out = block(q=out, rope=self.rope, cached_kv=icl_cache.kv[layer_idx])
            else:
                # Compute K/V for training data and store in cache
                out, k_proj, v_proj = block(q=out, train_size=train_size, rope=self.rope, need_kv=True)
                icl_cache.kv[layer_idx] = KVCacheEntry(key=k_proj, value=v_proj)

        return out


class SetTransformer(nn.Module):
    """Stack of induced self-attention blocks.

    A set transformer uses induced self-attention mechanism to efficiently
    process variable-sized sets while maintaining permutation invariance.

    Parameters
    ----------
    num_blocks : int
        Number of induced self-attention blocks in the stack.

    d_model : int
        Model dimension.

    nhead : int
        Number of attention heads and should be a divisor of ``d_model``.

    dim_feedforward : int
        Dimension of the feedforward network in each block.

    num_inds : int, default=16
        Number of inducing points used in self-attention blocks.

    dropout : float, default=0.0
        Dropout probability.

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable.

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward).

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers.

    ssmax : bool or str, default=False
        Type of scalable softmax to use in attention. Note that only the first
        attention layer of the induced self-attention blocks uses SSMax.
        If True, equivalent to "qassmax-mlp-elementwise".
        If False, equivalent to "none".
        If a string, uses the specified scalable softmax type.
        Options include:

        - "none": No scaling applied.
        - "ssmax": :math:`q_{\\text{scaled}} = q \\cdot (s \\cdot \\log n)` where
          :math:`s` is a learnable per-head parameter.
        - "ssmax-mlp": Uses MLP to compute scaling factors based on sequence length.
        - "ssmax-mlp-elementwise": Elementwise scaling per head dimension using MLP.
        - "qassmax-mlp": Query-aware scaling:
          :math:`\\text{scale} = \\text{base\\_mlp}(\\log n) \\cdot (1 + \\tanh(\\text{query\\_mlp}(q)))`.
        - "qassmax-mlp-elementwise": Elementwise query-aware scaling.

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of
        additional computation.

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
        bias_free_ln: bool = False,
        ssmax: Union[bool, str] = False,
        recompute: bool = False,
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
                    bias_free_ln=bias_free_ln,
                    ssmax=ssmax,
                )
                for _ in range(num_blocks)
            ]
        )
        self.recompute = recompute

    def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        """Process input through the stacked blocks.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model).

        train_size : Optional[int], default=None
            Position to split the input into training and test data. When provided,
            inducing points will only attend to training data in the first attention
            stage of induced self-attention blocks to prevent information leakage.

        Returns
        -------
        Tensor
            Output tensor with same shape as ``src``.
        """
        out = src
        for block in self.blocks:
            if self.recompute:
                out = checkpoint(partial(block, train_size=train_size), out, use_reentrant=False)
            else:
                out = block(out, train_size)

        return out

    def forward_with_cache(
        self,
        src: Tensor,
        col_cache: KVCache,
        train_size: Optional[int] = None,
        use_cache: bool = False,
        store_cache: bool = True,
    ) -> Tensor:
        """Process input through the stacked ISAB blocks with KV caching support.

        Each block has two attention stages:

        1. Stage 1: Inducing points attend to training data, producing ``hidden``.
        2. Stage 2: Input attends to ``hidden``, producing the output.

        We cache the K/V projections of ``hidden`` for Stage 2, which allows test
        samples to reuse the cached K/V without recomputing Stage 1.

        If ``store_cache=True``, this method:

        - Runs Stage 1: inducing points attend to training data to produce ``hidden``.
        - Caches K/V projections of ``hidden`` for each block.
        - Runs Stage 2: all samples attend to ``hidden``.

        If ``use_cache=True``, this method:

        - Skips Stage 1 (uses cached K/V from ``hidden``).
        - Runs Stage 2: test samples attend to cached K/V.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model).

        col_cache : KVCache
            Cache object for storing/retrieving K/V projections of ``hidden``.

        train_size : Optional[int], default=None
            Position to split the input into training and test data. If storing
            cache, it must be provided to ensure the cache is populated with
            training data correctly. If using cache, it is ignored.

        use_cache : bool, default=False
            Whether to use cached values to avoid redundant computation.

        store_cache : bool, default=True
            Whether to store computed values in cache.

        Returns
        -------
        Tensor
            Output tensor with same shape as ``src``.
        """

        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")

        if store_cache and train_size is None:
            raise ValueError("train_size must be provided when store_cache=True")

        out = src
        for block_idx, block in enumerate(self.blocks):
            out = block.forward_with_cache(out, col_cache, block_idx, train_size, use_cache, store_cache)

        return out
