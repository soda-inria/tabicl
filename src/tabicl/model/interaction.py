from __future__ import annotations

from typing import Optional, Tuple, Union
from functools import partial
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from .encoders import Encoder
from .inference import InferenceManager
from .inference_config import MgrConfig, InferenceConfig


class RowInteraction(nn.Module):
    """Context-aware row-wise interaction.

    This module captures interactions between features within each row using a transformer
    encoder with rotary positional encoding. It prepends learnable class tokens to the
    learned feature embeddings and uses these tokens to aggregate information.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.

    num_blocks : int
        Number of blocks used in the encoder.

    nhead : int
        Number of attention heads of the encoder.

    dim_feedforward : int
        Dimension of the feedforward network of the encoder.

    num_cls : int, default=4
        Number of learnable CLS tokens to prepend to the feature embeddings. The outputs
        of these CLS tokens are concatenated for the final representation per row.

    rope_base : float, default=100000
        Base scaling factor for rotary position encoding.

    rope_interleaved : bool, default=True
        If True, uses interleaved rotation where dimension pairs are (0,1), (2,3), etc.
        If False, uses non-interleaved rotation where the embedding is split into
        first half [0:d//2] and second half [d//2:d].

    dropout : float, default=0.0
        Dropout probability used in the encoder.

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable.

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward).

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers.

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of additional computation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_cls: int = 4,
        rope_base: float = 100000,
        rope_interleaved: bool = True,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        recompute: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_cls = num_cls
        self.norm_first = norm_first
        self.recompute = recompute

        self.tf_row = Encoder(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            use_rope=True,
            rope_base=rope_base,
            rope_interleaved=rope_interleaved,
            recompute=recompute,
        )

        self.cls_tokens = nn.Parameter(torch.empty(num_cls, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        self.out_ln = nn.LayerNorm(embed_dim, bias=not bias_free_ln) if norm_first else nn.Identity()
        self.inference_mgr = InferenceManager(enc_name="tf_row", out_dim=embed_dim * self.num_cls, out_no_seq=True)

    def _aggregate_embeddings(
        self,
        embeddings: Tensor,
        key_mask: Optional[Tensor] = None,
        return_features: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Process a batch of rows through a transformer encoder.

        This method:

        1. Processes embeddings through the transformer
        2. Extracts only the class token representations and applies normalization if pre-norm
        3. Concatenates the class tokens into a single vector per row

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings of shape (B, T, H+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features
             - C is the number of class tokens
             - E is the embedding dimension

        key_mask : Optional[Tensor], default=None
            Boolean mask of shape (B, T, H+C) where True indicates positions
            to ignore during attention (empty feature slots).

        return_features : bool, default=False
            If True, also return the per-feature outputs from the last block.
            In this path the last block runs over the full sequence (not only
            CLS queries), so per-column token outputs are available. The CLS
            outputs from this path are mathematically identical to the default
            path but may differ bit-wise (different matmul shapes). Use the
            default path when bit-exactness is required.

        Returns
        -------
        Tensor or tuple of Tensors
            If ``return_features`` is False: flattened class token outputs of
            shape (B, T, C*E).

            If ``return_features`` is True: tuple ``(cls_flat, feature_outputs)``
            where ``cls_flat`` has shape (B, T, C*E) and ``feature_outputs``
            has shape (B, T, H, E).
        """
        rope = self.tf_row.rope

        # Process all blocks except the last
        if self.recompute:
            for block in self.tf_row.blocks[:-1]:
                embeddings = checkpoint(
                    partial(block, key_padding_mask=key_mask, rope=rope), embeddings, use_reentrant=False
                )
        else:
            for block in self.tf_row.blocks[:-1]:
                embeddings = block(embeddings, key_padding_mask=key_mask, rope=rope)

        last_block = self.tf_row.blocks[-1]

        if return_features:
            # Full-sequence last block: q = k = v = embeddings. Keeps per-feature
            # outputs available alongside CLS outputs.
            if self.recompute:
                full_out = checkpoint(
                    lambda emb: last_block(q=emb, k=emb, v=emb, key_padding_mask=key_mask, rope=rope),
                    embeddings,
                    use_reentrant=False,
                )
            else:
                full_out = last_block(
                    q=embeddings, k=embeddings, v=embeddings, key_padding_mask=key_mask, rope=rope
                )
            del embeddings
            full_out = self.out_ln(full_out)                       # (B, T, H+C, E)
            cls_outputs = full_out[..., : self.num_cls, :]         # (B, T, C, E)
            feature_outputs = full_out[..., self.num_cls :, :]     # (B, T, H, E)
            return cls_outputs.flatten(-2), feature_outputs

        # Default path: q = CLS tokens only, k/v = full sequence.
        if self.recompute:
            cls_outputs = checkpoint(
                lambda emb: last_block(
                    q=emb[..., : self.num_cls, :], k=emb, v=emb, key_padding_mask=key_mask, rope=rope
                ),
                embeddings,
                use_reentrant=False,
            )
        else:
            cls_outputs = last_block(
                q=embeddings[..., : self.num_cls, :], k=embeddings, v=embeddings, key_padding_mask=key_mask, rope=rope
            )
        del embeddings
        cls_outputs = self.out_ln(cls_outputs)

        return cls_outputs.flatten(-2)  # (B, T, C*E)

    def _train_forward(
        self,
        embeddings: Tensor,
        d: Optional[Tensor] = None,
        return_features: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Transform feature embeddings into row representations for training.

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings of shape (B, T, H+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features
             - C is the number of class tokens
             - E is the embedding dimension

        d : Optional[Tensor], default=None
            The number of features per dataset. Used only in training mode.

        return_features : bool, default=False
            If True, also return per-feature last-block outputs of shape
            (B, T, H, E) alongside the row representations.

        Returns
        -------
        Tensor or tuple of Tensors
            Row representations of shape (B, T, C*E); optionally a second
            tensor ``feature_outputs`` of shape (B, T, H, E).
        """

        B, T, HC, E = embeddings.shape
        device = embeddings.device

        cls_tokens = self.cls_tokens.expand(B, T, self.num_cls, self.embed_dim)
        embeddings[:, :, : self.num_cls] = cls_tokens.to(embeddings.device)

        # Create mask to prevent from attending to empty features
        if d is None:
            key_mask = None
        else:
            d = d + self.num_cls
            indices = torch.arange(HC, device=device).view(1, 1, HC).expand(B, T, HC)
            key_mask = indices >= d.view(B, 1, 1)  # (B, T, HC)

        return self._aggregate_embeddings(embeddings, key_mask, return_features=return_features)

    def _inference_forward(
        self,
        embeddings: Tensor,
        mgr_config: MgrConfig = None,
        return_features: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Transform feature embeddings into row representations for inference.

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings of shape (B, T, H+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features
             - C is the number of class tokens
             - E is the embedding dimension

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager.

        return_features : bool, default=False
            If True, return per-feature last-block outputs alongside row
            representations. This path bypasses the InferenceManager's
            chunking — the whole (B, T, H+C, E) tensor is processed in one
            call. Memory cost scales linearly with ``T * H``.

        Returns
        -------
        Tensor or tuple of Tensors
            Row representations of shape (B, T, C*E); optionally a second
            tensor ``feature_outputs`` of shape (B, T, H, E).
        """
        # Configure inference parameters
        if mgr_config is None:
            mgr_config = InferenceConfig().ROW_CONFIG
        self.inference_mgr.configure(**mgr_config)

        B, T = embeddings.shape[:2]
        cls_tokens = self.cls_tokens.expand(B, T, self.num_cls, self.embed_dim)
        embeddings[:, :, : self.num_cls] = cls_tokens.to(embeddings.device)

        if return_features:
            # Bypass InferenceManager chunking: run the full aggregation in
            # one shot so the per-feature output tensor is contiguous.
            return self._aggregate_embeddings(embeddings, key_mask=None, return_features=True)

        representations = self.inference_mgr(
            self._aggregate_embeddings, inputs=OrderedDict([("embeddings", embeddings)])
        )

        return representations  # (B, T, C*E)

    def forward(
        self,
        embeddings: Tensor,
        d: Optional[Tensor] = None,
        mgr_config: MgrConfig = None,
        return_features: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Transform feature embeddings into row representations.

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings of shape (B, T, H+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features
             - C is the number of class tokens
             - E is the embedding dimension

        d : Optional[Tensor], default=None
            The number of features per dataset. Used only in training mode.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager. Used only in inference mode.

        return_features : bool, default=False
            If True, return ``(representations, feature_outputs)`` where
            ``feature_outputs`` has shape (B, T, H, E). See
            :meth:`_aggregate_embeddings` for details.

        Returns
        -------
        Tensor or tuple of Tensors
            Row representations of shape (B, T, C*E); optionally a second
            tensor ``feature_outputs`` of shape (B, T, H, E).
        """

        if self.training:
            return self._train_forward(embeddings, d, return_features=return_features)
        return self._inference_forward(embeddings, mgr_config, return_features=return_features)
