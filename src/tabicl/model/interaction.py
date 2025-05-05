from __future__ import annotations

from typing import Optional
from collections import OrderedDict

import torch
from torch import nn, Tensor

from .encoders import Encoder
from .inference import InferenceManager
from .inference_config import MgrConfig


class RowInteraction(nn.Module):
    """Context-aware row-wise interaction.

    This module captures interactions between features within each row using a transformer
    encoder with rotary positional encoding. It prepends learnable class tokens to the
    learned feature embeddings and uses these tokens to aggregate information.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension

    num_blocks : int
        Number of blocks used in the encoder

    nhead : int
        Number of attention heads of the encoder

    dim_feedforward : int
        Dimension of the feedforward network of the encoder

    num_cls : int, default=4
        Number of learnable CLS tokens to prepend to the feature embeddings. The outputs
        of these CLS tokens are concatenated for the final representation per row

    rope_base : float, default=100000
        Base scaling factor for rotary position encoding

    dropout : float, default=0.0
        Dropout probability used in the encoder

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_cls: int = 4,
        rope_base: float = 100000,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cls = num_cls
        self.norm_first = norm_first

        self.tf_row = Encoder(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            use_rope=True,
            rope_base=rope_base,
        )

        self.cls_tokens = nn.Parameter(torch.empty(num_cls, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        self.out_ln = nn.LayerNorm(embed_dim) if norm_first else nn.Identity()

        self.inference_mgr = InferenceManager(enc_name="tf_row", out_dim=embed_dim * self.num_cls, out_no_seq=True)

    def _aggregate_embeddings(self, embeddings: Tensor, key_mask: Optional[Tensor] = None) -> Tensor:
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

        Returns
        -------
        Tensor
            Flattened class token outputs of shape (B*T, C*E)
        """

        outputs = self.tf_row(embeddings, key_padding_mask=key_mask)  # (B, T, H+C, E)
        cls_outputs = outputs[..., : self.num_cls, :].clone()  # (B, T, C, E)
        del outputs  # Delete intermediate outputs to save memory

        cls_outputs = self.out_ln(cls_outputs)

        return cls_outputs.flatten(-2)  # (B, T, C*E)

    def _train_forward(self, embeddings: Tensor, d: Optional[Tensor] = None) -> Tensor:
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

        Returns
        -------
        Tensor
            Row representations of shape (B, T, C*E) where C is the number of class tokens
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

        representations = self._aggregate_embeddings(embeddings, key_mask)  # (B, T, C*E)

        return representations  # (B, T, C*E)

    def _inference_forward(self, embeddings: Tensor, mgr_config: MgrConfig = None) -> Tensor:
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
            Configuration for InferenceManager

        Returns
        -------
        Tensor
            Row representations of shape (B, T, C*E) where C is the number of class tokens
        """
        # Configure inference parameters
        if mgr_config is None:
            mgr_config = MgrConfig(
                min_batch_size=1,
                safety_factor=0.8,
                offload=False,
                auto_offload_pct=0.5,
                device=None,
                use_amp=True,
                verbose=False,
            )
        self.inference_mgr.configure(**mgr_config)

        B, T = embeddings.shape[:2]
        cls_tokens = self.cls_tokens.expand(B, T, self.num_cls, self.embed_dim)
        embeddings[:, :, : self.num_cls] = cls_tokens.to(embeddings.device)
        representations = self.inference_mgr(
            self._aggregate_embeddings, inputs=OrderedDict([("embeddings", embeddings)])
        )

        return representations  # (B, T, C*E)

    def forward(self, embeddings: Tensor, d: Optional[Tensor] = None, mgr_config: MgrConfig = None) -> Tensor:
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

        Returns
        -------
        Tensor
            Row representations of shape (B, T, C*E) where C is the number of class tokens
        """

        if self.training:
            representations = self._train_forward(embeddings, d)
        else:
            representations = self._inference_forward(embeddings, mgr_config)

        return representations  # (B, T, C*E)
