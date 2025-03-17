from __future__ import annotations
from typing import List, Optional
from collections import OrderedDict

import torch
from torch import nn, Tensor

from .layers import SkippableLinear
from .encoders import SetTransformer
from .inference import InferenceManager
from ..config import MgrConfig


class ColEmbedding(nn.Module):
    """Distribution-aware column-wise embedding.

    This module maps each scalar cell in a column to a high-dimensional embedding while
    capturing statistical regularities within the column. Unlike traditional approaches
    that use separate embedding layers per column, it employs a shared set transformer
    to process all features.

    ColEmbedding operates as follows:
    1. Each scalar cell is first linearly projected into the embedding dimension
    2. The set transformer generates distribution-aware weights and biases for each column
    3. The final column embeddings are computed as: column * weights + biases

    Parameters
    ----------
    embed_dim : int
        Embedding dimension

    num_blocks : int
        Number of induced self-attention blocks used in the set transformer

    nhead : int
        Number of attention heads of the set transformer

    dim_feedforward : int
        Dimension of the feedforward network of the set transformer

    num_inds : int
        Number of inducing points used in self-attention blocks of the set transformer

    dropout : float, default=0.0
        Dropout probability used in the set transformer

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)

    reserve_cls_tokens : int, default=4
        Number of slots to reserve for CLS tokens to avoid concatenation
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        reserve_cls_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.reserve_cls_tokens = reserve_cls_tokens
        self.in_linear = SkippableLinear(1, embed_dim)

        self.tf_col = SetTransformer(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_inds=num_inds,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

        self.out_w = SkippableLinear(embed_dim, embed_dim)
        self.ln_w = nn.LayerNorm(embed_dim) if norm_first else nn.Identity()

        self.out_b = SkippableLinear(embed_dim, embed_dim)
        self.ln_b = nn.LayerNorm(embed_dim) if norm_first else nn.Identity()

        self.inference_mgr = InferenceManager(enc_name="tf_col", out_dim=embed_dim)

    @staticmethod
    def map_feature_shuffle(reference_pattern: List[int], other_pattern: List[int]) -> List[int]:
        """Map feature shuffle pattern from the reference table to another table.

        Parameters
        ----------
        reference_pattern : List[int]
            The shuffle pattern of features in a reference table w.r.t. the original table

        other_pattern : List[int]
            The shuffle pattern of features in another table w.r.t. the original table

        Returns
        -------
        List[int]
            A mapping from the reference table's ordering to another table's ordering
        """

        orig_to_other = {feature: idx for idx, feature in enumerate(other_pattern)}
        mapping = [orig_to_other[feature] for feature in reference_pattern]

        return mapping

    def _compute_embeddings(self, features: Tensor, train_size: Optional[int] = None) -> Tensor:
        """Feature embedding using a shared set transformer

        Parameters
        ----------
        features : Tensor
            Input features of shape (N, T, 1) where:
             - N is the total number of features across all tables
             - T is the number of samples (rows)

        train_size : Optional[int], default=None
            Position to split the input into training and test data. When provided,
            inducing points will only attend to training data (positions < train_size)
            in the set transformer to prevent information leakage from test data.

        Returns
        -------
        Tensor
            Embeddings of shape (N, T, E) where E is the embedding dimension
        """

        src = self.in_linear(features)  # (N, T, 1) -> (N, T, E)
        src = self.tf_col(src, train_size)
        weights = self.ln_w(self.out_w(src))  # (N, T, E)
        biases = self.ln_b(self.out_b(src))  # (N, T, E)
        embeddings = features * weights + biases

        return embeddings

    def forward(
        self,
        X: Tensor,
        train_size: Optional[int] = None,
        feature_shuffles: Optional[List[List[int]]] = None,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        """Transform input table into embeddings.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)

        train_size : Optional[int], default=None
            Position to split the input into training and test data. When provided,
            inducing points will only attend to training data (positions < train_size)
            in the set transformer to prevent information leakage from test data.

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, embeddings are computed once and then shuffled accordingly.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager

        Returns
        -------
        Tensor
            Embeddings of shape (B, T, H+C, E) where:
             - C is the number of class tokens
             - E is embedding dimension
        """
        # Configure inference parameters
        if mgr_config is None:
            mgr_config = MgrConfig(
                min_batch_size=1,
                safety_factor=0.8,
                offload="auto",
                auto_offload_pct=0.5,
                device=None,
                use_amp=True,
                verbose=False,
            )
        self.inference_mgr.configure(**mgr_config)

        if feature_shuffles is None:
            # Processing all tables
            if self.reserve_cls_tokens > 0:
                # Pad with -100.0 to mark inputs that should be skipped in SkippableLinear and SetTransformer
                X = nn.functional.pad(X, (self.reserve_cls_tokens, 0), value=-100.0)

            features = X.transpose(1, 2).unsqueeze(-1)  # (B, H+C, T, 1)
            embeddings = self.inference_mgr(
                self._compute_embeddings, inputs=OrderedDict([("features", features), ("train_size", train_size)])
            )  # (B, H+C, T, E)
        else:
            B = X.shape[0]
            # Process only the first table and then shuffle features for other tables
            first_table = X[0]
            if self.reserve_cls_tokens > 0:
                # Pad with -100.0 to mark inputs that should be skipped in SkippableLinear and SetTransformer
                first_table = nn.functional.pad(first_table, (self.reserve_cls_tokens, 0), value=-100.0)

            features = first_table.transpose(0, 1).unsqueeze(-1)  # (H+C, T, 1)
            first_embeddings = self.inference_mgr(
                self._compute_embeddings,
                inputs=OrderedDict([("features", features), ("train_size", train_size)]),
                output_repeat=B,
            )  # (H+C, T, E)

            # Apply shuffles for tables after the first one
            embeddings = first_embeddings.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H+C, T, E)
            first_pattern = feature_shuffles[0]
            for i in range(1, B):
                mapping = self.map_feature_shuffle(first_pattern, feature_shuffles[i])
                if self.reserve_cls_tokens > 0:
                    mapping = [m + self.reserve_cls_tokens for m in mapping]
                    mapping = list(range(self.reserve_cls_tokens)) + mapping
                embeddings[i] = first_embeddings[mapping]

        return embeddings.transpose(1, 2)  # (B, T, H+C, E)
