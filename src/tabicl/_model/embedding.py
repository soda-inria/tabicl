from __future__ import annotations

from typing import List, Optional, Union, Literal
from collections import OrderedDict
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .layers import SkippableLinear, OneHotAndLinear
from .encoders import SetTransformer
from .kv_cache import KVCache
from .inference import InferenceManager
from .inference_config import MgrConfig, InferenceConfig


class ColEmbedding(nn.Module):
    """Distribution-aware column-wise embedding.

    This module maps each scalar cell in a column to a high-dimensional embedding while
    capturing statistical regularities within the column. Unlike traditional approaches
    that use separate embedding layers per column, it employs a shared set transformer
    to process all features.

    ColEmbedding operates in two modes depending on the `affine` parameter:

    When ``affine=True``:

    1. Each scalar cell is first linearly projected into the embedding dimension
    2. The set transformer generates distribution-aware weights and biases for each column
    3. The final column embeddings are computed as: :math:`\\text{column} \\times W + b`

    When ``affine=False``:

    1. Each scalar cell is first linearly projected into the embedding dimension
    2. The set transformer processes the projected features
    3. The final column embeddings are directly the set transformer's output

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.

    num_blocks : int
        Number of induced self-attention blocks used in the set transformer.

    nhead : int
        Number of attention heads of the set transformer.

    dim_feedforward : int
        Dimension of the feedforward network of the set transformer.

    num_inds : int
        Number of inducing points used in self-attention blocks of the set transformer.

    dropout : float, default=0.0
        Dropout probability used in the set transformer.

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable.

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward).

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers.

    affine : bool, default=True
        If True, computes embeddings as: :math:`\\text{features} \\times W + b`.
        If False, directly uses the set transformer output as embeddings.

    feature_group : bool or Literal["same", "valid"], default=False
        Feature grouping mode:
        - False: No grouping
        - True or "same": Group through circular permutation (output has same number of groups as features)
        - "valid": Group through padding and reshaping (output may have fewer groups)

    feature_group_size : int, default=3
        Number of features per group when feature_group is enabled.

    target_aware : bool, default=False
        If True, incorporates target information into embeddings during training.
        The target values are embedded into the same embedding space and added to each feature's embeddings,
        enabling the model to be aware of target information.

    max_classes : int, default=10
        Number of classes for classification task. If 0, assumes regression task.

    reserve_cls_tokens : int, default=4
        Number of slots to reserve for CLS tokens to avoid concatenation.

    ssmax : bool or str, default=False
        Type of scalable softmax to use in attention. Note that only the first attention layer of
        the induced self-attention blocks uses SSMax.
        If True, equivalent to "qassmax-mlp-elementwise".
        If False, equivalent to "none".
        If a string, uses the specified scalable softmax type.
        Options include:
            - "none": No scaling applied
            - "ssmax": :math:`q_{\\text{scaled}} = q \\cdot (s \\cdot \\log n)` where s is learnable per-head parameter
            - "ssmax-mlp": Uses MLP to compute scaling factors based on sequence length
            - "ssmax-mlp-elementwise": Elementwise scaling per head dimension using MLP
            - "qassmax-mlp": Query-aware scaling: :math:`\\text{scale} = \\text{base\\_mlp}(\\log n) \\cdot (1 + \\tanh(\\text{query\\_mlp}(q)))`
            - "qassmax-mlp-elementwise": Elementwise query-aware scaling

    mixed_radix_ensemble : bool, default=True
        If True, enables mixed-radix ensembling for many-class classification. Only effective
        if target_aware=True and max_classes > 0.

        When enabled and num_classes > max_classes, labels are decomposed into D digits
        via mixed-radix representation with balanced bases :math:`[k_0, \\ldots, k_{D-1}]` where each
        :math:`k_i \\leq` max_classes and :math:`\\prod_i k_i \\geq` num_classes. Each digit defines a coarser
        grouping of the original classes. The set transformer is run once per digit, and the
        final outputs are averaged across all digits.

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of additional computation.
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
        bias_free_ln: bool = False,
        affine: bool = True,
        feature_group: Union[bool, Literal["same", "valid"]] = False,
        feature_group_size: int = 3,
        target_aware: bool = False,
        max_classes: int = 10,
        reserve_cls_tokens: int = 4,
        ssmax: Union[bool, str] = False,
        mixed_radix_ensemble: bool = True,
        recompute: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.reserve_cls_tokens = reserve_cls_tokens
        self.feature_group = feature_group
        self.feature_group_size = feature_group_size
        self.target_aware = target_aware
        self.max_classes = max_classes
        self.affine = affine
        self.mixed_radix_ensemble = mixed_radix_ensemble
        self.in_linear = SkippableLinear(feature_group_size if feature_group else 1, embed_dim)

        self.tf_col = SetTransformer(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_inds=num_inds,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            ssmax=ssmax,
            recompute=recompute,
        )

        if target_aware:
            if max_classes > 0:  # Classification
                self.y_encoder = OneHotAndLinear(max_classes, embed_dim)
            else:  # Regression
                self.y_encoder = nn.Linear(1, embed_dim)

        if affine:
            self.out_w = SkippableLinear(embed_dim, embed_dim)
            self.ln_w = nn.LayerNorm(embed_dim, bias=not bias_free_ln) if norm_first else nn.Identity()

            self.out_b = SkippableLinear(embed_dim, embed_dim)
            self.ln_b = nn.LayerNorm(embed_dim, bias=not bias_free_ln) if norm_first else nn.Identity()

        self.inference_mgr = InferenceManager(enc_name="tf_col", out_dim=embed_dim)

    @staticmethod
    def map_feature_shuffle(reference_pattern: List[int], other_pattern: List[int]) -> List[int]:
        """Map feature shuffle pattern from the reference table to another table.

        Parameters
        ----------
        reference_pattern : List[int]
            The shuffle pattern of features in a reference table w.r.t. the original table.

        other_pattern : List[int]
            The shuffle pattern of features in another table w.r.t. the original table.

        Returns
        -------
        List[int]
            A mapping from the reference table's ordering to another table's ordering.
        """

        orig_to_other = {feature: idx for idx, feature in enumerate(other_pattern)}
        mapping = [orig_to_other[feature] for feature in reference_pattern]

        return mapping

    def feature_grouping(self, X: Tensor) -> Tensor:
        """Group features into fixed-size groups.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)

        Returns
        -------
        Tensor
            Grouped tensor of shape (B, T, G, feature_group_size) where G is the number of groups.
        """
        if not self.feature_group:
            return X.unsqueeze(-1)  # (B, T, H, 1)

        B, T, H = X.shape
        size = self.feature_group_size

        # Determine grouping mode
        mode = "same" if self.feature_group is True else self.feature_group

        if mode == "same":
            # Group through circular permutation
            idxs = torch.arange(H, dtype=torch.long, device=X.device)
            X = torch.stack([X[:, :, (idxs + 2**i) % H] for i in range(size)], dim=-1)
        else:
            # Group through padding and reshaping
            x_pad_cols = (size - H % size) % size
            if x_pad_cols > 0:
                X = F.pad(X, (0, x_pad_cols), value=0)
            X = X.reshape(B, T, -1, size)

        return X  # (B, T, G, size)

    def _compute_mixed_radix_bases(self, num_classes: int) -> List[int]:
        """Compute balanced bases for mixed-radix decomposition.

        For :math:`C >` max_classes, computes balanced bases :math:`[k_0, \\ldots, k_{D-1}]` where each
        :math:`k_i \\leq` max_classes and :math:`\\prod_i k_i \\geq C`. Each digit in the mixed-radix
        representation defines a coarser grouping of the original classes.

        For example, with num_classes=25 and max_classes=10, this method returns
        bases=[5, 5], i.e., both digits have 5 possible values.

        Detailed mapping for 25 classes with bases=[5, 5]:
        Each class label y is decomposed as (y // 5, y % 5) in mixed-radix representation:

            Class | Digit 0 (y//5) | Digit 1 (y%5)
            ------|----------------|---------------
              0   |       0        |       0
              1   |       0        |       1
              2   |       0        |       2
              3   |       0        |       3
              4   |       0        |       4
              5   |       1        |       0
              6   |       1        |       1
              ...
             24   |       4        |       4

        Each digit provides a different view (coarser grouping) of the original classes:
        - Digit 0 groups: {0-4}, {5-9}, {10-14}, {15-19}, {20-24}
        - Digit 1 groups: {0,5,10,15,20}, {1,6,11,16,21}, {2,7,12,17,22}, {3,8,13,18,23}, {4,9,14,19,24}
        Both digits have exactly 5 distinct values {0,1,2,3,4}.

        Parameters
        ----------
        num_classes : int
            Total number of unique classes.

        Returns
        -------
        List[int]
            List of bases :math:`[k_0, \\ldots, k_{D-1}]`, all <= max_classes, product >= num_classes.
        """
        if num_classes <= self.max_classes:
            return [num_classes]

        D = math.ceil(math.log(num_classes) / math.log(self.max_classes))
        k = math.ceil(num_classes ** (1.0 / D))
        k = min(k, self.max_classes)

        # If product is insufficient, increment bases one by one
        bases = [k] * D
        product = k**D
        idx = 0
        while product < num_classes and idx < D:
            if bases[idx] < self.max_classes:
                product = product // bases[idx] * (bases[idx] + 1)
                bases[idx] += 1
            idx += 1

        return bases

    def _extract_mixed_radix_digit(self, y: Tensor, digit_idx: int, bases: List[int]) -> Tensor:
        """Extract a specific digit from the mixed-radix representation of labels.

        For bases :math:`[k_0, k_1, \\ldots, k_{D-1}]`:

        .. math::

            \\text{Digit } 0 &: y \\mathbin{//} (k_1 \\cdot k_2 \\cdots k_{D-1}) \\mod k_0 \\\\
            \\text{Digit } 1 &: y \\mathbin{//} (k_2 \\cdot k_3 \\cdots k_{D-1}) \\mod k_1 \\\\
            &\\;\\vdots \\\\
            \\text{Digit } D{-}1 &: y \\mod k_{D-1}

        Parameters
        ----------
        y : Tensor
            Original class labels of shape (...).

        digit_idx : int
            Digit index.

        bases : List[int]
            List of bases :math:`[k_0, \\ldots, k_{D-1}]`.

        Returns
        -------
        Tensor
            Digit values in [0, bases[digit_idx]-1], same shape as y.
        """
        # Compute divisor: product of all bases after this digit
        divisor = 1
        for i in range(digit_idx + 1, len(bases)):
            divisor *= bases[i]

        return (y.long() // divisor) % bases[digit_idx]

    def _compute_embeddings(
        self, features: Tensor, train_size: int, y_train: Optional[Tensor] = None, embed_with_test: bool = False
    ) -> Tensor:
        """Feature embedding using a shared set transformer.

        Parameters
        ----------
        features : Tensor
            Input features of shape (..., T, in_dim) where:
             - ... represents arbitrary batch dimensions
             - T is the number of samples (rows)
             - in_dim is 1 (no grouping) or feature_group_size (with grouping)

        train_size : int
            Position to split the input into training and test data.
            Inducing points will only attend to training data (positions < train_size)
            in the set transformer to prevent information leakage from test data.

        y_train : Optional[Tensor], default=None
            Target values of shape (..., train_size). Required when target_aware=True.
            For classification, contains class labels. For regression, contains target values.

        embed_with_test : bool, default=False
            If True, inducing points attend to all samples (train + test).
            If False, inducing points only attend to training samples.

        Returns
        -------
        Tensor
            Embeddings of shape (..., T, E) where E is the embedding dimension.
        """

        src = self.in_linear(features)  # (..., T, in_dim) -> (..., T, E)

        if not self.target_aware:
            src = self.tf_col(src, train_size=None if embed_with_test else train_size)
        else:
            assert y_train is not None, "y_train must be provided when target_aware=True."

            # Determine if mixed-radix ensemble is needed
            num_classes = int(y_train.max().item()) + 1
            needs_mixed_radix = self.max_classes > 0 and num_classes > self.max_classes

            if not needs_mixed_radix:
                # Standard target-aware embedding
                if self.max_classes > 0:
                    y_emb = self.y_encoder(y_train.float())
                else:
                    y_emb = self.y_encoder(y_train.unsqueeze(-1))
                src[..., :train_size, :] = src[..., :train_size, :] + y_emb
                src = self.tf_col(src, train_size=None if embed_with_test else train_size)
            else:
                # Mixed-radix ensembling for many-class classification
                if not self.mixed_radix_ensemble:
                    raise ValueError(
                        f"Number of classes ({num_classes}) exceeds max_classes ({self.max_classes}). "
                        f"Set mixed_radix_ensemble=True to enable mixed-radix ensembling."
                    )

                # Compute balanced bases for mixed-radix decomposition
                bases = self._compute_mixed_radix_bases(num_classes)
                num_digits = len(bases)
                src_accum = torch.zeros_like(src)
                src_with_y = src.clone()

                # Run the set transformer for each digit, accumulate, and average
                for digit_idx in range(num_digits):
                    y_digit = self._extract_mixed_radix_digit(y_train, digit_idx, bases)
                    y_emb = self.y_encoder(y_digit.float())
                    src_with_y[..., :train_size, :] = src[..., :train_size, :] + y_emb
                    src_accum = src_accum + self.tf_col(src_with_y, train_size=None if embed_with_test else train_size)

                src = src_accum / num_digits

        if self.affine:
            weights = self.ln_w(self.out_w(src))
            biases = self.ln_b(self.out_b(src))
            embeddings = features * weights + biases
        else:
            embeddings = src

        return embeddings

    def _train_forward(
        self, X: Tensor, y_train: Tensor, d: Optional[Tensor] = None, embed_with_test: bool = False
    ) -> Tensor:
        """Transform input table into embeddings for training.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)

        y_train : Tensor
            Target values for training samples of shape (B, train_size).
            Used only for target-aware embedding.

        d : Optional[Tensor], default=None
            The number of features per dataset of shape (B,).
            Only supported when feature grouping is disabled.

        embed_with_test : bool, default=False
            If True, inducing points attend to all samples (train + test).
            If False, inducing points only attend to training samples.

        Returns
        -------
        Tensor
            Embeddings of shape (B, T, G+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - G is the number of feature groups
             - C is the number of class tokens
             - E is embedding dimension.
        """
        if self.feature_group:
            assert d is None, "d is not supported when feature grouping is enabled."
            return self._train_forward_with_feature_group(X, y_train, embed_with_test)
        else:
            return self._train_forward_without_feature_group(X, y_train, d, embed_with_test)

    def _train_forward_with_feature_group(self, X: Tensor, y_train: Tensor, embed_with_test: bool) -> Tensor:
        """Training path when feature grouping is enabled."""
        train_size = y_train.shape[1]
        X = self.feature_grouping(X)  # (B, T, G, group_size)
        if self.reserve_cls_tokens > 0:
            X = F.pad(X, (0, 0, self.reserve_cls_tokens, 0), value=-100.0)

        features = X.transpose(1, 2)  # (B, G+C, T, group_size)
        if self.target_aware:
            assert y_train is not None, "y_train must be provided when target_aware=True."
            y_train = y_train.unsqueeze(1).expand(-1, features.shape[1], -1)

        embeddings = self._compute_embeddings(features, train_size, y_train, embed_with_test)
        return embeddings.transpose(1, 2)  # (B, T, G+C, E)

    def _train_forward_without_feature_group(
        self, X: Tensor, y_train: Tensor, d: Optional[Tensor], embed_with_test: bool
    ) -> Tensor:
        """Training path without feature grouping, supporting variable number of features per table."""
        train_size = y_train.shape[1]

        if self.reserve_cls_tokens > 0:
            X = F.pad(X, (self.reserve_cls_tokens, 0), value=-100.0)

        if d is None:
            features = X.transpose(1, 2).unsqueeze(-1)  # (B, H+C, T, 1)
            if self.target_aware:
                assert y_train is not None, "y_train must be provided when target_aware=True."
                y_train = y_train.unsqueeze(1).expand(-1, features.shape[1], -1)
            embeddings = self._compute_embeddings(features, train_size, y_train, embed_with_test)
        else:
            if self.reserve_cls_tokens > 0:
                d = d + self.reserve_cls_tokens

            B, T, HC = X.shape
            X = X.transpose(1, 2)  # (B, H+C, T)

            # Create mask to extract non-empty features
            indices = torch.arange(HC, device=X.device).unsqueeze(0).expand(B, HC)
            mask = indices < d.unsqueeze(1)  # (B, H+C)
            features = X[mask].unsqueeze(-1)  # (N, T, 1) where N = sum(d)

            if self.target_aware:
                assert y_train is not None, "y_train must be provided when target_aware=True."
                # Expand y_train for each non-empty feature: (B, train_size) -> (N, train_size)
                y_train = y_train.unsqueeze(1).expand(-1, HC, -1)  # (B, H+C, train_size)
                y_train = y_train[mask]  # (N, train_size)

            effective_embeddings = self._compute_embeddings(features, train_size, y_train, embed_with_test)

            # Fill computed embeddings back into full tensor
            embeddings = torch.zeros(B, HC, T, self.embed_dim, device=X.device, dtype=effective_embeddings.dtype)
            embeddings[mask] = effective_embeddings

        return embeddings.transpose(1, 2)  # (B, T, H+C, E)

    def _inference_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        embed_with_test: bool = False,
        feature_shuffles: Optional[List[List[int]]] = None,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        """Transform input table into embeddings for inference.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)

        y_train : Tensor
            Target values for training samples of shape (B, train_size).
            Used only for target-aware embedding.

        embed_with_test : bool, default=False
            If True, inducing points attend to all samples (train + test).
            If False, inducing points only attend to training samples.

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch. It is only
            effective when feature grouping is disabled. When provided, indicates that
            X contains the same table with different feature orders. In this case,
            embeddings are computed once and then shuffled accordingly.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager.

        Returns
        -------
        Tensor
            Embeddings of shape (B, T, G+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - G is the number of feature groups
             - C is the number of class tokens
             - E is embedding dimension.
        """
        # Configure inference parameters
        if mgr_config is None:
            mgr_config = InferenceConfig().COL_CONFIG
        self.inference_mgr.configure(**mgr_config)

        train_size = y_train.shape[1]
        if self.feature_group:
            embeddings = self._inference_with_feature_group(X, y_train, train_size, embed_with_test)
        else:
            embeddings = self._inference_without_feature_group(
                X, y_train, train_size, embed_with_test, feature_shuffles
            )

        return embeddings.transpose(1, 2)  # (B, T, G+C, E)

    def _inference_with_feature_group(
        self, X: Tensor, y_train: Tensor, train_size: int, embed_with_test: bool
    ) -> Tensor:
        """Inference path when feature grouping is enabled."""

        X = self.feature_grouping(X)  # (B, T, G, group_size)
        if self.reserve_cls_tokens > 0:
            X = F.pad(X, (0, 0, self.reserve_cls_tokens, 0), value=-100.0)

        features = X.transpose(1, 2)  # (B, G+C, T, group_size)
        if self.target_aware:
            assert y_train is not None, "y_train must be provided when target_aware=True."
            y_train = y_train.unsqueeze(1).expand(-1, features.shape[1], -1)
        else:
            y_train = None

        return self.inference_mgr(
            self._compute_embeddings,
            inputs=OrderedDict(
                [
                    ("features", features),
                    ("train_size", train_size),
                    ("y_train", y_train),
                    ("embed_with_test", embed_with_test),
                ]
            ),
        )

    def _inference_without_feature_group(
        self,
        X: Tensor,
        y_train: Tensor,
        train_size: int,
        embed_with_test: bool,
        feature_shuffles: Optional[List[List[int]]],
    ) -> Tensor:
        """Inference path when feature grouping is disabled."""

        if feature_shuffles is None:
            if self.reserve_cls_tokens > 0:
                X = F.pad(X, (self.reserve_cls_tokens, 0), value=-100.0)

            features = X.transpose(1, 2).unsqueeze(-1)  # (B, H+C, T, 1)
            if self.target_aware:
                assert y_train is not None, "y_train must be provided when target_aware=True."
                y_train = y_train.unsqueeze(1).expand(-1, features.shape[1], -1)
            else:
                y_train = None

            embeddings = self.inference_mgr(
                self._compute_embeddings,
                inputs=OrderedDict(
                    [
                        ("features", features),
                        ("train_size", train_size),
                        ("y_train", y_train),
                        ("embed_with_test", embed_with_test),
                    ]
                ),
            )
        else:
            # Shuffle optimisation: compute once, reorder for each table
            B = X.shape[0]
            first_table = X[0]
            if self.reserve_cls_tokens > 0:
                first_table = F.pad(first_table, (self.reserve_cls_tokens, 0), value=-100.0)

            features = first_table.transpose(0, 1).unsqueeze(-1)  # (H+C, T, 1)
            if self.target_aware:
                assert y_train is not None, "y_train must be provided when target_aware=True."
                y_first = y_train[0].unsqueeze(0).expand(features.shape[0], -1)
            else:
                y_first = None

            first_embeddings = self.inference_mgr(
                self._compute_embeddings,
                inputs=OrderedDict(
                    [
                        ("features", features),
                        ("train_size", train_size),
                        ("y_train", y_first),
                        ("embed_with_test", embed_with_test),
                    ]
                ),
                output_repeat=B,
            )

            # Apply shuffles for tables after the first one
            embeddings = first_embeddings.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H+C, T, E)
            first_pattern = feature_shuffles[0]
            for i in range(1, B):
                mapping = self.map_feature_shuffle(first_pattern, feature_shuffles[i])
                if self.reserve_cls_tokens > 0:
                    mapping = [m + self.reserve_cls_tokens for m in mapping]
                    mapping = list(range(self.reserve_cls_tokens)) + mapping
                embeddings[i] = first_embeddings[mapping]

        return embeddings

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        embed_with_test: bool = False,
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

        y_train : Tensor
            Target values for training samples of shape (B, train_size).
            Used only for target-aware embedding.

        d : Optional[Tensor], default=None
            The number of features per dataset of shape (B,). Used only in training mode and
            when feature grouping is disabled. If feature grouping is enabled, it must be None.

        embed_with_test : bool, default=False
            If True, inducing points attend to all samples (train + test) in the set transformer.
            If False, inducing points only attend to training samples.

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch. Used only in inference mode.
            It is only effective when feature grouping is disabled. When provided, indicates that X contains
            the same table with different feature orders. In this case, embeddings are computed once and
            then shuffled accordingly.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager. Used only in inference mode.

        Returns
        -------
        Tensor
            Embeddings of shape (B, T, G+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - G is the number of feature groups
             - C is the number of class tokens
             - E is embedding dimension.
        """
        if self.training:
            embeddings = self._train_forward(X, y_train, d, embed_with_test)
        else:
            embeddings = self._inference_forward(X, y_train, embed_with_test, feature_shuffles, mgr_config)

        return embeddings  # (B, T, G+C, E)

    def _compute_embeddings_with_cache(
        self,
        features: Tensor,
        col_cache: KVCache,
        train_size: Optional[int] = None,
        y_train: Optional[Tensor] = None,
        use_cache: bool = False,
        store_cache: bool = True,
    ) -> Tensor:
        """Feature embedding using a shared set transformer with KV caching.

        Parameters
        ----------
        features : Tensor
            Input features of shape (..., T, in_dim).

        col_cache : KVCache
            Cache object for storing/retrieving ISAB K/V projections.

        train_size : Optional[int], default=None
            Position to split the input into training and test data.

        y_train : Optional[Tensor], default=None
            Target values of shape (..., train_size). Required when
            store_cache=True and target_aware=True; ignored when use_cache=True.

        use_cache : bool, default=False
            Whether to use cached values to avoid redundant computation.

        store_cache : bool, default=True
            Whether to store computed values in cache.

        Returns
        -------
        Tensor
            Embeddings of shape (..., T, E).
        """
        src = self.in_linear(features)

        if not self.target_aware:
            src = self.tf_col.forward_with_cache(
                src, col_cache=col_cache, train_size=train_size, use_cache=use_cache, store_cache=store_cache
            )
        else:
            # When using cache, skip y_train embedding — it's already baked
            # into the cached K/V projections from the store_cache pass.
            if store_cache:
                assert y_train is not None, "y_train must be provided when target_aware=True and store_cache=True."

                if self.max_classes > 0:
                    y_emb = self.y_encoder(y_train.float())
                else:
                    y_emb = self.y_encoder(y_train.unsqueeze(-1))
                src[..., :train_size, :] = src[..., :train_size, :] + y_emb

            src = self.tf_col.forward_with_cache(
                src, col_cache=col_cache, train_size=train_size, use_cache=use_cache, store_cache=store_cache
            )

        if self.affine:
            weights = self.ln_w(self.out_w(src))
            biases = self.ln_b(self.out_b(src))
            embeddings = features * weights + biases
        else:
            embeddings = src

        return embeddings

    def forward_with_cache(
        self,
        X: Tensor,
        col_cache: KVCache,
        y_train: Optional[Tensor] = None,
        use_cache: bool = False,
        store_cache: bool = True,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        """Transform input table into embeddings with KV caching support.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H).

        col_cache : KVCache
            Cache object for storing/retrieving ISAB K/V projections.

        y_train : Optional[Tensor], default=None
            Target values for training samples of shape (B, train_size).
            Required when store_cache=True; ignored when use_cache=True.

        use_cache : bool, default=False
            Whether to use cached values to avoid redundant computation.

        store_cache : bool, default=True
            Whether to store computed values in cache.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager. If None, uses the default
            COL_CONFIG from InferenceConfig.

        Returns
        -------
        Tensor
            Embeddings of shape (B, T, G+C, E).
        """

        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")

        if store_cache:
            assert y_train is not None, "y_train must be provided when store_cache=True"
            # many-class classification is not supported with caching
            if self.target_aware and self.max_classes > 0:
                num_classes = int(y_train.max().item()) + 1
                if num_classes > self.max_classes:
                    raise ValueError(
                        f"KV caching is not supported for classification with more classes "
                        f"({num_classes}) than max_classes ({self.max_classes}). Mixed-radix ensemble "
                        f"requires multiple forward passes which is incompatible with caching."
                    )

        if mgr_config is None:
            mgr_config = InferenceConfig().COL_CONFIG
        self.inference_mgr.configure(**mgr_config)

        if self.feature_group:
            X = self.feature_grouping(X)  # (B, T, G, group_size)
            if self.reserve_cls_tokens > 0:
                X = F.pad(X, (0, 0, self.reserve_cls_tokens, 0), value=-100.0)
            features = X.transpose(1, 2)  # (B, G+C, T, group_size)
        else:
            if self.reserve_cls_tokens > 0:
                X = F.pad(X, (self.reserve_cls_tokens, 0), value=-100.0)
            features = X.transpose(1, 2).unsqueeze(-1)  # (B, H+C, T, 1)

        if store_cache:
            train_size = y_train.shape[1]
            y_train = y_train.unsqueeze(1).expand(-1, features.shape[1], -1)
        else:
            train_size = None
            y_train = None

        embeddings = self.inference_mgr(
            self._compute_embeddings_with_cache,
            inputs=OrderedDict(
                [
                    ("features", features),
                    ("col_cache", col_cache),
                    ("train_size", train_size),
                    ("y_train", y_train),
                    ("use_cache", use_cache),
                    ("store_cache", store_cache),
                ]
            ),
        )
        return embeddings.transpose(1, 2)
