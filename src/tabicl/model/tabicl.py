from __future__ import annotations
from typing import Optional, List, Union, Literal

import torch
from torch import nn, Tensor

from .embedding import ColEmbedding
from .interaction import RowInteraction
from .learning import ICLearning
from .quantile_dist import QuantileToDistribution
from .kv_cache import TabICLCache
from .inference_config import InferenceConfig


class TabICL(nn.Module):
    """A Tabular In-Context Learning Foundation Model.

    TabICL is a transformer-based architecture for in-context learning on tabular data to make
    predictions without fine-tuning. It processes tabular data through three sequential stages:

    1. Column-wise embedding creates distribution-aware embeddings
    2. Row-wise interaction captures interactions between features within each row
    3. Dataset-wise in-context learning to learn patterns from labeled examples and make predictions

    Parameters
    ----------
    max_classes : int, default=10
        Determines the task type and output behavior:
        - If max_classes=0: The model performs regression using quantile prediction.
        - If max_classes>0: The model performs classification. This value specifies
          the number of classes the model supports natively. If the number of classes
          in the dataset exceeds this value, mixed-radix ensembling is used during
          column-wise embedding and hierarchical classification is used during in-context learning.

    num_quantiles : int, default=999
        Number of quantiles to predict for regression tasks. Only used when max_classes=0.
        The model directly predicts these quantile values.

    embed_dim : int, default=128
        Model dimension used in the column / row embedding transformers. For the in-context
        learning transformer, the dimension is this value multiplied by the number of CLS tokens.

    col_num_blocks : int, default=3
        Number of induced self-attention blocks in the column embedding transformer.

    col_nhead : int, default=8
        Number of attention heads in the column embedding transformer.

    col_num_inds : int, default=128
        Number of inducing points in the column embedding transformer.

    col_affine : bool, default=False
        If True, computes embeddings as: :math:`\\text{features} \\times W + b`.
        If False, directly uses the set transformer output as embeddings.

    col_feature_group : bool or Literal["same", "valid"], default="same"
        Feature grouping mode:
        - False: No grouping
        - True or "same": Group through circular permutation (output has same number of groups as features)
        - "valid": Group through padding and reshaping (output may have fewer groups)

    col_feature_group_size : int, default=3
        Number of features per group when feature grouping is enabled.

    col_target_aware : bool, default=True
        If True, incorporates target information into column-wise embeddings.

    col_ssmax : bool or str, default="qassmax-mlp-elementwise"
        Type of scalable softmax to use in the column embedding transformer. Note that only the first
        attention layer of the induced self-attention blocks uses SSMax.
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

    row_num_blocks : int, default=3
        Number of attention blocks in the row interaction transformer.

    row_nhead : int, default=8
        Number of attention heads in the row interaction transformer.

    row_num_cls : int, default=4
        Number of learnable CLS tokens used to aggregate feature information per row.

    row_rope_base : float, default=100000
        Base scaling factor for rotary position encoding in the row interaction transformer.

    row_rope_interleaved : bool, default=False
        If True, uses interleaved rotation where dimension pairs are (0,1), (2,3), etc.
        If False, uses non-interleaved rotation where the embedding is split into
        first half [0:d//2] and second half [d//2:d].

    icl_num_blocks : int, default=12
        Number of transformer blocks in the in-context learning transformer.

    icl_nhead : int, default=8
        Number of attention heads in the in-context learning transformer.

    icl_ssmax : bool or str, default="qassmax-mlp-elementwise"
        Type of scalable softmax to use in the in-context learning transformer.
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

    ff_factor : int, default=2
        Expansion factor for feedforward networks across all components.

    dropout : float, default=0.0
        Dropout probability across all components.

    activation : str or unary callable, default="gelu"
        Activation function used throughout the model.

    norm_first : bool, default=True
        If True, uses pre-norm architecture across all components.

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers (sets bias=False in nn.LayerNorm).

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of additional computation.
    """

    def __init__(
        self,
        max_classes: int = 10,
        num_quantiles: int = 999,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        col_nhead: int = 8,
        col_num_inds: int = 128,
        col_affine: bool = False,
        col_feature_group: Union[bool, Literal["same", "valid"]] = "same",
        col_feature_group_size: int = 3,
        col_target_aware: bool = True,
        col_ssmax: Union[
            bool,
            Literal[
                "none",
                "ssmax",
                "ssmax-mlp",
                "ssmax-mlp-elementwise",
                "qassmax-mlp",
                "qassmax-mlp-elementwise",
            ],
        ] = "qassmax-mlp-elementwise",
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100000,
        row_rope_interleaved: bool = False,
        icl_num_blocks: int = 12,
        icl_nhead: int = 8,
        icl_ssmax: Union[
            bool,
            Literal[
                "none",
                "ssmax",
                "ssmax-mlp",
                "ssmax-mlp-elementwise",
                "qassmax-mlp",
                "qassmax-mlp-elementwise",
            ],
        ] = "qassmax-mlp-elementwise",
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        recompute: bool = False,
    ):
        super().__init__()
        icl_dim = embed_dim * row_num_cls  # CLS tokens are concatenated for ICL

        # Determine task type
        if max_classes == 0:  # Regression
            if num_quantiles <= 0:
                raise ValueError("For regression (max_classes=0), num_quantiles must be greater than 0.")
            out_dim = num_quantiles
            self.quantile_dist = QuantileToDistribution(num_quantiles=num_quantiles)
        else:  # Classification
            out_dim = max_classes

        self.max_classes = max_classes
        self.num_quantiles = num_quantiles
        self.embed_dim = embed_dim
        self.col_num_blocks = col_num_blocks
        self.col_nhead = col_nhead
        self.col_num_inds = col_num_inds
        self.col_affine = col_affine
        self.col_feature_group = col_feature_group
        self.col_feature_group_size = col_feature_group_size
        self.col_target_aware = col_target_aware
        self.col_ssmax = col_ssmax
        self.row_num_blocks = row_num_blocks
        self.row_nhead = row_nhead
        self.row_num_cls = row_num_cls
        self.row_rope_base = row_rope_base
        self.row_rope_interleaved = row_rope_interleaved
        self.icl_num_blocks = icl_num_blocks
        self.icl_nhead = icl_nhead
        self.icl_ssmax = icl_ssmax
        self.ff_factor = ff_factor
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.bias_free_ln = bias_free_ln

        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=col_num_blocks,
            nhead=col_nhead,
            num_inds=col_num_inds,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            affine=col_affine,
            feature_group=col_feature_group,
            feature_group_size=col_feature_group_size,
            target_aware=col_target_aware,
            max_classes=max_classes,
            reserve_cls_tokens=row_num_cls,
            ssmax=col_ssmax,
            recompute=recompute,
        )

        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=row_num_blocks,
            nhead=row_nhead,
            dim_feedforward=embed_dim * ff_factor,
            num_cls=row_num_cls,
            rope_base=row_rope_base,
            rope_interleaved=row_rope_interleaved,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            recompute=recompute,
        )

        self.icl_predictor = ICLearning(
            out_dim=out_dim,
            max_classes=max_classes,
            d_model=icl_dim,
            num_blocks=icl_num_blocks,
            nhead=icl_nhead,
            dim_feedforward=icl_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            ssmax=icl_ssmax,
            recompute=recompute,
        )

        # KV cache for efficient inference
        self._cache: Optional[TabICLCache] = None

    @property
    def has_cache(self) -> bool:
        """Check if a valid cache is stored."""
        return self._cache is not None and not self._cache.is_empty()

    def clear_cache(self) -> None:
        """Clear the stored cache."""
        self._cache = None

    def _train_forward(
        self, X: Tensor, y_train: Tensor, d: Optional[Tensor] = None, embed_with_test: bool = False
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning for training.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        d : Optional[Tensor], default=None
            The number of features per dataset.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding.

        Returns
        -------
        Tensor
            Predictions of shape (B, test_size, out_dim):

            - For regression (max_classes=0): out_dim = num_quantiles
            - For classification (max_classes>0): out_dim = max_classes
        """

        B, T, H = X.shape
        train_size = y_train.shape[1]
        assert train_size <= T, "Number of training samples exceeds total samples"

        # Check if d is provided and has the same length as the number of features
        if d is not None and len(d.unique()) == 1 and d[0] == H:
            d = None

        # Column-wise embedding -> Row-wise interaction
        representations = self.row_interactor(
            self.col_embedder(
                X,
                y_train=y_train,
                d=d,
                embed_with_test=embed_with_test,
            ),
            d=d,
        )

        # Dataset-wise in-context learning
        return self.icl_predictor(representations, y_train=y_train)

    def _inference_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: Optional[InferenceConfig] = None,
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, column-wise embeddings are computed once and then shuffled accordingly.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding.

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        inference_config : Optional[InferenceConfig], default=None
            Inference configuration.

        Returns
        -------
        Tensor
            For regression (max_classes=0):
                Predictions of shape (B, test_size, num_quantiles), where test_size = T - train_size

            For classification (max_classes>0):
                If return_logits=True: Logits of shape (B, test_size, num_classes)
                If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """

        train_size = y_train.shape[1]
        assert train_size <= X.shape[1], "Number of training samples exceeds total samples"

        if inference_config is None:
            inference_config = InferenceConfig()

        # Column-wise embedding -> Row-wise interaction
        representations = self.row_interactor(
            self.col_embedder(
                X,
                y_train=y_train,
                embed_with_test=embed_with_test,
                feature_shuffles=feature_shuffles,
                mgr_config=inference_config.COL_CONFIG,
            ),
            mgr_config=inference_config.ROW_CONFIG,
        )

        # Dataset-wise in-context learning
        out = self.icl_predictor(
            representations,
            y_train=y_train,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
            mgr_config=inference_config.ICL_CONFIG,
        )

        return out

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        embed_with_test: bool = False,
        feature_shuffles: Optional[List[List[int]]] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: Optional[InferenceConfig] = None,
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        d : Optional[Tensor], default=None
            The number of features per dataset. Used only in training mode.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding.

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch. Used only in inference mode.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, column-wise embeddings are computed once and then shuffled accordingly.

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities. Used only in inference mode.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function. Used only in inference mode.

        inference_config : Optional[InferenceConfig], default=None
            Inference configuration. Used only in inference mode.

        Returns
        -------
        Tensor
            For training mode:
                Predictions of shape (B, test_size, out_dim):

                - For regression (max_classes=0): out_dim = num_quantiles
                - For classification (max_classes>0): out_dim = max_classes

            For inference mode:
                For regression (max_classes=0):
                    Predictions of shape (B, test_size, num_quantiles)

                For classification (max_classes>0):
                    If return_logits=True: Logits of shape (B, test_size, num_classes)
                    If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """

        if self.training:
            out = self._train_forward(X, y_train, d=d, embed_with_test=embed_with_test)
        else:
            out = self._inference_forward(
                X,
                y_train,
                feature_shuffles=feature_shuffles,
                embed_with_test=embed_with_test,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                inference_config=inference_config,
            )

        return out

    def predict_stats(
        self,
        X: Tensor,
        y_train: Tensor,
        output_type: str = "mean",
        alphas: Optional[List[float]] = None,
        embed_with_test: bool = False,
        inference_config: InferenceConfig = None,
    ) -> Tensor:
        """Compute summary statistics from predicted quantiles.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining
            positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        output_type : str or list of str, default="mean"
            Determines the type of output to return. Supported values:
            - "mean": Mean of the predicted quantiles (fast, no tail modeling).
            - "variance": Variance of the predicted quantiles (fast, no tail modeling).
            - "median": Median via inverse CDF interpolation.
            - "quantiles": Specific quantiles via inverse CDF. Use `alphas` to specify levels.
            If a list, returns a dict with the requested statistics.

        alphas : Optional[List[float]], default=None
            Probability levels for quantile output. Only used when "quantiles" is in `output_type`.
            Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding.

        inference_config : InferenceConfig
            Inference configuration.

        Returns
        -------
        Tensor or dict of Tensors
            - If `output_type` is a single string: returns the corresponding tensor.
            - If `output_type` is a list: returns a dict mapping names to tensors.

            Output shapes:

            - "mean", "variance", "median": (B, test_size)
            - "quantiles": (B, test_size, len(alphas))
        """
        assert self.max_classes == 0, "predict_stats is only applicable for regression tasks"

        raw_quantiles = self._inference_forward(
            X, y_train, embed_with_test=embed_with_test, inference_config=inference_config
        )  # (B, test_size, num_quantiles)

        dist = self.quantile_dist(raw_quantiles)
        raw_quantiles = dist.quantiles  # dist ensures that quantiles are monotonic

        output_type = [output_type] if isinstance(output_type, str) else output_type
        results = {}

        if "mean" in output_type:
            results["mean"] = raw_quantiles.mean(dim=-1)
        if "variance" in output_type:
            results["variance"] = raw_quantiles.var(dim=-1)
        if "median" in output_type:
            results["median"] = dist.icdf(
                alpha=torch.tensor(0.5, device=raw_quantiles.device, dtype=raw_quantiles.dtype)
            )
        if "quantiles" in output_type:
            if alphas is None:
                alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            results["quantiles"] = dist.icdf(
                alpha=torch.tensor(alphas, device=raw_quantiles.device, dtype=raw_quantiles.dtype)
            )

        if len(output_type) == 1:
            return results[output_type[0]]

        return results

    def forward_with_cache(
        self,
        X_train: Optional[Tensor] = None,
        y_train: Optional[Tensor] = None,
        X_test: Optional[Tensor] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        use_cache: bool = False,
        store_cache: bool = True,
        cache: Optional[TabICLCache] = None,
        cache_mode: str = "kv",
        inference_config: Optional[InferenceConfig] = None,
    ) -> Optional[Tensor]:
        """Forward pass with caching support for efficient inference.

        This method enables caching of training data computations to speed up
        repeated inference on the same training context. Two caching modes are
        supported:

        - ``"kv"``: Cache KV projections from both column embedding and ICL
          transformer layers. Fastest inference but uses more memory.
        - ``"repr"``: Cache column embedding KV projections and row interaction
          outputs (representations with y_train baked in). Uses ~24x less memory
          for the ICL part, at the cost of re-running the ICL transformer.

        Exactly one of `use_cache` or `store_cache` must be True.

        When ``store_cache=True``:
        - Requires X_train and y_train to be provided
        - Processes training data and stores cached values in self._cache
        - If X_test is also provided, returns predictions for test samples
        - If X_test is None, returns None (cache-only mode)

        When ``use_cache=True``:
        - Requires X_test and a populated self._cache
        - Uses cached values for training data

        Parameters
        ----------
        X_train : Optional[Tensor], default=None
            Training input of shape (B, train_size, H). Required when store_cache=True.

        y_train : Optional[Tensor], default=None
            Training target of shape (B, train_size). Required when store_cache=True.

        X_test : Optional[Tensor], default=None
            Test input of shape (B, test_size, H). Required when use_cache=True and optional
            when store_cache=True.

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        use_cache : bool, default=False
            Whether to use cached values to avoid redundant computation.

        store_cache : bool, default=True
            Whether to store computed values in cache.

        cache : Optional[TabICLCache], default=None
            External cache to use for inference. If provided, equivalent to
            setting use_cache=True and store_cache=False, but uses the provided
            cache instead of the model's internal self._cache.

        cache_mode : str, default="kv"
            Caching strategy: ``"kv"`` for KV projection caching, ``"repr"`` for
            representation caching. Ignored when ``use_cache=True`` (auto-detected
            from cache contents).

        inference_config : Optional[InferenceConfig], default=None
            Inference configuration.

        Returns
        -------
        Optional[Tensor]
            Predictions of shape (B, test_size, out_dim), or None if store_cache=True
            and X_test is not provided.

        Raises
        ------
        ValueError
            If use_cache == store_cache (exactly one must be True),
            if store_cache=True but X_train or y_train is None, or
            if use_cache=True but X_test is None or no cache exists.
        """

        if cache is not None:
            use_cache = True
            store_cache = False
            self._cache = cache

        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")

        if cache_mode not in ("kv", "repr"):
            raise ValueError(f"cache_mode must be 'kv' or 'repr', got '{cache_mode}'")

        if inference_config is None:
            inference_config = InferenceConfig()

        # Auto-detect cache mode from cache contents
        if use_cache and self._cache is not None and self._cache.cache_type == "repr":
            cache_mode = "repr"

        if store_cache:
            if X_train is None or y_train is None:
                raise ValueError("X_train and y_train are required when store_cache=True")

            # Initialize cache based on training data
            num_classes = len(torch.unique(y_train[0])) if self.max_classes > 0 else 0
            self._cache = TabICLCache(train_shape=X_train.shape, num_classes=num_classes)

            if X_test is None:
                X = X_train
            else:
                X = torch.cat([X_train, X_test], dim=1)

        if use_cache:
            if X_test is None:
                raise ValueError("X_test is required when use_cache=True")

            if self._cache is None or self._cache.is_empty():
                raise ValueError("No cache available. Call with store_cache=True first.")

            X = X_test
            y_train = None

        # Column-wise embedding with cache support -> Row-wise interaction
        representations = self.row_interactor(
            self.col_embedder.forward_with_cache(
                X,
                col_cache=self._cache.col_cache,
                y_train=y_train,
                use_cache=use_cache,
                store_cache=store_cache,
                mgr_config=inference_config.COL_CONFIG,
            ),
            mgr_config=inference_config.ROW_CONFIG,
        )

        # Dataset-wise in-context learning
        if cache_mode == "repr":
            if store_cache:
                train_size = y_train.shape[1]
                # Bake y_train into train portion of representations
                representations = self.icl_predictor.prepare_repr_cache(representations, y_train)
                self._cache.row_repr = representations[:, :train_size]

                if X_test is None:
                    return None
            else:
                # Concatenate cached train representations with test representations
                train_repr = self._cache.row_repr
                train_size = train_repr.shape[1]
                representations = torch.cat([train_repr.to(representations.device), representations], dim=1)

            out = self.icl_predictor.forward_with_repr_cache(
                representations,
                train_size=train_size,
                num_classes=self._cache.num_classes,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                mgr_config=inference_config.ICL_CONFIG,
            )
        else:
            out = self.icl_predictor.forward_with_cache(
                representations,
                icl_cache=self._cache.icl_cache,
                y_train=y_train,
                num_classes=self._cache.num_classes,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                use_cache=use_cache,
                store_cache=store_cache,
                mgr_config=inference_config.ICL_CONFIG,
            )

            if X_test is None:
                return None

        return out

    def predict_stats_with_cache(
        self,
        X_train: Optional[Tensor] = None,
        y_train: Optional[Tensor] = None,
        X_test: Optional[Tensor] = None,
        output_type: str = "mean",
        alphas: Optional[List[float]] = None,
        use_cache: bool = False,
        store_cache: bool = True,
        cache: Optional[TabICLCache] = None,
        cache_mode: str = "kv",
        inference_config: Optional[InferenceConfig] = None,
    ) -> Optional[Tensor]:
        """Compute summary statistics from predicted quantiles with KV caching.

        Parameters
        ----------
        X_train : Optional[Tensor], default=None
            Training input of shape (B, train_size, H). Required when store_cache=True.

        y_train : Optional[Tensor], default=None
            Training target of shape (B, train_size). Required when store_cache=True.

        X_test : Optional[Tensor], default=None
            Test input of shape (B, test_size, H). Required when use_cache=True and
            optional when store_cache=True.

        output_type : str or list of str, default="mean"
            Determines the type of output to return. Supported values:
            - "mean": Mean of the predicted quantiles (fast, no tail modeling).
            - "variance": Variance of the predicted quantiles (fast, no tail modeling).
            - "median": Median via inverse CDF interpolation.
            - "quantiles": Specific quantiles via inverse CDF. Use `alphas` to specify levels.
            If a list, returns a dict with the requested statistics.

        alphas : Optional[List[float]], default=None
            Probability levels for quantile output. Only used when "quantiles" is in
            `output_type`. Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].

        use_cache : bool, default=False
            Whether to use cached values to avoid redundant computation.

        store_cache : bool, default=True
            Whether to store computed values in cache.

        cache : Optional[TabICLCache], default=None
            External cache to use for inference. If provided, equivalent to
            setting use_cache=True and store_cache=False.

        cache_mode : str, default="kv"
            Caching strategy: ``"kv"`` for KV projection caching, ``"repr"`` for
            representation caching. Ignored when ``use_cache=True`` (auto-detected
            from cache contents).

        inference_config : Optional[InferenceConfig], default=None
            Inference configuration.

        Returns
        -------
        Tensor or dict of Tensors or None
            None if store_cache=True and X_test is not provided. Otherwise:

            - If `output_type` is a single string: returns the corresponding tensor.
            - If `output_type` is a list: returns a dict mapping names to tensors.

            Output shapes:

            - "mean", "variance", "median": (B, test_size)
            - "quantiles": (B, test_size, len(alphas))
        """
        assert self.max_classes == 0, "predict_stats_with_cache is only applicable for regression tasks"

        raw_quantiles = self.forward_with_cache(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            use_cache=use_cache,
            store_cache=store_cache,
            cache=cache,
            cache_mode=cache_mode,
            inference_config=inference_config,
        )

        if raw_quantiles is None:
            return None

        dist = self.quantile_dist(raw_quantiles)
        raw_quantiles = dist.quantiles

        output_type = [output_type] if isinstance(output_type, str) else output_type
        results = {}

        if "mean" in output_type:
            results["mean"] = raw_quantiles.mean(dim=-1)
        if "variance" in output_type:
            results["variance"] = raw_quantiles.var(dim=-1)
        if "median" in output_type:
            results["median"] = dist.icdf(
                alpha=torch.tensor(0.5, device=raw_quantiles.device, dtype=raw_quantiles.dtype)
            )
        if "quantiles" in output_type:
            if alphas is None:
                alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            results["quantiles"] = dist.icdf(
                alpha=torch.tensor(alphas, device=raw_quantiles.device, dtype=raw_quantiles.dtype)
            )

        if len(output_type) == 1:
            return results[output_type[0]]

        return results
