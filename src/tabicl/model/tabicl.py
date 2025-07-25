from __future__ import annotations

from typing import Optional, List
from torch import nn, Tensor

from .tabcompressor import CompressorProjector, TabCompressor
from .embedding import ColEmbedding
from .interaction import RowInteraction
from .learning import ICLearning
from .inference_config import InferenceConfig
from .tabpfn_arch.model.loading import load_model_criterion_config
from .tabpfn_arch.model.transformer import PerFeatureTransformer as ContextCompressionTransformer
from .tabpfn_arch.model.config import ModelConfig as TabPFNModelConfig

import torch


class TabICL(nn.Module):
    """A Tabular In-Context Learning Foundation Model.

    TabICL is a transformer-based architecture for in-context learning on tabular data to make
    predictions without fine-tuning. It processes tabular data through three sequential stages:

    1. Column-wise embedding creates distribution-aware embeddings
    2. Row-wise interaction captures interactions between features within each row
    3. Dataset-wise in-context learning to learn patterns from labeled examples and make predictions

    For datasets with more than `max_classes` classes, TabICL switches to hierarchical lassification
    to recursively partition classes into subgroups, forming a multi-level classification tree.

    Parameters
    ----------
    max_classes : int, default=10
        Number of classes that the model supports natively. If the number of classes
        in the dataset exceeds this value, hierarchical classification is used.

    embed_dim : int, default=128
        Model dimension used in the column / row embedding transformers. For the in-context
        learning transformer, the dimension is this value multiplied by the number of CLS tokens.

    col_num_blocks : int, default=3
        Number of induced self-attention blocks in the column embedding transformer

    col_nhead : int, default=4
        Number of attention heads in the column embedding transformer

    col_num_inds : int, default=128
        Number of inducing points in the column embedding transformer

    row_num_blocks : int, default=3
        Number of attention blocks in the row interaction transformer

    row_nhead : int, default=8
        Number of attention heads in the row interaction transformer

    row_num_cls : int, default=4
        Number of learnable CLS tokens used to aggregate feature information per row

    row_rope_base : float, default=100000
        Base scaling factor for rotary position encoding in the row interaction transformer

    icl_num_blocks : int, default=12
        Number of transformer blocks in the in-context learning transformer

    icl_nhead : int, default=4
        Number of attention heads in the in-context learning transformer

    ff_factor : int, default=2
        Expansion factor for feedforward networks across all components

    dropout : float, default=0.0
        Dropout probability across all components

    activation : str or unary callable, default="gelu"
        Activation function used throughout the model

    norm_first : bool, default=True
        If True, uses pre-norm architecture across all components
    """

    def __init__(
        self,
        max_classes: int = 10,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        col_nhead: int = 4,
        col_num_inds: int = 128,
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100000,
        icl_num_blocks: int = 12,
        icl_nhead: int = 4,
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        use_compressor: bool = True,
        row_compression_percentage: float = 10
    ):
        super().__init__()
        self.max_classes = max_classes
        self.embed_dim = embed_dim
        self.col_num_blocks = col_num_blocks
        self.col_nhead = col_nhead
        self.col_num_inds = col_num_inds
        self.row_num_blocks = row_num_blocks
        self.row_nhead = row_nhead
        self.row_num_cls = row_num_cls
        self.row_rope_base = row_rope_base
        self.icl_num_blocks = icl_num_blocks
        self.icl_nhead = icl_nhead
        self.ff_factor = ff_factor
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first

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

        icl_dim = embed_dim * row_num_cls  # CLS tokens are concatenated for ICL
        self.icl_predictor = ICLearning(
            max_classes=max_classes,
            d_model=icl_dim,
            num_blocks=icl_num_blocks,
            nhead=icl_nhead,
            dim_feedforward=icl_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        self.use_compressor = use_compressor
        self.row_compression_percentage = row_compression_percentage
        self.compressor_projector = None
        if self.use_compressor:
            self.context_compression_transformer = self._build_compressor()
        else:
            self.context_compression_transformer = None  # nothing to build


    def _build_compressor(self):
        return TabCompressor(
            embed_dim=self.embed_dim,
            col_num_blocks=self.col_num_blocks,
            col_nhead=self.col_nhead,
            col_num_inds=self.col_num_inds,
            row_num_blocks=self.row_num_blocks,
            row_nhead=self.row_nhead,
            row_num_cls=self.row_num_cls,
            row_rope_base=self.row_rope_base,
            ff_factor=self.ff_factor,
            dropout=self.dropout,
            activation=self.activation,
            norm_first=self.norm_first,
        )

    def _compress(self, X: Tensor, y_train: Tensor, train_size: int):
        """Compress the input tensor X using the context compression transformer.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)

        y : Tensor
            Labels of shape (B, T) where:
             - B is the number of tables
             - T is the number of samples (rows)

        train_size : int

        Returns
        -------
        Tensor
            Compressed tensor of shape (B, T, H') where H' is the compressed feature dimension.
        """
        B, _, H = X.shape
        if not self.use_compressor:
            return X, y_train

        enc_train = self.context_compression_transformer(X[:, :train_size])  # (B, train, H)

        if self.compressor_projector is None:
            self.compressor_projector = CompressorProjector(
                input_dim=enc_train.size(-1),  # d_enc from the headless TabICL encoder
                output_dim=H  # project back to original width
            ).to(enc_train.device)

        enc_train = self.compressor_projector(enc_train)  # (B, train, H)

        if self.row_compression_percentage > 0:
            keep = max(1, int(train_size * (1 - self.row_compression_percentage / 100)))
            perm = torch.randperm(train_size, device=X.device)[:keep]
            enc_train = enc_train[:, perm]
            y_train = y_train[:, perm]

        X_test = X[:, train_size:]
        X_compressed = torch.cat([enc_train, X_test], dim=1)  # (B, T, H)

        return X_compressed, y_train

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

        Returns
        -------
        Tensor
            Raw logits of shape (B, T, max_classes), which will be further handled by the training code.
        """

        B, T, H = X.shape
        train_size = y_train.shape[1]
        assert train_size <= T, "Number of training samples exceeds total samples"
        X_compressed, y_train_compressed = self._compress(X=X, y_train=y_train, train_size=train_size)
        effective_train_size = y_train_compressed.shape[1]
        # Check if d is provided and has the same length as the number of features
        if d is not None and len(d.unique()) == 1 and d[0] == H:
            d = None

        # Column-wise embedding -> Row-wise interaction
        representations = self.row_interactor(
            self.col_embedder(
                X_compressed,
                d=d,  # or mgr_config in inference
                train_size=None if embed_with_test else effective_train_size
            ),
            d=d
        )

        # Dataset-wise in-context learning
        out = self.icl_predictor(representations, y_train=y_train_compressed)

        return out

    def _inference_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: InferenceConfig = None,
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
            If True, allow training samples to attend to test samples during embedding

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities

        softmax_temperature : float, default=0.9
            Temperature for the softmax function

        inference_config: InferenceConfig
            Inference configuration

        Returns
        -------
        Tensor
            Raw logits or probabilities for test samples of shape (B, test_size, num_classes)
            where test_size = T - train_size
        """

        train_size = y_train.shape[1]
        assert train_size <= X.shape[1], "Number of training samples exceeds total samples"
        X_compressed, y_train_compressed = self._compress(X, y_train, train_size)
        effective_train_size = y_train_compressed.shape[1]
        if inference_config is None:
            inference_config = InferenceConfig()

        # Column-wise embedding -> Row-wise interaction
        representations = self.row_interactor(
            self.col_embedder(
                X_compressed,
                train_size=None if embed_with_test else effective_train_size,
                feature_shuffles=feature_shuffles,
                mgr_config=inference_config.COL_CONFIG,
            ),
            mgr_config=inference_config.ROW_CONFIG,
        )

        # Dataset-wise in-context learning
        out = self.icl_predictor(
            representations,
            y_train=y_train_compressed,
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
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: InferenceConfig = None,
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

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch. Used only in training mode.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, column-wise embeddings are computed once and then shuffled accordingly.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities. Used only in training mode.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function. Used only in training mode.

        inference_config: InferenceConfig
            Inferenece configuration. Used only in training mode.

        Returns
        -------
        Tensor
            For training mode:
              Raw logits of shape (B, T-train_size, max_classes), which will be further handled by the training code.

            For inference mode:
              Raw logits or probabilities for test samples of shape (B, T-train_size, num_classes).
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
