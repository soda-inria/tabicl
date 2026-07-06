from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F

from .graph_lib._base import Context, DatasetProperties
from .graph_lib._config import PriorConfig
from .graph_lib._properties import sample_categorical_sizes
from .graph_lib._dataset import RandomDataset

from ._reg2cls import standard_scaling, outlier_removing


class GraphSCM:
    """Generates a dataset with a causal graph structure.

    Parameters
    ----------
    regression : bool, default=False
        Whether the target is for regression or classification.

    seq_len : int, default=1024
        The number of samples (rows) to generate for the dataset.

    num_features : int, default=100
        The number of features.

    max_features : int, default=100
        Maximum number of features allowed.

    num_classes : int, default=2
        Number of classes for categorical target

    config : PriorConfig | None, default=None
        Configuration of the prior.

    cat_modes : List, default=None
        Categorical conversion modes that control how categorical features are processed.
        Each mode determines how continuous values are discretized and represented:
        - "id": Uses original continuous values for graph computation, returns category indices
        - "disc": Uses discretized cluster centers for graph computation, returns category indices
        - "disc_func": Applies random function to discretized centers, returns category indices
        - "int": Uses category indices as float values for graph computation, returns category indices
        - "softmax_id": Uses original values with softmax-based sampling, returns category indices
        - "softmax_disc": Uses softmax embeddings for graph computation, returns category indices
        - "softmax_int": Uses category indices from softmax sampling, returns category indices

        The discretization process creates clusters by finding nearest centers (for non-softmax modes)
        or using multinomial sampling with softmax probabilities (for softmax modes).

    fn_types : List, default=None
        List of function types to use for generating node mapping functions in the causal graph.
        Available function types:
        - "gp": Gaussian Process function using random Fourier features with Cauchy kernel
        - "generalized_gp": Enhanced GP function with optional product kernel and eigen-decay matrices
        - "nn": Neural Network function with random architecture (1-4 layers, configurable activations)
        - "tree": Random Forest of oblivious decision trees with random leaf values
        - "product": Product of two cheaper random functions for interaction modeling
        - "discretization": Nearest-neighbor discretization function with random target mapping
        - "linear": Simple linear transformation using random matrices
        - "flow": Normalizing flow function with iterative transformations

        Each function type creates different non-linear relationships between parent and child nodes
        in the causal graph, affecting the complexity and structure of generated features.

    permute_features : bool, default=True
        Whether to randomly permute features

    permute_labels : bool, default=True
        Whether to randomly permute class labels

    device : str, default="cpu"
        The computing device ('cpu' or 'cuda') where tensors will be allocated.

    **kwargs : dict
        Unused hyperparameters passed from parent configurations.
    """

    def __init__(
        self,
        regression: bool = False,
        seq_len: int = 1024,
        num_features: int = 100,
        max_features: int = 100,
        num_classes: int = 2,
        permute_features: bool = True,
        permute_labels: bool = True,
        config: Optional[PriorConfig] = None,
        device: str = "cpu",
        **kwargs,
    ):
        super(GraphSCM, self).__init__()
        self.regression = regression
        self.seq_len = seq_len
        self.num_features = num_features
        self.max_features = max_features
        self.num_classes = num_classes
        self.permute_features = permute_features
        self.permute_labels = permute_labels
        self.config = config
        self.device = device

    def __call__(self) -> None:
        """Generate a dataset and return features and target."""

        context = Context(config=self.config, device=self.device)
        properties = DatasetProperties(
            n_train=self.seq_len,
            n_test=0,
            cat_sizes={
                "x": sample_categorical_sizes(self.num_features, context, max_cat_size=200),
                "y": [0 if self.regression else self.num_classes],
            },
        )
        ds = RandomDataset(context).sample(properties)
        data = ds.get_concat_tensors()

        X_cat = data.get("x_cat", None)
        X_num = data.get("x_num", None)

        if X_cat is not None and X_num is not None:
            X = torch.cat([X_cat, X_num], dim=-1)
        elif X_cat is not None:
            X = X_cat
        elif X_num is not None:
            X = X_num
        else:
            raise ValueError("No features found in dataset")

        X = outlier_removing(X.float(), threshold=4)
        X = standard_scaling(X)

        if self.permute_features:
            feat_perm = torch.randperm(self.num_features, device=self.device)
            X = X[..., feat_perm]

        if self.num_features < self.max_features:
            X = F.pad(
                X, (0, self.max_features - self.num_features), mode="constant", value=0.0
            )  # (seq_len, max_features)

        if self.regression:
            y = data["y_num"]  # (seq_len, 1)
            y = outlier_removing(y.float(), threshold=4)
            y = standard_scaling(y)
            y = y.view(-1)  # (seq_len,)
        else:
            y = data["y_cat"].view(-1)  # (seq_len,)
            if self.permute_labels:
                class_perm = torch.randperm(self.num_classes, device=self.device)
                y = class_perm[y.long()]

        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
            X = torch.zeros_like(X)
            y = torch.full_like(y, -100.0)

        return X, y
