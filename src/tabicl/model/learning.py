from __future__ import annotations
from typing import Optional
from collections import OrderedDict
import math

import torch
from torch import nn, Tensor

from .layers import ClassNode, OneHotAndLinear
from .encoders import Encoder
from .inference import InferenceManager
from ..config import MgrConfig


class ICLearning(nn.Module):
    """Dataset-wise in-context learning with automatic hierarchical classification support.

    This module implements in-context learning that:
    1. Takes row representations and training labels as input
    2. Conditions the model on training examples
    3. Makes predictions for test examples based on learned patterns
    4. Automatically handles both small and large label spaces

    Parameters
    ----------
    max_classes : int
        Number of classes that the model supports natively. If the number of classes
        in the dataset exceeds this value, hierarchical classification is used.

    d_model : int
        Model dimension

    num_blocks : int
        Number of blocks used in the ICL encoder

    nhead : int
        Number of attention heads of the ICL encoder

    dim_feedforward : int
        Dimension of the feedforward network of the ICL encoder

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)
    """

    def __init__(
        self,
        max_classes: int,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.max_classes = max_classes
        self.norm_first = norm_first

        self.tf_icl = Encoder(
            num_blocks=num_blocks,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        if self.norm_first:
            self.ln = nn.LayerNorm(d_model)

        self.y_encoder = OneHotAndLinear(max_classes, d_model)
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, max_classes))

        self.inference_mgr = InferenceManager(enc_name="tf_icl", out_dim=max_classes)

    def _grouping(self, num_classes: int) -> tuple[Tensor, int]:
        """Divide classes into balanced groups for hierarchical classification.

        This method implements a balanced partitioning strategy that divides classes
        into approximately equal-sized groups to minimize tree depth.

        Parameters
        ----------
        num_classes : int
            Total number of unique classes to partition into groups

        Returns
        -------
        tuple[Tensor, int]
            - group_assignments: Tensor mapping each class index to its assigned group (0-indexed)
            - num_groups: Total number of groups created

        Notes
        -----
        For example, with max_classes=10 and num_classes=25:
        - Creates ceil(25/10) = 3 groups
        - Distributes classes as [9, 8, 8] elements per group
        - Returns assignments tensor [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2]
        """

        if num_classes <= self.max_classes:
            return torch.zeros(num_classes, dtype=torch.int), 1

        num_groups = math.ceil(num_classes / self.max_classes)
        group_assignments = torch.zeros(num_classes, dtype=torch.int)
        current_pos = 0

        remaining_classes = num_classes
        remaining_groups = num_groups
        for i in range(num_groups):
            group_size = math.ceil(remaining_classes / remaining_groups)
            group_assignments[current_pos : current_pos + group_size] = i
            current_pos += group_size
            remaining_classes -= group_size
            remaining_groups -= 1

        return group_assignments, num_groups

    def _fit_node(self, node: ClassNode, R: Tensor, y: Tensor, current_depth: int):
        """Recursively build a node in the hierarchical classification tree.

        For each node, this method either:
        1. Creates a leaf node if the number of classes is small enough to handle directly
        2. Splits classes into groups and recursively creates child nodes for each group

        Parameters
        ----------
        node : ClassNode
            Current node being constructed in the tree

        R : Tensor
            Row representations of shape (num_samples, D) where num_samples is the number of
            examples assigned to this node

        y : Tensor
            Targets of shape (num_samples,) corresponding to the samples in R

        current_depth : int
            Current depth in the hierarchical tree (root = 0)
        """

        unique_classes = torch.unique(y).int()
        node.classes_ = unique_classes

        if len(unique_classes) <= self.max_classes:
            # Create leaf node for direct classification
            node.is_leaf = True
            node.R = R
            node.y = y
            return

        # Merge classes into groups
        group_assignments, num_groups = self._grouping(len(unique_classes))

        # Create mapping from original class labels to their corresponding group numbers
        node.class_mapping = {c.item(): g.item() for c, g in zip(unique_classes, group_assignments)}
        node.group_indices = torch.tensor([node.class_mapping[c.item()] for c in y], dtype=torch.int)
        node.R = R
        node.y = y
        node.is_leaf = False

        # Create child nodes for each group
        for group in range(num_groups):
            mask = node.group_indices == group
            child_node = ClassNode(current_depth + 1)
            self._fit_node(child_node, R[mask], y[mask], current_depth + 1)
            node.child_nodes.append(child_node)

    def _fit_hierarchical(self, R_train: Tensor, y_train: Tensor):
        """Initialize the hierarchical classification tree.

        Parameters
        ----------
        R_train : Tensor
            Row representations of training data of shape (train_size, D)

        y_train : Tensor
            Training targets of shape (train_size,)
        """

        self.root = ClassNode(depth=0)
        self._fit_node(self.root, R_train, y_train, current_depth=0)

    def _label_encoding(self, y: Tensor) -> Tensor:
        """Remapping target values to contiguous integers starting from 0."""

        unique_vals, _ = torch.unique(y, return_inverse=True)
        indices = unique_vals.argsort()
        return indices[torch.searchsorted(unique_vals, y)]

    def _icl_predictions(self, R: Tensor, y_train: Tensor) -> Tensor:
        """In-context learning predictions.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor of shape (B, train_size)
            Training targets, where train_size is the position to split
            the input into training and test data
        """

        train_size = y_train.shape[1]
        R[:, :train_size] = R[:, :train_size] + self.y_encoder(y_train.float())
        src = self.tf_icl(R, attn_mask=train_size)
        if self.norm_first:
            src = self.ln(src)
        out = self.decoder(src)  # (B, T, max_classes)

        return out

    def _predict_standard(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = False,
        softmax_temperature: float = 0.9,
        auto_batch: bool = True,
    ) -> Tensor:
        """Generate predictions for standard classification with up to `max_classes` classes.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor of shape (B, train_size)
            Training targets, where train_size is the position to split
            the input into training and test data

        return_logits : bool, default=False
            If True, return logits instead of probabilities

        softmax_temperature : float, default=0.9
            Temperature for the softmax function

        auto_batch : bool, default=True
            Whether to use InferenceManager to automatically split inputs into smaller batches
        """

        train_size = y_train.shape[1]
        num_classes = len(torch.unique(y_train[0]))
        out = self.inference_mgr(
            self._icl_predictions, inputs=OrderedDict([("R", R), ("y_train", y_train)]), auto_batch=auto_batch
        )
        out = out[:, train_size:, :num_classes]

        if not return_logits:
            out = torch.softmax(out / softmax_temperature, dim=-1)

        return out

    def _predict_hierarchical(self, R_test: Tensor, softmax_temperature: float = 0.9) -> Tensor:
        """Generate predictions using the hierarchical classification tree.

        This method traverses the tree from leaves to root, computing probabilities at each level
        and combining them according to the probability chain rule.

        Parameters
        ----------
        R_test : Tensor
            Row representations of test data of shape (test_size, D)

        softmax_temperature : float, default=0.9
            Temperature for the softmax function

        Returns
        -------
        Tensor
            Probability over all classes, shape (test_size, C)
        """

        test_size = R_test.shape[0]
        device = R_test.device
        num_classes = len(self.root.classes_)

        def process_node(node, R_test):
            """Recursively process a node in the hierarchical tree.

            For leaf nodes: Directly predict class probabilities within the node's subset
            For internal nodes: Combine predictions from child nodes weighted by group probabilities
            """

            # Concatenate test data with node data
            node_R = torch.cat([node.R.to(device), R_test], dim=0)

            # Case 1: Leaf node - direct classification
            if node.is_leaf:
                node_y = self._label_encoding(node.y.to(device))
                # Get predictions for this leaf
                leaf_preds = self._predict_standard(
                    R=node_R.unsqueeze(0),
                    y_train=node_y.unsqueeze(0),
                    softmax_temperature=softmax_temperature,
                    auto_batch=False,
                ).squeeze(0)
                # Map leaf predictions to the global class space
                global_preds = torch.zeros((test_size, num_classes), device=device)
                for local_idx, global_idx in enumerate(node.classes_):
                    global_preds[:, global_idx] = leaf_preds[:, local_idx]

                return global_preds

            # Case 2: Internal node - classification into groups
            # Initialize output tensor for all classes
            final_probs = torch.zeros((test_size, num_classes), device=device)

            # Get group probabilities for this node
            node_y = node.group_indices.to(device)
            group_probs = self._predict_standard(
                R=node_R.unsqueeze(0),
                y_train=node_y.unsqueeze(0),
                softmax_temperature=softmax_temperature,
                auto_batch=False,
            ).squeeze(0)

            # Recursively process child nodes and combine predictions
            for group_idx, child_node in enumerate(node.child_nodes):
                child_probs = process_node(child_node, R_test)
                final_probs += child_probs * group_probs[:, group_idx : group_idx + 1]

            return final_probs

        return process_node(self.root, R_test)

    def forward(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        """In-context learning based on learned row representations.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor of shape (B, train_size)
            Training targets, where train_size is the position to split
            the input into training and test data

        return_logits : bool, default=True
            If True, return logits instead of probabilities

        softmax_temperature : float, default=0.9
            Temperature for the softmax function

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager

        Returns
        -------
        Tensor
            Raw logits or probabilities for test samples of shape (B, T-train_size, num_classes)
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

        num_classes = len(torch.unique(y_train[0]))
        assert all(
            len(torch.unique(yi)) == num_classes for yi in y_train
        ), "All tables must have the same number of classes"

        if num_classes <= self.max_classes:
            # Standard classification
            out = self._predict_standard(
                R, y_train, return_logits=return_logits, softmax_temperature=softmax_temperature
            )
        else:
            # Hierarchical classification
            out = []
            train_size = y_train.shape[1]
            for ri, yi in zip(R, y_train):
                if mgr_config.offload:
                    ri, yi = ri.cpu(), yi.cpu()
                else:
                    ri, yi = ri.to(mgr_config.device), yi.to(mgr_config.device)
                self._fit_hierarchical(ri[:train_size], yi)
                probs = self._predict_hierarchical(ri[train_size:])
                out.append(probs)
            out = torch.stack(out, dim=0)
            if return_logits:
                out = softmax_temperature * torch.log(out + 1e-6)

        return out
