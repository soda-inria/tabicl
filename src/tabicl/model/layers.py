from __future__ import annotations
from typing import List, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .rope import RotaryEmbedding
from .attention import multi_head_attention_forward


class ClassNode:
    """Node in the hierarchical classification tree for handling many-class problems.

    Attributes
    ----------
    depth : int
        Current depth level in the hierarchical tree

    is_leaf : bool
        Whether this node handles a small enough subset of classes directly

    classes_ : Tensor
        List of unique class indices this node is responsible for

    child_nodes : list
        Child nodes for non-leaf nodes, each handling a subset of classes

    class_mapping : dict
        Maps original class indices to group indices for internal nodes

    group_indices : Tensor
        Transformed labels after mapping original classes to their group indices

    R : Tensor
        Feature data associated with this node

    y : Tensor
        Target labels associated with this node
    """

    def __init__(self, depth=0):
        self.depth = depth
        self.is_leaf = False
        self.classes_ = None
        self.child_nodes = []
        self.class_mapping = {}
        self.group_indices = None
        self.R = None
        self.y = None


class OneHotAndLinear(nn.Linear):
    """Combines one-hot encoding and linear projection in a single efficient operation
    to convert categorical indices to embeddings.

    Parameters
    ----------
    num_classes : int
        Number of distinct categories for one-hot encoding

    embed_dim : int
        Output embedding dimension
    """

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__(num_classes, embed_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

    def forward(self, src: Tensor) -> Tensor:
        """Transform integer indices to dense embeddings.

        Parameters
        ----------
        src : Tensor
            Integer tensor of shape (batch_size, sequence_length) containing category indices

        Returns
        -------
        Tensor
            Embedded representation of shape (batch_size, sequence_length, embed_dim)
        """
        # Convert indices to one-hot vectors and apply linear projection
        one_hot = F.one_hot(src.long(), self.num_classes).to(src.dtype)
        return F.linear(one_hot, self.weight, self.bias)


class SkippableLinear(nn.Linear):
    """Linear layer that handles inputs where all values equal `skip_value`.

    First applies the linear transformation to all inputs, then replaces outputs for inputs
    where all values equal `skip_value` with the `skip_value`.

    Parameters
    ----------
    in_features : int
        Size of each input sample

    out_features : int
        Size of each output sample

    bias : bool, default=True
        If set to False, the layer will not learn an additive bias

    skip_value : float, default=-100.0
        Value used to mark inputs that should be skipped
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, skip_value: float = -100.0):
        super().__init__(in_features, out_features, bias)
        self.skip_value = skip_value

    def forward(self, src: Tensor) -> Tensor:
        """Forward pass that handles inputs flagged with `skip_value`.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., in_features)

        Returns
        -------
        Tensor
            Output tensor of shape (..., out_features) where rows corresponding
            to skipped inputs are filled with `skip_value`
        """

        out = F.linear(src, self.weight, self.bias)
        skip_mask = (src == self.skip_value).all(dim=-1)
        if skip_mask.any():
            out[skip_mask] = self.skip_value

        return out


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture.

    Parameters
    ----------
    in_dim : int
        Input feature dimension

    out_dim : Optional[int], default=None
        Output dimension. If None, uses the last hidden dimension

    hidden_dims : List[int], default=[256, 256, 256]
        Dimensions of hidden layers

    activation : str, default='gelu'
        Activation function: 'relu', 'gelu', 'leaky_relu', or 'tanh'

    bias : bool, default=True
        Whether to include bias terms in linear layers
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        hidden_dims: List[int] = [256, 256, 256],
        activation: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        # Build network architecture
        act_fn = self.get_activation(activation)
        layers = []

        # Create hidden layers with activations
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=bias))
            layers.append(act_fn())
            prev_dim = hidden_dim

        # Optional output projection
        if out_dim is not None:
            layers.append(nn.Linear(prev_dim, out_dim, bias=bias))

        self.net = nn.Sequential(*layers)

    @staticmethod
    def get_activation(activation: str) -> nn.Module:
        """Get activation function class from string name.

        Parameters
        ----------
        activation : str
            Name of activation function

        Returns
        -------
        class
            PyTorch activation function class
        """

        activation_map = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }

        if activation not in activation_map:
            raise ValueError(f"Unknown activation: {activation}. Supported: {list(activation_map.keys())}")

        return activation_map[activation]

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass through the MLP.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (..., in_dim)

        Returns
        -------
        Tensor
            Output tensor of shape (..., out_dim or last_hidden_dim)
        """
        return self.net(X)


class MultiheadAttention(nn.MultiheadAttention):
    """Enhanced multi-head attention with rotary positional embedding support.

    This extends PyTorch's MultiheadAttention to support rotary position embeddings (RoPE)
    and specialized attention masking when `attn_mask` is an integer. The implementation always
    uses `batch_first=True`, meaning all input tensors have shape (..., seq_len, embed_dim).

    Parameters
    ----------
    embed_dim : int
        Model dimension (total size of each attention head combined)

    num_heads : int
        Number of attention heads

    dropout : float, default=0.0
        Dropout probability applied to attention weights

    References
    ----------
    .. [1] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
           https://arxiv.org/abs/2104.09864
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__(embed_dim, num_heads, dropout, batch_first=True)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor | int] = None,
        rope: Optional[RotaryEmbedding] = None,
    ) -> Tensor:
        """Compute multi-head attention with support for rotary positional encoding.

        Parameters
        ----------
        query : Tensor
            Query tensor of shape (..., tgt_len, embed_dim)

        key : Tensor
            Key tensor of shape (..., src_len, embed_dim)

        value : Tensor
            Value tensor of shape (..., src_len, embed_dim)

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

        rope : Optional[RotaryEmbedding]
            Rotary positional encoding

        Returns
        -------
        Tensor
            Attention output of shape (..., tgt_len, embed_dim)
        """

        if isinstance(attn_mask, int):
            assert key_padding_mask is None, "key_padding_mask is not supported with attn_mask as int"
            assert rope is None, "Rotary position embedding is not supported with attn_mask as int"

        return multi_head_attention_forward(
            query,
            key,
            value,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rope=rope,
        )


class MultiheadAttentionBlock(nn.TransformerEncoderLayer):
    """Attention block supporting rotary positional encoding.

    Parameters
    ----------
    d_model : int
        Model dimension

    nhead : int
        Number of attention heads

    dim_feedforward : int
       Dimension of the feedforward network

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
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
    ):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, norm_first=norm_first, batch_first=True)
        del self.self_attn
        self.attn = MultiheadAttention(d_model, nhead, dropout)
        self.init_weights()

    def init_weights(self):
        """Initialize projection layers to zero for stable training."""
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor | int] = None,
        rope: Optional[RotaryEmbedding] = None,
    ) -> Tensor:
        """Process input through attention with optional rotary positional encoding.

        Parameters
        ----------
        q : Tensor
            Query tensor of shape (..., tgt_len, d_model)

        k : Tensor
            Key tensor of shape (..., src_len, d_model)
            If None, uses q for self-attention.

        v : Tensor
            Value tensor of shape (..., src_len, d_model)
            If None, uses q for self-attention.

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

        rope : Optional[RotaryEmbedding]
            Rotary positional encoding

        Returns
        -------
        Tensor
            Output tensor of shape (..., tgt_len, d_model)
        """

        if isinstance(attn_mask, int):
            assert key_padding_mask is None, "key_padding_mask is not supported with attn_mask as int"
            assert rope is None, "Rotary position embedding is not supported with attn_mask as int"
        else:
            # Convert masks to correct dtype for compatibility
            key_padding_mask = F._canonical_mask(
                mask=key_padding_mask,
                mask_name="key_padding_mask",
                other_type=F._none_or_dtype(attn_mask),
                other_name="src_mask",
                target_type=q.dtype,
            )
            attn_mask = F._canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=q.dtype,
                check_other=False,
            )

        # Use q as k,v if not provided
        k = q if k is None else k
        v = q if v is None else v

        # Apply layer depending on normalization order
        x = q
        if self.norm_first:
            # Pre-norm: normalize before attention and FFN
            attn = self._attn_block(self.norm1(q), self.norm1(k), self.norm1(v), key_padding_mask, attn_mask, rope)
            x = x + attn
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-norm: normalize after attention and FFN
            attn = self._attn_block(q, k, v, key_padding_mask, attn_mask, rope)
            x = self.norm1(x + attn)
            x = self.norm2(x + self._ff_block(x))

        return x

    def _attn_block(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor | int],
        rope: Optional[RotaryEmbedding],
    ) -> Tensor:
        attn = self.attn(q, k, v, key_padding_mask, attn_mask, rope)
        return self.dropout1(attn)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class InducedSelfAttentionBlock(nn.Module):
    """Induced Self-Attention for efficient O(n) attention on large sets.

    This module implements a bottleneck attention mechanism using a small set of
    learned inducing points that mediate interactions between input elements.
    The complexity is reduced from O(n²) to O(n) by:

    1. Projecting inputs onto inducing points (size m << n)
    2. Propagating information through these inducing points
    3. Projecting back to the original sequence

    Parameters
    ----------
    d_model : int
        Model dimension

    nhead : int
        Number of attention heads

    dim_feedforward : int
        Dimension of the feedforward network

    num_inds : int
        Number of inducing points (controls capacity vs. efficiency)

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)

    skip_value : float, default=-100.0
        Value used to mark inputs that should be skipped

    References
    ----------
    .. [1] Lee et al. "Set Transformer: A Framework for Attention-based
           Permutation-Invariant Neural Networks", ICML 2019
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        skip_value: float = -100.0,
    ):
        super().__init__()
        self.skip_value = skip_value

        # Two-stage attention mechanism
        self.multihead_attn1 = MultiheadAttentionBlock(d_model, nhead, dim_feedforward, dropout, activation, norm_first)
        self.multihead_attn2 = MultiheadAttentionBlock(d_model, nhead, dim_feedforward, dropout, activation, norm_first)

        # Learnable inducing points
        self.num_inds = num_inds
        self.ind_vectors = nn.Parameter(torch.empty(num_inds, d_model))
        nn.init.trunc_normal_(self.ind_vectors, std=0.02)

    def induced_attention(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        """Apply induced self-attention to input sequence.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model)

        train_size : Optional[int], default=None
            Position to split the input into training and test data

        Returns
        -------
        Tensor
            Output tensor with same shape as input
        """

        *batch_shape, _, d_model = src.shape
        ind_vectors = self.ind_vectors.expand(*batch_shape, self.num_inds, d_model)

        if train_size is None:
            hidden = self.multihead_attn1(ind_vectors, src, src)
        else:
            hidden = self.multihead_attn1(ind_vectors, src[..., :train_size, :], src[..., :train_size, :])

        out = self.multihead_attn2(src, hidden, hidden)

        return out

    def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        """Apply induced self-attention to input sequence.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model)

        train_size : Optional[int], default=None
            Position to split the input into training and test data. When provided,
            inducing points will only attend to training data in the first attention
            stage to prevent information leakage from test data during evaluation.

        Returns
        -------
        Tensor
            Output tensor with same shape as input
        """

        skip_mask = (src == self.skip_value).all(dim=(-2, -1))  # batch shape
        if skip_mask.any():
            if skip_mask.all():
                out = torch.full_like(src, self.skip_value)
            else:
                out = torch.empty_like(src)
                out[~skip_mask] = self.induced_attention(src[~skip_mask], train_size)
                out[skip_mask] = self.skip_value
        else:
            out = self.induced_attention(src, train_size)

        return out
