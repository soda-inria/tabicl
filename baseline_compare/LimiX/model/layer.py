from typing import Callable, Literal, Optional
import functools
import warnings

import torch
import torch.nn as nn
from torch.cuda import OutOfMemoryError
from torch.utils.checkpoint import checkpoint
from functools import partial
from torch.amp import autocast




try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func, flash_attn_varlen_qkvpacked_func
    from flash_attn import flash_attn_func
    HAVE_FLASH_ATTN = True
except (ModuleNotFoundError, ImportError):
    HAVE_FLASH_ATTN = False

FLASH_ATTN_RUNTIME_DISABLED = False

from typing_extensions import override

Activation = Literal['gelu']

ACTIVATION_FN: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'gelu': nn.GELU(), 
    'relu': nn.ReLU(),
}


def _is_flash_attention_runtime_error(err: RuntimeError) -> bool:
    return "invalid configuration argument" in str(err).lower()


def _disable_flash_attention_runtime(err: RuntimeError) -> None:
    global FLASH_ATTN_RUNTIME_DISABLED
    if not FLASH_ATTN_RUNTIME_DISABLED:
        warnings.warn(
            f"Disabling flash attention after CUDA runtime failure: {err}",
            RuntimeWarning,
            stacklevel=2,
        )
    FLASH_ATTN_RUNTIME_DISABLED = True

class LayerNormMixedPrecision(nn.LayerNorm):
    """
    When the embedding dimension is below 512, use half precision for computation to improve performance. 
    If the embedding dimension exceeds 512, it may cause training instability.
    """
    def forward(self, input: torch.Tensor):
        if input.dtype == torch.float16 and sum(self.normalized_shape) < 512:
            with autocast(device_type="cuda" if input.is_cuda else "cpu", enabled=False):
                return super().forward(input)
        else:
            return super().forward(input)

class MLP(torch.nn.Module):
    """Multi-Layer Perceptron"""
    def __init__(self, 
                 in_features: int, 
                 hidden_size:int, 
                 out_features: int, 
                 has_bias:bool, 
                 device: torch.device | None, 
                 dtype: torch.dtype | None,  
                 activation: Activation = 'gelu', 
                 depth:int=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.layers = []
       
        if depth == 1:
            self.layers.append(nn.Linear(in_features, out_features, bias=has_bias, device=device, dtype=dtype))
        else:
             # input layer
            self.layers.append(nn.Linear(in_features, hidden_size, bias=has_bias, device=device, dtype=dtype))
            self.layers.append(ACTIVATION_FN[self.activation])
            # hidden layers
            for i in range(depth - 2):
                self.layers.append(nn.Linear(hidden_size, hidden_size, bias=has_bias, device=device, dtype=dtype))
                self.layers.append(ACTIVATION_FN[self.activation])
            # output layer
            self.layers.append(nn.Linear(hidden_size, out_features, bias=has_bias, device=device, dtype=dtype))
            torch.nn.init.normal_(self.layers[-1].weight)
        self.mlp = nn.Sequential(*self.layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class MultiheadAttention(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        qkv_combined: bool = True,
        dropout:float=0,
        recompute:bool=False
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_combined = qkv_combined
        self.dropout = dropout
        self.recompute = recompute
        self.device = device
        self.dtype = dtype

        self.out_proj_weight = torch.nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.embed_dim, device=self.device, dtype=self.dtype))
        self.qkv_proj_weight = torch.nn.Parameter(torch.empty(3, self.num_heads, self.head_dim, self.embed_dim, device=device, dtype=dtype))

        torch.nn.init.normal_(self.out_proj_weight)
        nn.init.xavier_uniform_(self.qkv_proj_weight)

        self.q_proj_weight = None
        self.kv_proj_weight = None
        
        if recompute:
            self.forward = partial(checkpoint, self.forward, use_reentrant=False)  # type: ignore
    
    def get_cu_seqlens(self, batch_size: int, seqlen: int, device: torch.device) -> torch.Tensor:
        return torch.arange(
            0,
            (batch_size + 1) * seqlen,
            step=seqlen,
            dtype=torch.int32,
            device=device,
        )
    
    def compute_attention_by_torch(self, qkv:torch.Tensor|None, q:torch.Tensor|None, kv:torch.Tensor|None, attn_mask:torch.Tensor|None) -> torch.Tensor:
        '''Since flash attention does not support attn_mask, use scaled_dot_product_attention to compute attention when attn_mask is not None'''
        if qkv is not None:
            q, k, v = qkv.unbind(dim=-3)
        elif kv is not None and q is not None:
            k,v = kv.unbind(dim=-3)
        else:
            raise ValueError("When qkv is None, q and kv cannot both be None at the same time")
        assert q is not None and k is not None and v is not None, "q, k, and v must not be None"
        
        attention_outputs = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=attn_mask,
                dropout_p=self.dropout,
            )
        attention_outputs = attention_outputs.transpose(1, 2)
        return attention_outputs
    
    def compute_attention_by_flashattn(self, qkv:torch.Tensor|None, q:torch.Tensor|None, kv:torch.Tensor|None) -> torch.Tensor:
        "Compute attention using flash attention"
        assert HAVE_FLASH_ATTN, "Flash attention is not supported. Please install/reinstall flash attention."
        if self.qkv_combined and qkv is not None:
            B,S = qkv.shape[:2]
            atten_out = flash_attn_varlen_qkvpacked_func( # type: ignore
                        qkv.reshape(B * S, 3, self.num_heads, self.head_dim),
                        self.get_cu_seqlens(B, S, qkv.device),
                        S,
                        dropout_p=self.dropout,
                        softmax_scale=None,
                        causal=False,
                        return_attn_probs=False,
                        deterministic=False,
                    )
        elif not self.qkv_combined and q is not None and kv is not None:
            B,S = q.shape[:2]
            kv_shape = kv.shape
            atten_out = flash_attn_varlen_kvpacked_func( # type: ignore
                    q.reshape(B * S, self.num_heads, self.head_dim),
                    kv.reshape(B * kv_shape[1], 2, self.num_heads, self.head_dim),
                    self.get_cu_seqlens(B, S, q.device),
                    self.get_cu_seqlens(B, kv_shape[1], kv.device),
                    S,
                    kv_shape[1],
                    dropout_p=self.dropout,
                    causal=False,
                    return_attn_probs=False,
                    deterministic=False,
                )
        return atten_out # type: ignore
    
    def chunked_flash_attention(
            self,
            qkv: torch.Tensor | None,
            q: torch.Tensor | None,
            kv: torch.Tensor | None
    ) -> torch.Tensor:
        assert HAVE_FLASH_ATTN, "Flash attention is not supported. Please install/reinstall flash attention."

        chunk_size = 1000000000 // 2 // 192 // qkv.shape[1]

        if self.qkv_combined and qkv is not None:
            B, S = qkv.shape[:2]
            # Split into chunks along sequence dimension
            atten_out = torch.empty(B, S, self.num_heads * self.head_dim, device=qkv.device, dtype=qkv.dtype)

            for i in range(0, B, chunk_size):
                chunk_end = min(i + chunk_size, B)
                # qkv_chunk = qkv[:, i:chunk_end]
                qkv_chunk = qkv[i:chunk_end]

                chunk_B, chunk_S = qkv_chunk.shape[:2]
                chunk_out = flash_attn_varlen_qkvpacked_func(
                    qkv_chunk.reshape(chunk_B * chunk_S, 3, self.num_heads, self.head_dim),
                    self.get_cu_seqlens(chunk_B, chunk_S, qkv.device),
                    chunk_S,
                    dropout_p=self.dropout,
                    softmax_scale=None,
                    causal=False,
                    return_attn_probs=False,
                    deterministic=False,
                )
                atten_out[i:chunk_end] = chunk_out.reshape(chunk_B, chunk_S, -1)

        elif not self.qkv_combined and q is not None and kv is not None:
            B, S = q.shape[:2]
            kv_shape = kv.shape
            # Split into chunks along sequence dimension for Q
            atten_out = torch.empty(B, S, self.num_heads * self.head_dim, device=q.device, dtype=q.dtype)

            for i in range(0, B, chunk_size):
                chunk_end = min(i + chunk_size, B)
                q_chunk = q[i:chunk_end]

                chunk_B, chunk_S = q_chunk.shape[:2]
                chunk_out = flash_attn_varlen_kvpacked_func(
                    q_chunk.reshape(chunk_B * chunk_S, self.num_heads, self.head_dim),
                    kv.reshape(B * kv_shape[1], 2, self.num_heads, self.head_dim),
                    self.get_cu_seqlens(chunk_B, chunk_S, q.device),
                    self.get_cu_seqlens(B, kv_shape[1], kv.device),
                    chunk_S,
                    kv_shape[1],
                    dropout_p=self.dropout,
                    causal=False,
                    return_attn_probs=False,
                    deterministic=False,
                )
                atten_out[i:chunk_end] = chunk_out.reshape(chunk_B, chunk_S, -1)

            # Concatenate all query tiles
        return atten_out.reshape(B * S, self.num_heads, self.head_dim)

    def caculate_attention_score(self, q: torch.Tensor | None, k: torch.Tensor | None) -> torch.Tensor:
        if len(q.shape) == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
        logits = torch.einsum("b q h d, b k h d -> b q k h", q, k)
        logits *= torch.sqrt(torch.tensor(1.0 / q.shape[-1])).to(k.device)
        ps = torch.softmax(logits.float(), dim=2).to(torch.float16).mean(dim=-1)
        del logits
        return ps

    def chunked_caculate_attention_score(self, q: torch.Tensor | None, k: torch.Tensor | None) -> torch.Tensor:
        if len(q.shape) == 4:
            B, S, H, D = q.shape
        else:
            S, H, D = q.shape
        try:
            if len(q.shape) == 3:
                chunk_size = max(10000000 // 2 // 192 // q.shape[0], 1)
                ps = torch.zeros(1, q.shape[0], k.shape[0], device=q.device, dtype=torch.float16)
                for i in range(0, S, chunk_size):
                    chunk_end = min(i + chunk_size, q.shape[0])
                    q_chunk = q[i:chunk_end]
                    k_chunk = k
                    ps_chunk = self.caculate_attention_score(q_chunk.to(q.device), k_chunk.to(k.device))
                    ps[:, i:chunk_end] = ps_chunk.to(q.device)
            else:
                chunk_size = max(2000000000 // 2 // 192 // q.shape[1] // k.shape[1], 1)
                ps = torch.zeros(B, q.shape[1], k.shape[1], device=q.device, dtype=torch.float16)
                for i in range(0, B, chunk_size):
                    chunk_end = min(i + chunk_size, B)
                    q_chunk = q[i:chunk_end]
                    k_chunk = k[i:chunk_end]
                    ps_chunk = self.caculate_attention_score(q_chunk.to(q.device), k_chunk.to(k.device))
                    ps[i:chunk_end] = ps_chunk.to(q.device)
        except OutOfMemoryError as e:
            attention_device = torch.device("cpu")
            if len(q.shape) == 3:
                chunk_size = max(10000000 // 2 // 192 // q.shape[0], 1)
                ps = torch.zeros(1, q.shape[0], k.shape[0], device=q.device, dtype=torch.float16)
                for i in range(0, S, chunk_size):
                    chunk_end = min(i + chunk_size, q.shape[0])
                    q_chunk = q[i:chunk_end]
                    k_chunk = k
                    ps_chunk = self.caculate_attention_score(q_chunk.to(q.device), k_chunk.to(k.device))
                    ps[:, i:chunk_end] = ps_chunk.to(attention_device)
            else:
                chunk_size = max(2000000000 // 2 // 192 // q.shape[1] // k.shape[1], 1)
                ps = torch.zeros(B, q.shape[1], k.shape[1], device=attention_device, dtype=torch.float16)
                for i in range(0, B, chunk_size):
                    chunk_end = min(i + chunk_size, B)
                    q_chunk = q[i:chunk_end]
                    k_chunk = k[i:chunk_end]
                    ps_chunk = self.caculate_attention_score(q_chunk.to(q.device), k_chunk.to(k.device))
                    ps[i:chunk_end] = ps_chunk.to(attention_device)
        return ps.to(torch.float16)

    @override
    def forward(self, 
                x: torch.Tensor, 
                x_kv: Optional[torch.Tensor] = None, 
                copy_first_head_kv: bool = False,
                attn_mask: torch.Tensor | None = None, 
                calculate_sample_attention:bool=False, 
                calculate_feature_attention:bool=False) -> tuple[torch.Tensor,torch.Tensor | None,torch.Tensor | None]:
        """
        x: [batch_size, seq_len, feature, embed_dim]
        kv: Optional[batch_size, seq_len_kv, feature, embed_dim] — only needed if qkv_combined=False
        copy_first_head: Reuse the results from the first attention head
        """
        # feature attention: [B S F E]  
        # item attention: [B F S E]
        # B, T, C = x.shape
        B, S, _, _ = x.shape
        assert x.shape[-1] == self.embed_dim

        x = x.reshape(-1, *x.shape[-2:])
        BS, F, E = x.shape
        
        qkv = None
        q = None
        kv = None
        feature_attention=None
        sample_attention=None
        # batch_size = None
        # seqlen = None
        if self.qkv_combined:
            qkv = torch.einsum("... s, j h d s -> ... j h d", x, self.qkv_proj_weight)
        else:
            self.q_proj_weight = self.qkv_proj_weight[0]
            self.kv_proj_weight = self.qkv_proj_weight[1:]
            assert x_kv is not None, "kv combined attention requires kv input"
            x_kv = x_kv.reshape(-1, *x_kv.shape[-2:])
            q = torch.einsum("... s, h d s -> ... h d", x, self.q_proj_weight)
            if copy_first_head_kv:
                kv_weights = self.kv_proj_weight[:,:1]
                kv = torch.einsum("... s, j h d s -> ... j h d", x_kv, kv_weights)
                expand_shape = [-1 for _ in kv.shape]
                expand_shape[-2] = self.num_heads
                kv = kv.expand(*expand_shape)
            else:
                kv = torch.einsum("... s, j h d s -> ... j h d", x_kv, self.kv_proj_weight)

        use_flash_attention = (
            attn_mask is None
            and HAVE_FLASH_ATTN
            and not FLASH_ATTN_RUNTIME_DISABLED
            and self.qkv_proj_weight.device.type == "cuda"
        )

        try:
            if use_flash_attention:
                atten_out = self.compute_attention_by_flashattn(qkv, q, kv)
            else:
                atten_out = self.compute_attention_by_torch(qkv, q, kv, attn_mask)
        except OutOfMemoryError as e:
            atten_out = self.chunked_flash_attention(qkv, q, kv)
        except RuntimeError as e:
            if use_flash_attention and _is_flash_attention_runtime_error(e):
                _disable_flash_attention_runtime(e)
                atten_out = self.compute_attention_by_torch(qkv, q, kv, attn_mask)
            else:
                raise

        atten_out = atten_out.reshape(BS, F, self.num_heads, self.head_dim)

        if qkv is not None:
            q, k, v = qkv.unbind(dim=2)
        else:
            k,v=kv.unbind(dim=2)
        if calculate_feature_attention:
            try:
                feature_attention = self.caculate_attention_score(q, k)
            except (OutOfMemoryError, RuntimeError) as e:
                feature_attention = self.chunked_caculate_attention_score(q, k)

        if calculate_sample_attention:
            try:
                sample_attention = self.caculate_attention_score(q[-1], k[-1])
            except (OutOfMemoryError, RuntimeError) as e:
                sample_attention = self.chunked_caculate_attention_score(q[-1], k[-1])
        out = torch.einsum(
            "... h d, h d s -> ... s",
            atten_out,
            self.out_proj_weight,
        )

        return out.reshape(B, S, *out.shape[1:]),feature_attention,sample_attention

class EncoderBaseLayer(nn.Module):
    "Base encoder layer of the Transformer model"
    def __init__(self, 
                 nhead: int, 
                 embed_dim: int, 
                 hid_dim:int, 
                 dropout: float=0,
                 pre_norm: bool=False,
                 activation: str='gelu',
                 layer_norm_eps: float=1e-5,
                 device: torch.device|None=None,
                 dtype: torch.dtype|None=None,
                 recompute_attn: bool=False,
                 mlp_use_residual:bool=False,
                 layer_arch: str = 'fmfmsm',
                 seq_attn_isolated: bool = False,
                 seq_attn_serial: bool = False,  
                 self_share_all_kv_heads: bool = False,
                 cross_share_all_kv_heads: bool = True,
                 ):
        super().__init__()
        self.nhead = nhead
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.device = device
        self.dtype = dtype
        self.layer_arch = layer_arch
        self.head_dim = self.embed_dim // self.nhead
        self.recompute_attn = recompute_attn
        
        self.feature_attentions = []
        self.sequence_attentions = []
        self.mlp = []
        self.feature_attn_num = 1           # feature attention number
        self.seq_attn_num = 1             # sequence attention number
        self.mlp_num = 1                    # mlp number
        if layer_arch == 'fmfmsm':
            self.feature_attn_num = 2
            self.mlp_num = 3
        
        self.self_share_all_kv_heads = self_share_all_kv_heads
        self.cross_share_all_kv_heads = cross_share_all_kv_heads
        self.seq_attn_serial = seq_attn_serial
        self.seq_attn_isolated = seq_attn_isolated

        if self.seq_attn_isolated:
            self.seq_attn_num *= 2

        # attention+MLP
        self.feature_attentions = nn.ModuleList(
                                                    [
                                                        MultiheadAttention(
                                                                embed_dim=self.embed_dim,
                                                                num_heads=self.nhead,
                                                                device=self.device,
                                                                dtype=self.dtype,
                                                                qkv_combined=True,
                                                                dropout=self.dropout,
                                                                recompute=self.recompute_attn,
                                                        ) 
                                                        for _ in range(self.feature_attn_num)
                                                    ]
                                                )
        self.sequence_attentions = nn.ModuleList(
                                                    [
                                                        MultiheadAttention(
                                                            embed_dim=self.embed_dim,
                                                            num_heads=self.nhead,
                                                            device=self.device,
                                                            dtype=self.dtype,
                                                            qkv_combined=False,
                                                            dropout=self.dropout,
                                                            recompute=self.recompute_attn,
                                                        ) 
                                                        for _ in range(self.seq_attn_num)
                                                    ]
                                                )
        self.mlp = nn.ModuleList(
                                    [
                                        MLP(
                                            in_features=self.embed_dim,
                                            hidden_size=self.hid_dim,
                                            out_features=self.embed_dim,
                                            has_bias=False,
                                            device=self.device,
                                            dtype=self.dtype,
                                            activation=self.activation,
                                            depth=2,
                                        ) 
                                        for _ in range(self.mlp_num)
                                    ]
                                 )
        
        self.layer_steps = []
        if self.layer_arch == 'fmfmsm':
            assert len(self.feature_attentions) >= 2 and len(self.sequence_attentions) >= 1 and len(self.mlp) >= 3
            self.layer_steps = [
                                partial(
                                    self.call_features_attention,
                                    index=0
                                ),
                                self.mlp[0],
                                partial(
                                    self.call_features_attention,
                                    index=1
                                ),
                                self.mlp[1],
                                partial(
                                    self.call_sequence_attention,
                                    index=0
                                ),
                                self.mlp[2]
            ]
        elif self.layer_arch == 'smf':
            assert len(self.feature_attentions) >= 1 and len(self.sequence_attentions) >= 1 and len(self.mlp) >= 1
            self.layer_steps = [
                                partial(
                                    self.call_sequence_attention,
                                    index=0
                                ),
                                self.mlp[0],
                                partial(
                                    self.call_features_attention,
                                    index=0
                                )
            ]
        else:
            raise ValueError(f"Unsupport layr arch: {self.layer_arch}")
    
        self.layer_norms = nn.ModuleList(
            [
                LayerNormMixedPrecision(normalized_shape=self.embed_dim, eps=self.layer_norm_eps, 
                                        elementwise_affine=False, device=self.device, dtype=self.dtype)
                for _ in range(len(self.layer_steps))
            ]
        )
    
    def create_attn_mask(self, q_mask:torch.Tensor, k_mask:torch.Tensor)->torch.Tensor:
        """
        Create attention mask
        
        Args:
            q_mask (torch.Tensor): Query sequence mask, with shape [batch_size, head_count, q_seq_len]
            k_mask (torch.Tensor): Key sequence mask, with shape   [batch_size, head_count, k_seq_len]
        
        Returns:
            torch.Tensor: attention mask, with shape [batch_size, head_count, q_seq_len, k_seq_len]
        """
        _, _, q_seq_len = q_mask.shape
        _, _, k_seq_len = k_mask.shape
        
        q_mask_bool = q_mask.bool()  # [batch_size, head_count, q_seq_len]
        k_mask_bool = k_mask.bool()  # [batch_size, head_count, k_seq_len]
        
        q_expanded = q_mask_bool.unsqueeze(-1)
        k_expanded = k_mask_bool.unsqueeze(-2)
        
        valid_attn = q_expanded & k_expanded
        attn_mask = ~valid_attn
        _, _, q_seq_len, k_seq_len = attn_mask.shape
        attn_mask = attn_mask.reshape(-1, q_seq_len, k_seq_len)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, 6, -1, -1)
        
        return attn_mask

    def call_features_attention(self, x: torch.Tensor, feature_atten_mask: torch.Tensor | None, eval_pos: int,
                                index: int = 0,calculate_feature_attention:bool=False):
        assert len(self.feature_attentions) > index
        attn_mask = None
        if feature_atten_mask is not None:
            attn_mask = self.create_attn_mask(feature_atten_mask, feature_atten_mask)
        return self.feature_attentions[index](
                                x,
                                x_kv=None,
                                attn_mask=attn_mask,
                                calculate_feature_attention=calculate_feature_attention
                            )

    def call_sequence_attention(self, x: torch.Tensor, feature_atten_mask: torch.Tensor | None, eval_pos: int,
                                index: int = 0,calculate_sample_attention:bool=False):
        assert len(self.sequence_attentions) > index
        sample_attention=None
        index1 = index*2 if self.seq_attn_isolated else index
        index2 = index1+1 if self.seq_attn_isolated else index1
        assert index2 < len(self.sequence_attentions), f"Error: index2({index2}) >= len(self.sequence_attentions)({len(self.sequence_attentions)})"

        x_train = self.sequence_attentions[index1](
                        x = x[:, :eval_pos].transpose(1, 2),
                        x_kv = x[:, :eval_pos].transpose(1, 2),
                        copy_first_head_kv = True if self.self_share_all_kv_heads else False,
                    )[0].transpose(1, 2)

        if self.seq_attn_serial:
            x[:, :eval_pos] = x_train

        if eval_pos < x.shape[1]:
            x_test,_,sample_attention = self.sequence_attentions[index2](
                                                    x=x[:, eval_pos:].transpose(1, 2),
                                                    x_kv=x[:, :eval_pos].transpose(1, 2),
                                                    copy_first_head_kv=True if self.cross_share_all_kv_heads else False,
                                                    calculate_sample_attention=calculate_sample_attention
                                                )
            x_test=x_test.transpose(1, 2)
        else:
            x_test = None
            print(f"\033[30;43mWarning: eval_pos >= x.shape[1]!\033[0m")
        
        if x_test is not None:
            return torch.cat([x_train, x_test], dim=1),None,sample_attention
        else:
            return x_train

    def forward(self, x: torch.Tensor, feature_atten_mask: torch.Tensor, eval_pos: int,**kwargs) -> tuple[torch.Tensor,torch.Tensor | None,torch.Tensor | None]:
        calculate_sample_attention = kwargs.get("calculate_sample_attention", False)
        calculate_feature_attention = kwargs.get("calculate_feature_attention", False)
        layer_idx = kwargs.get("layer_idx", 11)

        feature_attenion=None
        sample_attention=None
        for idx, (sublayer, layer_norm) in enumerate(zip(self.layer_steps, self.layer_norms)):
            if self.pre_norm:
                residual = x
                x = layer_norm(x)
                if idx == 2 and calculate_feature_attention and layer_idx == 11:
                    x, feature_attenion, _ = sublayer(x, feature_atten_mask, eval_pos,calculate_feature_attention=True)
                elif idx == 4 and calculate_sample_attention and layer_idx == 11:
                    x, _, sample_attention = sublayer(x, feature_atten_mask, eval_pos,calculate_sample_attention=True)
                else:
                    if isinstance(sublayer, functools.partial):
                        x = sublayer(x, feature_atten_mask, eval_pos)
                        if isinstance(x, tuple):
                            x = x[0]
                    else:
                        x = sublayer(x)
                        if isinstance(x, tuple):
                            x = x[0]
                x = x + residual
            else:
                residual = x
                if idx == 2 and calculate_feature_attention and layer_idx == 11:
                    x, feature_attenion, _ = sublayer(x, feature_atten_mask, eval_pos,calculate_feature_attention=True)
                    x = x + residual
                elif idx == 0 and calculate_sample_attention and layer_idx == 11:
                    x, _, sample_attention = sublayer(x, feature_atten_mask, eval_pos, calculate_sample_attention=True)
                    x = x + residual
                else:
                    if  isinstance(sublayer, functools.partial):
                        x = sublayer(x, feature_atten_mask, eval_pos)
                        if isinstance(x, tuple):
                            x = x[0]
                        x = x + residual
                    else:
                        x = sublayer(x)
                        if isinstance(x, tuple):
                            x = x[0]
                        x = x + residual
                x=layer_norm(x)
        return x,feature_attenion,sample_attention
                
class LayerStack(nn.Module):
    """
    A flexible container module similar to ``nn.Sequential`` that allows 
    keyword arguments to be passed through to each layer.
    """
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, **kwargs):
        for idx,layer in enumerate(self.layers):
            kwargs["layer_idx"] = idx
            x,feature_attention,sample_attention = layer(x,**kwargs)
        return x,feature_attention,sample_attention
