"""
Rotary positional embedding

Copy from https://github.com/lucidrains/rotary-embedding-torch
"""

from __future__ import annotations
from typing import Literal

from math import pi
from einops import rearrange, repeat

import torch
from torch import nn, einsum, Tensor, broadcast_tensors


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def broadcat(tensors, dim=-1):
    """broadcat, as tortoise-tts was using it"""
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim=dim)


def rotate_half_interleaved(x):
    """Interleaved rotation: pairs ``(0,1), (2,3), (4,5)``, etc.

    Given input ``[..., d]``, rearranges to ``[..., d/2, 2]`` and rotates pairs.
    Used by default in most RoPE implementations (e.g. LLaMA).
    """
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def rotate_half_contiguous(x):
    """Rotate by splitting into contiguous first and second halves.

    Given input ``[..., d]``, splits into ``[..., :d/2]`` and ``[..., d/2:]``
    and rotates. Returns ``[-x2, x1]`` where ``x1`` is the first half and
    ``x2`` is the second half.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


@torch.autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2, interleaved=True):
    """Apply rotary embeddings to tensor.

    Computes :math:`x \\cdot \\cos(\\theta) + \\text{rotate}(x) \\cdot \\sin(\\theta)`
    for the portion of the input that overlaps with the frequency tensor.

    Parameters
    ----------
    freqs : Tensor
        Frequency tensor for rotation. For interleaved mode, shape is
        ``(..., dim)`` where ``dim = 2 * half``. For non-interleaved mode,
        shape is ``(..., half)``.

    t : Tensor
        Input tensor to rotate.

    start_index : int, default=0
        Starting index for rotation in the last dimension.

    scale : float, default=1.0
        Scaling factor for the rotation.

    seq_dim : int, default=-2
        Sequence dimension.

    interleaved : bool, default=True
        If True, uses interleaved rotation where dimension pairs are
        ``(0,1), (2,3)``, etc. If False, uses non-interleaved rotation
        where the embedding is split into first half ``[0:d//2]`` and
        second half ``[d//2:d]``.

    Returns
    -------
    Tensor
        Rotated tensor, same shape as ``t``.
    """
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    if interleaved:
        rot_dim = freqs.shape[-1]
    else:
        rot_dim = freqs.shape[-1] * 2  # Non-interleaved: freqs has shape (L, half)

    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place
    # Formula: x * cos + rotate(x) * sin
    if interleaved:
        # Interleaved mode: dimension pairs are (0,1), (2,3), etc.
        # freqs has shape (..., rot_dim) where rot_dim = dim
        t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half_interleaved(t_middle) * freqs.sin() * scale)
    else:
        # Non-interleaved mode: split embedding into first half [0:d//2] and second half [d//2:d]
        # freqs has shape (..., half) for non-interleaved mode
        # Expand to full dim by concatenating: [f0, f1, ..., f0, f1, ...]
        cos_freq = torch.cat([freqs.cos(), freqs.cos()], dim=-1) * scale
        sin_freq = torch.cat([freqs.sin(), freqs.sin()], dim=-1) * scale
        t_transformed = (t_middle * cos_freq) + (rotate_half_contiguous(t_middle) * sin_freq)

    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)


def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    """learned rotation helpers"""
    if exists(freq_ranges):
        rotations = einsum("..., f -> ... f", rotations, freq_ranges)
        rotations = rearrange(rotations, "... r f -> ... (r f)")

    rotations = repeat(rotations, "... n -> ... (n r)", r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings for use in transformer models.

    Rotary embeddings encode positional information in a way that allows
    continuous rotation of embeddings, enhancing the model's ability to
    capture long-range dependencies and positional relations.

    Attributes
    ----------
    dim : int
        The dimension of the embeddings.

    interleaved : bool
        If True, uses interleaved rotation where dimension pairs are
        ``(0,1), (2,3)``, etc. If False, uses non-interleaved rotation
        where the embedding is split into first half ``[0:d//2]`` and
        second half ``[d//2:d]``.

    custom_freqs : Tensor or None
        Custom frequency tensor for the embeddings.
        If None, default frequencies based on ``freqs_for`` are used.

    freqs_for : {'lang', 'pixel', 'constant'}
        Specifies the type of frequencies to use:
        ``'lang'`` for language, ``'pixel'`` for image data, or
        ``'constant'`` for a fixed frequency.

    theta : float
        A base scaling factor for the rotary embeddings.

    max_freq : int
        The maximum frequency for pixel-based embeddings.

    num_freqs : int
        The number of frequencies to use when ``freqs_for='constant'``.

    learned_freq : bool
        If True, the frequencies are learnable parameters.

    use_xpos : bool
        If True, uses extrapolatable rotary embeddings (XPOS).

    xpos_scale_base : float
        The base scaling factor used for XPOS.

    interpolate_factor : float
        Factor by which the sequence length is interpolated.

    theta_rescale_factor : float
        Rescaling factor applied to theta for longer sequence lengths.

    seq_before_head_dim : bool
        If True, sequences are handled before the head dimension.

    cache_if_possible : bool
        If True, caches computed frequencies and scales to optimize
        performance.
    """

    def __init__(
        self,
        dim,
        interleaved=True,
        custom_freqs: Tensor | None = None,
        freqs_for: Literal["lang", "pixel", "constant"] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
        seq_before_head_dim=False,
        cache_if_possible=True,
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible

        self.tmp_store("cached_freqs", None)
        self.tmp_store("cached_scales", None)

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.tmp_store("dummy", torch.tensor(0))

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        # interleaved rotation mode
        self.interleaved = interleaved

        # xpos

        self.use_xpos = use_xpos
        if not use_xpos:
            self.tmp_store("scale", None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store("scale", scale)

        # add apply_rotary_emb as static method

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent=False)

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, scale=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or exists(
            scale
        ), "you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings"

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset)

        if self.interleaved:
            freqs = self.forward(seq, seq_len=seq_len, offset=offset)
        else:
            # For non-interleaved mode, compute freqs without repetition
            # freqs shape: (seq_len, half) instead of (seq_len, dim)
            freqs = einsum("..., f -> ... f", seq.type(self.freqs.dtype), self.freqs)

        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")

        return apply_rotary_emb(freqs, t, scale=default(scale, 1.0), seq_dim=seq_dim, interleaved=self.interleaved)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim=None, offset=0):
        dtype, device, seq_dim = q.dtype, q.device, default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        q_scale = k_scale = 1.0

        if self.use_xpos:
            seq = self.get_seq_pos(k_len, dtype=dtype, device=device)

            q_scale = self.get_scale(seq[-q_len:]).type(dtype)
            k_scale = self.get_scale(seq).type(dtype)

        rotated_q = self.rotate_queries_or_keys(q, seq_dim=seq_dim, scale=q_scale, offset=k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim=seq_dim, scale=k_scale**-1)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)

        if self.interleaved:
            freqs = self.forward(seq, seq_len=seq_len)
        else:
            # For non-interleaved mode, compute freqs without repetition
            freqs = einsum("..., f -> ... f", seq.type(self.freqs.dtype), self.freqs)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")
            scale = rearrange(scale, "n d -> n 1 d")

        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim, interleaved=self.interleaved)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale**-1, seq_dim=seq_dim, interleaved=self.interleaved)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(self, t: Tensor, seq_len: int | None = None, offset=0):
        assert self.use_xpos

        should_cache = self.cache_if_possible and exists(seq_len)

        if should_cache and exists(self.cached_scales) and (seq_len + offset) <= self.cached_scales.shape[0]:
            return self.cached_scales[offset : (offset + seq_len)]

        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = torch.cat((scale, scale), dim=-1)

        if should_cache:
            self.tmp_store("cached_scales", scale)

        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == "pixel":
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)

            freqs = self.forward(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    @torch.autocast("cuda", enabled=False)
    def forward(self, t: Tensor, seq_len=None, offset=0):
        should_cache = (
            self.cache_if_possible and not self.learned_freq and exists(seq_len) and self.freqs_for != "pixel"
        )

        if should_cache and exists(self.cached_freqs) and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return self.cached_freqs[offset : (offset + seq_len)].detach()

        freqs = self.freqs

        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        if should_cache:
            self.tmp_store("cached_freqs", freqs.detach())

        return freqs
