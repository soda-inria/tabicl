from __future__ import annotations

import os
import uuid
import math
import shutil
import psutil
import weakref
import itertools
from enum import Enum, auto
from dataclasses import dataclass
from collections import OrderedDict
from typing import List, Tuple, Dict, Callable, Iterator, Literal, Optional, Any, Union

import numpy as np
from tqdm.auto import tqdm

import torch
from torch import Tensor

from .kv_cache import KVCache
from .attention import flash_attn3_toggle


class MemoryEstimator:
    """Estimates peak activation memory requirements for different attention-based components.

    Peak inference memory refers to the maximum amount of memory (typically GPU memory) used during
    the inference phase of a model. This is the highest memory consumption observed at any point.

    The coefficients and intercepts for each component are derived through memory profiling and regression:

    1. Collect memory usage data by running models with different batch sizes and sequence lengths
    2. Fit a polynomial regression:

       .. math::

           c_1 \\cdot \\text{batch\\_size} + c_2 \\cdot \\text{seq\\_len}
           + c_3 \\cdot (\\text{batch\\_size} \\times \\text{seq\\_len}) + \\text{intercept}

    3. Use the fitted coefficients to estimate memory for new batch sizes and sequence lengths

    Memory profiling was conducted using float32 without automatic mixed precision (AMP).
    When using AMP, actual memory usage will be lower than the estimates provided by this class.
    """

    # Coefficients and intercepts for memory estimation (from profiling)
    coefficients: Dict[str, list] = {
        "tf_col": [1.456149314e-01, 1.94081457e-05, 4.88223400e-03],
        "tf_row": [-2.06831848e-05, 2.27205969e-04, 5.37117114e-03],
        "tf_icl": [-4.03932756e-02, 5.42811085e-07, 1.95312473e-02],
    }
    intercepts: Dict[str, float] = {
        "tf_col": 142.91294659096457,
        "tf_row": 138.53653545318957,
        "tf_icl": 142.84243874417552,
    }

    @staticmethod
    def estimate_peak_mem(
        batch_size: int,
        seq_len: int,
        enc_name: str,
        include_inputs: bool = True,
        in_dim: Optional[int] = None,
    ) -> float:
        """Estimate peak memory usage for a given component with specified batch size and sequence length.

        Parameters
        ----------
        batch_size : int
            Batch size for inference.

        seq_len : int
            Sequence length for inference.

        enc_name : str
            Model encoder name to estimate memory for. One of:
            - ``"tf_col"``: Column embedding encoder
            - ``"tf_row"``: Row interaction encoder
            - ``"tf_icl"``: In-context learning encoder

        include_inputs : bool, default=True
            Whether to include memory usage for input tensors.

        in_dim : Optional[int], default=None
            Model dimension for the encoder.

        Returns
        -------
        float
            Estimated peak memory usage in MB for the specified encoder.
        """
        coefs = MemoryEstimator.coefficients[enc_name]
        inter = MemoryEstimator.intercepts[enc_name]
        peak_activation_mem = coefs[0] * batch_size + coefs[1] * seq_len + coefs[2] * batch_size * seq_len + inter

        if include_inputs:
            assert in_dim is not None, "Input dimension must be provided for input memory estimation"
            bytes_per_element = 4  # float32
            n_elements = batch_size * seq_len * in_dim
            mem_inputs = n_elements * bytes_per_element / (1024**2)
            peak_activation_mem += mem_inputs

        return peak_activation_mem

    @staticmethod
    def estimate_batch_size(
        seq_len: int,
        target_memory: float,
        enc_name: str,
        include_inputs: bool = True,
        in_dim: Optional[int] = None,
    ) -> int:
        """Estimate the batch size that fits within the target memory budget.

        The memory model from ``estimate_peak_mem`` is linear in ``bs``:

        .. math::

            \\text{mem} = c_1 \\cdot bs + c_2 \\cdot seq + c_3 \\cdot bs \\cdot seq + \\text{intercept}

        When ``include_inputs=True``, an additional input-memory term is added:

        .. math::

            \\text{mem} \\mathrel{+}= bs \\cdot seq \\cdot \\text{in\\_dim} \\cdot 4 / 1024^2

        Solving for ``bs`` gives:

        .. math::

            bs = \\frac{\\text{target} - c_2 \\cdot seq - \\text{intercept}}
                      {c_1 + c_3 \\cdot seq + seq \\cdot \\text{in\\_dim} \\cdot 4 / 1024^2}

        Parameters
        ----------
        seq_len : int
            Sequence length for inference.

        target_memory : float
            Target memory usage in MB.

        enc_name : str
            Model encoder name to estimate memory for.

        include_inputs : bool, default=True
            Whether to include memory usage for input tensors.

        in_dim : Optional[int], default=None
            Model dimension for the encoder.

        Returns
        -------
        int
            Estimated batch size that fits within target memory constraints.
        """
        if target_memory <= 0:
            return 1
        coefs = MemoryEstimator.coefficients[enc_name]
        intercept = MemoryEstimator.intercepts[enc_name]

        numerator = target_memory - coefs[1] * seq_len - intercept
        denominator = coefs[0] + coefs[2] * seq_len
        if include_inputs and in_dim is not None:
            denominator += seq_len * in_dim * 4 / (1024**2)

        if denominator <= 0:
            return 1

        return max(1, int(numerator / denominator))


class OffloadMode(Enum):
    """Offload mode for intermediate results."""

    GPU = auto()  # Keep everything on GPU (fastest if it fits)
    CPU = auto()  # Offload to CPU pinned memory
    DISK = auto()  # Offload to disk via memory-mapped files
    AUTO = auto()  # Automatically choose based on available memory


@dataclass
class OffloadReason:
    """Structured reason for offload mode resolution."""

    key: str
    detail: Optional[str] = None

    def __str__(self):
        return f"{self.key}: {self.detail}" if self.detail else self.key


@dataclass
class OffloadConfig:
    """Configuration for offloading behavior."""

    mode: OffloadMode = OffloadMode.AUTO

    # Thresholds for AUTO mode
    auto_offload_threshold: float = 0.5  # Offload if output > this fraction of GPU memory
    cpu_safety_factor: float = 0.85  # Safety margin for CPU memory
    disk_safety_factor: float = 0.95  # Safety margin for disk space

    # CPU options
    max_pinned_memory_mb: float = 32768.0  # Max pinned memory (32GB default), use regular memory above this

    # Disk options
    disk_offload_dir: Optional[str] = None  # Directory for memory-mapped files
    disk_min_free_mb: float = 1024.0  # Minimum free disk space to maintain
    disk_flush_mb: float = 2048.0  # Flush memmap after writing this many MB
    disk_cleanup: bool = True  # Auto-cleanup disk files via weakref
    disk_file_prefix: str = ""  # Prefix for disk files
    disk_dtype: Optional[torch.dtype] = None  # Override dtype for disk storage

    # Async options
    use_async: bool = True  # Use async D2H copy
    async_depth: int = 4  # Number of pending async copies before blocking


class PinnedBufferPool:
    """Pool of pinned CPU memory buffers for efficient GPU-to-CPU transfers.

    Pinned (page-locked) memory enables faster async GPU-to-CPU data transfers
    by allowing the GPU to directly access the CPU memory without involving
    the CPU. This class maintains a pool of such buffers to avoid the overhead
    of repeated allocation and deallocation.

    The pool is organized by (shape, dtype) pairs, allowing efficient reuse
    of buffers with matching dimensions.

    Parameters
    ----------
    max_buffers_per_shape : int, default=4
        Maximum number of buffers to keep pooled for each unique (shape, dtype)
        combination. Higher values use more memory but reduce allocation frequency.

    Notes
    -----
    - Pinned memory locks physical RAM and cannot be swapped to disk
    - For very large tensors, consider using non-pinned memory to avoid
      exhausting physical memory
    - Buffers are automatically returned to the pool after use via `put()`
    """

    def __init__(self, max_buffers_per_shape: int = 4):
        self._pool: Dict[Tuple[Tuple[int, ...], torch.dtype], List[Tensor]] = {}
        self._max_per_shape = max_buffers_per_shape

    def get(self, shape: Tuple[int, ...], dtype: torch.dtype) -> Tensor:
        """Get a pinned buffer, creating one if necessary."""
        key = (shape, dtype)
        pool = self._pool.get(key)
        if pool:
            return pool.pop()
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)

    def put(self, buf: Tensor) -> None:
        """Return a buffer to the pool for reuse."""
        if not buf.is_pinned():
            return  # Don't pool non-pinned buffers
        key = (tuple(buf.shape), buf.dtype)
        pool = self._pool.setdefault(key, [])
        if len(pool) < self._max_per_shape:
            pool.append(buf)

    def clear(self) -> None:
        """Clear all pooled buffers."""
        self._pool.clear()


class DiskTensor:
    """A tensor backed by a memory-mapped file on disk.

    Uses numpy memmap for storage and provides a torch tensor view
    for seamless integration with PyTorch operations. This enables
    storing tensors larger than available RAM by using disk space.

    Memory-mapped files allow efficient random access to large files
    without loading them entirely into memory. The operating system
    handles paging data between disk and memory as needed.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the tensor.

    dtype : torch.dtype
        Data type of the tensor.

    path : str
        Path to the memory-mapped file.

    cleanup : bool
        Whether to delete the file when this object is garbage collected.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        path: str,
        cleanup: bool = True,
    ):
        self.shape = shape
        self.dtype = dtype
        self.path = path
        self._cleanup = cleanup

        # Handle dtype mapping (numpy doesn't support bfloat16)
        self._storage_dtype, self._np_dtype, self._needs_view = self._resolve_dtype(dtype)

        # Create the memory-mapped file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        nbytes = int(np.dtype(self._np_dtype).itemsize * math.prod(shape))

        # Pre-allocate file
        with open(path, "wb") as f:
            f.truncate(nbytes)

        # Create memmap and torch view
        self._memmap = np.memmap(path, mode="r+", dtype=self._np_dtype, shape=shape)
        self._tensor = torch.from_numpy(self._memmap)

        if self._needs_view:
            # View as the target dtype (e.g., bfloat16)
            self._tensor = self._tensor.view(dtype)

        # Setup cleanup finalizer
        if cleanup:
            self._finalizer = weakref.finalize(self, self._cleanup_file, path)

    @staticmethod
    def _resolve_dtype(dtype: torch.dtype) -> Tuple[torch.dtype, np.dtype, bool]:
        """Resolve torch dtype to numpy dtype for storage."""
        if dtype == torch.bfloat16:
            # Store as uint16, view as bfloat16
            return torch.uint16, np.uint16, True
        elif dtype == torch.float16:
            return torch.float16, np.float16, False
        elif dtype == torch.float32:
            return torch.float32, np.float32, False
        elif dtype == torch.float64:
            return torch.float64, np.float64, False
        elif dtype == torch.int32:
            return torch.int32, np.int32, False
        elif dtype == torch.int64:
            return torch.int64, np.int64, False
        else:
            # Default to float32
            return torch.float32, np.float32, False

    @staticmethod
    def _cleanup_file(path: str) -> None:
        """Cleanup function called by weakref.finalize."""
        try:
            os.remove(path)
        except (FileNotFoundError, OSError):
            pass

    @property
    def tensor(self) -> Tensor:
        """Get the torch tensor view of this disk tensor."""
        return self._tensor

    def __getitem__(self, indices) -> Tensor:
        """Index into the tensor."""
        return self._tensor[indices]

    def __setitem__(self, indices, value: Tensor) -> None:
        """Write to the tensor (automatically persists to disk)."""
        if value.device.type != "cpu":
            value = value.cpu()
        self._tensor[indices] = value

    def flush(self) -> None:
        """Flush changes to disk."""
        self._memmap.flush()

    @property
    def nbytes(self) -> int:
        """Return total size in bytes."""
        return self._memmap.nbytes


class AsyncCopyManager:
    """Manages asynchronous GPU-to-CPU copies using CUDA streams.

    Uses a dedicated CUDA stream for device-to-host (D2H) transfers, allowing
    GPU computation to overlap with data movement. The workflow is:

    1. GPU tensor is copied to a pinned buffer on the copy stream
    2. An event is recorded to track completion
    3. When the event completes, data is written to final target (CPU/disk)
    4. The pinned buffer is returned to the pool for reuse

    Parameters
    ----------
    device : torch.device
        The CUDA device to use for transfers.

    max_pending : int, default=4
        Maximum number of pending async copies before blocking. Higher values
        increase memory usage but can improve throughput by hiding latency.

    buffer_pool : PinnedBufferPool, optional
        Pool of pinned memory buffers to use. If None, a new pool is created.

    Notes
    -----
    - Only works with CUDA devices; falls back to sync copy for CPU
    - Pending copies are automatically drained when max_pending is reached
    - Call ``drain_all()`` to ensure all copies complete before using the data
    """

    def __init__(self, device: torch.device, max_pending: int = 4, buffer_pool: Optional[PinnedBufferPool] = None):
        self.device = device
        self.max_pending = max_pending
        self.buffer_pool = buffer_pool or PinnedBufferPool(max_buffers_per_shape=8)

        # Create dedicated copy stream
        self._copy_stream: Optional[torch.cuda.Stream] = None
        if device.type == "cuda":
            self._copy_stream = torch.cuda.Stream(device=device)

        # Pending copies: (pinned_buffer, target, indices, event)
        self._pending: List[Tuple[Tensor, Any, tuple, Optional[torch.cuda.Event]]] = []
        self._bytes_written: float = 0.0  # Track bytes in MB for flush control

    def submit_copy(
        self,
        gpu_tensor: Tensor,
        target: Union[Tensor, DiskTensor],
        indices: tuple,
    ) -> None:
        """Submit an async copy from GPU to target storage.

        Parameters
        ----------
        gpu_tensor : Tensor
            Source tensor on GPU.

        target : Union[Tensor, DiskTensor]
            Target storage (CPU tensor or disk tensor).

        indices : tuple
            Indices where to write in the target.
        """
        if self._copy_stream is None:
            # Fallback to sync copy if no CUDA
            target[indices] = gpu_tensor.cpu()
            self._bytes_written += gpu_tensor.numel() * gpu_tensor.element_size() / (1024 * 1024)
            return

        # Get a pinned buffer
        pinned_buf = self.buffer_pool.get(tuple(gpu_tensor.shape), gpu_tensor.dtype)

        # Capture compute stream before switching context
        compute_stream = torch.cuda.current_stream(device=self.device)

        # Tell the caching allocator that copy_stream is also using gpu_tensor's memory,
        # so it won't be reclaimed until the copy completes.
        gpu_tensor.record_stream(self._copy_stream)

        # Async copy GPU -> pinned buffer on dedicated stream
        with torch.cuda.stream(self._copy_stream):
            # Wait for compute stream to finish producing gpu_tensor
            self._copy_stream.wait_stream(compute_stream)
            pinned_buf.copy_(gpu_tensor, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self._copy_stream)

        self._pending.append((pinned_buf, target, indices, event))

        # Drain if too many pending
        while len(self._pending) >= self.max_pending:
            self._drain_one()

    def _drain_one(self) -> float:
        """Complete one pending copy, return bytes written in MB."""
        if not self._pending:
            return 0.0

        pinned_buf, target, indices, event = self._pending.pop(0)
        if event is not None:
            event.synchronize()

        target[indices] = pinned_buf

        bytes_mb = pinned_buf.numel() * pinned_buf.element_size() / (1024 * 1024)
        self._bytes_written += bytes_mb

        # Return buffer to pool
        self.buffer_pool.put(pinned_buf)

        return bytes_mb

    def drain_all(self) -> float:
        """Complete all pending copies, return total bytes written in MB."""
        while self._pending:
            self._drain_one()
        return self._bytes_written

    def get_bytes_written(self) -> float:
        """Get total bytes written so far in MB."""
        return self._bytes_written

    def reset_bytes_counter(self) -> None:
        """Reset the bytes written counter."""
        self._bytes_written = 0.0

    def clear(self) -> None:
        """Clear pending copies without completing them."""
        self._pending.clear()


class InferenceManager:
    """Manages memory-efficient model inference by automatically batching inputs.

    Key Features
    ------------
    1. Automatic Batch Size Estimation: Estimates safe batch sizes based on
       available GPU memory using pre-computed memory coefficients from profiling.

    2. Multi-dimensional Batching: Splits large inputs into smaller batches
       across multiple batch dimensions, processing them sequentially.

    3. Flexible Offloading: Supports three storage backends for outputs:
       - GPU: Keep everything in GPU memory (fastest, but limited by VRAM)
       - CPU: Offload to CPU memory with optional pinned memory for faster transfers
       - Disk: Offload to memory-mapped files for unlimited capacity

    4. OOM Recovery: Automatically reduces batch size when out-of-memory errors
       occur, retrying until inference succeeds or minimum batch size is reached.

    5. Async Data Transfer: Uses dedicated CUDA streams for overlapping
       computation with GPU-to-CPU data movement.

    6. Result Merging: Combines outputs from multiple batches into a single
       contiguous tensor.

    Offloading Modes
    ----------------
    - ``"gpu"``: Keep output on GPU. Fastest option if the output fits in VRAM.
    - ``"cpu"``: Offload to CPU memory. Uses pinned memory for outputs smaller than
      ``max_pinned_memory_mb``, otherwise uses regular (swappable) memory.
    - ``"disk"``: Offload to memory-mapped files. Lowest memory footprint, useful
      for very large outputs that exceed both GPU and CPU memory.
    - ``"auto"``: Automatically choose based on available memory. Prefers GPU if
      output fits, then CPU, then disk.

    Parameters
    ----------
    enc_name : str
        Name of encoder for memory estimation. This determines which set of
        pre-computed memory coefficients to use. One of:
        - ``"tf_col"``: Column embedding encoder
        - ``"tf_row"``: Row interaction encoder
        - ``"tf_icl"``: In-context learning encoder

    out_dim : int
        Output dimension of the model. Used for pre-allocating output tensors
        with the correct shape.

    out_no_seq : bool, default=False
        Whether to remove the sequence dimension from output tensor. If True,
        output shape will be ``(..., out_dim)`` instead of ``(..., seq_len, out_dim)``.
    """

    def __init__(self, enc_name: str, out_dim: int, out_no_seq: bool = False):
        self.enc_name = enc_name
        self.out_dim = out_dim
        self.out_no_seq = out_no_seq
        self._is_configured = False

        # Internal state
        self._buffer_pool: Optional[PinnedBufferPool] = None
        self._disk_finalizers: List[weakref.finalize] = []

    def configure(
        self,
        min_batch_size: int = 1,
        safety_factor: float = 0.8,
        offload: Union[bool, Literal["auto", "gpu", "cpu", "disk"], OffloadMode] = "auto",
        auto_offload_threshold: float = 0.5,
        device: Optional[Union[str, torch.device]] = None,
        use_amp: bool = True,
        use_fa3: bool = True,
        verbose: bool = False,
        # Disk options
        disk_offload_dir: Optional[str] = None,
        disk_min_free_mb: float = 1024.0,
        disk_flush_mb: float = 8192.0,
        disk_cleanup: bool = True,
        disk_file_prefix: str = "",
        disk_dtype: Optional[torch.dtype] = None,
        # CPU options
        cpu_safety_factor: float = 0.85,
        disk_safety_factor: float = 0.95,
        max_pinned_memory_mb: float = 32768.0,  # 32GB
        # Async options
        use_async: bool = True,
        async_depth: int = 4,
    ):
        """Configure inference parameters. Must be called before using InferenceManager.

        Parameters
        ----------
        min_batch_size : int, default=1
            Minimum batch size to try before raising an error. If OOM occurs even
            with this batch size, inference cannot proceed and an error is raised.

        safety_factor : float, default=0.8
            Factor (0-1) to multiply estimated batch size by for conservative memory
            usage. Lower values are safer but may result in more batches.

        offload : Union[bool, str, OffloadMode], default="auto"
            Where to store output tensors during inference:

            - ``"gpu"`` or ``False``: Keep on GPU. Fastest but limited by VRAM.
            - ``"cpu"`` or ``True``: Offload to CPU memory. Uses pinned memory for
              faster transfers if output size <= ``max_pinned_memory_mb``.
            - ``"disk"``: Offload to memory-mapped files. Supports outputs larger than RAM.
            - ``"auto"``: Automatically choose based on available memory. Uses
              ``auto_offload_threshold`` to decide when to offload from GPU.

        auto_offload_threshold : float, default=0.5
            GPU memory threshold (0-1) for automatic offloading. Only used when
            ``offload="auto"``. If output size exceeds this fraction of available
            GPU memory, outputs are offloaded to CPU or disk.

        device : Optional[str or torch.device], default=None
            Device to use for computation. If None, defaults to CUDA if available,
            otherwise CPU.

        use_amp : bool, default=True
            Whether to use automatic mixed precision (AMP) during inference.
            AMP can significantly reduce memory usage and improve performance.

        use_fa3 : bool, default=True
            Whether to use Flash Attention 3 during inference. FA3 can improve
            performance for large sequences but may be slower for small datasets.
            Only effective when the flash_attn_interface package is installed.

        verbose : bool, default=False
            Whether to show progress bars and detailed logging information
            during inference.

        disk_offload_dir : Optional[str], default=None
            Directory for disk offloading (memory-mapped files). If None,
            disk offloading is disabled. When CPU memory is insufficient and
            ``disk_offload_dir`` is None, an error will be raised.

        disk_min_free_mb : float, default=1024.0
            Minimum free disk space (in MB) to maintain. The manager will not
            use disk offloading if it would reduce free space below this threshold.

        disk_flush_mb : float, default=8192.0
            Flush memory-mapped files to disk after writing this many MB.
            Larger values may improve performance but use more memory.

        disk_cleanup : bool, default=True
            Whether to automatically cleanup disk files when tensors are garbage
            collected. If False, files must be manually deleted.

        disk_file_prefix : str, default=""
            Prefix for disk file names. Useful for identifying files from
            different inference runs.

        disk_dtype : Optional[torch.dtype], default=None
            Override dtype for disk storage. If None, uses the input dtype.
            Can be used to save disk space by storing in lower precision.

        cpu_safety_factor : float, default=0.85
            Safety margin for CPU memory. Only this fraction of available CPU
            memory will be considered usable for offloading.

        disk_safety_factor : float, default=0.95
            Safety margin for disk space. Only this fraction of available disk
            space will be considered usable for offloading.

        max_pinned_memory_mb : float, default=32768.0
            Maximum size (in MB) for pinned CPU memory allocation. Outputs larger
            than this will use regular (non-pinned) memory which can be swapped.
            Pinned memory is faster for GPU-CPU transfers but locks physical RAM.
            Set to 0 to disable pinned memory entirely.

        use_async : bool, default=True
            Whether to use async D2H (device-to-host) copy with CUDA streams.
            Enables overlapping computation with data transfer for better throughput.

        async_depth : int, default=4
            Maximum number of pending async copies before blocking. Higher values
            can improve throughput but increase memory usage for pinned buffers.
        """
        self.min_batch_size = int(min_batch_size)
        self.safety_factor = float(safety_factor)
        self.auto_offload_threshold = float(auto_offload_threshold)
        self.use_amp = bool(use_amp)
        self.use_fa3 = bool(use_fa3)
        self.verbose = bool(verbose)

        # Disk settings
        self.disk_offload_dir = disk_offload_dir
        self.disk_min_free_mb = float(disk_min_free_mb)
        self.disk_flush_mb = float(disk_flush_mb)
        self.disk_cleanup = bool(disk_cleanup)
        self.disk_file_prefix = str(disk_file_prefix)
        self.disk_dtype = disk_dtype

        # Safety factors
        self.cpu_safety_factor = float(cpu_safety_factor)
        self.disk_safety_factor = float(disk_safety_factor)
        self.max_pinned_memory_mb = float(max_pinned_memory_mb)

        # Async settings
        self.use_async = bool(use_async)
        self.async_depth = max(1, int(async_depth))

        # Normalize offload mode
        self.offload_mode = self._normalize_offload(offload)

        # Setup device
        if device is None:
            self.exe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.exe_device = torch.device(device)
        else:
            self.exe_device = device

        # Initialize buffer pool
        self._buffer_pool = PinnedBufferPool()

        self._is_configured = True

    def _normalize_offload(self, offload: Any) -> OffloadMode:
        """Normalize various offload specifications to OffloadMode."""
        if isinstance(offload, OffloadMode):
            return offload
        if offload is None or offload is False:
            return OffloadMode.GPU
        if offload is True:
            return OffloadMode.CPU
        if isinstance(offload, str):
            s = offload.lower().strip()
            if s == "gpu":
                return OffloadMode.GPU
            if s == "cpu":
                return OffloadMode.CPU
            if s == "disk":
                return OffloadMode.DISK
            if s == "auto":
                return OffloadMode.AUTO
        raise ValueError(f"Invalid offload={offload!r}. Expected bool or one of 'auto', 'gpu', 'cpu', 'disk'.")

    def get_available_cpu_memory(self) -> float:
        """Get available CPU memory in MB.

        Returns
        -------
        float
            Available CPU memory in megabytes. This is the amount of memory
            that can be allocated without causing swap.
        """
        return psutil.virtual_memory().available / (1024 * 1024)

    def get_available_gpu_memory(self) -> float:
        """Get available GPU memory in MB.

        Synchronizes CUDA operations and clears cache before checking to get
        an accurate reading of available memory.

        Returns
        -------
        float
            Available GPU memory in megabytes, or 0.0 if CUDA is not
            available or execution device is CPU.
        """
        if not torch.cuda.is_available() or self.exe_device.type != "cuda":
            return 0.0
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return torch.cuda.mem_get_info(self.exe_device)[0] / (1024 * 1024)

    def get_available_disk_space(self, path: Optional[str]) -> float:
        """Get available disk space at the specified path in MB.

        Parameters
        ----------
        path : Optional[str]
            Path to check for available disk space. Directory will be created
            if it doesn't exist. If None, returns 0.0.

        Returns
        -------
        float
            Available disk space in megabytes, or 0.0 if path is None or
            cannot be accessed.
        """
        if path is None:
            return 0.0
        try:
            os.makedirs(path, exist_ok=True)
            return shutil.disk_usage(path).free / (1024 * 1024)
        except Exception:
            return 0.0

    def _estimate_tensor_mb(self, shape: Tuple[int, ...], dtype: torch.dtype, repeat: int = 1) -> float:
        """Estimate tensor size in MB.

        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the tensor.

        dtype : torch.dtype
            Data type of the tensor.

        repeat : int, default=1
            Multiplier for the size (e.g. if tensor will be duplicated).

        Returns
        -------
        float
            Estimated size in megabytes.
        """
        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        return bytes_per_element * math.prod(shape) * int(repeat) / (1024 * 1024)

    def estimate_safe_batch_size(
        self, seq_len: int, include_inputs: bool = True, in_dim: Optional[int] = None, max_bs: int = 50000
    ) -> Tuple[float, int]:
        """Estimate safe batch size based on available GPU memory.

        Uses pre-computed memory coefficients from profiling to estimate how many
        elements can be processed in a single batch without causing OOM.

        Parameters
        ----------
        seq_len : int
            Sequence length of the input data.

        include_inputs : bool, default=True
            Whether to include input tensor memory in the estimation.

        in_dim : int, optional
            Input dimension. Required if ``include_inputs`` is True.

        max_bs : int, default=50000
            Maximum batch size to return, regardless of available memory.
            This caps batch size to avoid CUDA errors with flash attention.

        Returns
        -------
        available_mem : float
            Available GPU memory in MB.

        safe_bs : int
            Estimated safe batch size (capped by ``max_bs``).
        """
        available_mem = self.get_available_gpu_memory()
        target_mem = available_mem * self.safety_factor

        estimated_bs = MemoryEstimator.estimate_batch_size(seq_len, target_mem, self.enc_name, include_inputs, in_dim)

        if estimated_bs > max_bs and self.verbose:
            print(
                f"Warning: Estimated batch size {estimated_bs} exceeds maximum safe limit. "
                f"Capping batch size to {max_bs}."
            )

        safe_bs = min(max(self.min_batch_size, estimated_bs), max_bs)
        return available_mem, safe_bs

    def _resolve_offload_mode(
        self, output_mb: float, gpu_free_mb: float, cpu_free_mb: float, disk_free_mb: float
    ) -> Tuple[OffloadMode, OffloadReason]:
        """Resolve actual offload mode, returns (mode, reason).

        For user-requested modes, the requested mode is used if it fits.
        Otherwise, modes fall back: GPU -> CPU -> DISK -> CPU(swap as last resort).

        For AUTO mode, the priority is:
            GPU (if within threshold) -> CPU -> DISK -> CPU(swap as last resort).

        Note: CPU mode can use either pinned or non-pinned memory.
        - Pinned memory: faster for async GPU-CPU transfers, but locks physical memory
        - Non-pinned memory: slower transfers, but can use virtual memory (swap)
        """

        has_gpu = gpu_free_mb > 0
        has_disk = self.disk_offload_dir is not None
        effective_disk = max(0, disk_free_mb - self.disk_min_free_mb) if has_disk else 0

        safe_cpu_mb = cpu_free_mb * self.cpu_safety_factor
        safe_disk_mb = effective_disk * self.disk_safety_factor

        gpu_fits = has_gpu and output_mb <= gpu_free_mb
        cpu_fits = output_mb <= safe_cpu_mb
        disk_fits = has_disk and output_mb <= safe_disk_mb

        # User-requested mode with cascading fallback
        # If the requested mode fails, downgrade one tier at a time: GPU -> CPU -> DISK -> CPU (swap as last resort)
        if self.offload_mode != OffloadMode.AUTO:
            requested = self.offload_mode

            if requested == OffloadMode.GPU:
                if gpu_fits:
                    return OffloadMode.GPU, OffloadReason(
                        "user_gpu_fits", f"{output_mb:.0f}MB <= {gpu_free_mb:.0f}MB gpu free"
                    )
                elif cpu_fits:
                    return OffloadMode.CPU, OffloadReason("user_gpu_fails", "gpu (requested) tight -> cpu")
                elif disk_fits:
                    return OffloadMode.DISK, OffloadReason("user_gpu_fails", "gpu (requested) tight, cpu tight -> disk")
                else:
                    return OffloadMode.CPU, OffloadReason(
                        "user_gpu_fails", "gpu (requested) tight, cpu tight, disk tight -> cpu (swap)"
                    )

            if requested == OffloadMode.CPU:
                if cpu_fits:
                    return OffloadMode.CPU, OffloadReason(
                        "user_cpu_fits", f"{output_mb:.0f}MB <= {safe_cpu_mb:.0f}MB safe cpu free"
                    )
                elif disk_fits:
                    return OffloadMode.DISK, OffloadReason("user_cpu_fails", "cpu (requested) tight -> disk")
                else:
                    return OffloadMode.CPU, OffloadReason(
                        "user_cpu_fails", "cpu (requested) tight, disk tight -> cpu (swap)"
                    )

            if requested == OffloadMode.DISK:
                if not has_disk:
                    raise ValueError(
                        "Disk offload requested but disk_offload_dir is not configured. "
                        "Please specify disk_offload_dir in the configuration."
                    )

                if disk_fits:
                    return OffloadMode.DISK, OffloadReason(
                        "user_disk_fits", f"{output_mb:.0f}MB <= {safe_disk_mb:.0f}MB safe disk free"
                    )
                else:
                    return OffloadMode.CPU, OffloadReason("user_disk_fails", "disk (requested) tight -> cpu (swap)")

        # AUTO mode
        output_pct = output_mb / max(gpu_free_mb, 1e-6) if has_gpu else 1.0
        gpu_within_threshold = has_gpu and output_pct <= self.auto_offload_threshold

        if gpu_within_threshold:
            return OffloadMode.GPU, OffloadReason(
                "auto_gpu_fits",
                f"{output_mb:.0f}MB <= {self.auto_offload_threshold * gpu_free_mb:.0f}MB safe gpu free",
            )
        elif cpu_fits:
            return OffloadMode.CPU, OffloadReason(
                "auto_cpu_fits", f"gpu tight -> cpu ({output_mb:.0f}MB <= {safe_cpu_mb:.0f}MB safe cpu free)"
            )
        elif disk_fits:
            return OffloadMode.DISK, OffloadReason(
                "auto_disk_fits",
                f"gpu tight, cpu tight -> disk ({output_mb:.0f}MB <= {safe_disk_mb:.0f}MB safe disk free)",
            )
        else:
            return OffloadMode.CPU, OffloadReason("auto_cpu_swap", "gpu tight, cpu tight, disk tight -> cpu (swap)")

    def _allocate_output_buffer(
        self,
        mode: OffloadMode,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> Tuple[Union[Tensor, DiskTensor], Dict[str, Any]]:
        """Allocate output buffer according to mode."""
        info: Dict[str, Any] = {"mode": mode.name, "shape": shape, "dtype": str(dtype)}
        output_mb = self._estimate_tensor_mb(shape, dtype)

        if mode == OffloadMode.GPU:
            try:
                out = torch.empty(shape, dtype=dtype, device=self.exe_device)
                return out, info
            except RuntimeError as e:
                info["alloc_error"] = str(e)
                # Fallback to CPU
                mode = OffloadMode.CPU

        if mode == OffloadMode.CPU:
            try:
                # Only use pinned memory for smaller allocations
                use_pinned = output_mb <= self.max_pinned_memory_mb
                if self.verbose and not use_pinned:
                    print(
                        f"  Using regular (non-pinned) CPU memory for {output_mb:.0f}MB output (max_pinned={self.max_pinned_memory_mb:.0f}MB)"
                    )
                out = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=use_pinned)
                info["pinned"] = use_pinned
                return out, info
            except RuntimeError as e:
                info["alloc_error"] = str(e)
                if self.disk_offload_dir is None:
                    raise RuntimeError(
                        f"CPU memory allocation failed ({e}) and disk offload is not available. "
                        f"Please specify disk_offload_dir in the configuration to enable disk offloading, "
                        f"or reduce the output size (estimated: {output_mb:.0f}MB)."
                    ) from e
                if self.verbose:
                    print(f"  CPU allocation failed: {e}, falling back to disk")
                mode = OffloadMode.DISK

        # Disk mode
        if self.disk_offload_dir is None:
            raise RuntimeError(
                "Disk offload requested but disk_offload_dir is not configured. "
                "Please specify disk_offload_dir in the configuration."
            )
        fname = f"{self.disk_file_prefix}{self.enc_name}_{uuid.uuid4().hex}.mmap"
        path = os.path.join(self.disk_offload_dir, fname)

        storage_dtype = self.disk_dtype or dtype
        disk_tensor = DiskTensor(shape, storage_dtype, path, cleanup=self.disk_cleanup)

        if self.disk_cleanup:
            self._disk_finalizers.append(disk_tensor._finalizer)
            # Prune dead finalizers
            self._disk_finalizers = [f for f in self._disk_finalizers if f.alive]

        info.update({"path": path, "storage_dtype": str(storage_dtype)})
        return disk_tensor, info

    def _to_exe_device(self, tensor: Tensor) -> Tensor:
        """Move tensor to execution device if not already there.

        Parameters
        ----------
        tensor : Tensor
            Input tensor to move to execution device.

        Returns
        -------
        Tensor
            Tensor on the execution device. If already on the correct
            device, returns the same tensor without copying.
        """
        if isinstance(tensor, torch.Tensor) and self.exe_device.type == "cuda" and not tensor.is_cuda:
            return tensor.to(self.exe_device, non_blocking=True)
        return tensor

    def _prepare_inputs(self, inputs: OrderedDict[str, Any]) -> Dict[str, Any]:
        """Move all inputs to execution device.

        Tensors are moved via to_exe_device, populated KVCaches are moved to
        exe_device, empty KVCaches and non-tensor values are passed through.
        """
        prepared: Dict[str, Any] = {}
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor):
                prepared[name] = self._to_exe_device(value)
            elif isinstance(value, KVCache):
                prepared[name] = value.to(self.exe_device) if value.is_populated() else value
            else:
                prepared[name] = value
        return prepared

    def _run_forward(self, forward_fn: Callable[..., Tensor], inputs: Dict[str, Any]) -> Tensor:
        """Execute forward function with no_grad and optional AMP."""
        with torch.no_grad(), flash_attn3_toggle(self.use_fa3):
            if self.use_amp and self.exe_device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    return forward_fn(**inputs)
            return forward_fn(**inputs)

    def __call__(
        self,
        forward_fn: Callable[..., Tensor],
        inputs: OrderedDict[str, Any],
        auto_batch: bool = True,
        output_repeat: int = 1,
    ) -> Tensor:
        """Execute forward pass with automatic batching and memory management.

        This method runs the provided forward function on the inputs, automatically
        handling batch splitting, memory offloading, and OOM recovery. It is the
        main entry point for running inference through the manager.

        Parameters
        ----------
        forward_fn : Callable[..., Tensor]
            Model forward function that takes keyword arguments matching the keys
            in ``inputs`` and returns a tensor output.

        inputs : OrderedDict[str, Any]
            OrderedDict of inputs where the first value must be a tensor of shape
            ``(..., seq_len, in_dim)``. The leading dimensions are treated as batch
            dimensions that may be split for memory efficiency. Non-tensor values
            are passed through unchanged.

        auto_batch : bool, default=True
            Whether to automatically split inputs into smaller batches based on
            estimated safe batch size. If False, processes all inputs at once.

        output_repeat : int, default=1
            Memory estimation multiplier for the output tensor. Use values > 1 when
            the output will be manipulated to create multiple derived outputs in
            subsequent operations (e.g. shuffled multiple times for feature ordering).

        Returns
        -------
        Tensor
            Combined output tensor from all batches. The tensor will be on the
            device determined by the offload mode:

            - GPU mode: Returns GPU tensor
            - CPU mode: Returns CPU tensor (pinned or non-pinned)
            - Disk mode: Returns CPU tensor backed by memory-mapped file

        Notes
        -----
        - For CPU execution (no CUDA), batching is not supported and the forward
          function is called once with all inputs.
        - When OOM occurs, batch size is halved and inference is retried.
        - Async copy is used when ``use_async=True`` and offloading to CPU/disk.
        """

        if not self._is_configured:
            raise RuntimeError("InferenceManager must be configured before running inference. Call configure() first.")

        # Non-batched execution
        if not auto_batch:
            return self._run_forward(forward_fn, self._prepare_inputs(inputs))

        # CPU execution: batching not supported currently
        if self.exe_device.type == "cpu":
            return forward_fn(**inputs)

        # Extract shape/dtype info
        first_value = next(iter(inputs.values()))
        if not isinstance(first_value, torch.Tensor):
            raise ValueError("First input must be a tensor.")
        if first_value.dim() < 3:
            raise ValueError(f"First tensor input must have at least 3 dimensions, got {first_value.dim()}")

        *batch_dims, seq_len, in_dim = first_value.shape
        input_dtype = first_value.dtype
        inputs_on_cuda = first_value.is_cuda
        total_bs = math.prod(batch_dims)

        # Estimate batch size
        gpu_mem, batch_size = self.estimate_safe_batch_size(seq_len, include_inputs=not inputs_on_cuda, in_dim=in_dim)

        if self.verbose:
            print(
                f"\nAvailable GPU memory: {gpu_mem / 1024:.2f}GB, seq_len: {seq_len}, "
                f"estimated batch size for {self.enc_name}: {batch_size}"
            )

        # Calculate output shape
        if self.out_no_seq:
            output_shape = (*batch_dims, self.out_dim)
        else:
            output_shape = (*batch_dims, seq_len, self.out_dim)

        # Estimate output size
        output_mb = self._estimate_tensor_mb(tuple(output_shape), input_dtype, repeat=output_repeat)
        cpu_free = self.get_available_cpu_memory()
        disk_free = self.get_available_disk_space(self.disk_offload_dir)

        # Resolve offload mode
        mode, reason = self._resolve_offload_mode(output_mb, gpu_mem, cpu_free, disk_free)

        if self.verbose:
            eff_disk = max(0, disk_free - self.disk_min_free_mb)
            print(
                f"Offload decision: mode={mode.name} (reason={reason})\n"
                f"Output size: {output_mb / 1024:.2f}GB (repeat={output_repeat}), "
                f"CPU free: {cpu_free / 1024:.2f}GB, "
                f"GPU free: {gpu_mem / 1024:.2f}GB, "
                f"Disk free (effective): {eff_disk / 1024:.2f}GB @ {self.disk_offload_dir}"
            )

        # Single batch case
        if batch_size >= total_bs:
            out = self._run_forward(forward_fn, self._prepare_inputs(inputs))

            if mode == OffloadMode.GPU:
                return out
            if mode == OffloadMode.CPU:
                return out.cpu()
            # DISK
            outputs, _ = self._allocate_output_buffer(mode, tuple(out.shape), input_dtype)
            if isinstance(outputs, DiskTensor):
                outputs[...] = out.cpu()
                outputs.flush()
                return outputs.tensor
            outputs.copy_(out.cpu())
            return outputs

        # Multi-batch execution
        outputs, _ = self._allocate_output_buffer(mode, tuple(output_shape), input_dtype)

        # Setup async copy manager if using async and offloading
        async_copy: Optional[AsyncCopyManager] = None
        if self.use_async and self.exe_device.type == "cuda" and mode != OffloadMode.GPU:
            async_copy = AsyncCopyManager(
                self.exe_device,
                max_pending=self.async_depth,
                buffer_pool=self._buffer_pool,
            )

        # Track in MB for consistency with disk_flush_mb
        bytes_since_flush_mb = 0.0

        # Identify store-mode caches that need to be accumulated back into the original inputs
        store_cache_keys = {name for name, v in inputs.items() if isinstance(v, KVCache) and not v.is_populated()}

        # Main inference loop with OOM recovery
        while True:
            try:
                split_sizes = self.compute_split_sizes(batch_dims, batch_size)
                n_batches = self.compute_n_batches(batch_dims, split_sizes)
                batch_iterator = self.create_multidim_batches(inputs, batch_dims, split_sizes)

                if self.verbose:
                    batch_iterator = tqdm(
                        batch_iterator, total=n_batches, desc=f"Processing {self.enc_name}", unit="batch"
                    )

                for batch_dict, indices in batch_iterator:
                    out = self._run_forward(forward_fn, batch_dict)

                    # Accumulate store-mode caches back into the original inputs.
                    # Each batch received a fresh empty KVCache and forward_fn fills
                    # it in-place (e.g., forward_with_cache stores K/V projections
                    # into the cache object it receives).  After the batch completes,
                    # we copy those per-batch entries into the pre-allocated cache.
                    for cache_key in store_cache_keys:
                        batch_cache = batch_dict[cache_key]
                        if batch_cache.is_populated():
                            original_cache = inputs[cache_key]
                            if not original_cache.is_populated():
                                # First populated batch: use its shapes as a template to
                                # pre-allocate zero tensors with the full batch dimensions.
                                original_cache.preallocate(batch_cache, tuple(batch_dims), device=self.exe_device)
                            original_cache[indices] = batch_cache

                    if mode == OffloadMode.GPU:
                        outputs[indices] = out
                    else:
                        if async_copy is not None:
                            async_copy.submit_copy(out, outputs, indices)
                            # Check if we need to flush (async manager tracks bytes internally)
                            if isinstance(outputs, DiskTensor) and self.disk_flush_mb > 0:
                                if async_copy.get_bytes_written() >= self.disk_flush_mb:
                                    outputs.flush()
                                    async_copy.reset_bytes_counter()
                        else:
                            out_cpu = out.cpu()
                            outputs[indices] = out_cpu
                            bytes_since_flush_mb += out_cpu.numel() * out_cpu.element_size() / (1024 * 1024)

                            # Periodic flush for disk
                            if isinstance(outputs, DiskTensor) and self.disk_flush_mb > 0:
                                if bytes_since_flush_mb >= self.disk_flush_mb:
                                    outputs.flush()
                                    bytes_since_flush_mb = 0.0

                    del out
                    del batch_dict

                # Drain async copies
                if async_copy is not None:
                    async_copy.drain_all()

                # Final flush
                if isinstance(outputs, DiskTensor):
                    outputs.flush()
                    return outputs.tensor

                return outputs

            except torch.cuda.OutOfMemoryError as e:
                if async_copy is not None:
                    async_copy.clear()

                # Clear pre-allocated cache entries before retry
                for cache_key in store_cache_keys:
                    inputs[cache_key].kv.clear()

                if batch_size <= self.min_batch_size:
                    raise RuntimeError(
                        f"Failed to execute even with minimum batch size {self.min_batch_size}. Error: {e}"
                    )

                if self.verbose:
                    print(f"OOM with batch_size={batch_size}, reducing to {max(self.min_batch_size, batch_size // 2)}")

                if self.exe_device.type == "cuda":
                    torch.cuda.empty_cache()

                batch_size = max(self.min_batch_size, batch_size // 2)

    @staticmethod
    def compute_split_sizes(batch_dims: Tuple[int], batch_size: int) -> List[int]:
        """Plan how to split batch dimensions based on memory constraints.

        Given batch dimensions and a target batch size, determines how many elements
        to process from each dimension in a single batch.

        Parameters
        ----------
        batch_dims : Tuple[int]
            Shape of batch dimensions (e.g. ``(10, 20)`` for 10x20 batches).

        batch_size : int
            Maximum number of elements to process at once.

        Returns
        -------
        List[int]
            Split sizes for each dimension. The product of split sizes will be
            at most ``batch_size``.
        """
        elements_left = batch_size
        split_sizes: List[int] = []

        for dim_size in batch_dims:
            if elements_left >= dim_size:
                split_sizes.append(dim_size)
                elements_left //= dim_size
            else:
                split_sizes.append(max(1, elements_left))
                elements_left = 1

        return split_sizes

    @staticmethod
    def compute_n_batches(batch_dims: Tuple[int], split_sizes: List[int]) -> int:
        """Compute total number of batches given batch dimensions and split sizes.

        Parameters
        ----------
        batch_dims : Tuple[int]
            Shape of batch dimensions.

        split_sizes : List[int]
            Split sizes for each dimension from ``compute_split_sizes``.

        Returns
        -------
        int
            Total number of batches needed to process all data.
        """
        n = 1
        for dim_size, bs in zip(batch_dims, split_sizes):
            n *= math.ceil(dim_size / bs)
        return n

    def create_multidim_batches(
        self,
        inputs: OrderedDict[str, Any],
        batch_dims: Tuple[int],
        split_sizes: List[int],
    ) -> Iterator[Tuple[Dict[str, Any], Tuple[slice, ...]]]:
        """Create a multi-dimensional batch iterator for splitting large inputs.

        Yields batches of inputs along with indices for placing results in the
        output tensor. Handles multi-dimensional batch dimensions by generating
        all combinations of slice indices.

        Parameters
        ----------
        inputs : OrderedDict[str, Any]
            Dictionary of inputs. Tensor values are sliced; others pass through.

        batch_dims : Tuple[int]
            Shape of batch dimensions to iterate over.

        split_sizes : List[int]
            Split sizes for each dimension.

        Yields
        ------
        batch_dict : Dict[str, Any]
            Dictionary of sliced inputs moved to execution device.

        indices : Tuple[slice, ...]
            Tuple of slice objects for placing results in output tensor.
        """
        slices: List[List[slice]] = []
        for dim_size, bs in zip(batch_dims, split_sizes):
            dim_slices: List[slice] = []
            for start in range(0, dim_size, bs):
                end = min(start + bs, dim_size)
                dim_slices.append(slice(start, end))
            slices.append(dim_slices)

        for slice_tuple in itertools.product(*slices):
            batch_dict: Dict[str, Any] = {}
            for name, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    batch_dict[name] = self._to_exe_device(value[slice_tuple])
                elif isinstance(value, KVCache):
                    if value.is_populated():
                        # use_cache mode: slice the populated cache along batch dims
                        # and move to execution device
                        batch_dict[name] = value[slice_tuple].to(self.exe_device)
                    else:
                        # store_cache mode: give each batch a fresh empty cache and
                        # forward_fn will populate it in-place
                        batch_dict[name] = KVCache()
                else:
                    batch_dict[name] = value
            yield batch_dict, slice_tuple
