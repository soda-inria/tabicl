from __future__ import annotations

import warnings
import itertools
from collections import OrderedDict
from typing import List, Tuple, Dict, Callable, Iterator, Literal, Optional, Any

import psutil
from tqdm.auto import tqdm

import math
from scipy.optimize import fsolve

import torch
from torch import Tensor


class MemoryEstimator:
    """Estimates peak activation memory requirements for different attention-based components.

    Peak inference memory refers to the maximum amount of memory (typically GPU memory) used during
    the inference phase of a model. This is the highest memory consumption observed at any point.

    The coefficients and intercepts for each component are derived through memory profiling and regression:
    1. Collect memory usage data by running models with different batch sizes and sequence lengths
    2. Fit a polynomial regression : c1 * batch_size + c2 * seq_len + c3 * (batch_size * seq_len) + intercept
    3. Use the fitted coefficients to estimate memory for new batch sizes and sequence lengths

    Memory profiling was conducted using float32 without automatic mixed precision (AMP).
    When using AMP, actual memory usage will be lower than the estimates provided by this class.
    """

    # Coefficients and intercepts for memory estimation
    coefficients: Dict[str, list] = {
        "tf_col": [7.079980260e-02, 7.29386080e-06, 3.90989142e-03],
        "tf_row": [-2.06831848e-05, 2.27205969e-04, 5.37117114e-03],
        "tf_icl": [-2.60068961e-01, 4.77470594e-07, 1.95310976e-02],
    }
    intercepts: Dict[str, float] = {
        "tf_col": 137.62474190864668,
        "tf_row": 138.53653545318957,
        "tf_icl": 140.58027172987750,
    }

    @staticmethod
    def estimate_peak_mem(
        batch_size: int, seq_len: int, enc_name: str, include_inputs: bool = True, in_dim: Optional[int] = None
    ) -> float:
        """Estimate peak memory usage for a given component with specified batch size and sequence length.

        Parameters
        ----------
        batch_size : int
            Batch size for inference

        seq_len : int
            Sequence length for inference

        enc_name : str
            Model encoder name to estimate memory for. One of:
            - "tf_col": Column embedding encoder
            - "tf_row": Row interaction encoder
            - "tf_icl": In-context learning encoder

        include_inputs : bool, default=True
            Whether to include memory usage for input tensors

        in_dim : Optional[int], default=None
            Model dimension for the encoder

        Returns
        -------
        float
            Estimated peak memory usage in MB for the specified encoder
        """
        coefs = MemoryEstimator.coefficients[enc_name]
        inter = MemoryEstimator.intercepts[enc_name]
        peak_activation_mem = coefs[0] * batch_size + coefs[1] * seq_len + coefs[2] * batch_size * seq_len + inter

        if include_inputs:
            assert in_dim is not None, "Input dimension must be provided for input memory estimation"
            bytes_per_element = 4  # float32
            n_elements = batch_size * seq_len * in_dim
            mem_inputs = n_elements * bytes_per_element / (1024**2)  # Convert to MB
            peak_activation_mem += mem_inputs

        return peak_activation_mem

    @staticmethod
    def estimate_batch_size(
        seq_len: int, target_memory: float, enc_name: str, include_inputs: bool = True, in_dim: Optional[int] = None
    ) -> int:
        """Estimate the batch size that would result in the target memory usage for a given sequence length.

        Parameters
        ----------
        seq_len : int
            Sequence length for inference

        target_memory : float
            Target memory usage in MB

        enc_name : str
            Model encoder name to estimate memory for. One of:
            - "tf_col": Column embedding encoder
            - "tf_row": Row interaction encoder
            - "tf_icl": In-context learning encoder

        include_inputs : bool, default=True
            Whether to include memory usage for input tensors

        in_dim : Optional[int], default=None
            Model dimension for the encoder

        Returns
        -------
        int
            Estimated batch size that fits within target memory constraints
        """

        def objective_function(bs: float) -> float:
            return MemoryEstimator.estimate_peak_mem(bs, seq_len, enc_name, include_inputs, in_dim) - target_memory

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            solution = fsolve(objective_function, x0=1)[0]

        # Ensure we return a valid batch size (at least 1)
        return max(1, int(solution))


class InferenceManager:
    """Manages memory-efficient model inference by automatically batching inputs.

    This class handles the complexities of running inference on large models with memory constraints.
    It automatically:
    1. Estimates safe batch sizes based on available GPU memory
    2. Splits large inputs into smaller batches
    3. Handles device placement (CPU/GPU offloading)
    4. Recovers from out-of-memory (OOM) errors by reducing batch size
    5. Merges results from multiple batches

    Parameters
    ----------
    enc_name : str
        Name of encoder for memory estimation. One of:
        - "tf_col": Column embedding encoder
        - "tf_row": Row interaction encoder
        - "tf_icl": In-context learning encoder

    out_dim : int
        Output dimension for pre-allocating output tensor

    out_no_seq : bool, default=False
        Whether to remove sequence dimension from output tensor. If True,
        output shape will be (..., out_dim) instead of (..., seq_len, out_dim)
    """

    def __init__(self, enc_name: str, out_dim: int, out_no_seq: bool = False):
        self.enc_name = enc_name
        self.out_dim = out_dim
        self.out_no_seq = out_no_seq
        self._is_configured = False  # Track if configure_inference has been called

    def configure(
        self,
        min_batch_size: int = 1,
        safety_factor: float = 0.8,
        offload: bool | Literal["auto"] = "auto",
        auto_offload_pct: float = 0.5,
        device: Optional[str | torch.device] = None,
        use_amp: bool = True,
        verbose: bool = False,
    ):
        """Configure inference parameters.

        Parameters
        ----------
        min_batch_size : int, default=1
            Minimum batch size to try before raising an error. If OOM occurs even with
            this batch size, inference cannot proceed.

        safety_factor : float, default=0.8
            Factor (0-1) to multiply estimated batch size by for conservative memory usage.
            Lower values are safer but may result in more batches.

        offload : bool or Literal["auto"], default="auto"
            Whether to offload intermediate results to CPU to save GPU memory.
            Options:
            - True: Always offload to CPU
            - False: Keep all tensors on GPU
            - "auto": Offload if output size exceeds `auto_offload_pct` of available GPU memory
            and enough CPU memory is available

        auto_offload_pct : float, default=0.5
            Threshold for automatic offloading when offload="auto".
            If output size exceeds this percentage of available GPU memory, intermediate
            results are offloaded to CPU in order to save GPU memory.

        device : Optional[str or torch.device], default=None
            Device to use for inference. If None, defaults to torch.device("cuda") if available,
            else torch.device("cpu")

        use_amp : bool, default=True
            Whether to use automatic mixed precision during inference

        verbose : bool, default=False
            Whether to show progress bars and logging information during inference
        """

        self.min_batch_size = min_batch_size
        self.safety_factor = safety_factor
        self.offload = offload
        self.auto_offload_pct = auto_offload_pct
        self.use_amp = use_amp
        self.verbose = verbose

        if device is None:
            self.exe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.exe_device = torch.device(device)
        else:
            self.exe_device = device

        self._is_configured = True  # Mark as configured

    def to_exe_device(self, tensor: Tensor) -> Tensor:
        """Move tensor to execution device if not already there.

        Parameters
        ----------
        tensor : Tensor
            Input tensor to move to execution device

        Returns
        -------
        Tensor
            Tensor on the execution device
        """
        if isinstance(tensor, torch.Tensor) and self.exe_device.type == "cuda" and not tensor.is_cuda:
            return tensor.to(self.exe_device)
        return tensor

    def get_available_cpu_memory(self) -> float:
        """Get available CPU memory in MB.

        Returns
        -------
        float
            Available CPU memory in MB
        """
        return psutil.virtual_memory().available / (1024 * 1024)  # Convert to MB

    def get_available_gpu_memory(self) -> float:
        """Get available GPU memory in MB.

        Returns
        -------
        float
            Available GPU memory in MB or infinity if CUDA is not available
        """
        if not torch.cuda.is_available() or self.exe_device.type != "cuda":
            return float("inf")

        # Synchronize and clear cache to get accurate memory reading
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return torch.cuda.mem_get_info(self.exe_device)[0] / (1024 * 1024)  # Convert to MB

    def estimate_safe_batch_size(
        self, seq_len: int, include_inputs: bool = True, in_dim: Optional[int] = None, max_bs: int = 50000
    ) -> Tuple[float, int]:
        """Estimate safe batch size based on available memory.

        Parameters
        ----------
        seq_len : int
            Sequence length for inference

        include_inputs : bool, default=True
            Whether to include memory usage for input tensors

        in_dim : Optional[int], default=None
            Model dimension for the encoder

        max_bs : int, default=50000
            Maximum allowable batch size to avoid CUDA errors like "invalid configuration argument"
            that occur with flash attention when batch size is too large. This is a hard cap
            regardless of available memory. See: https://github.com/pytorch/pytorch/issues/142228

        Returns
        -------
        Tuple[float, int]
            - Available memory in MB
            - Estimated safe batch size
        """
        available_mem = self.get_available_gpu_memory()
        target_mem = available_mem * self.safety_factor

        # Calculate batch size and ensure it's between min_batch_size and max_bs
        estimated_bs = MemoryEstimator.estimate_batch_size(seq_len, target_mem, self.enc_name, include_inputs, in_dim)

        # Apply maximum batch size cap to avoid CUDA errors with flash attention
        if estimated_bs > max_bs and self.verbose:
            print(
                f"Warning: Estimated batch size {estimated_bs} exceeds maximum safe limit. "
                f"Capping batch size to {max_bs} to avoid CUDA configuration errors."
            )

        safe_bs = min(max(self.min_batch_size, estimated_bs), max_bs)

        return available_mem, safe_bs

    def __call__(
        self,
        forward_fn: Callable[..., Tensor],
        inputs: OrderedDict[str, Any],
        auto_batch: bool = True,
        output_repeat: int = 1,
    ) -> Tensor:
        """Forward pass with automatic batch size adjustment to avoid OOM errors.

        Parameters
        ----------
        forward_fn : Callable[..., Tensor]
            Model forward function that takes dictionary inputs and returns tensor output

        inputs : OrderedDict[str, Any]
            OrderedDict of inputs where the first one must be a tensor of shape (..., seq_len, in_dim).
            For all tensor inputs, the first dimensions should be the batch dimensions.

        auto_batch : bool, default=True
            Whether to automatically split inputs into smaller batches

        output_repeat : int, default=1
            Memory estimation multiplier for the output tensor. This parameter is crucial when
            the output will be manipulated to create multiple derived outputs in subsequent
            operations. For example, in ColEmbedding, the same embedding output might be shuffled
            multiple times to handle different feature orderings, effectively requiring multiple
            copies of the output tensor in memory. Setting this parameter helps accurately estimate
            memory requirements and determine whether to offload to CPU to avoid OOM errors.

        Returns
        -------
        Tensor
            Combined output from all batches

        Raises
        ------
        ValueError
            If first input is not a tensor or doesn't have expected shape
        RuntimeError
            If inference fails even with minimum batch size
        """
        # Check if configure_inference has been called
        if not hasattr(self, "_is_configured") or not self._is_configured:
            raise RuntimeError(
                "InferenceManager must be configured before running inference. Call configure_inference() first."
            )

        if not auto_batch:
            # Move inputs to execution device
            inputs_on_exe = {}
            for name, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs_on_exe[name] = self.to_exe_device(value)
                else:
                    inputs_on_exe[name] = value

            with torch.no_grad():
                if self.use_amp and self.exe_device.type == "cuda":
                    with torch.autocast(device_type="cuda"):
                        outputs = forward_fn(**inputs_on_exe)
                else:
                    outputs = forward_fn(**inputs_on_exe)

            # Move to CPU if needed
            if self.offload:
                return outputs.to(device="cpu")
            else:
                return outputs

        # CPU does not support batching temporarily
        if self.exe_device.type == "cpu":
            with torch.no_grad():
                return forward_fn(**inputs)

        # Extract dimension and dtype information from first tensor
        first_value = next(iter(inputs.values()))

        if not isinstance(first_value, torch.Tensor):
            raise ValueError("First input must be a tensor.")

        if first_value.dim() < 3:
            raise ValueError(
                f"First tensor input must have at least 3 dimensions (batch_dim, seq_len, in_dim), "
                f"got {first_value.dim()}"
            )

        # Extract batch dimensions, sequence length, and input dimension
        *batch_dims, seq_len, in_dim = first_value.shape
        input_dtype = first_value.dtype
        inputs_on_cuda = first_value.is_cuda

        # Calculate total batch size across all batch dimensions
        total_bs = math.prod(batch_dims)

        # Estimate batch size based on available memory
        gpu_mem, batch_size = self.estimate_safe_batch_size(seq_len, include_inputs=not inputs_on_cuda, in_dim=in_dim)

        if self.verbose:
            print(
                f"\nAvailable memory: {gpu_mem / 1024:.2f}GB, sequence length: {seq_len}, "
                f"estimated batch elements per batch for {self.enc_name}: {batch_size}\n"
            )

        # Define output shape based on configuration
        if self.out_no_seq:
            output_shape = (*batch_dims, self.out_dim)
        else:
            output_shape = (*batch_dims, seq_len, self.out_dim)

        # Determine if offloading to CPU is needed
        if self.offload == "auto":
            # Calculate output tensor size in bytes
            bytes_per_element = torch.tensor([], dtype=input_dtype).element_size()
            output_mb = bytes_per_element * math.prod(output_shape) / (1024 * 1024)
            output_mb *= output_repeat  # Multiply by repeat factor

            # Check if output size exceeds threshold of available GPU memory
            output_pct = output_mb / gpu_mem
            excess_gpu = output_pct > self.auto_offload_pct

            # Check if there's enough CPU memory available
            cpu_mem = self.get_available_cpu_memory()
            enough_cpu = cpu_mem > output_mb

            self.offload = excess_gpu and enough_cpu
            if self.verbose:
                print(
                    f"Output size: {output_mb / 1024:.2f}GB, "
                    f"CPU memory: {cpu_mem / 1024:.2f}GB, "
                    f"GPU memory: {gpu_mem / 1024:.2f}GB\n"
                    f"Output size exceeds {self.auto_offload_pct * 100:.2f}% of GPU memory: {excess_gpu} "
                    f"and CPU memory is sufficient: {enough_cpu}\n"
                    f"Offloading to CPU: {self.offload}"
                )

        # If we can process all data in one batch, do it
        if batch_size >= total_bs:
            # Move inputs to execution device
            inputs_on_exe = {}
            for name, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs_on_exe[name] = self.to_exe_device(value)
                else:
                    inputs_on_exe[name] = value

            with torch.no_grad():
                if self.use_amp and self.exe_device.type == "cuda":
                    with torch.autocast(device_type="cuda"):
                        outputs = forward_fn(**inputs_on_exe)
                else:
                    outputs = forward_fn(**inputs_on_exe)

            # Move to CPU if needed
            if self.offload:
                return outputs.to(device="cpu")
            else:
                return outputs

        # Pre-allocate output tensor with same dtype as input
        output_device = torch.device("cpu") if self.offload else self.exe_device
        outputs = torch.empty(output_shape, dtype=input_dtype, device=output_device)

        # Main inference loop with OOM recovery
        while True:
            try:
                # Calculate how to split batch dimensions based on estimated batch size
                split_sizes = self.compute_split_sizes(batch_dims, batch_size)
                # Create batches based on the calculated structure
                n_batches = self.compute_n_batches(batch_dims, split_sizes)
                batch_iterator = self.create_multidim_batches(inputs, batch_dims, split_sizes)

                if self.verbose:
                    batch_iterator = tqdm(
                        batch_iterator, total=n_batches, desc=f"Processing {self.enc_name}", unit="batch"
                    )

                for batch_dict, indices in batch_iterator:
                    with torch.no_grad():
                        if self.use_amp and self.exe_device.type == "cuda":
                            with torch.autocast(device_type="cuda"):
                                output = forward_fn(**batch_dict)
                        else:
                            output = forward_fn(**batch_dict)

                        if self.offload:
                            # Move output to CPU before assigning to save GPU memory
                            outputs[indices] = output.to(device="cpu")
                            del output
                        else:
                            outputs[indices] = output

                    # Delete batch to free memory
                    del batch_dict

                return outputs

            except torch.cuda.OutOfMemoryError as e:
                if batch_size <= self.min_batch_size:
                    raise RuntimeError(
                        f"Failed to execute even with minimum batch size {self.min_batch_size}. Error: {e}"
                    )

                if self.verbose:
                    print(
                        f"OOM with batch_size={batch_size} for {self.enc_name}, "
                        f"reducing to {max(self.min_batch_size, batch_size // 2)}"
                    )

                # Clear CUDA memory and reduce batch size
                if self.exe_device.type == "cuda":
                    torch.cuda.empty_cache()

                batch_size = max(self.min_batch_size, batch_size // 2)

    @staticmethod
    def compute_split_sizes(batch_dims: Tuple[int], batch_size: int) -> List[int]:
        """Plan how to split batch dimensions based on memory constraints.

        Parameters
        ----------
        batch_dims : Tuple[int]
            Shape of batch dimensions

        batch_size : int
            Maximum number of elements to process in each batch

        Returns
        -------
        List[int]
        Dimension chunk sizes - a list where each element represents how many items
        from the corresponding batch dimension should be processed at once.
        """
        if not batch_dims:  # No batch dimensions
            return []

        # Calculate how many elements we can process at once
        elements_left = batch_size
        split_sizes = []

        # Try to fit as many complete dimensions as possible
        for dim_size in batch_dims:
            if elements_left >= dim_size:
                # We can process this entire dimension
                split_sizes.append(dim_size)
                elements_left //= dim_size
            else:
                # We need to split this dimension
                split_sizes.append(min(dim_size, max(1, elements_left)))
                elements_left = 0
                break

        # Fill remaining dimensions with 1 (process one slice at a time)
        split_sizes.extend([1] * (len(batch_dims) - len(split_sizes)))

        return split_sizes

    @staticmethod
    def compute_n_batches(batch_dims: Tuple[int], split_sizes: List[int]) -> int:
        """Compute the total number of batches needed.

        Parameters
        ----------
        batch_dims : Tuple[int]
            Shape of batch dimensions

        batch_sizes : List[int]
            List containing the batch size for each dimension

        Returns
        -------
        int
            Total number of batches
        """

        n_batches = 1
        for batch_dim, split_size in zip(batch_dims, split_sizes):
            n_batches *= math.ceil(batch_dim / split_size)

        return n_batches

    def create_multidim_batches(
        self, inputs: OrderedDict[str, Any], batch_dims: Tuple[int], split_sizes: List[int]
    ) -> Iterator:
        """Create batches from input dictionary with multidimensional batching.

        Parameters
        ----------
        inputs : OrderedDict[str, Any]
            OrderedDict of inputs

        batch_dims : Tuple[int]
            Shape of batch dimensions

        split_sizes : List[int]
            Dimension chunk sizes - a list where each element represents how many items
            from the corresponding batch dimension should be processed at once.

        Yields
        ------
        Tuple[Dict[str, Any], tuple]
            - Dictionary containing inputs for current batch
            - Tuple of indices for assigning results back to outputs
        """

        # Generate all possible slices based on batch_sizes
        slices = []
        for dim_size, batch_size in zip(batch_dims, split_sizes):
            dim_slices = []
            for start in range(0, dim_size, batch_size):
                end = min(start + batch_size, dim_size)
                dim_slices.append(slice(start, end))
            slices.append(dim_slices)

        # Generate all combinations of slices
        slice_tuples = itertools.product(*slices)

        for slice_tuple in slice_tuples:
            batch_dict = {}
            for name, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    # Move slice to execution device
                    batch_dict[name] = self.to_exe_device(value[slice_tuple])
                else:
                    # Non-tensor values are passed as is
                    batch_dict[name] = value

            yield batch_dict, slice_tuple
