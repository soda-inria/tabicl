from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


class MgrConfig:
    """Config class for ``InferenceManager``.

    Allowed Keys
    ------------
    **General:**

    - ``device``: Device to use for inference.
    - ``use_amp``: Whether to use automatic mixed precision during inference.
    - ``use_fa3``: Whether to use Flash Attention 3 during inference.
    - ``verbose``: Whether to print detailed information during inference.

    **Batching:**

    - ``min_batch_size``: Minimum batch size to try before raising an error.
    - ``safety_factor``: Factor to multiply estimated batch size by for conservative
      memory usage.

    **Offloading:**

    - ``offload``: Where to store output tensors during inference.

      - ``"gpu"`` or ``False``: Keep outputs on GPU. Fastest but limited by VRAM.
      - ``"cpu"`` or ``True``: Offload to CPU memory. Uses pinned memory for faster
        transfers if output size <= ``max_pinned_memory_mb``.
      - ``"disk"``: Offload to memory-mapped files. Supports outputs larger than RAM.
      - ``"auto"``: Automatically choose based on available memory (default).
        Uses ``auto_offload_threshold`` to decide when to offload from GPU.

    - ``auto_offload_threshold``: GPU memory threshold (0-1) for automatic offloading.
      When ``offload="auto"``, outputs exceeding this fraction of available GPU memory
      are offloaded to CPU or disk. Default is 0.5 (50%).

    **CPU offloading:**

    - ``cpu_safety_factor``: Safety margin (0-1) for CPU memory estimation.
    - ``max_pinned_memory_mb``: Maximum output size (MB) to use pinned memory.
      Larger outputs use regular memory which can be swapped.

    **Disk offloading:**

    - ``disk_offload_dir``: Directory for memory-mapped files. If None, disk
      offloading is disabled and an error is raised when CPU memory is insufficient.
    - ``disk_min_free_mb``: Minimum free disk space to maintain (MB).
    - ``disk_flush_mb``: Flush to disk after writing this many MB.
    - ``disk_cleanup``: Auto-cleanup files when tensors are garbage collected.
    - ``disk_file_prefix``: Prefix for memory-mapped file names.
    - ``disk_dtype``: Override dtype for disk storage (e.g. float16 to save space).
    - ``disk_safety_factor``: Safety margin (0-1) for disk space estimation.

    **Async transfer:**

    - ``use_async``: Use async CUDA streams for GPU-to-CPU transfers.
    - ``async_depth``: Max pending async copies before blocking.
    """

    _ALLOWED_KEYS = {
        # General
        "device",
        "use_amp",
        "use_fa3",
        "verbose",
        # Batching
        "min_batch_size",
        "safety_factor",
        # Offloading
        "offload",
        "auto_offload_threshold",
        # CPU offloading
        "cpu_safety_factor",
        "max_pinned_memory_mb",
        # Disk offloading
        "disk_offload_dir",
        "disk_min_free_mb",
        "disk_flush_mb",
        "disk_cleanup",
        "disk_file_prefix",
        "disk_dtype",
        "disk_safety_factor",
        # Async transfer
        "use_async",
        "async_depth",
    }
    _TYPE_SPECS = {
        # General
        "device": {
            "expected_type": (type(None), str, torch.device),
            "validator": None,
            "error_msg": "device must be a string or torch.device",
        },
        "use_amp": {"expected_type": bool, "validator": None, "error_msg": "use_amp must be a boolean"},
        "use_fa3": {"expected_type": bool, "validator": None, "error_msg": "use_fa3 must be a boolean"},
        "verbose": {"expected_type": bool, "validator": None, "error_msg": "verbose must be a boolean"},
        # Batching
        "min_batch_size": {
            "expected_type": int,
            "validator": lambda x: x >= 1,
            "error_msg": "min_batch_size must be an integer >= 1",
        },
        "safety_factor": {
            "expected_type": float,
            "validator": lambda x: 0.0 <= x <= 1.0,
            "error_msg": "safety_factor must be a float between 0 and 1",
        },
        # Offloading
        "offload": {
            "expected_type": (bool, str),
            "validator": lambda x: isinstance(x, bool) or x in ("auto", "gpu", "cpu", "disk"),
            "error_msg": "offload must be a boolean or one of 'auto', 'gpu', 'cpu', 'disk'",
        },
        "auto_offload_threshold": {
            "expected_type": float,
            "validator": lambda x: 0.0 <= x <= 1.0,
            "error_msg": "auto_offload_threshold must be a float between 0 and 1",
        },
        # CPU offloading
        "cpu_safety_factor": {
            "expected_type": float,
            "validator": lambda x: 0.0 <= x <= 1.0,
            "error_msg": "cpu_safety_factor must be a float between 0 and 1",
        },
        "max_pinned_memory_mb": {
            "expected_type": (int, float),
            "validator": lambda x: x >= 0,
            "error_msg": "max_pinned_memory_mb must be a non-negative number",
        },
        # Disk offloading
        "disk_offload_dir": {
            "expected_type": (type(None), str),
            "validator": None,
            "error_msg": "disk_offload_dir must be a string path or None",
        },
        "disk_min_free_mb": {
            "expected_type": (int, float),
            "validator": lambda x: x >= 0,
            "error_msg": "disk_min_free_mb must be a non-negative number",
        },
        "disk_flush_mb": {
            "expected_type": (int, float),
            "validator": lambda x: x >= 0,
            "error_msg": "disk_flush_mb must be a non-negative number",
        },
        "disk_cleanup": {"expected_type": bool, "validator": None, "error_msg": "disk_cleanup must be a boolean"},
        "disk_file_prefix": {"expected_type": str, "validator": None, "error_msg": "disk_file_prefix must be a string"},
        "disk_dtype": {
            "expected_type": (type(None), torch.dtype),
            "validator": None,
            "error_msg": "disk_dtype must be a torch.dtype or None",
        },
        "disk_safety_factor": {
            "expected_type": float,
            "validator": lambda x: 0.0 <= x <= 1.0,
            "error_msg": "disk_safety_factor must be a float between 0 and 1",
        },
        # Async transfer
        "use_async": {"expected_type": bool, "validator": None, "error_msg": "use_async must be a boolean"},
        "async_depth": {
            "expected_type": int,
            "validator": lambda x: x >= 1,
            "error_msg": "async_depth must be an integer >= 1",
        },
    }

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self._validate_and_set(key, value)

    def keys(self):
        """Return set of keys that have values set."""
        return {k for k in self._ALLOWED_KEYS if hasattr(self, k)}

    def items(self):
        """Return items as dict.items()."""
        return {k: getattr(self, k) for k in self.keys()}.items()

    def _validate_and_set(self, key, value):
        """Validate parameter type and value before setting."""
        if key not in self._ALLOWED_KEYS:
            raise KeyError(f"Invalid config key: {key}. Allowed keys: {self._ALLOWED_KEYS}")

        type_spec = self._TYPE_SPECS.get(key)
        expected_type = type_spec["expected_type"]
        validator = type_spec["validator"]
        error_msg = type_spec["error_msg"]

        if not isinstance(value, expected_type):
            raise TypeError(f"{error_msg}. Got {type(value).__name__}")

        if validator and not validator(value):
            raise ValueError(error_msg)

        setattr(self, key, value)

    def __iter__(self):
        """Return iterator over allowed keys that have values set."""
        return iter(k for k in self._ALLOWED_KEYS if hasattr(self, k))

    def __getitem__(self, key):
        if key not in self._ALLOWED_KEYS:
            raise KeyError(f"Invalid config key: {key}. Allowed keys: {self._ALLOWED_KEYS}")
        return getattr(self, key, None)

    def get(self, key, default=None):
        """Get value for key or raise KeyError if key doesn't exist.

        Parameters
        ----------
        key : str
            The configuration key to get.

        default : Any, default=None
            Default value to return if the key exists but the value is None.

        Returns
        -------
        Any
            The value for the key, or default if the value is None.
        """
        try:
            value = self[key]
            return default if value is None else value
        except KeyError:
            raise KeyError(f"Invalid config key: {key}. Allowed keys: {self._ALLOWED_KEYS}")

    def update(self, other):
        """Update configuration with values from another dict-like object."""
        if not isinstance(other, (dict, MgrConfig)):
            raise TypeError(f"Expected dict or MgrConfig, got {type(other)}")

        for key, value in other.items() if isinstance(other, dict) else vars(other).items():
            self._validate_and_set(key, value)
        return self


@dataclass
class InferenceConfig:
    """Configuration class for inference."""

    COL_CONFIG: MgrConfig = None
    ROW_CONFIG: MgrConfig = None
    ICL_CONFIG: MgrConfig = None

    def __post_init__(self):
        if isinstance(self.COL_CONFIG, dict):
            self.COL_CONFIG = MgrConfig(**self.COL_CONFIG)
        elif self.COL_CONFIG is None:
            self.COL_CONFIG = MgrConfig(
                # General
                device=None,
                use_amp=True,
                use_fa3=True,
                verbose=False,
                # Batching
                min_batch_size=1,
                safety_factor=0.8,
                # Offloading
                offload="auto",
                auto_offload_threshold=0.5,
                # CPU offloading
                cpu_safety_factor=0.85,
                max_pinned_memory_mb=32768.0,  # 32 GB
                # Disk offloading
                disk_offload_dir=None,
                disk_min_free_mb=1024.0,  # 1 GB
                disk_flush_mb=8192.0,  # 8 GB
                disk_cleanup=True,
                disk_file_prefix="",
                disk_dtype=None,
                disk_safety_factor=0.95,
                # Async transfer
                use_async=True,
                async_depth=4,
            )
        elif not isinstance(self.COL_CONFIG, MgrConfig):
            raise TypeError(f"COL_CONFIG must be a dict or MgrConfig, got {type(self.COL_CONFIG)}")

        if isinstance(self.ROW_CONFIG, dict):
            self.ROW_CONFIG = MgrConfig(**self.ROW_CONFIG)
        elif self.ROW_CONFIG is None:
            self.ROW_CONFIG = MgrConfig(
                # General
                device=None,
                use_amp=True,
                use_fa3=True,
                verbose=False,
                # Batching
                min_batch_size=1,
                safety_factor=0.8,
                # Offloading
                offload=False,
                auto_offload_threshold=0.5,
                # CPU offloading
                cpu_safety_factor=0.85,
                max_pinned_memory_mb=32768.0,  # 32 GB
                # Disk offloading
                disk_offload_dir=None,
                disk_min_free_mb=1024.0,  # 1 GB
                disk_flush_mb=8192.0,  # 8 GB
                disk_cleanup=True,
                disk_file_prefix="",
                disk_dtype=None,
                disk_safety_factor=0.95,
                # Async transfer
                use_async=True,
                async_depth=4,
            )
        elif not isinstance(self.ROW_CONFIG, MgrConfig):
            raise TypeError(f"ROW_CONFIG must be a dict or MgrConfig, got {type(self.ROW_CONFIG)}")

        if isinstance(self.ICL_CONFIG, dict):
            self.ICL_CONFIG = MgrConfig(**self.ICL_CONFIG)
        elif self.ICL_CONFIG is None:
            self.ICL_CONFIG = MgrConfig(
                # General
                device=None,
                use_amp=True,
                use_fa3=True,
                verbose=False,
                # Batching
                min_batch_size=1,
                safety_factor=0.8,
                # Offloading
                offload=False,
                auto_offload_threshold=0.5,
                # CPU offloading
                cpu_safety_factor=0.85,
                max_pinned_memory_mb=32768.0,  # 32 GB
                # Disk offloading
                disk_offload_dir=None,
                disk_min_free_mb=1024.0,  # 1 GB
                disk_flush_mb=8192.0,  # 8 GB
                disk_cleanup=True,
                disk_file_prefix="",
                disk_dtype=None,
                disk_safety_factor=0.95,
                # Async transfer
                use_async=True,
                async_depth=4,
            )
        elif not isinstance(self.ICL_CONFIG, MgrConfig):
            raise TypeError(f"ICL_CONFIG must be a dict or MgrConfig, got {type(self.ICL_CONFIG)}")

    def update_from_dict(self, config_dict: Dict[str, Dict]):
        """Update configurations from a dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Dict]
            Dictionary containing configuration updates for ``COL_CONFIG``,
            ``ROW_CONFIG``, and/or ``ICL_CONFIG``.

        Raises
        ------
        KeyError
            If dictionary contains keys other than the allowed configuration
            names.
        """
        allowed_keys = {"COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG"}
        for key in config_dict:
            if key not in allowed_keys:
                raise KeyError(f"Invalid config key: {key}. Allowed keys: {allowed_keys}")

            getattr(self, key).update(config_dict[key])
