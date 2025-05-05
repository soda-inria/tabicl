from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


class MgrConfig:
    """Config class for `InferenceManager`.

    Allowed keys:
    - min_batch_size: Minimum batch size to try before raising an error
    - safety_factor: Factor to multiply estimated batch size by for conservative memory usage
    - offload: Whether to offload intermediate results to CPU
    - auto_offload_pct: Threshold for automatic offloading when offload="auto"
    - device: Device to use for inference
    - use_amp: Whether to use automatic mixed precision during inference
    - verbose: Whether to print detailed information during inference
    """

    _ALLOWED_KEYS = {"min_batch_size", "safety_factor", "offload", "auto_offload_pct", "device", "use_amp", "verbose"}
    _TYPE_SPECS = {
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
        "offload": {
            "expected_type": (bool, str),
            "validator": lambda x: isinstance(x, bool) or x == "auto",
            "error_msg": "offload must be a boolean or the string 'auto'",
        },
        "auto_offload_pct": {
            "expected_type": float,
            "validator": lambda x: 0.0 <= x <= 1.0,
            "error_msg": "auto_offload_pct must be a float between 0 and 1",
        },
        "device": {
            "expected_type": (type(None), str, torch.device),
            "validator": None,
            "error_msg": "device must be a string or torch.device",
        },
        "use_amp": {"expected_type": bool, "validator": None, "error_msg": "use_amp must be a boolean"},
        "verbose": {"expected_type": bool, "validator": None, "error_msg": "verbose must be a boolean"},
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
            The configuration key to get

        default : Any, default=None
            Default value to return if the key exists but the value is None

        Returns
        -------
        Any
            The value for the key, or default if the value is None
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
                min_batch_size=1,
                safety_factor=0.8,
                offload="auto",
                auto_offload_pct=0.5,
                device=None,
                use_amp=True,
                verbose=False,
            )
        elif not isinstance(self.COL_CONFIG, MgrConfig):
            raise TypeError(f"COL_CONFIG must be a dict or MgrConfig, got {type(self.COL_CONFIG)}")

        if isinstance(self.ROW_CONFIG, dict):
            self.ROW_CONFIG = MgrConfig(**self.ROW_CONFIG)
        elif self.ROW_CONFIG is None:
            self.ROW_CONFIG = MgrConfig(
                min_batch_size=1,
                safety_factor=0.8,
                offload=False,
                auto_offload_pct=0.5,
                device=None,
                use_amp=True,
                verbose=False,
            )
        elif not isinstance(self.ROW_CONFIG, MgrConfig):
            raise TypeError(f"ROW_CONFIG must be a dict or MgrConfig, got {type(self.ROW_CONFIG)}")

        if isinstance(self.ICL_CONFIG, dict):
            self.ICL_CONFIG = MgrConfig(**self.ICL_CONFIG)
        elif self.ICL_CONFIG is None:
            self.ICL_CONFIG = MgrConfig(
                min_batch_size=1,
                safety_factor=0.8,
                offload=False,
                auto_offload_pct=0.5,
                device=None,
                use_amp=True,
                verbose=False,
            )
        elif not isinstance(self.ICL_CONFIG, MgrConfig):
            raise TypeError(f"ICL_CONFIG must be a dict or MgrConfig, got {type(self.ICL_CONFIG)}")

    def update_from_dict(self, config_dict: Dict[str, Dict]):
        """Update configurations from a dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Dict]
            Dictionary containing configuration updates for COL_CONFIG, ROW_CONFIG, and/or ICL_CONFIG

        Raises
        ------
        KeyError
            If dictionary contains keys other than the allowed configuration names
        """
        allowed_keys = {"COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG"}
        for key in config_dict:
            if key not in allowed_keys:
                raise KeyError(f"Invalid config key: {key}. Allowed keys: {allowed_keys}")

            getattr(self, key).update(config_dict[key])
