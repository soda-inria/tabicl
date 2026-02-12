from __future__ import annotations

import sys
import copy
import pickle
import warnings
from pathlib import Path
from typing import Optional
from collections import OrderedDict

import numpy as np
import torch

import sklearn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from tabicl import TabICL, InferenceConfig


def _check_version_compatibility(metadata: dict) -> None:
    """Warn if saved versions differ from current environment."""
    checks = [
        ("sklearn_version", sklearn.__version__, "scikit-learn"),
        ("torch_version", torch.__version__, "torch"),
        ("numpy_version", np.__version__, "numpy"),
    ]
    for key, current, name in checks:
        saved = metadata.get(key)
        if saved is not None and saved != current:
            warnings.warn(
                f"This file was saved with {name}=={saved} but you are running {name}=={current}. "
                f"This may cause errors or incorrect results.",
                UserWarning,
                stacklevel=3,
            )


class TabICLBaseEstimator(BaseEstimator):
    """Base class for TabICL scikit-learn estimators.

    Provides shared functionality for both :class:`TabICLClassifier` and
    :class:`TabICLRegressor`:

    - Device resolution and inference configuration
    - Model persistence (pickle serialization/deserialization)
    - Save/load convenience methods with options for model weights, KV cache,
      and training data

    This class should not be instantiated directly. Use
    :class:`TabICLClassifier` or :class:`TabICLRegressor` instead.
    """

    def _more_tags(self):
        return dict(non_deterministic=True)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags

    def _resolve_device(self) -> None:
        """Resolve the target device from the init parameter."""
        if self.device is None:
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(self.device, str):
            self.device_ = torch.device(self.device)
        else:
            self.device_ = self.device

    def _resolve_amp_fa3(self) -> tuple:
        """Resolve the ``"auto"`` option for ``use_amp`` and ``use_fa3``.

        Called by ``_build_inference_config`` at ``fit()`` and ``__setstate__``.
        Explicit bool values are returned as-is, while ``"auto"`` triggers a simple
        heuristic based on ``n_samples_in_`` and ``n_features_in_``:

        +--------------------------------------+-------+-------+
        | Regime                               |  AMP  |  FA3  |
        +======================================+=======+=======+
        | Small  (n < 1024 & feat < 60)        |  off  |  off  |
        +--------------------------------------+-------+-------+
        | Medium (above small, n < 10240)      |  on   |  off  |
        +--------------------------------------+-------+-------+
        | Large  (n >= 10240)                  |  on   |  on   |
        +--------------------------------------+-------+-------+

        When ``use_amp=False`` (explicitly disabled by the user) and the data
        is above the small threshold, FA3 is enabled as a fallback accelerator.

        The thresholds are based on preliminary observations and are not rigorously
        tuned. It assumes that the training set is large relative to the test set
        and does not account for KV-cache scenarios. If the auto behaviour is suboptimal
        for your workload, set explicit bool values for ``use_amp`` and ``use_fa3``.

        Returns
        -------
        use_amp : bool
            Whether to enable automatic mixed precision.

        use_fa3 : bool
            Whether to enable Flash Attention 3.
        """
        n_samples = getattr(self, "n_samples_in_", 0)
        n_features = getattr(self, "n_features_in_", 0)
        small_data = n_samples < 1024 and n_features < 60

        # -- AMP --
        if self.use_amp == "auto":
            use_amp = not small_data
        else:
            use_amp = bool(self.use_amp)

        # -- FA3 --
        if self.use_fa3 == "auto":
            if small_data:
                use_fa3 = False
            elif not use_amp:
                # AMP is off and use FA3 as the main accelerator for attention
                use_fa3 = True
            else:
                # AMP is on and FA3 only adds meaningful benefit at large scale
                use_fa3 = n_samples >= 10240
        else:
            use_fa3 = bool(self.use_fa3)

        return use_amp, use_fa3

    def _build_inference_config(self) -> None:
        """Build the inference configuration from init parameters.

        This is called during fit() and also during __setstate__ to reconstruct
        the inference config after loading from a persisted file.
        """
        use_amp, use_fa3 = self._resolve_amp_fa3()

        init_config = {
            "COL_CONFIG": {
                "device": self.device_,
                "use_amp": use_amp,
                "use_fa3": use_fa3,
                "verbose": self.verbose,
                "offload": self.offload_mode,
                "disk_offload_dir": self.disk_offload_dir,
            },
            "ROW_CONFIG": {"device": self.device_, "use_amp": use_amp, "use_fa3": use_fa3, "verbose": self.verbose},
            "ICL_CONFIG": {"device": self.device_, "use_amp": use_amp, "use_fa3": use_fa3, "verbose": self.verbose},
        }
        if self.inference_config is None:
            self.inference_config_ = InferenceConfig()
            self.inference_config_.update_from_dict(init_config)
        elif isinstance(self.inference_config, dict):
            self.inference_config_ = InferenceConfig()
            for key, value in self.inference_config.items():
                if key in init_config:
                    init_config[key].update(value)
            self.inference_config_.update_from_dict(init_config)
        else:
            self.inference_config_ = self.inference_config

    def __getstate__(self):
        """Customize pickle serialization.

        The fitted state is partitioned into three categories:

        **Always excluded (reconstructed on load by ``__setstate__``):**

        - ``model_`` — The nn.Module is never fine-tuned, so its weights are
          identical to the checkpoint and can be reloaded. Excluding it avoids
          saving 10s-100s MB of redundant data.
        - ``device_`` — Hardware-specific (e.g. ``cuda:0``); meaningless on a
          different machine. Re-derived from the ``device`` init parameter.
        - ``inference_config_`` — Contains ``MgrConfig`` objects with device
          references. Rebuilt from init parameters.
        - ``model_path_`` — Absolute filesystem path to the checkpoint; almost
          certainly invalid on another machine.

        **Optionally included (controlled by ``save()`` flags):**

        - Model weights — When ``save_model_weights=True``, the state dict is
          saved on CPU, making the file self-contained (no checkpoint needed).
        - KV cache — Saved by default (moved to CPU for device portability).
          Excluding it yields a smaller file but requires rebuilding the cache.
        - Training data — Saved by default. Can be excluded when KV cache is
          present, which is useful for data privacy since the KV cache encodes
          the training context and enables prediction without raw data.

        **Always included:**

        - ``ensemble_generator_``, ``X_encoder_``, ``y_encoder_``/``y_scaler_``,
          ``classes_``, ``n_classes_``, ``n_features_in_``, and all init
          parameters — required for prediction and sklearn compatibility.
        - Version metadata — embedded for compatibility checking on load.
        """
        state = self.__dict__.copy()

        # Read save options set by save(), or use defaults for direct pickle
        save_model_weights = state.pop("_save_model_weights", False)
        save_kv_cache = state.pop("_save_kv_cache", True)
        save_training_data = state.pop("_save_training_data", True)

        # Always exclude device-specific and reconstructable attributes
        state.pop("device_", None)
        state.pop("inference_config_", None)
        state.pop("model_path_", None)

        # Handle model weights
        if save_model_weights:
            # Save state dict (not nn.Module itself)
            state["_model_state_dict"] = {k: v.cpu() for k, v in self.model_.state_dict().items()}
            # model_config_ stays in state
        else:
            state.pop("model_config_", None)
        state.pop("model_", None)

        # Handle KV cache
        if save_kv_cache and state.get("model_kv_cache_") is not None:
            cpu_cache = OrderedDict()
            for method, cache in state["model_kv_cache_"].items():
                cpu_cache[method] = cache.to("cpu")
            state["model_kv_cache_"] = cpu_cache
        else:
            state["model_kv_cache_"] = None

        # Handle training data
        if not save_training_data:
            eg = copy.deepcopy(state["ensemble_generator_"])
            eg.X_ = None
            eg.y_ = None
            for prep in eg.preprocessors_.values():
                prep.X_transformed_ = None
            state["ensemble_generator_"] = eg

        # Version metadata
        state["_persistence_metadata"] = {
            "sklearn_version": sklearn.__version__,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "saved_model_weights": save_model_weights,
            "saved_kv_cache": save_kv_cache,
            "saved_training_data": save_training_data,
        }

        return state

    def __setstate__(self, state):
        """Customize pickle deserialization.

        Reconstructs device-specific attributes and reloads the model.
        """

        metadata = state.pop("_persistence_metadata", None)
        model_state_dict = state.pop("_model_state_dict", None)

        # Restore instance state
        self.__dict__.update(state)

        # Check version compatibility
        if metadata:
            _check_version_compatibility(metadata)

        # Resolve device
        self._resolve_device()

        # Reload or reconstruct model
        if model_state_dict is not None and hasattr(self, "model_config_"):
            # Model weights were saved and reconstruct without checkpoint
            self.model_ = TabICL(**self.model_config_)
            self.model_.load_state_dict(model_state_dict)
            self.model_.eval()
        else:
            # Reload from checkpoint
            self._load_model()
        self.model_.to(self.device_)

        # Reconstruct inference config
        self._build_inference_config()

        # Move KV cache to device
        if hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None:
            device_cache = OrderedDict()
            for method, cache in self.model_kv_cache_.items():
                device_cache[method] = cache.to(self.device_)
            self.model_kv_cache_ = device_cache

    def save(
        self,
        path: str | Path,
        save_model_weights: bool = False,
        save_training_data: bool = True,
        save_kv_cache: bool = True,
    ) -> None:
        """Save the fitted estimator to a file.

        The saved file can be loaded back using the class's ``load()`` method,
        ``pickle.load()``, or ``joblib.load()``.

        Parameters
        ----------
        path : str or Path
            File path to save to.

        save_model_weights : bool, default=False
            If True, include the pretrained TabICL model weights in the file,
            making it fully self-contained. If False (default), model weights
            are reloaded from the checkpoint on load, resulting in a smaller file.

        save_training_data : bool, default=True
            If True (default), include the training data used for in-context
            learning. If False, training data is excluded for a smaller file
            and better data privacy. This is only allowed when ``save_kv_cache``
            is True and a KV cache exists, since predictions require either
            cached KV projections or the original training data.

        save_kv_cache : bool, default=True
            If True (default) and KV cache exists, include the cached key-value
            projections. If False, the KV cache is excluded, resulting in a
            smaller file but requiring a rebuild for fast inference.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted.

        ValueError
            If training data is excluded without a KV cache to fall back on.
        """
        check_is_fitted(self)

        has_kv_cache = hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None
        if not save_training_data and not (save_kv_cache and has_kv_cache):
            raise ValueError(
                "Cannot exclude training data when KV cache is not available or not being saved. "
                "Either set save_training_data=True, or fit with kv_cache=True and set save_kv_cache=True."
            )

        # Set temporary flags for __getstate__
        self._save_model_weights = save_model_weights
        self._save_kv_cache = save_kv_cache
        self._save_training_data = save_training_data
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=5)
        finally:
            del self._save_model_weights
            del self._save_kv_cache
            del self._save_training_data

    @classmethod
    def load(cls, path: str | Path, device: Optional[str | torch.device] = None) -> TabICLBaseEstimator:
        """Load a fitted estimator from a file.

        Parameters
        ----------
        path : str or Path
            File path to load from.

        device : str, torch.device, or None, default=None
            Device to use after loading. If None, the device is determined
            by the estimator's ``device`` init parameter (which defaults to
            auto-detection: CUDA if available, else CPU).

        Returns
        -------
        estimator : TabICLBaseEstimator
            The loaded fitted estimator.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)

        if device is not None:
            obj.device = device
            obj._resolve_device()
            obj.model_.to(obj.device_)
            obj._build_inference_config()
            if hasattr(obj, "model_kv_cache_") and obj.model_kv_cache_ is not None:
                device_cache = OrderedDict()
                for method, cache in obj.model_kv_cache_.items():
                    device_cache[method] = cache.to(obj.device_)
                obj.model_kv_cache_ = device_cache

        return obj

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1, temperature: float = 0.9) -> np.ndarray:
        """Compute softmax values with temperature scaling using NumPy.

        Computes :math:`\text{softmax}(x / \tau)` where :math:`\tau` is the
        temperature parameter.

        Parameters
        ----------
        x : ndarray
            Input array of logits.

        axis : int, default=-1
            Axis along which to compute softmax.

        temperature : float, default=0.9
            Temperature scaling parameter :math:`\tau`.

        Returns
        -------
        ndarray
            Softmax probabilities along the specified axis, with the same shape
            as the input.
        """
        x = x / temperature
        # Subtract max for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        # Compute softmax
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
