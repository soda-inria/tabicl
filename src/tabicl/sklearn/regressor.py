from __future__ import annotations

import warnings
from pathlib import Path
import multiprocessing as mp
from collections import OrderedDict
from typing import Optional, List, Dict

import numpy as np
import torch

from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError

from .base import TabICLBaseEstimator
from .preprocessing import TransformToNumerical, EnsembleGenerator
from .sklearn_utils import validate_data, _num_samples

from tabicl import TabICL, TabICLCache, InferenceConfig


class TabICLRegressor(RegressorMixin, TabICLBaseEstimator):
    """Tabular In-Context Learning (TabICL) Regressor with scikit-learn interface.

    This regressor applies TabICL to tabular data regression, using an ensemble
    of transformed dataset views to improve predictions. The ensemble members are
    created by applying different normalization methods and feature permutations.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of estimators for ensemble predictions.

    norm_methods : str or list[str] or None, default=None
        Normalization methods to apply:
        - 'none': No normalization
        - 'power': Yeo-Johnson power transform
        - 'quantile': Transform features to an approximately normal distribution.
        - 'quantile_rtdl': Quantile transform that adds noise to training data before fitting.
        - 'robust': Scale using median and quantiles
        Can be a single string or a list of methods to use across ensemble members.
        When set to None, it will use ["none", "power"].

    feat_shuffle_method : str, default='latin'
        Feature permutation strategy:
        - 'none': No shuffling and preserve original feature order
        - 'shift': Circular shifting of feature columns
        - 'random': Random permutation of features
        - 'latin': Latin square patterns for systematic feature permutations

    outlier_threshold : float, default=4.0
        Z-score threshold for outlier detection and clipping. Values with
        :math:`|z| > \text{threshold}` are considered outliers.

    batch_size : Optional[int], default=8
        Batch size for inference. If None, all ensemble members are processed in a single batch.
        Adjust this parameter based on available memory. Lower values use less memory but may
        be slower.

    model_path : Optional[str or Path], default=None
        Path to the pre-trained model checkpoint file.
        - If provided and the file exists, it's loaded directly.
        - If provided but the file doesn't exist and `allow_auto_download` is true, the version
          specified by `checkpoint_version` is downloaded from Hugging Face Hub (repo: 'jingang/TabICL')
          to this path.
        - If `None` (default), the version specified by `checkpoint_version` is downloaded from
          Hugging Face Hub (repo: 'jingang/TabICL') and cached locally in the default
          Hugging Face cache directory (typically `~/.cache/huggingface/hub`).

    allow_auto_download : bool, default=True
        Whether to allow automatic download if the pretrained checkpoint cannot be found at the
        specified `model_path`.

    checkpoint_version : str, default='tabicl-regressor-v2-20260212.ckpt'
        Specifies which version of the pre-trained model checkpoint to use when `model_path`
        is `None` or points to a non-existent file (and `allow_auto_download` is true).
        Checkpoints are downloaded from https://huggingface.co/jingang/TabICL.

    device : Optional[str or torch.device], default=None
        Device to use for inference. If None, defaults to CUDA if available, else CPU.
        Can be specified as a string ('cuda', 'cpu') or a torch.device object.

    use_amp : bool or "auto", default="auto"
        Controls automatic mixed precision (AMP) for inference.
        - True / False: force on / off.
        - "auto": Automatically enable AMP based on input data size using the following heuristic:

            +--------------------------------------+-------+-------+
            | Regime                               |  AMP  |  FA3  |
            +======================================+=======+=======+
            | Small  (n < 1024 & feat < 60)        |  off  |  off  |
            +--------------------------------------+-------+-------+
            | Medium (above small, n < 10240)      |  on   |  off  |
            +--------------------------------------+-------+-------+
            | Large  (n >= 10240)                  |  on   |  on   |
            +--------------------------------------+-------+-------+

            The above heuristic is based on the observation that AMP can introduce overhead that outweighs
            its benefits for small inputs. In addition, it assumes that the training set is large relative to
            the test set and does not account for KV-cache scenarios. If it is suboptimal for your workload,
            set it explicitly.

    use_fa3 : bool or "auto", default="auto"
        Whether to use Flash Attention 3 that can speed up inference for large datasets on NVIDIA Hopper
        GPUs like H100. Only effective when FA3 is installed.
        - True / False: force on / off.
        - "auto": Automatically enable FA3 based on input data size using a simple heuristic (see above).

    offload_mode : str or bool, default='auto'
        Controls where column-wise embedding outputs are stored during inference.
        Column-wise embedding produces a large tensor of shape
        (batch_size, n_rows, n_columns, embed_dim) which is the main memory bottleneck.
        Available options:
        - ``'auto'``: Automatically choose based on available memory (default).
        - ``'gpu'`` or ``False``: Keep on GPU. Fastest but limited by VRAM.
        - ``'cpu'`` or ``True``: Offload to CPU memory.
        - ``'disk'``: Offload to memory-mapped files (requires ``disk_offload_dir``).

        It only affects column-wise embedding (COL_CONFIG). For finer-grained control
        over all components, use ``inference_config``.

    disk_offload_dir : Optional[str], default=None
        Directory for memory-mapped files used when ``offload_mode='disk'`` or when
        ``offload_mode='auto'`` falls back to disk offloading.
        It only affects column-wise embedding (COL_CONFIG). For finer-grained control
        over all components, use ``inference_config``.

    random_state : int or None, default=42
        Random seed for reproducibility of ensemble generation, affecting feature
        shuffling and other randomized operations.

    n_jobs : int or None, default=None
        Number of threads to use for PyTorch in case the model is run on CPU.
        None means using the PyTorch default, which is the number of physical CPU cores.
        Negative numbers mean that :math:`\\max(1, n_{\\text{logical\\_cores}} + 1 + \\text{n\\_jobs})`
        threads will be used. In particular, ``n_jobs=-1`` means that all logical cores
        will be used.

    verbose : bool, default=False
        Whether to print detailed information during inference.

    inference_config : Optional[InferenceConfig | Dict[str, Dict[str, Any]]], default=None
        Configuration for inference settings. This parameter provides fine-grained control
        over the three transformers in TabICL (column-wise, row-wise, and in-context learning).

        WARNING: This parameter should only be used by advanced users who understand the internal
        architecture of TabICL and need precise control over inference.

        When None (default):
            - A new InferenceConfig object is created with default settings
            - The ``device``, ``use_amp``, ``use_fa3``, ``offload_mode``, ``disk_offload_dir``, and ``verbose``
              parameters from the class initialization are applied to the relevant components

        When Dict with allowed top-level keys "COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG":
            - A new InferenceConfig object is created with default settings
            - Any values explicitly specified in the dictionary will override default defaults
            - ``device``, ``use_amp``, ``use_fa3``, ``offload_mode``, ``disk_offload_dir``, and ``verbose``
              from the class initialization are used if they are not specified in the dictionary

        When InferenceConfig:
            - The provided InferenceConfig object is used directly without modification
            - ``device``, ``use_amp``, ``use_fa3``, ``offload_mode``, ``disk_offload_dir``, and ``verbose``
              from the class initialization are ignored
            - All settings must be explicitly defined in the provided InferenceConfig object

    Attributes
    ----------
    n_features_in_ : int
        Number of features in the training data.

    n_samples_in_ : int
        Number of samples in the training data.

    feature_names_in_ : ndarray of shape (n_features_in_,) or None
        Feature names seen during ``fit``. Only set when the input ``X`` has
        feature names (e.g., a pandas DataFrame with string column names).

    X_encoder_ : TransformToNumerical
        Encoder for transforming input features to numerical values.

    y_scaler_ : StandardScaler
        Scaler for transforming target values.

    ensemble_generator_ : EnsembleGenerator
        Fitted ensemble generator that creates multiple dataset views.

    model_ : TabICL
        The loaded TabICL model used for predictions.

    model_path_ : str
        Path to the loaded checkpoint file.

    model_config_ : dict
        Configuration dictionary from the loaded checkpoint.

    device_ : torch.device
        The device where the model is loaded and computations are performed.

    inference_config_ : InferenceConfig
        The inference configuration.

    cache_mode_ : str or None
        The caching mode used when ``fit()`` was called with ``kv_cache``.
        One of ``"kv"``, ``"repr"``, or ``None`` (when no caching is used).

    model_kv_cache_ : OrderedDict[str, TabICLCache] or None
        Pre-computed KV caches for training data, keyed by normalization method.
        Created when ``fit()`` is called with ``kv_cache=True``. When set, ``predict()``
        reuses the cached key-value projections instead of re-processing training data,
        enabling faster inference on multiple test sets.
    """

    def __init__(
        self,
        n_estimators: int = 8,
        norm_methods: Optional[str | List[str]] = None,
        feat_shuffle_method: str = "latin",
        outlier_threshold: float = 4.0,
        batch_size: Optional[int] = 8,
        model_path: Optional[str | Path] = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = "tabicl-regressor-v2-20260212.ckpt",
        device: Optional[str | torch.device] = None,
        use_amp: bool | str = "auto",
        use_fa3: bool | str = "auto",
        offload_mode: str | bool = "auto",
        disk_offload_dir: Optional[str] = None,
        random_state: int | None = 42,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        inference_config: Optional[InferenceConfig | Dict] = None,
    ):
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.batch_size = batch_size
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.use_amp = use_amp
        self.use_fa3 = use_fa3
        self.offload_mode = offload_mode
        self.disk_offload_dir = disk_offload_dir
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.inference_config = inference_config

    def _load_model(self) -> None:
        """Load a model from a given path or download it if not available.

        It uses `model_path` and `checkpoint_version` to determine the source.
         - If `model_path` is specified and exists, it's used directly.
         - If `model_path` is specified but doesn't exist (and auto-download is enabled),
           the version specified by `checkpoint_version` is downloaded to `model_path`.
         - If `model_path` is None, the version specified by `checkpoint_version` is downloaded
           from Hugging Face Hub and cached in the default Hugging Face cache directory.

        Raises
        ------
        AssertionError
            If the checkpoint doesn't contain the required 'config' or 'state_dict' keys.

        ValueError
            If a checkpoint cannot be found or downloaded based on the settings.
        """

        repo_id = "jingang/TabICL"
        filename = self.checkpoint_version

        if self.model_path is None:
            # Scenario 1: the model path is not provided, so download from HF Hub based on the checkpoint version
            try:
                model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True))
            except LocalEntryNotFoundError:
                if self.allow_auto_download:
                    print(f"Checkpoint '{filename}' not cached.\n Downloading from Hugging Face Hub ({repo_id}).\n")
                    model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename))
                else:
                    raise ValueError(
                        f"Checkpoint '{filename}' not cached and automatic download is disabled.\n"
                        f"Set allow_auto_download=True to download the checkpoint from Hugging Face Hub ({repo_id})."
                    )
            if model_path_:
                checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
        else:
            # Scenario 2: the model path is provided
            model_path_ = Path(self.model_path) if isinstance(self.model_path, str) else self.model_path
            if model_path_.exists():
                # Scenario 2a: the model path exists, load it directly
                checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
            else:
                # Scenario 2b: the model path does not exist, download the checkpoint version to this path
                if self.allow_auto_download:
                    print(
                        f"Checkpoint not found at '{model_path_}'.\n"
                        f"Downloading '{filename}' from Hugging Face Hub ({repo_id}) to this location.\n"
                    )
                    model_path_.parent.mkdir(parents=True, exist_ok=True)
                    cache_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path_.parent)
                    Path(cache_path).rename(model_path_)
                    checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
                else:
                    raise ValueError(
                        f"Checkpoint not found at '{model_path_}' and automatic download is disabled.\n"
                        f"Either provide a valid checkpoint path, or set allow_auto_download=True to download "
                        f"'{filename}' from Hugging Face Hub ({repo_id})."
                    )

        assert "config" in checkpoint, "The checkpoint doesn't contain the model configuration."
        assert "state_dict" in checkpoint, "The checkpoint doesn't contain the model state."

        self.model_path_ = model_path_

        config = checkpoint["config"]
        self.model_ = TabICL(**config)
        self.model_config_ = config
        self.model_.load_state_dict(checkpoint["state_dict"])
        self.model_.eval()

    def fit(self, X: np.ndarray, y: np.ndarray, kv_cache: bool | str = False) -> TabICLRegressor:
        """Fit the regressor to training data.

        Prepares the model for prediction by:

        1. Scaling target values using StandardScaler
        2. Converting input features to numerical values
        3. Fitting the ensemble generator to create transformed dataset views
        4. Loading the pre-trained TabICL model
        5. Optionally pre-computing KV caches for training data to speed up inference

        The model itself is not trained on the data; it uses in-context learning
        at inference time. This method only prepares the data transformations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.

        y : array-like of shape (n_samples,)
            Training target values.

        kv_cache : bool or str, default=False
            Controls caching of training data computations to speed up subsequent
            ``predict`` calls.
            - False: No caching.
            - True or "kv": Cache key-value projections from both column embedding
              and ICL transformer layers. Fast inference but memory-heavy for large
              training sets.
            - "repr": Cache column embedding KV projections and row interaction outputs
              (representations). Uses ~24x less memory than "kv" for the ICL part,
              at the cost of re-running the ICL transformer at predict time.

        Returns
        -------
        self : TabICLRegressor
            Fitted regressor instance.
        """

        if y is None:
            raise ValueError("This regressor requires y to be passed, but the target y is None.")

        X, y = validate_data(self, X, y, dtype=None, skip_check_array=True)

        # Ensure y is numeric
        y = np.asarray(y, dtype=np.float32)

        # Warn and flatten 2D column-vector y
        if y.ndim == 2 and y.shape[1] == 1:
            from sklearn.exceptions import DataConversionWarning

            warnings.warn(
                "A column-vector y was passed when a 1d array was expected. Please change "
                "the shape of y to (n_samples, ), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )
            y = y.ravel()

        # Device setup
        self._resolve_device()

        # Inference configuration
        self.n_samples_in_ = _num_samples(X)
        self._build_inference_config()

        # Load the pre-trained TabICL model
        self._load_model()
        self.model_.to(self.device_)

        # Scale target values
        self.y_scaler_ = StandardScaler()
        y_scaled = self.y_scaler_.fit_transform(y.reshape(-1, 1)).flatten()

        # Transform input features
        self.X_encoder_ = TransformToNumerical(verbose=self.verbose)
        X = self.X_encoder_.fit_transform(X)

        # Fit ensemble generator to create multiple dataset views
        self.ensemble_generator_ = EnsembleGenerator(
            classification=False,
            n_estimators=self.n_estimators,
            norm_methods=self.norm_methods or ["none", "power"],
            feat_shuffle_method=self.feat_shuffle_method,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
        )
        self.ensemble_generator_.fit(X, y_scaled)

        self.model_kv_cache_ = None
        if kv_cache:
            if kv_cache is True or kv_cache == "kv":
                self.cache_mode_ = "kv"
            elif kv_cache == "repr":
                self.cache_mode_ = "repr"
            else:
                raise ValueError(f"Invalid kv_cache value '{kv_cache}'. Expected False, True, 'kv', or 'repr'.")
            self._build_kv_cache()

        return self

    def _build_kv_cache(self) -> None:
        """Pre-compute KV caches for training data across all ensemble batches."""

        # X=None is required in transform() even though it is the default value
        # because sklearn's _SetOutputMixin wraps transform() with a signature
        # that enforces X as a positional argument.
        train_data = self.ensemble_generator_.transform(X=None, mode="train")
        self.model_kv_cache_ = OrderedDict()

        for norm_method, (Xs, ys) in train_data.items():
            batch_size = self.batch_size or Xs.shape[0]
            n_batches = int(np.ceil(Xs.shape[0] / batch_size))
            Xs_split = np.array_split(Xs, n_batches)
            ys_split = np.array_split(ys, n_batches)

            caches = []
            for X_batch, y_batch in zip(Xs_split, ys_split):
                X_batch = torch.from_numpy(X_batch).float().to(self.device_)
                y_batch = torch.from_numpy(y_batch).float().to(self.device_)
                with torch.no_grad():
                    self.model_.predict_stats_with_cache(
                        X_train=X_batch,
                        y_train=y_batch,
                        use_cache=False,
                        store_cache=True,
                        cache_mode=self.cache_mode_,
                        inference_config=self.inference_config_,
                    )
                caches.append(self.model_._cache)
                self.model_.clear_cache()

            # Merge all batch caches into a single cache
            self.model_kv_cache_[norm_method] = TabICLCache.concat(caches)

    def _batch_forward(
        self,
        Xs: np.ndarray,
        ys: np.ndarray,
        output_type: str | list[str] = "mean",
        alphas: Optional[List[float]] = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Process model forward passes in batches to manage memory efficiently.

        This method handles the batched inference through the TabICL model,
        dividing the ensemble members into smaller batches to avoid out-of-memory errors.

        Parameters
        ----------
        Xs : np.ndarray
            Input features of shape ``(n_datasets, n_samples, n_features)``, where
            ``n_datasets`` is the number of ensemble members.

        ys : np.ndarray
            Training labels of shape ``(n_datasets, train_size)``, where ``train_size``
            is the number of samples used for in-context learning.

        output_type : str or list of str, default="mean"
            Type of output to return (``"mean"``, ``"median"``, ``"variance"``,
            or ``"quantiles"``).

        alphas : list of float or None, default=None
            Probability levels to return if ``output_type`` includes ``"quantiles"``.

        Returns
        -------
        np.ndarray or dict[str, np.ndarray]
            Model outputs. Shape depends on ``output_type``.
        """

        batch_size = self.batch_size or Xs.shape[0]
        n_batches = np.ceil(Xs.shape[0] / batch_size)
        Xs = np.array_split(Xs, n_batches)
        ys = np.array_split(ys, n_batches)

        output_type = [output_type] if isinstance(output_type, str) else output_type
        results = {key: [] for key in output_type}

        for X_batch, y_batch in zip(Xs, ys):
            X_batch = torch.from_numpy(X_batch).float().to(self.device_)
            y_batch = torch.from_numpy(y_batch).float().to(self.device_)

            with torch.no_grad():
                out = self.model_.predict_stats(
                    X_batch,
                    y_batch,
                    output_type=output_type,
                    alphas=alphas,
                    inference_config=self.inference_config_,
                )
                if isinstance(out, dict):
                    for key in output_type:
                        results[key].append(out[key].float().cpu().numpy())
                else:
                    results[output_type[0]].append(out.float().cpu().numpy())

        # Concatenate batches
        for key in results:
            results[key] = np.concatenate(results[key], axis=0)

        if len(output_type) == 1:
            return results[output_type[0]]

        return results

    def _batch_forward_with_cache(
        self,
        Xs: np.ndarray,
        kv_cache: TabICLCache,
        output_type: str | list[str] = "mean",
        alphas: Optional[List[float]] = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Process model forward passes using a pre-computed KV cache.

        The cache is sliced along the batch dimension to match each batch.

        Parameters
        ----------
        Xs : np.ndarray
            Test features of shape ``(n_datasets, test_size, n_features)``.

        kv_cache : TabICLCache
            Single KV cache for all estimators of a normalization method.

        output_type : str or list of str, default="mean"
            Type of output to return (``"mean"``, ``"median"``, ``"variance"``,
            or ``"quantiles"``).

        alphas : list of float or None, default=None
            Probability levels to return if ``output_type`` includes ``"quantiles"``.

        Returns
        -------
        np.ndarray or dict[str, np.ndarray]
            Model outputs. Shape depends on ``output_type``.
        """
        n_total = Xs.shape[0]
        batch_size = self.batch_size or n_total
        n_batches = int(np.ceil(n_total / batch_size))
        Xs_split = np.array_split(Xs, n_batches)

        output_type = [output_type] if isinstance(output_type, str) else output_type
        results = {key: [] for key in output_type}

        offset = 0
        for X_batch in Xs_split:
            bs = X_batch.shape[0]
            cache_subset = kv_cache.slice_batch(offset, offset + bs)
            offset += bs

            X_batch = torch.from_numpy(X_batch).float().to(self.device_)
            with torch.no_grad():
                out = self.model_.predict_stats_with_cache(
                    X_test=X_batch,
                    output_type=output_type,
                    alphas=alphas,
                    cache=cache_subset,
                    inference_config=self.inference_config_,
                )
                if isinstance(out, dict):
                    for key in output_type:
                        results[key].append(out[key].float().cpu().numpy())
                else:
                    results[output_type[0]].append(out.float().cpu().numpy())

        # Concatenate batches
        for key in results:
            results[key] = np.concatenate(results[key], axis=0)

        if len(output_type) == 1:
            return results[output_type[0]]

        return results

    def predict(
        self, X: np.ndarray, output_type: str | list[str] = "mean", alphas: Optional[List[float]] = None
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Predict target values for test samples.

        Applies the ensemble of TabICL models to make predictions, with each ensemble
        member providing predictions that are then averaged. The method:

        1. Transforms input data using the fitted encoders
        2. Applies the ensemble generator to create multiple views
        3. Forwards each view through the model
        4. Averages predictions across ensemble members
        5. Inverse transforms predictions to original scale

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples for prediction.

        output_type : str or list of str, default="mean"
            Determines the type of output to return.
            - If ``"mean"``, returns the mean over the predicted distribution.
            - If ``"median"``, returns the median over the predicted distribution.
            - If ``"quantiles"``, returns the quantiles of the predicted distribution.
              The parameter ``alphas`` determines which quantiles are returned.
            - If a list of str, returns multiple types of outputs as specified in the list.

        alphas : list of float or None, default=None
            The probability levels to return if ``output_type="quantiles"``.

            By default, the ``[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]``
            quantiles are returned. The predictions per quantile match
            the input order.

        Returns
        -------
        np.ndarray of shape (n_samples,) or dict[str, np.ndarray]
            An array of shape ``(n_samples,)`` if ``output_type`` is ``"mean"`` or
            ``"median"``, or an array of shape ``(n_samples, n_quantiles)`` if
            ``output_type`` is ``"quantiles"``.

            If ``output_type`` is a list of str, returns a dictionary with keys as
            specified in the list and values as the corresponding predictions.
        """
        check_is_fitted(self)
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            # Reject 1D arrays to maintain sklearn compatibility
            raise ValueError("The provided input X is one-dimensional. Reshape your data.")

        # Check if prediction is possible
        has_kv_cache = hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None
        has_training_data = (
            hasattr(self, "ensemble_generator_") and getattr(self.ensemble_generator_, "X_", None) is not None
        )
        if not has_kv_cache and not has_training_data:
            raise RuntimeError(
                "Cannot predict: this estimator was saved without training data and has no KV cache. "
                "Predictions require either cached KV projections or the original training data. "
                "Re-fit the estimator or load from a file saved with save_training_data=True or "
                "save_kv_cache=True."
            )

        if self.n_jobs is not None:
            assert self.n_jobs != 0
            old_n_threads = torch.get_num_threads()
            n_logical_cores = mp.cpu_count()

            if self.n_jobs > 0:
                if self.n_jobs > n_logical_cores:
                    warnings.warn(
                        f"TabICL got n_jobs={self.n_jobs} but there are only {n_logical_cores} logical cores available."
                        f" Only {n_logical_cores} threads will be used."
                    )
                n_threads = max(n_logical_cores, self.n_jobs)
            else:
                n_threads = max(1, mp.cpu_count() + 1 + self.n_jobs)

            torch.set_num_threads(n_threads)

        # Preserve DataFrame structure to retain column names and types for correct feature transformation
        X = validate_data(self, X, reset=False, dtype=None, skip_check_array=True)
        X = self.X_encoder_.transform(X)

        output_type = [output_type] if isinstance(output_type, str) else list(output_type)

        if hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None:
            # Cache exists: forward only test data and use the pre-computed cache for training data
            test_data = self.ensemble_generator_.transform(X, mode="test")
            results = {key: [] for key in output_type}
            for norm_method, (Xs_test,) in test_data.items():
                kv_cache = self.model_kv_cache_[norm_method]
                batch_out = self._batch_forward_with_cache(Xs_test, kv_cache, output_type=output_type, alphas=alphas)
                if isinstance(batch_out, dict):
                    for key in output_type:
                        results[key].append(batch_out[key])
                else:
                    results[output_type[0]].append(batch_out)
        else:
            # No cache: forward both training and test data
            data = self.ensemble_generator_.transform(X, mode="both")
            results = {key: [] for key in output_type}
            for Xs, ys in data.values():
                batch_out = self._batch_forward(Xs, ys, output_type=output_type, alphas=alphas)
                if isinstance(batch_out, dict):
                    for key in output_type:
                        results[key].append(batch_out[key])
                else:
                    results[output_type[0]].append(batch_out)

        # Concatenate across ensemble members and apply inverse transform
        final_results = {}
        for key in output_type:
            arr = np.concatenate(results[key], axis=0)
            n_estimators = arr.shape[0]
            n_samples = arr.shape[1]

            if arr.ndim == 2:
                # mean, variance, or median: (n_estimators, n_samples)
                arr = self.y_scaler_.inverse_transform(arr.reshape(-1, 1)).reshape(n_estimators, n_samples)
                final_results[key] = np.mean(arr, axis=0)
            else:
                # quantiles: (n_estimators, n_samples, n_quantiles)
                n_quantiles = arr.shape[2]
                arr = self.y_scaler_.inverse_transform(arr.reshape(-1, 1)).reshape(n_estimators, n_samples, n_quantiles)
                final_results[key] = np.mean(arr, axis=0)

        if self.n_jobs is not None:
            torch.set_num_threads(old_n_threads)

        if len(output_type) == 1:
            return final_results[output_type[0]]

        return final_results
