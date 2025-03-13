from __future__ import annotations

import warnings
from pathlib import Path
from packaging import version
from typing import Optional, List

import numpy as np
import torch

import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError

from .preprocessing import TransformToNumerical, EnsembleGenerator
from .model.tabicl import TabICL

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
OLD_SKLEARN = version.parse(sklearn.__version__) < version.parse("1.6")


class TabICLClassifier(ClassifierMixin, BaseEstimator):
    """Tabular In-Context Learning (TabICL) with scikit-learn interface.

    This classifier applies TabICL to tabular data classification, using an ensemble
    of transformed dataset views to improve prediction accuracy. The ensemble members
    are created by applying different normalization methods, feature permutations,
    and class label shifts.

    Parameters
    ----------
    n_estimators : int, default=32
        Number of estimators for ensemble predictions.

    norm_methods : str or list[str] or None, default=None
        Normalization methods to apply:
        - 'none': No normalization
        - 'power': Yeo-Johnson power transform
        - 'quantile': Transform features using quantiles information
        - 'robust': Scale using median and quantiles
        Can be a single string or a list of methods to use across ensemble members.
        When set to None, it will use ["none", "power"].

    feat_shuffle_method : str, default='latin'
        Feature permutation strategy:
        - 'none': No shuffling and preserve original feature order
        - 'shift': Circular shifting of feature columns
        - 'random': Random permutation of features
        - 'latin': Latin square patterns for systematic feature permutations

    class_shift : bool, default=True
        Whether to apply cyclic shifts to class labels across ensemble members.

    outlier_threshold : float, default=4.0
        Z-score threshold for outlier detection and clipping.

    softmax_temperature : float, default=0.9
        Temperature for the softmax function. Lower values make predictions more
        confident, higher values make them more conservative.

    average_logits : bool, default=True
        Whether to average the logits (True) or probabilities (False) of ensemble members.
        Averaging logits often produces better calibrated probabilities.

    use_hierarchical : bool, default=True
        Whether to enable hierarchical classification for datasets with many classes.
        Required when the number of classes exceeds the model's max_classes limit.

    use_amp : bool, default=True
        Whether to use automatic mixed precision for faster inference with minimal
        impact on prediction accuracy.

    batch_size : Optional[int] = 8
        Batch size for inference. If None, all ensemble members are processed in a single batch.
        Adjust this parameter based on available memory. Lower values use less memory but may
        be slower.

    model_path : Optional[str | Path] = None
        Path to the pre-trained model checkpoint file. If None, the model will be downloaded
        automatically from Hugging Face Hub (repo: 'jingang/TabICL-clf', file: 'tabicl-classifier.ckpt')
        and stored in the default Hugging Face cache directory (typically '~/.cache/huggingface/hub'
        where '~' represents the user's home directory). The checkpoint should contain both 'config'
        and 'state_dict' keys. If the path is provided but the file doesn't exist, it will be
        downloaded to the specified location.

    allow_auto_download: bool = True
        Whether to allow automatic download if the pretrained checkpoint cannot be found at the
        specified `model_path`.

    device : Optional[str or torch.device], default=None
        Device to use for inference. If None, defaults to CUDA if available, else CPU.
        Can be specified as a string ('cuda', 'cpu') or a torch.device object.

    random_state : int | None = 42
        Random seed for reproducibility of ensemble generation, affecting feature
        shuffling and other randomized operations.

    verbose : bool, default=False
        Whether to print detailed information during inference

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.

    n_classes_ : int
        Number of classes in the training data.

    n_features_in_ : int
        Number of features in the training data.

    X_encoder_ : TransformToNumerical
        Encoder for transforming input features to numerical values.

    y_encoder_ : LabelEncoder
        Encoder for transforming class labels to integers and back.

    ensemble_generator_ : EnsembleGenerator
        Fitted ensemble generator that creates multiple dataset views.

    model_ : TabICL
        The loaded TabICL model used for predictions.

    device_ : torch.device
        The device where the model is loaded and computations are performed.
    """

    def __init__(
        self,
        n_estimators: int = 32,
        norm_methods: Optional[str | List[str]] = None,
        feat_shuffle_method: str = "latin",
        class_shift: bool = True,
        outlier_threshold: float = 4.0,
        softmax_temperature: float = 0.9,
        average_logits: bool = True,
        use_hierarchical: bool = True,
        use_amp: bool = True,
        batch_size: Optional[int] = 8,
        model_path: Optional[str | Path] = None,
        allow_auto_download: bool = True,
        device: Optional[str | torch.device] = None,
        random_state: int | None = 42,
        verbose: bool = False,
    ):
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shift = class_shift
        self.outlier_threshold = outlier_threshold
        self.softmax_temperature = softmax_temperature
        self.average_logits = average_logits
        self.use_hierarchical = use_hierarchical
        self.use_amp = use_amp
        self.batch_size = batch_size
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _load_model(self):
        """Load a model from a given path or download it if not available.

        This method is responsible for loading the TabICL model from a checkpoint file.
        If model_path is None or the file doesn't exist at the specified location,
        it will download the model from the Hugging Face Hub.

        The checkpoint file should contain:
        - 'config': Model configuration parameters
        - 'state_dict': The trained model weights

        Raises
        ------
        AssertionError
            If the checkpoint doesn't contain the required 'config' or 'state_dict' keys.
        """

        repo_id = "jingang/TabICL-clf"
        filename = "tabicl-classifier.ckpt"

        if self.model_path is None:
            try:
                model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            except LocalEntryNotFoundError:
                if self.allow_auto_download:
                    print(f"Checkpoint not cached. Downloading from Hugging Face Hub ({repo_id})")
                    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
                else:
                    raise ValueError(
                        f"Checkpoint not cached and automatic download is disabled.\n"
                        f"Set allow_auto_download=True to download the checkpoint from Hugging Face Hub ({repo_id})."
                    )
        else:
            if isinstance(self.model_path, str):
                model_path = Path(self.model_path)
            else:
                model_path = self.model_path

            if model_path.exists():
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            else:
                if self.allow_auto_download:
                    print(f"Checkpoint not found. Downloading from Hugging Face Hub ({repo_id}) to {model_path}")
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path.parent)
                    Path(cache_path).rename(model_path)
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
                else:
                    raise ValueError(
                        f"Checkpoint not found at {model_path} and automatic download is disabled.\n"
                        f"Either provide a valid checkpoint path, or set auto_download=True to download "
                        f"the checkpoint from Hugging Face Hub ({repo_id})."
                    )

        assert "config" in checkpoint, "The checkpoint doesn't contain the model configuration."
        assert "state_dict" in checkpoint, "The checkpoint doesn't contain the model state."

        self.model_path_ = model_path
        self.model_ = TabICL(**checkpoint["config"])
        self.model_.load_state_dict(checkpoint["state_dict"])
        self.model_.eval()

    def fit(self, X, y):
        """Fit the classifier to training data.

        Prepares the model for prediction by:
        1. Encoding class labels using LabelEncoder
        2. Converting input features to numerical values
        3. Fitting the ensemble generator to create transformed dataset views
        4. Loading the pre-trained TabICL model

        The model itself is not trained on the data; it uses in-context learning
        at inference time. This method only prepares the data transformations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.

        y : array-like of shape (n_samples,)
            Training target labels.

        Returns
        -------
        self : TabICLClassifier
            Fitted classifier instance.

        Raises
        ------
        ValueError
            If the number of classes exceeds the model's maximum supported classes
            and hierarchical classification is disabled.
        """

        if OLD_SKLEARN:
            # Workaround for compatibility with scikit-learn prior to v1.6
            X, y = self._validate_data(X, y, dtype=None, cast_to_ndarray=False)
        else:
            X, y = self._validate_data(X, y, dtype=None, skip_check_array=True)

        check_classification_targets(y)

        if self.device is None:
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(self.device, str):
            self.device_ = torch.device(self.device)
        else:
            self.device_ = self.device

        self._load_model()
        self.model_.to(self.device_)

        self.y_encoder_ = LabelEncoder()
        y = self.y_encoder_.fit_transform(y)
        self.classes_ = self.y_encoder_.classes_
        self.n_classes_ = len(self.y_encoder_.classes_)

        if self.n_classes_ > self.model_.max_classes and not self.use_hierarchical:
            raise ValueError(
                f"The number of classes ({self.n_classes_}) exceeds the max number of classes ({self.model_.max_classes}) "
                f"natively supported by the model. Consider enabling hierarchical classification."
            )

        if self.n_classes_ > self.model_.max_classes and self.verbose:
            print(
                f"The number of classes ({self.n_classes_}) exceeds the max number of classes ({self.model_.max_classes}) "
                f"natively supported by the model. Therefore, hierarchical classification is used."
            )

        self.X_encoder_ = TransformToNumerical()
        X = self.X_encoder_.fit_transform(X)

        self.ensemble_generator_ = EnsembleGenerator(
            n_estimators=self.n_estimators,
            norm_methods=self.norm_methods or ["none", "power"],
            feat_shuffle_method=self.feat_shuffle_method,
            class_shift=self.class_shift,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
        )
        self.ensemble_generator_.fit(X, y)

        return self

    def _batch_forward(self, Xs, ys, shuffle_patterns=None):
        """Process model forward passes in batches to manage memory efficiently.

        This method handles the batched inference through the TabICL model,
        dividing the ensemble members into smaller batches to avoid out-of-memory errors.

        Parameters
        ----------
        Xs : np.ndarray
            Input features of shape (n_datasets, n_samples, n_features), where n_datasets
            is the number of ensemble members.

        ys : np.ndarray
            Training labels of shape (n_datasets, train_size), where train_size is the
            number of samples used for in-context learning.

        shuffle_patterns : List or None, default=None
            Lists of feature shuffle patterns to be applied to each ensemble member.
            If None, no feature shuffling is applied.

        Returns
        -------
        np.ndarray
            Model outputs (logits or probabilities) of shape (n_datasets, test_size, n_classes)
            where test_size = n_samples - train_size.
        """

        batch_size = self.batch_size or Xs.shape[0]
        n_batches = np.ceil(Xs.shape[0] / batch_size)
        Xs = np.array_split(Xs, n_batches)
        ys = np.array_split(ys, n_batches)
        if shuffle_patterns is None:
            shuffle_patterns = [None] * n_batches
        else:
            shuffle_patterns = np.array_split(shuffle_patterns, n_batches)

        outputs = []
        for X_batch, y_batch, pattern_batch in zip(Xs, ys, shuffle_patterns):
            X_batch = torch.from_numpy(X_batch).float().to(self.device_)
            y_batch = torch.from_numpy(y_batch).float().to(self.device_)
            if pattern_batch is not None:
                pattern_batch = pattern_batch.tolist()

            with torch.no_grad():
                out = self.model_(
                    X_batch,
                    y_batch,
                    feature_shuffles=pattern_batch,
                    return_logits=True if self.average_logits else False,
                    softmax_temperature=self.softmax_temperature,
                    device=self.device_,
                    use_amp=self.use_amp,
                    verbose=self.verbose,
                )
            outputs.append(out.cpu().numpy())

        return np.concatenate(outputs, axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for test samples.

        Applies the ensemble of TabICL models to make predictions, with each ensemble
        member providing predictions that are then averaged. The method:
        1. Transforms input data using the fitted encoders
        2. Applies the ensemble generator to create multiple views
        3. Forwards each view through the model
        4. Corrects for class shifts
        5. Averages predictions across ensemble members

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples for prediction.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Class probabilities for each test sample.
        """
        check_is_fitted(self)
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            # Reject 1D arrays to maintain sklearn compatibility
            raise ValueError(f"The provided input X is one-dimensional. Reshape your data.")

        # Preserve DataFrame structure to retain column names and types for correct feature transformation
        if OLD_SKLEARN:
            # Workaround for compatibility with scikit-learn prior to v1.6
            X = self._validate_data(X, reset=False, dtype=None, cast_to_ndarray=False)
        else:
            X = self._validate_data(X, reset=False, dtype=None, skip_check_array=True)

        X = self.X_encoder_.transform(X)

        data = self.ensemble_generator_.transform(X)
        outputs = []
        for norm_method, (Xs, ys) in data.items():
            shuffle_patterns = self.ensemble_generator_.feature_shuffle_patterns_[norm_method]
            outputs.append(self._batch_forward(Xs, ys, shuffle_patterns))
        outputs = np.concatenate(outputs, axis=0)

        # Extract class shift offsets from ensemble generator
        class_shift_offsets = []
        for offsets in self.ensemble_generator_.class_shift_offsets_.values():
            class_shift_offsets.extend(offsets)

        # Determine actual number of ensemble members
        # May be fewer than requested if dataset has quite limited features and classes
        n_estimators = len(class_shift_offsets)

        # Aggregate predictions from all ensemble members, correcting for class shifts
        avg = None
        for i, offset in enumerate(class_shift_offsets):
            out = outputs[i]
            # Reverse the class shift
            out = np.concatenate([out[..., offset:], out[..., :offset]], axis=-1)

            if avg is None:
                avg = out
            else:
                avg += out

        # Calculate ensemble average
        avg /= n_estimators

        # Convert logits to probabilities if required
        if self.average_logits:
            avg = self.softmax(avg, axis=-1, temperature=self.softmax_temperature)

        return avg

    def predict(self, X):
        """Predict class labels for test samples.

        Uses predict_proba to get class probabilities and returns the class with
        the highest probability for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples for prediction.

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted class labels for each test sample.
        """
        proba = self.predict_proba(X)
        y = np.argmax(proba, axis=1)

        return self.y_encoder_.inverse_transform(y)

    @staticmethod
    def softmax(x, axis: int = -1, temperature: float = 0.9):
        """Compute softmax values with temperature scaling using NumPy.

        Parameters
        ----------
        x : ndarray
            Input array of logits.

        axis : int, default=-1
            Axis along which to compute softmax.

        temperature : float, default=0.9
            Temperature scaling parameter.

        Returns
        -------
        ndarray
            Softmax probabilities along the specified axis, with the same shape as the input.
        """
        x = x / temperature
        # Subtract max for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        # Compute softmax
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
