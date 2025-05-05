from __future__ import annotations

import random
import itertools
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
    StandardScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
)
from sklearn.utils.validation import check_is_fitted


class TransformToNumerical(TransformerMixin, BaseEstimator):
    """Transforms non-numerical data in a DataFrame to numerical representations.

    This transformer automatically detects and converts categorical variables, text features,
    and boolean data types into numerical representations suitable for machine learning models.

    Parameters
    ----------
    verbose : bool, default=False
        Whether to print information about column classifications.

    Attributes
    ----------
    tfm_ : ColumnTransformer or FunctionTransformer
        The fitted transformer that handles the conversion of different column types.
         - If input is a DataFrame: A ColumnTransformer with OrdinalEncoder for categorical
           columns and SimpleImputer for numeric columns
         - If input is not a DataFrame: A FunctionTransformer that passes data through unchanged
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def fit(self, X, y=None):
        """Configure transformers for different column types in the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data. If a DataFrame, column types are used to determine
            appropriate transformations.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        if not hasattr(X, "columns"):  # proxy way to check whether X is a dataframe without importing pandas
            # no dataframe
            self.tfm_ = FunctionTransformer()
            return self

        cat_cols = make_column_selector(dtype_include=["string", "object", "category", "boolean"])(X)
        cat_pos = [X.columns.get_loc(col) for col in cat_cols]

        numeric_cols = make_column_selector(dtype_include="number")(X)
        numeric_pos = [X.columns.get_loc(col) for col in numeric_cols]

        self.tfm_ = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OrdinalEncoder(
                        dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
                    ),
                    cat_pos,
                ),
                ("continuous", SimpleImputer(), numeric_pos),
            ]
        )
        self.tfm_.fit(X)

        if self.verbose:
            selected_cols = []
            for name, tfm, pos in self.tfm_.transformers_:
                if tfm != "drop":
                    cols = list(X.columns[pos])
                    selected_cols.extend(cols)
                    print(f"Columns classified as {name}: {cols}")

            dropped_cols = set(X.columns).difference(set(selected_cols))
            if len(dropped_cols) >= 1:
                print(f"The following columns are not used due to their data type: {list(dropped_cols)}")

        return self

    def transform(self, X):
        """Transform features using the fitted transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed array with numerical representations.
        """
        return self.tfm_.transform(X)


class UniqueFeatureFilter(TransformerMixin, BaseEstimator):
    """Filter that removes features with only one unique value in the training set.

    Parameters
    ----------
    threshold : int, default=1
        Features with unique values less than or equal to this threshold will be removed.

    Attributes
    ----------
    n_features_in_ : int
        Number of features in the training data.

    n_features_out_ : int
        Number of features after filtering.

    features_to_keep_ : ndarray
        Boolean mask for features to keep.

    Notes
    -----
    1. Features with unique values <= threshold are removed.
    2. When the input dataset has very few samples (n_samples <= threshold), all features are preserved
       regardless of their unique value counts. This is a safety mechanism because:
       - With few samples, it's difficult to reliably assess feature variability
       - A feature might appear constant in few samples but vary in the complete dataset
    """

    def __init__(self, threshold: int = 1):
        self.threshold = threshold

    def fit(self, X, y=None):
        """Learn which features to keep based on unique value counts.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        X = self._validate_data(X)

        # If there are very few samples, keep all features
        if X.shape[0] <= self.threshold:
            self.features_to_keep_ = np.ones(self.n_features_in_, dtype=bool)
        else:
            # For each feature, check if it has more than threshold unique values
            self.features_to_keep_ = np.array(
                [len(np.unique(X[:, i])) > self.threshold for i in range(self.n_features_in_)]
            )

        self.n_features_out_ = np.sum(self.features_to_keep_)

        return self

    def transform(self, X):
        """Filter features according to unique value counts.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features_out_)
            Transformed array with selected features.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        return X[:, self.features_to_keep_]


class OutlierRemover(TransformerMixin, BaseEstimator):
    """Transformer that clips extreme values based on training data distribution.

    This implementation uses a two-stage Z-score based approach to identify and clip outliers:
    1. First stage: Identify values beyond z standard deviations and mark as missing
    2. Second stage: Recompute statistics without outliers for more robust bounds
    3. Final stage: Apply log-based clipping to maintain data distribution

    Parameters
    ----------
    threshold : float, default=4.0
        Values beyond this number of standard deviations are considered outliers.

    copy : bool, default=True
        If False, try to avoid a copy and do inplace clipping instead.

    Attributes
    ----------
    n_features_in_ : int
        Number of features in the training data.

    means_ : ndarray of shape (n_features_in_,)
        Mean values per feature after removing outliers.

    stds_ : ndarray of shape (n_features_in_,)
        Standard deviation values per feature after removing outliers.

    lower_bounds_ : ndarray of shape (n_features_in_,)
        Lower bounds for clipping.

    upper_bounds_ : ndarray of shape (n_features_in_,)
        Upper bounds for clipping.
    """

    def __init__(self, threshold: float = 4.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        """Learn clipping bounds from training data using two-stage Z-score method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        X = self._validate_data(X)

        # First stage: Identify outliers using initial statistics
        self.means_ = np.nanmean(X, axis=0)
        self.stds_ = np.nanstd(X, axis=0, ddof=1 if X.shape[0] > 1 else 0)

        # Ensure standard deviations are not zero
        self.stds_ = np.maximum(self.stds_, 1e-6)

        # Create a clean copy with outliers replaced by NaN
        X_clean = X.copy()
        lower_bounds = self.means_ - self.threshold * self.stds_
        upper_bounds = self.means_ + self.threshold * self.stds_

        # Create masks for values outside bounds
        lower_mask = X < lower_bounds[np.newaxis, :]
        upper_mask = X > upper_bounds[np.newaxis, :]
        outlier_mask = np.logical_or(lower_mask, upper_mask)

        # Set outliers to NaN
        X_clean[outlier_mask] = np.nan

        # Second stage: Recompute statistics without outliers
        self.means_ = np.nanmean(X_clean, axis=0)
        self.stds_ = np.nanstd(X_clean, axis=0, ddof=1 if X.shape[0] > 1 else 0)

        # Ensure standard deviations are not zero
        self.stds_ = np.maximum(self.stds_, 1e-6)

        # Compute final bounds
        self.lower_bounds_ = self.means_ - self.threshold * self.stds_
        self.upper_bounds_ = self.means_ + self.threshold * self.stds_

        return self

    def transform(self, X):
        """Clip values based on learned bounds with log-based adjustments.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed array with clipped values.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        X = np.maximum(-np.log1p(np.abs(X)) + self.lower_bounds_, X)
        X = np.minimum(np.log1p(np.abs(X)) + self.upper_bounds_, X)

        return X


class CustomStandardScaler(TransformerMixin, BaseEstimator):
    """Custom implementation of standard scaling with clipping.

    This scaler computes the mean and standard deviation of the training data,
    adds a small epsilon to the standard deviation to avoid division by zero,
    and clips the transformed values to a reasonable range.

    Parameters
    ----------
    clip_min : float, default=-100
        Lower bound for clipping transformed values.

    clip_max : float, default=100
        Upper bound for clipping transformed values.

    epsilon : float, default=1e-6
        Small constant added to the standard deviation to avoid division by zero.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        The mean value for each feature in the training set.

    scale_ : ndarray of shape (n_features,)
        The standard deviation for each feature in the training set with epsilon added.
    """

    def __init__(self, clip_min: float = -100, clip_max: float = 100, epsilon: float = 1e-6):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon

    def fit(self, X, y=None):
        """Compute the mean and std to be used for scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        X = self._validate_data(X)

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0) + self.epsilon

        return self

    def transform(self, X):
        """Standardize features by removing the mean and scaling to unit variance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed array after scaling and clipping.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        X_scaled = (X - self.mean_) / self.scale_
        X_clipped = np.clip(X_scaled, self.clip_min, self.clip_max)

        return X_clipped


class RTDLQuantileTransformer(BaseEstimator, TransformerMixin):
    """Quantile transformer adapted for tabular deep learning models.

    This implementation is based on research from the RTDL group and adds noise to training
    data before applying quantile transformation, improving robustness and generalization.
    It also dynamically adjusts the number of quantiles based on data size.

    Parameters
    ----------
    noise : float, default=1e-3
        Magnitude of Gaussian noise to add relative to feature standard deviations.
        Set to 0 to disable noise addition.

    n_quantiles : int, default=1000
        Maximum number of quantiles to use. The actual number used is dynamically
        determined as min(X.shape[0] // 30, n_quantiles) with a minimum of 10.

    subsample : int, default=1_000_000_000
        Maximum number of samples used to estimate the quantiles for computational efficiency.

    output_distribution : {'uniform', 'normal'}, default='normal'
        Marginal distribution for the transformed data.

    random_state : int or None, default=None
        Seed for random number generation for reproducible noise and quantile sampling.

    Attributes
    ----------
    normalizer_ : QuantileTransformer
        Fitted transformer used to transform the data.

    Notes
    -----
    Adapted from https://github.com/yandex-research/tabular-dl-tabr/blob/75105013189c76bc4f247633c2fb856bc948e579/lib/data.py#L262
    following https://github.com/dholzmueller/pytabkit/blob/949bf81e3964f65a33dd2c252c3713c239c17b2d/pytabkit/models/utils.py#L431
    """

    def __init__(
        self,
        noise: float = 1e-3,
        n_quantiles: int = 1000,
        subsample: int = 1_000_000_000,
        output_distribution: str = "normal",
        random_state: Optional[int] = None,
    ):
        self.noise = noise
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the quantile transformer to training data with optional noise addition.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data to fit the transformer.

        y : None
            Ignored. Kept for compatibility with scikit-learn's transformer interface.

        Returns
        -------
        self : object
            Returns self.
        """
        # Calculate the number of quantiles based on data size
        n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)

        # Initialize QuantileTransformer
        normalizer = QuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state,
        )

        # Add noise if required
        X_modified = self._add_noise(X) if self.noise > 0 else X

        # Fit the normalizer
        normalizer.fit(X_modified)

        # Show that it's fitted
        self.normalizer_ = normalizer

        return self

    def transform(self, X, y=None):
        """Transform data using the fitted quantile transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be transformed.

        y : None
            Ignored. Kept for compatibility with scikit-learn's transformer interface.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            The transformed data with distribution specified by output_distribution.
        """
        check_is_fitted(self)
        return self.normalizer_.transform(X)

    def _add_noise(self, X):
        """Add noise to the input data proportional to feature standard deviations.

        The noise magnitude is controlled by the 'noise' parameter and is scaled
        inversely to the standard deviation of each feature to ensure
        consistent noise levels across features of different scales.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to add noise to.

        Returns
        -------
        X_noisy : ndarray of shape (n_samples, n_features)
            The input data with added Gaussian noise.
        """
        stds = np.std(X, axis=0, keepdims=True)
        noise_std = self.noise / np.maximum(stds, self.noise)
        rng = np.random.default_rng(self.random_state)
        X_noisy = X + noise_std * rng.standard_normal(X.shape)
        return X_noisy


class PreprocessingPipeline(TransformerMixin, BaseEstimator):
    """Preprocessing pipeline for tabular data.

    This pipeline combines scaling, normalization, and outlier handling.

    Parameters
    ----------
    normalization_method : str, default='power'
        Method for normalization: 'power', 'quantile', 'quantile_tabr', 'robust', 'none'.

    outlier_threshold : float, default=4.0
        Z-score threshold for outlier detection.

    random_state : int or None, default=None
        Random seed for reproducible normalization.

    Attributes
    ----------
    n_features_in_ : int
        Number of features in the training data.

    standard_scaler_ : CustomStandardScaler
        The fitted standard scaler.

    normalizer_ : sklearn transformers
        The fitted normalization transformer (PowerTransformer, QuantileTransformer, RTDLQuantileTransformer, RobustScaler).

    outlier_remover_ : OutlierRemover
        The fitted outlier remover.

    X_transformed_ : ndarray of shape (n_samples, n_features)
        The transofrmed training input data. Save it for later use to avoid recomputation.
    """

    def __init__(
        self, normalization_method: str = "power", outlier_threshold: float = 4.0, random_state: Optional[int] = None
    ):
        self.normalization_method = normalization_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the preprocessing pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        X = self._validate_data(X)

        # 1. Apply standard scaling
        self.standard_scaler_ = CustomStandardScaler()
        X_scaled = self.standard_scaler_.fit_transform(X)

        # 2. Apply normalization
        if self.normalization_method != "none":
            if self.normalization_method == "power":
                self.normalizer_ = PowerTransformer(method="yeo-johnson", standardize=True)
            elif self.normalization_method == "quantile":
                self.normalizer_ = QuantileTransformer(output_distribution="normal", random_state=self.random_state)
            elif self.normalization_method == "quantile_rtdl":
                self.normalizer_ = Pipeline(
                    [
                        (
                            "quantile_rtdl",
                            RTDLQuantileTransformer(output_distribution="normal", random_state=self.random_state),
                        ),
                        ("std", StandardScaler()),
                    ]
                )
            elif self.normalization_method == "robust":
                self.normalizer_ = RobustScaler(unit_variance=True)
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")

            self.X_min_ = np.min(X_scaled, axis=0, keepdims=True)
            self.X_max_ = np.max(X_scaled, axis=0, keepdims=True)
            X_normalized = self.normalizer_.fit_transform(X_scaled)
        else:
            self.normalizer_ = None
            X_normalized = X_scaled

        # 3. Handle outliers
        self.outlier_remover_ = OutlierRemover(threshold=self.outlier_threshold)
        self.X_transformed_ = self.outlier_remover_.fit_transform(X_normalized)

        return self

    def transform(self, X):
        """Apply the preprocessing pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_out : ndarray
            Preprocessed data.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, copy=True)
        # Standard scaling
        X = self.standard_scaler_.transform(X)
        # Normalization
        if self.normalizer_ is not None:
            try:
                # this can fail in rare cases if there is an outlier in X that was not present in fit()
                X = self.normalizer_.transform(X)
            except ValueError:
                # clip values to train min/max
                X = np.clip(X, self.X_min_, self.X_max_)
                X = self.normalizer_.transform(X)
        # Outlier removal
        X = self.outlier_remover_.transform(X)

        return X


class FeatureShuffler:
    """Utility that generates feature permutations for ensemble creation.

    This class provides methods to create different types of feature permutations
    that can be used when creating ensemble variants of datasets.

    Parameters
    ----------
    n_features : int
        Number of features in the dataset.

    method : str, default='latin'
        Method used for feature shuffling:
        - 'none': No shuffling
        - 'random': Random permutation
        - 'latin': Latin square permutation
        - 'shift': Circular shift of features

    random_state : int or None, default=None
        Random seed for reproducible shuffling.
    """

    def __init__(self, n_features: int, method: str = "latin", random_state: Optional[int] = None):
        self.n_features = n_features
        self.method = method
        self.random_state = random_state

    def shuffle(self, n_estimators: int) -> List[np.ndarray]:
        """Generate feature shuffling patterns for ensemble diversity.

        Creates permutations of feature indices according to the specified method,
        which can be used to reorder features when creating ensemble variants.

        Parameters
        ----------
        n_estimators : int
            Number of feature permutations to generate.
            - For 'none' method: Always returns a single pattern with no shuffling
            - For 'shift' method: Generates all possible circular shifts of features
            - For 'latin' method: Generates Latin square permutations
            - For 'random' method: For small feature sets (≤5), samples from all possible
              permutations; otherwise generates random permutations

        Returns
        -------
        list of ndarray
            List of feature permutation arrays, where each array contains
            indices that can be used to shuffle features.
        """

        self.rng_ = random.Random(self.random_state)
        feature_indices = list(range(self.n_features))

        # No shuffling
        if self.method == "none" or n_estimators == 1:
            shuffle_patterns = [feature_indices]

        # Generate permutations based on method
        if self.method == "shift":
            # All possible circular shifts
            shuffle_patterns = [feature_indices[-i:] + feature_indices[:-i] for i in range(self.n_features)]
        elif self.method == "random":
            # Random permutations
            if self.n_features <= 5:
                all_perms = [list(perm) for perm in itertools.permutations(feature_indices)]
                shuffle_patterns = self.rng_.sample(all_perms, min(n_estimators, len(all_perms)))
            else:
                shuffle_patterns = [self.rng_.sample(feature_indices, self.n_features) for _ in range(n_estimators)]
        elif self.method == "latin":
            # Latin square permutations
            shuffle_patterns = self._latin_squares()
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'shift', 'random', 'latin', or 'none'.")

        return shuffle_patterns

    def _latin_squares(self):
        """Generate Latin squares for feature shuffling.

        Returns
        -------
        list
            List of feature permutations forming a Latin square.
        """

        def _shuffle_transpose_shuffle(matrix):
            square = deepcopy(matrix)
            self.rng_.shuffle(square)
            trans = list(zip(*square))
            self.rng_.shuffle(trans)
            return trans

        def _rls(symbols):
            n = len(symbols)
            if n == 1:
                return [symbols]
            else:
                sym = self.rng_.choice(symbols)
                symbols.remove(sym)
                square = _rls(symbols)
                square.append(square[0].copy())
                for i in range(n):
                    square[i].insert(i, sym)
                return square

        symbols = list(range(self.n_features))
        square = _rls(symbols)
        feature_shuffles = _shuffle_transpose_shuffle(square)

        return [list(shuffle) for shuffle in feature_shuffles]


class EnsembleGenerator(TransformerMixin, BaseEstimator):
    """Generate ensemble variants for robust tabular prediction with TabICL.

    This class creates diverse data variants through:
    1. Applying different normalization techniques
    2. Permuting feature orders to exploit position-invariance in transformer architectures
    3. Shifting class labels to prevent overfitting to specific class index patterns

    Parameters
    ----------
    n_estimators : int
        Number of ensemble variants to generate.

    norm_methods : str or list[str] or None, default=None
        Normalization methods to apply:
        - 'none': No normalization
        - 'power': Yeo-Johnson power transform
        - 'quantile': Transform feature distribution to approximately Gaussian, using the empirical quantiles.
        - 'quantile_rtdl': Version of the quantile transform used typically in papers by the RTDL group.
        - 'robust': Scale using median and quantiles
    If set to None, ['none', 'power'] will be applied.

    feat_shuffle_method : str, default='latin'
        Feature permutation strategy:
        - 'none': No shuffling and preserve original feature order
        - 'shift': Circular shifting
        - 'random': Random permutation
        - 'latin': Latin square patterns

    class_shift : bool, default=True
        Whether to apply cyclic shifts to class labels.

    outlier_threshold : float, default=4.0
        Z-score threshold for outlier detection and clipping.

    random_state : int or None, default=None
        Seed for reproducible ensemble generation.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features after filtering.

    n_classes_ : int
        Number of unique target classes.

    unique_filter_ : UniqueFeatureFilter
        Filter that removes features with only one unique value.

    preprocessors_ : dict
        Maps normalization methods to fitted preprocessing pipelines.

    ensemble_configs_ : OrderedDict
        Generated ensemble configurations, organized by normalization method.
        Keys are normalization methods and values are lists of (feature_pattern, shift_offset) tuples.

    feature_shuffle_patterns_ : OrderedDict
        Maps normalization methods to lists of feature index permutations.

    class_shift_offsets_ : OrderedDict
        Maps normalization methods to lists of class shift offsets.

    X_ : ndarray
        Training feature data after filtering.

    y_ : ndarray
        Training target values.
    """

    def __init__(
        self,
        n_estimators: int,
        norm_methods: str | List[str] | None = None,
        feat_shuffle_method: str = "latin",
        class_shift: bool = True,
        outlier_threshold: float = 4.0,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shift = class_shift
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y):
        """Create ensemble configurations and fit preprocessing pipelines.

        This method:
        1. Removes features with only one unique value
        2. Generates diverse ensemble configurations
        3. Fits preprocessing pipelines for each normalization method

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature data.

        y : array-like of shape (n_samples,)
            Training target values.

        Returns
        -------
        self : object
            Fitted generator.
        """
        self._validate_data(X, y)

        if self.norm_methods is None:
            self.norm_methods_ = ["none", "power"]
        else:
            if isinstance(self.norm_methods, str):
                self.norm_methods_ = [self.norm_methods]
            else:
                self.norm_methods_ = self.norm_methods

        # Filter unique features
        self.unique_filter_ = UniqueFeatureFilter()
        X = self.unique_filter_.fit_transform(X)

        self.X_ = X
        self.y_ = y

        # override n_features_in_ to account for unique feature filtering
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))

        self.rng_ = random.Random(self.random_state)
        self.ensemble_configs_, self.feature_shuffle_patterns_, self.class_shift_offsets_ = self._generate_ensemble()

        self.preprocessors_ = {}
        for norm_method in self.ensemble_configs_:
            if norm_method not in self.preprocessors_:
                preprocessor = PreprocessingPipeline(
                    normalization_method=norm_method,
                    outlier_threshold=self.outlier_threshold,
                    random_state=self.random_state,
                )
                preprocessor.fit(X)
                self.preprocessors_[norm_method] = preprocessor

        return self

    def _generate_ensemble(self):
        """Create diverse ensemble configurations grouped by normalization method.

        Returns
        -------
        tuple
            Three-element tuple containing:
            - ensemble_configs: OrderedDict mapping normalization methods to shuffle-shift configs
            - shuffle_patterns: OrderedDict mapping normalization methods to lists of feature shuffle patterns
            - shift_offsets: OrderedDict mapping normalization methods to lists of class shift offsets
        """

        shuffler = FeatureShuffler(self.n_features_in_, self.feat_shuffle_method, self.random_state)
        shuffle_patterns = shuffler.shuffle(self.n_estimators)

        if self.class_shift and self.n_estimators > 1:
            shift_offsets = self.rng_.sample(range(self.n_classes_), self.n_classes_)
        else:
            shift_offsets = [0]

        shuffle_shift_configs = list(itertools.product(shuffle_patterns, shift_offsets))
        self.rng_.shuffle(shuffle_shift_configs)

        shuffle_shift_norm_configs = list(itertools.product(shuffle_shift_configs, self.norm_methods_))
        shuffle_shift_norm_configs = shuffle_shift_norm_configs[: self.n_estimators]

        # Reorganize configs so that those with the same normalization method are grouped together
        used_methods = list(set([config[1] for config in shuffle_shift_norm_configs]))

        ensemble_configs = OrderedDict()
        shuffle_patterns = OrderedDict()
        shift_offsets = OrderedDict()

        for method in used_methods:
            shuffle_shift_configs = [config[0] for config in shuffle_shift_norm_configs if config[1] == method]
            shuffle_patterns[method] = [config[0] for config in shuffle_shift_configs]
            shift_offsets[method] = [config[1] for config in shuffle_shift_configs]
            ensemble_configs[method] = shuffle_shift_configs

        return ensemble_configs, shuffle_patterns, shift_offsets

    def transform(self, X):
        """Combines training and test data to create different in-context learning prompts.

        For each normalization method:
        1. Preprocesses data once using the fitted pipeline for that method
        2. Applies each feature permutation and class shift within that method's group

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test input data to be combined with training data.

        Returns
        -------
        OrderedDict
            Dictionary mapping normalization methods to tuples of (X_ensemble, y_ensemble) where:
            - X_ensemble: array of shape (n_variants_for_method, n_samples_combined, n_features)
              Contains the preprocessed and feature-permuted combined data
            - y_ensemble: array of shape (n_variants_for_method, n_samples_train)
              Contains the class-shifted training labels
        """

        check_is_fitted(self, ["ensemble_configs_"])

        # Unique feature filtering
        X = self.unique_filter_.transform(X)
        y = self.y_

        data = OrderedDict()
        for norm_method, shuffle_shift_configs in self.ensemble_configs_.items():
            # Apply preprocessing
            preprocessor = self.preprocessors_[norm_method]
            X_variant = np.concatenate(
                [preprocessor.X_transformed_, preprocessor.transform(X)],
                axis=0,
            )
            # Shuffle features and shift class labels
            X_ensemble = []
            y_ensemble = []
            for shuffle_pattern, shift_offset in shuffle_shift_configs:
                X_ensemble.append(X_variant[:, shuffle_pattern])
                y_ensemble.append((y + shift_offset) % self.n_classes_)
            data[norm_method] = (np.stack(X_ensemble, axis=0), np.stack(y_ensemble, axis=0))

        return data
