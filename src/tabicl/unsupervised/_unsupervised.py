from __future__ import annotations

import numpy as np
import torch

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from tabicl.sklearn.preprocessing import Shuffler
from tabicl.model.quantile_dist import QuantileDistribution
from tabicl import TabICLClassifier, TabICLRegressor


class TabICLUnsupervised(BaseEstimator):
    """Unsupervised learning with TabICL.

    Supports three unsupervised tasks:
    - **Imputation**: Fill NaN values by conditioning on observed features.
    - **Outlier detection**: Score samples by their estimated joint density.
    - **Synthetic data generation**: Autoregressive sampling from the learned density.

    Estimates the joint density by decomposing it via the chain rule of
    probability:

        ``P(X) = P(X_1) * P(X_2 | X_1) * ... * P(X_d | X_1, ..., X_{d-1})``

    Each conditional ``P(X_k | X_{<k})`` is predicted by a TabICL classifier
    (for categorical features) or regressor (for numerical features).  Multiple
    random feature orderings (permutations) are averaged to reduce the
    dependence on any single ordering.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ensemble estimators per conditional prediction.

    categorical_features : list[int] or None, default=None
        Indices of categorical features. If None, auto-detected from data
        based on ``max_categories``.

    max_categories : int, default=10
        Maximum unique values for auto-detection of categorical features.

    batch_size : int or None, default=8
        Batch size for inner estimator inference.

    random_state : int or None, default=42
        Random seed for reproducibility.

    device : str or None, default=None
        Device for inference. None auto-selects CUDA or CPU.

    estimator_params : dict or None, default=None
        Additional keyword arguments forwarded to the inner
        ``TabICLClassifier`` and ``TabICLRegressor`` (e.g.
        ``norm_methods``, ``outlier_threshold``).

    Attributes
    ----------
    X_ : np.ndarray of shape (n_samples, n_features)
        Copy of the training data, used as conditioning context for all
        predictions.

    n_features_in_ : int
        Number of features seen during ``fit()``.

    categorical_features_ : list[int]
        Indices of categorical features (user-supplied or auto-detected).

    numerical_features_ : list[int]
        Indices of numerical features (complement of ``categorical_features_``).

    categories_ : dict[int, np.ndarray]
        Mapping from categorical feature index to its sorted unique values.

    _clf_model : torch.nn.Module or None
        Shared classifier model weights (loaded once in ``fit()``).

    _reg_model : torch.nn.Module or None
        Shared regressor model weights (loaded once in ``fit()``).

    Examples
    --------
    >>> import numpy as np
    >>> from tabicl import TabICLUnsupervised
    >>> X = np.random.standard_normal((50, 3))
    >>> model = TabICLUnsupervised(n_estimators=4, device="cpu")
    >>> model.fit(X)
    >>> scores = model.score_samples(X, n_permutations=4)
    >>> X_synth = model.generate(n_samples=10)
    """

    # Minimum number of training samples required
    # to fit a conditional model for a feature
    _MIN_SAMPLES_PER_CONDITIONAL: int = 5

    def __init__(
        self,
        n_estimators: int = 8,
        categorical_features: list[int] | None = None,
        max_categories: int = 10,
        batch_size: int | None = 8,
        random_state: int | None = 42,
        device: str | None = None,
        estimator_params: dict | None = None,
    ):
        self.n_estimators = n_estimators
        self.categorical_features = categorical_features
        self.max_categories = max_categories
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self.estimator_params = estimator_params or {}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.non_deterministic = True
        return tags

    @property
    def _estimator_kwargs(self) -> dict:
        """Keyword arguments shared by all inner TabICL estimators."""
        return {
            **self.estimator_params,
            "n_estimators": self.n_estimators,
            "batch_size": self.batch_size,
            "kv_cache": False,
            "random_state": self.random_state,
            "device": self.device,
            "verbose": False,
        }

    def fit(self, X: np.ndarray, y=None) -> TabICLUnsupervised:
        """Store training data, detect categorical features, and load shared models.

        The raw training data is stored in ``self.X_`` and used as conditioning
        context for all downstream predictions.  Shared model weights are loaded
        once here and injected into per-column estimators to avoid redundant
        ``torch.load()`` calls.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. May contain NaN values.

        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self : TabICLUnsupervised
        """

        X = np.asarray(X, dtype=np.float32)
        self.X_ = X.copy()
        self.n_features_in_ = X.shape[1]

        # Detect categorical features
        if self.categorical_features is None:
            self.categorical_features_ = self._infer_categorical_features(X)
        else:
            self.categorical_features_ = list(self.categorical_features)

        # Store unique categories for each categorical feature
        self.categories_ = {}
        for j in self.categorical_features_:
            col = X[:, j]
            valid = col[~np.isnan(col)]
            self.categories_[j] = np.unique(valid).astype(int)

        self.numerical_features_ = [j for j in range(self.n_features_in_) if j not in self.categorical_features_]

        # Load shared models once to avoid repeated torch.load calls.
        need_clf = len(self.categorical_features_) > 0
        need_reg = len(self.numerical_features_) > 0

        self._clf_model = self._load_shared_model(TabICLClassifier) if need_clf else None
        self._reg_model = self._load_shared_model(TabICLRegressor) if need_reg else None

        return self

    def score_samples(self, X: np.ndarray, n_permutations: int = 4) -> np.ndarray:
        """Compute outlier scores via chain-rule log-probability.

        Estimates the joint density by factoring it as a product of conditionals:

            ``score(x) = exp((1/K) Sigma_k log P(x_{pi(k)} | x_{pi(<k)}))``

        where ``pi`` is a random permutation and averaging is over ``K``
        permutations. Higher scores indicate more normal data points;
        lower scores indicate outliers.

        For numerical features, ``P(x_k | ...)`` is the density from the
        quantile-based distribution (log_prob on the learned ICDF). For
        categorical features, it is the predicted class probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.

        n_permutations : int, default=4
            Number of random feature orderings to average over.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Outlier scores. Higher = more normal, lower = more outlier.
        """
        check_is_fitted(self)

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            raise ValueError("Expected 2D array, got 1D array.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                f"was fitted with {self.n_features_in_} features."
            )
        rng = np.random.default_rng(self.random_state)
        permutations = Shuffler(self.n_features_in_, random_state=self.random_state).shuffle(n_permutations)

        log_densities = []
        for perm in permutations:
            log_densities.append(self._compute_log_density(X, perm, rng))

        return np.exp(np.mean(log_densities, axis=0))

    def impute(self, X: np.ndarray, temperature: float = 1e-8, n_iterations: int = 2) -> np.ndarray:
        """Fill NaN values by conditioning on all other features.

        For numerical features, predictions are drawn from the quantile-based
        ICDF: :math:`x \\sim F^{-1}(u)` where :math:`u` is temperature-scaled
        around 0.5.  For categorical features, classes are sampled from the
        temperature-scaled predictive distribution.

        Multiple iterations (``n_iterations > 1``) refine imputed values
        iteratively: each pass conditions on the current best estimates of all
        other columns.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data with NaN values to impute.

        temperature : float, default=1e-8
            Temperature for sampling. Near 0 gives deterministic (median/mode),
            1.0 gives full distribution sampling.

        n_iterations : int, default=2
            Number of iterative refinement passes. With ``n_iterations=1`` the
            method performs a single left-to-right sweep; higher values cycle
            through the missing columns repeatedly, each time conditioning on
            the most recently imputed values of other columns.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Data with NaN values filled.
        """
        check_is_fitted(self)

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            raise ValueError("Expected 2D array, got 1D array.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                f"was fitted with {self.n_features_in_} features."
            )

        X_imp = X.copy()

        rng = np.random.default_rng(self.random_state)

        # Find columns with NaN values, sorted by missingness rate ascending.
        # Columns with fewer missing values get imputed first so that later
        # columns benefit from better-conditioned neighbours.
        columns_with_nan = sorted(
            (j for j in range(self.n_features_in_) if np.any(np.isnan(X_imp[:, j]))),
            key=lambda j: np.isnan(X_imp[:, j]).mean(),
        )

        if not columns_with_nan:
            return X_imp

        # Store original NaN masks before warm-start so the main loop always
        # knows which cells to fill, regardless of how many iterations run.
        missing_masks = {j: np.isnan(X_imp[:, j]) for j in columns_with_nan}

        # Warm-start: fill NaN cells with training median (numerical) or mode
        # (categorical) so that conditioning features are never NaN when passed
        # to the inner estimators.  These coarse initial values are overwritten
        # column by column in the iterative loop below.
        for j in columns_with_nan:
            train_col = self.X_[:, j]
            if j in self.categorical_features_:
                vals, counts = np.unique(train_col[~np.isnan(train_col)], return_counts=True)
                fill = vals[np.argmax(counts)] if len(vals) > 0 else 0.0
            else:
                fill = float(np.nanmedian(train_col))
            X_imp[missing_masks[j], j] = fill

        for _ in range(n_iterations):
            for col_idx in columns_with_nan:
                missing_mask = missing_masks[col_idx]

                # All other features are conditioning features
                other_features = [j for j in range(self.n_features_in_) if j != col_idx]

                # Rows where col_idx is not NaN in the training data
                train_mask = ~np.isnan(self.X_[:, col_idx])
                if train_mask.sum() < self._MIN_SAMPLES_PER_CONDITIONAL:
                    # Not enough data to train a conditional model for this feature.
                    # Fall back to the mean of observed training values.
                    X_imp[missing_mask, col_idx] = np.nanmean(self.X_[:, col_idx])
                    continue

                X_imp[missing_mask, col_idx] = self._sample_column(
                    col_idx=col_idx,
                    cond_features=other_features,
                    X_test=X_imp[missing_mask],
                    train_mask=train_mask,
                    temperature=temperature,
                    rng=rng,
                )

        return X_imp

    def generate(self, n_samples: int = 100, temperature: float = 1.0) -> np.ndarray:
        """Generate synthetic data by autoregressive sampling.

        Features are sampled in the original feature order: each feature
        ``x_k`` is sampled from ``P(x_k | x_{<k})``.

        Parameters
        ----------
        n_samples : int, default=100
            Number of synthetic samples to generate.

        temperature : float, default=1.0
            Temperature for sampling. Higher values produce more diverse data.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Generated synthetic data.
        """
        check_is_fitted(self)

        rng = np.random.default_rng(self.random_state)
        X_synth = np.empty((n_samples, self.n_features_in_), dtype=np.float32)

        for col_idx in range(self.n_features_in_):
            cond_features = list(range(col_idx))
            train_mask = ~np.isnan(self.X_[:, col_idx])

            if train_mask.sum() < self._MIN_SAMPLES_PER_CONDITIONAL:
                # Not enough data to train a conditional model for this feature.
                # Fall back to random draws from observed training values.
                valid = self.X_[~np.isnan(self.X_[:, col_idx]), col_idx]
                X_synth[:, col_idx] = rng.choice(valid, size=n_samples, replace=True) if len(valid) > 0 else 0.0
                continue

            X_synth[:, col_idx] = self._sample_column(
                col_idx=col_idx,
                cond_features=cond_features,
                X_test=X_synth,
                train_mask=train_mask,
                temperature=temperature,
                rng=rng,
            )

        return X_synth

    def _load_shared_model(self, estimator_cls):
        """Instantiate a temporary estimator and extract its loaded model weights.

        Parameters
        ----------
        estimator_cls : type
            ``TabICLClassifier`` or ``TabICLRegressor``.

        Returns
        -------
        torch.nn.Module
            The loaded model, transferred to the resolved device.
        """
        est = estimator_cls(**self._estimator_kwargs)
        est._resolve_device()
        est._load_model()
        est.model_.to(est.device_)

        return est.model_

    def _fit_conditional_estimator(
        self,
        col_idx: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[TabICLClassifier | TabICLRegressor, bool]:
        """Create a fitted estimator for predicting feature ``col_idx``.

        Parameters
        ----------
        col_idx : int
            Index of the target feature being predicted.

        X_train : np.ndarray of shape (n_train, n_cond_features)
            Conditioning features for training samples.

        y_train : np.ndarray of shape (n_train,)
            Target values for training samples.

        Returns
        -------
        est : TabICLClassifier or TabICLRegressor
            Fitted estimator.

        is_categorical : bool
            Whether the target feature is categorical.
        """
        is_categorical = col_idx in self.categorical_features_
        kwargs = self._estimator_kwargs

        if is_categorical:
            est = TabICLClassifier(**kwargs)
            est.model_ = self._clf_model
            y_train = y_train.astype(int)
        else:
            est = TabICLRegressor(**kwargs)
            est.model_ = self._reg_model
            y_train = y_train.astype(np.float32)

        # Skip model loading: the shared model weights are
        # already set on est.model_ above. This prevents
        # redundant torch.load() calls.
        est._load_model = lambda: None
        est.fit(X_train, y_train)

        return est, is_categorical

    def _infer_categorical_features(self, X: np.ndarray) -> list[int]:
        """Return indices of integer-valued columns with at most ``self.max_categories`` distinct values."""

        def is_categorical(col: np.ndarray) -> bool:
            valid = col[~np.isnan(col)]
            return valid.size > 0 and not np.any(valid % 1.0) and np.unique(valid).size <= self.max_categories

        return [j for j in range(X.shape[1]) if is_categorical(X[:, j])]

    def _prepare_conditional_data(
        self,
        tgt_idx: int,
        cond_features: list[int],
        train_mask: np.ndarray,
        X_test: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assemble conditioning data for a single conditional prediction.

        Handles the empty-conditioning case (random noise) and filters NaN
        values from the target column.

        Parameters
        ----------
        tgt_idx : int
            Index of the target feature.

        cond_features : list[int]
            Indices of conditioning features.

        train_mask : np.ndarray of shape (n_train,)
            Boolean mask for valid training rows (non-NaN in target).

        X_test : np.ndarray of shape (n_test, n_features)
            Test data to extract conditioning columns from.

        rng : np.random.Generator
            Random number generator (used when ``cond_features`` is empty).

        Returns
        -------
        tuple of (X_train_cond, y_train_cond, X_test_cond)
             X_train_cond : np.ndarray of shape (n_train, n_cond_features)
             y_train_cond : np.ndarray of shape (n_train,)
             X_test_cond : np.ndarray of shape (n_test, n_cond_features)
        """
        n_test = X_test.shape[0]

        if len(cond_features) == 0:
            # No conditioning features: use random noise as a dummy input
            X_train_cond = rng.standard_normal((train_mask.sum(), 1)).astype(np.float32)
            X_test_cond = rng.standard_normal((n_test, 1)).astype(np.float32)
        else:
            X_train_cond = self.X_[train_mask][:, cond_features]
            X_test_cond = X_test[:, cond_features]

        y_train_cond = self.X_[train_mask, tgt_idx]

        return X_train_cond, y_train_cond, X_test_cond

    def _compute_log_density(self, X: np.ndarray, perm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Accumulate per-feature log-probability contributions for one permutation.

        Iterates through features in the order given by ``perm``, conditioning
        each feature on all preceding features in that ordering.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to score.

        perm : np.ndarray of shape (n_features,)
            Feature ordering for this permutation.

        rng : np.random.Generator
            Random number generator passed to ``_prepare_conditional_data``.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Sum of per-feature log-probabilities.
        """
        log_p = np.zeros(X.shape[0])

        for i, col_idx in enumerate(perm):
            train_mask = ~np.isnan(self.X_[:, col_idx])
            if train_mask.sum() < self._MIN_SAMPLES_PER_CONDITIONAL:
                continue

            cond_features = list(perm[:i])
            X_train_cond, y_train_cond, X_test_cond = self._prepare_conditional_data(
                tgt_idx=col_idx,
                cond_features=cond_features,
                train_mask=train_mask,
                X_test=X,
                rng=rng,
            )
            est, is_categorical = self._fit_conditional_estimator(col_idx, X_train_cond, y_train_cond)

            if is_categorical:
                log_p += self._log_prob_categorical(est, X_test_cond, X[:, col_idx])
            else:
                log_p += self._log_prob_numerical(est, X_test_cond, X[:, col_idx])

        return log_p

    def _log_prob_categorical(
        self,
        est: TabICLClassifier,
        X_test: np.ndarray,
        observed_col: np.ndarray,
    ) -> np.ndarray:
        """Compute per-sample log-probability for a categorical conditional.

        Parameters
        ----------
        est : TabICLClassifier
            Fitted classifier for this conditional.

        X_test : np.ndarray of shape (n_samples, n_cond_features)

        observed_col : np.ndarray of shape (n_samples,)
            Observed values of the target column.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Log-probability contribution (0.0 for NaN observations).
        """
        n_samples = observed_col.shape[0]
        pred = est.predict_proba(X_test)  # (n_samples, n_classes)

        # Handle missing values by assigning them a dummy class
        # and masking out their contributions later
        is_missing = np.isnan(observed_col)
        observed_col = np.where(is_missing, 0, observed_col).astype(int)

        # For each sample, find the column in proba for the observed class.
        # est.classes_ is sorted, so searchsorted gives the position.
        class_idx = np.clip(np.searchsorted(est.classes_, observed_col), 0, len(est.classes_) - 1)
        in_classes = est.classes_[class_idx] == observed_col

        proba = np.full(n_samples, 1e-10)
        valid = np.where(~is_missing & in_classes)[0]
        proba[valid] = pred[valid, class_idx[valid]]

        # Missing observations get 0; unseen classes get log(1e-10)
        return np.where(is_missing, 0.0, np.log(np.maximum(proba, 1e-10)))

    def _log_prob_numerical(
        self,
        est: TabICLRegressor,
        X_test: np.ndarray,
        observed_col: np.ndarray,
    ) -> np.ndarray:
        """Compute per-sample log-probability for a numerical conditional.

        Parameters
        ----------
        est : TabICLRegressor
            Fitted regressor for this conditional.

        X_test : np.ndarray of shape (n_samples, n_cond_features)

        observed_col : np.ndarray of shape (n_samples,)
            Observed values of the target column (original scale).

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Log-probability contribution from the quantile distribution.
        """
        # predict returns ensemble-averaged quantiles of shape (n_samples, n_q) in the original y-space
        raw_q = torch.from_numpy(est.predict(X_test, output_type="raw_quantiles")).to(est.device_)
        dist = est.model_.quantile_dist(raw_q)
        observed_col = torch.from_numpy(observed_col).to(dtype=raw_q.dtype, device=raw_q.device)

        # unsqueeze to (n_samples, 1) so log_prob returns (n_samples, 1) not (n_samples, n_samples)
        lp = dist.log_prob(observed_col.unsqueeze(-1)).squeeze(-1)  # (n_samples,)

        return lp.cpu().numpy()

    def _sample_column(
        self,
        col_idx: int,
        cond_features: list[int],
        X_test: np.ndarray,
        train_mask: np.ndarray,
        temperature: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Build conditioning data, fit an estimator, and sample values for one column.

        Parameters
        ----------
        col_idx : int
            Index of the target feature.

        cond_features : list[int]
            Indices of conditioning features.

        X_test : np.ndarray of shape (n_test, n_features)
            Test rows to predict for.

        train_mask : np.ndarray of shape (n_train,)
            Boolean mask selecting valid training rows.

        temperature : float
            Sampling temperature.

        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray of shape (n_test,)
            Sampled values.
        """
        X_train_cond, y_train_cond, X_test_cond = self._prepare_conditional_data(
            tgt_idx=col_idx,
            cond_features=cond_features,
            train_mask=train_mask,
            X_test=X_test,
            rng=rng,
        )
        est, is_categorical = self._fit_conditional_estimator(col_idx, X_train_cond, y_train_cond)

        if is_categorical:
            sampled_col = self._sample_categorical(est.predict_proba(X_test_cond), est.classes_, temperature, rng)
        else:
            raw_q = torch.from_numpy(est.predict(X_test_cond, output_type="raw_quantiles")).to(est.device_)
            sampled_col = self._sample_numerical(est.model_.quantile_dist(raw_q), temperature, rng)

        return sampled_col

    @staticmethod
    def _sample_categorical(
        proba: np.ndarray,
        classes: np.ndarray,
        temperature: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample from categorical distribution with temperature scaling.

        Applies temperature scaling to log-probabilities:
        :math:`\\log p'_k = \\log p_k / \\tau`, then samples via inverse
        transform sampling.

        Parameters
        ----------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities.

        classes : np.ndarray of shape (n_classes,)
            Class labels.

        temperature : float
            Temperature. Near 0 returns the mode; 1.0 gives unmodified sampling.

        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Sampled class labels.
        """
        if temperature <= 1e-8:
            return classes[np.argmax(proba, axis=1)]

        if temperature != 1.0:
            log_p = np.log(np.clip(proba, 1e-10, None)) / temperature
            log_p -= log_p.max(axis=1, keepdims=True)
            proba = np.exp(log_p)
            proba /= proba.sum(axis=1, keepdims=True)

        # Sample from the categorical distribution using inverse transform sampling
        cumprob = np.cumsum(proba, axis=1)
        u = rng.random(proba.shape[0])[:, np.newaxis]
        indices = (u >= cumprob).sum(axis=1)
        indices = np.clip(indices, 0, len(classes) - 1)

        return classes[indices]

    @staticmethod
    def _sample_numerical(
        dist: QuantileDistribution,
        temperature: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample from the quantile distribution via ICDF inversion.

        Computes :math:`x = F^{-1}(0.5 + (u - 0.5) \\cdot \\tau)` where
        :math:`u \\sim \\mathrm{Uniform}(0, 1)` and :math:`\\tau` is the
        temperature.

        Parameters
        ----------
        dist : QuantileDistribution
            Fitted distribution.

        temperature : float
            Temperature. Near 0 returns the median; higher values sample
            further into the tails.

        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Sampled values in the original y-space.
        """
        n_samples = dist.quantiles.shape[0]
        device, dtype = dist.quantiles.device, dist.quantiles.dtype

        u = torch.tensor(rng.random(n_samples), device=device, dtype=dtype)
        u = 0.5 + (u - 0.5) * temperature
        u = torch.clamp(u, 1e-6, 1.0 - 1e-6)

        # unsqueeze to (n_samples, 1) so log_prob returns (n_samples, 1) not (n_samples, n_samples)
        return dist.icdf(u.unsqueeze(-1)).squeeze(-1).cpu().numpy()
