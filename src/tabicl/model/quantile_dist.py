from typing import List, Tuple, Literal, Optional
import warnings

import numpy as np
import torch
from torch import nn, Tensor
from torch.distributions import Distribution

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class QuantileDistributionConfig:
    """Configuration constants for QuantileDistribution.

    Numerical Stability Parameters
    ------------------------------
    TOL : float, default=1e-6
        General numerical tolerance for avoiding division by zero.

    MIN_SLOPE : float, default=1e-6
        Minimum allowed slope :math:`dQ/d\\alpha` in spline regions.
        Prevents infinite PDF values. Smaller values allow sharper peaks.

    MAX_SLOPE : float, default=1e6
        Maximum allowed slope :math:`dQ/d\\alpha` in spline regions.
        Prevents numerical overflow. Larger values allow heavier tails.

    Tail Parameters
    ---------------
    MIN_BETA : float, default=0.01
        Minimum tail scale parameter :math:`\\beta`. Prevents degenerate
        (zero-width) tails.

    MAX_BETA : float, default=100.0
        Maximum tail scale parameter :math:`\\beta`. Prevents extremely heavy
        tails that cause overflow.

    MIN_ETA : float, default=-0.49
        Minimum GPD shape parameter :math:`\\eta`.
        :math:`\\eta > 0` ensures heavy tails; :math:`\\eta = 0` gives
        exponential; :math:`\\eta < 0` gives light tails.
        Negative values allow modeling bounded/light-tailed distributions.

    MAX_ETA : float, default=0.49
        Maximum GPD shape parameter :math:`\\eta`.
        Must be :math:`< 0.5` for finite variance, :math:`< 1` for finite mean.

    ETA_TOLERANCE : float, default=0.01
        Threshold for treating :math:`\\eta \\approx 0` (use exponential
        approximation).

    Computational Limits
    --------------------
    MAX_LOG_RATIO : float, default=15.0
        Maximum :math:`\\log(\\text{ratio})` in tail computations. Prevents
        :math:`\\exp()` overflow. :math:`\\exp(15) \\approx 3.3 \\times 10^6`.

    MAX_EXPONENT : float, default=15.0
        Maximum exponent before :math:`\\exp()` call. Prevents
        overflow/underflow.

    MAX_CRPS : float, default=1e4
        Maximum CRPS value (for clamping extreme values).

    Tail Inference Parameters
    -------------------------
    TAIL_QUANTILES_FOR_ESTIMATION : int, default=20
        Number of quantiles in each tail used for parameter estimation.
    """

    TOL: float = 1e-6
    MIN_SLOPE: float = 1e-6
    MAX_SLOPE: float = 1e6
    MIN_BETA: float = 0.01
    MAX_BETA: float = 100.0
    MIN_ETA: float = -0.49
    MAX_ETA: float = 0.49
    ETA_TOLERANCE: float = 0.01
    MAX_LOG_RATIO: float = 15.0
    MAX_EXPONENT: float = 15.0
    MAX_CRPS: float = 1e4
    TAIL_QUANTILES_FOR_ESTIMATION: int = 20


def isotonic_regression_pava(y: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    """Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.

    Solves:

    .. math::

        \\arg\\min_x \\sum_i w_i (y_i - x_i)^2 \\quad \\text{subject to} \\quad x_i \\le x_{i+1}

    Parameters
    ----------
    y : Tensor
        Input values to make monotonically non-decreasing.
        Shape: ``(*batch_shape, n)``.

    weights : Tensor, optional
        Case weights. Must be positive.
        Shape: ``(*batch_shape, n)`` or None for uniform weights.

    Returns
    -------
    Tensor
        Isotonic (monotonically non-decreasing) values.
        Shape: ``(*batch_shape, n)``.
    """
    if not NUMBA_AVAILABLE:
        warnings.warn(
            "Numba not available. Install with 'pip install numba' for fast isotonic regression."
            " Use sorting as fallback."
        )
        return torch.sort(y, dim=-1).values

    batch_shape = y.shape[:-1]
    n = y.shape[-1]
    device = y.device
    dtype = y.dtype

    y_np = y.reshape(-1, n).detach().cpu().numpy().astype(np.float64)
    if weights is None:
        result_np = _pava_batch_numba_no_weights(y_np)
    else:
        w_np = weights.reshape(-1, n).detach().cpu().numpy().astype(np.float64)
        result_np = _pava_batch_numba(y_np, w_np)

    result = torch.from_numpy(result_np).to(dtype=dtype, device=device)
    return result.reshape(*batch_shape, n)


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _pava_single_numba_no_weights(y: np.ndarray) -> np.ndarray:
        """Single-sequence PAVA without weights (optimized path)."""
        n = len(y)
        if n <= 1:
            return y.copy()

        block_values = np.empty(n, dtype=y.dtype)
        block_counts = np.empty(n, dtype=np.int64)
        block_ends = np.empty(n, dtype=np.int64)
        num_blocks = 0

        for i in range(n):
            block_values[num_blocks] = y[i]
            block_counts[num_blocks] = 1
            block_ends[num_blocks] = i
            num_blocks += 1

            # Merge while violation exists
            while num_blocks > 1 and block_values[num_blocks - 2] > block_values[num_blocks - 1]:
                v1, c1 = block_values[num_blocks - 2], block_counts[num_blocks - 2]
                v2, c2 = block_values[num_blocks - 1], block_counts[num_blocks - 1]
                end2 = block_ends[num_blocks - 1]

                # Uniform weights: simple average
                merged_count = c1 + c2
                merged_val = (v1 * c1 + v2 * c2) / merged_count

                block_values[num_blocks - 2] = merged_val
                block_counts[num_blocks - 2] = merged_count
                block_ends[num_blocks - 2] = end2
                num_blocks -= 1

        # Reconstruct
        result = np.empty(n, dtype=y.dtype)
        start = 0
        for b in range(num_blocks):
            val = block_values[b]
            end = block_ends[b]
            for j in range(start, end + 1):
                result[j] = val
            start = end + 1

        return result

    @njit(cache=True)
    def _pava_single_numba(y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Single-sequence PAVA with weights."""
        n = len(y)
        if n <= 1:
            return y.copy()

        block_values = np.empty(n, dtype=y.dtype)
        block_weights = np.empty(n, dtype=w.dtype)
        block_ends = np.empty(n, dtype=np.int64)
        num_blocks = 0

        for i in range(n):
            block_values[num_blocks] = y[i]
            block_weights[num_blocks] = w[i]
            block_ends[num_blocks] = i
            num_blocks += 1

            # Merge while violation exists
            while num_blocks > 1 and block_values[num_blocks - 2] > block_values[num_blocks - 1]:
                v1, w1 = block_values[num_blocks - 2], block_weights[num_blocks - 2]
                v2, w2 = block_values[num_blocks - 1], block_weights[num_blocks - 1]
                end2 = block_ends[num_blocks - 1]

                merged_w = w1 + w2
                merged_v = (v1 * w1 + v2 * w2) / merged_w

                block_values[num_blocks - 2] = merged_v
                block_weights[num_blocks - 2] = merged_w
                block_ends[num_blocks - 2] = end2
                num_blocks -= 1

        # Reconstruct
        result = np.empty(n, dtype=y.dtype)
        start = 0
        for b in range(num_blocks):
            val = block_values[b]
            end = block_ends[b]
            for j in range(start, end + 1):
                result[j] = val
            start = end + 1

        return result

    @njit(parallel=True, cache=True)
    def _pava_batch_numba_no_weights(y_batch: np.ndarray) -> np.ndarray:
        """Batch PAVA without weights"""
        batch_size = y_batch.shape[0]
        result = np.empty_like(y_batch)
        for b in prange(batch_size):
            result[b] = _pava_single_numba_no_weights(y_batch[b])
        return result

    @njit(parallel=True, cache=True)
    def _pava_batch_numba(y_batch: np.ndarray, w_batch: np.ndarray) -> np.ndarray:
        """Batch PAVA with weights"""
        batch_size = y_batch.shape[0]
        result = np.empty_like(y_batch)
        for b in prange(batch_size):
            result[b] = _pava_single_numba(y_batch[b], w_batch[b])
        return result


def enforce_monotonicity(quantiles: Tensor, method: str = "sort", weights: Optional[Tensor] = None) -> Tensor:
    """Enforce monotonicity of quantiles to fix crossing.

    Parameters
    ----------
    quantiles : Tensor
        Predicted quantiles that may have crossing violations.
        Shape: ``(*batch_shape, num_quantiles)``.

    method : str, default="sort"
        Method for fixing crossing:

        - ``"sort"``: Sort values (fast, default)
        - ``"isotonic"``: Pool Adjacent Violators (optimal L2, :math:`O(N)`)
        - ``"cummax"``: Cumulative maximum (fast but distorts distribution)

    weights : Tensor, optional
        Case weights for PAVA method.
        Shape: ``(*batch_shape, num_quantiles)`` or None.

    Returns
    -------
    Tensor
        Monotonically non-decreasing quantiles.
        Shape: ``(*batch_shape, num_quantiles)``.
    """
    if method == "isotonic":
        return isotonic_regression_pava(quantiles, weights)
    elif method == "cummax":
        return torch.cummax(quantiles, dim=-1).values
    elif method == "sort":
        return torch.sort(quantiles, dim=-1).values
    else:
        raise ValueError(f"Unknown method: {method}. Use 'isotonic', 'cummax', or 'sort'.")


def estimate_exp_tail_params(
    quantiles: Tensor,
    alpha_levels: Tensor,
    num_tail_quantiles: int = 20,
) -> Tuple[Tensor, Tensor]:
    """Estimate exponential tail parameters using log-space linear regression.

    For exponential tails:

    .. math::

        \\text{Left:} \\quad Q(\\alpha) = \\beta_L \\cdot \\ln(\\alpha) + c_L

        \\text{Right:} \\quad Q(\\alpha) = -\\beta_R \\cdot \\ln(1 - \\alpha) + c_R

    We estimate :math:`\\beta` by regressing :math:`Q` against
    :math:`\\ln(\\alpha)` (or :math:`\\ln(1 - \\alpha)` for right tail).

    Parameters
    ----------
    quantiles : Tensor
        Quantile values after monotonicity correction.
        Shape: ``(*batch_shape, num_quantiles)``.

    alpha_levels : Tensor
        Probability levels corresponding to quantiles.
        Shape: ``(num_quantiles,)``.

    num_tail_quantiles : int, default=20
        Number of quantiles in each tail to use for estimation.

    Returns
    -------
    beta_l : Tensor
        Left tail scale parameter. Shape: ``(*batch_shape,)``.

    beta_r : Tensor
        Right tail scale parameter. Shape: ``(*batch_shape,)``.
    """
    cfg = QuantileDistributionConfig
    n = quantiles.shape[-1]
    k = min(num_tail_quantiles, n // 4)

    # === Left tail: Q(α) = β_L·ln(α) + c_L ===
    alpha_left = alpha_levels[:k]
    q_left = quantiles[..., :k]  # Shape: (*batch_shape, k)

    # Log-transform alpha
    ln_alpha_left = torch.log(alpha_left.clamp(min=cfg.TOL))  # Shape: (k,)

    # Linear regression: β = Cov(Q, ln(α)) / Var(ln(α))
    ln_alpha_mean = ln_alpha_left.mean()
    ln_alpha_centered = ln_alpha_left - ln_alpha_mean  # Shape: (k,)

    q_left_mean = q_left.mean(dim=-1, keepdim=True)  # Shape: (*batch_shape, 1)
    q_left_centered = q_left - q_left_mean  # Shape: (*batch_shape, k)

    # Compute covariance and variance
    cov_left = (q_left_centered * ln_alpha_centered).mean(dim=-1)  # Shape: (*batch_shape,)
    var_ln_alpha_left = (ln_alpha_centered**2).mean()  # Scalar

    beta_l = cov_left / var_ln_alpha_left.clamp(min=cfg.TOL)
    beta_l = torch.clamp(beta_l.abs(), min=cfg.MIN_BETA, max=cfg.MAX_BETA)

    # === Right tail: Q(α) = -β_R·ln(1-α) + c_R ===
    alpha_right = alpha_levels[-k:]
    q_right = quantiles[..., -k:]  # Shape: (*batch_shape, k)

    # Log-transform (1 - alpha)
    ln_one_minus_alpha = torch.log((1 - alpha_right).clamp(min=cfg.TOL))  # Shape: (k,)

    # Linear regression
    ln_1ma_mean = ln_one_minus_alpha.mean()
    ln_1ma_centered = ln_one_minus_alpha - ln_1ma_mean

    q_right_mean = q_right.mean(dim=-1, keepdim=True)
    q_right_centered = q_right - q_right_mean

    cov_right = (q_right_centered * ln_1ma_centered).mean(dim=-1)
    var_ln_1ma = (ln_1ma_centered**2).mean()

    # Note: Q = -β·ln(1-α) + c, so coefficient is -β
    beta_r = -cov_right / var_ln_1ma.clamp(min=cfg.TOL)
    beta_r = torch.clamp(beta_r.abs(), min=cfg.MIN_BETA, max=cfg.MAX_BETA)

    return beta_l, beta_r


def estimate_gpd_tail_params(
    quantiles: Tensor,
    alpha_levels: Tensor,
    num_tail_quantiles: int = 20,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Estimate GPD tail parameters using Pickands-like estimator.

    For GPD with shape :math:`\\eta` and scale :math:`\\mu`:

    .. math::

        \\text{Left:} \\quad Q(\\alpha) = q_L - \\frac{\\mu}{\\eta}
        \\left( \\left(\\frac{\\alpha_L}{\\alpha}\\right)^\\eta - 1 \\right)

        \\text{Right:} \\quad Q(\\alpha) = q_R + \\frac{\\mu}{\\eta}
        \\left( \\left(\\frac{1 - \\alpha_R}{1 - \\alpha}\\right)^\\eta - 1 \\right)

    We use a modified Pickands estimator based on quantile spacing ratios.

    Parameters
    ----------
    quantiles : Tensor
        Quantile values after monotonicity correction.
        Shape: ``(*batch_shape, num_quantiles)``.

    alpha_levels : Tensor
        Probability levels.
        Shape: ``(num_quantiles,)``.

    num_tail_quantiles : int, default=20
        Number of quantiles in each tail to use for estimation.

    Returns
    -------
    eta_l : Tensor
        Left tail shape parameter. Shape: ``(*batch_shape,)``.

    mu_l : Tensor
        Left tail scale parameter. Shape: ``(*batch_shape,)``.

    eta_r : Tensor
        Right tail shape parameter. Shape: ``(*batch_shape,)``.

    mu_r : Tensor
        Right tail scale parameter. Shape: ``(*batch_shape,)``.
    """
    cfg = QuantileDistributionConfig
    n = quantiles.shape[-1]
    k = min(num_tail_quantiles, n // 4)

    # First get exponential estimates for μ (scale parameter)
    beta_l, beta_r = estimate_exp_tail_params(quantiles, alpha_levels, num_tail_quantiles)

    # === Left tail - Pickands-like estimation ===
    # Use three quantile levels: idx_1 (most extreme), idx_2 (medium), idx_3 (less extreme)
    idx_1 = 0
    idx_2 = k // 3
    idx_3 = 2 * k // 3

    q1_left = quantiles[..., idx_1]
    q2_left = quantiles[..., idx_2]
    q3_left = quantiles[..., idx_3]

    alpha_1 = alpha_levels[idx_1]
    alpha_2 = alpha_levels[idx_2]
    alpha_3 = alpha_levels[idx_3]

    # For left tail, smaller α = more extreme = smaller Q
    # Q spacing ratios
    delta_q_12 = q2_left - q1_left  # Q2 > Q1 for left tail
    delta_q_23 = q3_left - q2_left  # Q3 > Q2

    # Log-alpha spacing ratios
    ln_alpha_12 = torch.log((alpha_2 / alpha_1).clamp(min=cfg.TOL))
    ln_alpha_23 = torch.log((alpha_3 / alpha_2).clamp(min=cfg.TOL))

    # For exponential tail: ΔQ_12 / ΔQ_23 = ln(α2/α1) / ln(α3/α2)
    expected_q_ratio = ln_alpha_12 / ln_alpha_23.clamp(min=cfg.TOL)
    actual_q_ratio = delta_q_12 / delta_q_23.clamp(min=cfg.TOL)

    # η estimate: deviation from exponential behavior
    # If actual_ratio > expected_ratio: heavier tail (η > 0)
    # If actual_ratio < expected_ratio: lighter tail (η < 0)
    ratio_deviation = actual_q_ratio / expected_q_ratio.clamp(min=cfg.TOL)

    # Map deviation to η using log relationship
    # η ≈ (ln(ratio_deviation)) / (ln(α2/α1))
    eta_l_raw = torch.where(
        ratio_deviation > cfg.TOL,
        torch.log(ratio_deviation.clamp(min=cfg.TOL)) / ln_alpha_12.abs().clamp(min=cfg.TOL),
        torch.zeros_like(ratio_deviation),
    )
    eta_l = torch.clamp(eta_l_raw, min=cfg.MIN_ETA, max=cfg.MAX_ETA)
    mu_l = beta_l

    # === Right tail - Pickands-like estimation ===
    idx_1_r = n - 1
    idx_2_r = n - 1 - k // 3
    idx_3_r = n - 1 - 2 * k // 3

    q1_right = quantiles[..., idx_1_r]
    q2_right = quantiles[..., idx_2_r]
    q3_right = quantiles[..., idx_3_r]

    alpha_1_r = alpha_levels[idx_1_r]
    alpha_2_r = alpha_levels[idx_2_r]
    alpha_3_r = alpha_levels[idx_3_r]

    # For right tail, larger α = larger Q
    delta_q_12_r = q1_right - q2_right  # Q1 > Q2 for right tail
    delta_q_23_r = q2_right - q3_right

    # Use (1-α) ratios
    one_m_1 = (1 - alpha_1_r).clamp(min=cfg.TOL)
    one_m_2 = (1 - alpha_2_r).clamp(min=cfg.TOL)
    one_m_3 = (1 - alpha_3_r).clamp(min=cfg.TOL)

    ln_1ma_12 = torch.log((one_m_2 / one_m_1).clamp(min=cfg.TOL))
    ln_1ma_23 = torch.log((one_m_3 / one_m_2).clamp(min=cfg.TOL))

    expected_q_ratio_r = ln_1ma_12 / ln_1ma_23.clamp(min=cfg.TOL)
    actual_q_ratio_r = delta_q_12_r / delta_q_23_r.clamp(min=cfg.TOL)

    ratio_deviation_r = actual_q_ratio_r / expected_q_ratio_r.clamp(min=cfg.TOL)
    eta_r_raw = torch.where(
        ratio_deviation_r > cfg.TOL,
        torch.log(ratio_deviation_r.clamp(min=cfg.TOL)) / ln_1ma_12.abs().clamp(min=cfg.TOL),
        torch.zeros_like(ratio_deviation_r),
    )
    eta_r = torch.clamp(eta_r_raw, min=cfg.MIN_ETA, max=cfg.MAX_ETA)
    mu_r = beta_r

    return eta_l, mu_l, eta_r, mu_r


class QuantileDistribution(Distribution):
    """Probability distribution constructed from predicted quantiles.

    Wraps a set of predicted quantiles into a proper distribution with:

    - Monotonicity enforcement (fixes quantile crossing)
    - Tail extrapolation (exponential or GPD) with data-inferred parameters
    - Analytical statistics (CDF, PDF, CRPS, mean, variance)

    Parameters
    ----------
    quantiles : Tensor
        Predicted quantile values.
        Shape: ``(*batch_shape, num_quantiles)``.

    alpha_levels : Tensor, optional
        Probability levels corresponding to each quantile.
        Shape: ``(num_quantiles,)``.
        Default: ``linspace(0.0, 1.0, num_quantiles + 2)[1:-1]``.

    tail_type : str, default="exp"
        Type of tail extrapolation beyond the observed quantile range.

        - ``"exp"``: Exponential tails (lighter, suitable for sub-exponential)
        - ``"gpd"``: Generalized Pareto Distribution (heavier, for power-law tails)

    fix_crossing : bool, default=True
        Whether to enforce monotonicity to fix quantile crossing.

    crossing_method : str, default="sort"
        Method for fixing crossing: ``"sort"``, ``"isotonic"``, or ``"cummax"``.
    """

    arg_constraints = {}
    support = torch.distributions.constraints.real
    has_rsample = False

    def __init__(
        self,
        quantiles: Tensor,
        alpha_levels: Optional[Tensor] = None,
        tail_type: Literal["exp", "gpd"] = "exp",
        fix_crossing: bool = True,
        crossing_method: str = "sort",
    ):
        self.cfg = QuantileDistributionConfig
        self.tol = self.cfg.TOL
        self.tail_type = tail_type

        # Store shapes
        self._batch_shape = quantiles.shape[:-1]
        self.num_quantiles = quantiles.shape[-1]

        # Default alpha levels
        if alpha_levels is None:
            alpha_levels = torch.linspace(
                0.0, 1.0, self.num_quantiles + 2, device=quantiles.device, dtype=quantiles.dtype
            )[1:-1]
        self.alpha_levels = alpha_levels

        # Fix quantile crossing
        if fix_crossing:
            quantiles = enforce_monotonicity(quantiles, method=crossing_method)

        self.quantiles = quantiles

        # Setup internal structures
        self._setup_spline()
        self._setup_tails()

        super().__init__(batch_shape=self._batch_shape, validate_args=False)

    def _setup_spline(self):
        """Setup linear spline segments between quantile knots.

        Creates segment boundaries and slopes for piecewise linear interpolation.
        """
        # Expand alpha_levels for batch operations
        alpha = self.alpha_levels
        while alpha.dim() < len(self._batch_shape) + 1:
            alpha = alpha.unsqueeze(0)
        alpha = alpha.expand(*self._batch_shape, -1)

        # Segment boundaries: (*batch_shape, num_segments) where num_segments = num_quantiles - 1
        self.alpha_lo = alpha[..., :-1]  # α_i for segment i
        self.alpha_hi = alpha[..., 1:]  # α_{i+1} for segment i
        self.q_lo = self.quantiles[..., :-1]  # Q(α_i)
        self.q_hi = self.quantiles[..., 1:]  # Q(α_{i+1})

        # Segment properties
        self.delta_alpha = self.alpha_hi - self.alpha_lo  # Shape: (*batch_shape, num_segments)
        self.delta_q = self.q_hi - self.q_lo
        self.slopes = self.delta_q / torch.clamp(self.delta_alpha, min=self.tol)

        self.num_segments = self.num_quantiles - 1

        # Boundary values (scalars for alpha, tensors for q)
        self.alpha_l = self.alpha_levels[0].item()
        self.alpha_r = self.alpha_levels[-1].item()
        self.q_l = self.quantiles[..., 0]  # Shape: (*batch_shape,)
        self.q_r = self.quantiles[..., -1]  # Shape: (*batch_shape,)

    def _setup_tails(self):
        """Setup tail parameters inferred from boundary quantiles."""
        cfg = self.cfg
        device = self.quantiles.device
        dtype = self.quantiles.dtype

        num_tail_q = cfg.TAIL_QUANTILES_FOR_ESTIMATION

        if self.tail_type == "exp":
            # Estimate β using log-space regression
            self.beta_l, self.beta_r = estimate_exp_tail_params(
                self.quantiles,
                self.alpha_levels,
                num_tail_quantiles=num_tail_q,
            )

            # Compute tail coefficients
            # Left: Q(α) = a_l·ln(α) + b_l where a_l = β_l
            alpha_l_safe = max(self.alpha_l, self.tol)
            self.tail_a_l = self.beta_l
            self.tail_b_l = self.q_l - self.tail_a_l * torch.log(torch.tensor(alpha_l_safe, device=device, dtype=dtype))

            # Right: Q(α) = a_r·ln(1-α) + b_r where a_r = -β_r
            alpha_r_safe = min(self.alpha_r, 1 - self.tol)
            self.tail_a_r = -self.beta_r
            self.tail_b_r = self.q_r - self.tail_a_r * torch.log(
                torch.tensor(1 - alpha_r_safe, device=device, dtype=dtype)
            )
        else:  # GPD tails
            # Estimate GPD parameters using Pickands-like estimator
            self.eta_l, self.mu_l, self.eta_r, self.mu_r = estimate_gpd_tail_params(
                self.quantiles,
                self.alpha_levels,
                num_tail_quantiles=num_tail_q,
            )

    def icdf(self, alpha: Tensor) -> Tensor:
        """Compute quantile function :math:`Q(\\alpha) = F^{-1}(\\alpha)`.

        Parameters
        ----------
        alpha : Tensor
            Probability levels at which to evaluate quantiles.
            Shape: scalar, ``(n,)``, or ``(*batch_shape, n)``.

            If shape is ``(n,)``, broadcasts to ``(*batch_shape, n)``.
            If shape is ``(*batch_shape, n)``, must match distribution batch shape.

        Returns
        -------
        Tensor
            Quantile values :math:`Q(\\alpha)`.
            Shape: ``(*batch_shape, n)`` or ``(*batch_shape,)`` if alpha is scalar.
        """

        squeeze_output = False
        if alpha.dim() == 0:
            alpha = alpha.unsqueeze(0)
            squeeze_output = True

        alpha_shape = alpha.shape

        # Expand alpha for batch dimensions if needed
        if alpha.dim() == 1:
            # Shape (n,) -> (*batch_shape, n)
            alpha_expanded = alpha
            for _ in range(len(self._batch_shape)):
                alpha_expanded = alpha_expanded.unsqueeze(0)
            alpha_expanded = alpha_expanded.expand(*self._batch_shape, *alpha_shape)
        else:
            alpha_expanded = alpha

        # Compute in each region
        q_left = self._icdf_left_tail(alpha_expanded)
        q_right = self._icdf_right_tail(alpha_expanded)
        q_spline = self._icdf_spline(alpha_expanded)

        # Select based on region
        result = torch.where(
            alpha_expanded < self.alpha_l, q_left, torch.where(alpha_expanded > self.alpha_r, q_right, q_spline)
        )

        if squeeze_output:
            result = result.squeeze(-1)

        return result

    def _expand_to_alpha(self, param: Tensor, alpha: Tensor) -> Tensor:
        """Expand batch parameter to match alpha shape.

        Parameters
        ----------
        param : Tensor
            Parameter with shape ``(*batch_shape,)`` or ``(*batch_shape, k)``.

        alpha : Tensor
            Target tensor with shape ``(*batch_shape, n)``.

        Returns
        -------
        Tensor
            Expanded parameter matching alpha shape.
        """
        n_expand = alpha.dim() - param.dim()
        result = param
        for _ in range(n_expand):
            result = result.unsqueeze(-1)
        return result.expand_as(alpha)

    def _icdf_left_tail(self, alpha: Tensor) -> Tensor:
        """Left tail quantile :math:`Q(\\alpha)` for :math:`\\alpha < \\alpha_L`."""
        if self.tail_type == "exp":
            a = self._expand_to_alpha(self.tail_a_l, alpha)
            b = self._expand_to_alpha(self.tail_b_l, alpha)
            alpha_safe = torch.clamp(alpha, min=self.tol)
            return a * torch.log(alpha_safe) + b
        else:
            return self._icdf_gpd_left(alpha)

    def _icdf_right_tail(self, alpha: Tensor) -> Tensor:
        """Right tail quantile :math:`Q(\\alpha)` for :math:`\\alpha > \\alpha_R`."""
        if self.tail_type == "exp":
            a = self._expand_to_alpha(self.tail_a_r, alpha)
            b = self._expand_to_alpha(self.tail_b_r, alpha)
            one_m_alpha = torch.clamp(1 - alpha, min=self.tol)
            return a * torch.log(one_m_alpha) + b
        else:
            return self._icdf_gpd_right(alpha)

    def _icdf_gpd_left(self, alpha: Tensor) -> Tensor:
        r"""GPD left tail: :math:`Q(\alpha) = q_L - \frac{\mu}{\eta}\left((\alpha_L / \alpha)^\eta - 1\right)`."""
        cfg = self.cfg
        eta = self._expand_to_alpha(self.eta_l, alpha)
        mu = self._expand_to_alpha(self.mu_l, alpha)
        q_L = self._expand_to_alpha(self.q_l, alpha)

        is_exp_approx = torch.abs(eta) < cfg.ETA_TOLERANCE
        eta_safe = torch.where(is_exp_approx, torch.full_like(eta, cfg.ETA_TOLERANCE), eta)

        alpha_safe = torch.clamp(alpha, min=self.tol)
        ratio = self.alpha_l / alpha_safe
        log_ratio = torch.clamp(torch.log(ratio), max=cfg.MAX_LOG_RATIO)
        ratio_pow = torch.exp(torch.clamp(eta_safe * log_ratio, max=cfg.MAX_EXPONENT))

        gpd_result = q_L - mu / eta_safe * (ratio_pow - 1.0)
        exp_result = q_L - mu * log_ratio

        return torch.where(is_exp_approx, exp_result, gpd_result)

    def _icdf_gpd_right(self, alpha: Tensor) -> Tensor:
        r"""GPD right tail: :math:`Q(\alpha) = q_R + \frac{\mu}{\eta}\left(\left(\frac{1-\alpha_R}{1-\alpha}\right)^\eta - 1\right)`."""
        cfg = self.cfg
        eta = self._expand_to_alpha(self.eta_r, alpha)
        mu = self._expand_to_alpha(self.mu_r, alpha)
        q_R = self._expand_to_alpha(self.q_r, alpha)

        is_exp_approx = torch.abs(eta) < cfg.ETA_TOLERANCE
        eta_safe = torch.where(is_exp_approx, torch.full_like(eta, cfg.ETA_TOLERANCE), eta)

        one_m_alpha = torch.clamp(1 - alpha, min=self.tol)
        ratio = (1 - self.alpha_r) / one_m_alpha
        log_ratio = torch.clamp(torch.log(ratio), max=cfg.MAX_LOG_RATIO)
        ratio_pow = torch.exp(torch.clamp(eta_safe * log_ratio, max=cfg.MAX_EXPONENT))

        gpd_result = q_R + mu / eta_safe * (ratio_pow - 1.0)
        exp_result = q_R + mu * log_ratio

        return torch.where(is_exp_approx, exp_result, gpd_result)

    def _icdf_spline(self, alpha: Tensor) -> Tensor:
        """Piecewise linear quantile interpolation for :math:`\\alpha_L \\le \\alpha \\le \\alpha_R`.

        Parameters
        ----------
        alpha : Tensor
            Shape: ``(*batch_shape, n)``.

        Returns
        -------
        Tensor
            Shape: ``(*batch_shape, n)``.
        """
        n_extra = alpha.dim() - len(self._batch_shape)
        seg_dim = len(self._batch_shape)

        # Expand segment data for broadcasting
        # From (*batch_shape, num_segments) to (*batch_shape, num_segments, 1, ..., 1)
        alpha_lo = self.alpha_lo
        alpha_hi = self.alpha_hi
        q_lo = self.q_lo
        q_hi = self.q_hi

        for _ in range(n_extra):
            alpha_lo = alpha_lo.unsqueeze(-1)
            alpha_hi = alpha_hi.unsqueeze(-1)
            q_lo = q_lo.unsqueeze(-1)
            q_hi = q_hi.unsqueeze(-1)

        # Expand alpha for segment dimension: (*batch_shape, 1, n)
        alpha_exp = alpha.unsqueeze(seg_dim)

        # Find which segment each alpha belongs to
        in_segment = (alpha_exp >= alpha_lo) & (alpha_exp < alpha_hi)

        # Linear interpolation: Q = q_lo + t * (q_hi - q_lo) where t = (α - α_lo) / Δα
        t = (alpha_exp - alpha_lo) / torch.clamp(alpha_hi - alpha_lo, min=self.tol)
        t = torch.clamp(t, 0.0, 1.0)
        q_all_seg = q_lo + t * (q_hi - q_lo)

        # Select correct segment via masked sum
        mask = in_segment.float()
        mask_sum = mask.sum(dim=seg_dim, keepdim=True).clamp(min=1.0)
        result = (q_all_seg * mask).sum(dim=seg_dim) / mask_sum.squeeze(seg_dim)

        # Handle α >= α_R
        q_r_exp = self._expand_to_alpha(self.q_r, alpha)
        result = torch.where(alpha >= self.alpha_r, q_r_exp, result)

        return result

    def cdf(self, z: Tensor) -> Tensor:
        """Compute cumulative distribution function :math:`F(z) = P(Z \\le z)`.

        Parameters
        ----------
        z : Tensor
            Values at which to evaluate CDF.
            Shape: scalar, ``(n,)``, or ``(*batch_shape, ...)``.

            If shape is ``(n,)``, broadcasts to ``(*batch_shape, n)``.

        Returns
        -------
        Tensor
            CDF values in :math:`[0, 1]`.
            Shape: ``(*batch_shape, ...)`` matching z after broadcasting.
        """
        # Handle 1D input by broadcasting (only if z is not already batch_shape)
        if z.dim() == 1 and len(self._batch_shape) > 0 and z.shape != self._batch_shape:
            z = z.unsqueeze(0).expand(*self._batch_shape, -1)

        n_extra = z.dim() - len(self._batch_shape)

        q_l_exp = self.q_l
        q_r_exp = self.q_r
        for _ in range(n_extra):
            q_l_exp = q_l_exp.unsqueeze(-1)
            q_r_exp = q_r_exp.unsqueeze(-1)

        cdf_left = self._cdf_left_tail(z)
        cdf_right = self._cdf_right_tail(z)
        cdf_spline = self._cdf_spline(z)

        return torch.where(z < q_l_exp, cdf_left, torch.where(z > q_r_exp, cdf_right, cdf_spline))

    def _expand_to_z(self, param: Tensor, z: Tensor) -> Tensor:
        """Expand batch parameter to match z shape."""
        n_expand = z.dim() - param.dim()
        result = param
        for _ in range(n_expand):
            result = result.unsqueeze(-1)
        return result.expand_as(z)

    def _cdf_left_tail(self, z: Tensor) -> Tensor:
        """CDF in left tail region (:math:`z < q_L`)."""
        cfg = self.cfg
        if self.tail_type == "exp":
            a = self._expand_to_z(self.tail_a_l, z)
            b = self._expand_to_z(self.tail_b_l, z)
            a_safe = torch.clamp(a.abs(), min=self.tol)
            log_alpha = torch.clamp((z - b) / a_safe, max=0.0)
            alpha = torch.exp(log_alpha)
            return torch.clamp(alpha, min=0.0, max=self.alpha_l)
        else:
            eta = self._expand_to_z(self.eta_l, z)
            mu = self._expand_to_z(self.mu_l, z)
            q_L = self._expand_to_z(self.q_l, z)

            is_exp = torch.abs(eta) < cfg.ETA_TOLERANCE
            eta_safe = torch.where(is_exp, torch.full_like(eta, cfg.ETA_TOLERANCE), eta)
            mu_safe = torch.clamp(mu, min=self.tol)

            psi = torch.clamp((q_L - z) / mu_safe, min=0.0, max=cfg.MAX_EXPONENT)
            inner = torch.clamp(1.0 + eta_safe * psi, min=self.tol)
            exp_arg = torch.clamp(-torch.log(inner) / eta_safe, min=-cfg.MAX_EXPONENT, max=cfg.MAX_EXPONENT)

            gpd_alpha = self.alpha_l * torch.exp(exp_arg)
            exp_alpha = self.alpha_l * torch.exp(-psi)
            alpha = torch.where(is_exp, exp_alpha, gpd_alpha)
            return torch.clamp(alpha, min=0.0, max=self.alpha_l)

    def _cdf_right_tail(self, z: Tensor) -> Tensor:
        """CDF in right tail region (:math:`z > q_R`)."""
        cfg = self.cfg
        if self.tail_type == "exp":
            a = self._expand_to_z(self.tail_a_r, z)
            b = self._expand_to_z(self.tail_b_r, z)
            a_safe = torch.clamp((-a).abs(), min=self.tol)
            log_one_m = torch.clamp((z - b) / (-a_safe), max=0.0)
            one_m_alpha = torch.exp(log_one_m)
            return torch.clamp(1.0 - one_m_alpha, min=self.alpha_r, max=1.0)
        else:
            eta = self._expand_to_z(self.eta_r, z)
            mu = self._expand_to_z(self.mu_r, z)
            q_R = self._expand_to_z(self.q_r, z)

            is_exp = torch.abs(eta) < cfg.ETA_TOLERANCE
            eta_safe = torch.where(is_exp, torch.full_like(eta, cfg.ETA_TOLERANCE), eta)
            mu_safe = torch.clamp(mu, min=self.tol)

            psi = torch.clamp((z - q_R) / mu_safe, min=0.0, max=cfg.MAX_EXPONENT)
            inner = torch.clamp(1.0 + eta_safe * psi, min=self.tol)
            exp_arg = torch.clamp(-torch.log(inner) / eta_safe, min=-cfg.MAX_EXPONENT, max=cfg.MAX_EXPONENT)

            one_m_r = 1 - self.alpha_r
            gpd_one_m = one_m_r * torch.exp(exp_arg)
            exp_one_m = one_m_r * torch.exp(-psi)
            one_m = torch.where(is_exp, exp_one_m, gpd_one_m)
            return torch.clamp(1.0 - one_m, min=self.alpha_r, max=1.0)

    def _cdf_spline(self, z: Tensor) -> Tensor:
        """CDF in spline region via inverse linear interpolation."""
        n_extra = z.dim() - len(self._batch_shape)
        seg_dim = len(self._batch_shape)

        # Expand segment data
        alpha_lo = self.alpha_lo
        alpha_hi = self.alpha_hi
        q_lo = self.q_lo
        q_hi = self.q_hi

        for _ in range(n_extra):
            alpha_lo = alpha_lo.unsqueeze(-1)
            alpha_hi = alpha_hi.unsqueeze(-1)
            q_lo = q_lo.unsqueeze(-1)
            q_hi = q_hi.unsqueeze(-1)

        z_exp = z.unsqueeze(seg_dim)

        in_segment = (z_exp >= q_lo) & (z_exp < q_hi)

        # Inverse interpolation: α = α_lo + t * Δα where t = (z - q_lo) / Δq
        t = (z_exp - q_lo) / torch.clamp(q_hi - q_lo, min=self.tol)
        t = torch.clamp(t, 0.0, 1.0)
        alpha_all = alpha_lo + t * (alpha_hi - alpha_lo)

        mask = in_segment.float()
        mask_sum = mask.sum(dim=seg_dim, keepdim=True).clamp(min=1.0)
        result = (alpha_all * mask).sum(dim=seg_dim) / mask_sum.squeeze(seg_dim)

        q_r_exp = self._expand_to_z(self.q_r, z)
        result = torch.where(z >= q_r_exp, torch.tensor(self.alpha_r, device=z.device, dtype=z.dtype), result)

        return result

    def _icdf_derivative(self, alpha: Tensor) -> Tensor:
        """Compute :math:`dQ/d\\alpha` at given probability levels.

        Parameters
        ----------
        alpha : Tensor
            Probability levels.
            Shape: ``(*batch_shape, n)``.

        Returns
        -------
        Tensor
            Derivative values :math:`dQ/d\\alpha` (always positive).
            Shape: ``(*batch_shape, n)``.
        """
        cfg = self.cfg

        alpha_l_t = torch.tensor(self.alpha_l, device=alpha.device, dtype=alpha.dtype)
        alpha_r_t = torch.tensor(self.alpha_r, device=alpha.device, dtype=alpha.dtype)

        if alpha.dim() > len(self._batch_shape):
            alpha_l_t = self._expand_to_alpha(alpha_l_t.expand(*self._batch_shape), alpha)
            alpha_r_t = self._expand_to_alpha(alpha_r_t.expand(*self._batch_shape), alpha)

        deriv_left = self._deriv_left_tail(alpha)
        deriv_right = self._deriv_right_tail(alpha)
        deriv_spline = self._deriv_spline(alpha)

        deriv = torch.where(alpha < alpha_l_t, deriv_left, torch.where(alpha > alpha_r_t, deriv_right, deriv_spline))

        return torch.clamp(deriv, min=cfg.MIN_SLOPE, max=cfg.MAX_SLOPE)

    def _deriv_left_tail(self, alpha: Tensor) -> Tensor:
        """Left tail derivative: :math:`dQ/d\\alpha` for :math:`\\alpha < \\alpha_L`."""
        if self.tail_type == "exp":
            a = self._expand_to_alpha(self.tail_a_l, alpha)
            alpha_safe = torch.clamp(alpha, min=self.tol)
            return a / alpha_safe
        else:
            cfg = self.cfg
            eta = self._expand_to_alpha(self.eta_l, alpha)
            mu = self._expand_to_alpha(self.mu_l, alpha)

            is_exp = torch.abs(eta) < cfg.ETA_TOLERANCE
            eta_safe = torch.where(is_exp, torch.full_like(eta, cfg.ETA_TOLERANCE), eta)

            alpha_safe = torch.clamp(alpha, min=self.tol)
            ratio = self.alpha_l / alpha_safe
            log_ratio = torch.clamp(torch.log(ratio), max=cfg.MAX_LOG_RATIO)
            ratio_pow = torch.exp(torch.clamp(eta_safe * log_ratio, max=cfg.MAX_EXPONENT))

            gpd_deriv = mu * ratio_pow / alpha_safe
            exp_deriv = mu / alpha_safe
            return torch.where(is_exp, exp_deriv, gpd_deriv)

    def _deriv_right_tail(self, alpha: Tensor) -> Tensor:
        """Right tail derivative: :math:`dQ/d\\alpha` for :math:`\\alpha > \\alpha_R`."""
        if self.tail_type == "exp":
            a = self._expand_to_alpha(self.tail_a_r, alpha)
            one_m = torch.clamp(1 - alpha, min=self.tol)
            return (-a) / one_m
        else:
            cfg = self.cfg
            eta = self._expand_to_alpha(self.eta_r, alpha)
            mu = self._expand_to_alpha(self.mu_r, alpha)

            is_exp = torch.abs(eta) < cfg.ETA_TOLERANCE
            eta_safe = torch.where(is_exp, torch.full_like(eta, cfg.ETA_TOLERANCE), eta)

            one_m = torch.clamp(1 - alpha, min=self.tol)
            ratio = (1 - self.alpha_r) / one_m
            log_ratio = torch.clamp(torch.log(ratio), max=cfg.MAX_LOG_RATIO)
            ratio_pow = torch.exp(torch.clamp(eta_safe * log_ratio, max=cfg.MAX_EXPONENT))

            gpd_deriv = mu * ratio_pow / one_m
            exp_deriv = mu / one_m
            return torch.where(is_exp, exp_deriv, gpd_deriv)

    def _deriv_spline(self, alpha: Tensor) -> Tensor:
        """Spline derivative: piecewise constant slopes."""
        n_extra = alpha.dim() - len(self._batch_shape)
        seg_dim = len(self._batch_shape)

        slopes = self.slopes
        alpha_lo = self.alpha_lo
        alpha_hi = self.alpha_hi

        for _ in range(n_extra):
            slopes = slopes.unsqueeze(-1)
            alpha_lo = alpha_lo.unsqueeze(-1)
            alpha_hi = alpha_hi.unsqueeze(-1)

        alpha_exp = alpha.unsqueeze(seg_dim)
        in_segment = (alpha_exp >= alpha_lo) & (alpha_exp < alpha_hi)

        mask = in_segment.float()
        mask_sum = mask.sum(dim=seg_dim, keepdim=True).clamp(min=1.0)
        result = (slopes * mask).sum(dim=seg_dim) / mask_sum.squeeze(seg_dim)

        no_segment = mask.sum(dim=seg_dim) < 0.5
        result = torch.where(no_segment, torch.ones_like(result), result)

        return result

    def log_prob(self, z: Tensor) -> Tensor:
        """Compute log probability density :math:`\\log f(z)`.

        Uses the relation :math:`f(z) = 1 / Q'(F(z))`, so
        :math:`\\log f(z) = -\\log Q'(F(z))`.

        Parameters
        ----------
        z : Tensor
            Values at which to evaluate log density.
            Shape: scalar, ``(n,)``, or ``(*batch_shape, ...)``.

            If shape is ``(n,)``, broadcasts to ``(*batch_shape, n)``.

        Returns
        -------
        Tensor
            Log probability density values.
            Shape: ``(*batch_shape, ...)`` matching z after broadcasting.
        """
        # Handle 1D input (only if z is not already batch_shape)
        if z.dim() == 1 and len(self._batch_shape) > 0 and z.shape != self._batch_shape:
            z = z.unsqueeze(0).expand(*self._batch_shape, -1)

        alpha = self.cdf(z)
        q_deriv = self._icdf_derivative(alpha)
        return -torch.log(q_deriv)

    def pdf(self, z: Tensor) -> Tensor:
        """Compute probability density function :math:`f(z)`.

        Parameters
        ----------
        z : Tensor
            Values at which to evaluate PDF.
            Shape: scalar, ``(n,)``, or ``(*batch_shape, ...)``.

            If shape is ``(n,)``, broadcasts to ``(*batch_shape, n)``.

        Returns
        -------
        Tensor
            PDF values (non-negative).
            Shape: ``(*batch_shape, ...)`` matching z after broadcasting.
        """
        return torch.exp(self.log_prob(z))

    def crps(self, z: Tensor) -> Tensor:
        """Compute analytical CRPS (Continuous Ranked Probability Score).

        .. math::

            \\text{CRPS} = E[|Z - z|] - 0.5 \\cdot E[|Z - Z'|]
            = \\int_0^{F(z)} 2\\alpha(z - Q(\\alpha))\\,d\\alpha
            + \\int_{F(z)}^1 2(1 - \\alpha)(Q(\\alpha) - z)\\,d\\alpha

        Parameters
        ----------
        z : Tensor
            Observation values.
            Shape: ``(*batch_shape,)`` or ``(*batch_shape, ...)``.

        Returns
        -------
        Tensor
            CRPS values (non-negative, lower is better).
            Shape: same as z.
        """
        cfg = self.cfg

        alpha_z = self.cdf(z)

        crps_left = self._crps_left_tail(z, alpha_z)
        crps_spline = self._crps_spline(z, alpha_z)
        crps_right = self._crps_right_tail(z, alpha_z)

        return torch.clamp(crps_left + crps_spline + crps_right, min=0.0, max=cfg.MAX_CRPS)

    def _crps_left_tail(self, z: Tensor, alpha_z: Tensor) -> Tensor:
        """CRPS contribution from left tail :math:`[0, \\alpha_L]`."""
        if self.tail_type == "exp":
            return self._crps_left_tail_exp(z, alpha_z)
        else:
            return self._crps_left_tail_gpd(z, alpha_z)

    def _crps_left_tail_exp(self, z: Tensor, alpha_z: Tensor) -> Tensor:
        """Exponential left tail CRPS (compact formula)."""
        a = self._expand_to_z(self.tail_a_l, z)
        b = self._expand_to_z(self.tail_b_l, z)
        q_L = self._expand_to_z(self.q_l, z)
        alpha_L = self.alpha_l

        alpha_L_safe = max(alpha_L, self.tol)
        alpha_tilde = torch.clamp(alpha_z, min=self.tol, max=alpha_L_safe)

        ln_alpha_L = torch.log(torch.tensor(alpha_L_safe, device=z.device, dtype=z.dtype))

        term1 = (z - b) * (alpha_L_safe**2 - 2 * alpha_L_safe + 2 * alpha_tilde)
        term2 = alpha_L_safe**2 * a * (-ln_alpha_L + 0.5)
        term2 = term2 + 2 * torch.where(
            z < q_L,
            alpha_L_safe * a * (ln_alpha_L - 1) + alpha_tilde * (-z + b + a),
            torch.zeros_like(z),
        )

        return term1 + term2

    def _crps_left_tail_gpd(self, z: Tensor, alpha_z: Tensor) -> Tensor:
        """GPD left tail CRPS."""
        cfg = self.cfg

        eta = self._expand_to_z(self.eta_l, z)
        mu = self._expand_to_z(self.mu_l, z)
        q_L = self._expand_to_z(self.q_l, z)
        alpha_L = self.alpha_l

        is_exp = torch.abs(eta) < cfg.ETA_TOLERANCE
        eta_safe = torch.where(is_exp, torch.full_like(eta, cfg.ETA_TOLERANCE), eta)

        two_m_eta = torch.clamp((2.0 - eta_safe).abs(), min=cfg.ETA_TOLERANCE)
        two_m_eta = torch.where(2.0 - eta_safe >= 0, two_m_eta, -two_m_eta)
        mu_over_2me = torch.clamp(mu / two_m_eta, min=-cfg.MAX_CRPS, max=cfg.MAX_CRPS)

        alpha_sq = alpha_L**2
        simple_crps = alpha_sq * (z - q_L + mu_over_2me)

        alpha_tilde = torch.clamp(alpha_z, min=self.tol, max=alpha_L - self.tol)
        psi = torch.clamp((q_L - z) / torch.clamp(mu, min=self.tol), min=0.0, max=cfg.MAX_EXPONENT)
        T = torch.clamp(1.0 + eta_safe * psi, min=self.tol)
        exp_power = torch.clamp((1.0 - 2.0 / eta_safe) * torch.log(T), min=-cfg.MAX_EXPONENT, max=cfg.MAX_EXPONENT)
        T_power = torch.exp(exp_power)

        I2 = alpha_sq * mu_over_2me * T_power
        I1 = 2.0 * (q_L - z) * (alpha_L - alpha_tilde) + alpha_sq * mu_over_2me * (1.0 - T_power)
        gpd_crps = I1 + I2

        in_tail = z < q_L
        return torch.where(in_tail, gpd_crps, simple_crps)

    def _crps_right_tail(self, z: Tensor, alpha_z: Tensor) -> Tensor:
        """CRPS contribution from right tail :math:`[\\alpha_R, 1]`."""
        if self.tail_type == "exp":
            return self._crps_right_tail_exp(z, alpha_z)
        else:
            return self._crps_right_tail_gpd(z, alpha_z)

    def _crps_right_tail_exp(self, z: Tensor, alpha_z: Tensor) -> Tensor:
        """Exponential right tail CRPS (compact formula)."""
        a = self._expand_to_z(self.tail_a_r, z)
        b = self._expand_to_z(self.tail_b_r, z)
        q_R = self._expand_to_z(self.q_r, z)
        alpha_R = self.alpha_r

        alpha_R_safe = min(alpha_R, 1 - self.tol)
        one_m_R = max(1 - alpha_R_safe, self.tol)
        alpha_tilde = torch.clamp(alpha_z, min=alpha_R_safe, max=1 - self.tol)

        ln_one_m_R = torch.log(torch.tensor(one_m_R, device=z.device, dtype=z.dtype))

        term1 = (z - b) * (-1 - alpha_R_safe**2 + 2 * alpha_tilde)
        term2 = a * (-0.5 * (alpha_R_safe + 1) ** 2 + (alpha_R_safe**2 - 1) * ln_one_m_R + 2 * alpha_tilde)
        term2 = term2 + 2 * torch.where(
            z > q_R,
            (1 - alpha_tilde) * (z - b),
            a * one_m_R * ln_one_m_R,
        )

        return term1 + term2

    def _crps_right_tail_gpd(self, z: Tensor, alpha_z: Tensor) -> Tensor:
        """GPD right tail CRPS."""
        cfg = self.cfg

        eta = self._expand_to_z(self.eta_r, z)
        mu = self._expand_to_z(self.mu_r, z)
        q_R = self._expand_to_z(self.q_r, z)
        alpha_R = self.alpha_r

        is_exp = torch.abs(eta) < cfg.ETA_TOLERANCE
        eta_safe = torch.where(is_exp, torch.full_like(eta, cfg.ETA_TOLERANCE), eta)

        two_m_eta = torch.clamp((2.0 - eta_safe).abs(), min=cfg.ETA_TOLERANCE)
        two_m_eta = torch.where(2.0 - eta_safe >= 0, two_m_eta, -two_m_eta)
        mu_over_2me = torch.clamp(mu / two_m_eta, min=-cfg.MAX_CRPS, max=cfg.MAX_CRPS)

        one_m_R = max(1 - alpha_R, self.tol)
        one_m_sq = one_m_R**2

        simple_crps = one_m_sq * (q_R - z + mu_over_2me)

        alpha_tilde = torch.clamp(alpha_z, min=alpha_R + self.tol, max=1 - self.tol)
        psi = torch.clamp((z - q_R) / torch.clamp(mu, min=self.tol), min=0.0, max=cfg.MAX_EXPONENT)
        T = torch.clamp(1.0 + eta_safe * psi, min=self.tol)
        exp_power = torch.clamp((1.0 - 2.0 / eta_safe) * torch.log(T), min=-cfg.MAX_EXPONENT, max=cfg.MAX_EXPONENT)
        T_power = torch.exp(exp_power)

        I2 = one_m_sq * mu_over_2me * T_power
        I1 = 2.0 * (z - q_R) * (alpha_tilde - alpha_R) - one_m_sq * mu_over_2me * (1.0 - T_power)
        gpd_crps = I1 + I2

        in_tail = z > q_R
        return torch.where(in_tail, gpd_crps, simple_crps)

    def _crps_spline(self, z: Tensor, alpha_z: Tensor) -> Tensor:
        """CRPS contribution from spline region using unified clamp formula."""
        n_extra = z.dim() - len(self._batch_shape)
        seg_dim = len(self._batch_shape)

        # Expand segment data
        alpha_i = self.alpha_lo
        alpha_ip1 = self.alpha_hi
        q_i = self.q_lo
        m = self.slopes

        for _ in range(n_extra):
            alpha_i = alpha_i.unsqueeze(-1)
            alpha_ip1 = alpha_ip1.unsqueeze(-1)
            q_i = q_i.unsqueeze(-1)
            m = m.unsqueeze(-1)

        z_exp = z.unsqueeze(seg_dim)
        alpha_z_exp = alpha_z.unsqueeze(seg_dim)

        # Unified formula via clamp trick
        r = torch.clamp(alpha_z_exp, alpha_i, alpha_ip1)

        r2, r3 = r**2, r**3
        ai2, ai3 = alpha_i**2, alpha_i**3
        aip12, aip13 = alpha_ip1**2, alpha_ip1**3

        # I1 = ∫_{α_i}^r 2α(z-Q)dα
        I1 = (z_exp - q_i) * (r2 - ai2) - 2 * m * (r3 / 3 - alpha_i * r2 / 2 + ai3 / 6)

        # I2 = ∫_r^{α_{i+1}} 2(1-α)(Q-z)dα
        A = q_i - z_exp
        diff_a = alpha_ip1 - r
        diff_a2 = aip12 - r2
        diff_a3 = aip13 - r3

        int_Qmz = A * diff_a + m * (diff_a2 / 2 - alpha_i * diff_a)
        int_aQmz = A * diff_a2 / 2 + m * (diff_a3 / 3 - alpha_i * diff_a2 / 2)

        I2 = 2 * int_Qmz - 2 * int_aQmz

        seg_crps = I1 + I2
        return seg_crps.sum(dim=seg_dim)

    def pinball(self, z: Tensor, num_quantiles: int = 999) -> Tensor:
        """Numerical CRPS via pinball loss (for validation).

        Parameters
        ----------
        z : Tensor
            Observation values.
            Shape: ``(*batch_shape,)``.

        num_quantiles : int, default=999
            Number of quantile levels for numerical integration.

        Returns
        -------
        Tensor
            Approximate CRPS values.
            Shape: ``(*batch_shape,)``.
        """
        device, dtype = z.device, z.dtype
        alphas = torch.linspace(0.0, 1.0, num_quantiles + 2, device=device, dtype=dtype)[1:-1]
        pred_q = self.icdf(alphas)
        z_exp = z.unsqueeze(-1)
        diff = z_exp - pred_q
        loss = torch.where(diff >= 0, alphas * diff, (alphas - 1) * diff)
        return 2 * loss.mean(dim=-1)

    def mean(self) -> Tensor:
        """Compute mean :math:`E[Z]` using analytical tail-corrected formula.

        Returns
        -------
        Tensor
            Expected value.
            Shape: ``(*batch_shape,)``.
        """
        if self.tail_type == "exp":
            return self._mean_exp_analytical()
        else:
            return self._mean_gpd_analytical()

    def _mean_exp_analytical(self) -> Tensor:
        """Analytical mean for exponential tails."""
        # Left tail: alpha_L * (q_L - beta_L)
        left_int = self.alpha_l * (self.q_l - self.tail_a_l)

        # Spline: trapezoid rule
        spline_int = (self.delta_alpha * (self.q_lo + self.q_hi) / 2).sum(dim=-1)

        # Right tail: (1-α_R) * (q_R - a_R) = (1-α_R) * (q_R + β_R)
        right_int = (1 - self.alpha_r) * (self.q_r - self.tail_a_r)

        return left_int + spline_int + right_int

    def _mean_gpd_analytical(self) -> Tensor:
        """Analytical mean for GPD tails."""
        cfg = self.cfg

        eta_l_safe = torch.clamp(self.eta_l, min=cfg.ETA_TOLERANCE, max=1 - cfg.ETA_TOLERANCE)
        eta_r_safe = torch.clamp(self.eta_r, min=cfg.ETA_TOLERANCE, max=1 - cfg.ETA_TOLERANCE)

        # Left: α_L * (q_L - μ/(1-η))
        left_int = self.alpha_l * (self.q_l - self.mu_l / (1 - eta_l_safe))

        # Spline
        spline_int = (self.delta_alpha * (self.q_lo + self.q_hi) / 2).sum(dim=-1)

        # Right: (1-α_R) * (q_R + μ/(1-η))
        right_int = (1 - self.alpha_r) * (self.q_r + self.mu_r / (1 - eta_r_safe))

        return left_int + spline_int + right_int

    def variance(self) -> Tensor:
        """Compute variance :math:`\\text{Var}[Z] = E[Z^2] - E[Z]^2` using analytical tail-corrected formula.

        Returns
        -------
        Tensor
            Variance (non-negative).
            Shape: ``(*batch_shape,)``.
        """
        if self.tail_type == "exp":
            return self._variance_exp_analytical()
        else:
            return self._variance_gpd_analytical()

    def _variance_exp_analytical(self) -> Tensor:
        """Analytical variance for exponential tails."""
        a_l = self.tail_a_l
        a_r = self.tail_a_r

        # E[Z²] left: α_L * (q_L² - 2*a*q_L + 2*a²)
        e_z2_left = self.alpha_l * (self.q_l**2 - 2 * a_l * self.q_l + 2 * a_l**2)

        # E[Z²] spline: ∑ Δα * (q_i² + q_i*q_{i+1} + q_{i+1}²) / 3
        e_z2_spline = (self.delta_alpha * (self.q_lo**2 + self.q_lo * self.q_hi + self.q_hi**2) / 3).sum(dim=-1)

        # E[Z²] right: (1-α_R) * (q_R² - 2*a*q_R + 2*a²)
        e_z2_right = (1 - self.alpha_r) * (self.q_r**2 - 2 * a_r * self.q_r + 2 * a_r**2)

        e_z2 = e_z2_left + e_z2_spline + e_z2_right
        e_z = self._mean_exp_analytical()

        return torch.clamp(e_z2 - e_z**2, min=0.0)

    def _variance_gpd_analytical(self) -> Tensor:
        """Analytical variance for GPD tails."""
        cfg = self.cfg

        eta_l_safe = torch.clamp(self.eta_l, min=cfg.ETA_TOLERANCE, max=0.49)
        eta_r_safe = torch.clamp(self.eta_r, min=cfg.ETA_TOLERANCE, max=0.49)

        # Left tail E[Z²]
        c_l = self.q_l + self.mu_l / eta_l_safe
        d_l = self.mu_l / eta_l_safe
        one_m_eta_l = 1 - eta_l_safe
        one_m_2eta_l = torch.clamp(1 - 2 * eta_l_safe, min=cfg.ETA_TOLERANCE)
        e_z2_left = self.alpha_l * (c_l**2 - 2 * c_l * d_l / one_m_eta_l + d_l**2 / one_m_2eta_l)

        # Spline E[Z²]
        e_z2_spline = (self.delta_alpha * (self.q_lo**2 + self.q_lo * self.q_hi + self.q_hi**2) / 3).sum(dim=-1)

        # Right tail E[Z²]
        c_r = self.q_r - self.mu_r / eta_r_safe
        d_r = self.mu_r / eta_r_safe
        one_m_eta_r = 1 - eta_r_safe
        one_m_2eta_r = torch.clamp(1 - 2 * eta_r_safe, min=cfg.ETA_TOLERANCE)
        e_z2_right = (1 - self.alpha_r) * (c_r**2 + 2 * c_r * d_r / one_m_eta_r + d_r**2 / one_m_2eta_r)

        e_z2 = e_z2_left + e_z2_spline + e_z2_right
        e_z = self._mean_gpd_analytical()

        return torch.clamp(e_z2 - e_z**2, min=0.0)

    def stddev(self) -> Tensor:
        """Compute standard deviation.

        Returns
        -------
        Tensor
            Standard deviation (non-negative).
            Shape: ``(*batch_shape,)``.
        """
        return torch.sqrt(torch.clamp(self.variance(), min=self.tol))

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw samples from the distribution.

        Parameters
        ----------
        sample_shape : torch.Size, default=()
            Shape of the sample.
            Default: ``()`` for a single sample per batch element.

        Returns
        -------
        Tensor
            Samples from the distribution.
            Shape: ``(*sample_shape, *batch_shape)``.
        """
        n_samples = max(1, torch.Size(sample_shape).numel())

        # Generate uniforms and apply inverse CDF: (*batch_shape, n_samples)
        u = torch.rand(*self.batch_shape, n_samples, device=self.quantiles.device, dtype=self.quantiles.dtype)
        q = self.icdf(u)

        if not sample_shape:
            return q.squeeze(-1)  # (*batch_shape,)

        # Reshape: (*batch_shape, n_samples) -> (*sample_shape, *batch_shape)
        n_batch = len(self.batch_shape)
        q = q.view(*self.batch_shape, *sample_shape)
        return q.movedim(tuple(range(n_batch)), tuple(range(-n_batch, 0)))


class QuantileToDistribution(nn.Module):
    """Module wrapper that converts predicted quantiles to QuantileDistribution.

    Use this to wrap your quantile prediction model's output into a proper
    probability distribution with analytical statistics.

    Parameters
    ----------
    alpha_levels : List[float] or Tensor, optional
        The quantile levels corresponding to predictions.
        Default: ``linspace(0.0, 1.0, num_quantiles + 2)[1:-1]``.

    num_quantiles : int, default=999
        Number of quantile levels (used if ``alpha_levels`` not provided).

    tail_type : str, default="exp"
        Type of tail extrapolation: ``"exp"`` or ``"gpd"``.

    fix_crossing : bool, default=True
        Whether to enforce monotonicity.

    crossing_method : str, default="sort"
        Method for fixing crossing: ``"isotonic"``, ``"cummax"``, or ``"sort"``.
    """

    def __init__(
        self,
        alpha_levels: Optional[List[float]] = None,
        num_quantiles: int = 999,
        tail_type: Literal["exp", "gpd"] = "exp",
        fix_crossing: bool = True,
        crossing_method: str = "sort",
    ):
        super().__init__()

        self.tail_type = tail_type
        self.fix_crossing = fix_crossing
        self.crossing_method = crossing_method

        # Register alpha levels as buffer
        if alpha_levels is None:
            self.alpha_levels = torch.linspace(0.0, 1.0, num_quantiles + 2)[1:-1]
        else:
            self.alpha_levels = torch.tensor(alpha_levels)

    def forward(self, quantiles: Tensor) -> QuantileDistribution:
        """Convert predicted quantiles to distribution.

        Parameters
        ----------
        quantiles : Tensor
            Predicted quantiles from your model.
            Shape: ``(*batch_shape, num_quantiles)``.

        Returns
        -------
        QuantileDistribution
            Distribution object with analytical methods.
        """
        return QuantileDistribution(
            quantiles=quantiles,
            alpha_levels=self.alpha_levels.to(quantiles.device, dtype=quantiles.dtype),
            tail_type=self.tail_type,
            fix_crossing=self.fix_crossing,
            crossing_method=self.crossing_method,
        )
