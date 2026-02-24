from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import fft
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf

from tabicl.forecast.transforms.base import TimeTransform
from tabicl.forecast.transforms.calendar import FourierEncoder


logger = logging.getLogger(__name__)


@dataclass
class PeriodicDetectionConfig:
    """Configuration for automatic periodicity detection via FFT.

    Parameters
    ----------
    max_top_k : int
        Maximum number of dominant periods to detect.

    do_detrend : bool
        Whether to remove trend before FFT.

    detrend_type : {"first_diff", "loess", "linear", "constant"}
        Detrending method.

    use_peaks_only : bool
        Whether to consider only local peaks in the FFT spectrum.

    apply_hann_window : bool
        Whether to apply a Hann window to reduce spectral leakage.

    zero_padding_factor : int
        Factor by which to zero-pad the signal for finer frequency
        resolution.

    round_to_closest_integer : bool
        Whether to round detected periods to the nearest integer.

    validate_with_acf : bool
        Whether to validate detected periods against autocorrelation.

    sampling_interval : float
        Time interval between consecutive samples.

    magnitude_threshold : float | None
        Threshold to filter out less significant frequency components.

    relative_threshold : bool
        Whether ``magnitude_threshold`` is relative to the maximum FFT
        magnitude.

    exclude_zero : bool
        Whether to exclude periods of 0 from the results.
    """

    max_top_k: int = 5
    do_detrend: bool = True
    detrend_type: Literal["first_diff", "loess", "linear", "constant"] = "linear"
    use_peaks_only: bool = True
    apply_hann_window: bool = True
    zero_padding_factor: int = 2
    round_to_closest_integer: bool = True
    validate_with_acf: bool = False
    sampling_interval: float = 1.0
    magnitude_threshold: float | None = 0.05
    relative_threshold: bool = True
    exclude_zero: bool = True


def _remove_trend(
    x: np.ndarray,
    detrend_type: Literal["first_diff", "loess", "linear", "constant"],
) -> np.ndarray:
    """Remove trend from a time series signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal.

    detrend_type : {"first_diff", "loess", "linear", "constant"}
        Detrending method:

        - ``"first_diff"``: First-order differencing.
        - ``"loess"``: Local regression (LOWESS) trend removal.
        - ``"linear"``: Linear trend removal via polynomial fit.
        - ``"constant"``: Mean subtraction.

    Returns
    -------
    np.ndarray
        Detrended signal.

    Raises
    ------
    ValueError
        If ``detrend_type`` is not a recognized method.
    """
    if detrend_type == "first_diff":
        return np.diff(x, prepend=x[0])

    elif detrend_type == "loess":
        from statsmodels.api import nonparametric

        indices = np.arange(len(x))
        lowess = nonparametric.lowess(x, indices, frac=0.1)
        trend = lowess[:, 1]
        return x - trend

    elif detrend_type == "linear":
        indices = np.arange(len(x))
        coeffs = np.polyfit(indices, x, 1, rcond=None)
        trend = np.polyval(coeffs, indices)
        return x - trend

    elif detrend_type == "constant":
        return x - np.mean(x)

    else:
        raise ValueError(f"Invalid detrend method: {detrend_type}")


def _prepare_signal(
    values: np.ndarray,
    do_detrend: bool,
    detrend_type: str,
    apply_hann_window: bool,
) -> tuple[np.ndarray, int]:
    """Remove NaNs, optionally detrend, and apply windowing.

    Returns the processed signal and its original (pre-padding) length.
    """
    values = values[~np.isnan(values)]
    n_original = len(values)

    if do_detrend:
        values = _remove_trend(values, detrend_type)

    if apply_hann_window:
        window = np.hanning(n_original)
        values = values * window

    return values, n_original


def _compute_spectrum(
    values: np.ndarray,
    n_original: int,
    zero_padding_factor: int,
    sampling_interval: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Zero-pad, compute FFT, and return magnitudes + frequency bins."""
    if zero_padding_factor > 1:
        padded_length = int(n_original * zero_padding_factor)
        padded_values = np.zeros(padded_length)
        padded_values[:n_original] = values
        values = padded_values
        n = padded_length
    else:
        n = n_original

    fft_values = fft.rfft(values)
    fft_magnitudes = np.abs(fft_values)
    freqs = np.fft.rfftfreq(n, d=sampling_interval)

    # Exclude the DC component (0 Hz)
    fft_magnitudes[0] = 0.0

    return fft_magnitudes, freqs, n


def _extract_dominant_periods(
    fft_magnitudes: np.ndarray,
    freqs: np.ndarray,
    max_top_k: int,
    use_peaks_only: bool,
    magnitude_threshold: float | None,
    relative_threshold: bool,
    round_to_closest_integer: bool,
    exclude_zero: bool,
) -> list[tuple[float, float]]:
    """Find peaks, threshold, round, deduplicate, and return (period, magnitude) pairs."""
    # Determine the absolute threshold value
    if magnitude_threshold is not None and relative_threshold:
        threshold_value = magnitude_threshold * np.max(fft_magnitudes)
    else:
        threshold_value = magnitude_threshold

    # Identify dominant frequencies
    if use_peaks_only:
        if threshold_value is not None:
            peak_indices, _ = find_peaks(fft_magnitudes, height=threshold_value)
        else:
            peak_indices, _ = find_peaks(fft_magnitudes)
        if len(peak_indices) == 0:
            peak_indices = np.arange(len(fft_magnitudes))
        sorted_peak_indices = peak_indices[np.argsort(fft_magnitudes[peak_indices])[::-1]]
        top_indices = sorted_peak_indices[:max_top_k]
    else:
        sorted_indices = np.argsort(fft_magnitudes)[::-1]
        if threshold_value is not None:
            sorted_indices = np.array([i for i in sorted_indices if fft_magnitudes[i] >= threshold_value])
        top_indices = sorted_indices[:max_top_k]

    # Convert frequencies to periods
    periods = np.zeros_like(freqs)
    non_zero = freqs > 0
    periods[non_zero] = 1.0 / freqs[non_zero]
    top_periods = periods[top_indices]
    logger.debug(f"Top periods: {top_periods}")

    if round_to_closest_integer:
        top_periods = np.round(top_periods)

    if exclude_zero:
        non_zero_mask = top_periods != 0
        top_periods = top_periods[non_zero_mask]
        top_indices = top_indices[non_zero_mask]

    # Deduplicate
    if len(top_periods) > 0:
        unique_period_indices = np.unique(top_periods, return_index=True)[1]
        top_periods = top_periods[unique_period_indices]
        top_indices = top_indices[unique_period_indices]

    return [(top_periods[i], fft_magnitudes[top_indices[i]]) for i in range(len(top_indices))]


def _validate_with_acf(
    results: list[tuple[float, float]],
    target_values: np.ndarray,
    n_original: int,
) -> list[tuple[float, float]]:
    """Cross-check detected periods against autocorrelation peaks."""
    acf_values = acf(target_values[:n_original], nlags=n_original, fft=True)
    acf_peak_indices, _ = find_peaks(acf_values, height=1.96 / np.sqrt(n_original))
    validated = []
    for period, mag in results:
        period_int = int(round(period))
        if period_int < len(acf_values) and any(abs(period_int - peak) <= 1 for peak in acf_peak_indices):
            validated.append((period, mag))

    return validated if validated else results


def detect_periodicities(
    target_values: pd.Series,
    max_top_k: int = 10,
    do_detrend: bool = True,
    detrend_type: Literal["first_diff", "loess", "linear", "constant"] = "first_diff",
    use_peaks_only: bool = True,
    apply_hann_window: bool = True,
    zero_padding_factor: int = 2,
    round_to_closest_integer: bool = True,
    validate_with_acf_flag: bool = False,
    sampling_interval: float = 1.0,
    magnitude_threshold: float | None = 0.05,
    relative_threshold: bool = True,
    exclude_zero: bool = False,
) -> list[tuple[float, float]]:
    """Identify dominant seasonal periods in a time series using FFT.

    Parameters
    ----------
    target_values : pd.Series
        Input time series data.

    max_top_k : int, default=10
        Maximum number of dominant periods to return.

    do_detrend : bool, default=True
        If ``True``, remove the trend from the signal before FFT.

    detrend_type : {"first_diff", "loess", "linear", "constant"}, default="first_diff"
        Detrending method to apply.

    use_peaks_only : bool, default=True
        If ``True``, consider only local peaks in the FFT magnitude
        spectrum.

    apply_hann_window : bool, default=True
        If ``True``, apply a Hann window to reduce spectral leakage.

    zero_padding_factor : int, default=2
        Factor by which to zero-pad the signal for finer frequency
        resolution.

    round_to_closest_integer : bool, default=True
        If ``True``, round detected periods to the nearest integer.

    validate_with_acf_flag : bool, default=False
        If ``True``, validate detected periods against the
        autocorrelation function.

    sampling_interval : float, default=1.0
        Time interval between consecutive samples.

    magnitude_threshold : float | None, default=0.05
        Threshold to filter out less significant frequency components.
        Interpreted as a fraction of the maximum FFT magnitude when
        ``relative_threshold`` is ``True``.

    relative_threshold : bool, default=True
        If ``True``, ``magnitude_threshold`` is interpreted as a fraction
        of the maximum FFT magnitude. Otherwise, treated as an absolute
        value.

    exclude_zero : bool, default=False
        If ``True``, exclude periods of 0 from the results.

    Returns
    -------
    list[tuple[float, float]]
        List of ``(period, magnitude)`` tuples, sorted in descending
        order by magnitude.
    """

    raw_values = np.array(target_values, dtype=float)
    signal, n_original = _prepare_signal(raw_values, do_detrend, detrend_type, apply_hann_window)
    fft_magnitudes, freqs, _ = _compute_spectrum(signal, n_original, zero_padding_factor, sampling_interval)

    results = _extract_dominant_periods(
        fft_magnitudes,
        freqs,
        max_top_k,
        use_peaks_only,
        magnitude_threshold,
        relative_threshold,
        round_to_closest_integer,
        exclude_zero,
    )

    if validate_with_acf_flag:
        results = _validate_with_acf(results, raw_values, n_original)

    results.sort(key=lambda x: x[1], reverse=True)
    return results


class AutoPeriodicEncoder(TimeTransform):
    """Transform that automatically detects and encodes seasonal periods.

    Uses FFT-based spectral analysis to identify dominant seasonal periods
    in the target time series, then generates sin/cosine features for each
    detected period.

    Parameters
    ----------
    config : PeriodicDetectionConfig | dict | None, default=None
        Configuration for periodicity detection. Accepts a dataclass instance,
        a dict of overrides, or ``None`` for defaults.
    """

    def __init__(self, config: PeriodicDetectionConfig | dict | None = None):
        if config is None:
            self.config = PeriodicDetectionConfig()
        elif isinstance(config, dict):
            self.config = PeriodicDetectionConfig(**config)
        else:
            self.config = config

        self._validate_config()
        logger.debug(f"Initialized AutoPeriodicEncoder with config: {self.config}")

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.max_top_k < 1:
            logger.warning("max_top_k must be at least 1, setting to 1")
            self.config.max_top_k = 1

        if self.config.zero_padding_factor < 1:
            logger.warning("zero_padding_factor must be at least 1, setting to 1")
            self.config.zero_padding_factor = 1

        if self.config.detrend_type not in ("first_diff", "loess", "linear", "constant"):
            logger.warning(f"Invalid detrend_type: {self.config.detrend_type}, using 'linear'")
            self.config.detrend_type = "linear"

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate sin/cosine features for automatically detected seasonal periods.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with a ``target`` column.

        Returns
        -------
        pd.DataFrame
            DataFrame with added ``sin_#i`` and ``cos_#i`` columns for each
            detected period, plus zero-filled placeholders up to
            ``max_top_k``.
        """
        df = df.copy()

        cfg = self.config
        detected = detect_periodicities(
            df.target,
            max_top_k=cfg.max_top_k,
            do_detrend=cfg.do_detrend,
            detrend_type=cfg.detrend_type,
            use_peaks_only=cfg.use_peaks_only,
            apply_hann_window=cfg.apply_hann_window,
            zero_padding_factor=cfg.zero_padding_factor,
            round_to_closest_integer=cfg.round_to_closest_integer,
            validate_with_acf_flag=cfg.validate_with_acf,
            sampling_interval=cfg.sampling_interval,
            magnitude_threshold=cfg.magnitude_threshold,
            relative_threshold=cfg.relative_threshold,
            exclude_zero=cfg.exclude_zero,
        )

        logger.debug(f"Found {len(detected)} seasonal periods: {detected}")

        periods = [period for period, _ in detected]

        if periods:
            encoder = FourierEncoder(periods=periods)
            df = encoder.generate(df)

        # Standardize column names for consistency across time series
        renamed_columns = {}
        for i, period in enumerate(periods):
            renamed_columns[f"sin_{period}"] = f"sin_#{i}"
            renamed_columns[f"cos_{period}"] = f"cos_#{i}"

        df = df.rename(columns=renamed_columns)

        # Add placeholder zero columns for missing periods up to max_top_k
        for i in range(len(periods), cfg.max_top_k):
            df[f"sin_#{i}"] = 0.0
            df[f"cos_#{i}"] = 0.0

        return df
