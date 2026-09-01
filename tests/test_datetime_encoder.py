"""Tests for DatetimeEncoder sin/cos encoding (GitHub issue #136)."""

import numpy as np
import pandas as pd
import pytest
import statsmodels  # noqa: F401

from tabicl.forecast.transforms._calendar import DatetimeEncoder


@pytest.fixture
def one_week_df():
    """DataFrame spanning one full week (Mon-Sun)."""
    dates = pd.date_range("2026-01-05", periods=7, freq="D")  # Mon to Sun
    idx = pd.MultiIndex.from_arrays([dates], names=["timestamp"])
    return pd.DataFrame({"value": range(7)}, index=idx)


@pytest.fixture
def one_year_df():
    """DataFrame spanning one full year of daily data."""
    dates = pd.date_range("2026-01-01", periods=365, freq="D")
    idx = pd.MultiIndex.from_arrays([dates], names=["timestamp"])
    return pd.DataFrame({"value": range(365)}, index=idx)


class TestDatetimeEncoderNoDuplicateAngles:
    """Sin/cos pairs must be distinct for every element within a period."""

    def test_day_of_week_all_distinct(self, one_week_df):
        enc = DatetimeEncoder(
            components=[],
            seasonal_features={"day_of_week": [7]},
        )
        result = enc.generate(one_week_df)
        sin_cos = result[["day_of_week_sin", "day_of_week_cos"]].values
        # All 7 (sin, cos) pairs must be unique
        unique_pairs = set(map(tuple, np.round(sin_cos, 10)))
        assert len(unique_pairs) == 7

    def test_hour_of_day_all_distinct(self):
        dates = pd.date_range("2026-01-05", periods=24, freq="h")
        idx = pd.MultiIndex.from_arrays([dates], names=["timestamp"])
        df = pd.DataFrame({"value": range(24)}, index=idx)
        enc = DatetimeEncoder(
            components=[],
            seasonal_features={"hour_of_day": [24]},
        )
        result = enc.generate(df)
        sin_cos = result[["hour_of_day_sin", "hour_of_day_cos"]].values
        unique_pairs = set(map(tuple, np.round(sin_cos, 10)))
        assert len(unique_pairs) == 24

    def test_month_of_year_all_distinct(self):
        dates = pd.date_range("2026-01-15", periods=12, freq="ME")
        idx = pd.MultiIndex.from_arrays([dates], names=["timestamp"])
        df = pd.DataFrame({"value": range(12)}, index=idx)
        enc = DatetimeEncoder(
            components=[],
            seasonal_features={"month_of_year": [12]},
        )
        result = enc.generate(df)
        sin_cos = result[["month_of_year_sin", "month_of_year_cos"]].values
        unique_pairs = set(map(tuple, np.round(sin_cos, 10)))
        assert len(unique_pairs) == 12

    def test_first_and_last_day_of_week_differ(self, one_week_df):
        """Specific regression: Monday and Sunday must NOT collide."""
        enc = DatetimeEncoder(
            components=[],
            seasonal_features={"day_of_week": [7]},
        )
        result = enc.generate(one_week_df)
        monday = result.iloc[0][["day_of_week_sin", "day_of_week_cos"]].values
        sunday = result.iloc[6][["day_of_week_sin", "day_of_week_cos"]].values
        assert not np.allclose(monday, sunday), (
            f"Monday and Sunday have identical encodings: {monday}"
        )

    def test_encoding_values_correct(self, one_week_df):
        """Verify the formula: angle = 2*pi*feature/period."""
        enc = DatetimeEncoder(
            components=[],
            seasonal_features={"day_of_week": [7]},
        )
        result = enc.generate(one_week_df)
        for i in range(7):
            expected_sin = np.sin(2 * np.pi * i / 7)
            expected_cos = np.cos(2 * np.pi * i / 7)
            actual_sin = result.iloc[i]["day_of_week_sin"]
            actual_cos = result.iloc[i]["day_of_week_cos"]
            assert np.isclose(actual_sin, expected_sin), f"day {i}: sin mismatch"
            assert np.isclose(actual_cos, expected_cos), f"day {i}: cos mismatch"

    def test_leap_year_day_of_year_no_collision(self):
        """First and last day of leap year must NOT collide with default encoder.

        Regression test for GitHub issue #136: with day_of_year default of 366,
        Jan 1 (day 0) and Dec 31 (day 365) in a leap year must have distinct encodings.
        """
        # 2024 is a leap year (366 days)
        dates = pd.to_datetime(["2024-01-01", "2024-12-31"])
        idx = pd.MultiIndex.from_arrays([dates], names=["timestamp"])
        df = pd.DataFrame({"value": [0, 365]}, index=idx)

        # Use default DatetimeEncoder to test the default day_of_year=366
        enc = DatetimeEncoder(components=[])
        result = enc.generate(df)

        day_1 = result.iloc[0][["day_of_year_sin", "day_of_year_cos"]].values
        day_366 = result.iloc[1][["day_of_year_sin", "day_of_year_cos"]].values
        assert not np.allclose(day_1, day_366, atol=1e-10), (
            f"Day 1 and day 366 have identical encodings: {day_1}"
        )

    def test_iso_week_53_no_collision(self):
        """First and last ISO week must NOT collide with default encoder.

        Regression test for GitHub issue #136: with week_of_year default of 53,
        ISO week 1 (week 0) and week 53 (week 52) in a week-53 year must have distinct encodings.
        """
        # 2020 has ISO week 53
        # 2019-12-30 is Monday of ISO week 1 of 2020
        # 2020-12-28 is Monday of ISO week 53 of 2020
        dates = pd.to_datetime(["2019-12-30", "2020-12-28"])
        idx = pd.MultiIndex.from_arrays([dates], names=["timestamp"])
        df = pd.DataFrame({"value": range(len(dates))}, index=idx)

        # Use default DatetimeEncoder to test the default week_of_year=53
        enc = DatetimeEncoder(components=[])
        result = enc.generate(df)

        week_1 = result.iloc[0][["week_of_year_sin", "week_of_year_cos"]].values
        week_53 = result.iloc[1][["week_of_year_sin", "week_of_year_cos"]].values
        assert not np.allclose(week_1, week_53, atol=1e-10), (
            f"Week 1 and week 53 have identical encodings: {week_1}"
        )
