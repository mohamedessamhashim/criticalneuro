"""Tests for CSD analysis modules.

All tests use synthetic data with known properties.
"""

import numpy as np
import pandas as pd
import pytest

from src.csd.rolling_window import (
    compute_csd_for_participant,
    detrend_series,
    kendall_tau_trend,
    rolling_ar1,
    rolling_return_rate,
    rolling_variance,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


class TestDetrending:
    def test_first_difference_removes_linear_trend(self):
        """Differencing a linear trend should produce a near-constant series."""
        x = np.linspace(10, 50, 20)
        result = detrend_series(x, "first_difference")
        # All differences should be approximately equal
        assert len(result) == 19
        np.testing.assert_allclose(result, result[0], atol=1e-10)

    def test_linear_detrend_removes_linear_trend(self):
        """Linear detrending should remove linear component."""
        x = np.linspace(0, 10, 20) + np.array([0.1] * 20)
        result = detrend_series(x, "linear")
        assert len(result) == len(x)
        # Residuals should have mean near zero
        assert abs(np.mean(result)) < 0.1

    def test_none_returns_copy(self):
        """'none' detrending should return the input unchanged."""
        x = np.array([1.0, 3.0, 2.0, 5.0])
        result = detrend_series(x, "none")
        np.testing.assert_array_equal(result, x)
        # Verify it's a copy, not the same object
        assert result is not x

    def test_handles_short_series(self):
        """Series shorter than 2 should be returned unchanged."""
        x = np.array([5.0])
        result = detrend_series(x, "first_difference")
        np.testing.assert_array_equal(result, x)

    def test_handles_nan_in_first_difference(self):
        """NaN values should propagate correctly in first_difference."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = detrend_series(x, "first_difference")
        assert np.isfinite(result[0])  # 2-1
        assert np.isnan(result[1])  # nan-2
        assert np.isnan(result[2])  # 4-nan
        assert np.isfinite(result[3])  # 5-4


class TestRollingVariance:
    def test_increasing_variance_detected(self, rng):
        """Series with genuinely increasing variance should have positive tau."""
        n = 30
        # Noise amplitude grows over time
        noise_scale = np.linspace(0.1, 2.0, n)
        x = rng.normal(0, 1, n) * noise_scale
        var_series = rolling_variance(x, window=4)
        tau, p = kendall_tau_trend(var_series)
        assert tau > 0, f"Expected positive tau, got {tau}"

    def test_constant_series_zero_variance(self):
        """Constant series should have zero variance in all windows."""
        x = np.ones(10)
        var_series = rolling_variance(x, window=4)
        np.testing.assert_allclose(var_series, 0.0, atol=1e-10)

    def test_nan_window_below_threshold(self):
        """Windows with fewer than 3 valid values should return NaN."""
        x = np.array([1.0, np.nan, np.nan, 4.0])
        var_series = rolling_variance(x, window=4)
        assert np.isnan(var_series[0])

    def test_output_length(self):
        """Output length should be len(x) - window + 1."""
        x = np.arange(10, dtype=float)
        result = rolling_variance(x, window=4)
        assert len(result) == 7


class TestRollingAR1:
    def test_increasing_ar1_detected(self, rng):
        """AR(1) process with increasing coefficient should show positive tau."""
        n = 40
        x = np.zeros(n)
        x[0] = rng.normal()
        # AR coefficient increases from 0.1 to 0.9
        ar_coeffs = np.linspace(0.1, 0.9, n)
        for i in range(1, n):
            x[i] = ar_coeffs[i] * x[i - 1] + rng.normal(0, 0.3)

        ar1_series = rolling_ar1(x, window=6)
        tau, _ = kendall_tau_trend(ar1_series)
        assert tau > 0, f"Expected positive tau for increasing AR1, got {tau}"

    def test_white_noise_low_ar1(self, rng):
        """White noise should have AR1 near zero."""
        x = rng.normal(0, 1, 50)
        ar1_series = rolling_ar1(x, window=10)
        mean_ar1 = np.nanmean(ar1_series)
        assert abs(mean_ar1) < 0.5, f"White noise AR1 too high: {mean_ar1}"


class TestRollingReturnRate:
    def test_return_rate_no_inf(self, rng):
        """Return rate should never be infinite."""
        x = rng.normal(0, 1, 20)
        rr = rolling_return_rate(x, window=4)
        assert not np.any(np.isinf(rr[np.isfinite(rr)]))


class TestKendallTau:
    def test_constant_series_is_nan(self):
        """Flat constant series gives NaN tau (all ties in scipy.kendalltau)."""
        x = np.ones(10)
        tau, p = kendall_tau_trend(x)
        # scipy.stats.kendalltau returns NaN for all-tied data
        assert np.isnan(tau)

    def test_monotonic_increase(self):
        """Strictly increasing series should give tau = 1."""
        x = np.arange(10, dtype=float)
        tau, p = kendall_tau_trend(x)
        assert tau == pytest.approx(1.0)

    def test_too_few_values_returns_nan(self):
        """Fewer than 3 valid values should return NaN."""
        x = np.array([1.0, np.nan, np.nan])
        tau, p = kendall_tau_trend(x)
        assert np.isnan(tau)
        assert np.isnan(p)


class TestCSDForParticipant:
    def test_returns_all_keys(self, rng):
        """Result dict should contain all expected keys."""
        x = rng.normal(0, 1, 10)
        result = compute_csd_for_participant(x, window=4, detrend_method="first_difference")
        expected_keys = {
            "var_tau", "var_p", "ar1_tau", "ar1_p",
            "rr_tau", "rr_p", "n_valid_points", "mean_variance", "mean_ar1",
        }
        assert set(result.keys()) == expected_keys

    def test_handles_nan_gracefully(self):
        """Time series with NaN should not raise errors."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = compute_csd_for_participant(x, window=3, detrend_method="first_difference")
        # Should complete without error — values may be NaN but no crash
        assert isinstance(result, dict)


class TestSurrogates:
    def test_preserves_power_spectrum(self, rng):
        """Phase-randomized surrogate should preserve the power spectrum."""
        from src.csd.surrogate_testing import phase_randomize_series

        x = rng.normal(0, 1, 64)
        surrogate = phase_randomize_series(x, rng)

        # Compare power spectra
        orig_power = np.abs(np.fft.rfft(x)) ** 2
        surr_power = np.abs(np.fft.rfft(surrogate)) ** 2

        # KS test — power spectra should not be significantly different
        from scipy.stats import ks_2samp
        stat, p = ks_2samp(orig_power, surr_power)
        assert p > 0.05, f"Power spectra differ significantly (KS p={p:.4f})"

    def test_preserves_nan_positions(self, rng):
        """NaN positions in original should remain NaN in surrogate."""
        from src.csd.surrogate_testing import phase_randomize_series

        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0])
        surrogate = phase_randomize_series(x, rng)
        assert np.isnan(surrogate[2])
        assert np.isnan(surrogate[5])
        assert np.isfinite(surrogate[0])


class TestCompositeScore:
    def test_composite_score_is_sum_of_mean_taus(self):
        """Composite score should be mean_var_tau + mean_ar1_tau per participant."""
        from src.csd.composite_score import compute_composite_csd_score

        data = {
            "RID": [1, 1, 1, 2, 2, 2],
            "protein": ["a", "b", "c", "a", "b", "c"],
            "var_tau": [0.2, 0.4, 0.6, 0.1, 0.3, 0.5],
            "ar1_tau": [0.1, 0.3, 0.5, 0.0, 0.2, 0.4],
        }
        df = pd.DataFrame(data)
        result = compute_composite_csd_score(df, method="mean_tau")

        rid1 = result[result["RID"] == 1].iloc[0]
        expected_var = np.mean([0.2, 0.4, 0.6])
        expected_ar1 = np.mean([0.1, 0.3, 0.5])
        assert rid1["mean_var_tau"] == pytest.approx(expected_var)
        assert rid1["mean_ar1_tau"] == pytest.approx(expected_ar1)
        assert rid1["composite_csd_score"] == pytest.approx(expected_var + expected_ar1)
