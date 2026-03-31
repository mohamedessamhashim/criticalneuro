"""Surrogate testing for CSD signals.

Validates CSD findings by comparing observed Kendall tau trends against
distributions generated from phase-randomized surrogate time series.
Phase randomization preserves the power spectrum (autocorrelation structure)
while destroying the temporal ordering, following Theiler et al. 1992.
"""

import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.csd.rolling_window import (
    compute_csd_for_participant,
    detrend_series,
    kendall_tau_trend,
    rolling_ar1,
    rolling_variance,
)

logger = logging.getLogger(__name__)


def phase_randomize_series(
    x: np.ndarray, random_state: np.random.RandomState
) -> np.ndarray:
    """Generate a phase-randomized surrogate time series.

    Preserves the power spectrum while destroying temporal ordering.
    NaN positions in the original are preserved in the surrogate.

    Parameters
    ----------
    x : np.ndarray
        1D time series (may contain NaN).
    random_state : np.random.RandomState
        Random state for reproducibility.

    Returns
    -------
    np.ndarray
        Phase-randomized surrogate of the same length.
    """
    # Record and handle NaN positions
    nan_mask = np.isnan(x)
    x_filled = x.copy()
    if nan_mask.any():
        # Fill NaN with interpolated values for FFT
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) < 2:
            return x.copy()
        x_filled[nan_mask] = np.interp(
            np.where(nan_mask)[0], valid_idx, x[valid_idx]
        )

    n = len(x_filled)
    fft_coeffs = np.fft.rfft(x_filled)

    # Generate random phases
    n_freqs = len(fft_coeffs)
    random_phases = random_state.uniform(0, 2 * np.pi, n_freqs)

    # DC component (index 0) keeps its phase
    random_phases[0] = 0.0
    # Nyquist component keeps its phase for even-length series
    if n % 2 == 0:
        random_phases[-1] = 0.0

    # Apply phase rotation
    fft_surrogate = fft_coeffs * np.exp(1j * random_phases)

    # Inverse FFT
    surrogate = np.fft.irfft(fft_surrogate, n=n).real

    # Restore NaN positions
    surrogate[nan_mask] = np.nan

    return surrogate


def compute_surrogate_kendall_tau(
    x: np.ndarray,
    n_surrogates: int,
    window: int,
    detrend_method: str,
    random_seed: int,
) -> dict:
    """Compare observed CSD trends against surrogate distributions.

    Parameters
    ----------
    x : np.ndarray
        Original time series.
    n_surrogates : int
        Number of surrogates to generate.
    window : int
        Rolling window size.
    detrend_method : str
        Detrending method.
    random_seed : int
        Base random seed.

    Returns
    -------
    dict
        Observed taus, surrogate tau distributions, and empirical p-values.
    """
    # Compute observed CSD
    observed = compute_csd_for_participant(x, window, detrend_method)
    obs_var_tau = observed["var_tau"]
    obs_ar1_tau = observed["ar1_tau"]

    # Generate surrogates and compute taus
    surrogate_var_taus = np.full(n_surrogates, np.nan)
    surrogate_ar1_taus = np.full(n_surrogates, np.nan)

    for i in range(n_surrogates):
        rng = np.random.RandomState(random_seed + i)
        surrogate = phase_randomize_series(x, rng)
        surr_csd = compute_csd_for_participant(surrogate, window, detrend_method)
        surrogate_var_taus[i] = surr_csd["var_tau"]
        surrogate_ar1_taus[i] = surr_csd["ar1_tau"]

    # Empirical p-values (one-tailed: observed >= surrogates)
    # +1 correction to avoid p=0
    if np.isfinite(obs_var_tau):
        n_geq_var = np.sum(surrogate_var_taus[np.isfinite(surrogate_var_taus)] >= obs_var_tau)
        n_valid_var = np.isfinite(surrogate_var_taus).sum()
        empirical_p_var = (n_geq_var + 1) / (n_valid_var + 1)
    else:
        empirical_p_var = np.nan

    if np.isfinite(obs_ar1_tau):
        n_geq_ar1 = np.sum(surrogate_ar1_taus[np.isfinite(surrogate_ar1_taus)] >= obs_ar1_tau)
        n_valid_ar1 = np.isfinite(surrogate_ar1_taus).sum()
        empirical_p_ar1 = (n_geq_ar1 + 1) / (n_valid_ar1 + 1)
    else:
        empirical_p_ar1 = np.nan

    return {
        "observed_var_tau": obs_var_tau,
        "observed_ar1_tau": obs_ar1_tau,
        "surrogate_var_taus": surrogate_var_taus,
        "surrogate_ar1_taus": surrogate_ar1_taus,
        "empirical_p_var": empirical_p_var,
        "empirical_p_ar1": empirical_p_ar1,
    }


def run_surrogate_validation(
    df: pd.DataFrame,
    seq_cols: list[str],
    significant_proteins: list[str],
    config: dict,
) -> pd.DataFrame:
    """Validate CSD signals using surrogate testing across converter participants.

    For each significant protein, tests whether the observed CSD trend is
    stronger than expected from phase-randomized surrogates. A protein is
    "robustly CSD-detected" if >= 50% of converters show a validated signal.

    Parameters
    ----------
    df : pd.DataFrame
        Processed proteomics DataFrame.
    seq_cols : list[str]
        All protein column names.
    significant_proteins : list[str]
        Proteins that passed FDR correction.
    config : dict
        Configuration with csd section.

    Returns
    -------
    pd.DataFrame
        Per-protein validation results.
    """
    csd_cfg = config["csd"]
    adni_cfg = config.get("adni", config.get("ppmi", {}))
    converter_label = adni_cfg.get("converter_group", adni_cfg.get("fast_progressor_group"))

    window = csd_cfg["primary_window"]
    detrend_method = csd_cfg["primary_detrending"]
    n_surrogates = csd_cfg["n_surrogates"]
    alpha = csd_cfg["alpha"]
    random_seed = config["random_seed"]

    # Filter to converter participants
    converters = df[df["TRAJECTORY"] == converter_label]
    converter_rids = converters["RID"].unique()

    logger.info(
        "Surrogate validation: %d proteins × %d converters × %d surrogates",
        len(significant_proteins),
        len(converter_rids),
        n_surrogates,
    )

    results = []
    for protein in tqdm(significant_proteins, desc="Surrogate validation"):
        n_tested = 0
        n_validated = 0

        for rid in converter_rids:
            participant_df = converters[converters["RID"] == rid].sort_values("EXAMDATE")
            series = participant_df[protein].values.astype(float)

            if np.isfinite(series).sum() < csd_cfg["primary_window"] + 1:
                continue

            n_tested += 1
            surr_result = compute_surrogate_kendall_tau(
                series, n_surrogates, window, detrend_method, random_seed + n_tested
            )

            # Validated if empirical p < alpha for variance
            if surr_result["empirical_p_var"] < alpha:
                n_validated += 1

        fraction = n_validated / n_tested if n_tested > 0 else 0.0
        results.append(
            {
                "protein": protein,
                "n_converters_tested": n_tested,
                "n_validated": n_validated,
                "fraction_validated": fraction,
                "robust": fraction >= 0.50,
            }
        )

    results_df = pd.DataFrame(results)
    n_robust = results_df["robust"].sum() if len(results_df) > 0 else 0
    logger.info(
        "Surrogate validation complete: %d/%d proteins robustly validated",
        n_robust,
        len(results_df),
    )
    return results_df
