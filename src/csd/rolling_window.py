"""Rolling-window CSD (Critical Slowing Down) analysis.

Mathematical core of the project. Computes rolling-window variance,
autocorrelation, and return rate on detrended protein time series,
then tests for monotonic trends using Kendall's tau. Group-level
comparisons use Wilcoxon rank-sum with BH-FDR correction.

Every function handles NaN gracefully and is numerically stable.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from src.preprocessing.somascan_qc import identify_seq_columns

logger = logging.getLogger(__name__)


def detrend_series(x: np.ndarray, method: str) -> np.ndarray:
    """Detrend a time series before computing rolling CSD statistics.

    Parameters
    ----------
    x : np.ndarray
        1D time series (may contain NaN).
    method : str
        One of 'first_difference', 'linear', 'loess', 'none'.

    Returns
    -------
    np.ndarray
        Detrended series. For first_difference, length is len(x)-1.
    """
    if len(x) < 2:
        return x.copy()

    if method == "first_difference":
        result = np.full(len(x) - 1, np.nan)
        for i in range(len(x) - 1):
            if np.isfinite(x[i]) and np.isfinite(x[i + 1]):
                result[i] = x[i + 1] - x[i]
        return result

    elif method == "linear":
        valid = np.isfinite(x)
        if valid.sum() < 2:
            return x.copy()
        indices = np.arange(len(x))
        coeffs = np.polyfit(indices[valid], x[valid], 1)
        trend = np.polyval(coeffs, indices)
        result = x.copy()
        result[valid] = x[valid] - trend[valid]
        return result

    elif method == "loess":
        valid = np.isfinite(x)
        if valid.sum() < 4:
            # Fall back to first_difference if too few points for loess
            return detrend_series(x, "first_difference")
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            indices = np.arange(len(x))
            smoothed = lowess(
                x[valid], indices[valid], frac=0.5, return_sorted=False
            )
            result = x.copy()
            # Interpolate smoothed values at all valid positions
            full_smooth = np.interp(indices, indices[valid], smoothed)
            result[valid] = x[valid] - full_smooth[valid]
            return result
        except ImportError:
            logger.warning(
                "statsmodels not available for loess, falling back to linear"
            )
            return detrend_series(x, "linear")

    elif method == "none":
        return x.copy()

    else:
        raise ValueError(f"Unknown detrending method: {method}")


def rolling_variance(x: np.ndarray, window: int) -> np.ndarray:
    """Compute variance within each rolling window.

    Parameters
    ----------
    x : np.ndarray
        1D time series (may contain NaN).
    window : int
        Window size.

    Returns
    -------
    np.ndarray
        Rolling variance, length len(x) - window + 1.
        NaN for windows with fewer than 3 non-NaN values.
    """
    n = len(x)
    if n < window:
        return np.array([np.nan])

    n_windows = n - window + 1
    result = np.full(n_windows, np.nan)

    for i in range(n_windows):
        window_slice = x[i : i + window]
        valid = window_slice[np.isfinite(window_slice)]
        if len(valid) >= 3:
            result[i] = np.var(valid, ddof=1)

    return result


def rolling_ar1(x: np.ndarray, window: int) -> np.ndarray:
    """Compute lag-1 autocorrelation within each rolling window.

    Pearson correlation between consecutive pairs within each window.

    Parameters
    ----------
    x : np.ndarray
        1D time series (may contain NaN).
    window : int
        Window size.

    Returns
    -------
    np.ndarray
        Rolling AR1, length len(x) - window + 1.
        NaN for windows with fewer than 3 valid consecutive pairs.
    """
    n = len(x)
    if n < window:
        return np.array([np.nan])

    n_windows = n - window + 1
    result = np.full(n_windows, np.nan)

    for i in range(n_windows):
        window_slice = x[i : i + window]
        # Extract valid consecutive pairs
        x1 = window_slice[:-1]
        x2 = window_slice[1:]
        valid = np.isfinite(x1) & np.isfinite(x2)

        if valid.sum() >= 3:
            x1_valid = x1[valid]
            x2_valid = x2[valid]
            # Handle constant series (std = 0)
            if np.std(x1_valid) > 0 and np.std(x2_valid) > 0:
                result[i] = np.corrcoef(x1_valid, x2_valid)[0, 1]
            else:
                result[i] = 0.0

    return result


def rolling_return_rate(x: np.ndarray, window: int) -> np.ndarray:
    """Compute return rate from rolling AR1.

    Return rate = -log(|AR1|). Approaches zero as the system loses resilience.

    Parameters
    ----------
    x : np.ndarray
        1D time series.
    window : int
        Window size.

    Returns
    -------
    np.ndarray
        Rolling return rate.
    """
    ar1 = rolling_ar1(x, window)
    # Clip |AR1| to avoid log(0)
    abs_ar1 = np.clip(np.abs(ar1), 1e-10, None)
    result = -np.log(abs_ar1)
    # Preserve NaN from ar1
    result[np.isnan(ar1)] = np.nan
    return result


def kendall_tau_trend(indicator_series: np.ndarray) -> tuple[float, float]:
    """Compute Kendall's rank correlation between indicator and time indices.

    Parameters
    ----------
    indicator_series : np.ndarray
        1D array of indicator values over time.

    Returns
    -------
    tuple[float, float]
        (tau, p_value). Returns (nan, nan) if fewer than 3 valid values.
    """
    valid_mask = np.isfinite(indicator_series)
    if valid_mask.sum() < 3:
        return (np.nan, np.nan)

    valid_values = indicator_series[valid_mask]
    time_indices = np.arange(len(indicator_series))[valid_mask]

    tau, p = stats.kendalltau(time_indices, valid_values)
    return (tau, p)


def compute_csd_for_participant(
    protein_series: np.ndarray,
    window: int,
    detrend_method: str,
) -> dict:
    """Compute complete CSD profile for one protein's time series.

    Parameters
    ----------
    protein_series : np.ndarray
        1D protein values over visits for one participant.
    window : int
        Rolling window size.
    detrend_method : str
        Detrending method.

    Returns
    -------
    dict
        CSD statistics: var_tau, var_p, ar1_tau, ar1_p, rr_tau, rr_p,
        n_valid_points, mean_variance, mean_ar1.
    """
    detrended = detrend_series(protein_series, detrend_method)

    var_series = rolling_variance(detrended, window)
    ar1_series = rolling_ar1(detrended, window)
    rr_series = rolling_return_rate(detrended, window)

    var_tau, var_p = kendall_tau_trend(var_series)
    ar1_tau, ar1_p = kendall_tau_trend(ar1_series)
    rr_tau, rr_p = kendall_tau_trend(rr_series)

    return {
        "var_tau": var_tau,
        "var_p": var_p,
        "ar1_tau": ar1_tau,
        "ar1_p": ar1_p,
        "rr_tau": rr_tau,
        "rr_p": rr_p,
        "n_valid_points": int(np.isfinite(detrended).sum()),
        "mean_variance": float(np.nanmean(var_series)),
        "mean_ar1": float(np.nanmean(ar1_series)),
    }


def _filter_valid_visits(participant_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Filter visits by inter-visit gap constraints from config.

    Drops visits where the gap to the previous visit is outside the
    valid range [visit_gap_min_months, visit_gap_max_months].

    Parameters
    ----------
    participant_df : pd.DataFrame
        Single participant's visits, sorted by EXAMDATE.
    config : dict
        Configuration with adni/ppmi section containing gap params.

    Returns
    -------
    pd.DataFrame
        Filtered visits.
    """
    cohort_cfg = config.get("adni", config.get("ppmi", {}))
    gap_min = cohort_cfg.get("visit_gap_min_months")
    gap_max = cohort_cfg.get("visit_gap_max_months")

    if gap_min is None and gap_max is None:
        return participant_df

    if len(participant_df) < 2:
        return participant_df

    dates = participant_df["EXAMDATE"]
    gaps_months = dates.diff().dt.days / 30.44

    # First visit is always kept (gap is NaN)
    valid = gaps_months.isna()
    if gap_min is not None:
        valid = valid | (gaps_months >= gap_min)
    if gap_max is not None:
        valid = valid & (gaps_months.isna() | (gaps_months <= gap_max))

    n_dropped = (~valid).sum()
    if n_dropped > 0:
        logger.debug(
            "RID %s: dropped %d visits with invalid inter-visit gaps",
            participant_df["RID"].iloc[0],
            n_dropped,
        )

    return participant_df.loc[valid]


def compute_csd_all_proteins(
    df: pd.DataFrame,
    seq_cols: list[str],
    config: dict,
) -> pd.DataFrame:
    """Compute CSD statistics for all proteins and participants.

    Iterates over participants then proteins. Only includes participants
    with at least min_visits_longitudinal visits. Validates inter-visit
    gaps using visit_gap_min_months and visit_gap_max_months from config.

    Parameters
    ----------
    df : pd.DataFrame
        Processed proteomics DataFrame with RID, VISCODE, EXAMDATE, seq.* columns.
    seq_cols : list[str]
        Protein column names.
    config : dict
        Configuration with adni/ppmi and csd sections.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with one row per (participant, protein).
    """
    csd_cfg = config["csd"]
    window = csd_cfg["primary_window"]
    detrend_method = csd_cfg["primary_detrending"]

    # Determine min visits from config
    min_visits = config.get("adni", config.get("ppmi", {})).get(
        "min_visits_longitudinal", 4
    )

    # Filter participants with enough visits
    visit_counts = df.groupby("RID").size()
    eligible_rids = visit_counts[visit_counts >= min_visits].index
    logger.info(
        "CSD analysis: %d/%d participants have >= %d visits",
        len(eligible_rids),
        df["RID"].nunique(),
        min_visits,
    )

    results = []
    n_gap_filtered = 0
    for rid in tqdm(eligible_rids, desc="CSD analysis"):
        participant_df = df[df["RID"] == rid].sort_values("EXAMDATE")

        # Validate inter-visit gaps
        participant_df = _filter_valid_visits(participant_df, config)
        if len(participant_df) < min_visits:
            n_gap_filtered += 1
            continue

        for protein in seq_cols:
            series = participant_df[protein].values.astype(float)

            # Skip if too few non-NaN values
            if np.isfinite(series).sum() < min_visits:
                continue

            csd = compute_csd_for_participant(series, window, detrend_method)
            csd["RID"] = rid
            csd["protein"] = protein
            results.append(csd)

    if n_gap_filtered > 0:
        logger.info(
            "%d participants dropped after visit gap filtering (< %d valid visits)",
            n_gap_filtered,
            min_visits,
        )

    results_df = pd.DataFrame(results)
    logger.info(
        "CSD analysis complete: %d participant-protein pairs from %d participants",
        len(results_df),
        len(eligible_rids) - n_gap_filtered,
    )
    return results_df


def compute_group_csd_statistics(
    csd_results: pd.DataFrame,
    trajectory_map: pd.Series,
    config: dict,
) -> pd.DataFrame:
    """Compare CSD statistics between converters and stable MCI.

    For each protein, runs Wilcoxon rank-sum test on Kendall tau values,
    computes effect sizes, and applies BH-FDR correction.

    Parameters
    ----------
    csd_results : pd.DataFrame
        Long-format CSD results from compute_csd_all_proteins.
    trajectory_map : pd.Series
        Series mapping RID -> TRAJECTORY label.
    config : dict
        Configuration with adni section.

    Returns
    -------
    pd.DataFrame
        Per-protein statistics sorted by FDR-corrected p-value.
    """
    from statsmodels.stats.multitest import multipletests

    adni_cfg = config["adni"]
    # Support both ADNI ("converter_group") and PPMI ("fast_progressor_group") key names
    converter_label = adni_cfg.get("converter_group") or adni_cfg.get("fast_progressor_group")
    stable_label = adni_cfg.get("stable_group") or adni_cfg.get("slow_progressor_group")

    # Merge trajectory labels — reset_index() names the column after the Series index name
    traj_df = trajectory_map.rename("TRAJECTORY").reset_index()
    rid_col = traj_df.columns[0]  # "RID" or "index" depending on whether index was named
    if rid_col != "RID":
        traj_df = traj_df.rename(columns={rid_col: "RID"})
    merged = csd_results.merge(traj_df, on="RID", how="left")

    converters = merged[merged["TRAJECTORY"] == converter_label]
    stable = merged[merged["TRAJECTORY"] == stable_label]

    logger.info(
        "Group comparison: %d converter entries, %d stable entries",
        len(converters),
        len(stable),
    )

    protein_stats = []
    proteins = csd_results["protein"].unique()

    for protein in proteins:
        conv_tau = converters.loc[converters["protein"] == protein, "var_tau"].dropna()
        stab_tau = stable.loc[stable["protein"] == protein, "var_tau"].dropna()

        if len(conv_tau) < 3 or len(stab_tau) < 3:
            continue

        U, p = stats.mannwhitneyu(conv_tau, stab_tau, alternative="greater")
        n1, n2 = len(conv_tau), len(stab_tau)
        rank_biserial = 1 - (2 * U) / (n1 * n2)

        # Also compute for AR1
        conv_ar1 = converters.loc[
            converters["protein"] == protein, "ar1_tau"
        ].dropna()
        stab_ar1 = stable.loc[stable["protein"] == protein, "ar1_tau"].dropna()

        ar1_U, ar1_p = (np.nan, np.nan)
        if len(conv_ar1) >= 3 and len(stab_ar1) >= 3:
            ar1_U, ar1_p = stats.mannwhitneyu(
                conv_ar1, stab_ar1, alternative="greater"
            )

        protein_stats.append(
            {
                "protein": protein,
                "median_var_tau_converter": conv_tau.median(),
                "median_var_tau_stable": stab_tau.median(),
                "var_U_statistic": U,
                "var_p_value": p,
                "rank_biserial_r": rank_biserial,
                "median_ar1_tau_converter": conv_ar1.median()
                if len(conv_ar1) > 0
                else np.nan,
                "median_ar1_tau_stable": stab_ar1.median()
                if len(stab_ar1) > 0
                else np.nan,
                "ar1_p_value": ar1_p,
                "n_converters": n1,
                "n_stable": n2,
            }
        )

    results = pd.DataFrame(protein_stats)

    if len(results) > 0:
        # BH-FDR correction on variance p-values
        _, fdr_p, _, _ = multipletests(
            results["var_p_value"], method="fdr_bh", alpha=config["csd"]["alpha"]
        )
        results["var_fdr_p"] = fdr_p

        # BH-FDR on AR1 p-values
        valid_ar1 = results["ar1_p_value"].notna()
        if valid_ar1.any():
            _, ar1_fdr, _, _ = multipletests(
                results.loc[valid_ar1, "ar1_p_value"],
                method="fdr_bh",
                alpha=config["csd"]["alpha"],
            )
            results.loc[valid_ar1, "ar1_fdr_p"] = ar1_fdr

        results = results.sort_values("var_fdr_p")

    n_sig = (results["var_fdr_p"] < config["csd"]["alpha"]).sum() if len(results) > 0 else 0
    logger.info(
        "Group CSD statistics: %d proteins tested, %d significant at FDR < %.2f",
        len(results),
        n_sig,
        config["csd"]["alpha"],
    )
    return results


def run_csd_sensitivity_analysis(
    df: pd.DataFrame,
    seq_cols: list[str],
    config: dict,
) -> pd.DataFrame:
    """Run CSD analysis across all window sizes and detrending methods.

    Tests robustness of CSD findings by repeating the analysis with every
    combination of window_sizes and detrending_methods from config.

    Parameters
    ----------
    df : pd.DataFrame
        Processed proteomics DataFrame.
    seq_cols : list[str]
        Protein column names.
    config : dict
        Configuration with csd.window_sizes and csd.detrending_methods.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with one row per (window, method) combination,
        containing median tau statistics across all participant-protein pairs.
    """
    from pathlib import Path

    csd_cfg = config["csd"]
    window_sizes = csd_cfg["window_sizes"]
    detrending_methods = csd_cfg["detrending_methods"]

    results_dir = Path(config["paths"]["results_csd"]) / "sensitivity"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for window in window_sizes:
        for method in detrending_methods:
            logger.info(
                "Sensitivity analysis: window=%d, detrending=%s", window, method
            )

            # Create a modified config for this combination
            sens_config = config.copy()
            sens_config["csd"] = dict(csd_cfg)
            sens_config["csd"]["primary_window"] = window
            sens_config["csd"]["primary_detrending"] = method

            csd_results = compute_csd_all_proteins(df, seq_cols, sens_config)

            # Save per-combination results
            output_file = results_dir / f"csd_w{window}_{method}.csv"
            csd_results.to_csv(output_file, index=False)

            # Summarize
            summary_rows.append(
                {
                    "window": window,
                    "detrending": method,
                    "n_pairs": len(csd_results),
                    "median_var_tau": csd_results["var_tau"].median()
                    if len(csd_results) > 0
                    else np.nan,
                    "median_ar1_tau": csd_results["ar1_tau"].median()
                    if len(csd_results) > 0
                    else np.nan,
                    "mean_var_tau": csd_results["var_tau"].mean()
                    if len(csd_results) > 0
                    else np.nan,
                    "mean_ar1_tau": csd_results["ar1_tau"].mean()
                    if len(csd_results) > 0
                    else np.nan,
                    "frac_positive_var_tau": (csd_results["var_tau"] > 0).mean()
                    if len(csd_results) > 0
                    else np.nan,
                    "frac_positive_ar1_tau": (csd_results["ar1_tau"] > 0).mean()
                    if len(csd_results) > 0
                    else np.nan,
                }
            )

    summary = pd.DataFrame(summary_rows)
    logger.info(
        "Sensitivity analysis complete: %d combinations tested", len(summary)
    )
    return summary
