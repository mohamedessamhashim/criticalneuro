"""Composite CSD score computation and temporal specificity analysis.

Summarizes each participant's CSD profile into a single number and
analyzes how CSD scores change as a function of time-to-conversion.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def compute_composite_csd_score(
    csd_results: pd.DataFrame,
    method: str = "mean_tau",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Summarize each participant's CSD profile into a single score.

    Parameters
    ----------
    csd_results : pd.DataFrame
        Long-format CSD results with columns: RID, protein, var_tau, ar1_tau.
    method : str
        'mean_tau': average Kendall tau across proteins, sum var + ar1.
        'pc1_tau': first principal component of protein-level tau matrix.
    random_seed : int
        Random seed for PCA.

    Returns
    -------
    pd.DataFrame
        Per-participant scores: RID, composite_csd_score, mean_var_tau, mean_ar1_tau.
    """
    if method == "mean_tau":
        grouped = csd_results.groupby("RID").agg(
            mean_var_tau=("var_tau", "mean"),
            mean_ar1_tau=("ar1_tau", "mean"),
        ).reset_index()

        grouped["composite_csd_score"] = (
            grouped["mean_var_tau"] + grouped["mean_ar1_tau"]
        )

        logger.info(
            "Composite CSD scores (mean_tau): %d participants, "
            "mean score = %.4f ± %.4f",
            len(grouped),
            grouped["composite_csd_score"].mean(),
            grouped["composite_csd_score"].std(),
        )
        return grouped

    elif method == "pc1_tau":
        # Pivot to wide format: RID x protein var_tau
        pivot = csd_results.pivot_table(
            index="RID", columns="protein", values="var_tau", aggfunc="first"
        ).fillna(0)

        pca = PCA(n_components=1, random_state=random_seed)
        pc1_scores = pca.fit_transform(pivot.values).flatten()

        result = pd.DataFrame(
            {
                "RID": pivot.index,
                "composite_csd_score": pc1_scores,
                "mean_var_tau": csd_results.groupby("RID")["var_tau"].mean().values,
                "mean_ar1_tau": csd_results.groupby("RID")["ar1_tau"].mean().values,
            }
        )

        logger.info(
            "Composite CSD scores (pc1_tau): %d participants, "
            "explained variance = %.2f%%",
            len(result),
            pca.explained_variance_ratio_[0] * 100,
        )
        return result

    else:
        raise ValueError(f"Unknown composite score method: {method}")


def temporal_specificity_analysis(
    df: pd.DataFrame,
    composite_scores: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Analyze CSD scores by time-to-conversion windows.

    Tests whether CSD scores differ across time windows using
    Kruskal-Wallis. The expected pattern under CSD theory is a peak
    at 12-24 months, not monotonic increase.

    Parameters
    ----------
    df : pd.DataFrame
        Proteomics DataFrame with TRAJECTORY and MONTHS_TO_CONVERSION.
    composite_scores : pd.DataFrame
        Per-participant composite CSD scores.
    config : dict
        Configuration with adni section.

    Returns
    -------
    pd.DataFrame
        CSD score statistics per time window.
    """
    adni_cfg = config["adni"]
    converter_label = adni_cfg["converter_group"]

    # Merge scores with clinical data
    merged = composite_scores.merge(
        df[["RID", "TRAJECTORY", "MONTHS_TO_CONVERSION"]].drop_duplicates("RID"),
        on="RID",
        how="inner",
    )

    # Filter to converters
    converters = merged[merged["TRAJECTORY"] == converter_label].copy()

    if len(converters) == 0:
        logger.warning("No converter participants found for temporal analysis")
        return pd.DataFrame()

    # Bin by time-to-conversion
    # Use the minimum MONTHS_TO_CONVERSION per participant (closest to conversion)
    participant_min_months = (
        df[df["TRAJECTORY"] == converter_label]
        .groupby("RID")["MONTHS_TO_CONVERSION"]
        .min()
    )

    converters = converters.merge(
        participant_min_months.rename("min_months_to_conv"),
        on="RID",
        how="left",
    )

    bins = [
        (">=36 months", converters["min_months_to_conv"] >= 36),
        ("24-36 months", (converters["min_months_to_conv"] >= 24) & (converters["min_months_to_conv"] < 36)),
        ("12-24 months", (converters["min_months_to_conv"] >= 12) & (converters["min_months_to_conv"] < 24)),
        ("<12 months", converters["min_months_to_conv"] < 12),
    ]

    results = []
    bin_scores = []
    for label, mask in bins:
        scores = converters.loc[mask, "composite_csd_score"]
        results.append(
            {
                "time_window": label,
                "mean_csd": scores.mean() if len(scores) > 0 else np.nan,
                "sem_csd": scores.sem() if len(scores) > 1 else np.nan,
                "n_participants": len(scores),
            }
        )
        if len(scores) > 0:
            bin_scores.append(scores.values)

    # Kruskal-Wallis test across bins
    kw_h, kw_p = (np.nan, np.nan)
    if len(bin_scores) >= 2:
        valid_bins = [b for b in bin_scores if len(b) >= 2]
        if len(valid_bins) >= 2:
            kw_h, kw_p = stats.kruskal(*valid_bins)

    results_df = pd.DataFrame(results)

    logger.info(
        "Temporal specificity: Kruskal-Wallis H=%.2f, p=%.4f across %d windows",
        kw_h if np.isfinite(kw_h) else 0,
        kw_p if np.isfinite(kw_p) else 1,
        len(results_df),
    )
    logger.info("  Window scores: %s", results_df[["time_window", "mean_csd", "n_participants"]].to_string(index=False))

    # Add test statistics as metadata
    results_df.attrs["kruskal_wallis_H"] = kw_h
    results_df.attrs["kruskal_wallis_p"] = kw_p

    return results_df
