"""Single-sample Dynamic Network Biomarker (sDNB).

Computes a DNB score for individual participants relative to a reference
population without requiring pre-defined groups. This allows ranking
every individual by their proximity to the molecular tipping point.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def _compute_reference_correlations(
    reference_values: np.ndarray,
    reference_outside: np.ndarray | None,
) -> tuple[float, float]:
    """Precompute PCC_D and PCC_O from the reference population.

    These are properties of the reference correlation structure and do
    not change across participants.

    Returns
    -------
    tuple[float, float]
        (PCC_D, PCC_O)
    """
    n_core = reference_values.shape[1]

    # PCC_D: mean absolute within-group correlation
    if n_core < 2:
        PCC_D = 1.0
    else:
        corr_matrix = np.corrcoef(reference_values.T)
        upper_idx = np.triu_indices(n_core, k=1)
        pairwise = corr_matrix[upper_idx]
        pairwise_finite = pairwise[np.isfinite(pairwise)]
        PCC_D = np.abs(pairwise_finite).mean() if len(pairwise_finite) > 0 else 0.0

    # PCC_O: mean absolute correlation between core and non-core
    if reference_outside is not None and reference_outside.shape[1] > 0:
        X_all = np.hstack([reference_values, reference_outside])
        col_means = np.nanmean(X_all, axis=0)
        nan_idx = np.where(np.isnan(X_all))
        if nan_idx[0].size > 0:
            X_all[nan_idx] = np.take(col_means, nan_idx[1])
        corr_all = np.corrcoef(X_all.T)
        cross_block = corr_all[:n_core, n_core:]
        valid_cross = cross_block[np.isfinite(cross_block)]
        PCC_O = np.abs(valid_cross).mean() if len(valid_cross) > 0 else 0.0
    else:
        PCC_O = 0.0

    return PCC_D, PCC_O


def compute_sdnb_score(
    participant_values: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    PCC_D: float,
    PCC_O: float,
    epsilon: float = 1e-8,
) -> float:
    """Compute single-sample DNB score for one participant.

    Mirrors the stage-level DNB formula (Chen et al. 2012) adapted for a
    single sample:

        sDNB = (sigma_D × PCC_D) / (PCC_O + epsilon)

    where sigma_D is the mean absolute z-score of the participant's core
    proteins (deviation from reference), PCC_D and PCC_O are precomputed
    from the reference correlation structure.

    Parameters
    ----------
    participant_values : np.ndarray
        Shape (n_proteins,). Core protein values for one participant.
    ref_mean : np.ndarray
        Shape (n_proteins,). Reference mean per protein.
    ref_std : np.ndarray
        Shape (n_proteins,). Reference std per protein.
    PCC_D : float
        Precomputed mean absolute within-group correlation.
    PCC_O : float
        Precomputed mean absolute cross-group correlation.
    epsilon : float
        Denominator stabilizer.

    Returns
    -------
    float
        sDNB score.
    """
    if len(participant_values) == 0:
        return 0.0

    z_scores = (participant_values - ref_mean) / ref_std

    valid = np.isfinite(z_scores)
    if valid.sum() < 1:
        return 0.0

    sigma_D = np.abs(z_scores[valid]).mean()

    return (sigma_D * PCC_D) / (PCC_O + epsilon)


def run_sdnb_analysis(
    df: pd.DataFrame,
    seq_cols: list[str],
    reference_mask: pd.Series,
    config: dict,
    core_protein_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute sDNB scores for all participants.

    Parameters
    ----------
    df : pd.DataFrame
        Proteomics DataFrame with TRAJECTORY and MONTHS_TO_CONVERSION.
    seq_cols : list[str]
        All protein column names (fallback if core_protein_cols not provided).
    reference_mask : pd.Series
        Boolean mask for reference population.
    config : dict
        Configuration with dnb section.
    core_protein_cols : list[str] or None
        If provided, restrict sDNB scoring to these DNB core proteins only.
        This focuses the score on the pre-identified transition network rather
        than diluting it across thousands of background proteins.

    Returns
    -------
    pd.DataFrame
        Per-participant sDNB scores with clinical annotations.
    """
    epsilon = config["dnb"]["epsilon"]
    # Use DNB core proteins if provided; fall back to all proteins
    score_cols = core_protein_cols if core_protein_cols else seq_cols
    reference_values = df.loc[reference_mask, score_cols].values

    # Non-core proteins for PCC_O computation
    if core_protein_cols:
        outside_cols = [c for c in seq_cols if c not in set(core_protein_cols)]
        reference_outside = df.loc[reference_mask, outside_cols].values if outside_cols else None
    else:
        outside_cols = []
        reference_outside = None

    # Precompute reference statistics (constant across participants)
    ref_mean = np.nanmean(reference_values, axis=0)
    ref_std = np.nanstd(reference_values, axis=0, ddof=1)
    ref_std = np.where(ref_std > 0, ref_std, 1e-10)
    PCC_D, PCC_O = _compute_reference_correlations(reference_values, reference_outside)

    logger.info(
        "sDNB analysis: %d participants, %d reference samples, %d core proteins, "
        "%d outside proteins, PCC_D=%.4f, PCC_O=%.4f%s",
        df["RID"].nunique(),
        reference_mask.sum(),
        len(score_cols),
        len(outside_cols),
        PCC_D, PCC_O,
        " (DNB core proteins)" if core_protein_cols else " (all proteins — no core proteins provided)",
    )

    # Compute sDNB for each participant.
    # Visit selection: last pre-conversion visit for converters (smallest
    # positive MONTHS_TO_CONVERSION), last visit by date for stable/other.
    # Max-across-visits is retained as a diagnostic column for AUC comparison.
    adni_cfg = config.get("adni", config.get("ppmi", {}))
    converter_label = adni_cfg.get("converter_group", adni_cfg.get("fast_progressor_group"))

    results = []
    for rid, group in df.groupby("RID"):
        visit_scores = []
        for _, row in group.iterrows():
            participant_visit = row[score_cols].values.astype(float)
            visit_score = compute_sdnb_score(participant_visit, ref_mean, ref_std, PCC_D, PCC_O, epsilon)
            visit_scores.append({
                "score": visit_score,
                "months_to_conv": row.get("MONTHS_TO_CONVERSION", np.nan),
                "examdate": row.get("EXAMDATE", pd.NaT),
            })

        if not visit_scores:
            continue

        trajectory = group["TRAJECTORY"].iloc[0]
        max_score = max(v["score"] for v in visit_scores)

        # --- Principled visit selection ---
        selected = None
        if trajectory == converter_label:
            # Last pre-conversion visit: smallest positive MONTHS_TO_CONVERSION
            pre_conv = [v for v in visit_scores
                        if pd.notna(v["months_to_conv"]) and v["months_to_conv"] > 0]
            if pre_conv:
                selected = min(pre_conv, key=lambda v: v["months_to_conv"])

        if selected is None:
            # Stable/other, or converter fallback: latest EXAMDATE
            dated = [v for v in visit_scores if pd.notna(v["examdate"])]
            if dated:
                selected = max(dated, key=lambda v: v["examdate"])
            else:
                selected = visit_scores[-1]

        results.append({
            "RID": rid,
            "sdnb_score": selected["score"],
            "sdnb_score_max": max_score,
            "TRAJECTORY": trajectory,
            "MONTHS_TO_CONVERSION": selected["months_to_conv"],
        })

    results_df = pd.DataFrame(results)

    # Compute correlation with time-to-conversion for converters
    converters = results_df[
        (results_df["TRAJECTORY"] == converter_label)
        & results_df["MONTHS_TO_CONVERSION"].notna()
    ]

    if len(converters) >= 3:
        rho, p = stats.spearmanr(
            converters["sdnb_score"], converters["MONTHS_TO_CONVERSION"]
        )
        logger.info(
            "sDNB vs time-to-conversion: Spearman rho=%.3f, p=%.4f (n=%d)",
            rho, p, len(converters),
        )
    else:
        logger.info("Too few converters for correlation analysis")

    # ROC AUC: sDNB predicting converter vs stable
    stable_label = adni_cfg.get("stable_group", adni_cfg.get("slow_progressor_group"))
    binary_df = results_df[
        results_df["TRAJECTORY"].isin([converter_label, stable_label])
    ].copy()

    if len(binary_df) >= 10:
        binary_df["is_converter"] = (binary_df["TRAJECTORY"] == converter_label).astype(int)
        valid = binary_df["sdnb_score"].notna()
        valid_max = binary_df["sdnb_score_max"].notna()
        if valid.sum() >= 10 and binary_df.loc[valid, "is_converter"].nunique() == 2:
            auc = roc_auc_score(
                binary_df.loc[valid, "is_converter"],
                binary_df.loc[valid, "sdnb_score"],
            )
            logger.info("sDNB ROC AUC (last-pre-conversion visit): %.3f", auc)

            # Diagnostic: compare with max-across-visits to detect inflation
            if valid_max.sum() >= 10 and binary_df.loc[valid_max, "is_converter"].nunique() == 2:
                auc_max = roc_auc_score(
                    binary_df.loc[valid_max, "is_converter"],
                    binary_df.loc[valid_max, "sdnb_score_max"],
                )
                logger.info("sDNB ROC AUC (max-across-visits, diagnostic): %.3f", auc_max)
                logger.info("AUC difference (max - principled): %.3f", auc_max - auc)

    # Drop diagnostic column before returning
    results_df = results_df.drop(columns=["sdnb_score_max"])

    logger.info("sDNB analysis complete: %d participants scored", len(results_df))
    return results_df
