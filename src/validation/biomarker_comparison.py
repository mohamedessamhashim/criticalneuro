"""Biomarker comparison and incremental prediction analysis.

Compares CSD and DNB scores against established biomarkers (p-tau217,
NfL, GFAP, Abeta42/40) using Spearman correlations, subgroup analysis
in biomarker-negative participants, and incremental logistic regression
with bootstrap confidence intervals.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def compute_biomarker_correlations(
    df: pd.DataFrame,
    csd_scores: pd.DataFrame,
    biomarker_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Spearman correlations between CSD/DNB scores and biomarkers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with biomarker columns and RID.
    csd_scores : pd.DataFrame
        DataFrame with RID, composite_csd_score, and optionally sdnb_score.
    biomarker_cols : list[str]
        Biomarker column names to include.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (correlation_matrix, p_value_matrix)
    """
    # Merge scores with biomarkers
    merged = df[["RID"] + [c for c in biomarker_cols if c in df.columns]].drop_duplicates("RID")
    merged = merged.merge(csd_scores, on="RID", how="inner")

    # Columns to correlate
    score_cols = []
    if "composite_csd_score" in merged.columns:
        score_cols.append("composite_csd_score")
    if "sdnb_score" in merged.columns:
        score_cols.append("sdnb_score")

    available_biomarkers = [c for c in biomarker_cols if c in merged.columns]
    all_cols = score_cols + available_biomarkers

    logger.info(
        "Biomarker correlations: %d participants, %d variables",
        len(merged),
        len(all_cols),
    )

    # Compute Spearman correlations
    n = len(all_cols)
    corr_matrix = pd.DataFrame(np.nan, index=all_cols, columns=all_cols)
    p_matrix = pd.DataFrame(np.nan, index=all_cols, columns=all_cols)

    for i, col_i in enumerate(all_cols):
        for j, col_j in enumerate(all_cols):
            valid = merged[[col_i, col_j]].dropna()
            # Ensure both columns are numeric before computing correlation
            try:
                a = pd.to_numeric(valid[col_i], errors="coerce")
                b = pd.to_numeric(valid[col_j], errors="coerce")
                ab = pd.concat([a, b], axis=1).dropna()
                if len(ab) >= 3:
                    result = stats.spearmanr(ab.iloc[:, 0], ab.iloc[:, 1])
                    corr_matrix.loc[col_i, col_j] = float(result.statistic)
                    p_matrix.loc[col_i, col_j] = float(result.pvalue)
            except Exception:
                pass

    return corr_matrix, p_matrix


def biomarker_negative_subgroup_analysis(
    df: pd.DataFrame,
    csd_scores: pd.DataFrame,
    threshold_col: str,
    threshold_value: float,
    outcome_col: str,
) -> dict:
    """Test CSD prediction within biomarker-negative participants.

    The most clinically important analysis: if CSD detects conversion
    before biomarkers cross thresholds, this has implications for
    early intervention.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with biomarker and outcome columns.
    csd_scores : pd.DataFrame
        Per-participant CSD scores.
    threshold_col : str
        Biomarker column for subsetting.
    threshold_value : float
        Threshold below which participants are "biomarker-negative".
    outcome_col : str
        Binary outcome column (1 = converter).

    Returns
    -------
    dict
        Subgroup analysis results.
    """
    merged = df.merge(csd_scores, on="RID", how="inner")

    if threshold_col not in merged.columns:
        logger.warning("Threshold column '%s' not found", threshold_col)
        return {"error": f"Column {threshold_col} not found"}

    # Filter to biomarker-negative
    subgroup = merged[merged[threshold_col] < threshold_value].copy()

    if len(subgroup) < 10:
        logger.warning("Biomarker-negative subgroup too small: %d", len(subgroup))
        return {"n_subgroup": len(subgroup), "error": "Subgroup too small"}

    valid = subgroup[["composite_csd_score", outcome_col]].dropna()
    if len(valid) < 10 or valid[outcome_col].nunique() < 2:
        return {"n_subgroup": len(subgroup), "error": "Insufficient valid data"}

    # Mann-Whitney U test
    converters = valid[valid[outcome_col] == 1]["composite_csd_score"]
    non_converters = valid[valid[outcome_col] == 0]["composite_csd_score"]

    U, p = stats.mannwhitneyu(converters, non_converters, alternative="greater")

    # ROC AUC
    auc = roc_auc_score(valid[outcome_col], valid["composite_csd_score"])

    result = {
        "n_subgroup": len(subgroup),
        "n_converter": len(converters),
        "n_stable": len(non_converters),
        "U_statistic": U,
        "p_value": p,
        "auc": auc,
    }

    logger.info(
        "Biomarker-negative subgroup (%s < %.2f): n=%d, AUC=%.3f, p=%.4f",
        threshold_col,
        threshold_value,
        len(subgroup),
        auc,
        p,
    )
    return result


def incremental_prediction_analysis(
    df: pd.DataFrame,
    csd_scores: pd.DataFrame,
    biomarker_cols: list[str],
    outcome_col: str,
    config: dict,
) -> dict:
    """Compare base model vs augmented model with CSD score.

    Base: age + sex + APOE4 + established biomarkers
    Augmented: base + composite_csd_score

    Computes likelihood ratio test, AUC difference, NRI, and IDI
    with bootstrap confidence intervals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with covariates and outcome.
    csd_scores : pd.DataFrame
        Per-participant CSD scores.
    biomarker_cols : list[str]
        Biomarker columns for base model.
    outcome_col : str
        Binary outcome column.
    config : dict
        Configuration with validation section.

    Returns
    -------
    dict
        All metrics with bootstrap CIs.
    """
    n_bootstrap = config["validation"]["bootstrap_n"]
    random_seed = config["random_seed"]

    merged = df.merge(csd_scores, on="RID", how="inner")

    # Identify available predictors
    base_predictors = ["AGE", "SEX", "APOE4"]
    available_biomarkers = [c for c in biomarker_cols if c in merged.columns]
    base_predictors.extend(available_biomarkers)
    base_predictors = [c for c in base_predictors if c in merged.columns]

    extra_predictors = [c for c in ["composite_csd_score", "sdnb_score"]
                        if c in merged.columns]
    augmented_predictors = base_predictors + extra_predictors

    # Drop rows with missing values
    all_cols = augmented_predictors + [outcome_col]
    clean = merged[[c for c in all_cols if c in merged.columns]].dropna()

    if len(clean) < 20 or clean[outcome_col].nunique() < 2:
        logger.warning("Insufficient data for incremental prediction analysis")
        return {"error": "Insufficient data"}

    X_base = clean[base_predictors].values
    X_augmented = clean[augmented_predictors].values
    y = clean[outcome_col].values.astype(int)

    # Standardize
    scaler_base = StandardScaler()
    scaler_aug = StandardScaler()
    X_base_scaled = scaler_base.fit_transform(X_base)
    X_aug_scaled = scaler_aug.fit_transform(X_augmented)

    # Fit models
    base_model = LogisticRegression(max_iter=1000, random_state=random_seed)
    aug_model = LogisticRegression(max_iter=1000, random_state=random_seed)
    base_model.fit(X_base_scaled, y)
    aug_model.fit(X_aug_scaled, y)

    # Predictions
    base_proba = base_model.predict_proba(X_base_scaled)[:, 1]
    aug_proba = aug_model.predict_proba(X_aug_scaled)[:, 1]

    # AUCs
    auc_base = roc_auc_score(y, base_proba)
    auc_aug = roc_auc_score(y, aug_proba)

    # Log-likelihood ratio test
    ll_base = np.sum(y * np.log(np.clip(base_proba, 1e-10, 1)) +
                     (1 - y) * np.log(np.clip(1 - base_proba, 1e-10, 1)))
    ll_aug = np.sum(y * np.log(np.clip(aug_proba, 1e-10, 1)) +
                    (1 - y) * np.log(np.clip(1 - aug_proba, 1e-10, 1)))
    lr_chi2 = -2 * (ll_base - ll_aug)
    lr_p = 1 - stats.chi2.cdf(lr_chi2, df=1)

    # NRI (Net Reclassification Improvement) at threshold 0.5
    base_class = (base_proba >= 0.5).astype(int)
    aug_class = (aug_proba >= 0.5).astype(int)
    events = y == 1
    non_events = y == 0
    nri_events = ((aug_class[events] > base_class[events]).sum() -
                  (aug_class[events] < base_class[events]).sum()) / events.sum()
    nri_non_events = ((aug_class[non_events] < base_class[non_events]).sum() -
                      (aug_class[non_events] > base_class[non_events]).sum()) / non_events.sum()
    nri = nri_events + nri_non_events

    # IDI (Integrated Discrimination Improvement)
    idi = (aug_proba[events].mean() - base_proba[events].mean()) - \
          (aug_proba[non_events].mean() - base_proba[non_events].mean())

    # Bootstrap CIs
    rng = np.random.RandomState(random_seed)
    boot_aucs_diff = []
    boot_nris = []
    boot_idis = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y), len(y), replace=True)
        y_b = y[idx]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue

        X_base_b = X_base_scaled[idx]
        X_aug_b = X_aug_scaled[idx]

        try:
            bm = LogisticRegression(max_iter=1000, random_state=random_seed)
            am = LogisticRegression(max_iter=1000, random_state=random_seed)
            bm.fit(X_base_b, y_b)
            am.fit(X_aug_b, y_b)

            bp = bm.predict_proba(X_base_b)[:, 1]
            ap = am.predict_proba(X_aug_b)[:, 1]

            boot_aucs_diff.append(roc_auc_score(y_b, ap) - roc_auc_score(y_b, bp))

            bc = (bp >= 0.5).astype(int)
            ac = (ap >= 0.5).astype(int)
            ev = y_b == 1
            ne = y_b == 0
            if ev.sum() > 0 and ne.sum() > 0:
                nri_b = ((ac[ev] > bc[ev]).sum() - (ac[ev] < bc[ev]).sum()) / ev.sum() + \
                        ((ac[ne] < bc[ne]).sum() - (ac[ne] > bc[ne]).sum()) / ne.sum()
                boot_nris.append(nri_b)
                idi_b = (ap[ev].mean() - bp[ev].mean()) - (ap[ne].mean() - bp[ne].mean())
                boot_idis.append(idi_b)
        except Exception:
            continue

    result = {
        "auc_base": auc_base,
        "auc_augmented": auc_aug,
        "auc_difference": auc_aug - auc_base,
        "auc_diff_ci_lower": np.percentile(boot_aucs_diff, 2.5) if boot_aucs_diff else np.nan,
        "auc_diff_ci_upper": np.percentile(boot_aucs_diff, 97.5) if boot_aucs_diff else np.nan,
        "lr_chi2": lr_chi2,
        "lr_p_value": lr_p,
        "nri": nri,
        "nri_ci_lower": np.percentile(boot_nris, 2.5) if boot_nris else np.nan,
        "nri_ci_upper": np.percentile(boot_nris, 97.5) if boot_nris else np.nan,
        "idi": idi,
        "idi_ci_lower": np.percentile(boot_idis, 2.5) if boot_idis else np.nan,
        "idi_ci_upper": np.percentile(boot_idis, 97.5) if boot_idis else np.nan,
        "n_samples": len(y),
        "n_events": int(y.sum()),
        "base_predictors": base_predictors,
    }

    logger.info(
        "Incremental prediction: AUC base=%.3f, augmented=%.3f, "
        "diff=%.3f [%.3f, %.3f], LRT p=%.4f, NRI=%.3f, IDI=%.3f",
        auc_base, auc_aug, auc_aug - auc_base,
        result["auc_diff_ci_lower"], result["auc_diff_ci_upper"],
        lr_p, nri, idi,
    )
    return result
