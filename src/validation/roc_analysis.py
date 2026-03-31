"""ROC curve computation with bootstrap confidence intervals.

Computes ROC curves for multiple predictors at multiple time horizons,
with DeLong's test for comparing AUCs between predictors.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def _delong_test(y_true: np.ndarray, y_score1: np.ndarray, y_score2: np.ndarray) -> tuple[float, float]:
    """DeLong's test for comparing two AUCs.

    Uses the placement value approach from Sun & Xu 2014.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_score1, y_score2 : np.ndarray
        Predicted scores from two models.

    Returns
    -------
    tuple[float, float]
        (z_statistic, p_value)
    """
    positive = y_true == 1
    negative = y_true == 0
    n1 = positive.sum()
    n0 = negative.sum()

    if n1 < 2 or n0 < 2:
        return (np.nan, np.nan)

    # Placement values for model 1
    V10_1 = np.array([np.mean(y_score1[negative] < s) for s in y_score1[positive]])
    V01_1 = np.array([np.mean(y_score1[positive] > s) for s in y_score1[negative]])

    # Placement values for model 2
    V10_2 = np.array([np.mean(y_score2[negative] < s) for s in y_score2[positive]])
    V01_2 = np.array([np.mean(y_score2[positive] > s) for s in y_score2[negative]])

    # Variance of AUC difference
    S10 = np.cov(V10_1, V10_2)
    S01 = np.cov(V01_1, V01_2)

    # Variance of the difference
    var_diff = (S10[0, 0] - 2 * S10[0, 1] + S10[1, 1]) / n1 + \
               (S01[0, 0] - 2 * S01[0, 1] + S01[1, 1]) / n0

    if var_diff <= 0:
        return (np.nan, np.nan)

    auc1 = roc_auc_score(y_true, y_score1)
    auc2 = roc_auc_score(y_true, y_score2)
    z = (auc1 - auc2) / np.sqrt(var_diff)
    p = 2 * stats.norm.sf(abs(z))

    return (z, p)


def compute_roc_curves(
    df: pd.DataFrame,
    predictor_cols: list[str],
    outcome_col: str,
    time_horizons: list[int],
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute ROC curves for each predictor at each time horizon.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with predictor columns, outcome, and MONTHS_TO_CONVERSION.
    predictor_cols : list[str]
        Names of predictor columns.
    outcome_col : str
        Column indicating converter status (1/0).
    time_horizons : list[int]
        Months before conversion to evaluate (e.g., [12, 24, 36]).
    n_bootstrap : int
        Number of bootstrap resamples for CIs.
    random_seed : int
        Random seed.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (roc_results, delong_comparisons)
    """
    rng = np.random.RandomState(random_seed)
    available_predictors = [c for c in predictor_cols if c in df.columns]

    logger.info(
        "ROC analysis: %d predictors × %d time horizons",
        len(available_predictors),
        len(time_horizons),
    )

    roc_results = []
    delong_results = []

    for horizon in time_horizons:
        # Define outcome at this time horizon
        # Converter within horizon months: cases where MONTHS_TO_CONVERSION <= horizon
        horizon_df = df.copy()

        if "MONTHS_TO_CONVERSION" in horizon_df.columns:
            # For converters: positive if conversion within horizon
            is_converter = horizon_df[outcome_col] == 1
            within_horizon = horizon_df["MONTHS_TO_CONVERSION"] <= horizon
            horizon_df["horizon_outcome"] = (
                is_converter & within_horizon
            ).astype(int)
        else:
            horizon_df["horizon_outcome"] = horizon_df[outcome_col]

        for predictor in available_predictors:
            valid = horizon_df[[predictor, "horizon_outcome"]].dropna()
            if len(valid) < 20 or valid["horizon_outcome"].nunique() < 2:
                continue

            y = valid["horizon_outcome"].values
            scores = valid[predictor].values

            # Main AUC
            auc = roc_auc_score(y, scores)
            fpr, tpr, thresholds = roc_curve(y, scores)

            # Bootstrap CI
            boot_aucs = []
            for _ in range(n_bootstrap):
                idx = rng.choice(len(y), len(y), replace=True)
                y_b, s_b = y[idx], scores[idx]
                if y_b.sum() > 0 and y_b.sum() < len(y_b):
                    boot_aucs.append(roc_auc_score(y_b, s_b))

            ci_lower = np.percentile(boot_aucs, 2.5) if boot_aucs else np.nan
            ci_upper = np.percentile(boot_aucs, 97.5) if boot_aucs else np.nan

            roc_results.append(
                {
                    "predictor": predictor,
                    "time_horizon": horizon,
                    "auc": auc,
                    "auc_ci_lower": ci_lower,
                    "auc_ci_upper": ci_upper,
                    "n_samples": len(y),
                    "n_events": int(y.sum()),
                }
            )

        # DeLong pairwise comparisons at this horizon
        for i, pred_i in enumerate(available_predictors):
            for j, pred_j in enumerate(available_predictors):
                if j <= i:
                    continue

                valid = horizon_df[[pred_i, pred_j, "horizon_outcome"]].dropna()
                if len(valid) < 20 or valid["horizon_outcome"].nunique() < 2:
                    continue

                y = valid["horizon_outcome"].values
                z, p = _delong_test(y, valid[pred_i].values, valid[pred_j].values)

                delong_results.append(
                    {
                        "predictor_1": pred_i,
                        "predictor_2": pred_j,
                        "time_horizon": horizon,
                        "z_statistic": z,
                        "p_value": p,
                    }
                )

    roc_df = pd.DataFrame(roc_results)
    delong_df = pd.DataFrame(delong_results)

    if len(roc_df) > 0:
        logger.info("ROC results:\n%s", roc_df.to_string(index=False))

    return roc_df, delong_df
