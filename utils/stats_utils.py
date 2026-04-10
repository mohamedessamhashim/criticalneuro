"""Statistical utility functions for CriticalNeuroMap pipeline."""

import numpy as np
from scipy import stats


def mann_whitney_test(
    group1: np.ndarray, group2: np.ndarray, alternative: str = "two-sided"
) -> tuple[float, float]:
    """Mann-Whitney U test with NaN handling.

    Parameters
    ----------
    group1, group2 : np.ndarray
        Sample values for each group.
    alternative : str
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    tuple[float, float]
        (U statistic, p-value)
    """
    g1 = group1[~np.isnan(group1)]
    g2 = group2[~np.isnan(group2)]

    if len(g1) < 2 or len(g2) < 2:
        return np.nan, np.nan

    u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative=alternative)
    return float(u_stat), float(p_val)


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman correlation with NaN handling.

    Parameters
    ----------
    x, y : np.ndarray
        Paired observations.

    Returns
    -------
    tuple[float, float]
        (rho, p-value)
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan, np.nan

    rho, p_val = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p_val)


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC-ROC with NaN handling.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0/1).
    y_score : np.ndarray
        Continuous scores.

    Returns
    -------
    float
        AUC score, or NaN if insufficient data.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_t = y_true[mask]
    y_s = y_score[mask]

    if len(y_t) < 2 or len(np.unique(y_t)) < 2:
        return np.nan

    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_t, y_s))
