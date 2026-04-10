"""Dynamic Network Biomarker (DNB) score computation.

Implements the DNB scoring formula from Chen et al. 2012.
Module-level DNB identification is handled by WGCNA → BioTIP → l-DNB
(see pipeline/stage4_wgcna.R, pipeline/stage4_biotip.R, pipeline/stage4_ldnb.R).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_dnb_score(
    X_group: np.ndarray,
    X_outside: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Compute the DNB score for a group of proteins.

    DNB = (sigma_D × PCC_D) / (|PCC_O| + epsilon)

    where sigma_D is mean within-group standard deviation, PCC_D is mean
    absolute pairwise correlation within the group, and PCC_O is mean
    absolute correlation between group and outside proteins.

    Parameters
    ----------
    X_group : np.ndarray
        Shape (n_samples, n_group_proteins).
    X_outside : np.ndarray
        Shape (n_samples, n_outside_proteins).
    epsilon : float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        DNB score.
    """
    n_group = X_group.shape[1]

    if n_group == 0:
        return 0.0

    # sigma_D: mean within-group standard deviation
    sigma_D = np.nanstd(X_group, axis=0, ddof=1).mean()

    # PCC_D: mean absolute pairwise correlation within group
    if n_group == 1:
        PCC_D = 1.0
    else:
        corr_matrix = np.corrcoef(X_group.T)
        # Extract upper triangle (excluding diagonal)
        upper_idx = np.triu_indices(n_group, k=1)
        pairwise_corr = corr_matrix[upper_idx]
        # Handle NaN correlations
        valid_corr = pairwise_corr[np.isfinite(pairwise_corr)]
        PCC_D = np.abs(valid_corr).mean() if len(valid_corr) > 0 else 0.0

    # PCC_O: mean absolute correlation between group and outside
    # Vectorized: compute full cross-correlation matrix in one shot
    if X_outside.shape[1] == 0:
        PCC_O = 0.0
    else:
        # Stack group and outside, compute full correlation matrix once
        X_all = np.hstack([X_group, X_outside])
        # Replace NaN with column means for correlation computation
        col_means = np.nanmean(X_all, axis=0)
        inds = np.where(np.isnan(X_all))
        X_all[inds] = np.take(col_means, inds[1])
        corr_all = np.corrcoef(X_all.T)
        # Extract group x outside block
        cross_block = corr_all[:n_group, n_group:]
        valid_cross = cross_block[np.isfinite(cross_block)]
        PCC_O = np.abs(valid_cross).mean() if len(valid_cross) > 0 else 0.0

    return (sigma_D * PCC_D) / (PCC_O + epsilon)
