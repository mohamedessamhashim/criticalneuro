"""SomaScan proteomics quality control pipeline.

Runs QC in a fixed order that must not be changed:
1. Filter proteins by detectability
2. Remove sample outliers
3. Median normalize
4. Log2 transform
5. Residualize covariates

Imputation is deliberately excluded from the main pipeline and called
separately only for the cross-sectional DNB analysis subset (Rule 10).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def identify_seq_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns starting with 'seq.' (Rule 2).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame potentially containing SomaScan analyte columns.

    Returns
    -------
    list[str]
        Sorted list of column names starting with 'seq.'.
    """
    seq_cols = sorted([c for c in df.columns if c.startswith("seq.")])
    logger.info("Identified %d seq.* protein columns", len(seq_cols))
    return seq_cols


def filter_proteins_by_detectability(
    df: pd.DataFrame, seq_cols: list[str], min_detectability: float
) -> tuple[pd.DataFrame, list[str]]:
    """Remove proteins detected in fewer than min_detectability fraction of samples.

    Uses the 10th percentile of each protein as an LLOD proxy when no
    explicit LLOD file is available. A sample is considered "detected"
    if its value is above this proxy threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with protein columns.
    seq_cols : list[str]
        List of protein column names.
    min_detectability : float
        Minimum fraction of samples that must be above LLOD (e.g. 0.80).

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (filtered DataFrame, list of remaining protein columns)
    """
    n_samples = len(df)
    protein_matrix = df[seq_cols]

    # Use 10th percentile per protein as LLOD proxy
    llod_proxy = protein_matrix.quantile(0.10)
    detectability = (protein_matrix > llod_proxy).sum() / n_samples

    passing = detectability[detectability >= min_detectability].index.tolist()
    removed = len(seq_cols) - len(passing)

    logger.info(
        "Protein detectability filter: %d/%d proteins passed (%.1f%% threshold), "
        "%d removed",
        len(passing),
        len(seq_cols),
        min_detectability * 100,
        removed,
    )

    drop_cols = [c for c in seq_cols if c not in passing]
    df_filtered = df.drop(columns=drop_cols)
    return df_filtered, passing


def remove_sample_outliers(
    df: pd.DataFrame,
    seq_cols: list[str],
    sd_threshold: float,
    n_components: int = 50,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Remove sample outliers using PCA-based distance.

    Fits PCA on the protein matrix, computes Euclidean distance from the
    centroid in PC space, and removes samples whose distance exceeds
    sd_threshold standard deviations above the mean distance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with protein columns.
    seq_cols : list[str]
        List of protein column names.
    sd_threshold : float
        Number of SDs above mean distance for outlier cutoff.
    n_components : int
        Number of PCA components (capped automatically).

    Returns
    -------
    pd.DataFrame
        DataFrame with outlier samples removed.
    """
    protein_matrix = df[seq_cols].values.copy()

    # Handle NaN for PCA: fill with column median
    col_medians = np.nanmedian(protein_matrix, axis=0)
    nan_mask = np.isnan(protein_matrix)
    for j in range(protein_matrix.shape[1]):
        protein_matrix[nan_mask[:, j], j] = col_medians[j]

    n_components_actual = min(n_components, protein_matrix.shape[1], len(df) - 1)
    logger.info(
        "PCA outlier detection: %d samples, %d proteins, %d components",
        len(df),
        len(seq_cols),
        n_components_actual,
    )

    pca = PCA(n_components=n_components_actual, random_state=random_seed)
    scores = pca.fit_transform(protein_matrix)

    # Euclidean distance from centroid in PC space
    centroid = scores.mean(axis=0)
    distances = np.sqrt(((scores - centroid) ** 2).sum(axis=1))

    mean_dist = distances.mean()
    std_dist = distances.std()
    threshold = mean_dist + sd_threshold * std_dist

    outlier_mask = distances > threshold
    n_outliers = outlier_mask.sum()

    if n_outliers > 0 and "RID" in df.columns:
        outlier_rids = df.loc[outlier_mask, "RID"].tolist()
        logger.info(
            "Removed %d outlier samples (RIDs: %s)", n_outliers, outlier_rids
        )
    else:
        logger.info("Removed %d outlier samples", n_outliers)

    return df.loc[~outlier_mask].reset_index(drop=True)


def median_normalize(df: pd.DataFrame, seq_cols: list[str]) -> pd.DataFrame:
    """Per-sample median normalization.

    For each sample, divides all protein values by the sample's own median,
    then multiplies by the cohort-wide median of medians. This corrects for
    differences in total protein load between samples.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with protein columns.
    seq_cols : list[str]
        List of protein column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with median-normalized protein values.
    """
    df = df.copy()
    protein_matrix = df[seq_cols]

    sample_medians = protein_matrix.median(axis=1)
    cohort_median = sample_medians.median()

    # Avoid division by zero for samples with median == 0
    sample_medians = sample_medians.replace(0, np.nan)

    normalized = protein_matrix.div(sample_medians, axis=0) * cohort_median
    df[seq_cols] = normalized

    logger.info(
        "Median normalization: cohort median = %.4f, %d samples normalized",
        cohort_median,
        len(df),
    )
    return df


def log2_transform(df: pd.DataFrame, seq_cols: list[str]) -> pd.DataFrame:
    """Apply log2 transformation to protein values.

    Clips all values to a minimum of 1.0 before transformation to avoid
    log(0) or log of negative values. This assumes RFU values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with protein columns.
    seq_cols : list[str]
        List of protein column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with log2-transformed protein values.
    """
    df = df.copy()
    protein_matrix = df[seq_cols]

    # Clip to floor of 1.0 to prevent log(0) or log(negative)
    clipped = protein_matrix.clip(lower=1.0)
    df[seq_cols] = np.log2(clipped)

    n_clipped = (protein_matrix < 1.0).sum().sum()
    if n_clipped > 0:
        logger.info(
            "Log2 transform: %d values clipped to floor of 1.0 before transform",
            n_clipped,
        )
    logger.info("Log2 transform applied to %d proteins", len(seq_cols))
    return df


def residualize_covariates(
    df: pd.DataFrame, seq_cols: list[str], covariates: list[str]
) -> pd.DataFrame:
    """Regress out covariates from each protein, keeping residuals + mean.

    For each protein, fits a linear regression of protein ~ covariates
    using only samples where all covariates are non-missing. Replaces
    protein values with residuals plus the original protein mean.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with protein and covariate columns.
    seq_cols : list[str]
        List of protein column names.
    covariates : list[str]
        List of covariate column names to regress out.

    Returns
    -------
    pd.DataFrame
        DataFrame with residualized protein values.
    """
    df = df.copy()

    # Check which covariates are available
    available_covariates = [c for c in covariates if c in df.columns]
    missing_covariates = [c for c in covariates if c not in df.columns]

    if missing_covariates:
        logger.warning(
            "Covariates not found in DataFrame, skipping: %s", missing_covariates
        )

    if not available_covariates:
        logger.warning("No covariates available for residualization, skipping")
        return df

    # Mask for samples with all covariates present
    covariate_matrix = df[available_covariates]
    valid_mask = covariate_matrix.notna().all(axis=1)
    n_valid = valid_mask.sum()

    logger.info(
        "Residualizing %d proteins against covariates %s (%d/%d samples with "
        "complete covariates)",
        len(seq_cols),
        available_covariates,
        n_valid,
        len(df),
    )

    X = covariate_matrix.loc[valid_mask].values.astype(float)

    for col in seq_cols:
        y = df.loc[valid_mask, col].values.astype(float)

        # Skip proteins with NaN in the valid subset
        protein_valid = ~np.isnan(y)
        if protein_valid.sum() < len(available_covariates) + 1:
            continue

        X_fit = X[protein_valid]
        y_fit = y[protein_valid]
        y_mean = y_fit.mean()

        model = LinearRegression()
        model.fit(X_fit, y_fit)

        # Compute residuals for all valid-covariate samples
        predicted = model.predict(X)
        residuals = y - predicted + y_mean

        df.loc[valid_mask, col] = residuals

    return df


def impute_missing_values(
    df: pd.DataFrame, seq_cols: list[str], method: str
) -> pd.DataFrame:
    """Impute missing protein values.

    IMPORTANT: Only apply to the cross-sectional DNB analysis subset.
    Never impute values for the longitudinal CSD analysis (Rule 10).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with protein columns.
    seq_cols : list[str]
        List of protein column names.
    method : str
        Imputation method. Currently supports 'half_min'.

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values.
    """
    df = df.copy()

    if method == "half_min":
        n_imputed = 0
        for col in seq_cols:
            na_mask = df[col].isna()
            n_na = na_mask.sum()
            if n_na > 0:
                col_min = df[col].min()
                df.loc[na_mask, col] = col_min / 2.0
                n_imputed += n_na

        logger.info(
            "Imputed %d missing values across %d proteins using %s method",
            n_imputed,
            len(seq_cols),
            method,
        )
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    return df


def run_somascan_qc_pipeline(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, list[str]]:
    """Execute the full SomaScan QC pipeline in fixed order.

    Order: detectability filter -> outlier removal -> median normalization ->
    log2 transform -> residualize covariates.

    Imputation is NOT included here. It must be called separately and only
    on the cross-sectional DNB analysis subset (Rule 10).

    Parameters
    ----------
    df : pd.DataFrame
        Raw proteomics DataFrame.
    config : dict
        Configuration dictionary with 'proteomics' section.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (cleaned DataFrame, list of protein columns that passed QC)
    """
    proteomics_cfg = config["proteomics"]

    logger.info(
        "Starting SomaScan QC pipeline: %d samples, %d columns",
        len(df),
        len(df.columns),
    )

    # Step 1: Identify protein columns
    seq_cols = identify_seq_columns(df)
    if not seq_cols:
        raise ValueError("No seq.* columns found in DataFrame")

    # Step 2: Filter by detectability
    df, seq_cols = filter_proteins_by_detectability(
        df, seq_cols, proteomics_cfg["min_detectability"]
    )

    # Step 3: Remove sample outliers
    df = remove_sample_outliers(
        df,
        seq_cols,
        sd_threshold=proteomics_cfg["outlier_mahalanobis_sd"],
        n_components=proteomics_cfg.get("outlier_pca_components", 50),
        random_seed=config["random_seed"],
    )

    # Step 4: Median normalize
    df = median_normalize(df, seq_cols)

    # Step 5: Log2 transform
    df = log2_transform(df, seq_cols)

    # Step 6: Residualize covariates
    covariates = proteomics_cfg.get("covariates_to_residualize", [])
    if covariates:
        df = residualize_covariates(df, seq_cols, covariates)

    logger.info(
        "SomaScan QC pipeline complete: %d samples, %d proteins retained",
        len(df),
        len(seq_cols),
    )
    return df, seq_cols
