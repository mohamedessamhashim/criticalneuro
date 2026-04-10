"""Batch correction for SomaScan proteomics data.

Provides ComBat batch correction via rpy2/sva with an automatic
Python fallback (per-batch median centering) if R is unavailable.
Also provides PCA-based validation of batch correction quality.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

_R_AVAILABLE = None  # None = not yet tested; True = works; False = unavailable


def _check_r_available() -> bool:
    """Lazily test rpy2 availability. Result cached in _R_AVAILABLE."""
    global _R_AVAILABLE
    if _R_AVAILABLE is None:
        try:
            import rpy2.robjects  # noqa: F401
            _R_AVAILABLE = True
        except Exception:
            _R_AVAILABLE = False
            logger.info("rpy2 not available — will use Python fallback for batch correction")
    return bool(_R_AVAILABLE)


def run_combat_r(
    df: pd.DataFrame,
    seq_cols: list[str],
    batch_col: str,
    protected_cols: list[str],
    use_r_combat: bool = True,
) -> pd.DataFrame:
    """Apply ComBat batch correction.

    Uses R's sva::ComBat via rpy2 if available and use_r_combat is True,
    otherwise falls back to Python median-centering. Protected columns
    (biological variables) are included in the model matrix so they are
    not removed by batch correction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with protein and batch columns.
    seq_cols : list[str]
        List of protein column names.
    batch_col : str
        Name of the batch column (e.g., 'PlateId').
    protected_cols : list[str]
        Biological variables to protect (e.g., ['TRAJECTORY', 'AGE']).
    use_r_combat : bool
        If False, skip R and use Python fallback directly.

    Returns
    -------
    pd.DataFrame
        DataFrame with batch-corrected protein values.
    """
    if batch_col not in df.columns:
        logger.warning(
            "Batch column '%s' not found in DataFrame — skipping batch correction",
            batch_col,
        )
        return df

    n_batches = df[batch_col].nunique()
    if n_batches <= 1:
        logger.info(
            "Only %d batch(es) found — batch correction not needed", n_batches
        )
        return df

    logger.info(
        "Batch correction: %d samples, %d proteins, %d batches",
        len(df),
        len(seq_cols),
        n_batches,
    )

    if use_r_combat and _check_r_available():
        try:
            return _run_combat_r_impl(df, seq_cols, batch_col, protected_cols)
        except Exception as e:
            logger.warning(
                "R ComBat failed (%s) — falling back to Python median centering", e
            )
            return _median_center_fallback(df, seq_cols, batch_col)
    else:
        if not use_r_combat:
            logger.info("use_r_combat=False in config — using Python median-centering")
        else:
            logger.info("rpy2 not available — using Python median-centering fallback")
        return _median_center_fallback(df, seq_cols, batch_col)


def _run_combat_r_impl(
    df: pd.DataFrame,
    seq_cols: list[str],
    batch_col: str,
    protected_cols: list[str],
) -> pd.DataFrame:
    """Internal: run ComBat via rpy2."""
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.packages import importr
    numpy2ri.activate()
    pandas2ri.activate()
    sva = importr("sva")

    # Prepare protein matrix (proteins x samples, as ComBat expects)
    protein_matrix = df[seq_cols].values.T  # shape: (n_proteins, n_samples)

    # Prepare batch vector
    batch = df[batch_col].values

    # Build model matrix for protected variables
    available_protected = [c for c in protected_cols if c in df.columns]

    if available_protected:
        # Create design matrix from protected variables
        design_df = df[available_protected].copy()
        # Convert categorical to dummy variables
        for col in available_protected:
            if design_df[col].dtype == object or design_df[col].dtype.name == "category":
                dummies = pd.get_dummies(design_df[col], prefix=col, drop_first=True)
                design_df = design_df.drop(columns=[col])
                design_df = pd.concat([design_df, dummies], axis=1)

        # Fill NaN with column means for the model matrix
        design_df = design_df.fillna(design_df.mean())
        # Add intercept
        design_df.insert(0, "intercept", 1.0)
        mod = design_df.values
    else:
        mod = np.ones((len(df), 1))

    # Convert to R objects
    # Use flatten('F') for Fortran (column-major) order to match R's matrix() filling
    r_matrix = ro.r["matrix"](
        ro.FloatVector(protein_matrix.flatten('F')),
        nrow=protein_matrix.shape[0],
        ncol=protein_matrix.shape[1],
    )
    r_batch = ro.StrVector(batch.astype(str))
    r_mod = ro.r["matrix"](
        ro.FloatVector(mod.flatten('F')), nrow=mod.shape[0], ncol=mod.shape[1]
    )

    # Run ComBat
    corrected = sva.ComBat(dat=r_matrix, batch=r_batch, mod=r_mod)
    corrected_np = np.array(corrected).T  # back to samples x proteins

    df = df.copy()
    df[seq_cols] = corrected_np

    logger.info("ComBat batch correction completed via R/sva")
    return df


def _median_center_fallback(
    df: pd.DataFrame, seq_cols: list[str], batch_col: str
) -> pd.DataFrame:
    """Python fallback: per-batch median centering.

    For each batch, subtracts the batch median and adds the grand median
    per protein. This removes systematic batch offsets while preserving
    biological variation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with protein and batch columns.
    seq_cols : list[str]
        List of protein column names.
    batch_col : str
        Name of the batch column.

    Returns
    -------
    pd.DataFrame
        DataFrame with batch-corrected protein values.
    """
    df = df.copy()

    # Grand median per protein (across all batches)
    grand_median = df[seq_cols].median(axis=0)

    for batch, batch_df in df.groupby(batch_col):
        batch_median = batch_df[seq_cols].median(axis=0)
        correction = grand_median - batch_median
        df.loc[batch_df.index, seq_cols] = batch_df[seq_cols].add(correction, axis=1)

    logger.info(
        "Python median-centering batch correction completed (%d batches)",
        df[batch_col].nunique(),
    )
    return df


def validate_batch_correction(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    seq_cols: list[str],
    batch_col: str,
    biological_col: str,
    output_dir: str,
    random_seed: int = 42,
) -> None:
    """Generate PCA validation plots for batch correction.

    Produces a 2x2 grid: (before/after correction) x (colored by batch /
    colored by biological group). Saved as PDF.

    Parameters
    ----------
    df_before : pd.DataFrame
        DataFrame before batch correction.
    df_after : pd.DataFrame
        DataFrame after batch correction.
    seq_cols : list[str]
        List of protein column names.
    batch_col : str
        Batch column name.
    biological_col : str
        Biological grouping column (e.g., 'TRAJECTORY').
    output_dir : str
        Directory to save the validation plot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    datasets = [
        ("Before correction", df_before),
        ("After correction", df_after),
    ]
    color_cols = [batch_col, biological_col]
    col_labels = [f"Colored by {batch_col}", f"Colored by {biological_col}"]

    for row, (title, data) in enumerate(datasets):
        # Compute PCA
        protein_matrix = data[seq_cols].values.copy()
        col_medians = np.nanmedian(protein_matrix, axis=0)
        nan_mask = np.isnan(protein_matrix)
        for j in range(protein_matrix.shape[1]):
            protein_matrix[nan_mask[:, j], j] = col_medians[j]

        n_components = min(2, protein_matrix.shape[1], len(data) - 1)
        pca = PCA(n_components=n_components, random_state=random_seed)
        scores = pca.fit_transform(protein_matrix)

        for col_idx, (color_col, col_label) in enumerate(
            zip(color_cols, col_labels)
        ):
            ax = axes[row, col_idx]

            if color_col in data.columns:
                groups = data[color_col].fillna("Unknown")
                unique_groups = groups.unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))

                for g, c in zip(unique_groups, colors):
                    mask = groups == g
                    ax.scatter(
                        scores[mask, 0],
                        scores[mask, 1],
                        c=[c],
                        label=str(g)[:20],
                        alpha=0.5,
                        s=10,
                    )

                if len(unique_groups) <= 10:
                    ax.legend(fontsize=6, markerscale=2)
            else:
                ax.scatter(scores[:, 0], scores[:, 1], alpha=0.5, s=10)

            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            ax.set_title(f"{title}\n{col_label}")

    plt.tight_layout()
    fig_path = output_path / "pca_batch_correction_validation.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Batch correction validation plot saved to %s", fig_path)
