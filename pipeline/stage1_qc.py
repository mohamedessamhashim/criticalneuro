"""Stage 1: Quality Control.

Handles both cross-sectional (ADNI/PPMI) and longitudinal (Knight-ADRC) data.
Respects data provenance flags — skips QC if data is already QC'd.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_METADATA_COLS_LONGITUDINAL = [
    "SampleID", "SubjectID", "Age", "Sex", "APOE4", "Plate",
    "Diagnosis", "VisitDate", "VisitNumber", "Converter", "VisitsToDx",
]


def load_knight_adrc_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Knight-ADRC expression matrix and metadata.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (expression_df, metadata_df)
    """
    expr_path = Path(cfg["input"]["expression_matrix"])
    meta_path = Path(cfg["input"]["metadata"])

    if not expr_path.exists():
        raise FileNotFoundError(
            f"Expression matrix not found at {expr_path}.\n"
            "Set input.expression_matrix in config.yaml to your data file path."
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {meta_path}.\n"
            "Set input.metadata in config.yaml to your metadata file path."
        )

    expr = pd.read_csv(expr_path, index_col=0)
    metadata = pd.read_csv(meta_path)

    logger.info("Loaded expression: %d samples x %d proteins", expr.shape[0], expr.shape[1])
    logger.info("Loaded metadata: %d rows", len(metadata))

    # Validate metadata columns
    missing_cols = [c for c in REQUIRED_METADATA_COLS_LONGITUDINAL if c not in metadata.columns]
    if missing_cols:
        raise ValueError(
            f"Metadata is missing required columns: {missing_cols}\n"
            "See data/README_data.md for column specifications."
        )

    # Validate sample overlap
    expr_samples = set(expr.index.astype(str))
    meta_samples = set(metadata["SampleID"].astype(str))
    overlap = expr_samples & meta_samples

    if len(overlap) == 0:
        raise ValueError(
            "No sample IDs match between expression matrix row names and metadata SampleID column.\n"
            "Ensure expression matrix index matches metadata SampleID."
        )

    if len(overlap) < len(expr_samples):
        logger.warning(
            "%d/%d expression samples not in metadata — these will be excluded",
            len(expr_samples) - len(overlap), len(expr_samples),
        )

    logger.info("Sample overlap: %d samples", len(overlap))
    return expr, metadata


def run_qc_stage(cfg: dict) -> pd.DataFrame:
    """Run QC stage with provenance-aware routing.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    pd.DataFrame
        QC'd expression matrix (proteins x samples).
    """
    prov = cfg["data_provenance"]
    mode = cfg["analysis_mode"]

    if mode == "longitudinal":
        expr, metadata = load_knight_adrc_data(cfg)

        if prov["cruchaga_qc_already_applied"]:
            logger.info(
                "DATA PROVENANCE: Cruchaga Lab QC already applied. "
                "Skipping QC filters (detectability, outlier removal)."
            )
            logger.warning(
                "ASSUMPTION: Expression matrix is post-ANML, post-Cruchaga-QC. "
                "If this is wrong, set cruchaga_qc_already_applied: false in config.yaml."
            )
            # Still identify protein columns
            seq_cols = [c for c in expr.columns if c.startswith("seq.")]
            if not seq_cols:
                # Try all numeric columns
                seq_cols = expr.select_dtypes(include=[np.number]).columns.tolist()
            logger.info("Identified %d protein columns", len(seq_cols))
        else:
            logger.info("DATA PROVENANCE: Raw SomaScan data. Running full QC.")
            seq_cols = [c for c in expr.columns if c.startswith("seq.")]
            expr, seq_cols = _run_full_qc(expr, seq_cols, cfg)

        # Save intermediate
        output_dir = Path(cfg["output"]["dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Merge expression with metadata for downstream stages
        expr_t = expr[seq_cols].T  # proteins x samples
        expr_t.to_csv(output_dir / "expression_qc.csv")
        metadata.to_csv(output_dir / "metadata_staged.csv", index=False)

        logger.info("QC stage complete: %d proteins x %d samples", len(seq_cols), expr.shape[0])
        return expr

    else:
        # Cross-sectional mode: delegate to existing pipeline
        logger.info("Cross-sectional mode: use pipelines/run_full_pipeline.py for ADNI/PPMI QC")
        raise NotImplementedError(
            "Cross-sectional QC uses the existing pipelines/run_full_pipeline.py. "
            "Run: python pipelines/run_full_pipeline.py --stage adni_preprocess"
        )


def _run_full_qc(
    expr: pd.DataFrame, seq_cols: list[str], cfg: dict
) -> tuple[pd.DataFrame, list[str]]:
    """Run full QC pipeline on raw SomaScan data.

    Reuses existing QC functions from src/preprocessing/somascan_qc.py.
    """
    from src.preprocessing.somascan_qc import (
        filter_proteins_by_detectability,
        log2_transform,
        median_normalize,
        remove_sample_outliers,
    )

    qc_cfg = cfg["qc"]

    # Step 1: Detectability filter
    expr, seq_cols = filter_proteins_by_detectability(
        expr, seq_cols, qc_cfg["detectability_threshold"]
    )

    # Step 2: Outlier removal
    expr = remove_sample_outliers(
        expr, seq_cols, qc_cfg["outlier_sd_threshold"],
        n_components=qc_cfg["outlier_pca_components"],
        random_seed=cfg["reproducibility"]["global_seed"],
    )

    # Step 3: Median normalization
    expr = median_normalize(expr, seq_cols)

    # Step 4: Log2 transform (only if not already done)
    if not cfg["data_provenance"]["log2_already_applied"]:
        expr = log2_transform(expr, seq_cols)

    logger.info("Full QC complete: %d proteins x %d samples", len(seq_cols), len(expr))
    return expr, seq_cols
