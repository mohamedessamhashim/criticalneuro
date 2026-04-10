"""Stage 3b: Batch correction via ComBat.

Wraps existing src/preprocessing/batch_correction.py with provenance awareness.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def run_batch_correction_stage(cfg: dict, expr: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Apply ComBat batch correction if needed.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration.
    expr : pd.DataFrame
        Expression matrix (samples x proteins).
    metadata : pd.DataFrame
        Sample metadata with batch variable.

    Returns
    -------
    pd.DataFrame
        Batch-corrected expression matrix.
    """
    prov = cfg["data_provenance"]
    batch_cfg = cfg["batch_correction"]

    if prov["combat_already_applied"]:
        logger.info("ComBat already applied — skipping batch correction.")
        return expr

    if batch_cfg["method"] == "none":
        logger.info("Batch correction method set to 'none' — skipping.")
        return expr

    batch_var = batch_cfg["batch_variable"]
    if batch_var not in metadata.columns:
        logger.warning(
            "Batch variable '%s' not found in metadata — skipping batch correction.",
            batch_var,
        )
        return expr

    # Merge batch info into expression
    import numpy as np
    seq_cols = [c for c in expr.columns if c.startswith("seq.")]
    if not seq_cols:
        seq_cols = expr.select_dtypes(include=[np.number]).columns.tolist()

    from src.preprocessing.batch_correction import run_combat_r

    # Ensure batch column is in the dataframe
    if batch_var not in expr.columns:
        expr = expr.merge(
            metadata[["SampleID", batch_var]].drop_duplicates(),
            left_index=True, right_on="SampleID", how="left",
        ).set_index("SampleID")

    protected = batch_cfg.get("protected_variables", [])
    expr_corrected = run_combat_r(
        expr, seq_cols, batch_var, protected,
        use_r_combat=True,
    )

    # Save intermediate
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    expr_corrected.to_csv(output_dir / "expression_batch_corrected.csv")

    logger.info("Batch correction complete")
    return expr_corrected
