"""Stage 2: Normalization.

Applies log2 transformation and median normalization as needed,
respecting data provenance flags.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_normalization_stage(cfg: dict, expr: pd.DataFrame) -> pd.DataFrame:
    """Apply normalization with provenance-aware routing.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration.
    expr : pd.DataFrame
        Expression matrix (samples x proteins).

    Returns
    -------
    pd.DataFrame
        Normalized expression matrix.
    """
    prov = cfg["data_provenance"]
    seq_cols = [c for c in expr.columns if c.startswith("seq.")]
    if not seq_cols:
        seq_cols = expr.select_dtypes(include=[np.number]).columns.tolist()

    if prov["cruchaga_qc_already_applied"]:
        logger.info("Cruchaga QC applied — skipping full normalization.")

        # Only log2 transform if not already done
        if not prov["log2_already_applied"]:
            logger.info("Applying log2 transform (values are in RFU space).")
            expr[seq_cols] = np.log2(expr[seq_cols].clip(lower=1e-10))
        else:
            logger.info("Log2 already applied — skipping.")

        # Skip median normalization (already done in Cruchaga pipeline)
        if prov.get("cruchaga_normalization_applied", True):
            logger.info("ANML normalization already applied — skipping median normalization.")
        else:
            from src.preprocessing.somascan_qc import median_normalize
            expr = median_normalize(expr, seq_cols)
    else:
        # Full normalization
        from src.preprocessing.somascan_qc import log2_transform, median_normalize

        expr = median_normalize(expr, seq_cols)
        expr = log2_transform(expr, seq_cols)

    # Save intermediate
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    expr.to_csv(output_dir / "expression_normalized.csv")

    logger.info("Normalization complete: %d proteins x %d samples", len(seq_cols), len(expr))
    return expr
