"""Olink NPX data loading and QC.

Loads Olink Explore NPX files for ADNI and PPMI cohorts. Olink NPX
values are already log2-transformed — this loader never re-transforms.
All column naming follows the NPX_PROTEINNAME convention (Rule 2).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.somascan_qc import residualize_covariates

logger = logging.getLogger(__name__)


def _identify_npx_columns(df: pd.DataFrame) -> list[str]:
    """Return sorted list of Olink NPX protein columns (NPX_* prefix)."""
    return sorted([c for c in df.columns if c.startswith("NPX_")])


def load_olink_adni(config: dict) -> pd.DataFrame:
    """Load all Olink NPX CSV files from ADNI and pivot to wide format.

    Olink NPX files typically arrive in long format (one row per
    sample-protein pair). This function pivots to wide format where
    each protein gets a column named NPX_PROTEINNAME.

    Parameters
    ----------
    config : dict
        Configuration with paths.adni_olink_dir.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with columns: RID, VISCODE, EXAMDATE,
        PLATFORM, and NPX_* protein columns.

    Raises
    ------
    FileNotFoundError
        If no Olink CSV files are found.
    """
    olink_dir = Path(config["paths"]["adni_olink_dir"])
    csv_files = sorted(olink_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No Olink CSV files found in {olink_dir}. "
            "Download Olink NPX files from ADNI IDA."
        )

    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        frames.append(df)
        logger.info("Loaded Olink file: %s (%d rows)", f.name, len(df))

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined Olink data: %d rows from %d files", len(combined), len(csv_files))

    # Standardize column names to uppercase for matching
    combined.columns = [c.upper() for c in combined.columns]

    # Parse RID and VISCODE from sample identifiers
    # ADNI Olink files may have RID/VISCODE directly or as a combined SampleId
    if "RID" not in combined.columns:
        if "SAMPLEID" in combined.columns:
            # Parse RID_VISCODE pattern from SampleId
            parts = combined["SAMPLEID"].str.extract(r"(\d+)_(\S+)")
            combined["RID"] = pd.to_numeric(parts[0], errors="coerce")
            combined["VISCODE"] = parts[1]
        else:
            raise ValueError(
                "Olink data has no RID or SAMPLEID column. "
                "Check file format and update olink_loader.py."
            )

    if "VISCODE" not in combined.columns and "VISIT" in combined.columns:
        combined = combined.rename(columns={"VISIT": "VISCODE"})

    # Parse EXAMDATE if present
    for date_col in ["EXAMDATE", "DATE", "ANALYSISDATE"]:
        if date_col in combined.columns:
            combined["EXAMDATE"] = pd.to_datetime(combined[date_col], errors="coerce")
            break

    # Determine if data is long or wide format
    # Long format indicators: columns like ASSAY/OLINKID/PROTEINNAME and NPX
    is_long = any(c in combined.columns for c in ["ASSAY", "OLINKID", "ASSAYNAME"])

    if is_long:
        # Identify protein name and NPX value columns
        protein_col = None
        for candidate in ["ASSAY", "ASSAYNAME", "PROTEINNAME", "OLINKID"]:
            if candidate in combined.columns:
                protein_col = candidate
                break

        npx_col = None
        for candidate in ["NPX", "NPX_VALUE", "VALUE"]:
            if candidate in combined.columns:
                npx_col = candidate
                break

        if protein_col is None or npx_col is None:
            raise ValueError(
                f"Cannot identify protein name and NPX columns. "
                f"Available columns: {list(combined.columns)}"
            )

        # Pivot to wide format
        id_cols = ["RID", "VISCODE"]
        if "EXAMDATE" in combined.columns:
            id_cols.append("EXAMDATE")

        pivot = combined.pivot_table(
            index=id_cols,
            columns=protein_col,
            values=npx_col,
            aggfunc="first",
        ).reset_index()

        # Add NPX_ prefix to protein columns
        protein_names = [c for c in pivot.columns if c not in id_cols]
        rename_map = {name: f"NPX_{name}" for name in protein_names}
        pivot = pivot.rename(columns=rename_map)
        combined = pivot

    # Add PLATFORM column (Rule 3)
    combined["PLATFORM"] = "olink"

    npx_cols = _identify_npx_columns(combined)
    logger.info(
        "ADNI Olink loaded: %d participants, %d visits, %d proteins",
        combined["RID"].nunique(),
        len(combined),
        len(npx_cols),
    )

    return combined


def load_olink_ppmi(config: dict) -> pd.DataFrame:
    """Load all Olink NPX CSV files from PPMI.

    Same logic as load_olink_adni but renames PATNO→RID and
    EVENT_ID→VISCODE immediately (Rule 3).

    Parameters
    ----------
    config : dict
        Configuration with paths.ppmi_olink_dir.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with NPX_* columns.
    """
    olink_dir = Path(config["paths"]["ppmi_olink_dir"])
    csv_files = sorted(olink_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No Olink CSV files found in {olink_dir}. "
            "Download Olink NPX files from PPMI."
        )

    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        frames.append(df)
        logger.info("Loaded PPMI Olink file: %s (%d rows)", f.name, len(df))

    combined = pd.concat(frames, ignore_index=True)
    combined.columns = [c.upper() for c in combined.columns]

    # PPMI uses PATNO → RID, EVENT_ID → VISCODE
    if "PATNO" in combined.columns:
        combined = combined.rename(columns={"PATNO": "RID"})
    if "EVENT_ID" in combined.columns:
        combined = combined.rename(columns={"EVENT_ID": "VISCODE"})

    # Same long→wide pivot logic as ADNI
    is_long = any(c in combined.columns for c in ["ASSAY", "OLINKID", "ASSAYNAME"])

    if is_long:
        protein_col = None
        for candidate in ["ASSAY", "ASSAYNAME", "PROTEINNAME", "OLINKID"]:
            if candidate in combined.columns:
                protein_col = candidate
                break

        npx_col = None
        for candidate in ["NPX", "NPX_VALUE", "VALUE"]:
            if candidate in combined.columns:
                npx_col = candidate
                break

        if protein_col and npx_col:
            id_cols = ["RID", "VISCODE"]
            pivot = combined.pivot_table(
                index=id_cols,
                columns=protein_col,
                values=npx_col,
                aggfunc="first",
            ).reset_index()

            protein_names = [c for c in pivot.columns if c not in id_cols]
            rename_map = {name: f"NPX_{name}" for name in protein_names}
            pivot = pivot.rename(columns=rename_map)
            combined = pivot

    combined["PLATFORM"] = "olink"

    npx_cols = _identify_npx_columns(combined)
    logger.info(
        "PPMI Olink loaded: %d participants, %d proteins",
        combined["RID"].nunique(),
        len(npx_cols),
    )

    return combined


def apply_olink_qc(
    df: pd.DataFrame,
    npx_cols: list[str],
    config: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply QC pipeline to Olink NPX data.

    Olink NPX values are already log2-transformed — this function
    never re-transforms them (config olink.npx_already_log_transformed).

    QC steps:
    1. Remove proteins where < min_qc_pass_fraction of samples pass QC
    2. Residualize covariates (age, sex, APOE4) identically to SomaScan

    Parameters
    ----------
    df : pd.DataFrame
        Olink DataFrame with NPX_* columns.
    npx_cols : list[str]
        List of NPX protein column names.
    config : dict
        Configuration with olink section.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (cleaned DataFrame, list of proteins passing QC)
    """
    olink_cfg = config["olink"]
    df = df.copy()

    # Step 1: Filter proteins by QC pass fraction
    min_qc = olink_cfg["min_qc_pass_fraction"]

    # If QC_PASS column exists, use it; otherwise use detectability (non-NaN fraction)
    if "QC_PASS" in df.columns or "QC_WARNING" in df.columns:
        qc_col = "QC_PASS" if "QC_PASS" in df.columns else "QC_WARNING"
        passing_proteins = []
        for col in npx_cols:
            # Count fraction of samples where protein passed QC
            pass_frac = (df[qc_col] == "PASS").mean() if df[qc_col].dtype == object else 1.0
            if pass_frac >= min_qc:
                passing_proteins.append(col)
    else:
        # Fallback: use non-NaN fraction as proxy
        passing_proteins = []
        for col in npx_cols:
            detect_frac = df[col].notna().mean()
            if detect_frac >= min_qc:
                passing_proteins.append(col)

    n_removed = len(npx_cols) - len(passing_proteins)
    logger.info(
        "Olink QC: %d/%d proteins pass (removed %d with <%.0f%% detection)",
        len(passing_proteins),
        len(npx_cols),
        n_removed,
        min_qc * 100,
    )

    # Verify NPX values are NOT re-log-transformed
    if olink_cfg.get("npx_already_log_transformed", True):
        logger.info("Olink NPX values are already log2-transformed — skipping log transform")

    # Step 2: Residualize covariates
    covariates = olink_cfg.get("covariates_to_residualize", [])
    if covariates:
        df = residualize_covariates(df, passing_proteins, covariates)
        logger.info("Residualized covariates: %s", covariates)

    return df, passing_proteins
