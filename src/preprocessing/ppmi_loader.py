"""PPMI data loading and preprocessing.

Mirror of adni_loader.py for the Parkinson's Progression Markers Initiative.
Key differences: PATNO -> RID, EVENT_ID -> VISCODE, progression labels
based on MDS-UPDRS III tertiles instead of diagnostic conversion.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.somascan_qc import identify_seq_columns

logger = logging.getLogger(__name__)


def load_ppmi_clinical(config: dict) -> pd.DataFrame:
    """Load and merge PPMI clinical data files.

    Reads MDS-UPDRS Part III, Demographics, Diagnosis History, and MoCA,
    renames PATNO -> RID and EVENT_ID -> VISCODE (Rule 3), parses dates,
    and merges all on RID + VISCODE.

    Parameters
    ----------
    config : dict
        Configuration dictionary with paths.ppmi_clinical_dir.

    Returns
    -------
    pd.DataFrame
        Merged clinical DataFrame.
    """
    clinical_dir = Path(config["paths"]["ppmi_clinical_dir"])

    if not clinical_dir.exists():
        raise FileNotFoundError(
            f"PPMI clinical directory not found at {clinical_dir}. "
            "Download clinical data from ppmi-info.org."
        )

    expected_files = {
        "updrs": "MDS-UPDRS_Part_III.csv",
        "demographics": "Demographics.csv",
        "diagnosis": "Diagnosis_History.csv",
        "moca": "Montreal_Cognitive_Assessment__MoCA.csv",
    }

    loaded = {}
    for key, filename in expected_files.items():
        filepath = clinical_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, low_memory=False)
            df.columns = df.columns.str.upper()
            loaded[key] = df
            logger.info("Loaded PPMI %s: %d rows", key, len(df))
        else:
            logger.warning("PPMI clinical file not found: %s", filepath)

    if not loaded:
        raise FileNotFoundError(
            f"No clinical files found in {clinical_dir}. "
            "Expected: MDS_UPDRS_Part_III.csv, Demographics.csv, etc."
        )

    # Start with the first available file
    merged = list(loaded.values())[0]

    # Standardize ID columns before merging
    merged = _standardize_ppmi_ids(merged)

    for key, df in list(loaded.items())[1:]:
        df = _standardize_ppmi_ids(df)
        merge_cols = ["RID", "VISCODE"]
        available_merge = [c for c in merge_cols if c in df.columns and c in merged.columns]

        if len(available_merge) >= 1:
            new_cols = [
                c for c in df.columns
                if c not in merged.columns or c in available_merge
            ]
            merged = merged.merge(df[new_cols], on=available_merge, how="outer")

    # Force-apply SEX and AGE from Demographics.csv, regardless of merge order.
    # Other PPMI tables (e.g. MDS-UPDRS) carry a SEX column with all-NaN values
    # which blocks demographics SEX from being added by the standard merge logic.
    demo_path = clinical_dir / "Demographics.csv"
    if demo_path.exists():
        demo = pd.read_csv(demo_path, low_memory=False)
        demo.columns = demo.columns.str.upper()
        if "PATNO" in demo.columns:
            demo = demo.rename(columns={"PATNO": "RID"})
        demo["RID"] = pd.to_numeric(demo["RID"], errors="coerce")
        demo_keep = [c for c in ["RID", "SEX", "BIRTHDT"] if c in demo.columns]
        demo_bl = (
            demo[demo_keep]
            .dropna(subset=["RID"])
            .drop_duplicates(subset=["RID"])
        )
        # Drop stale SEX/BIRTHDT that may have come from other tables (all NaN)
        merged = merged.drop(
            columns=[c for c in ["SEX", "BIRTHDT"] if c in merged.columns],
            errors="ignore",
        )
        merged = merged.merge(demo_bl, on="RID", how="left")
        # Compute AGE from BIRTHDT ("MM/YYYY") and EXAMDATE
        if "BIRTHDT" in merged.columns and "EXAMDATE" in merged.columns:
            birth_year = pd.to_datetime(
                merged["BIRTHDT"], format="%m/%Y", errors="coerce"
            ).dt.year
            exam_year = pd.to_datetime(
                merged["EXAMDATE"], errors="coerce"
            ).dt.year
            merged["AGE"] = (exam_year - birth_year).astype(float)
            merged = merged.drop(columns=["BIRTHDT"])
        n_sex = merged["SEX"].notna().sum() if "SEX" in merged.columns else 0
        n_age = merged["AGE"].notna().sum() if "AGE" in merged.columns else 0
        logger.info(
            "Demographics force-applied: SEX non-NaN=%d, AGE non-NaN=%d",
            n_sex, n_age,
        )

    logger.info(
        "PPMI clinical merged: %d rows, %d participants",
        len(merged),
        merged["RID"].nunique() if "RID" in merged.columns else 0,
    )
    return merged


def _standardize_ppmi_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Rename PATNO -> RID and EVENT_ID -> VISCODE, parse dates."""
    if "PATNO" in df.columns:
        df = df.rename(columns={"PATNO": "RID"})
    if "EVENT_ID" in df.columns:
        df = df.rename(columns={"EVENT_ID": "VISCODE"})
        df["VISCODE"] = df["VISCODE"].astype(str).str.lower()

    # Parse date columns -> EXAMDATE
    for candidate in ["INFODT", "ORIG_ENTRY", "EXAMDATE", "LAST_UPDATE"]:
        if candidate in df.columns:
            df["EXAMDATE"] = pd.to_datetime(df[candidate], errors="coerce")
            break

    return df


def load_somascan_ppmi(config: dict) -> pd.DataFrame:
    """Load PPMI SomaScan CSF proteomics data.

    Parameters
    ----------
    config : dict
        Configuration dictionary with paths.ppmi_somascan_dir.

    Returns
    -------
    pd.DataFrame
        SomaScan DataFrame with RID, VISCODE, PlateId, and seq.* columns.
    """
    somascan_dir = Path(config["paths"]["ppmi_somascan_dir"])

    if not somascan_dir.exists():
        raise FileNotFoundError(
            f"PPMI SomaScan directory not found at {somascan_dir}. "
            "Download SomaScan data from ppmi-info.org."
        )

    csv_files = sorted(somascan_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {somascan_dir}.")

    dfs = []
    for f in csv_files:
        logger.info("Reading PPMI SomaScan file: %s", f.name)
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    logger.info(
        "Concatenated %d PPMI SomaScan files: %d total rows",
        len(csv_files),
        len(df),
    )

    # Detect long format (one row per protein): TESTNAME + TESTVALUE columns
    if "TESTNAME" in df.columns and "TESTVALUE" in df.columns:
        logger.info("Detected long-format PPMI SomaScan data — pivoting to wide")
        df = _pivot_ppmi_somascan_long(df)
    else:
        # Wide format: standardize column names and apply R-export rename
        df.columns = df.columns.str.replace("-", ".")
        rename_map = {
            c: "seq." + c[1:]
            for c in df.columns
            if re.match(r"^X\d+\.\d+$", c)
        }
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info(
                "Renamed %d R-format SomaScan columns (X#.# -> seq.#.#)",
                len(rename_map),
            )

        # Parse RID / VISCODE from SampleId if present
        sample_id_col = None
        for candidate in ["SampleId", "SAMPLEID", "Sample_ID", "SampleID"]:
            if candidate in df.columns:
                sample_id_col = candidate
                break

        if sample_id_col:
            parsed = df[sample_id_col].astype(str).str.extract(
                r"(?:PPMI_)?(\d+)_([a-zA-Z0-9_.]+)$"
            )
            if parsed.notna().all(axis=1).any():
                df["RID"] = pd.to_numeric(parsed[0], errors="coerce")
                df["VISCODE"] = parsed[1].str.lower()
                n_failed = df["RID"].isna().sum()
                if n_failed > 0:
                    logger.warning(
                        "PPMI SampleId parsing: %d rows failed", n_failed
                    )

        df = _standardize_ppmi_ids(df)

    seq_cols = identify_seq_columns(df)

    # Identify PlateId
    plate_col = None
    for candidate in ["PlateId", "PLATEID", "Plate_ID"]:
        if candidate in df.columns:
            plate_col = candidate
            break

    keep_cols = ["RID", "VISCODE"]
    if plate_col:
        df = df.rename(columns={plate_col: "PlateId"})
        keep_cols.append("PlateId")
    keep_cols.extend(seq_cols)

    df = df[[c for c in keep_cols if c in df.columns]]

    logger.info(
        "PPMI SomaScan loaded: %d samples, %d proteins, %d participants",
        len(df),
        len(seq_cols),
        df["RID"].nunique() if "RID" in df.columns else 0,
    )
    return df


def _pivot_ppmi_somascan_long(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot PPMI SomaScan from long format to wide format.

    Long format: one row per (PATNO, CLINICAL_EVENT, TESTNAME).
    Wide format: one row per (RID, VISCODE), columns = seq.* proteins.

    TESTNAME format is 'NNNNN-NN_V' (e.g. '5632-6_3') where the suffix
    after '_' is a version number. This maps to seq.NNNNN.NN (Rule 2).
    """
    # Standardize IDs
    df = df.rename(columns={"PATNO": "RID", "CLINICAL_EVENT": "VISCODE"})
    df["RID"] = pd.to_numeric(df["RID"], errors="coerce")
    df["VISCODE"] = df["VISCODE"].astype(str).str.lower()

    # Convert TESTNAME: 'NNNNN-NN_V' -> 'seq.NNNNN.NN'
    def _to_seq_id(name: str) -> str:
        m = re.match(r"^(\d+)-(\d+)(?:_\d+)?$", str(name))
        if m:
            return f"seq.{m.group(1)}.{m.group(2)}"
        return name

    df["TESTNAME"] = df["TESTNAME"].apply(_to_seq_id)
    df["TESTVALUE"] = pd.to_numeric(df["TESTVALUE"], errors="coerce")

    # Carry PlateId per (RID, VISCODE) — take first plate for each sample
    plate_col = None
    for candidate in ["PLATEID", "PlateId", "Plate_ID"]:
        if candidate in df.columns:
            plate_col = candidate
            break

    meta_cols = ["RID", "VISCODE"]
    if plate_col:
        plate_map = (
            df.groupby(["RID", "VISCODE"])[plate_col]
            .first()
            .reset_index()
            .rename(columns={plate_col: "PlateId"})
        )

    # Pivot: rows = (RID, VISCODE), columns = seq.*
    wide = df.pivot_table(
        index=["RID", "VISCODE"],
        columns="TESTNAME",
        values="TESTVALUE",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    if plate_col:
        wide = wide.merge(plate_map, on=["RID", "VISCODE"], how="left")

    logger.info(
        "Pivoted long SomaScan: %d samples, %d protein columns",
        len(wide),
        sum(1 for c in wide.columns if c.startswith("seq.")),
    )
    return wide


def load_csf_biomarkers_ppmi(config: dict) -> pd.DataFrame | None:
    """Load PPMI CSF biomarker files.

    Reads alpha-synuclein, t-tau, p-tau181, Abeta42, NfL and APOE
    from the PPMI biomarkers directory. Long-format files are pivoted
    to wide format before returning.

    Parameters
    ----------
    config : dict
        Configuration dictionary with paths.ppmi_biomarkers_dir.

    Returns
    -------
    pd.DataFrame or None
        Merged biomarker DataFrame, or None if no files found.
    """
    biomarker_dir = Path(config["paths"]["ppmi_biomarkers_dir"])

    if not biomarker_dir.exists():
        logger.warning("PPMI biomarkers directory not found at %s", biomarker_dir)
        return None

    csv_files = sorted(biomarker_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("No biomarker CSV files found in %s", biomarker_dir)
        return None

    merged = None
    for f in csv_files:
        logger.info("Reading PPMI biomarker file: %s", f.name)
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.upper()

        # Detect long format (TESTNAME + TESTVALUE) and pivot to wide
        if "TESTNAME" in df.columns and "TESTVALUE" in df.columns:
            df = _pivot_ppmi_biomarkers_long(df)
        else:
            df = _standardize_ppmi_ids(df)

        if "RID" not in df.columns:
            logger.warning("Biomarker file %s missing RID, skipping", f.name)
            continue

        if merged is None:
            merged = df
        else:
            merge_cols = ["RID"]
            if "VISCODE" in df.columns and "VISCODE" in merged.columns:
                merge_cols.append("VISCODE")
            new_cols = [
                c for c in df.columns
                if c not in merged.columns or c in merge_cols
            ]
            merged = merged.merge(df[new_cols], on=merge_cols, how="outer")

    if merged is not None:
        logger.info(
            "PPMI biomarkers loaded: %d rows, %d participants",
            len(merged),
            merged["RID"].nunique(),
        )
    return merged


def _pivot_ppmi_biomarkers_long(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long-format PPMI biomarker file to wide format.

    Filters to key clinical biomarkers (alpha-syn, tau, Abeta42, NfL,
    APOE) to keep memory usage manageable, then pivots to one row per
    (RID, VISCODE).
    """
    # Standardize IDs
    df = df.rename(columns={
        "PATNO": "RID",
        "CLINICAL_EVENT": "VISCODE",
    })
    if "RID" in df.columns:
        df["RID"] = pd.to_numeric(df["RID"], errors="coerce")
    if "VISCODE" in df.columns:
        df["VISCODE"] = df["VISCODE"].astype(str).str.lower()

    # Filter to key biomarkers — keeps memory footprint small
    _KEY_BIOMARKERS = {
        "CSF Alpha-synuclein": "ALPHA_SYN",
        "a-Synuclein": "ALPHA_SYN",
        "total alpha-Syn ELISA": "ALPHA_SYN",
        "tTau": "T_TAU",
        "BD tTau": "T_TAU",
        "pTau181": "P_TAU181",
        "pTau": "P_TAU181",
        "ABeta42": "ABETA42",
        "ABeta 1-42": "ABETA42",
        "NfL": "NFL",
        "NFL": "NFL",
        "ApoE Genotype": "APOE_GENOTYPE",
        "APOE GENOTYPE": "APOE_GENOTYPE",
        "PTAU217": "P_TAU217",
        "p217+tau": "P_TAU217",
        "NPTAU217": "NP_TAU217",
    }

    df = df[df["TESTNAME"].isin(_KEY_BIOMARKERS)].copy()
    if df.empty:
        return pd.DataFrame(columns=["RID", "VISCODE"])

    df["TESTNAME"] = df["TESTNAME"].map(_KEY_BIOMARKERS)
    df["TESTVALUE"] = pd.to_numeric(df["TESTVALUE"], errors="coerce")

    # Pivot: one row per (RID, VISCODE)
    wide = df.pivot_table(
        index=["RID", "VISCODE"],
        columns="TESTNAME",
        values="TESTVALUE",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    logger.info(
        "Pivoted long biomarkers: %d rows, %d biomarker columns",
        len(wide),
        sum(1 for c in wide.columns if c not in ("RID", "VISCODE")),
    )
    return wide


def assign_progression_labels(
    clinical: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Assign PD progression labels based on MDS-UPDRS III change.

    Computes per-participant slope of MDS-UPDRS III over time, then
    splits into tertiles: top = PD_FAST, bottom = PD_SLOW,
    middle = PD_INTERMEDIATE.

    Parameters
    ----------
    clinical : pd.DataFrame
        PPMI clinical DataFrame with RID, EXAMDATE, and progression metric.
    config : dict
        Configuration with ppmi section.

    Returns
    -------
    pd.DataFrame
        DataFrame with TRAJECTORY column added.
    """
    ppmi_cfg = config["ppmi"]
    metric = ppmi_cfg["progression_metric"]

    df = clinical.copy()
    df["TRAJECTORY"] = "other"

    if metric not in df.columns:
        logger.warning(
            "Progression metric '%s' not found in DataFrame. "
            "Cannot assign progression labels.",
            metric,
        )
        return df

    # Compute per-participant slope of metric over time
    slopes = {}
    for rid, group in df.groupby("RID"):
        group = group.sort_values("EXAMDATE").dropna(subset=[metric, "EXAMDATE"])
        if len(group) < 2:
            continue

        # Compute months from first visit
        first_date = group["EXAMDATE"].iloc[0]
        months = (
            (group["EXAMDATE"] - first_date).dt.days / 30.44
        ).values.astype(float)
        values = group[metric].values.astype(float)

        if months[-1] - months[0] < 1:
            continue

        # Linear slope via least squares
        slope = np.polyfit(months, values, 1)[0]
        slopes[rid] = slope

    if not slopes:
        logger.warning("No valid slopes computed for progression labeling")
        return df

    slope_series = pd.Series(slopes)

    # Split into tertiles
    terciles = slope_series.quantile([1 / 3, 2 / 3])
    slow_threshold = terciles.iloc[0]
    fast_threshold = terciles.iloc[1]

    fast_rids = set(slope_series[slope_series >= fast_threshold].index)
    slow_rids = set(slope_series[slope_series <= slow_threshold].index)
    intermediate_rids = set(slope_series.index) - fast_rids - slow_rids

    df.loc[df["RID"].isin(fast_rids), "TRAJECTORY"] = ppmi_cfg["fast_progressor_group"]
    df.loc[df["RID"].isin(slow_rids), "TRAJECTORY"] = ppmi_cfg["slow_progressor_group"]
    df.loc[df["RID"].isin(intermediate_rids), "TRAJECTORY"] = ppmi_cfg["intermediate_group"]

    logger.info("PPMI progression labels assigned:")
    logger.info("  PD_FAST: %d participants", len(fast_rids))
    logger.info("  PD_SLOW: %d participants", len(slow_rids))
    logger.info("  PD_INTERMEDIATE: %d participants", len(intermediate_rids))

    return df


def merge_ppmi_data(
    clinical: pd.DataFrame,
    somascan: pd.DataFrame,
    biomarkers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge PPMI clinical, proteomics, and biomarker data.

    Parameters
    ----------
    clinical : pd.DataFrame
        Clinical data with trajectory labels.
    somascan : pd.DataFrame
        SomaScan proteomics.
    biomarkers : pd.DataFrame or None
        CSF biomarkers.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    logger.info(
        "Merging PPMI: clinical (%d rows) + SomaScan (%d rows)",
        len(clinical),
        len(somascan),
    )

    merged = clinical.merge(somascan, on=["RID", "VISCODE"], how="inner")
    logger.info("After clinical-proteomics inner join: %d rows", len(merged))

    if biomarkers is not None:
        pre_merge = len(merged)
        merge_cols = ["RID"]
        if "VISCODE" in biomarkers.columns:
            merge_cols.append("VISCODE")
        merged = merged.merge(biomarkers, on=merge_cols, how="left")
        logger.info(
            "After biomarker left join: %d rows (was %d)", len(merged), pre_merge
        )

    n_participants = merged["RID"].nunique()
    logger.info(
        "Final PPMI dataset: %d visits from %d participants",
        len(merged),
        n_participants,
    )
    return merged
