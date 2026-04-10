"""Load and validate config.yaml for CriticalNeuroMap pipeline."""

import logging
import random
import sys
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and validate configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to config.yaml file.

    Returns
    -------
    dict
        Validated configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If config file does not exist.
    ValueError
        If required fields are missing or invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            "Create config.yaml or specify --config path."
        )

    with open(path) as f:
        cfg = yaml.safe_load(f)

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    """Validate required config fields based on analysis mode."""
    # Required top-level fields
    mode = cfg.get("analysis_mode")
    if mode not in ("cross_sectional", "longitudinal"):
        raise ValueError(
            f"analysis_mode must be 'cross_sectional' or 'longitudinal', got '{mode}'"
        )

    cohort = cfg.get("cohort")
    if cohort not in ("adni", "ppmi", "knight_adrc"):
        raise ValueError(
            f"cohort must be 'adni', 'ppmi', or 'knight_adrc', got '{cohort}'"
        )

    # Validate data provenance
    prov = cfg.get("data_provenance")
    if prov is None:
        raise ValueError(
            "data_provenance section is required in config.yaml.\n"
            "See data/README_data.md for the three questions you must answer."
        )

    for key in ("cruchaga_qc_already_applied", "log2_already_applied", "combat_already_applied"):
        if key not in prov:
            raise ValueError(f"data_provenance.{key} is required in config.yaml")

    # Validate input paths for longitudinal mode
    if mode == "longitudinal":
        input_cfg = cfg.get("input", {})
        expr_path = Path(input_cfg.get("expression_matrix", ""))
        meta_path = Path(input_cfg.get("metadata", ""))

        if not expr_path.name:
            raise ValueError("input.expression_matrix path is required for longitudinal mode")
        if not meta_path.name:
            raise ValueError("input.metadata path is required for longitudinal mode")

        # Warn (don't fail) if files don't exist yet — Muhammad may not have placed them
        if not expr_path.exists():
            logger.warning(
                "Expression matrix not found at %s — pipeline will fail at data loading stage",
                expr_path,
            )
        if not meta_path.exists():
            logger.warning(
                "Metadata file not found at %s — pipeline will fail at data loading stage",
                meta_path,
            )

    # Validate longitudinal section
    if mode == "longitudinal":
        long_cfg = cfg.get("longitudinal")
        if long_cfg is None:
            raise ValueError("longitudinal section is required when analysis_mode is 'longitudinal'")

    # Validate WGCNA section
    wgcna_cfg = cfg.get("wgcna")
    if wgcna_cfg is None:
        raise ValueError("wgcna section is required in config.yaml")

    # Validate BioTIP section
    biotip_cfg = cfg.get("biotip")
    if biotip_cfg is None:
        raise ValueError("biotip section is required in config.yaml")

    # Validate l-DNB section
    ldnb_cfg = cfg.get("ldnb")
    if ldnb_cfg is None:
        raise ValueError("ldnb section is required in config.yaml")

    logger.info(
        "Config validated: mode=%s, cohort=%s, provenance: qc_done=%s, log2_done=%s, combat_done=%s",
        mode, cohort,
        prov["cruchaga_qc_already_applied"],
        prov["log2_already_applied"],
        prov["combat_already_applied"],
    )


def set_all_seeds(cfg: dict) -> None:
    """Set all random seeds globally for reproducibility.

    Parameters
    ----------
    cfg : dict
        Configuration with reproducibility section.
    """
    repro = cfg.get("reproducibility", {})
    python_seed = repro.get("python_seed", 42)
    global_seed = repro.get("global_seed", 42)

    random.seed(python_seed)
    np.random.seed(global_seed)

    logger.info("Random seeds set: python=%d, numpy=%d", python_seed, global_seed)


def validate_input_value_range(expr_path: str, log2_already_applied: bool) -> None:
    """Validate that expression values match the declared provenance.

    Checks whether values look like RFU (~100-100000) or log2 (~7-17)
    and warns if there's a mismatch with the config declaration.

    Parameters
    ----------
    expr_path : str
        Path to expression matrix CSV.
    log2_already_applied : bool
        Whether config says data is already log2-transformed.
    """
    import pandas as pd

    try:
        # Read just the first few rows to check value ranges
        df = pd.read_csv(expr_path, nrows=5, index_col=0)
        seq_cols = [c for c in df.columns if c.startswith("seq.")]
        if not seq_cols:
            logger.warning("No seq.* columns found in expression matrix — cannot validate value range")
            return

        sample_values = df[seq_cols].values.flatten()
        sample_values = sample_values[~np.isnan(sample_values)]

        if len(sample_values) == 0:
            return

        median_val = np.median(sample_values)
        max_val = np.max(sample_values)

        if log2_already_applied:
            # Expect values in ~7-17 range
            if median_val > 100:
                logger.error(
                    "VALUE RANGE MISMATCH: config says log2_already_applied=true but "
                    "median value is %.1f (looks like RFU space, not log2). "
                    "Set log2_already_applied: false in config.yaml or the pipeline "
                    "will produce WRONG RESULTS.",
                    median_val,
                )
                raise ValueError(
                    f"Expression values (median={median_val:.1f}) look like RFU space "
                    "but log2_already_applied is true. Fix config.yaml."
                )
        else:
            # Expect values in ~100-100000 range (RFU)
            if max_val < 50:
                logger.error(
                    "VALUE RANGE MISMATCH: config says log2_already_applied=false but "
                    "max value is %.1f (looks like log2 space, not RFU). "
                    "Set log2_already_applied: true in config.yaml or the pipeline "
                    "will double-log-transform and produce WRONG RESULTS.",
                    max_val,
                )
                raise ValueError(
                    f"Expression values (max={max_val:.1f}) look like log2 space "
                    "but log2_already_applied is false. Fix config.yaml."
                )

        logger.info(
            "Value range check passed: median=%.1f, max=%.1f, log2_declared=%s",
            median_val, max_val, log2_already_applied,
        )
    except pd.errors.EmptyDataError:
        logger.warning("Expression matrix appears empty — cannot validate value range")
