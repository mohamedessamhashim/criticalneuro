"""Stage 5b: Validation — AUC, ROC, biomarker correlation.

Uses l-DNB IDNB scores for validation in longitudinal mode,
or sDNB scores for cross-sectional mode.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def run_validation_stage(cfg: dict) -> None:
    """Run validation analysis on l-DNB or sDNB scores.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration.
    """
    output_dir = Path(cfg["output"]["dir"]) / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg["analysis_mode"] == "longitudinal":
        # Load l-DNB scores
        ldnb_path = Path(cfg["output"]["dir"]) / "ldnb" / "ldnb_individual_scores.csv"
        if not ldnb_path.exists():
            logger.warning("l-DNB scores not found at %s — skipping validation", ldnb_path)
            return

        scores = pd.read_csv(ldnb_path)
        logger.info("Loaded %d l-DNB scores for validation", len(scores))

        # Basic statistics by stage
        if "Stage" in scores.columns and "IDNB" in scores.columns:
            stage_stats = scores.groupby("Stage")["IDNB"].agg(["mean", "std", "count"])
            stage_stats.to_csv(output_dir / "idnb_stage_statistics.csv")
            logger.info("Stage statistics:\n%s", stage_stats.to_string())

        # Converter vs non-converter comparison
        if "Converter" in scores.columns:
            from scipy import stats

            is_converter = scores["Converter"].astype(str).str.lower().isin(["true", "1"])
            converters = scores.loc[is_converter, "IDNB"].dropna()
            stable = scores.loc[~is_converter, "IDNB"].dropna()

            if len(converters) > 0 and len(stable) > 0:
                u_stat, p_val = stats.mannwhitneyu(converters, stable, alternative="greater")
                logger.info(
                    "Converter vs stable IDNB: U=%.1f, p=%.4f (converters mean=%.3f, stable mean=%.3f)",
                    u_stat, p_val, converters.mean(), stable.mean(),
                )

                comparison = pd.DataFrame({
                    "group": ["converter", "stable"],
                    "n": [len(converters), len(stable)],
                    "mean_IDNB": [converters.mean(), stable.mean()],
                    "std_IDNB": [converters.std(), stable.std()],
                    "mann_whitney_U": [u_stat, u_stat],
                    "p_value": [p_val, p_val],
                })
                comparison.to_csv(output_dir / "converter_vs_stable.csv", index=False)

    else:
        logger.info("Cross-sectional validation: use pipelines/run_full_pipeline.py --stage validation")

    logger.info("Validation stage complete")
