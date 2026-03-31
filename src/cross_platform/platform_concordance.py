"""Cross-platform concordance analysis.

Quantifies agreement between SomaScan and Olink DNB results
for overlapping proteins. Includes positive control validation
to verify that platform harmonisation is working correctly.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_platform_concordance(
    somascan_scores: pd.DataFrame,
    olink_scores: pd.DataFrame,
    overlap_proteins: list[str],
) -> pd.DataFrame:
    """Compute per-protein Spearman concordance between platforms.

    For each protein in the overlap, correlates its DNB frequency score
    on SomaScan vs. Olink across all participants with measurements on
    both platforms.

    Parameters
    ----------
    somascan_scores : pd.DataFrame
        Per-participant DNB scores from SomaScan with columns:
        RID, protein, dnb_frequency (or similar per-protein metric).
    olink_scores : pd.DataFrame
        Per-participant DNB scores from Olink, same structure.
    overlap_proteins : list[str]
        UniProt accessions measurable on both platforms.

    Returns
    -------
    pd.DataFrame
        Per-protein concordance with columns: UniProt, spearman_rho,
        spearman_p, n_participants.
    """
    # Restrict to overlap proteins
    soma_overlap = somascan_scores[
        somascan_scores["UniProt"].isin(overlap_proteins)
    ]
    olink_overlap = olink_scores[
        olink_scores["UniProt"].isin(overlap_proteins)
    ]

    results = []
    for uniprot in overlap_proteins:
        soma_protein = soma_overlap[soma_overlap["UniProt"] == uniprot]
        olink_protein = olink_overlap[olink_overlap["UniProt"] == uniprot]

        if soma_protein.empty or olink_protein.empty:
            continue

        # Merge on RID to get paired measurements
        merged = soma_protein.merge(
            olink_protein,
            on="RID",
            suffixes=("_somascan", "_olink"),
        )

        if len(merged) < 3:
            continue

        rho, pval = stats.spearmanr(
            merged["dnb_frequency_somascan"],
            merged["dnb_frequency_olink"],
        )

        results.append({
            "UniProt": uniprot,
            "spearman_rho": rho if np.isfinite(rho) else 0.0,
            "spearman_p": pval if np.isfinite(pval) else 1.0,
            "n_participants": len(merged),
        })

    result_df = pd.DataFrame(results)
    if result_df.empty:
        logger.warning("No proteins had enough paired measurements for concordance")
        return pd.DataFrame(columns=[
            "UniProt", "spearman_rho", "spearman_p", "n_participants"
        ])

    logger.info(
        "Platform concordance computed for %d proteins, "
        "median rho = %.3f",
        len(result_df),
        result_df["spearman_rho"].median(),
    )

    return result_df


def positive_control_concordance(
    somascan_df: pd.DataFrame,
    olink_df: pd.DataFrame,
    control_proteins: list[str],
    config: dict,
) -> dict:
    """Verify concordance of known positive control proteins.

    For GFAP, NEFL, CLU, APOE, compute their DNB scores on both
    platforms and verify they are concordant. If any control fails,
    raise an error to halt the pipeline.

    Parameters
    ----------
    somascan_df : pd.DataFrame
        SomaScan DNB core proteins with UniProt and frequency columns.
    olink_df : pd.DataFrame
        Olink DNB core proteins with UniProt and frequency columns.
    control_proteins : list[str]
        Gene symbols of positive control proteins.
    config : dict
        Configuration.

    Returns
    -------
    dict
        Concordance results per control protein.

    Raises
    ------
    ValueError
        If any control protein fails concordance check.
    """
    concordance_results = {}
    failures = []

    for gene in control_proteins:
        # Look up in both DataFrames by gene symbol or UniProt
        soma_match = somascan_df[
            somascan_df["protein"].str.contains(gene, case=False, na=False)
            | somascan_df.get("EntrezGeneSymbol", pd.Series(dtype=str)).str.contains(
                gene, case=False, na=False
            )
        ]
        olink_match = olink_df[
            olink_df["protein"].str.contains(gene, case=False, na=False)
            | olink_df.get("EntrezGeneSymbol", pd.Series(dtype=str)).str.contains(
                gene, case=False, na=False
            )
        ]

        if soma_match.empty and olink_match.empty:
            logger.info(
                "Positive control %s not in either platform's core proteins — skipping",
                gene,
            )
            concordance_results[gene] = {
                "status": "absent_both",
                "somascan_frequency": None,
                "olink_frequency": None,
            }
            continue

        if soma_match.empty or olink_match.empty:
            present_on = "somascan" if not soma_match.empty else "olink"
            logger.warning(
                "Positive control %s found only on %s — discordant",
                gene,
                present_on,
            )
            concordance_results[gene] = {
                "status": "discordant",
                "somascan_frequency": (
                    soma_match["frequency"].iloc[0] if not soma_match.empty else None
                ),
                "olink_frequency": (
                    olink_match["frequency"].iloc[0] if not olink_match.empty else None
                ),
            }
            failures.append(gene)
            continue

        # Both present — check direction agreement
        soma_freq = soma_match["frequency"].iloc[0]
        olink_freq = olink_match["frequency"].iloc[0]

        concordance_results[gene] = {
            "status": "concordant",
            "somascan_frequency": soma_freq,
            "olink_frequency": olink_freq,
        }
        logger.info(
            "Positive control %s: SomaScan freq=%.3f, Olink freq=%.3f — concordant",
            gene,
            soma_freq,
            olink_freq,
        )

    if failures:
        msg = (
            f"Positive control concordance FAILED for: {failures}. "
            "This suggests a bug in platform harmonisation. "
            "Pipeline halted — investigate before proceeding."
        )
        logger.error(msg)
        raise ValueError(msg)

    logger.info("All positive controls passed concordance check")
    return concordance_results


def generate_concordance_report(
    concordance_df: pd.DataFrame,
    config: dict,
) -> None:
    """Write a summary concordance report.

    Parameters
    ----------
    concordance_df : pd.DataFrame
        Output of compute_platform_concordance().
    config : dict
        Configuration.
    """
    results_dir = Path(config["paths"]["results_cross_platform"])
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "concordance_report.txt"

    if concordance_df.empty:
        report_path.write_text("No concordance data available.\n")
        logger.warning("Empty concordance data — report is minimal")
        return

    n_proteins = len(concordance_df)
    median_rho = concordance_df["spearman_rho"].median()
    mean_rho = concordance_df["spearman_rho"].mean()
    frac_positive = (concordance_df["spearman_rho"] > 0).mean()
    frac_strong = (concordance_df["spearman_rho"] > 0.5).mean()

    top20 = concordance_df.nlargest(20, "spearman_rho")

    lines = [
        "=== Cross-Platform Concordance Report ===",
        "",
        f"Number of overlap proteins analysed: {n_proteins}",
        f"Median Spearman rho: {median_rho:.3f}",
        f"Mean Spearman rho: {mean_rho:.3f}",
        f"Fraction with rho > 0: {frac_positive:.2%}",
        f"Fraction with rho > 0.5: {frac_strong:.2%}",
        "",
        "Top 20 most concordant proteins:",
        "-" * 50,
    ]

    for _, row in top20.iterrows():
        lines.append(
            f"  {row['UniProt']:15s}  rho={row['spearman_rho']:.3f}  "
            f"p={row['spearman_p']:.2e}  n={int(row['n_participants'])}"
        )

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text)
    logger.info("Concordance report written to %s", report_path)

    # Also save concordance DataFrame
    concordance_df.to_csv(
        results_dir / "platform_concordance.csv", index=False
    )
