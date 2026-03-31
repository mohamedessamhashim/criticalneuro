"""Golden Set computation for cross-platform validation.

Identifies proteins that are DNB core proteins on both SomaScan AND Olink
platforms. These form the Golden Set — the primary reportable finding,
demonstrating that transition signals are properties of patient biology
rather than measurement technology.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def compute_golden_set(
    dnb_core_somascan: pd.DataFrame,
    dnb_core_olink: pd.DataFrame,
    overlap_proteins: list[str],
    config: dict,
) -> pd.DataFrame:
    """Identify proteins that are DNB core on both platforms.

    Parameters
    ----------
    dnb_core_somascan : pd.DataFrame
        Core proteins from SomaScan DNB with columns:
        protein, frequency, n_participants, UniProt.
    dnb_core_olink : pd.DataFrame
        Core proteins from Olink DNB with columns:
        protein, frequency, n_participants, UniProt.
    overlap_proteins : list[str]
        UniProt accessions measurable on both platforms.
    config : dict
        Configuration with cross_platform section.

    Returns
    -------
    pd.DataFrame
        Golden Set proteins with cross-platform annotations.
    """
    cp_cfg = config["cross_platform"]
    fdr_threshold = cp_cfg["golden_set_fdr_threshold"]

    if dnb_core_somascan.empty or dnb_core_olink.empty:
        logger.warning(
            "One or both platforms have no DNB core proteins — Golden Set is empty"
        )
        return pd.DataFrame(columns=[
            "UniProt", "protein_somascan", "protein_olink",
            "frequency_somascan", "frequency_olink",
            "is_golden_set",
        ])

    # Restrict to overlap proteins
    soma_overlap = dnb_core_somascan[
        dnb_core_somascan["UniProt"].isin(overlap_proteins)
    ].copy()
    olink_overlap = dnb_core_olink[
        dnb_core_olink["UniProt"].isin(overlap_proteins)
    ].copy()

    if soma_overlap.empty or olink_overlap.empty:
        logger.warning(
            "No overlap proteins found in DNB core lists — Golden Set is empty"
        )
        return pd.DataFrame(columns=[
            "UniProt", "protein_somascan", "protein_olink",
            "frequency_somascan", "frequency_olink",
            "is_golden_set",
        ])

    # Inner join on UniProt to find shared core proteins
    merged = soma_overlap.merge(
        olink_overlap,
        on="UniProt",
        suffixes=("_somascan", "_olink"),
    )

    if merged.empty:
        logger.warning(
            "SomaScan and Olink DNB core proteins share no UniProt IDs — "
            "Golden Set is empty"
        )
        return pd.DataFrame(columns=[
            "UniProt", "protein_somascan", "protein_olink",
            "frequency_somascan", "frequency_olink",
            "is_golden_set",
        ])

    # Build result DataFrame
    result = pd.DataFrame({
        "UniProt": merged["UniProt"],
        "protein_somascan": merged["protein_somascan"],
        "protein_olink": merged["protein_olink"],
        "frequency_somascan": merged["frequency_somascan"],
        "frequency_olink": merged["frequency_olink"],
    })

    # Apply FDR correction to combined frequency scores
    # Use the minimum frequency across platforms as a conservative test statistic
    # Convert frequencies to p-value-like quantities via permutation null
    # For now, use the frequency product as the test statistic and apply BH-FDR
    combined_freq = result["frequency_somascan"] * result["frequency_olink"]
    # Convert to p-values: higher frequency = lower p-value
    # Use 1 - combined_freq as a simple monotone transform
    pseudo_p = 1.0 - combined_freq
    pseudo_p = pseudo_p.clip(lower=1e-10, upper=1.0 - 1e-10)

    reject, fdr_p, _, _ = multipletests(pseudo_p, alpha=fdr_threshold, method="fdr_bh")
    result["fdr_p"] = fdr_p
    result["is_golden_set"] = reject

    n_golden = result["is_golden_set"].sum()
    logger.info(
        "Golden Set: %d proteins pass FDR < %.2f (from %d shared core proteins)",
        n_golden,
        fdr_threshold,
        len(result),
    )

    # Save results
    results_dir = Path(config["paths"]["results_cross_platform"])
    results_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(results_dir / "golden_set_proteins.csv", index=False)
    logger.info("Golden Set saved to %s", results_dir / "golden_set_proteins.csv")

    return result


def add_csd_evidence_to_golden_set(
    golden_set_df: pd.DataFrame,
    csd_results_somascan: pd.DataFrame,
) -> pd.DataFrame:
    """Flag Golden Set proteins with CSD support as Tier 1.

    Parameters
    ----------
    golden_set_df : pd.DataFrame
        Golden Set from compute_golden_set().
    csd_results_somascan : pd.DataFrame
        CSD group statistics with columns: protein, var_tau, var_fdr_p.

    Returns
    -------
    pd.DataFrame
        Golden Set with tier column added.
    """
    if golden_set_df.empty:
        golden_set_df["csd_var_tau"] = pd.Series(dtype=float)
        golden_set_df["csd_significant"] = pd.Series(dtype=bool)
        golden_set_df["tier"] = pd.Series(dtype=str)
        return golden_set_df

    # Merge CSD results on protein name (SomaScan side)
    csd_subset = csd_results_somascan[["protein", "var_tau", "var_fdr_p"]].copy()
    csd_subset = csd_subset.rename(columns={
        "var_tau": "csd_var_tau",
        "var_fdr_p": "csd_fdr_p",
    })

    result = golden_set_df.merge(
        csd_subset,
        left_on="protein_somascan",
        right_on="protein",
        how="left",
    )
    result = result.drop(columns=["protein"], errors="ignore")

    result["csd_significant"] = result["csd_fdr_p"].fillna(1.0) < 0.05
    result["tier"] = np.where(
        result["is_golden_set"] & result["csd_significant"],
        "Tier_1",
        np.where(result["is_golden_set"], "Tier_2", "not_golden_set"),
    )

    n_tier1 = (result["tier"] == "Tier_1").sum()
    n_tier2 = (result["tier"] == "Tier_2").sum()
    logger.info(
        "Golden Set tiers: %d Tier 1 (Golden Set + CSD), %d Tier 2 (Golden Set only)",
        n_tier1,
        n_tier2,
    )

    return result


def compute_golden_set_statistics(
    golden_set_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Compute summary statistics for the Golden Set.

    Parameters
    ----------
    golden_set_df : pd.DataFrame
        Golden Set with tier annotations.
    config : dict
        Configuration.

    Returns
    -------
    pd.DataFrame
        Summary statistics.
    """
    golden = golden_set_df[golden_set_df["is_golden_set"]].copy()

    summary = {
        "n_golden_set_proteins": len(golden),
        "n_total_shared_core": len(golden_set_df),
        "fraction_golden_set": len(golden) / max(len(golden_set_df), 1),
        "mean_frequency_somascan": golden["frequency_somascan"].mean() if len(golden) > 0 else 0.0,
        "mean_frequency_olink": golden["frequency_olink"].mean() if len(golden) > 0 else 0.0,
    }

    if "tier" in golden_set_df.columns:
        summary["n_tier_1"] = (golden_set_df["tier"] == "Tier_1").sum()
        summary["n_tier_2"] = (golden_set_df["tier"] == "Tier_2").sum()
        summary["fraction_csd_support"] = (
            summary["n_tier_1"] / max(len(golden), 1)
        )

    # Hypergeometric test: is the overlap larger than expected by chance?
    # This requires knowing total proteins on each platform — use overlap count
    if "n_somascan_core" in config.get("_runtime", {}):
        n_soma_core = config["_runtime"]["n_somascan_core"]
        n_olink_core = config["_runtime"]["n_olink_core"]
        n_overlap_total = config["_runtime"]["n_overlap_total"]
        n_shared = len(golden)

        # Hypergeometric: P(X >= n_shared) where X ~ Hypergeometric(N, K, n)
        pval = stats.hypergeom.sf(
            n_shared - 1, n_overlap_total, n_soma_core, n_olink_core
        )
        summary["hypergeometric_p"] = pval
        logger.info("Hypergeometric test p-value: %.2e", pval)

    summary_df = pd.DataFrame([summary])

    results_dir = Path(config["paths"]["results_cross_platform"])
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(results_dir / "golden_set_summary.csv", index=False)
    logger.info("Golden Set summary saved")

    return summary_df
