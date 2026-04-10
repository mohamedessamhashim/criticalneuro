"""WGCNA-guided Dynamic Network Biomarker (DNB) analysis.

Uses WGCNA co-expression modules to identify biologically grounded
transition modules. Each WGCNA module is scored using the standard
DNB formula, and the top-scoring module's hub proteins form the core
DNB set. This is the primary DNB identification method for cross-sectional
analysis (ADNI/PPMI).

For longitudinal analysis (Knight-ADRC), see the BioTIP + l-DNB pipeline
in pipeline/stage4_biotip.R and pipeline/stage4_ldnb.R.

References:
    - Chen et al. (2012) — DNB framework
    - Langfelder & Horvath (2008) — WGCNA
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.dnb.dnb_computation import compute_dnb_score

logger = logging.getLogger(__name__)


def load_wgcna_modules(wgcna_results_dir: str) -> pd.DataFrame:
    """Load WGCNA module assignments from CSV.

    Parameters
    ----------
    wgcna_results_dir : str
        Path to directory containing wgcna_modules.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: protein, module, kME.

    Raises
    ------
    FileNotFoundError
        If wgcna_modules.csv does not exist.
    """
    modules_path = Path(wgcna_results_dir) / "wgcna_modules.csv"
    if not modules_path.exists():
        raise FileNotFoundError(
            f"WGCNA modules not found at {modules_path}. "
            "Run 'Rscript R/wgcna_module_detection.R' first."
        )

    df = pd.read_csv(modules_path)
    logger.info(
        "Loaded WGCNA modules: %d proteins across %d modules",
        len(df),
        df["module"].nunique(),
    )
    return df


def score_modules_dnb(
    df: pd.DataFrame,
    protein_cols: list[str],
    reference_mask: pd.Series,
    converter_mask: pd.Series,
    wgcna_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Score each WGCNA module using the DNB formula.

    For each module, filters proteins by kME threshold, then computes:
        DNB = (sigma_D × PCC_D) / (|PCC_O| + epsilon)

    using the converter group's protein expression vs. all outside proteins.

    Parameters
    ----------
    df : pd.DataFrame
        Full proteomics DataFrame (already imputed).
    protein_cols : list[str]
        All protein column names.
    reference_mask : pd.Series
        Boolean mask for reference population (CN_amyloid_negative).
    converter_mask : pd.Series
        Boolean mask for converter group (MCI_to_Dementia).
    wgcna_df : pd.DataFrame
        WGCNA module assignments (protein, module, kME).
    config : dict
        Configuration with wgcna and dnb sections.

    Returns
    -------
    pd.DataFrame
        Ranked module scores with columns:
        module, dnb_score, n_proteins, n_proteins_filtered, mean_kME.
    """
    wgcna_cfg = config["wgcna"]
    dnb_cfg = config["dnb"]
    kme_threshold = wgcna_cfg["kme_threshold"]
    min_module_size = wgcna_cfg["min_module_size"]
    epsilon = dnb_cfg["epsilon"]

    # Build set of available protein columns for fast lookup
    available_proteins = set(protein_cols)

    # Get converter data for DNB scoring
    X_converter = df.loc[converter_mask]

    modules = wgcna_df["module"].unique()
    # Exclude grey module (unassigned proteins)
    modules = [m for m in modules if m != "grey"]

    results = []
    for module in modules:
        # All proteins in this module
        module_proteins_all = wgcna_df[wgcna_df["module"] == module]["protein"].tolist()

        # Filter by kME threshold (quality filter for tight module membership)
        module_proteins_strong = wgcna_df[
            (wgcna_df["module"] == module) & (wgcna_df["kME"] >= kme_threshold)
        ]["protein"].tolist()

        # Keep only proteins that exist in the data
        module_proteins_strong = [
            p for p in module_proteins_strong if p in available_proteins
        ]

        if len(module_proteins_strong) < min_module_size:
            logger.debug(
                "Module '%s': %d proteins after kME filter (< %d minimum), skipping",
                module,
                len(module_proteins_strong),
                min_module_size,
            )
            continue

        # Outside proteins: everything NOT in this module
        module_set = set(module_proteins_strong)
        outside_proteins = [p for p in protein_cols if p not in module_set]

        # Extract converter expression matrices
        X_group = X_converter[module_proteins_strong].values
        X_outside = X_converter[outside_proteins].values

        # Compute DNB score using the existing function
        score = compute_dnb_score(X_group, X_outside, epsilon=epsilon)

        # Module statistics
        module_kmes = wgcna_df[
            (wgcna_df["module"] == module) & (wgcna_df["kME"] >= kme_threshold)
        ]["kME"]

        results.append(
            {
                "module": module,
                "dnb_score": score,
                "n_proteins_total": len(module_proteins_all),
                "n_proteins_filtered": len(module_proteins_strong),
                "mean_kME": float(module_kmes.mean()),
                "max_kME": float(module_kmes.max()),
            }
        )

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values("dnb_score", ascending=False).reset_index(
            drop=True
        )

    logger.info(
        "WGCNA module DNB scoring: %d/%d modules scored (kME >= %.2f, n >= %d)",
        len(results_df),
        len(modules),
        kme_threshold,
        min_module_size,
    )

    if len(results_df) > 0:
        top = results_df.iloc[0]
        logger.info(
            "Top module: '%s' — DNB score = %.4f, %d proteins",
            top["module"],
            top["dnb_score"],
            top["n_proteins_filtered"],
        )

    return results_df


def identify_transition_module(module_scores: pd.DataFrame) -> str | None:
    """Identify the transition module (highest DNB score).

    Parameters
    ----------
    module_scores : pd.DataFrame
        Output of score_modules_dnb().

    Returns
    -------
    str or None
        Name of the transition module (color), or None if no modules scored.
    """
    if module_scores.empty:
        logger.warning("No modules scored — cannot identify transition module")
        return None

    transition_module = module_scores.iloc[0]["module"]
    logger.info(
        "Transition module identified: '%s' (DNB = %.4f)",
        transition_module,
        module_scores.iloc[0]["dnb_score"],
    )
    return transition_module


def extract_wgcna_core_proteins(
    df: pd.DataFrame,
    protein_cols: list[str],
    reference_mask: pd.Series,
    converter_mask: pd.Series,
    transition_module: str,
    wgcna_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Extract core DNB proteins from the transition module using dual-filter.

    Filter 1: Top N% variance ratio (same as original DNB method).
    Filter 2: Rank by kME, keep top K proteins.

    Parameters
    ----------
    df : pd.DataFrame
        Full proteomics DataFrame.
    protein_cols : list[str]
        All protein column names.
    reference_mask : pd.Series
        Boolean mask for reference population.
    converter_mask : pd.Series
        Boolean mask for converter group.
    transition_module : str
        WGCNA module color name.
    wgcna_df : pd.DataFrame
        WGCNA module assignments.
    config : dict
        Configuration with wgcna and dnb sections.

    Returns
    -------
    pd.DataFrame
        Core proteins with columns: protein, module, kME, var_ratio.
    """
    wgcna_cfg = config["wgcna"]
    dnb_cfg = config["dnb"]
    kme_threshold = wgcna_cfg["kme_threshold"]
    core_count = wgcna_cfg["core_protein_count"]
    percentile = dnb_cfg["primary_variance_percentile"]

    # Get module proteins that pass kME threshold
    module_df = wgcna_df[
        (wgcna_df["module"] == transition_module) & (wgcna_df["kME"] >= kme_threshold)
    ].copy()

    # Keep only proteins present in data
    available = set(protein_cols)
    module_df = module_df[module_df["protein"].isin(available)].copy()

    if len(module_df) == 0:
        logger.warning(
            "No proteins in transition module '%s' pass kME threshold",
            transition_module,
        )
        return pd.DataFrame(columns=["protein", "module", "kME", "var_ratio"])

    module_proteins = module_df["protein"].tolist()

    # Dual-filter 1: Variance ratio relative to reference
    X_converter = df.loc[converter_mask, module_proteins].values
    X_reference = df.loc[reference_mask, module_proteins].values

    var_converter = np.nanvar(X_converter, axis=0, ddof=1)
    var_reference = np.nanvar(X_reference, axis=0, ddof=1)
    var_ref_safe = np.where(var_reference > 0, var_reference, 1e-10)
    var_ratio = var_converter / var_ref_safe

    module_df = module_df.reset_index(drop=True)
    module_df["var_ratio"] = var_ratio

    # Apply variance percentile filter across module proteins
    var_threshold = np.nanpercentile(var_ratio, 100 - percentile)
    var_threshold = max(var_threshold, 1e-10)
    high_var_mask = var_ratio >= var_threshold
    module_df_filtered = module_df[high_var_mask].copy()

    logger.info(
        "Dual-filter step 1 (variance): %d/%d module proteins pass top %d%% threshold",
        len(module_df_filtered),
        len(module_df),
        percentile,
    )

    if len(module_df_filtered) == 0:
        # Fall back to all module proteins if variance filter is too strict
        logger.warning(
            "Variance filter removed all proteins — falling back to kME-only ranking"
        )
        module_df_filtered = module_df.copy()

    # Dual-filter 2: Rank by kME, keep top K
    module_df_filtered = module_df_filtered.sort_values("kME", ascending=False)
    core_df = module_df_filtered.head(core_count).copy()

    logger.info(
        "WGCNA core proteins: %d proteins from module '%s' "
        "(kME >= %.2f, top %d%% variance, top %d by kME)",
        len(core_df),
        transition_module,
        kme_threshold,
        percentile,
        core_count,
    )

    return core_df[["protein", "module", "kME", "var_ratio"]]


def run_wgcna_dnb_analysis(
    df: pd.DataFrame,
    protein_cols: list[str],
    config: dict,
    platform_label: str,
    wgcna_results_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full WGCNA-guided DNB analysis pipeline.

    Orchestrates: load modules → score modules → identify transition →
    extract core proteins → run sDNB.

    Parameters
    ----------
    df : pd.DataFrame
        Processed, imputed proteomics DataFrame.
    protein_cols : list[str]
        Protein column names.
    config : dict
        Full config dict.
    platform_label : str
        'somascan' or 'olink'.
    wgcna_results_dir: str, optional
        Path to the WGCNA results directory. Defaults to config['wgcna']['results_dir'].

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (module_scores, core_proteins, sdnb_scores)
    """
    logger.info("=== WGCNA-guided DNB Analysis (%s) ===", platform_label)

    dnb_cfg = config["dnb"]
    wgcna_cfg = config["wgcna"]

    # Load WGCNA modules
    wgcna_results_dir = wgcna_results_dir or wgcna_cfg["results_dir"]
    wgcna_df = load_wgcna_modules(wgcna_results_dir)

    # Define masks
    reference_mask = df["TRAJECTORY"] == dnb_cfg["reference_group"]
    adni_cfg = config.get("adni", config.get("ppmi", {}))
    converter_label = adni_cfg.get(
        "converter_group", adni_cfg.get("fast_progressor_group")
    )
    converter_mask = df["TRAJECTORY"] == converter_label

    if reference_mask.sum() < 5:
        logger.warning(
            "Fewer than 5 reference samples on %s — skipping WGCNA DNB",
            platform_label,
        )
        empty = pd.DataFrame()
        return empty, empty, empty

    if converter_mask.sum() < 5:
        logger.warning(
            "Fewer than 5 converter samples on %s — skipping WGCNA DNB",
            platform_label,
        )
        empty = pd.DataFrame()
        return empty, empty, empty

    # Step 1: Score all WGCNA modules with DNB formula
    module_scores = score_modules_dnb(
        df, protein_cols, reference_mask, converter_mask, wgcna_df, config
    )

    if module_scores.empty:
        logger.warning("No modules scored — aborting WGCNA DNB")
        empty = pd.DataFrame()
        return module_scores, empty, empty

    # Step 2: Identify the transition module
    transition_module = identify_transition_module(module_scores)
    if transition_module is None:
        empty = pd.DataFrame()
        return module_scores, empty, empty

    # Step 3: Extract core proteins via dual-filter
    core_proteins = extract_wgcna_core_proteins(
        df,
        protein_cols,
        reference_mask,
        converter_mask,
        transition_module,
        wgcna_df,
        config,
    )

    if len(core_proteins) == 0:
        logger.warning("No core proteins extracted — aborting sDNB")
        return module_scores, core_proteins, pd.DataFrame()

    # Step 4: Run sDNB with WGCNA core proteins
    from src.dnb.sdnb import run_sdnb_analysis

    core_protein_list = core_proteins["protein"].tolist()
    logger.info(
        "Running sDNB with %d WGCNA core proteins (module '%s')",
        len(core_protein_list),
        transition_module,
    )

    # Determine which seq_cols are available in df for outside computation
    all_seq_cols = [c for c in protein_cols if c in df.columns]

    sdnb_scores = run_sdnb_analysis(
        df,
        all_seq_cols,
        reference_mask,
        config,
        core_protein_cols=core_protein_list,
    )

    # Save results
    results_dir = Path(config["paths"].get(
        f"results_dnb_{platform_label}",
        f"data/results/dnb/{platform_label}",
    )) / "wgcna"
    results_dir.mkdir(parents=True, exist_ok=True)

    module_scores.to_csv(results_dir / "module_dnb_scores.csv", index=False)
    core_proteins.to_csv(results_dir / "dnb_core_proteins_wgcna.csv", index=False)
    sdnb_scores.to_csv(results_dir / "sdnb_scores_wgcna.csv", index=False)

    # Annotate with UniProt if SomaScan
    if platform_label == "somascan" and len(core_proteins) > 0:
        uniprot_map_path = Path(config["paths"].get(
            "somascan_uniprot_map", "data/reference/somascan_uniprot_map.csv"
        ))
        if uniprot_map_path.exists():
            try:
                name_map = pd.read_csv(uniprot_map_path, dtype=str)
                name_map["protein"] = "seq." + name_map["Analytes"].str[1:]
                core_annotated = core_proteins.merge(
                    name_map[["protein", "UniProt", "EntrezGeneSymbol", "TargetFullName"]],
                    on="protein",
                    how="left",
                )
                core_annotated.to_csv(
                    results_dir / "dnb_core_proteins_wgcna_annotated.csv", index=False
                )
                logger.info(
                    "Annotated %d/%d WGCNA core proteins with UniProt IDs",
                    core_annotated["UniProt"].notna().sum(),
                    len(core_annotated),
                )
            except Exception as exc:
                logger.warning("Could not annotate WGCNA core proteins: %s", exc)

    logger.info(
        "WGCNA DNB on %s complete: transition module '%s', "
        "%d core proteins, %d participants scored",
        platform_label,
        transition_module,
        len(core_proteins),
        len(sdnb_scores),
    )

    return module_scores, core_proteins, sdnb_scores
