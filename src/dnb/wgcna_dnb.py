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


def _compute_stage_components(
    X_group: np.ndarray,
    X_outside: np.ndarray,
    epsilon: float,
) -> dict:
    """Compute DNB score and its components for a single stage slice."""
    dnb_score = compute_dnb_score(X_group, X_outside, epsilon=epsilon)
    mean_sd = float(np.nanstd(X_group, axis=0, ddof=1).mean())

    n_prot = X_group.shape[1]
    if n_prot >= 2:
        corr_mat = np.corrcoef(X_group.T)
        upper = corr_mat[np.triu_indices(n_prot, k=1)]
        valid = upper[np.isfinite(upper)]
        mean_pcc_within = float(np.abs(valid).mean()) if len(valid) > 0 else 0.0
    else:
        mean_pcc_within = 1.0

    if X_outside.shape[1] > 0:
        X_all = np.hstack([X_group, X_outside])
        col_means = np.nanmean(X_all, axis=0)
        inds = np.where(np.isnan(X_all))
        if inds[0].size > 0:
            X_all[inds] = np.take(col_means, inds[1])
        corr_all = np.corrcoef(X_all.T)
        cross_block = corr_all[:n_prot, n_prot:]
        valid_cross = cross_block[np.isfinite(cross_block)]
        mean_pcc_outside = float(np.abs(valid_cross).mean()) if len(valid_cross) > 0 else 0.0
    else:
        mean_pcc_outside = 0.0

    return {
        "dnb_score": dnb_score,
        "mean_sd": mean_sd,
        "mean_pcc_within": mean_pcc_within,
        "mean_pcc_outside": mean_pcc_outside,
    }


def compute_per_stage_dnb(
    df: pd.DataFrame,
    module_proteins: list[str],
    outside_proteins: list[str],
    trajectory_groups: list[str],
    epsilon: float = 1e-8,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute DNB score of a module separately for each trajectory stage.

    This produces the per-stage comparison table showing how DNB rises
    as disease progresses from CN through MCI to Dementia, with bootstrap
    95% confidence intervals and permutation p-values.

    Parameters
    ----------
    df : pd.DataFrame
        Full proteomics DataFrame with TRAJECTORY column.
    module_proteins : list[str]
        kME-filtered proteins from the transition module.
    outside_proteins : list[str]
        All non-module proteins.
    trajectory_groups : list[str]
        Trajectory labels to score (in clinical order).
    epsilon : float
        Denominator stabilizer.
    n_bootstrap : int
        Number of bootstrap resamples for CIs (default 1000).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (stage_scores, permutation_results)
        stage_scores: Per-stage DNB scores with CI columns.
        permutation_results: Pairwise permutation test p-values.
    """
    rng = np.random.default_rng(random_seed)
    results = []

    # Subsample outside proteins for bootstrap/permutation tractability.
    # With ~60 module proteins x 500 outside, we get 30,000 cross-correlations
    # per iteration — more than sufficient for stable PCC_outside estimates.
    # Observed (point estimate) scores use the full outside set.
    max_outside = 500
    if len(outside_proteins) > max_outside:
        outside_sub_idx = rng.choice(len(outside_proteins), max_outside, replace=False)
        outside_sub = [outside_proteins[i] for i in sorted(outside_sub_idx)]
    else:
        outside_sub = outside_proteins

    for stage in trajectory_groups:
        mask = df["TRAJECTORY"] == stage
        n = mask.sum()
        if n < 3:
            logger.warning("Stage '%s': %d samples (< 3), skipping", stage, n)
            continue

        X_group = df.loc[mask, module_proteins].values
        X_outside_full = df.loc[mask, outside_proteins].values
        X_outside_sub = df.loc[mask, outside_sub].values

        # Point estimates use full outside set
        obs = _compute_stage_components(X_group, X_outside_full, epsilon)

        # Bootstrap CIs use subsampled outside for speed
        boot_dnb = []
        boot_pcc_out = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            b_group = X_group[idx]
            b_outside = X_outside_sub[idx]
            b = _compute_stage_components(b_group, b_outside, epsilon)
            boot_dnb.append(b["dnb_score"])
            boot_pcc_out.append(b["mean_pcc_outside"])

        results.append({
            "stage": stage,
            "dnb_score": obs["dnb_score"],
            "dnb_ci_lower": float(np.percentile(boot_dnb, 2.5)),
            "dnb_ci_upper": float(np.percentile(boot_dnb, 97.5)),
            "n_samples": n,
            "mean_sd": obs["mean_sd"],
            "mean_pcc_within": obs["mean_pcc_within"],
            "mean_pcc_outside": obs["mean_pcc_outside"],
            "pcc_outside_ci_lower": float(np.percentile(boot_pcc_out, 2.5)),
            "pcc_outside_ci_upper": float(np.percentile(boot_pcc_out, 97.5)),
        })

    results_df = pd.DataFrame(results)

    # Permutation tests: pairwise differences in PCC_outside
    # PCC_outside is the key DNB indicator of approaching a critical
    # transition (monotonic decline), whereas the composite DNB score
    # is dominated by SD variance and lacks discriminative power.
    perm_results = []
    stage_data_full = {}
    stage_data_sub = {}
    for stage in trajectory_groups:
        mask = df["TRAJECTORY"] == stage
        if mask.sum() >= 3:
            stage_data_full[stage] = (
                df.loc[mask, module_proteins].values,
                df.loc[mask, outside_proteins].values,
            )
            stage_data_sub[stage] = (
                df.loc[mask, module_proteins].values,
                df.loc[mask, outside_sub].values,
            )

    stages_list = list(stage_data_full.keys())
    for i in range(len(stages_list)):
        for j in range(i + 1, len(stages_list)):
            s_a, s_b = stages_list[i], stages_list[j]
            # Observed values use full outside set
            obs_a = _compute_stage_components(*stage_data_full[s_a], epsilon)
            obs_b = _compute_stage_components(*stage_data_full[s_b], epsilon)
            obs_dnb_diff = obs_b["dnb_score"] - obs_a["dnb_score"]
            obs_pcc_diff = obs_b["mean_pcc_outside"] - obs_a["mean_pcc_outside"]

            # Permutation uses subsampled outside for speed
            X_a_grp, X_a_out = stage_data_sub[s_a]
            X_b_grp, X_b_out = stage_data_sub[s_b]
            n_a, n_b = len(X_a_grp), len(X_b_grp)
            pooled_grp = np.vstack([X_a_grp, X_b_grp])
            pooled_out = np.vstack([X_a_out, X_b_out])

            # Observed PCC_outside difference with subsampled outside
            # for consistent comparison with permuted differences
            obs_a_sub = _compute_stage_components(X_a_grp, X_a_out, epsilon)
            obs_b_sub = _compute_stage_components(X_b_grp, X_b_out, epsilon)
            obs_pcc_diff_sub = (
                obs_b_sub["mean_pcc_outside"] - obs_a_sub["mean_pcc_outside"]
            )

            n_exceed = 0
            for _ in range(n_bootstrap):
                perm_idx = rng.permutation(n_a + n_b)
                p_a_grp = pooled_grp[perm_idx[:n_a]]
                p_a_out = pooled_out[perm_idx[:n_a]]
                p_b_grp = pooled_grp[perm_idx[n_a:]]
                p_b_out = pooled_out[perm_idx[n_a:]]
                perm_a = _compute_stage_components(p_a_grp, p_a_out, epsilon)
                perm_b = _compute_stage_components(p_b_grp, p_b_out, epsilon)
                perm_diff = (
                    perm_b["mean_pcc_outside"] - perm_a["mean_pcc_outside"]
                )
                if abs(perm_diff) >= abs(obs_pcc_diff_sub):
                    n_exceed += 1

            p_value = (n_exceed + 1) / (n_bootstrap + 1)
            perm_results.append({
                "stage_a": s_a,
                "stage_b": s_b,
                "pcc_outside_diff": obs_pcc_diff,
                "dnb_diff": obs_dnb_diff,
                "p_value": p_value,
            })

    perm_df = pd.DataFrame(perm_results)

    if len(results_df) > 0:
        logger.info(
            "Per-stage DNB scores (with 95%% CI):\n%s",
            results_df.to_string(index=False),
        )
    if len(perm_df) > 0:
        logger.info(
            "Permutation test results:\n%s",
            perm_df.to_string(index=False),
        )

    return results_df, perm_df


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

    # Step 2b: Per-stage DNB scoring for the transition module
    wgcna_cfg = config["wgcna"]
    kme_threshold = wgcna_cfg["kme_threshold"]
    available_proteins = set(protein_cols)
    stage_module_proteins = wgcna_df[
        (wgcna_df["module"] == transition_module) & (wgcna_df["kME"] >= kme_threshold)
    ]["protein"].tolist()
    stage_module_proteins = [p for p in stage_module_proteins if p in available_proteins]
    stage_outside_proteins = [p for p in protein_cols if p not in set(stage_module_proteins)]

    trajectory_groups = [
        g for g in ["CN_amyloid_negative", "CN_amyloid_positive", "stable_MCI", "MCI_to_Dementia"]
        if (df["TRAJECTORY"] == g).sum() >= 3
    ]
    bootstrap_n = config.get("validation", {}).get("bootstrap_n", 1000)
    seed = config.get("random_seed", 42)
    stage_dnb_scores, stage_perm = compute_per_stage_dnb(
        df, stage_module_proteins, stage_outside_proteins,
        trajectory_groups, epsilon=config["dnb"]["epsilon"],
        n_bootstrap=bootstrap_n, random_seed=seed,
    )

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
    if len(stage_dnb_scores) > 0:
        stage_dnb_scores.to_csv(results_dir / "stage_dnb_scores.csv", index=False)
    if len(stage_perm) > 0:
        stage_perm.to_csv(results_dir / "stage_dnb_permutation.csv", index=False)

    # Step 5: Module eigengene ROC for clinical utility
    try:
        from sklearn.decomposition import PCA
        from sklearn.metrics import roc_auc_score

        converter_label_val = converter_label or "MCI_to_Dementia"
        stable_label = "stable_MCI"
        roc_mask = df["TRAJECTORY"].isin([stable_label, converter_label_val])
        roc_df = df.loc[roc_mask].copy()

        if len(roc_df) >= 10 and roc_df["TRAJECTORY"].nunique() == 2:
            X_eigen = roc_df[stage_module_proteins].values
            # Impute NaN with column means
            col_means_e = np.nanmean(X_eigen, axis=0)
            nan_inds = np.where(np.isnan(X_eigen))
            if nan_inds[0].size > 0:
                X_eigen[nan_inds] = np.take(col_means_e, nan_inds[1])

            pca = PCA(n_components=1, random_state=seed)
            eigengene = pca.fit_transform(X_eigen).ravel()

            y_roc = (roc_df["TRAJECTORY"] == converter_label_val).astype(int).values
            auc = roc_auc_score(y_roc, eigengene)
            # Flip if AUC < 0.5
            if auc < 0.5:
                eigengene = -eigengene
                auc = 1.0 - auc

            # Bootstrap CI
            rng_roc = np.random.default_rng(seed)
            boot_aucs = []
            for _ in range(bootstrap_n):
                idx = rng_roc.choice(len(y_roc), len(y_roc), replace=True)
                y_b, s_b = y_roc[idx], eigengene[idx]
                if y_b.sum() > 0 and y_b.sum() < len(y_b):
                    b_auc = roc_auc_score(y_b, s_b)
                    boot_aucs.append(b_auc)

            ci_lo = float(np.percentile(boot_aucs, 2.5)) if boot_aucs else np.nan
            ci_hi = float(np.percentile(boot_aucs, 97.5)) if boot_aucs else np.nan

            eigen_roc = pd.DataFrame([{
                "module": transition_module,
                "n_proteins": len(stage_module_proteins),
                "n_stable_mci": int((y_roc == 0).sum()),
                "n_converters": int((y_roc == 1).sum()),
                "auc": auc,
                "auc_ci_lower": ci_lo,
                "auc_ci_upper": ci_hi,
                "variance_explained": float(pca.explained_variance_ratio_[0]),
            }])
            eigen_roc.to_csv(results_dir / "eigengene_roc_results.csv", index=False)
            logger.info(
                "Eigengene ROC: AUC = %.3f (95%% CI: %.3f–%.3f), variance explained = %.1f%%",
                auc, ci_lo, ci_hi, pca.explained_variance_ratio_[0] * 100,
            )
        else:
            logger.warning("Insufficient samples for eigengene ROC (%d)", len(roc_df))
    except Exception as exc:
        logger.warning("Eigengene ROC analysis failed: %s", exc)

    # Annotate with UniProt if SomaScan
    if platform_label.startswith("somascan") and len(core_proteins) > 0:
        uniprot_map_path = Path(config["paths"].get(
            "somascan_uniprot_map", "data/reference/somascan_uniprot_map.csv"
        ))
        if uniprot_map_path.exists():
            try:
                name_map = pd.read_csv(uniprot_map_path, dtype=str)
                # Primary join: Analyte-derived protein ID (e.g. X7011.8 -> seq.7011.8)
                name_map["protein"] = "seq." + name_map["Analytes"].str[1:]
                annot_cols = ["UniProt", "EntrezGeneSymbol", "TargetFullName"]
                core_annotated = core_proteins.merge(
                    name_map[["protein"] + annot_cols],
                    on="protein",
                    how="left",
                )
                # Fallback join via SomaId for unmatched proteins (PPMI uses
                # different aptamer versions that share SomaId but have
                # different SeqId numbers, e.g. SL007011 covers both
                # X11591.43 in ADNI and seq.7011.8 in PPMI)
                # Manual annotations for PPMI aptamers not in ADNI reference
                _manual_annot = {
                    "seq.14757.144": ("P55075", "FGF8", "Fibroblast growth factor 8"),
                    "seq.10485.56": ("P43358", "MAGEA4", "MAGE family member A4"),
                }
                for prot, (uniprot, symbol, fullname) in _manual_annot.items():
                    mask_man = core_annotated["protein"] == prot
                    if mask_man.any() and core_annotated.loc[mask_man, "UniProt"].isna().all():
                        core_annotated.loc[mask_man, "UniProt"] = uniprot
                        core_annotated.loc[mask_man, "EntrezGeneSymbol"] = symbol
                        core_annotated.loc[mask_man, "TargetFullName"] = fullname

                unmapped = core_annotated["UniProt"].isna()
                if unmapped.any() and "SomaId" in name_map.columns:
                    # Extract numeric part from protein ID: seq.7011.8 -> 7011
                    core_annotated["_seq_base"] = (
                        core_annotated["protein"]
                        .str.replace(r"^seq\.", "", regex=True)
                        .str.split(".")
                        .str[0]
                    )
                    # Build SomaId lookup: SL007011 -> 7011
                    soma_lookup = name_map.drop_duplicates("SomaId").copy()
                    soma_lookup["_soma_base"] = (
                        soma_lookup["SomaId"]
                        .str.replace(r"^SL0*", "", regex=True)
                    )
                    for idx in core_annotated[unmapped].index:
                        seq_base = core_annotated.loc[idx, "_seq_base"]
                        match = soma_lookup[soma_lookup["_soma_base"] == seq_base]
                        if len(match) > 0:
                            for col in annot_cols:
                                core_annotated.loc[idx, col] = match.iloc[0][col]
                    core_annotated = core_annotated.drop(columns=["_seq_base"])
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
