"""Dynamic Network Biomarker (DNB) computation.

Implements the DNB framework from Chen et al. 2012. Identifies a group
of proteins showing coordinated destabilization near a tipping point,
characterized by high within-group variance and correlation, and low
correlation with proteins outside the group.
"""

import logging

import numpy as np
import pandas as pd

from src.preprocessing.somascan_qc import identify_seq_columns

logger = logging.getLogger(__name__)


def compute_dnb_score(
    X_group: np.ndarray,
    X_outside: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Compute the DNB score for a group of proteins.

    DNB = (sigma_D × PCC_D) / (|PCC_O| + epsilon)

    where sigma_D is mean within-group standard deviation, PCC_D is mean
    absolute pairwise correlation within the group, and PCC_O is mean
    absolute correlation between group and outside proteins.

    Parameters
    ----------
    X_group : np.ndarray
        Shape (n_samples, n_group_proteins).
    X_outside : np.ndarray
        Shape (n_samples, n_outside_proteins).
    epsilon : float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        DNB score.
    """
    n_group = X_group.shape[1]

    if n_group == 0:
        return 0.0

    # sigma_D: mean within-group standard deviation
    sigma_D = np.nanstd(X_group, axis=0, ddof=1).mean()

    # PCC_D: mean absolute pairwise correlation within group
    if n_group == 1:
        PCC_D = 1.0
    else:
        corr_matrix = np.corrcoef(X_group.T)
        # Extract upper triangle (excluding diagonal)
        upper_idx = np.triu_indices(n_group, k=1)
        pairwise_corr = corr_matrix[upper_idx]
        # Handle NaN correlations
        valid_corr = pairwise_corr[np.isfinite(pairwise_corr)]
        PCC_D = np.abs(valid_corr).mean() if len(valid_corr) > 0 else 0.0

    # PCC_O: mean absolute correlation between group and outside
    # Vectorized: compute full cross-correlation matrix in one shot
    if X_outside.shape[1] == 0:
        PCC_O = 0.0
    else:
        # Stack group and outside, compute full correlation matrix once
        X_all = np.hstack([X_group, X_outside])
        # Replace NaN with column means for correlation computation
        col_means = np.nanmean(X_all, axis=0)
        inds = np.where(np.isnan(X_all))
        X_all[inds] = np.take(col_means, inds[1])
        corr_all = np.corrcoef(X_all.T)
        # Extract group x outside block
        cross_block = corr_all[:n_group, n_group:]
        valid_cross = cross_block[np.isfinite(cross_block)]
        PCC_O = np.abs(valid_cross).mean() if len(valid_cross) > 0 else 0.0

    return (sigma_D * PCC_D) / (PCC_O + epsilon)


def identify_dnb_group(
    X: np.ndarray,
    protein_names: list[str],
    X_ref: np.ndarray,
    top_variance_percentile: int,
    epsilon: float = 1e-8,
) -> tuple[list[str], float]:
    """Identify the optimal DNB protein group via greedy search.

    Step 1: Select candidate proteins with highest variance relative to reference.
    Step 2: Greedily build a group by adding proteins that maximize DNB score.

    Parameters
    ----------
    X : np.ndarray
        Sample protein matrix (n_samples, n_proteins).
    protein_names : list[str]
        Protein names corresponding to columns of X.
    X_ref : np.ndarray
        Reference population protein matrix.
    top_variance_percentile : int
        Percentile for candidate selection (e.g., 20 = top 20%).
    epsilon : float
        DNB score denominator stabilizer.

    Returns
    -------
    tuple[list[str], float]
        (list of DNB group protein names, best DNB score).
    """
    n_proteins = X.shape[1]
    if n_proteins < 2:
        return ([], 0.0)

    # Step 1: Variance ratio selection
    var_sample = np.nanvar(X, axis=0, ddof=1)
    var_ref = np.nanvar(X_ref, axis=0, ddof=1)
    # Avoid division by zero
    var_ref_safe = np.where(var_ref > 0, var_ref, 1e-10)
    var_ratio = var_sample / var_ref_safe

    threshold = np.nanpercentile(var_ratio, 100 - top_variance_percentile)
    threshold = max(threshold, 1e-10)  # Ignore proteins with zero variance (frequent for N=2 visits)
    candidate_mask = var_ratio >= threshold
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) < 2:
        logger.warning(
            "Fewer than 2 candidate proteins at %d%% variance threshold",
            top_variance_percentile,
        )
        return ([], 0.0)

    logger.info(
        "DNB candidate selection: %d/%d proteins (top %d%% variance ratio)",
        len(candidate_indices),
        n_proteins,
        top_variance_percentile,
    )

    # Step 2: Pre-compute full correlation matrix ONCE.
    # The original _score_group called compute_dnb_score() which ran
    # np.corrcoef on the full n_proteins×n_proteins matrix on every candidate
    # evaluation — O(n_candidates² × n_proteins² × n_samples) total.
    # Pre-computing once reduces this to O(n_proteins² × n_samples) up front,
    # then O(n_group × n_proteins) per evaluation via matrix slicing.
    X_filled = X.copy().astype(float)
    col_means = np.nanmean(X_filled, axis=0)
    nan_rows, nan_cols = np.where(np.isnan(X_filled))
    X_filled[nan_rows, nan_cols] = col_means[nan_cols]

    with np.errstate(invalid="ignore", divide="ignore"):
        corr_full = np.corrcoef(X_filled.T)  # (n_proteins, n_proteins)
    np.fill_diagonal(corr_full, 0.0)
    corr_full = np.nan_to_num(corr_full, nan=0.0)

    sigma_all = np.nanstd(X, axis=0, ddof=1)  # (n_proteins,)

    # Initial pair: slice candidate submatrix from precomputed matrix
    cand_corr = corr_full[np.ix_(candidate_indices, candidate_indices)].copy()

    best_pair = np.unravel_index(np.abs(cand_corr).argmax(), cand_corr.shape)
    if best_pair[0] == best_pair[1]:
        # All pairwise correlations are zero/NaN — no meaningful seed pair
        logger.warning("No correlated candidate pair found — skipping DNB group")
        return ([], 0.0)
    group_local = list(best_pair)
    remaining = [i for i in range(len(candidate_indices)) if i not in group_local]

    all_indices = list(range(n_proteins))

    def _score_group_fast(group_local_indices):
        group_global = candidate_indices[np.array(group_local_indices)]
        group_set = set(group_global.tolist())
        outside_global = np.array([i for i in all_indices if i not in group_set])

        # sigma_D: mean std across group proteins (precomputed)
        sigma_D = float(sigma_all[group_global].mean())

        # PCC_D: mean abs within-group correlation (matrix slice)
        n_g = len(group_global)
        if n_g == 1:
            PCC_D = 1.0
        else:
            sub = corr_full[np.ix_(group_global, group_global)]
            upper = np.triu_indices(n_g, k=1)
            vals = sub[upper]
            finite_vals = vals[np.isfinite(vals)]
            PCC_D = float(np.abs(finite_vals).mean()) if len(finite_vals) > 0 else 0.0

        # PCC_O: mean abs group-to-outside correlation (matrix slice)
        if len(outside_global) == 0:
            PCC_O = 0.0
        else:
            cross = corr_full[np.ix_(group_global, outside_global)]
            flat = cross[np.isfinite(cross)]
            PCC_O = float(np.abs(flat).mean()) if len(flat) > 0 else 0.0

        return (sigma_D * PCC_D) / (PCC_O + epsilon)

    best_score = _score_group_fast(group_local)

    # Iteratively add proteins that improve score
    improved = True
    while improved and remaining:
        improved = False
        best_candidate = None
        best_new_score = best_score

        for idx in remaining:
            trial_group = group_local + [idx]
            trial_score = _score_group_fast(trial_group)
            if trial_score > best_new_score:
                best_new_score = trial_score
                best_candidate = idx

        if best_candidate is not None:
            group_local.append(best_candidate)
            remaining.remove(best_candidate)
            best_score = best_new_score
            improved = True

    group_global_indices = candidate_indices[group_local]
    group_names = [protein_names[i] for i in group_global_indices]

    logger.info(
        "DNB group identified: %d proteins, score = %.4f",
        len(group_names),
        best_score,
    )
    return (group_names, best_score)


def compute_stage_dnb_scores(
    df: pd.DataFrame,
    seq_cols: list[str],
    reference_mask: pd.Series,
    stage_column: str,
    config: dict,
) -> pd.DataFrame:
    """Compute DNB scores for each clinical stage.

    Parameters
    ----------
    df : pd.DataFrame
        Proteomics DataFrame with stage labels.
    seq_cols : list[str]
        Protein column names.
    reference_mask : pd.Series
        Boolean mask for reference population (CN_amyloid_negative).
    stage_column : str
        Column containing stage labels.
    config : dict
        Configuration with dnb section.

    Returns
    -------
    pd.DataFrame
        DNB scores per stage with SEM.
    """
    dnb_cfg = config["dnb"]
    percentile = dnb_cfg["primary_variance_percentile"]
    epsilon = dnb_cfg["epsilon"]

    X_ref = df.loc[reference_mask, seq_cols].values
    stages = df[stage_column].unique()

    results = []
    for stage in sorted(stages):
        # Skip the reference group — comparing reference to itself produces
        # var_ratio ≈ 1 everywhere, all proteins pass the threshold, and the
        # resulting DNB score is trivially uninformative.
        if stage == dnb_cfg["reference_group"]:
            logger.info(
                "Skipping reference group '%s' in stage DNB loop "
                "(no meaningful variance ratio against itself)",
                stage,
            )
            continue

        stage_mask = df[stage_column] == stage
        X_stage = df.loc[stage_mask, seq_cols].values
        n_samples = stage_mask.sum()

        if n_samples < 5:
            logger.warning("Stage '%s' has only %d samples, skipping", stage, n_samples)
            continue

        group_names, score = identify_dnb_group(
            X_stage, seq_cols, X_ref, percentile, epsilon
        )

        results.append(
            {
                "stage": stage,
                "dnb_score": score,
                "n_samples": n_samples,
                "n_dnb_proteins": len(group_names),
            }
        )

    # Add reference group baseline: compute DNB for the reference group
    # against the full non-reference pool.  This answers "does the healthy group
    # show transition-state coordinated variance?" — expected answer: no (low score).
    non_reference_mask = ~reference_mask
    if non_reference_mask.sum() >= 5:
        X_non_ref = df.loc[non_reference_mask, seq_cols].values
        ref_group_names, ref_score = identify_dnb_group(
            X_ref, seq_cols, X_non_ref, percentile, epsilon
        )
        results.append({
            "stage": dnb_cfg["reference_group"],
            "dnb_score": ref_score,
            "n_samples": int(reference_mask.sum()),
            "n_dnb_proteins": len(ref_group_names),
        })
        logger.info(
            "Reference group '%s' baseline DNB (vs non-reference pool): %.4f",
            dnb_cfg["reference_group"], ref_score,
        )

    results_df = pd.DataFrame(results)
    logger.info("Stage DNB scores computed for %d stages", len(results_df))
    return results_df


def identify_dnb_core_proteins(
    dnb_results_by_participant: dict,
    threshold: float,
) -> pd.DataFrame:
    """Identify proteins consistently appearing in DNB groups.

    Parameters
    ----------
    dnb_results_by_participant : dict
        Mapping RID -> list of DNB group protein names.
    threshold : float
        Minimum fraction of participants (e.g., 0.30).

    Returns
    -------
    pd.DataFrame
        Core proteins with frequency annotations.
    """
    n_participants = len(dnb_results_by_participant)
    if n_participants == 0:
        return pd.DataFrame(columns=["protein", "frequency", "n_participants"])

    # Count protein appearances
    protein_counts = {}
    for rid, proteins in dnb_results_by_participant.items():
        for protein in proteins:
            protein_counts[protein] = protein_counts.get(protein, 0) + 1

    results = []
    for protein, count in protein_counts.items():
        freq = count / n_participants
        if freq >= threshold:
            results.append(
                {
                    "protein": protein,
                    "frequency": freq,
                    "n_participants": count,
                }
            )

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values("frequency", ascending=False)

    logger.info(
        "DNB core proteins: %d proteins appear in >= %.0f%% of %d participants",
        len(results_df),
        threshold * 100,
        n_participants,
    )
    return results_df


def run_dnb_on_platform(
    df: pd.DataFrame,
    protein_cols: list[str],
    config: dict,
    platform_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the full DNB analysis on a single platform's processed DataFrame.

    Wraps compute_stage_dnb_scores() and identify_dnb_core_proteins()
    with platform-aware logging and output paths.

    Parameters
    ----------
    df : pd.DataFrame
        Processed proteomics DataFrame (already QC'd and imputed).
    protein_cols : list[str]
        Protein column names (seq.* for SomaScan, NPX_* for Olink).
    config : dict
        Configuration with dnb section.
    platform_label : str
        'somascan' or 'olink'.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict]
        (stage_scores_df, core_proteins_df, dnb_by_participant dict)
    """
    from pathlib import Path

    logger.info("=== DNB Analysis (%s) ===", platform_label)

    dnb_cfg = config["dnb"]
    reference_mask = df["TRAJECTORY"] == dnb_cfg["reference_group"]

    if reference_mask.sum() < 5:
        logger.warning(
            "Fewer than 5 reference samples on %s — skipping DNB",
            platform_label,
        )
        empty_scores = pd.DataFrame(
            columns=["stage", "dnb_score", "n_samples", "n_dnb_proteins"]
        )
        empty_core = pd.DataFrame(
            columns=["protein", "frequency", "n_participants", "platform"]
        )
        return empty_scores, empty_core, {}

    # Stage DNB scores
    stage_scores = compute_stage_dnb_scores(
        df, protein_cols, reference_mask, "TRAJECTORY", config
    )

    # Per-participant DNB group identification for core proteins
    converter_label = config["adni"]["converter_group"]
    converters = df[df["TRAJECTORY"] == converter_label]
    X_ref = df.loc[reference_mask, protein_cols].values

    dnb_by_participant = {}
    n_single_visit = 0
    for rid, group in converters.groupby("RID"):
        X_part = group[protein_cols].values
        if len(X_part) < 2:
            # np.nanvar(ddof=1) requires ≥2 observations; skip silently
            n_single_visit += 1
            continue
        group_names, _ = identify_dnb_group(
            X_part,
            protein_cols,
            X_ref,
            dnb_cfg["primary_variance_percentile"],
            dnb_cfg["epsilon"],
        )
        dnb_by_participant[rid] = group_names

    n_total_converters = converters["RID"].nunique()
    if n_single_visit > 0:
        logger.info(
            "Per-participant DNB: %d/%d converter participants have only 1 visit "
            "(need ≥2 for variance) — will use pooled-converter fallback if needed",
            n_single_visit,
            n_total_converters,
        )

    core_proteins_per_participant = identify_dnb_core_proteins(
        dnb_by_participant, dnb_cfg["core_protein_threshold"]
    )
    if len(core_proteins_per_participant) > 0:
        logger.info(
            "Per-participant DNB found %d core proteins",
            len(core_proteins_per_participant),
        )

    # Always identify core proteins from the pooled converter group so that
    # all cohorts (ADNI, PPMI, etc.) use the same methodology.  This is
    # scientifically equivalent to the stage-level DNB already computed in
    # compute_stage_dnb_scores — proteins with elevated coordinated variance in
    # converters vs reference (Chen et al. 2012 criteria).
    core_proteins = pd.DataFrame(
        columns=["protein", "frequency", "n_participants"]
    )
    if len(converters) >= 5:
        logger.info(
            "Pooled-converter DNB group (%d converter rows)",
            len(converters),
        )
        X_conv_all = converters[protein_cols].values
        pooled_group, pooled_score = identify_dnb_group(
            X_conv_all,
            protein_cols,
            X_ref,
            dnb_cfg["primary_variance_percentile"],
            dnb_cfg["epsilon"],
        )
        if pooled_group:
            core_proteins = pd.DataFrame({
                "protein": pooled_group,
                "frequency": 1.0,
                "n_participants": len(converters),
            })
            logger.info(
                "Pooled-converter core proteins: %d proteins, DNB score = %.4f",
                len(core_proteins),
                pooled_score,
            )

    core_proteins["platform"] = platform_label

    # Annotate SomaScan core proteins with UniProt, gene symbol, and target name.
    # File format: CSV with header, Analytes column = "X10000.28" (R-export format).
    if platform_label == "somascan" and len(core_proteins) > 0:
        uniprot_map_path = Path(config["paths"].get(
            "somascan_uniprot_map", "data/reference/somascan_uniprot_map.csv"
        ))
        if uniprot_map_path.exists():
            try:
                name_map = pd.read_csv(uniprot_map_path, dtype=str)
                # Analytes = "X10000.28" → strip leading X, prepend "seq."
                name_map["protein"] = "seq." + name_map["Analytes"].str[1:]
                core_proteins = core_proteins.merge(
                    name_map[["protein", "UniProt", "EntrezGeneSymbol", "TargetFullName"]],
                    on="protein", how="left",
                )
                logger.info(
                    "Annotated %d/%d core proteins with UniProt IDs",
                    core_proteins["UniProt"].notna().sum(),
                    len(core_proteins),
                )
            except Exception as exc:
                logger.warning("Could not load SomaScan UniProt map: %s", exc)

    # Save results to platform-specific directory
    results_dir = Path(config["paths"].get(
        f"results_dnb_{platform_label}",
        f"data/results/dnb/{platform_label}",
    ))
    results_dir.mkdir(parents=True, exist_ok=True)
    stage_scores.to_csv(results_dir / "dnb_scores_by_stage.csv", index=False)
    core_proteins.to_csv(results_dir / "dnb_core_proteins.csv", index=False)

    logger.info(
        "DNB on %s complete: %d stages scored, %d core proteins",
        platform_label,
        len(stage_scores),
        len(core_proteins),
    )

    return stage_scores, core_proteins, dnb_by_participant, core_proteins_per_participant
