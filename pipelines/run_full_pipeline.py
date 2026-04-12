"""Full pipeline orchestrator for CriticalNeuroMap (cross-sectional mode).

Runs all preprocessing and analysis steps in sequence with checkpointing.
Each stage checks for existing output and skips if already completed.

Stage order:
  0 - env_check            Environment verification
  1 - adni_preprocess      ADNI SomaScan preprocessing
  2 - ppmi_preprocess      PPMI SomaScan preprocessing
  3 - wgcna                WGCNA module detection (R)
  4 - dnb_wgcna_somascan   WGCNA-guided DNB analysis
  5 - netmedpy             Interactome proximity (NetMedPy)
  6 - cross_platform       Cross-platform Golden Set validation
  7 - csd                  CSD rolling-window analysis
  8 - validation           Biomarker comparison and ROC analysis
  9 - wgcna_ppmi           WGCNA PPMI replication
 10 - ppmi_replication     Replicate on PPMI data
 11 - figures              Generate all publication figures

For longitudinal Knight-ADRC analysis, use run_pipeline.py instead.

Usage:
    python pipelines/run_full_pipeline.py                        # run all stages
    python pipelines/run_full_pipeline.py --stage wgcna          # run single stage
    python pipelines/run_full_pipeline.py --force                # force rerun all
"""

import argparse
import logging
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work when the script
# is run from any working directory (e.g. `python pipelines/run_full_pipeline.py`).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config/config.yaml not found. Run from project root.")
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_environment(config: dict) -> None:
    """Stage 0: Verify data files, Python packages, and R packages."""
    logger.info("=== Stage 0: Environment Check ===")

    # Check critical Python imports
    required_packages = [
        "numpy", "pandas", "scipy", "sklearn", "statsmodels",
        "pyarrow", "yaml", "matplotlib", "seaborn", "tqdm",
    ]
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.error("Missing Python packages: %s", missing)
        logger.error("Run: pip install -r requirements.txt")
        sys.exit(1)

    logger.info("All required Python packages available")

    # Check for data files (warnings, not errors — user may not have all data yet)
    adnimerge_path = Path(config["paths"]["adnimerge_csv"])
    if not adnimerge_path.exists():
        logger.warning(
            "ADNIMERGE.csv not found at %s. Run 'Rscript R/adnimerge_extract.R' first.",
            adnimerge_path,
        )

    somascan_dir = Path(config["paths"]["adni_somascan_dir"])
    if somascan_dir.exists() and not any(somascan_dir.glob("*.csv")):
        logger.warning("No SomaScan CSV files found in %s", somascan_dir)

    # Check for Olink data (informational)
    olink_dir = Path(config["paths"]["adni_olink_dir"])
    if not olink_dir.exists() or not any(olink_dir.glob("*.csv")):
        logger.warning(
            "No Olink data found in %s — cross-platform validation will be skipped",
            olink_dir,
        )

    logger.info("Environment check complete")


def run_adni_preprocessing(config: dict) -> None:
    """Stage 1: Load, merge, QC, and batch-correct ADNI SomaScan data."""
    logger.info("=== Stage 1: ADNI Preprocessing ===")
    import pandas as pd

    from src.preprocessing.adni_loader import (
        assign_conversion_labels,
        load_adnimerge,
        load_plasma_biomarkers_adni,
        load_somascan_adni,
        merge_adni_data,
    )
    from src.preprocessing.batch_correction import run_combat_r, validate_batch_correction
    from src.preprocessing.somascan_qc import identify_seq_columns, run_somascan_qc_pipeline

    # Load data
    adnimerge = load_adnimerge(config)
    somascan = load_somascan_adni(config)
    biomarkers = load_plasma_biomarkers_adni(config)

    # Assign conversion labels
    adnimerge = assign_conversion_labels(adnimerge, config)

    # Merge
    merged = merge_adni_data(adnimerge, somascan, biomarkers)

    # QC pipeline
    df_before = merged.copy()
    cleaned, seq_cols = run_somascan_qc_pipeline(merged, config)

    # Batch correction
    batch_cfg = config["batch_correction"]
    if batch_cfg["batch_column"] in cleaned.columns:
        df_corrected = run_combat_r(
            cleaned, seq_cols, batch_cfg["batch_column"], batch_cfg["protected_columns"],
            use_r_combat=batch_cfg.get("use_r_combat", True),
        )
        validate_batch_correction(
            cleaned, df_corrected, seq_cols,
            batch_cfg["batch_column"], "TRAJECTORY",
            config["paths"]["batch_correction_qc"],
            random_seed=config["random_seed"],
        )
        cleaned = df_corrected

    # Save as Parquet (Rule 5)
    output_path = Path(config["paths"]["adni_clean_parquet"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False)
    logger.info("ADNI preprocessed data saved to %s", output_path)


def run_ppmi_preprocessing(config: dict) -> None:
    """Stage 2: Load, merge, QC, and batch-correct PPMI data."""
    logger.info("=== Stage 2: PPMI Preprocessing ===")

    from src.preprocessing.batch_correction import run_combat_r
    from src.preprocessing.ppmi_loader import (
        assign_progression_labels,
        load_csf_biomarkers_ppmi,
        load_ppmi_clinical,
        load_somascan_ppmi,
        merge_ppmi_data,
    )
    from src.preprocessing.somascan_qc import run_somascan_qc_pipeline

    clinical = load_ppmi_clinical(config)
    somascan = load_somascan_ppmi(config)
    biomarkers = load_csf_biomarkers_ppmi(config)

    clinical = assign_progression_labels(clinical, config)
    merged = merge_ppmi_data(clinical, somascan, biomarkers)

    cleaned, seq_cols = run_somascan_qc_pipeline(merged, config)

    batch_cfg = config["batch_correction"]
    if batch_cfg["batch_column"] in cleaned.columns:
        cleaned = run_combat_r(
            cleaned, seq_cols, batch_cfg["batch_column"], batch_cfg["protected_columns"],
            use_r_combat=batch_cfg.get("use_r_combat", True),
        )

    output_path = Path(config["paths"]["ppmi_clean_parquet"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False)
    logger.info("PPMI preprocessed data saved to %s", output_path)


def run_wgcna(config: dict) -> None:
    """Stage 3a: WGCNA co-expression module detection (R script).

    Runs WGCNA on the full ADNI proteomics matrix to identify
    co-expression modules and module membership (kME) scores.
    Output: data/results/wgcna/wgcna_modules.csv
    """
    logger.info("=== Stage 3a: WGCNA Module Detection ===")

    wgcna_script = Path("R/wgcna_module_detection.R")
    if not wgcna_script.exists():
        logger.error("WGCNA R script not found at %s", wgcna_script)
        return

    input_path = Path(config["paths"]["adni_clean_parquet"])
    if not input_path.exists():
        raise FileNotFoundError(
            f"Preprocessed ADNI data not found at {input_path}. "
            "Run adni_preprocess first."
        )

    logger.info("Running WGCNA module detection via R...")
    result = subprocess.run(
        ["Rscript", str(wgcna_script)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error("WGCNA R script failed:\n%s", result.stderr)
        raise RuntimeError("WGCNA module detection failed")

    # Log R output
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            logger.info("[R] %s", line.strip())

    # Verify output exists
    wgcna_dir = Path(config["wgcna"]["results_dir"])
    modules_csv = wgcna_dir / "wgcna_modules.csv"
    if modules_csv.exists():
        import pandas as pd
        wgcna_df = pd.read_csv(modules_csv)
        n_modules = wgcna_df["module"].nunique()
        n_proteins = len(wgcna_df)
        logger.info(
            "WGCNA complete: %d modules detected across %d proteins",
            n_modules,
            n_proteins,
        )
    else:
        logger.warning("WGCNA output not found at %s", modules_csv)


def run_dnb_wgcna_somascan(config: dict) -> None:
    """Stage 3b: WGCNA-guided DNB analysis on ADNI SomaScan data.

    Uses WGCNA co-expression modules to score DNB at the module level,
    identifies the transition module, and extracts core proteins via
    dual-filter (variance ratio + kME ranking).
    """
    logger.info("=== Stage 3b: WGCNA-guided DNB Analysis — SomaScan ===")
    import pandas as pd

    from src.dnb.wgcna_dnb import run_wgcna_dnb_analysis
    from src.preprocessing.somascan_qc import identify_seq_columns, impute_missing_values

    # Check WGCNA modules exist
    wgcna_dir = Path(config["wgcna"]["results_dir"])
    if not (wgcna_dir / "wgcna_modules.csv").exists():
        logger.warning(
            "WGCNA modules not found at %s — run 'wgcna' stage first. Skipping.",
            wgcna_dir,
        )
        return

    input_path = Path(config["paths"]["adni_clean_parquet"])
    if not input_path.exists():
        raise FileNotFoundError(f"Preprocessed ADNI data not found at {input_path}")

    df = pd.read_parquet(input_path)
    seq_cols = identify_seq_columns(df)

    # Impute for cross-sectional DNB
    df_imputed = impute_missing_values(
        df, seq_cols, config["proteomics"]["imputation_method"]
    )

    # Run WGCNA-guided DNB
    module_scores, core_proteins, sdnb_scores = run_wgcna_dnb_analysis(
        df_imputed, seq_cols, config, "somascan"
    )

    if len(core_proteins) > 0:
        logger.info(
            "WGCNA DNB identified %d core proteins",
            len(core_proteins),
        )
    else:
        logger.warning("WGCNA DNB produced no core proteins")

    logger.info("WGCNA-guided DNB SomaScan analysis complete")


def run_wgcna_ppmi(config: dict) -> None:
    """Stage 8b: WGCNA co-expression module detection for PPMI (R script)."""
    logger.info("=== Stage 8b: WGCNA Module Detection (PPMI) ===")

    wgcna_script = Path("R/wgcna_module_detection.R")
    if not wgcna_script.exists():
        logger.error("WGCNA R script not found at %s", wgcna_script)
        return

    input_path = Path(config["paths"]["ppmi_clean_parquet"])
    if not input_path.exists():
        logger.warning("PPMI preprocessed data not found, skipping WGCNA PPMI")
        return

    out_dir = Path("data/results/ppmi/wgcna")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running WGCNA module detection via R for PPMI...")
    result = subprocess.run(
        ["Rscript", str(wgcna_script), str(input_path), str(out_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error("WGCNA PPMI R script failed:\n%s", result.stderr)
        raise RuntimeError("WGCNA PPMI module detection failed")

    for line in result.stdout.strip().split("\n"):
        if line.strip():
            logger.info("[R] %s", line.strip())

    logger.info("WGCNA PPMI stage complete")


def run_netmedpy(config: dict) -> None:
    """Stage 3c: Interactome proximity analysis via NetMedPy."""
    logger.info("=== Stage 3b: Interactome Proximity Analysis (NetMedPy) ===")
    result = subprocess.run(
        [sys.executable, 'src/network_medicine/interactome_proximity.py'],
        check=False  # Don't abort full pipeline if interactome missing
    )
    if result.returncode != 0:
        logger.warning("Stage 3b (NetMedPy) failed — continuing pipeline. "
                       "Ensure data/reference/interactome_ppi.pkl exists (see README for download instructions).")


def run_cross_platform(config: dict) -> None:
    """Stage 5: Cross-platform validation and Golden Set construction.

    Requires both SomaScan and Olink DNB results. Skips gracefully
    if Olink DNB was skipped.
    """
    logger.info("=== Stage 5: Cross-Platform Validation (Golden Set) ===")
    import pandas as pd

    from src.cross_platform.golden_set import (
        compute_golden_set,
        compute_golden_set_statistics,
    )
    from src.cross_platform.platform_concordance import (
        generate_concordance_report,
        positive_control_concordance,
    )

    # Check that both platform DNB results exist
    soma_core_path = Path(config["paths"]["results_dnb_somascan"]) / "dnb_core_proteins.csv"
    olink_core_path = Path(config["paths"]["results_dnb_olink"]) / "dnb_core_proteins.csv"

    if not soma_core_path.exists():
        logger.error(
            "SomaScan DNB core proteins not found at %s — run dnb_somascan first",
            soma_core_path,
        )
        return

    if not olink_core_path.exists():
        logger.warning(
            "Olink DNB core proteins not found at %s — "
            "skipping cross-platform validation",
            olink_core_path,
        )
        return

    dnb_core_somascan = pd.read_csv(soma_core_path)
    dnb_core_olink = pd.read_csv(olink_core_path)

    # Load overlap proteins (requires platform_harmoniser.py — Phase 2)
    overlap_path = Path(config["paths"]["platform_protein_overlap"])
    if overlap_path.exists():
        overlap_df = pd.read_csv(overlap_path)
        overlap_proteins = overlap_df["UniProt"].tolist()
    else:
        logger.warning(
            "Platform overlap file not found at %s — "
            "using all proteins with UniProt annotations as overlap",
            overlap_path,
        )
        # Fall back to intersection of UniProt columns if present
        soma_uniprots = set(dnb_core_somascan.get("UniProt", pd.Series()).dropna())
        olink_uniprots = set(dnb_core_olink.get("UniProt", pd.Series()).dropna())
        overlap_proteins = list(soma_uniprots | olink_uniprots)

    if not overlap_proteins:
        logger.warning("No overlap proteins available — skipping cross-platform validation")
        return

    # Positive control concordance check
    control_proteins = config["cross_platform"]["platform_comparison_biomarkers"]
    try:
        positive_control_concordance(
            dnb_core_somascan, dnb_core_olink, control_proteins, config
        )
    except ValueError:
        logger.error("Positive control concordance check failed — halting cross-platform stage")
        return

    # Build Golden Set
    golden_set = compute_golden_set(
        dnb_core_somascan, dnb_core_olink, overlap_proteins, config
    )

    # Summary statistics
    compute_golden_set_statistics(golden_set, config)

    # Run GSEA on Golden Set if any proteins found
    if golden_set["is_golden_set"].any():
        gsea_script = Path("R/gsea_analysis.R")
        if gsea_script.exists():
            logger.info("Running GSEA on Golden Set via R...")
            result = subprocess.run(
                ["Rscript", str(gsea_script), "--golden-set"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                logger.warning("R GSEA on Golden Set failed: %s", result.stderr)

    logger.info("Cross-platform validation complete")


def run_csd_analysis(config: dict) -> None:
    """Stage 6: CSD rolling window analysis on ADNI data (SECONDARY)."""
    logger.info("=== Stage 6: CSD Analysis — ADNI (SECONDARY) ===")
    import pandas as pd

    from src.csd.composite_score import compute_composite_csd_score, temporal_specificity_analysis
    from src.csd.rolling_window import (
        compute_csd_all_proteins,
        compute_group_csd_statistics,
        run_csd_sensitivity_analysis,
    )
    from src.csd.surrogate_testing import run_surrogate_validation
    from src.preprocessing.somascan_qc import identify_seq_columns

    input_path = Path(config["paths"]["adni_clean_parquet"])
    if not input_path.exists():
        raise FileNotFoundError(f"Preprocessed ADNI data not found at {input_path}")

    df = pd.read_parquet(input_path)
    seq_cols = identify_seq_columns(df)
    results_dir = Path(config["paths"]["results_csd"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Rolling window CSD
    csd_results = compute_csd_all_proteins(df, seq_cols, config)
    csd_results.to_csv(results_dir / "csd_all_proteins.csv", index=False)

    if csd_results.empty:
        logger.warning(
            "CSD produced 0 results — no participants met the minimum visit requirement "
            "(%d visits). This dataset appears to be cross-sectional. "
            "CSD analysis requires longitudinal data with >= %d visits per participant. "
            "Skipping all downstream CSD steps.",
            config["csd"]["primary_window"],
            config["csd"]["primary_window"],
        )
        # Write empty placeholder files so downstream stages don't crash on missing files
        pd.DataFrame(columns=["protein", "var_fdr_p"]).to_csv(
            results_dir / "group_csd_statistics.csv", index=False
        )
        pd.DataFrame(columns=["RID", "composite_csd_score", "TRAJECTORY"]).to_csv(
            results_dir / "composite_csd_scores.csv", index=False
        )
        pd.DataFrame().to_csv(results_dir / "temporal_specificity.csv", index=False)
        pd.DataFrame().to_csv(results_dir / "sensitivity_summary.csv", index=False)
        logger.info("CSD analysis complete (no longitudinal data)")
        return

    # Group statistics
    trajectory_map = df.groupby("RID")["TRAJECTORY"].first()
    # reset_index() names the RID column by the index name; rename to "RID" explicitly
    trajectory_reset = trajectory_map.rename("TRAJECTORY").reset_index()
    trajectory_reset.columns = ["RID", "TRAJECTORY"]
    group_stats = compute_group_csd_statistics(csd_results, trajectory_map, config)
    group_stats.to_csv(results_dir / "group_csd_statistics.csv", index=False)

    # Surrogate validation for significant proteins
    sig_proteins = group_stats[group_stats["var_fdr_p"] < config["csd"]["alpha"]]["protein"].tolist()
    if sig_proteins:
        surrogate_results = run_surrogate_validation(df, seq_cols, sig_proteins, config)
        surrogate_results.to_csv(results_dir / "surrogate_validation.csv", index=False)

    # Composite scores
    composite = compute_composite_csd_score(csd_results, method="mean_tau")
    composite = composite.merge(trajectory_reset, on="RID", how="left")
    composite.to_csv(results_dir / "composite_csd_scores.csv", index=False)

    # Temporal specificity
    temporal = temporal_specificity_analysis(df, composite, config)
    temporal.to_csv(results_dir / "temporal_specificity.csv", index=False)

    # Sensitivity analysis across window sizes and detrending methods
    logger.info("Running CSD sensitivity analysis...")
    sensitivity_summary = run_csd_sensitivity_analysis(df, seq_cols, config)
    sensitivity_summary.to_csv(results_dir / "sensitivity_summary.csv", index=False)

    # Annotate CSD results with Golden Set membership if available
    golden_set_path = Path(config["paths"]["results_cross_platform"]) / "golden_set_proteins.csv"
    if golden_set_path.exists():
        from src.cross_platform.golden_set import add_csd_evidence_to_golden_set

        golden_set = pd.read_csv(golden_set_path)
        golden_set = add_csd_evidence_to_golden_set(golden_set, group_stats)
        golden_set.to_csv(golden_set_path, index=False)
        logger.info("Golden Set updated with CSD evidence (Tier 1/Tier 2)")

    # Cross-validate with R earlywarnings package
    earlywarnings_script = Path("R/earlywarnings_analysis.R")
    if earlywarnings_script.exists():
        logger.info("Running R earlywarnings cross-validation...")
        try:
            result = subprocess.run(
                ["Rscript", str(earlywarnings_script)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                logger.warning("R earlywarnings script failed: %s", result.stderr)
            else:
                logger.info("R earlywarnings cross-validation complete")
        except FileNotFoundError:
            logger.warning("Rscript not found — skipping R earlywarnings cross-validation")
    else:
        logger.info("R/earlywarnings_analysis.R not found, skipping cross-validation")

    logger.info("CSD analysis complete")


def run_validation(config: dict) -> None:
    """Stage 7: Biomarker comparison and ROC analysis."""
    logger.info("=== Stage 7: Validation (ADNI) ===")
    import pandas as pd

    from src.validation.biomarker_comparison import (
        biomarker_negative_subgroup_analysis,
        compute_biomarker_correlations,
        incremental_prediction_analysis,
    )
    from src.validation.roc_analysis import compute_roc_curves

    input_path = Path(config["paths"]["adni_clean_parquet"])
    df = pd.read_parquet(input_path)
    results_dir = Path(config["paths"]["results_validation"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load CSD and sDNB scores (CSD stage removed — cross-sectional data only)
    _csd_path = Path(config["paths"]["results_csd"]) / "composite_csd_scores.csv"
    csd_scores = pd.read_csv(_csd_path) if _csd_path.exists() else pd.DataFrame(columns=["RID"])

    # Load SomaScan sDNB scores
    sdnb_path = Path(config["paths"]["results_dnb_somascan"]) / "wgcna" / "sdnb_scores_wgcna.csv"
    if not sdnb_path.exists():
        sdnb_path = Path(config["paths"]["results_dnb_somascan"]) / "sdnb_scores.csv"
    if not sdnb_path.exists():
        sdnb_path = Path(config["paths"]["results_dnb"]) / "sdnb_scores.csv"
    if sdnb_path.exists():
        sdnb_scores = pd.read_csv(sdnb_path)[["RID", "sdnb_score"]]
        if csd_scores.empty:
            # CSD produced no results (cross-sectional data) — use sDNB scores alone
            csd_scores = sdnb_scores
        else:
            csd_scores = csd_scores.merge(sdnb_scores, on="RID", how="left")

    # Load Golden Set DNB score if available
    golden_set_path = Path(config["paths"]["results_cross_platform"]) / "golden_set_proteins.csv"
    if golden_set_path.exists():
        logger.info("Golden Set results found — including in validation")

    # Biomarker correlations — include clinical surrogates available in ADNIMERGE
    # so the heatmap is non-empty even before plasma biomarker files are downloaded.
    biomarker_cols = ["PTAU181", "PTAU217", "NFL", "GFAP", "ABETA42_40_RATIO"]
    clinical_surrogates = ["CDRSB", "AMYLOID_STATUS", "APOE4"]
    biomarker_cols = biomarker_cols + [c for c in clinical_surrogates if c in df.columns]
    corr_matrix, p_matrix = compute_biomarker_correlations(df, csd_scores, biomarker_cols)
    corr_matrix.to_csv(results_dir / "biomarker_correlations.csv")
    p_matrix.to_csv(results_dir / "biomarker_correlations_pvalues.csv")

    # Create binary outcome column
    converter_label = config["adni"]["converter_group"]
    stable_label = config["adni"]["stable_group"]
    df["IS_CONVERTER"] = (df["TRAJECTORY"] == converter_label).astype(int)
    analysis_df = df[df["TRAJECTORY"].isin([converter_label, stable_label])]

    # Incremental prediction
    incremental = incremental_prediction_analysis(
        analysis_df, csd_scores, biomarker_cols, "IS_CONVERTER", config
    )
    pd.DataFrame([incremental]).to_csv(results_dir / "incremental_prediction.csv", index=False)

    # ROC curves
    predictor_cols = ["composite_csd_score", "sdnb_score"] + biomarker_cols
    merged_for_roc = analysis_df.merge(csd_scores, on="RID", how="inner")

    # Exclude post-diagnosis samples (negative MONTHS_TO_CONVERSION)
    if "MONTHS_TO_CONVERSION" in merged_for_roc.columns:
        post_dx = merged_for_roc["MONTHS_TO_CONVERSION"].fillna(0) < 0
        n_excluded = post_dx.sum()
        if n_excluded > 0:
            logger.info(
                "Excluding %d post-diagnosis samples (negative MONTHS_TO_CONVERSION)",
                n_excluded,
            )
            merged_for_roc = merged_for_roc[~post_dx]

    roc_df, delong_df = compute_roc_curves(
        merged_for_roc, predictor_cols, "IS_CONVERTER",
        config["validation"]["time_horizons_months"],
        config["validation"]["bootstrap_n"],
        config["random_seed"],
    )
    roc_df.to_csv(results_dir / "roc_results.csv", index=False)
    delong_df.to_csv(results_dir / "delong_comparisons.csv", index=False)

    logger.info("Validation complete")


def run_ppmi_replication(config: dict) -> None:
    """Stage 8: Replicate WGCNA DNB analysis on PPMI data."""
    logger.info("=== Stage 8: PPMI Replication ===")
    import pandas as pd

    from src.dnb.wgcna_dnb import run_wgcna_dnb_analysis
    from src.preprocessing.somascan_qc import identify_seq_columns, impute_missing_values

    input_path = Path(config["paths"]["ppmi_clean_parquet"])
    if not input_path.exists():
        logger.warning("PPMI preprocessed data not found, skipping replication")
        return

    df = pd.read_parquet(input_path)
    seq_cols = identify_seq_columns(df)
    results_dir = Path(config["paths"]["results_ppmi"])
    results_dir.mkdir(parents=True, exist_ok=True)

    df_imputed = impute_missing_values(df, seq_cols, config["proteomics"]["imputation_method"])

    # Override reference group for PPMI context
    ppmi_dnb_config = config.copy()
    ppmi_dnb_config["dnb"] = config["dnb"].copy()
    ppmi_dnb_config["dnb"]["reference_group"] = config["ppmi"]["slow_progressor_group"]
    ppmi_dnb_config["adni"] = config["ppmi"].copy()
    ppmi_dnb_config["adni"]["converter_group"] = config["ppmi"]["fast_progressor_group"]
    ppmi_dnb_config["paths"] = config["paths"].copy()
    ppmi_dnb_config["paths"]["results_dnb_somascan"] = str(
        Path(config["paths"]["results_ppmi"]) / "somascan"
    )

    # WGCNA DNB analysis for PPMI
    ppmi_wgcna_dir = "data/results/ppmi/wgcna"
    if Path(ppmi_wgcna_dir).exists():
        logger.info("--- PPMI DNB via WGCNA ---")
        run_wgcna_dnb_analysis(
            df_imputed, seq_cols, ppmi_dnb_config, "somascan_ppmi", wgcna_results_dir=ppmi_wgcna_dir
        )
    else:
        logger.warning("PPMI WGCNA modules not found at %s — run wgcna_ppmi stage first", ppmi_wgcna_dir)

    logger.info("PPMI replication complete")


def run_figures(config: dict) -> None:
    """Stage 9: Generate all publication figures."""
    logger.info("=== Stage 9: Figure Generation ===")
    from src.visualization.figures import generate_all_figures

    generate_all_figures(config)
    logger.info("Figure generation complete")


# ---- Stage registry ----
# Stage order: preprocess → WGCNA → WGCNA DNB → NetMedPy → cross-platform → CSD → validation → PPMI → figures
STAGES = OrderedDict(
    [
        ("env_check", (check_environment, None)),
        ("adni_preprocess", (run_adni_preprocessing, "adni_clean_parquet")),
        ("ppmi_preprocess", (run_ppmi_preprocessing, "ppmi_clean_parquet")),
        ("wgcna", (run_wgcna, None)),
        ("dnb_wgcna_somascan", (run_dnb_wgcna_somascan, None)),
        ("netmedpy", (run_netmedpy, None)),
        ("cross_platform", (run_cross_platform, "results_cross_platform")),
        ("csd", (run_csd_analysis, "results_csd")),
        ("validation", (run_validation, "results_validation")),
        ("wgcna_ppmi", (run_wgcna_ppmi, None)),
        ("ppmi_replication", (run_ppmi_replication, "results_ppmi")),
        ("figures", (run_figures, "results_figures")),
    ]
)


def _stage_output_exists(config: dict, output_key: str) -> bool:
    """Check if a stage's primary output already exists."""
    if output_key is None:
        return False

    path_value = config["paths"].get(output_key, "")
    if not path_value:
        return False

    path = Path(path_value)

    if path.suffix == ".parquet":
        return path.exists()
    else:
        # For directory-based outputs, check if any real files exist recursively (ignore .gitkeep)
        return path.exists() and any(
            f for f in path.rglob("*") if f.is_file() and f.name != ".gitkeep"
        )


def main():
    parser = argparse.ArgumentParser(
        description="CriticalNeuroMap full analysis pipeline"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=list(STAGES.keys()),  # includes 'netmedpy'
        default=None,
        help="Run a single stage (default: run all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if outputs exist",
    )
    args = parser.parse_args()

    config = load_config()

    stages_to_run = (
        [(args.stage, STAGES[args.stage])]
        if args.stage
        else list(STAGES.items())
    )

    total_start = time.time()

    for stage_name, (stage_func, output_key) in stages_to_run:
        if not args.force and output_key and _stage_output_exists(config, output_key):
            logger.info("Stage '%s': output exists, skipping (use --force to rerun)", stage_name)
            continue

        start = time.time()
        try:
            stage_func(config)
            elapsed = time.time() - start
            logger.info("Stage '%s' completed in %.1f seconds", stage_name, elapsed)
        except FileNotFoundError as e:
            logger.error("Stage '%s' failed — missing data: %s", stage_name, e)
            if args.stage:
                sys.exit(1)
            logger.info("Continuing with remaining stages...")
        except Exception as e:
            logger.error("Stage '%s' failed: %s", stage_name, e, exc_info=True)
            if args.stage:
                sys.exit(1)
            logger.info("Continuing with remaining stages...")

    total_elapsed = time.time() - total_start
    logger.info("Pipeline complete. Total time: %.1f seconds (%.1f minutes)",
                total_elapsed, total_elapsed / 60)


if __name__ == "__main__":
    main()
