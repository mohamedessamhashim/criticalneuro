#!/usr/bin/env python3
"""
CriticalNeuroMap — Main Pipeline Entry Point
============================================
Usage:
    python run_pipeline.py                          # uses config.yaml
    python run_pipeline.py --config my_config.yaml  # custom config
    python run_pipeline.py --stage wgcna            # run single stage
    python run_pipeline.py --resume                 # resume from last checkpoint

For Muhammad (Knight-ADRC):
    1. Edit config.yaml: set input.expression_matrix and input.metadata paths
    2. Answer the 3 data provenance questions (see data/README_data.md)
    3. Run: conda activate criticalneuro && python run_pipeline.py
    4. Results appear in results/ directory
"""

import argparse
import logging
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import yaml

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="CriticalNeuroMap analysis pipeline (WGCNA → BioTIP → l-DNB)"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--stage", type=str, default=None,
        help="Run a single stage (default: run all)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint (skip completed stages)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force rerun even if checkpoints exist",
    )
    args = parser.parse_args()

    # Load and validate config
    from pipeline.config_loader import load_config, set_all_seeds, validate_input_value_range
    from pipeline.logger import setup_logging

    cfg = load_config(args.config)
    logger = setup_logging(
        log_file=cfg.get("logging", {}).get("log_file", "results/pipeline.log"),
        level=cfg.get("logging", {}).get("level", "INFO"),
    )

    logger.info("=" * 60)
    logger.info("CriticalNeuroMap Pipeline")
    logger.info("Config: %s", args.config)
    logger.info("Mode: %s | Cohort: %s", cfg["analysis_mode"], cfg["cohort"])
    logger.info("=" * 60)

    # Set all random seeds
    set_all_seeds(cfg)

    # ── Data provenance routing ──────────────────────────────────────────
    prov = cfg["data_provenance"]

    if prov["cruchaga_qc_already_applied"]:
        logger.info(
            "DATA PROVENANCE: Cruchaga Lab QC already applied. "
            "Skipping Stage 1 (QC) and Stage 2 (normalization). "
            "Starting at Stage 3 (residualization)."
        )
        logger.warning(
            "ASSUMPTION: Expression matrix is post-ANML, post-Cruchaga-QC. "
            "If this is wrong, set cruchaga_qc_already_applied: false in config.yaml."
        )
        run_stage1 = False
        run_stage2_qc = False
        run_stage2_normalization = False
        run_stage2_log2 = not prov["log2_already_applied"]
        run_stage3_combat = not prov["combat_already_applied"]
    else:
        logger.info(
            "DATA PROVENANCE: Raw SomaScan data. Running full QC and normalization."
        )
        run_stage1 = True
        run_stage2_qc = True
        run_stage2_normalization = True
        run_stage2_log2 = True
        run_stage3_combat = True

    # Validate input value ranges if files exist
    if cfg["analysis_mode"] == "longitudinal":
        expr_path = cfg["input"]["expression_matrix"]
        if Path(expr_path).exists():
            validate_input_value_range(expr_path, prov["log2_already_applied"])

    # ── Setup directories ────────────────────────────────────────────────
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(cfg["output"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Define stages ────────────────────────────────────────────────────
    if cfg["analysis_mode"] == "longitudinal":
        stages = _build_longitudinal_stages(
            cfg, args.config,
            run_stage1, run_stage2_log2, run_stage3_combat,
        )
    else:
        stages = _build_cross_sectional_stages(cfg)

    # Filter to single stage if requested
    if args.stage:
        stage_names = [s[0] for s in stages]
        if args.stage not in stage_names:
            logger.error("Unknown stage '%s'. Available: %s", args.stage, stage_names)
            sys.exit(1)
        stages = [(n, f) for n, f in stages if n == args.stage]

    # ── Execute stages ───────────────────────────────────────────────────
    total_start = time.time()

    for stage_name, stage_func in stages:
        checkpoint_file = checkpoint_dir / f"{stage_name}.done"

        if args.resume and not args.force and checkpoint_file.exists():
            logger.info("Stage '%s': checkpoint exists, skipping (use --force to rerun)", stage_name)
            continue

        logger.info("=" * 40)
        logger.info("Stage: %s", stage_name)
        logger.info("=" * 40)

        start = time.time()
        try:
            stage_func()
            elapsed = time.time() - start

            # Write checkpoint
            checkpoint_file.write_text(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
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
    logger.info("=" * 60)
    logger.info("Pipeline complete. Total time: %.1f seconds (%.1f minutes)",
                total_elapsed, total_elapsed / 60)
    logger.info("Results in: %s", output_dir)


def _build_longitudinal_stages(cfg, config_path, run_stage1, run_log2, run_combat):
    """Build stage list for longitudinal Knight-ADRC analysis."""
    output_dir = Path(cfg["output"]["dir"])
    figures_dir = cfg["output"]["figures_dir"]

    stages = []

    # Stage 1: Load and QC
    if run_stage1:
        def _stage1():
            from pipeline.stage1_qc import run_qc_stage
            run_qc_stage(cfg)
        stages.append(("load_and_qc", _stage1))
    else:
        def _stage1_load_only():
            from pipeline.stage1_qc import load_knight_adrc_data
            import numpy as np
            expr, metadata = load_knight_adrc_data(cfg)
            seq_cols = [c for c in expr.columns if c.startswith("seq.")]
            if not seq_cols:
                seq_cols = expr.select_dtypes(include=[np.number]).columns.tolist()
            output_dir.mkdir(parents=True, exist_ok=True)
            expr.to_csv(output_dir / "expression_qc.csv")
            metadata.to_csv(output_dir / "metadata_staged.csv", index=False)
        stages.append(("load_data", _stage1_load_only))

    # Stage 2: Normalization (log2 only if needed)
    if run_log2:
        def _stage2():
            import pandas as pd
            from pipeline.stage2_normalization import run_normalization_stage
            expr = pd.read_csv(output_dir / "expression_qc.csv", index_col=0)
            run_normalization_stage(cfg, expr)
        stages.append(("normalize", _stage2))

    # Stage 3a: Residualization (lme4 mixed models)
    def _stage3a():
        input_file = output_dir / "expression_normalized.csv"
        if not input_file.exists():
            input_file = output_dir / "expression_qc.csv"
        _run_r_script(
            "pipeline/stage3_residualization.R",
            config=config_path,
            input=str(input_file),
            metadata=str(output_dir / "metadata_staged.csv"),
            output=str(output_dir),
        )
    stages.append(("residualize", _stage3a))

    # Stage 3b: Batch correction (ComBat)
    if run_combat:
        def _stage3b():
            import pandas as pd
            from pipeline.stage3_batch_correction import run_batch_correction_stage
            expr = pd.read_csv(output_dir / "expression_residualized.csv", index_col=0)
            metadata = pd.read_csv(output_dir / "metadata_staged.csv")
            run_batch_correction_stage(cfg, expr, metadata)
        stages.append(("batch_correct", _stage3b))

    # Stage 4a: WGCNA
    def _stage4a():
        # Use the most processed expression file available
        for fname in ["expression_batch_corrected.csv", "expression_residualized.csv",
                      "expression_normalized.csv", "expression_qc.csv"]:
            input_file = output_dir / fname
            if input_file.exists():
                break
        wgcna_output = output_dir / "wgcna"
        _run_r_script(
            "pipeline/stage4_wgcna.R",
            config=config_path,
            input=str(input_file),
            output=str(wgcna_output),
        )
    stages.append(("wgcna", _stage4a))

    # Stage 4b: BioTIP
    def _stage4b():
        for fname in ["expression_batch_corrected.csv", "expression_residualized.csv",
                      "expression_normalized.csv", "expression_qc.csv"]:
            input_file = output_dir / fname
            if input_file.exists():
                break
        biotip_output = output_dir / "biotip"
        _run_r_script(
            "pipeline/stage4_biotip.R",
            config=config_path,
            expr=str(input_file),
            metadata=str(output_dir / "metadata_staged.csv"),
            modules=str(output_dir / "wgcna" / "wgcna_module_assignments.csv"),
            output=str(biotip_output),
        )
    stages.append(("biotip", _stage4b))

    # Stage 4c: l-DNB
    def _stage4c():
        for fname in ["expression_batch_corrected.csv", "expression_residualized.csv",
                      "expression_normalized.csv", "expression_qc.csv"]:
            input_file = output_dir / fname
            if input_file.exists():
                break
        ldnb_output = output_dir / "ldnb"
        _run_r_script(
            "pipeline/stage4_ldnb.R",
            config=config_path,
            expr=str(input_file),
            metadata=str(output_dir / "metadata_staged.csv"),
            cts=str(output_dir / "biotip" / "biotip_cts_proteins.csv"),
            output=str(ldnb_output),
        )
    stages.append(("ldnb", _stage4c))

    # Stage 5a: Network medicine
    def _stage5a():
        from pipeline.stage5_network_medicine import run_network_medicine_stage
        run_network_medicine_stage(cfg)
    stages.append(("network_medicine", _stage5a))

    # Stage 5b: Validation
    def _stage5b():
        from pipeline.stage5_validation import run_validation_stage
        run_validation_stage(cfg)
    stages.append(("validation", _stage5b))

    # Stage 6: Figures
    def _stage6():
        _run_r_script(
            "pipeline/stage6_figures.R",
            config=config_path,
            results_dir=str(output_dir),
            output=str(figures_dir),
        )
    stages.append(("figures", _stage6))

    return stages


def _build_cross_sectional_stages(cfg):
    """Build stage list for cross-sectional ADNI/PPMI analysis."""
    def _run_legacy():
        logger = logging.getLogger("pipeline")
        logger.info(
            "Cross-sectional mode: delegating to pipelines/run_full_pipeline.py. "
            "Run: python pipelines/run_full_pipeline.py"
        )
        result = subprocess.run(
            [sys.executable, "pipelines/run_full_pipeline.py"],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("Legacy pipeline failed")

    return [("cross_sectional_pipeline", _run_legacy)]


def _run_r_script(script_path: str, **kwargs) -> None:
    """Run an R script via subprocess with named arguments.

    Parameters
    ----------
    script_path : str
        Path to R script relative to project root.
    **kwargs
        Named arguments passed as --name value pairs.
    """
    logger = logging.getLogger("pipeline")

    script = Path(script_path)
    if not script.exists():
        raise FileNotFoundError(f"R script not found: {script}")

    cmd = ["Rscript", str(script)]
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Log stdout
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                logger.info("[R] %s", line.strip())

    if result.returncode != 0:
        logger.error("R script failed:\n%s", result.stderr)
        raise RuntimeError(f"R script {script_path} failed (exit code {result.returncode})")


if __name__ == "__main__":
    main()
