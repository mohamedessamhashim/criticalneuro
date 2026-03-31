"""Tests for cross-platform validation module.

All tests use synthetic data — no real ADNI or Olink data required.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestGoldenSetEmptyWhenNoOverlap:
    """Golden Set should be empty with warning when platforms share no proteins."""

    def test_no_shared_proteins_returns_empty(self):
        from src.cross_platform.golden_set import compute_golden_set

        somascan_core = pd.DataFrame({
            "protein": ["seq.1000.01", "seq.1001.02"],
            "frequency": [0.5, 0.4],
            "n_participants": [10, 8],
            "UniProt": ["P01234", "P01235"],
            "platform": ["somascan", "somascan"],
        })
        olink_core = pd.DataFrame({
            "protein": ["NPX_PROTX", "NPX_PROTY"],
            "frequency": [0.6, 0.3],
            "n_participants": [12, 6],
            "UniProt": ["P09999", "P09998"],
            "platform": ["olink", "olink"],
        })
        overlap_proteins = ["P01234", "P01235", "P09999", "P09998"]

        config = {
            "cross_platform": {"golden_set_fdr_threshold": 0.05},
            "paths": {"results_cross_platform": tempfile.mkdtemp()},
        }

        result = compute_golden_set(somascan_core, olink_core, overlap_proteins, config)
        assert len(result) == 0 or not result["is_golden_set"].any()

    def test_empty_input_returns_empty(self):
        from src.cross_platform.golden_set import compute_golden_set

        empty = pd.DataFrame(columns=["protein", "frequency", "n_participants", "UniProt", "platform"])
        config = {
            "cross_platform": {"golden_set_fdr_threshold": 0.05},
            "paths": {"results_cross_platform": tempfile.mkdtemp()},
        }

        result = compute_golden_set(empty, empty, [], config)
        assert result.empty

    def test_shared_proteins_produce_golden_set(self):
        from src.cross_platform.golden_set import compute_golden_set

        shared_uniprot = "P12345"
        somascan_core = pd.DataFrame({
            "protein": ["seq.1000.01"],
            "frequency": [0.8],
            "n_participants": [16],
            "UniProt": [shared_uniprot],
            "platform": ["somascan"],
        })
        olink_core = pd.DataFrame({
            "protein": ["NPX_GFAP"],
            "frequency": [0.7],
            "n_participants": [14],
            "UniProt": [shared_uniprot],
            "platform": ["olink"],
        })
        overlap_proteins = [shared_uniprot]

        config = {
            "cross_platform": {"golden_set_fdr_threshold": 0.05},
            "paths": {"results_cross_platform": tempfile.mkdtemp()},
        }

        result = compute_golden_set(somascan_core, olink_core, overlap_proteins, config)
        assert len(result) == 1
        assert result["UniProt"].iloc[0] == shared_uniprot


class TestConcordanceScoreBounds:
    """Spearman concordance values must be valid correlations [-1, 1]."""

    def test_concordance_in_valid_range(self):
        from src.cross_platform.platform_concordance import compute_platform_concordance

        np.random.seed(42)
        n_participants = 20
        n_proteins = 5

        rids = list(range(n_participants))
        rows_soma = []
        rows_olink = []

        for uniprot_id in [f"P{i:05d}" for i in range(n_proteins)]:
            for rid in rids:
                freq = np.random.uniform(0, 1)
                rows_soma.append({
                    "RID": rid,
                    "UniProt": uniprot_id,
                    "dnb_frequency": freq + np.random.normal(0, 0.1),
                })
                rows_olink.append({
                    "RID": rid,
                    "UniProt": uniprot_id,
                    "dnb_frequency": freq + np.random.normal(0, 0.1),
                })

        somascan_scores = pd.DataFrame(rows_soma)
        olink_scores = pd.DataFrame(rows_olink)
        overlap = [f"P{i:05d}" for i in range(n_proteins)]

        result = compute_platform_concordance(somascan_scores, olink_scores, overlap)

        assert len(result) == n_proteins
        assert (result["spearman_rho"] >= -1.0).all()
        assert (result["spearman_rho"] <= 1.0).all()

    def test_empty_overlap_returns_empty(self):
        from src.cross_platform.platform_concordance import compute_platform_concordance

        somascan = pd.DataFrame(columns=["RID", "UniProt", "dnb_frequency"])
        olink = pd.DataFrame(columns=["RID", "UniProt", "dnb_frequency"])

        result = compute_platform_concordance(somascan, olink, [])
        assert result.empty


class TestConcordanceReportWritten:
    """Concordance report file must be created after generate_concordance_report."""

    def test_report_file_created(self):
        from src.cross_platform.platform_concordance import generate_concordance_report

        concordance_df = pd.DataFrame({
            "UniProt": ["P00001", "P00002", "P00003"],
            "spearman_rho": [0.85, 0.42, -0.1],
            "spearman_p": [0.001, 0.05, 0.8],
            "n_participants": [50, 45, 40],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "paths": {"results_cross_platform": tmpdir},
            }
            generate_concordance_report(concordance_df, config)

            report_path = Path(tmpdir) / "concordance_report.txt"
            assert report_path.exists()

            content = report_path.read_text()
            assert "Cross-Platform Concordance Report" in content
            assert "Median Spearman rho" in content
            assert "P00001" in content

    def test_empty_concordance_produces_minimal_report(self):
        from src.cross_platform.platform_concordance import generate_concordance_report

        empty_df = pd.DataFrame(columns=[
            "UniProt", "spearman_rho", "spearman_p", "n_participants"
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "paths": {"results_cross_platform": tmpdir},
            }
            generate_concordance_report(empty_df, config)

            report_path = Path(tmpdir) / "concordance_report.txt"
            assert report_path.exists()


class TestGoldenSetCSDTiers:
    """Golden Set proteins with CSD support should be flagged Tier 1."""

    def test_tier_assignment(self):
        from src.cross_platform.golden_set import add_csd_evidence_to_golden_set

        golden_set = pd.DataFrame({
            "UniProt": ["P001", "P002", "P003"],
            "protein_somascan": ["seq.100.01", "seq.200.02", "seq.300.03"],
            "protein_olink": ["NPX_A", "NPX_B", "NPX_C"],
            "frequency_somascan": [0.8, 0.6, 0.5],
            "frequency_olink": [0.7, 0.5, 0.4],
            "is_golden_set": [True, True, False],
        })

        csd_results = pd.DataFrame({
            "protein": ["seq.100.01", "seq.200.02", "seq.300.03", "seq.400.04"],
            "var_tau": [0.45, 0.10, 0.30, 0.55],
            "var_fdr_p": [0.001, 0.50, 0.02, 0.001],
        })

        result = add_csd_evidence_to_golden_set(golden_set, csd_results)

        assert "tier" in result.columns
        # P001: golden_set=True, csd_fdr_p=0.001 < 0.05 → Tier_1
        assert result.loc[result["UniProt"] == "P001", "tier"].iloc[0] == "Tier_1"
        # P002: golden_set=True, csd_fdr_p=0.50 > 0.05 → Tier_2
        assert result.loc[result["UniProt"] == "P002", "tier"].iloc[0] == "Tier_2"
        # P003: golden_set=False → not_golden_set
        assert result.loc[result["UniProt"] == "P003", "tier"].iloc[0] == "not_golden_set"


class TestPositiveControlConcordance:
    """Positive control check should pass when controls are on both platforms."""

    def test_concordant_controls_pass(self):
        from src.cross_platform.platform_concordance import positive_control_concordance

        somascan_core = pd.DataFrame({
            "protein": ["seq.GFAP.01", "seq.NEFL.02"],
            "frequency": [0.7, 0.6],
        })
        olink_core = pd.DataFrame({
            "protein": ["NPX_GFAP", "NPX_NEFL"],
            "frequency": [0.65, 0.55],
        })

        config = {}
        result = positive_control_concordance(
            somascan_core, olink_core, ["GFAP", "NEFL"], config
        )

        assert result["GFAP"]["status"] == "concordant"
        assert result["NEFL"]["status"] == "concordant"

    def test_discordant_controls_raise_error(self):
        from src.cross_platform.platform_concordance import positive_control_concordance

        somascan_core = pd.DataFrame({
            "protein": ["seq.GFAP.01"],
            "frequency": [0.7],
        })
        # GFAP missing from Olink
        olink_core = pd.DataFrame({
            "protein": ["NPX_NEFL"],
            "frequency": [0.55],
        })

        config = {}
        with pytest.raises(ValueError, match="concordance FAILED"):
            positive_control_concordance(
                somascan_core, olink_core, ["GFAP"], config
            )
