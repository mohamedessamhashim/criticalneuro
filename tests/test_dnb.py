"""Tests for DNB analysis modules.

All tests use synthetic data with known properties.
"""

import numpy as np
import pandas as pd
import pytest

from src.dnb.dnb_computation import compute_dnb_score


@pytest.fixture
def rng():
    return np.random.RandomState(42)


class TestDNBScore:
    def test_dnb_score_formula(self):
        """Verify DNB score with known inputs."""
        # 2 group proteins, perfectly correlated, high variance
        # 2 outside proteins, uncorrelated with group
        rng = np.random.RandomState(42)
        n = 50
        base = rng.normal(0, 5, n)
        X_group = np.column_stack([base + rng.normal(0, 0.1, n), base + rng.normal(0, 0.1, n)])
        X_outside = rng.normal(0, 1, (n, 2))

        score = compute_dnb_score(X_group, X_outside)
        # High within-group correlation, high variance, low outside correlation
        # -> score should be positive and substantial
        assert score > 1.0, f"Expected high DNB score, got {score}"

    def test_single_protein_group(self, rng):
        """Single protein in group should give PCC_D = 1.0."""
        n = 50
        X_group = rng.normal(0, 5, (n, 1))
        X_outside = rng.normal(0, 1, (n, 3))
        score = compute_dnb_score(X_group, X_outside)
        assert score > 0

    def test_empty_group_returns_zero(self):
        """Empty group should return score 0."""
        X_group = np.empty((50, 0))
        X_outside = np.random.randn(50, 5)
        score = compute_dnb_score(X_group, X_outside)
        assert score == 0.0

    def test_no_outside_proteins(self, rng):
        """With no outside proteins, score should still be computed."""
        n = 50
        X_group = rng.normal(0, 5, (n, 3))
        X_outside = np.empty((n, 0))
        score = compute_dnb_score(X_group, X_outside)
        assert score > 0


class TestSDNB:
    def test_sdnb_increases_for_destabilized_group(self, rng):
        """sDNB scores should be higher for destabilized samples."""
        from src.dnb.sdnb import compute_sdnb_score, _compute_reference_correlations

        n_ref = 100
        n_proteins = 20
        n_outside = 50

        # Reference: stable, low variance
        ref_values = rng.normal(0, 1, (n_ref, n_proteins))
        ref_outside = rng.normal(0, 1, (n_ref, n_outside))

        # Precompute reference statistics
        ref_mean = np.nanmean(ref_values, axis=0)
        ref_std = np.nanstd(ref_values, axis=0, ddof=1)
        ref_std = np.where(ref_std > 0, ref_std, 1e-10)
        PCC_D, PCC_O = _compute_reference_correlations(ref_values, ref_outside)

        # Stable participant: close to reference
        stable_values = rng.normal(0, 1, n_proteins)

        # Destabilized participant: high deviation from reference
        destabilized_values = rng.normal(0, 5, n_proteins)

        score_stable = compute_sdnb_score(stable_values, ref_mean, ref_std, PCC_D, PCC_O)
        score_destab = compute_sdnb_score(destabilized_values, ref_mean, ref_std, PCC_D, PCC_O)

        assert score_destab > score_stable, (
            f"Destabilized sDNB ({score_destab:.3f}) should exceed "
            f"stable ({score_stable:.3f})"
        )


class TestDNBOnOlink:
    """Tests for DNB running on Olink NPX columns."""

    def test_dnb_runs_on_olink_columns(self, rng):
        """DNB score should be computable from NPX_* column data."""
        n = 50
        # Simulate Olink NPX data (already log2-transformed)
        base = rng.normal(0, 3, n)
        X_group = np.column_stack([
            base + rng.normal(0, 0.2, n),
            base + rng.normal(0, 0.2, n),
        ])
        X_outside = rng.normal(0, 1, (n, 3))

        score = compute_dnb_score(X_group, X_outside)
        assert score > 0, "DNB score should be positive for correlated Olink-like data"


class TestGoldenSetIsIntersection:
    """Golden Set must be the intersection of SomaScan and Olink DNB core proteins."""

    def test_golden_set_is_intersection(self):
        """Golden Set result should only contain proteins found on both platforms."""
        from src.cross_platform.golden_set import compute_golden_set

        import tempfile

        shared_uniprot = "P11111"
        somascan_core = pd.DataFrame({
            "protein": ["seq.100.01", "seq.200.02"],
            "frequency": [0.8, 0.6],
            "n_participants": [16, 12],
            "UniProt": [shared_uniprot, "P22222"],
            "platform": ["somascan", "somascan"],
        })
        olink_core = pd.DataFrame({
            "protein": ["NPX_GFAP", "NPX_CLU"],
            "frequency": [0.7, 0.5],
            "n_participants": [14, 10],
            "UniProt": [shared_uniprot, "P33333"],
            "platform": ["olink", "olink"],
        })
        overlap_proteins = [shared_uniprot, "P22222", "P33333"]

        config = {
            "cross_platform": {"golden_set_fdr_threshold": 0.05},
            "paths": {"results_cross_platform": tempfile.mkdtemp()},
        }

        result = compute_golden_set(somascan_core, olink_core, overlap_proteins, config)

        # Only the shared UniProt (P11111) should appear in merged results
        # (P22222 is SomaScan-only, P33333 is Olink-only — neither should be in result)
        assert len(result) == 1
        assert result["UniProt"].iloc[0] == shared_uniprot


class TestPositiveControlsConcordant:
    """Positive controls must pass concordance check."""

    def test_positive_controls_concordant(self):
        """Known biomarkers present on both platforms should be concordant."""
        from src.cross_platform.platform_concordance import positive_control_concordance

        somascan_core = pd.DataFrame({
            "protein": ["seq.GFAP.01", "seq.NEFL.02"],
            "frequency": [0.7, 0.6],
        })
        olink_core = pd.DataFrame({
            "protein": ["NPX_GFAP", "NPX_NEFL"],
            "frequency": [0.65, 0.55],
        })

        result = positive_control_concordance(
            somascan_core, olink_core, ["GFAP", "NEFL"], {}
        )

        for protein in ["GFAP", "NEFL"]:
            assert result[protein]["status"] == "concordant"
