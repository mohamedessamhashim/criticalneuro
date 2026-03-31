"""Tests for DNB analysis modules.

All tests use synthetic data with known properties.
"""

import numpy as np
import pandas as pd
import pytest

from src.dnb.dnb_computation import (
    compute_dnb_score,
    identify_dnb_core_proteins,
    identify_dnb_group,
)


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


class TestDNBGroup:
    def test_group_has_higher_within_correlation(self, rng):
        """DNB group proteins should have higher pairwise correlation than random."""
        n = 100
        n_proteins = 30

        # Create a block of 5 correlated proteins (simulating DNB group)
        base = rng.normal(0, 3, n)
        correlated_block = np.column_stack(
            [base + rng.normal(0, 0.5, n) for _ in range(5)]
        )
        # Plus 25 uncorrelated proteins
        uncorrelated = rng.normal(0, 1, (n, 25))
        X = np.column_stack([correlated_block, uncorrelated])

        # Reference has low variance
        X_ref = rng.normal(0, 0.5, (n, n_proteins))

        protein_names = [f"p{i}" for i in range(n_proteins)]
        group_names, score = identify_dnb_group(
            X, protein_names, X_ref, top_variance_percentile=30
        )

        if len(group_names) >= 2:
            # Check within-group correlation is higher than random
            group_idx = [protein_names.index(p) for p in group_names]
            X_g = X[:, group_idx]
            corr = np.corrcoef(X_g.T)
            upper = np.triu_indices(len(group_idx), k=1)
            mean_within = np.abs(corr[upper]).mean()

            # Random group of same size
            random_idx = rng.choice(n_proteins, len(group_idx), replace=False)
            X_r = X[:, random_idx]
            corr_r = np.corrcoef(X_r.T)
            upper_r = np.triu_indices(len(random_idx), k=1)
            mean_random = np.abs(corr_r[upper_r]).mean()

            assert mean_within > mean_random, (
                f"Within-group correlation ({mean_within:.3f}) should exceed "
                f"random ({mean_random:.3f})"
            )

    def test_returns_empty_for_insufficient_proteins(self, rng):
        """With fewer than 2 proteins, should return empty group."""
        X = rng.normal(0, 1, (50, 1))
        X_ref = rng.normal(0, 1, (50, 1))
        group, score = identify_dnb_group(X, ["p0"], X_ref, 20)
        assert group == []
        assert score == 0.0


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


class TestCoreProteins:
    def test_threshold_filtering(self):
        """Proteins below threshold fraction should be excluded."""
        dnb_by_participant = {
            1: ["p1", "p2", "p3"],
            2: ["p1", "p2", "p4"],
            3: ["p1", "p5", "p6"],
            4: ["p1", "p2", "p3"],
            5: ["p7", "p8", "p9"],
        }
        # threshold=0.30 means protein must appear in >= 30% of 5 participants = 1.5 -> 2+
        result = identify_dnb_core_proteins(dnb_by_participant, threshold=0.30)

        # p1 appears in 4/5 = 0.80 -> core
        # p2 appears in 3/5 = 0.60 -> core
        # p3 appears in 2/5 = 0.40 -> core
        # p4, p5, p6, p7, p8, p9 appear in 1/5 = 0.20 -> not core
        assert "p1" in result["protein"].values
        assert "p2" in result["protein"].values
        assert "p3" in result["protein"].values
        assert "p7" not in result["protein"].values

    def test_empty_input(self):
        """Empty input should return empty DataFrame."""
        result = identify_dnb_core_proteins({}, threshold=0.30)
        assert len(result) == 0


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
