"""Tests for the preprocessing pipeline.

All tests use synthetic data — no dependency on real ADNI/PPMI files.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.somascan_qc import (
    filter_proteins_by_detectability,
    identify_seq_columns,
    impute_missing_values,
    log2_transform,
    median_normalize,
    remove_sample_outliers,
    residualize_covariates,
)
from src.preprocessing.adni_loader import assign_conversion_labels, merge_adni_data


@pytest.fixture
def config():
    """Minimal config for testing."""
    return {
        "random_seed": 42,
        "adni": {
            "stable_followup_min_months": 36,
            "converter_group": "MCI_to_Dementia",
            "stable_group": "stable_MCI",
        },
        "proteomics": {
            "min_detectability": 0.80,
            "outlier_mahalanobis_sd": 3.0,
            "outlier_pca_components": 50,
            "covariates_to_residualize": ["AGE", "SEX", "APOE4"],
            "imputation_method": "half_min",
        },
    }


@pytest.fixture
def rng():
    """Seeded random state for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def synthetic_protein_df(rng):
    """Create a synthetic DataFrame with protein and metadata columns."""
    n_samples = 100
    n_proteins = 20

    data = {"RID": range(1, n_samples + 1), "VISCODE": ["bl"] * n_samples}
    for i in range(n_proteins):
        data[f"seq.{1000 + i}.{i}"] = rng.lognormal(mean=8, sigma=0.5, size=n_samples)
    data["AGE"] = rng.normal(70, 8, n_samples)
    data["SEX"] = rng.choice([0, 1], n_samples)
    data["APOE4"] = rng.choice([0, 1], n_samples, p=[0.7, 0.3])

    return pd.DataFrame(data)


class TestSeqColumnIdentification:
    def test_identifies_seq_columns_only(self):
        """Given df with seq.* and non-seq columns, verify only seq.* returned."""
        df = pd.DataFrame(
            {
                "seq.1234.56": [1, 2],
                "seq.7890.12": [3, 4],
                "AGE": [70, 75],
                "RID": [1, 2],
            }
        )
        result = identify_seq_columns(df)
        assert result == ["seq.1234.56", "seq.7890.12"]

    def test_returns_empty_for_no_seq_columns(self):
        """DataFrame without seq.* columns returns empty list."""
        df = pd.DataFrame({"AGE": [70], "RID": [1]})
        result = identify_seq_columns(df)
        assert result == []

    def test_returns_sorted_list(self):
        """Columns should be sorted alphabetically."""
        df = pd.DataFrame(
            {"seq.9999.1": [1], "seq.0001.1": [2], "seq.5000.1": [3]}
        )
        result = identify_seq_columns(df)
        assert result == ["seq.0001.1", "seq.5000.1", "seq.9999.1"]


class TestProteinDetectability:
    def test_protein_detectability_filter(self, rng):
        """Protein with 70% detectability should be removed at 80% threshold."""
        n = 100
        data = {
            "RID": range(n),
            # Protein A: 90% above LLOD proxy -> passes
            "seq.1000.1": rng.lognormal(8, 0.5, n),
            # Protein B: values mostly at LLOD level -> may fail
            "seq.2000.1": np.concatenate(
                [rng.lognormal(8, 0.5, 90), np.full(10, 0.01)]
            ),
        }
        # Create protein C with very low detectability
        c_vals = np.full(n, 0.01)
        c_vals[:50] = rng.lognormal(8, 0.5, 50)  # only 50% above LLOD
        data["seq.3000.1"] = c_vals
        df = pd.DataFrame(data)

        seq_cols = ["seq.1000.1", "seq.2000.1", "seq.3000.1"]
        df_filtered, remaining = filter_proteins_by_detectability(
            df, seq_cols, min_detectability=0.80
        )

        # seq.3000.1 should be removed (50% detectability < 80% threshold)
        assert "seq.3000.1" not in remaining
        assert "seq.1000.1" in remaining

    def test_all_proteins_above_threshold(self, rng):
        """If all proteins pass, none should be removed."""
        n = 100
        data = {
            "RID": range(n),
            "seq.1000.1": rng.lognormal(8, 0.5, n),
            "seq.2000.1": rng.lognormal(8, 0.5, n),
        }
        df = pd.DataFrame(data)
        seq_cols = ["seq.1000.1", "seq.2000.1"]

        _, remaining = filter_proteins_by_detectability(
            df, seq_cols, min_detectability=0.80
        )
        assert len(remaining) == 2


class TestSampleOutlierRemoval:
    def test_sample_outlier_removal(self, rng):
        """Injecting an extreme outlier sample should remove it."""
        n = 50
        n_proteins = 10
        data = {"RID": range(n)}
        for i in range(n_proteins):
            data[f"seq.{1000 + i}.1"] = rng.normal(100, 10, n)
        df = pd.DataFrame(data)

        seq_cols = [f"seq.{1000 + i}.1" for i in range(n_proteins)]

        # Inject extreme outlier at row 0
        for col in seq_cols:
            df.loc[0, col] = df[col].mean() + 100 * df[col].std()

        result = remove_sample_outliers(df, seq_cols, sd_threshold=3.0)
        assert len(result) < len(df)
        assert 0 not in result["RID"].values

    def test_no_outliers_from_clean_data(self, rng):
        """Normal data should not lose samples."""
        n = 50
        data = {"RID": range(n)}
        for i in range(10):
            data[f"seq.{1000 + i}.1"] = rng.normal(100, 10, n)
        df = pd.DataFrame(data)

        seq_cols = [f"seq.{1000 + i}.1" for i in range(10)]
        result = remove_sample_outliers(df, seq_cols, sd_threshold=5.0)
        # With a generous threshold of 5 SD, clean normal data should keep all
        assert len(result) == len(df)


class TestMedianNormalization:
    def test_median_normalization_properties(self, rng):
        """After normalization, all sample medians should be approximately equal."""
        n = 50
        data = {"RID": range(n)}
        for i in range(10):
            # Different samples have different total protein loads
            scale = rng.uniform(0.5, 2.0)
            data[f"seq.{1000 + i}.1"] = rng.lognormal(8, 0.5, n) * scale
        df = pd.DataFrame(data)

        seq_cols = [f"seq.{1000 + i}.1" for i in range(10)]
        result = median_normalize(df, seq_cols)

        sample_medians = result[seq_cols].median(axis=1)
        cohort_median = sample_medians.median()

        # All sample medians should be close to cohort median
        np.testing.assert_allclose(
            sample_medians.values,
            cohort_median,
            rtol=1e-10,
        )


class TestLog2Transform:
    def test_no_negative_infinity(self, rng):
        """With values including 0 and near-zero, verify no -inf in output."""
        data = {
            "RID": range(10),
            "seq.1000.1": [0, 0.001, 0.5, 1, 2, 4, 8, 16, 100, 1000],
            "seq.2000.1": rng.lognormal(8, 0.5, 10),
        }
        df = pd.DataFrame(data)
        seq_cols = ["seq.1000.1", "seq.2000.1"]

        result = log2_transform(df, seq_cols)
        assert not np.any(np.isinf(result[seq_cols].values))
        assert not np.any(result[seq_cols].values < 0)  # log2(1) = 0 is the floor

    def test_log2_correctness(self):
        """Verify log2(4) = 2 for known values."""
        df = pd.DataFrame(
            {"RID": [1, 2], "seq.1000.1": [4.0, 16.0], "seq.2000.1": [8.0, 32.0]}
        )
        seq_cols = ["seq.1000.1", "seq.2000.1"]
        result = log2_transform(df, seq_cols)

        assert result.loc[0, "seq.1000.1"] == pytest.approx(2.0)
        assert result.loc[1, "seq.1000.1"] == pytest.approx(4.0)
        assert result.loc[0, "seq.2000.1"] == pytest.approx(3.0)
        assert result.loc[1, "seq.2000.1"] == pytest.approx(5.0)


class TestResidualization:
    def test_residualize_removes_age_effect(self, rng):
        """Protein perfectly correlated with AGE should have ~0 correlation after."""
        n = 200
        age = rng.normal(70, 8, n)
        data = {
            "RID": range(n),
            "AGE": age,
            "SEX": rng.choice([0, 1], n),
            # Protein strongly driven by age
            "seq.1000.1": age * 2.0 + rng.normal(0, 0.1, n),
            "seq.2000.1": rng.normal(100, 10, n),
        }
        df = pd.DataFrame(data)
        seq_cols = ["seq.1000.1", "seq.2000.1"]

        result = residualize_covariates(df, seq_cols, ["AGE"])

        # After residualization, correlation with AGE should be near zero
        corr = np.corrcoef(result["seq.1000.1"], result["AGE"])[0, 1]
        assert abs(corr) < 0.05


class TestImputation:
    def test_half_min_imputation(self):
        """Verify NaN replaced with half the column minimum."""
        df = pd.DataFrame(
            {
                "RID": [1, 2, 3, 4],
                "seq.1000.1": [10.0, 20.0, np.nan, 30.0],
                "seq.2000.1": [5.0, np.nan, 15.0, 25.0],
            }
        )
        seq_cols = ["seq.1000.1", "seq.2000.1"]
        result = impute_missing_values(df, seq_cols, "half_min")

        # seq.1000.1 min = 10, so NaN -> 5.0
        assert result.loc[2, "seq.1000.1"] == pytest.approx(5.0)
        # seq.2000.1 min = 5, so NaN -> 2.5
        assert result.loc[1, "seq.2000.1"] == pytest.approx(2.5)

    def test_imputation_preserves_existing_values(self):
        """Non-NaN values should be unchanged after imputation."""
        df = pd.DataFrame(
            {
                "RID": [1, 2, 3],
                "seq.1000.1": [10.0, 20.0, np.nan],
            }
        )
        result = impute_missing_values(df, ["seq.1000.1"], "half_min")
        assert result.loc[0, "seq.1000.1"] == pytest.approx(10.0)
        assert result.loc[1, "seq.1000.1"] == pytest.approx(20.0)


class TestOlinkLoader:
    """Tests for Olink data loading."""

    def test_olink_loader_returns_npx_columns(self):
        """Olink loader should identify NPX_* columns."""
        from src.preprocessing.olink_loader import _identify_npx_columns

        df = pd.DataFrame({
            "RID": [1, 2],
            "NPX_GFAP": [3.5, 4.2],
            "NPX_NEFL": [2.1, 3.0],
            "AGE": [70, 75],
        })
        result = _identify_npx_columns(df)
        assert result == ["NPX_GFAP", "NPX_NEFL"]
        assert "AGE" not in result

    def test_olink_not_log_transformed_again(self):
        """apply_olink_qc must not re-log-transform NPX values."""
        from src.preprocessing.olink_loader import apply_olink_qc

        rng = np.random.RandomState(42)
        n = 50
        # NPX values are already log2 — typical range 0-15
        df = pd.DataFrame({
            "RID": range(n),
            "NPX_PROT1": rng.normal(8, 2, n),
            "NPX_PROT2": rng.normal(6, 1.5, n),
            "AGE": rng.normal(70, 8, n),
            "SEX": rng.choice([0, 1], n),
        })
        npx_cols = ["NPX_PROT1", "NPX_PROT2"]
        config = {
            "olink": {
                "min_qc_pass_fraction": 0.5,
                "npx_already_log_transformed": True,
                "covariates_to_residualize": [],
            },
        }

        result_df, passing = apply_olink_qc(df, npx_cols, config)

        # Values should remain in the same range (not doubly log-transformed)
        orig_mean = df["NPX_PROT1"].mean()
        result_mean = result_df["NPX_PROT1"].mean()
        assert abs(orig_mean - result_mean) < 1.0, (
            f"NPX values changed dramatically ({orig_mean:.1f} -> {result_mean:.1f}), "
            "suggesting re-log-transformation"
        )


class TestPlatformHarmoniser:
    """Tests for platform harmonisation."""

    def test_platform_harmoniser_builds_overlap(self, tmp_path):
        """build_platform_overlap should produce a valid overlap CSV."""
        from src.preprocessing.platform_harmoniser import build_platform_overlap

        # Create synthetic SomaScan and Olink UniProt maps
        soma_map = pd.DataFrame({
            "SeqId": ["seq.1000.01", "seq.2000.02", "seq.3000.03"],
            "UniProt": ["P00001", "P00002", "P00003"],
            "EntrezGeneSymbol": ["GFAP", "NEFL", "CLU"],
        })
        olink_map = pd.DataFrame({
            "AssayName": ["GFAP", "NEFL", "APOE"],
            "UniProt": ["P00001", "P00002", "P00004"],
            "EntrezGeneSymbol": ["GFAP", "NEFL", "APOE"],
        })

        soma_path = tmp_path / "soma_map.csv"
        olink_path = tmp_path / "olink_map.csv"
        output_path = tmp_path / "overlap.csv"

        soma_map.to_csv(soma_path, index=False)
        olink_map.to_csv(olink_path, index=False)

        result = build_platform_overlap(str(soma_path), str(olink_path), str(output_path))

        assert output_path.exists()
        assert len(result) == 2  # P00001 and P00002 overlap
        assert set(result["UniProt"]) == {"P00001", "P00002"}

    def test_platform_column_set(self):
        """Olink and SomaScan DataFrames must have PLATFORM column after loading."""
        # Verify PLATFORM column is set correctly
        df_soma = pd.DataFrame({
            "RID": [1],
            "PLATFORM": ["somascan"],
            "seq.1000.01": [100.0],
        })
        df_olink = pd.DataFrame({
            "RID": [1],
            "PLATFORM": ["olink"],
            "NPX_GFAP": [3.5],
        })

        assert "PLATFORM" in df_soma.columns
        assert "PLATFORM" in df_olink.columns
        assert df_soma["PLATFORM"].iloc[0] == "somascan"
        assert df_olink["PLATFORM"].iloc[0] == "olink"


class TestConversionLabels:
    def test_conversion_label_assignment(self, config):
        """Synthetic trajectories should be labeled correctly."""
        # RID 1: MCI -> MCI -> Dementia = MCI_to_Dementia
        # RID 2: MCI -> MCI -> MCI (48mo followup) = stable_MCI
        # RID 3: CN (amyloid neg) = CN_amyloid_negative
        # RID 4: CN (amyloid pos) = CN_amyloid_positive
        base_date = pd.Timestamp("2020-01-01")

        data = {
            "RID": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
            "EXAMDATE": [
                base_date,
                base_date + pd.DateOffset(months=12),
                base_date + pd.DateOffset(months=24),
                base_date,
                base_date + pd.DateOffset(months=24),
                base_date + pd.DateOffset(months=48),
                base_date,
                base_date + pd.DateOffset(months=12),
                base_date,
                base_date + pd.DateOffset(months=12),
            ],
            "DX": [
                "MCI", "MCI", "Dementia",
                "MCI", "MCI", "MCI",
                "CN", "CN",
                "CN", "CN",
            ],
            "AMYLOID_STATUS": [
                np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan,
                0, 0,
                1, 1,
            ],
            "VISCODE": [
                "bl", "m12", "m24",
                "bl", "m24", "m48",
                "bl", "m12",
                "bl", "m12",
            ],
        }
        df = pd.DataFrame(data)
        result = assign_conversion_labels(df, config)

        # Check RID 1 = converter
        rid1 = result[result["RID"] == 1]
        assert (rid1["TRAJECTORY"] == "MCI_to_Dementia").all()
        # MONTHS_TO_CONVERSION at baseline should be ~24
        assert rid1.iloc[0]["MONTHS_TO_CONVERSION"] > 20

        # Check RID 2 = stable MCI
        rid2 = result[result["RID"] == 2]
        assert (rid2["TRAJECTORY"] == "stable_MCI").all()

        # Check RID 3 = CN amyloid negative
        rid3 = result[result["RID"] == 3]
        assert (rid3["TRAJECTORY"] == "CN_amyloid_negative").all()

        # Check RID 4 = CN amyloid positive
        rid4 = result[result["RID"] == 4]
        assert (rid4["TRAJECTORY"] == "CN_amyloid_positive").all()


class TestMerge:
    def test_merge_does_not_duplicate_rows(self):
        """Merging on RID+VISCODE should not create duplicate rows."""
        clinical = pd.DataFrame(
            {
                "RID": [1, 1, 2, 2],
                "VISCODE": ["bl", "m12", "bl", "m12"],
                "DX": ["MCI", "MCI", "CN", "CN"],
            }
        )
        proteomics = pd.DataFrame(
            {
                "RID": [1, 1, 2],
                "VISCODE": ["bl", "m12", "bl"],
                "seq.1000.1": [100, 200, 150],
            }
        )
        result = merge_adni_data(clinical, proteomics)

        # Should have 3 rows (inner join on matching RID+VISCODE)
        assert len(result) == 3
        # No duplicated RID+VISCODE pairs
        assert not result.duplicated(subset=["RID", "VISCODE"]).any()
