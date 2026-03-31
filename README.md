# CriticalNeuroMap

**Dynamic Network Biomarker and Critical Slowing Down in Longitudinal Proteomics as Early Warning Signals for Neurodegenerative Disease Transitions**

---

## Background

Complex biological systems approaching a tipping point exhibit *critical slowing down* (CSD): rising variance, increasing autocorrelation, and growing inter-component correlations. The **Dynamic Network Biomarker (DNB)** framework (Chen et al. 2012, *Nature Communications*) operationalises this into a single score — the ratio of within-group protein co-variation to between-group co-variation — that peaks at the pre-transition state, just before a system tips irreversibly.

This repository applies DNB analysis to high-dimensional plasma and CSF SomaScan proteomics to ask: can coordinated protein network destabilisation serve as an early warning signal for neurodegenerative disease transitions? Two independent cohorts are analysed:

- **ADNI** — SomaScan plasma (~6,946 proteins, 332 participants) for the MCI → Alzheimer's dementia transition
- **PPMI** — SomaScan CSF (~4,785 proteins, 1,250 samples) for slow vs. rapid Parkinson's disease progression

A per-participant single-sample DNB (sDNB) score is derived for individual risk stratification, and results are validated against established biomarkers (amyloid PET, APOE4, ABETA42/40 ratio).

---

## Installation

Requires [conda](https://docs.conda.io/en/latest/) and R ≥ 4.4.

```bash
conda env create -f environment.yml
conda activate criticalneuro
```

Verify R packages (installed automatically by conda):

```bash
Rscript -e "library(fgsea); library(sva); library(arrow); cat('R packages OK\n')"
```

---

## Data Access

Data are not included in this repository and must be requested directly from the study portals. Both studies require registration and data use agreement.

### ADNI (Alzheimer's Disease Neuroimaging Initiative)

1. Register at [adni.loni.usc.edu](https://adni.loni.usc.edu) and apply for data access.
2. Download **SomaScan Aptamer Data** (plasma, baseline visit): `ADNI → Download → Study Data → Biospecimen → SomaScan Proteomics`.
3. Download **ADNIMERGE** clinical summary table: `ADNI → Download → Study Data → Study Info → ADNIMERGE`.
4. Place all ADNI files in `data/raw/ADNI/`.
5. Run the R extraction script once to generate the clinical CSV:
   ```bash
   Rscript R/adnimerge_extract.R
   ```

### PPMI (Parkinson's Progression Markers Initiative)

1. Register at [ppmi-info.org](https://www.ppmi-info.org) and apply for data access.
2. Download **SomaScan CSF Proteomics** files: `PPMI → Access Data → Proteomics`.
3. Place all PPMI files in `data/raw/PPMI/`.

### Reference files

The following reference files are bundled in `data/reference/` and require no separate download:

| File | Purpose |
|------|---------|
| `somascan_uniprot_map.csv` | SomaScan aptamer → UniProt ID mapping |
| `msigdb_hallmark.gmt.txt` | MSigDB Hallmark gene sets (GSEA) |
| `Reactome_Pathways.gmt` | Reactome pathway gene sets (GSEA) |

---

## Running the Pipeline

```bash
# Run the full pipeline (all stages)
python pipelines/run_full_pipeline.py

# Run a single stage
python pipelines/run_full_pipeline.py --stage dnb_somascan

# Force re-run a completed stage
python pipelines/run_full_pipeline.py --stage validation --force
```

**Available stages:**

| Stage name | Description |
|-----------|-------------|
| `env_check` | Verify Python and R dependencies |
| `adni_preprocess` | Load, QC, and batch-correct ADNI SomaScan data |
| `ppmi_preprocess` | Load and QC PPMI SomaScan data |
| `dnb_somascan` | DNB analysis on ADNI SomaScan (primary) |
| `dnb_olink` | DNB analysis on ADNI Olink (skipped if no Olink data) |
| `cross_platform` | Golden Set construction (requires both platforms) |
| `validation` | sDNB vs. established biomarkers (ROC, Spearman) |
| `ppmi_replication` | PPMI DNB replication analysis |
| `figures` | Generate all publication figures |

Outputs are written to `data/results/`. See [`docs/pipeline_results.md`](docs/pipeline_results.md) for a full annotated results report.

---

## Tests

```bash
pytest tests/ -v --tb=short
```

---

## Project Structure

```
src/
  preprocessing/   # Data loading, QC, batch correction (ADNI, PPMI, Olink)
  dnb/             # DNB computation, single-sample DNB (sDNB)
  csd/             # Critical Slowing Down (rolling-window; requires longitudinal data)
  validation/      # Biomarker comparison, ROC/AUC analysis
  visualization/   # Publication figure generation
  cross_platform/  # Golden Set construction, platform concordance
R/                 # R scripts: ADNIMERGE extraction, ComBat, GSEA
pipelines/         # End-to-end pipeline orchestrator with checkpointing
notebooks/         # Exploratory Jupyter notebooks (01–07)
tests/             # pytest test suite
config/            # config.yaml — single source of truth for all parameters
tools/             # Development utilities (pipeline monitor)
docs/              # Pipeline results report, user guide
data/
  raw/             # Raw data (not committed)
  processed/       # Preprocessed parquet files (not committed)
  results/         # Analysis outputs (not committed)
  reference/       # Bundled reference files (committed)
```

---

## Results Summary

See [`docs/pipeline_results.md`](docs/pipeline_results.md) for the full annotated results report, including:

- ADNI DNB stage scores (MCI→Dementia = 7.49, stable MCI = 6.18)
- 13 ADNI DNB core proteins with biological annotation
- sDNB predictive performance (AUC = 0.57 vs. AMYLOID_STATUS AUC = 0.78)
- PPMI replication: DNB peaks at PD_INTERMEDIATE (0.565)
- Cross-disease shared protein: CRYBB2 (Beta-crystallin B2) identified in both ADNI and PPMI

---

## Citation

If you use this code in your research, please cite:

> Hashim, M.E. (2026). *CriticalNeuroMap: Dynamic Network Biomarker and Critical Slowing Down in Proteomics for Neurodegenerative Disease Transitions* (v2.0.0). Manuscript in preparation.

See also [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

---

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

---

## Author

**Mohamed Essam Hashim, M.B.B.S.**
