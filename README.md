# CriticalNeuroMap

**Dynamic Network Biomarker and Critical Slowing Down in Longitudinal Proteomics as Early Warning Signals for Neurodegenerative Disease Transitions**

---

## Quick Start for Knight-ADRC Analysis 

### 1. Install dependencies (once)
```bash
conda activate criticalneuro
pip install -r requirements_python.txt
Rscript install_dependencies.R
```

### 2. Prepare your data files
Place your Knight-ADRC files at the paths specified in config.yaml:
- Expression matrix: rows = samples, columns = SeqIds (e.g. seq.1234.56)
- Metadata CSV: must include the columns listed in config.yaml

See `data/README_data.md` for exact column specifications and format.

**IMPORTANT**: Answer the 3 data provenance questions in `data/README_data.md` before running.
The pipeline needs to know if your data is already QC'd to avoid double-normalization.

### 3. Configure (minimal changes needed)
Edit `config.yaml`:
```yaml
analysis_mode: "longitudinal"
cohort: "knight_adrc"
input:
  expression_matrix: "path/to/your/expression.csv"
  metadata: "path/to/your/metadata.csv"
data_provenance:
  cruchaga_qc_already_applied: true    # YES if post-Cruchaga-QC data
  log2_already_applied: false           # NO if values are in RFU space (~100-100000)
  combat_already_applied: false         # NO (Cruchaga Lab doesn't routinely apply ComBat)
```

### 4. Run
```bash
conda activate criticalneuro
python run_pipeline.py
```

### 5. Resume if interrupted
```bash
python run_pipeline.py --resume
```

### 6. Results
All outputs appear in `results/`:
- `results/tables/` — all CSV outputs
- `results/figures/` — all plots (PDF + PNG)
- `results/pipeline.log` — full run log

---

## Background

Complex biological systems approaching a tipping point exhibit *critical slowing down* (CSD): rising variance, increasing autocorrelation, and growing inter-component correlations. The **Dynamic Network Biomarker (DNB)** framework (Chen et al. 2012, *Nature Communications*) operationalises this into a single score — the ratio of within-group protein co-variation to between-group co-variation — that peaks at the pre-transition state, just before a system tips irreversibly.

This repository applies DNB analysis to high-dimensional plasma and CSF SomaScan proteomics to ask: can coordinated protein network destabilisation serve as an early warning signal for neurodegenerative disease transitions? Two independent cohorts are analysed:

- **ADNI** — SomaScan plasma (~6,946 proteins, 332 participants) for the MCI → Alzheimer's dementia transition
- **PPMI** — SomaScan CSF (~4,785 proteins, 1,250 samples) for slow vs. rapid Parkinson's disease progression

A per-participant single-sample DNB (sDNB) score is derived for individual risk stratification, and results are validated against established biomarkers (amyloid PET, APOE4, ABETA42/40 ratio).

---

## Installation

Requires [conda](https://docs.conda.io/en/latest/) and R >= 4.4.

```bash
# Create environment (first time only)
conda env create -f environment.yml

# Always activate before running
conda activate criticalneuro

# Install R packages not in conda (BioTIP, lme4, etc.)
Rscript install_dependencies.R
```

Verify all dependencies:
```bash
conda activate criticalneuro
Rscript -e "library(WGCNA); library(BioTIP); library(lme4); cat('R packages OK\n')"
python -c "import numpy, pandas, scipy, yaml; print('Python packages OK')"
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

### Longitudinal Analysis (Knight-ADRC)
```bash
conda activate criticalneuro

# Run the full WGCNA → BioTIP → l-DNB pipeline
python run_pipeline.py

# Run a single stage
python run_pipeline.py --stage wgcna

# Resume from last checkpoint
python run_pipeline.py --resume
```

### Cross-Sectional Analysis (ADNI/PPMI)
```bash
conda activate criticalneuro

# Run the full cross-sectional pipeline
python pipelines/run_full_pipeline.py

# Run a single stage
python pipelines/run_full_pipeline.py --stage wgcna

# Force re-run
python pipelines/run_full_pipeline.py --force
```

**Longitudinal stages** (Knight-ADRC via `run_pipeline.py`):

| Stage | Description |
|-------|-------------|
| `load_data` | Load and validate Knight-ADRC data |
| `normalize` | Log2 transform (if needed) |
| `residualize` | Covariate residualization (lme4 mixed models) |
| `batch_correct` | ComBat batch correction (if needed) |
| `wgcna` | WGCNA co-expression module detection |
| `biotip` | BioTIP tipping-point scoring (MCI + permutation test) |
| `ldnb` | Individual l-DNB scoring |
| `network_medicine` | Interactome proximity analysis |
| `validation` | Converter vs. stable comparison |
| `figures` | Publication figures |

**Cross-sectional stages** (ADNI/PPMI via `pipelines/run_full_pipeline.py`):

| Stage | Description |
|-------|-------------|
| `adni_preprocess` | Load, QC, and batch-correct ADNI SomaScan data |
| `ppmi_preprocess` | Load and QC PPMI SomaScan data |
| `wgcna` | WGCNA module detection |
| `dnb_wgcna_somascan` | WGCNA-guided DNB analysis |
| `csd` | Critical Slowing Down analysis |
| `validation` | sDNB vs. established biomarkers (ROC, Spearman) |
| `ppmi_replication` | PPMI replication |
| `figures` | Generate all publication figures |

Outputs are written to `results/` (longitudinal) or `data/results/` (cross-sectional).

---

## Tests

```bash
pytest tests/ -v --tb=short
```

---

## Project Structure

```
run_pipeline.py      # Main entry point (longitudinal Knight-ADRC)
config.yaml          # Central configuration file

pipeline/            # NEW: WGCNA -> BioTIP -> l-DNB pipeline stages
  config_loader.py   #   Config loading and validation
  logger.py          #   Centralized logging
  stage1_qc.py       #   QC with provenance-aware routing
  stage2_normalization.py
  stage3_residualization.R  # lme4 mixed models (longitudinal)
  stage3_batch_correction.py
  stage4_wgcna.R     #   WGCNA module detection
  stage4_biotip.R    #   BioTIP tipping-point scoring
  stage4_ldnb.R      #   l-DNB individual scoring
  stage5_network_medicine.py
  stage5_validation.py
  stage6_figures.R   #   Publication figures

utils/               # Shared utilities
  longitudinal_utils.R  # Timepoint alignment, trajectory validation
  somascan_utils.R      # SeqId mapping helpers
  stats_utils.py        # Statistical helpers

src/                 # Original analysis modules (cross-sectional)
  preprocessing/     #   Data loading, QC, batch correction
  dnb/               #   DNB score computation, WGCNA-guided DNB, sDNB
  csd/               #   Critical Slowing Down analysis
  validation/        #   ROC/AUC, biomarker comparison
  cross_platform/    #   Golden Set, platform concordance
  network_medicine/  #   Interactome proximity (NetMedPy)

pipelines/           # Cross-sectional pipeline orchestrator
R/                   # R scripts (ADNIMERGE, GSEA, earlywarnings)
config/              # Legacy config (cross-sectional mode)
tests/               # pytest test suite
data/
  raw/               # Raw data (not committed)
  processed/         # Preprocessed files (not committed)
  results/           # Cross-sectional outputs
  reference/         # Bundled reference files
```

---

## Author

**Mohamed Essam Hashim, M.B.B.S.**
