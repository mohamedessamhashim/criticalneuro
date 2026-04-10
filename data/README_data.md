# Data Format Specification for CriticalNeuroMap

## BEFORE RUNNING: Answer Three Questions

### Question 1: Has the Cruchaga Lab QC pipeline already been applied?

Does The Cruchaga Lab apply a specific QC pipeline after receiving data from SomaLogic?:
- Removes aptamers where max|cal_SF - median_SF| >= 0.5
- Removes aptamers with median cross-plate CV > 0.15
- Removes aptamers where >85% of values fall outside 1.5xIQR

If you are providing the standard Knight-ADRC SomaScan delivery
(the post-QC protein matrix, not raw ADAT files), answer: **YES**

Set in config.yaml:
```yaml
cruchaga_qc_already_applied: true
```

### Question 2: Is the expression matrix already log2-transformed?

Does the Cruchaga Lab typically deliver RFU values in untransformed space
even after QC. Check: are values in the range ~100-100,000 (RFU space)?
Or are they in the range ~7-17 (log2 space)?

If values are in RFU space (typical): set `log2_already_applied: false`
If values are already log2: set `log2_already_applied: true`

### Question 3: Has ComBat batch correction already been applied?

Answer is probably: **NO**

Set in config.yaml:
```yaml
combat_already_applied: false
```

---

## Expression Matrix Format

**File**: CSV with samples as rows and aptamers as columns.

```
,seq.1234.56,seq.2345.67,seq.3456.78,...
SAMPLE001,1234.5,5678.9,2345.6,...
SAMPLE002,1345.6,4567.8,3456.7,...
```

- **Row index**: SampleID (must match metadata SampleID column)
- **Column names**: SeqId format `seq.XXXX.XX` (e.g., `seq.3072.4`)
- **Values**: Post-SomaLogic-normalization RFU values (if not pre-log2'd)
  - Typical range: 100 to 100,000 (RFU space)
  - If already log2: 7 to 17

## Metadata Format

**File**: CSV with one row per sample-visit.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| SampleID | string | Unique sample identifier | "KADRC_001_V1" |
| SubjectID | string | Participant ID (repeated across visits) | "KADRC_001" |
| Age | numeric | Age at blood draw | 72.5 |
| Sex | string | "M" or "F" | "F" |
| APOE4 | integer | Number of APOE e4 alleles (0, 1, or 2) | 1 |
| Plate | string | SomaScan plate/subarray ID | "Plate_A01" |
| Diagnosis | string | "CO" or "AD" at each visit | "CO" |
| VisitDate | date | Date of blood draw (YYYY-MM-DD) | "2020-03-15" |
| VisitNumber | integer | Visit sequence (1=first) | 3 |
| Converter | boolean | Did this subject convert CO->AD? | TRUE |
| VisitsToDx | integer | **(Optional)** Visits until AD diagnosis (0=AD visit, NA for non-converters) | 2 |

### VisitsToDx (auto-computed if missing)

If your metadata does **not** include a `VisitsToDx` column, the pipeline will
automatically compute it from `Diagnosis` and `VisitNumber`:
- For each converter, it finds the first visit where `Diagnosis == "AD"`
- Then counts backwards: `VisitsToDx = AD_visit - VisitNumber`
- Non-converters get `VisitsToDx = NA`

Example for a converter with trajectory CO->CO->CO->AD:
- Visit 1 (CO): VisitsToDx = 3
- Visit 2 (CO): VisitsToDx = 2
- Visit 3 (CO): VisitsToDx = 1
- Visit 4 (AD): VisitsToDx = 0

For non-converters: VisitsToDx = NA for all visits.

If you provide `VisitsToDx` in your metadata, the pipeline will use it as-is.
