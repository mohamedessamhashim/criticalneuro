# R/adnimerge_extract.R
library(yaml)
library(ADNIMERGE2)

config <- yaml::read_yaml("config/config.yaml")
output_path <- config$paths$adnimerge_csv
output_dir <- dirname(output_path)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

message("Using ADNIMERGE2 package — building merged table from individual datasets")

# ---- Core tables ----
registry  <- as.data.frame(REGISTRY)
dxsum     <- as.data.frame(DXSUM)
cdrtab    <- as.data.frame(CDR)
mmsetab   <- as.data.frame(MMSE)
demog     <- as.data.frame(PTDEMOG)
apoe      <- as.data.frame(APOERES)

# Standardize column names to uppercase
for (obj in c("registry","dxsum","cdrtab","mmsetab","demog","apoe")) {
  df <- get(obj)
  colnames(df) <- toupper(colnames(df))
  assign(obj, df)
}

# ---- Build join keys ----
# Use RID + VISCODE as the merge key throughout
merge2 <- function(x, y, ...) merge(x, y, by=c("RID","VISCODE"), all.x=TRUE, ...)

# Start from registry (all participants and visits)
merged <- registry[, intersect(colnames(registry),
  c("RID","VISCODE","EXAMDATE","PTID","SITEID","COLPROT","ORIGPROT","USERDATE"))]

# Diagnosis
if (all(c("RID","VISCODE") %in% colnames(dxsum))) {
  dx_cols <- intersect(colnames(dxsum), c("RID","VISCODE","DIAGNOSIS","DX","DXCHANGE","DXCURREN","DXCONV"))
  merged <- merge2(merged, dxsum[, dx_cols])
}

# CDR
if (all(c("RID","VISCODE") %in% colnames(cdrtab))) {
  cdr_cols <- intersect(colnames(cdrtab), c("RID","VISCODE","CDGLOBAL","CDRSB","CDSUMBOX"))
  merged <- merge2(merged, cdrtab[, cdr_cols])
  # Rename to expected column names
  if ("CDSUMBOX" %in% colnames(merged) && !"CDRSB" %in% colnames(merged))
    colnames(merged)[colnames(merged)=="CDSUMBOX"] <- "CDRSB"
}

# MMSE
if (all(c("RID","VISCODE") %in% colnames(mmsetab))) {
  mmse_cols <- intersect(colnames(mmsetab), c("RID","VISCODE","MMSCORE","MMSE"))
  merged <- merge2(merged, mmsetab[, mmse_cols])
  if ("MMSCORE" %in% colnames(merged) && !"MMSE" %in% colnames(merged))
    colnames(merged)[colnames(merged)=="MMSCORE"] <- "MMSE"
}

# Demographics (baseline only — merge on RID)
if ("RID" %in% colnames(demog)) {
  colnames(demog) <- toupper(colnames(demog))
  demog_cols <- intersect(colnames(demog), c("RID","PTGENDER","PTEDUCAT","PTETHCAT","PTRACCAT","PTMARRY","PTDOBYY"))
  demog_bl <- demog[!duplicated(demog$RID), demog_cols]
  merged <- merge(merged, demog_bl, by="RID", all.x=TRUE)
  if ("PTGENDER" %in% colnames(merged)) {
    merged$PTGENDER_BIN <- as.integer(merged$PTGENDER == "Female" | merged$PTGENDER == 2)
  }
  if ("PTDOBYY" %in% colnames(merged) && "EXAMDATE" %in% colnames(merged)) {
    merged$AGE <- as.numeric(format(as.Date(merged$EXAMDATE), "%Y")) - merged$PTDOBYY
  }
}

# APOE (baseline only — merge on RID)
if ("RID" %in% colnames(apoe)) {
  apoe_cols <- intersect(colnames(apoe), c("RID","APGEN1","APGEN2","GENOTYPE"))
  apoe_bl <- apoe[!duplicated(apoe$RID), apoe_cols]
  merged <- merge(merged, apoe_bl, by="RID", all.x=TRUE)
  # Create APOE4 binary carrier flag
  if ("APGEN1" %in% colnames(merged) && "APGEN2" %in% colnames(merged)) {
    merged$APOE4 <- as.integer(merged$APGEN1 == 4 | merged$APGEN2 == 4)
  } else if ("GENOTYPE" %in% colnames(merged)) {
    merged$APOE4 <- as.integer(grepl("4", as.character(merged$GENOTYPE)))
  }
}

# ---- Amyloid PET ----
if (exists("UCBERKELEY_AMY_6MM")) {
  amy <- as.data.frame(UCBERKELEY_AMY_6MM)
  colnames(amy) <- toupper(colnames(amy))
  amy_cols <- intersect(colnames(amy), c("RID","VISCODE","SUMMARY_SUVR","CENTILOIDS","AMYLOID_STATUS"))
  if (length(amy_cols) > 2) {
    merged <- merge2(merged, amy[, amy_cols])
    # AV45 SUVR threshold
    suvr_col <- intersect(colnames(merged), c("SUMMARY_SUVR","AV45"))[1]
    if (!is.na(suvr_col)) {
      merged$AMYLOID_STATUS <- NA_integer_
      merged$AMYLOID_STATUS[!is.na(merged[[suvr_col]]) & merged[[suvr_col]] >= 1.11] <- 1L
      merged$AMYLOID_STATUS[!is.na(merged[[suvr_col]]) & merged[[suvr_col]] < 1.11]  <- 0L
    }
    if ("CENTILOIDS" %in% colnames(merged)) {
      cl <- merged$CENTILOIDS
      merged$AMYLOID_STATUS[!is.na(cl) & cl >= 20] <- 1L
      merged$AMYLOID_STATUS[!is.na(cl) & cl <  20] <- 0L
    }
  }
}

# ---- Tau PET ----
if (exists("UCBERKELEY_TAU_6MM")) {
  tau <- as.data.frame(UCBERKELEY_TAU_6MM)
  colnames(tau) <- toupper(colnames(tau))
  tau_cols <- intersect(colnames(tau), c("RID","VISCODE","META_TEMPORAL_SUVR","TAU_SUVR"))
  if (length(tau_cols) > 2) {
    merged <- merge2(merged, tau[, tau_cols])
    tau_suvr_col <- intersect(colnames(merged), c("META_TEMPORAL_SUVR","TAU_SUVR"))[1]
    if (!is.na(tau_suvr_col)) {
      merged$TAU_PET_STATUS <- NA_integer_
      tv <- merged[[tau_suvr_col]]
      merged$TAU_PET_STATUS[!is.na(tv) & tv >= 1.30] <- 1L
      merged$TAU_PET_STATUS[!is.na(tv) & tv <  1.30] <- 0L
    }
  }
}

# ---- Months from baseline ----
if ("EXAMDATE" %in% colnames(merged)) {
  merged$EXAMDATE <- as.Date(merged$EXAMDATE)
  merged <- merged[order(merged$RID, merged$EXAMDATE), ]
  bl_dates <- tapply(merged$EXAMDATE, merged$RID, min, na.rm=TRUE)
  bl_dates <- as.Date(bl_dates, origin="1970-01-01")
  merged$MONTHS_FROM_BL <- as.numeric(
    difftime(merged$EXAMDATE, bl_dates[as.character(merged$RID)], units="days")
  ) / 30.44
}

# ---- Diagnosis label consolidation ----
# Prefer DIAGNOSIS (ADNI3+) then DX then DXCHANGE
if (!"DX" %in% colnames(merged)) {
  if ("DIAGNOSIS" %in% colnames(merged)) {
    merged$DX <- as.character(merged$DIAGNOSIS)
  } else if ("DXCHANGE" %in% colnames(merged)) {
    merged$DX <- as.character(merged$DXCHANGE)
  }
}

# ---- Summary ----
message(sprintf("Final: %d visits from %d participants", nrow(merged), length(unique(merged$RID))))
if ("DX" %in% colnames(merged)) {
  dx_table <- table(merged$DX, useNA="ifany")
  message("Diagnosis distribution:")
  for (dx in names(dx_table)) message(sprintf("  %s: %d", dx, dx_table[dx]))
}

write.csv(merged, output_path, row.names=FALSE)
message(sprintf("ADNIMERGE.csv written to %s", output_path))
