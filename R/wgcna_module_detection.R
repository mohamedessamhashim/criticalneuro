# R/wgcna_module_detection.R
# ============================
# WGCNA co-expression module detection for CriticalNeuroMap.
#
# Runs WGCNA on the post-QC, batch-corrected, residualized protein matrix
# to identify co-expression modules and compute module membership (kME).
#
# Input:  data/processed/adni_proteomics_clean.parquet
# Output: data/results/wgcna/wgcna_modules.csv       (protein → module + kME)
#         data/results/wgcna/wgcna_summary.csv        (module summary statistics)
#         data/results/wgcna/soft_threshold_fit.csv    (scale-free topology fit)
#
# Usage: Rscript R/wgcna_module_detection.R
#
# Reference: Langfelder & Horvath (2008) BMC Bioinformatics

library(yaml)
library(arrow)
library(WGCNA)

# Allow multi-threading for WGCNA
allowWGCNAThreads()

# ---- Load configuration ----
config <- yaml::read_yaml("config/config.yaml")
set.seed(config$random_seed)

wgcna_cfg <- config$wgcna
soft_power_max   <- wgcna_cfg$soft_power_max       # 20
min_mod_size     <- wgcna_cfg$min_module_size_wgcna # 30
merge_cut_height <- wgcna_cfg$merge_cut_height      # 0.25

# ---- Load processed ADNI proteomics data ----
input_path <- config$paths$adni_clean_parquet
if (!file.exists(input_path)) {
  stop(sprintf("Processed ADNI data not found at %s. Run preprocessing first.", input_path))
}

df <- arrow::read_parquet(input_path)
message(sprintf("Loaded ADNI data: %d samples, %d columns", nrow(df), ncol(df)))

# ---- Identify protein columns ----
seq_cols <- grep("^seq\\.", colnames(df), value = TRUE)
message(sprintf("Found %d protein columns (seq.*)", length(seq_cols)))

if (length(seq_cols) < 50) {
  stop("Fewer than 50 protein columns found — WGCNA requires a larger feature set")
}

# ---- Prepare expression matrix ----
# WGCNA expects samples as rows, genes/proteins as columns
# Use the entire cohort to establish the co-expression architecture
expr_matrix <- as.matrix(df[, seq_cols])

# Impute NaN with column medians for WGCNA (it cannot handle NAs)
for (j in seq_len(ncol(expr_matrix))) {
  na_mask <- is.na(expr_matrix[, j])
  if (any(na_mask)) {
    expr_matrix[na_mask, j] <- median(expr_matrix[!na_mask, j], na.rm = TRUE)
  }
}

# Check for zero-variance columns and remove them
col_vars <- apply(expr_matrix, 2, var, na.rm = TRUE)
zero_var_mask <- col_vars == 0 | is.na(col_vars)
if (any(zero_var_mask)) {
  message(sprintf("Removing %d zero-variance proteins", sum(zero_var_mask)))
  expr_matrix <- expr_matrix[, !zero_var_mask]
  seq_cols <- seq_cols[!zero_var_mask]
}

message(sprintf("Expression matrix for WGCNA: %d samples x %d proteins",
                nrow(expr_matrix), ncol(expr_matrix)))

# ---- Check for good genes and good samples ----
gsg <- goodSamplesGenes(expr_matrix, verbose = 3)
if (!gsg$allOK) {
  if (sum(!gsg$goodGenes) > 0) {
    message(sprintf("Removing %d genes flagged by goodSamplesGenes", sum(!gsg$goodGenes)))
  }
  if (sum(!gsg$goodSamples) > 0) {
    message(sprintf("Removing %d samples flagged by goodSamplesGenes", sum(!gsg$goodSamples)))
  }
  expr_matrix <- expr_matrix[gsg$goodSamples, gsg$goodGenes]
  seq_cols <- seq_cols[gsg$goodGenes]
}

message(sprintf("After QC: %d samples x %d proteins", nrow(expr_matrix), ncol(expr_matrix)))

# ---- Step 1: Soft-thresholding power selection ----
message("=== Step 1: Picking soft-thresholding power ===")
powers <- c(seq(1, 10, by = 1), seq(12, soft_power_max, by = 2))
sft <- pickSoftThreshold(expr_matrix,
                         powerVector = powers,
                         RsquaredCut = 0.85,
                         networkType = "signed",
                         verbose = 3)

# Save scale-free topology fit table
sft_df <- sft$fitIndices
output_dir <- wgcna_cfg$results_dir
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
write.csv(sft_df, file.path(output_dir, "soft_threshold_fit.csv"), row.names = FALSE)

# Select power: use WGCNA's recommendation, fall back to 6 if none meets R^2 > 0.85
soft_power <- sft$powerEstimate
if (is.na(soft_power) || is.null(soft_power)) {
  soft_power <- 6
  message(sprintf("No power met R^2 threshold — using fallback power = %d", soft_power))
} else {
  message(sprintf("Selected soft-thresholding power = %d (R^2 = %.3f)",
                  soft_power, sft_df$SFT.R.sq[sft_df$Power == soft_power]))
}

# ---- Step 2: Blockwise module detection ----
message("=== Step 2: Blockwise module detection ===")
message(sprintf("Parameters: power=%d, minModuleSize=%d, mergeCutHeight=%.2f",
                soft_power, min_mod_size, merge_cut_height))

net <- blockwiseModules(
  expr_matrix,
  power              = soft_power,
  networkType        = "signed",
  TOMType            = "signed",
  minModuleSize      = min_mod_size,
  mergeCutHeight     = merge_cut_height,
  reassignThreshold  = 0,
  numericLabels      = FALSE,    # Use color labels
  pamRespectsDendro   = FALSE,
  saveTOMs            = FALSE,   # Don't save TOM matrices (large files)
  verbose             = 3
)

module_colors <- net$colors
n_modules <- length(unique(module_colors)) - ("grey" %in% unique(module_colors))
message(sprintf("Detected %d modules (+ grey/unassigned)", n_modules))

# Module sizes
module_sizes <- table(module_colors)
message("Module sizes:")
for (mod in names(sort(module_sizes, decreasing = TRUE))) {
  message(sprintf("  %s: %d proteins", mod, module_sizes[mod]))
}

# ---- Step 3: Module eigengenes and kME ----
message("=== Step 3: Computing module eigengenes and kME ===")
MEs <- net$MEs

# kME: correlation of each protein with each module eigengene
# signedKME returns a matrix with columns named kME<color>
kME_all <- signedKME(expr_matrix, MEs)

# For each protein, get its assigned module's kME
kME_values <- numeric(ncol(expr_matrix))
for (i in seq_along(seq_cols)) {
  assigned_module <- module_colors[i]
  kme_col <- paste0("kME", assigned_module)
  if (kme_col %in% colnames(kME_all)) {
    kME_values[i] <- kME_all[i, kme_col]
  } else {
    kME_values[i] <- NA
  }
}

# ---- Step 4: Write outputs ----
message("=== Step 4: Writing outputs ===")

# Per-protein module assignments with kME
modules_df <- data.frame(
  protein = seq_cols,
  module  = module_colors,
  kME     = round(kME_values, 4),
  stringsAsFactors = FALSE
)
modules_path <- file.path(output_dir, "wgcna_modules.csv")
write.csv(modules_df, modules_path, row.names = FALSE)
message(sprintf("Module assignments written: %s (%d proteins)", modules_path, nrow(modules_df)))

# Summary statistics
summary_rows <- list()
for (mod in unique(module_colors)) {
  mod_mask <- module_colors == mod
  mod_kmes <- kME_values[mod_mask]
  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    module       = mod,
    n_proteins   = sum(mod_mask),
    mean_kME     = round(mean(mod_kmes, na.rm = TRUE), 4),
    median_kME   = round(median(mod_kmes, na.rm = TRUE), 4),
    min_kME      = round(min(mod_kmes, na.rm = TRUE), 4),
    max_kME      = round(max(mod_kmes, na.rm = TRUE), 4),
    stringsAsFactors = FALSE
  )
}
summary_df <- do.call(rbind, summary_rows)
summary_df <- summary_df[order(-summary_df$n_proteins), ]

# Add metadata row
meta_row <- data.frame(
  module = "META",
  n_proteins = ncol(expr_matrix),
  mean_kME = soft_power,
  median_kME = n_modules,
  min_kME = nrow(expr_matrix),
  max_kME = merge_cut_height,
  stringsAsFactors = FALSE
)
summary_df <- rbind(meta_row, summary_df)

summary_path <- file.path(output_dir, "wgcna_summary.csv")
write.csv(summary_df, summary_path, row.names = FALSE)
message(sprintf("Summary written: %s", summary_path))

message(sprintf("=== WGCNA complete: %d modules from %d proteins ===",
                n_modules, ncol(expr_matrix)))
