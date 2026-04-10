# ============================================================
# Stage 4a: WGCNA Module Detection
# ============================================================
# Called by run_pipeline.py via subprocess
# Args: --config, --input (residualized matrix CSV), --output

suppressPackageStartupMessages({
  library(WGCNA)
  library(yaml)
  library(optparse)
})

# Allow multi-threading for WGCNA
allowWGCNAThreads()

# --- Parse arguments ---
option_list <- list(
  make_option("--config", type = "character"),
  make_option("--input",  type = "character"),  # residualized matrix (samples x proteins)
  make_option("--output", type = "character")
)
opt <- parse_args(OptionParser(option_list = option_list))
cfg <- yaml.load_file(opt$config)

# --- Set seed ---
set.seed(cfg$reproducibility$r_seed)

# --- Load data ---
expr <- read.csv(opt$input, row.names = 1, check.names = FALSE)
cat(sprintf("[WGCNA] Loaded: %d samples x %d columns\n", nrow(expr), ncol(expr)))

# Identify protein columns
seq_cols <- grep("^seq\\.", colnames(expr), value = TRUE)
if (length(seq_cols) == 0) {
  seq_cols <- colnames(expr)[sapply(expr, is.numeric)]
}
cat(sprintf("[WGCNA] Found %d protein columns\n", length(seq_cols)))

# Prepare expression matrix: WGCNA needs samples x proteins
datExpr <- as.matrix(expr[, seq_cols])

# Impute NaN with column medians for WGCNA (cannot handle NAs)
for (j in seq_len(ncol(datExpr))) {
  na_mask <- is.na(datExpr[, j])
  if (any(na_mask)) {
    datExpr[na_mask, j] <- median(datExpr[!na_mask, j], na.rm = TRUE)
  }
}

# Remove zero-variance columns
col_vars <- apply(datExpr, 2, var, na.rm = TRUE)
zero_var_mask <- col_vars == 0 | is.na(col_vars)
if (any(zero_var_mask)) {
  cat(sprintf("[WGCNA] Removing %d zero-variance proteins\n", sum(zero_var_mask)))
  datExpr <- datExpr[, !zero_var_mask]
  seq_cols <- seq_cols[!zero_var_mask]
}

# --- Validate: check for missing values ---
gsg <- goodSamplesGenes(datExpr, verbose = 0)
if (!gsg$allOK) {
  datExpr <- datExpr[gsg$goodSamples, gsg$goodGenes]
  seq_cols <- seq_cols[gsg$goodGenes]
  cat(sprintf("[WGCNA] Removed %d bad samples, %d bad genes\n",
              sum(!gsg$goodSamples), sum(!gsg$goodGenes)))
}

cat(sprintf("[WGCNA] Expression matrix: %d samples x %d proteins\n",
            nrow(datExpr), ncol(datExpr)))

# --- Soft threshold selection ---
wgcna_cfg <- cfg$wgcna
powers <- wgcna_cfg$soft_threshold_range[1]:wgcna_cfg$soft_threshold_range[2]

# Set seed before soft threshold selection
set.seed(cfg$reproducibility$r_seed)

sft <- pickSoftThreshold(datExpr,
                          powerVector = powers,
                          networkType = wgcna_cfg$network_type,
                          corFnc = wgcna_cfg$cor_method,
                          verbose = 0)

# Auto-select threshold: first power where R^2 >= 0.80
r2 <- sft$fitIndices[, "SFT.R.sq"]
softPower <- powers[which(r2 >= 0.80)[1]]
if (is.na(softPower)) {
  softPower <- powers[which.max(r2)]
  warning(sprintf("[WGCNA] R^2 never reached 0.80. Using power=%d (R^2=%.3f). Consider increasing soft_threshold_range.",
                  softPower, max(r2, na.rm = TRUE)))
}
cat(sprintf("[WGCNA] Selected soft threshold power = %d (R^2 = %.3f)\n",
            softPower, r2[powers == softPower]))

# --- Build network and detect modules ---
# Set seed before module detection
set.seed(cfg$reproducibility$r_seed)

net <- blockwiseModules(
  datExpr,
  power              = softPower,
  networkType        = wgcna_cfg$network_type,
  corType            = wgcna_cfg$cor_method,
  TOMType            = wgcna_cfg$network_type,
  minModuleSize      = wgcna_cfg$min_module_size,
  mergeCutHeight     = wgcna_cfg$merge_cut_height,
  maxBlockSize       = wgcna_cfg$max_block_size,
  numericLabels      = FALSE,
  pamRespectsDendro  = FALSE,
  saveTOMs           = FALSE,
  verbose            = 0,
  randomSeed         = cfg$reproducibility$r_seed
)

moduleColors <- net$colors
MEs <- net$MEs  # Module eigengenes

n_modules <- length(unique(moduleColors)) - 1  # Exclude "grey" (unassigned)
cat(sprintf("[WGCNA] Detected %d modules across %d proteins\n",
            n_modules, ncol(datExpr)))

# Module sizes
module_sizes <- table(moduleColors)
for (mod in names(sort(module_sizes, decreasing = TRUE))) {
  cat(sprintf("  %s: %d proteins\n", mod, module_sizes[mod]))
}

# --- Module membership (kME) ---
kME <- signedKME(datExpr, MEs, corFnc = wgcna_cfg$cor_method)

# --- Save outputs ---
dir.create(opt$output, showWarnings = FALSE, recursive = TRUE)

# Module assignments
module_df <- data.frame(
  AptName = colnames(datExpr),
  Module  = moduleColors,
  stringsAsFactors = FALSE
)
# Add kME for each protein's own module
module_df$kME <- sapply(seq_len(nrow(module_df)), function(i) {
  mod <- module_df$Module[i]
  col <- paste0("kME", mod)
  if (col %in% colnames(kME)) kME[i, col] else NA
})
write.csv(module_df,
          file.path(opt$output, "wgcna_module_assignments.csv"),
          row.names = FALSE)

# Module eigengenes
write.csv(MEs,
          file.path(opt$output, "wgcna_module_eigengenes.csv"),
          row.names = TRUE)

# Soft threshold diagnostics
write.csv(sft$fitIndices,
          file.path(opt$output, "wgcna_soft_threshold_selection.csv"),
          row.names = FALSE)

# Save power used (for reproducibility record)
writeLines(as.character(softPower),
           file.path(opt$output, "wgcna_soft_power_used.txt"))

cat(sprintf("[WGCNA] Complete. Outputs saved to %s\n", opt$output))
