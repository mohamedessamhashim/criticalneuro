# ============================================================
# Stage 4b: BioTIP Tipping-Point Detection
# ============================================================
# Scores WGCNA modules using Module-Criticality Index (MCI),
# identifies Critical Transition Signal (CTS) proteins,
# and runs permutation testing for significance.
#
# Called by run_pipeline.py via subprocess
# Args: --config, --expr, --metadata, --modules, --output

suppressPackageStartupMessages({
  library(BioTIP)
  library(yaml)
  library(optparse)
})

# --- Parse arguments ---
option_list <- list(
  make_option("--config",   type = "character"),
  make_option("--expr",     type = "character"),  # residualized matrix CSV
  make_option("--metadata", type = "character"),  # sample metadata with stage labels
  make_option("--modules",  type = "character"),  # WGCNA module assignments CSV
  make_option("--output",   type = "character")
)
opt <- parse_args(OptionParser(option_list = option_list))
cfg <- yaml.load_file(opt$config)
set.seed(cfg$reproducibility$r_seed)

# --- Load inputs ---
expr     <- as.matrix(read.csv(opt$expr, row.names = 1, check.names = FALSE))
metadata <- read.csv(opt$metadata, stringsAsFactors = FALSE)
modules  <- read.csv(opt$modules, stringsAsFactors = FALSE)

cat(sprintf("[BioTIP] Expression: %d features x %d samples\n", nrow(expr), ncol(expr)))
cat(sprintf("[BioTIP] Metadata: %d rows\n", nrow(metadata)))
cat(sprintf("[BioTIP] WGCNA modules: %d proteins\n", nrow(modules)))

# --- Ensure expression is proteins x samples ---
# If rows are samples (more columns than rows of seq.* pattern), transpose
if (nrow(expr) < ncol(expr)) {
  # Already proteins x samples or samples x proteins?
  # Check if rownames match protein names
  if (any(grepl("^seq\\.", rownames(expr)))) {
    cat("[BioTIP] Expression appears to be proteins x samples (correct orientation)\n")
  } else {
    cat("[BioTIP] Transposing expression matrix to proteins x samples\n")
    expr <- t(expr)
  }
} else {
  if (!any(grepl("^seq\\.", rownames(expr)))) {
    cat("[BioTIP] Transposing expression matrix to proteins x samples\n")
    expr <- t(expr)
  }
}

# --- Build stage sample lists ---
if (cfg$analysis_mode == "longitudinal") {
  stage_map <- cfg$longitudinal$stage_definitions
  metadata$Stage <- sapply(metadata$VisitsToDx, function(v) {
    if (is.na(v)) return("stable_CO")
    if (v < 0)    return("post_AD")   # Post-diagnosis follow-ups — exclude from analysis
    key <- as.character(v)
    if (key %in% names(stage_map)) stage_map[[key]] else "stable_CO"
  })

  # Ordered stages for tipping-point analysis
  stage_order <- c("T3_baseline", "T2_early", "T1_preClinical", "T0_AD")

} else {
  # Cross-sectional (ADNI): use existing TRAJECTORY column
  stage_order <- c("CN_amyloid_negative", "Stable_MCI",
                   "CN_amyloid_positive", "MCI_to_Dementia")
  metadata$Stage <- metadata$TRAJECTORY
}

# Create named list: stage -> vector of SampleIDs
samplesL <- lapply(stage_order, function(s) {
  ids <- metadata$SampleID[metadata$Stage == s]
  # Only keep IDs that exist in expression columns
  intersect(as.character(ids), colnames(expr))
})
names(samplesL) <- stage_order

# Validate: check each stage has enough samples
for (s in stage_order) {
  n <- length(samplesL[[s]])
  if (n < cfg$biotip$mci_bottom) {
    stop(sprintf("[BioTIP] Stage '%s' has only %d samples (minimum %d). Check metadata.",
                 s, n, cfg$biotip$mci_bottom))
  }
  cat(sprintf("[BioTIP] Stage '%s': n=%d samples\n", s, n))
}

# --- Build module gene lists from WGCNA output ---
# Exclude grey module (unassigned proteins)
module_list <- split(modules$AptName, modules$Module)
module_list <- module_list[names(module_list) != "grey"]

# Filter module list to proteins in expression matrix
module_list <- lapply(module_list, function(prots) {
  intersect(prots, rownames(expr))
})
# Remove empty modules
module_list <- module_list[sapply(module_list, length) >= cfg$biotip$mci_bottom]
cat(sprintf("[BioTIP] Testing %d WGCNA modules\n", length(module_list)))

# --- Convert WGCNA modules to BioTIP cluster membership format ---
# BioTIP getMCI expects:
#   groups:  list (per state) of named integer vectors (gene → cluster number)
#   countsL: list (per state) of expression matrices (genes × samples)
set.seed(cfg$reproducibility$r_seed)

# Build membership vector from WGCNA module assignments
# Only include proteins present in expression matrix
valid_modules <- modules[modules$AptName %in% rownames(expr), ]
module_colors <- valid_modules$Module
module_int <- as.integer(factor(module_colors))
names(module_int) <- valid_modules$AptName

# Replicate membership for each state (WGCNA gives global modules)
groups <- lapply(stage_order, function(s) module_int)
names(groups) <- stage_order

# Build countsL: expression subsets per stage (genes × samples)
countsL <- lapply(stage_order, function(s) {
  sids <- samplesL[[s]]
  expr[names(module_int), sids, drop = FALSE]
})
names(countsL) <- stage_order

cat(sprintf("[BioTIP] Membership vector: %d proteins in %d modules\n",
            length(module_int), length(unique(module_int))))

# --- Score modules using MCI (Module-Criticality Index) ---
MCI_results <- getMCI(groups, countsL, fun = "BioTIP")

# --- Get maximum MCI members (biomodule per state) ---
maxMCI_members <- getMaxMCImember(
  MCI_results[[1]],   # membersL: module membership per state
  MCI_results[[2]],   # MCIl: MCI scores per state
  minsize = cfg$biotip$min_cts_size
)

# --- Get peak-stage statistics ---
maxStats <- getMaxStats(MCI_results[["MCI"]], maxMCI_members[[1]])

# --- Extract Critical Transition Signal (CTS) ---
CTS_list <- getCTS(maxStats, maxMCI_members[[2]])

if (length(CTS_list) == 0 || length(CTS_list[[1]]) == 0) {
  cat("[BioTIP] WARNING: No CTS proteins identified. Check module sizes and sample counts.\n")
  # Save empty results
  dir.create(opt$output, showWarnings = FALSE, recursive = TRUE)
  write.csv(data.frame(AptName = character(0)),
            file.path(opt$output, "biotip_cts_proteins.csv"),
            row.names = FALSE)
  write.csv(data.frame(peak_stage = NA, observed_MCI = NA, empirical_p = NA,
                        n_cts_proteins = 0, n_permutations = 0),
            file.path(opt$output, "biotip_summary.csv"),
            row.names = FALSE)
  cat("[BioTIP] Exiting with no CTS proteins.\n")
  quit(status = 0)
}

cts_proteins <- CTS_list[[1]]
cat(sprintf("[BioTIP] CTS identified: %d proteins\n", length(cts_proteins)))

# --- Permutation testing ---
cat(sprintf("[BioTIP] Running %d permutations for significance testing...\n",
            cfg$biotip$n_permutations))
set.seed(cfg$reproducibility$r_seed)

Ic_random <- simulation_Ic(
  obs.x   = length(cts_proteins),
  sampleL = samplesL,
  counts  = expr,
  B       = cfg$biotip$n_permutations,
  fun     = "BioTIP"
)

# Compute observed Ic score for CTS proteins
observed_Ic <- getIc(
  counts  = expr,
  sampleL = samplesL,
  genes   = cts_proteins,
  fun     = "BioTIP"
)
observed_MCI <- max(observed_Ic, na.rm = TRUE)

empirical_p <- mean(Ic_random >= observed_MCI, na.rm = TRUE)

# Identify peak stage from MCI scores
# MCI_results[["MCI"]] is a list per state with MCI scores per module
mci_per_stage <- sapply(stage_order, function(s) {
  mci_vals <- MCI_results[["MCI"]][[s]]
  if (!is.null(mci_vals) && is.numeric(mci_vals) && length(mci_vals) > 0) {
    max(mci_vals, na.rm = TRUE)
  } else NA
})
peak_stage <- stage_order[which.max(mci_per_stage)]

cat(sprintf("[BioTIP] Peak stage: %s | Observed MCI: %.4f | Empirical p: %.4f\n",
            peak_stage, observed_MCI, empirical_p))

# --- Save outputs ---
dir.create(opt$output, showWarnings = FALSE, recursive = TRUE)

# Stage-level MCI scores
mci_df <- data.frame(
  Stage    = stage_order,
  N        = sapply(stage_order, function(s) length(samplesL[[s]])),
  MCI      = mci_per_stage,
  stringsAsFactors = FALSE
)
write.csv(mci_df,
          file.path(opt$output, "biotip_stage_mci_scores.csv"),
          row.names = FALSE)

# CTS protein list
write.csv(data.frame(AptName = cts_proteins),
          file.path(opt$output, "biotip_cts_proteins.csv"),
          row.names = FALSE)

# Permutation null distribution
write.csv(data.frame(Ic_random = Ic_random),
          file.path(opt$output, "biotip_permutation_null.csv"),
          row.names = FALSE)

# Summary statistics
summary_df <- data.frame(
  peak_stage     = peak_stage,
  observed_MCI   = observed_MCI,
  empirical_p    = empirical_p,
  n_cts_proteins = length(cts_proteins),
  n_permutations = cfg$biotip$n_permutations
)
write.csv(summary_df,
          file.path(opt$output, "biotip_summary.csv"),
          row.names = FALSE)

cat(sprintf("[BioTIP] Complete. CTS = %d proteins. Results in %s\n",
            length(cts_proteins), opt$output))
