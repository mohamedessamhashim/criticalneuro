# R/earlywarnings_analysis.R
# ==========================
# Cross-validates Python CSD analysis using the earlywarnings R package.
# Reads processed ADNI proteomics, computes PC1 per converter participant,
# runs generic_ews() and surrogates(), writes results to CSV.
#
# Usage: Rscript R/earlywarnings_analysis.R

library(yaml)
library(arrow)
library(earlywarnings)

# ---- Load configuration ----
config <- yaml::read_yaml("config/config.yaml")
random_seed <- config$random_seed
set.seed(random_seed)

# ---- Load processed data ----
input_path <- config$paths$adni_clean_parquet
if (!file.exists(input_path)) {
  stop(sprintf("Processed ADNI data not found at %s. Run preprocessing first.", input_path))
}

df <- arrow::read_parquet(input_path)
message(sprintf("Loaded processed ADNI data: %d rows, %d columns", nrow(df), ncol(df)))

# ---- Identify protein columns ----
seq_cols <- grep("^seq\\.", colnames(df), value = TRUE)
message(sprintf("Found %d protein columns", length(seq_cols)))

# ---- Filter to converters ----
converter_label <- config$adni$converter_group
if (!"TRAJECTORY" %in% colnames(df)) {
  stop("TRAJECTORY column not found. Run assign_conversion_labels first.")
}

converters <- df[df$TRAJECTORY == converter_label, ]
converter_rids <- unique(converters$RID)
message(sprintf("Converter participants: %d", length(converter_rids)))

# ---- Configuration ----
primary_window <- config$csd$primary_window
min_visits <- config$adni$min_visits_longitudinal

# ---- Run earlywarnings per participant ----
results <- data.frame()

for (rid in converter_rids) {
  participant <- converters[converters$RID == rid, ]
  participant <- participant[order(participant$EXAMDATE), ]

  if (nrow(participant) < min_visits) {
    next
  }

  # Compute PC1 of protein matrix for this participant
  protein_matrix <- as.matrix(participant[, seq_cols])

  # Remove columns with all NA
  valid_cols <- colSums(!is.na(protein_matrix)) > 0
  protein_matrix <- protein_matrix[, valid_cols]

  if (ncol(protein_matrix) < 2 || nrow(protein_matrix) < min_visits) {
    next
  }

  # Fill NA with column medians for PCA
  for (j in seq_len(ncol(protein_matrix))) {
    na_mask <- is.na(protein_matrix[, j])
    if (any(na_mask)) {
      protein_matrix[na_mask, j] <- median(protein_matrix[!na_mask, j], na.rm = TRUE)
    }
  }

  # PCA - use first component as composite time series
  pca_result <- tryCatch(
    prcomp(protein_matrix, center = TRUE, scale. = TRUE),
    error = function(e) NULL
  )

  if (is.null(pca_result)) {
    next
  }

  pc1_series <- pca_result$x[, 1]

  # earlywarnings expects winsize as fraction of time series length
  winsize_frac <- primary_window / length(pc1_series)
  winsize_frac <- max(winsize_frac, 0.25)  # minimum 25%
  winsize_frac <- min(winsize_frac, 0.75)  # maximum 75%

  # Run generic_ews
  ews_result <- tryCatch(
    generic_ews(pc1_series, winsize = winsize_frac, detrending = "first-diff"),
    error = function(e) NULL
  )

  if (is.null(ews_result)) {
    next
  }

  # Extract Kendall tau trends
  n_ews <- nrow(ews_result)
  if (n_ews < 3) {
    next
  }

  var_tau <- tryCatch(cor.test(seq_len(n_ews), ews_result$ar1, method = "kendall")$estimate, error = function(e) NA)
  ar1_tau <- tryCatch(cor.test(seq_len(n_ews), ews_result$sd, method = "kendall")$estimate, error = function(e) NA)

  results <- rbind(results, data.frame(
    RID = rid,
    n_visits = nrow(participant),
    pc1_var_explained = summary(pca_result)$importance[2, 1],
    r_var_tau = var_tau,
    r_ar1_tau = ar1_tau,
    winsize_frac = winsize_frac,
    stringsAsFactors = FALSE
  ))
}

# ---- Write results ----
output_path <- file.path(config$paths$results_csd, "r_earlywarnings_results.csv")
output_dir <- dirname(output_path)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

write.csv(results, output_path, row.names = FALSE)
message(sprintf("R earlywarnings results written to %s (%d participants)", output_path, nrow(results)))
