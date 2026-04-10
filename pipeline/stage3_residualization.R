# ============================================================
# Stage 3a: Covariate Residualization
# ============================================================
# Cross-sectional: OLS residualization (Age, Sex, APOE4)
# Longitudinal: lme4 mixed model with random intercept per subject
#
# Called by run_pipeline.py via subprocess
# Args: --config, --input, --metadata, --output

suppressPackageStartupMessages({
  library(yaml)
  library(optparse)
})

# --- Parse arguments ---
option_list <- list(
  make_option("--config",   type = "character"),
  make_option("--input",    type = "character"),  # expression matrix CSV (samples x proteins)
  make_option("--metadata", type = "character"),  # metadata CSV
  make_option("--output",   type = "character")   # output directory
)
opt <- parse_args(OptionParser(option_list = option_list))
cfg <- yaml.load_file(opt$config)

# --- Set seed ---
set.seed(cfg$reproducibility$r_seed)

# --- Load data ---
expr <- read.csv(opt$input, row.names = 1, check.names = FALSE)
metadata <- read.csv(opt$metadata, stringsAsFactors = FALSE)

cat(sprintf("[Residualize] Loaded expression: %d samples x %d proteins\n",
            nrow(expr), ncol(expr)))
cat(sprintf("[Residualize] Loaded metadata: %d rows\n", nrow(metadata)))

# --- Match metadata to expression ---
# Ensure row order matches
common_ids <- intersect(rownames(expr), as.character(metadata$SampleID))
if (length(common_ids) == 0) {
  stop("[Residualize] No matching SampleIDs between expression and metadata")
}
expr <- expr[common_ids, , drop = FALSE]
metadata <- metadata[match(common_ids, as.character(metadata$SampleID)), ]
cat(sprintf("[Residualize] Matched %d samples\n", length(common_ids)))

# Identify protein columns
seq_cols <- grep("^seq\\.", colnames(expr), value = TRUE)
if (length(seq_cols) == 0) {
  # Fall back to all numeric columns
  seq_cols <- colnames(expr)[sapply(expr, is.numeric)]
}
cat(sprintf("[Residualize] Found %d protein columns\n", length(seq_cols)))

# --- Mode-specific residualization ---
if (cfg$analysis_mode == "longitudinal") {
  cat("[Residualize] Running longitudinal residualization (lme4 mixed models)\n")

  suppressPackageStartupMessages(library(lme4))

  covars <- cfg$residualization$covariates_longitudinal
  re_var <- cfg$residualization$longitudinal_random_effect

  # Validate covariates exist
  missing_covars <- setdiff(covars, colnames(metadata))
  if (length(missing_covars) > 0) {
    stop(sprintf("[Residualize] Missing covariates in metadata: %s",
                 paste(missing_covars, collapse = ", ")))
  }
  if (!(re_var %in% colnames(metadata))) {
    stop(sprintf("[Residualize] Random effect variable '%s' not in metadata", re_var))
  }

  # Build formula
  fixed_terms <- paste(covars, collapse = " + ")
  formula_str <- sprintf("y ~ %s + (1|%s)", fixed_terms, re_var)
  cat(sprintf("[Residualize] Formula: %s\n", formula_str))

  residuals_mat <- matrix(NA, nrow = nrow(expr), ncol = length(seq_cols),
                           dimnames = list(rownames(expr), seq_cols))

  n_proteins <- length(seq_cols)
  n_lme_success <- 0
  n_ols_fallback <- 0

  for (i in seq_len(n_proteins)) {
    if (i %% 500 == 0) {
      cat(sprintf("  ... %d / %d proteins\n", i, n_proteins))
    }

    fit_data <- data.frame(y = as.numeric(expr[, seq_cols[i]]), metadata,
                            check.names = FALSE)

    tryCatch({
      fit <- lmer(as.formula(formula_str), data = fit_data,
                   REML = FALSE,
                   control = lmerControl(optimizer = "bobyqa",
                                          optCtrl = list(maxfun = 1e5)))
      residuals_mat[, i] <- residuals(fit)
      n_lme_success <- n_lme_success + 1
    }, error = function(e) {
      # If lme4 fails, fall back to OLS
      ols_formula <- as.formula(sub(" \\+ \\(1\\|.*\\)", "", formula_str))
      fit_ols <- lm(ols_formula, data = fit_data)
      residuals_mat[, i] <<- residuals(fit_ols)
      n_ols_fallback <<- n_ols_fallback + 1
    })
  }

  cat(sprintf("[Residualize] Complete: %d lme4, %d OLS fallback\n",
              n_lme_success, n_ols_fallback))

  # Replace expression values with residuals
  expr[, seq_cols] <- residuals_mat

} else {
  cat("[Residualize] Running cross-sectional residualization (OLS)\n")

  covars <- cfg$residualization$covariates_cross_sectional

  # Validate covariates exist
  missing_covars <- setdiff(covars, colnames(metadata))
  if (length(missing_covars) > 0) {
    cat(sprintf("[Residualize] Warning: Missing covariates: %s. Using available ones.\n",
                paste(missing_covars, collapse = ", ")))
    covars <- intersect(covars, colnames(metadata))
  }

  if (length(covars) == 0) {
    cat("[Residualize] No covariates available — skipping residualization\n")
  } else {
    fixed_terms <- paste(covars, collapse = " + ")
    formula_str <- sprintf("y ~ %s", fixed_terms)
    cat(sprintf("[Residualize] Formula: %s\n", formula_str))

    n_proteins <- length(seq_cols)
    for (i in seq_len(n_proteins)) {
      if (i %% 500 == 0) {
        cat(sprintf("  ... %d / %d proteins\n", i, n_proteins))
      }

      fit_data <- data.frame(y = as.numeric(expr[, seq_cols[i]]), metadata,
                              check.names = FALSE)

      tryCatch({
        fit <- lm(as.formula(formula_str), data = fit_data)
        expr[, seq_cols[i]] <- residuals(fit)
      }, error = function(e) {
        # Leave unchanged if model fails
      })
    }

    cat("[Residualize] Cross-sectional residualization complete\n")
  }
}

# --- Save output ---
dir.create(opt$output, showWarnings = FALSE, recursive = TRUE)
output_path <- file.path(opt$output, "expression_residualized.csv")
write.csv(expr, output_path)
cat(sprintf("[Residualize] Output saved to %s\n", output_path))
