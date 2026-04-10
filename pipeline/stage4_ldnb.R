# ============================================================
# Stage 4c: l-DNB Individual-Level Scoring
# ============================================================
# Uses CTS proteins from BioTIP as the DNB module.
# Computes individual-level l-DNB scores (IDNB) for each sample.
# Plots individual tipping-point trajectories (longitudinal mode).
#
# Called by run_pipeline.py via subprocess
# Args: --config, --expr, --metadata, --cts, --output

suppressPackageStartupMessages({
  library(yaml)
  library(optparse)
  library(ggplot2)
  library(dplyr)
})

# --- Parse arguments ---
option_list <- list(
  make_option("--config",   type = "character"),
  make_option("--expr",     type = "character"),  # residualized matrix CSV
  make_option("--metadata", type = "character"),
  make_option("--cts",      type = "character"),  # CTS proteins from BioTIP
  make_option("--output",   type = "character")
)
opt <- parse_args(OptionParser(option_list = option_list))
cfg <- yaml.load_file(opt$config)
set.seed(cfg$reproducibility$r_seed)

# --- Load inputs ---
expr     <- as.matrix(read.csv(opt$expr, row.names = 1, check.names = FALSE))
metadata <- read.csv(opt$metadata, stringsAsFactors = FALSE)
cts      <- read.csv(opt$cts, stringsAsFactors = FALSE)$AptName

cat(sprintf("[l-DNB] Expression: %d x %d\n", nrow(expr), ncol(expr)))
cat(sprintf("[l-DNB] CTS proteins: %d\n", length(cts)))

# --- Ensure expression is proteins x samples ---
if (!any(grepl("^seq\\.", rownames(expr)))) {
  expr <- t(expr)
}

# Validate CTS proteins are in expression matrix
cts_valid <- intersect(cts, rownames(expr))
if (length(cts_valid) < length(cts)) {
  cat(sprintf("[l-DNB] Warning: %d CTS proteins not in expression matrix. Using %d.\n",
              length(cts) - length(cts_valid), length(cts_valid)))
  cts <- cts_valid
}
if (length(cts) < 5) {
  stop("[l-DNB] Fewer than 5 valid CTS proteins. Cannot compute l-DNB.")
}

# --- Add stage labels to metadata ---
if (cfg$analysis_mode == "longitudinal") {
  stage_map <- cfg$longitudinal$stage_definitions
  metadata$Stage <- sapply(metadata$VisitsToDx, function(v) {
    if (is.na(v)) return("stable_CO")
    key <- as.character(v)
    if (key %in% names(stage_map)) stage_map[[key]] else "stable_CO"
  })
}

# --- Define reference population ---
ref_stage <- cfg$ldnb$reference_stage
if (cfg$analysis_mode == "longitudinal") {
  ref_ids <- metadata$SampleID[!is.na(metadata$Stage) & metadata$Stage == ref_stage]
} else {
  ref_ids <- metadata$SampleID[metadata$TRAJECTORY == "CN_amyloid_negative"]
}
ref_ids <- intersect(as.character(ref_ids), colnames(expr))
cat(sprintf("[l-DNB] Reference population: %d samples from '%s'\n",
            length(ref_ids), ref_stage))

if (length(ref_ids) < 3) {
  stop(sprintf("[l-DNB] Only %d reference samples — need at least 3.", length(ref_ids)))
}

# --- Compute reference statistics (on CTS proteins only) ---
ref_mat   <- expr[cts, ref_ids, drop = FALSE]
ref_means <- rowMeans(ref_mat, na.rm = TRUE)
ref_sds   <- apply(ref_mat, 1, sd, na.rm = TRUE)
# Replace zero SDs with small value to avoid division by zero
ref_sds[ref_sds < 1e-6] <- 1e-6

# --- l-DNB score per sample ---
top_k <- min(cfg$ldnb$top_k, length(cts))
all_sample_ids <- colnames(expr)
cat(sprintf("[l-DNB] Computing IDNB for %d samples (top_k=%d)...\n",
            length(all_sample_ids), top_k))

idnb_scores <- numeric(length(all_sample_ids))
names(idnb_scores) <- all_sample_ids

for (j in seq_along(all_sample_ids)) {
  sid <- all_sample_ids[j]
  test_vec <- expr[cts, sid]

  # Z-scores relative to reference
  z_scores <- (test_vec - ref_means) / ref_sds

  # IDNB: mean of top-k |z-scores|
  local_scores <- abs(z_scores)
  k <- min(top_k, length(local_scores))
  top_idx <- order(local_scores, decreasing = TRUE)[1:k]
  idnb_scores[j] <- mean(local_scores[top_idx], na.rm = TRUE)
}

# --- Merge with metadata ---
scores_df <- data.frame(
  SampleID = all_sample_ids,
  IDNB     = idnb_scores,
  stringsAsFactors = FALSE
)
scores_df <- merge(scores_df, metadata, by = "SampleID", all.x = TRUE)

# Z-score IDNB within each stage for comparability
if (cfg$analysis_mode == "longitudinal" && "Stage" %in% colnames(scores_df)) {
  scores_df <- scores_df %>%
    group_by(Stage) %>%
    mutate(IDNB_zscore = as.numeric(scale(IDNB))) %>%
    ungroup()
}

# --- Save outputs ---
dir.create(opt$output, showWarnings = FALSE, recursive = TRUE)

write.csv(scores_df,
          file.path(opt$output, "ldnb_individual_scores.csv"),
          row.names = FALSE)

# --- Figure: individual trajectories (longitudinal only) ---
figures_dir <- cfg$output$figures_dir
if (!is.null(figures_dir)) {
  dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)
}

if (cfg$analysis_mode == "longitudinal" && "VisitsToDx" %in% colnames(scores_df)) {
  converter_scores <- scores_df %>%
    filter(!is.na(VisitsToDx)) %>%
    mutate(VisitsToDx = as.numeric(VisitsToDx))

  if (nrow(converter_scores) > 0 && !is.null(figures_dir)) {
    p_traj <- ggplot(converter_scores,
                     aes(x = -VisitsToDx, y = IDNB, group = SubjectID)) +
      geom_line(alpha = 0.4, color = "steelblue") +
      geom_point(alpha = 0.6, color = "steelblue") +
      stat_summary(aes(group = 1), fun = mean, geom = "line",
                   color = "red", linewidth = 1.5) +
      scale_x_continuous(
        breaks = c(-3, -2, -1, 0),
        labels = c("T-3\n(Baseline)", "T-2", "T-1\n(Last CO)", "T0\n(AD Dx)")
      ) +
      labs(
        title    = "Individual l-DNB Trajectories: CO->AD Converters",
        subtitle = sprintf("n=%d converters | CTS proteins = %d",
                           length(unique(converter_scores$SubjectID)),
                           length(cts)),
        x = "Visits Before AD Diagnosis",
        y = "IDNB Score (network instability)"
      ) +
      theme_classic(base_size = 14)

    ggsave(file.path(figures_dir, "ldnb_individual_trajectories.pdf"),
           p_traj, width = 8, height = 5)
    ggsave(file.path(figures_dir, "ldnb_individual_trajectories.png"),
           p_traj, width = 8, height = 5, dpi = 300)
    cat("[l-DNB] Trajectory figure saved\n")
  }
}

# --- Print summary ---
cat(sprintf("[l-DNB] Complete. Mean IDNB by stage:\n"))
if ("Stage" %in% colnames(scores_df)) {
  stage_summary <- scores_df %>%
    group_by(Stage) %>%
    summarise(mean_IDNB = mean(IDNB, na.rm = TRUE), n = n(), .groups = "drop")
  print(as.data.frame(stage_summary))
}
