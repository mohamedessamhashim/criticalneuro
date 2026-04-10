# ============================================================
# Longitudinal Utility Functions
# ============================================================
# Functions for aligning samples by visits-before-conversion
# and validating converter trajectories.

suppressPackageStartupMessages({
  library(dplyr)
})


#' Align samples by visits-before-conversion
#'
#' For each converter, find the visit where AD diagnosis occurs,
#' then count backwards to assign VisitsToDx.
#'
#' @param metadata data.frame with SubjectID, VisitNumber, Diagnosis, Converter
#' @param cfg config list with longitudinal$stage_definitions
#' @return metadata with VisitsToDx and Stage columns added
align_by_conversion <- function(metadata, cfg) {
  metadata <- metadata %>%
    group_by(SubjectID) %>%
    mutate(
      AD_visit = ifelse(any(Diagnosis == "AD"),
                        min(VisitNumber[Diagnosis == "AD"]),
                        NA),
      VisitsToDx = ifelse(!is.na(AD_visit),
                           AD_visit - VisitNumber,
                           NA)
    ) %>%
    ungroup()

  # Apply stage labels from config
  stage_map <- cfg$longitudinal$stage_definitions
  metadata$Stage <- sapply(metadata$VisitsToDx, function(v) {
    if (is.na(v)) return("stable_CO")
    key <- as.character(v)
    if (key %in% names(stage_map)) stage_map[[key]] else "stable_CO"
  })

  return(metadata)
}


#' Validate converter trajectories
#'
#' Filter to subjects matching required trajectory patterns
#' (e.g., "CO->CO->CO->AD").
#'
#' @param metadata data.frame with aligned longitudinal data
#' @param required_trajectories character vector of required trajectory patterns
#' @return data.frame of valid converters + all non-converters
validate_trajectories <- function(metadata, required_trajectories) {
  # Build trajectory string per subject
  subject_trajectories <- metadata %>%
    group_by(SubjectID) %>%
    arrange(VisitNumber) %>%
    summarise(
      Trajectory = paste(Diagnosis, collapse = "->"),
      N_Visits   = n(),
      .groups    = "drop"
    )

  valid_subjects <- subject_trajectories %>%
    filter(Trajectory %in% required_trajectories) %>%
    pull(SubjectID)

  cat(sprintf("[Longitudinal] Valid converters: %d subjects\n",
              length(valid_subjects)))
  cat("[Longitudinal] Trajectories found:\n")
  traj_table <- table(subject_trajectories$Trajectory[
    subject_trajectories$SubjectID %in% valid_subjects
  ])
  print(traj_table)

  # Keep valid converters + all non-converters
  is_converter <- metadata$Converter %in% c(TRUE, "TRUE", "true", 1)
  result <- metadata[metadata$SubjectID %in% valid_subjects | !is_converter, ]

  return(result)
}


#' Summarize longitudinal staging
#'
#' Print a summary table of samples per stage and per trajectory.
#'
#' @param metadata data.frame with Stage and SubjectID columns
#' @return invisible NULL (prints summary)
summarize_staging <- function(metadata) {
  cat("\n=== Longitudinal Staging Summary ===\n")

  # Samples per stage
  stage_counts <- metadata %>%
    group_by(Stage) %>%
    summarise(
      n_samples  = n(),
      n_subjects = n_distinct(SubjectID),
      .groups    = "drop"
    ) %>%
    arrange(desc(n_samples))

  cat("\nSamples per stage:\n")
  print(as.data.frame(stage_counts))

  # Converter status
  if ("Converter" %in% colnames(metadata)) {
    converter_counts <- metadata %>%
      group_by(Converter) %>%
      summarise(
        n_subjects = n_distinct(SubjectID),
        n_samples  = n(),
        .groups    = "drop"
      )
    cat("\nConverter status:\n")
    print(as.data.frame(converter_counts))
  }

  invisible(NULL)
}
