# ============================================================
# SomaScan Utility Functions
# ============================================================
# Helpers for SeqId mapping, protein annotation, and
# SomaScan-specific data handling.


#' Identify SomaScan protein columns (seq.* prefix)
#'
#' @param df data.frame or matrix
#' @return character vector of column names matching "seq.*" pattern
identify_seq_columns <- function(df) {
  grep("^seq\\.", colnames(df), value = TRUE)
}


#' Map SeqIds to gene symbols using UniProt map
#'
#' @param seq_ids character vector of SeqIds (e.g. "seq.1234.56")
#' @param map_path path to somascan_uniprot_map.csv
#' @return data.frame with columns: SeqId, UniProt, EntrezGeneSymbol, TargetFullName
map_seqid_to_gene <- function(seq_ids, map_path) {
  if (!file.exists(map_path)) {
    warning(sprintf("UniProt map not found at %s", map_path))
    return(data.frame(
      SeqId = seq_ids,
      UniProt = NA_character_,
      EntrezGeneSymbol = NA_character_,
      TargetFullName = NA_character_,
      stringsAsFactors = FALSE
    ))
  }

  map_df <- read.csv(map_path, stringsAsFactors = FALSE)

  # Convert R-export format (X10000.28) to seq.* format
  if ("Analytes" %in% colnames(map_df)) {
    map_df$SeqId <- paste0("seq.", sub("^X", "", map_df$Analytes))
  }

  result <- data.frame(SeqId = seq_ids, stringsAsFactors = FALSE)
  result <- merge(result, map_df[, c("SeqId", "UniProt", "EntrezGeneSymbol", "TargetFullName")],
                   by = "SeqId", all.x = TRUE, sort = FALSE)

  return(result)
}


#' Validate SomaScan expression matrix format
#'
#' Checks that the expression matrix has the expected structure.
#'
#' @param expr data.frame or matrix
#' @return list with n_samples, n_proteins, seq_cols, and any warnings
validate_somascan_matrix <- function(expr) {
  seq_cols <- identify_seq_columns(expr)

  result <- list(
    n_samples  = nrow(expr),
    n_proteins = length(seq_cols),
    seq_cols   = seq_cols,
    warnings   = character(0)
  )

  if (length(seq_cols) == 0) {
    result$warnings <- c(result$warnings,
      "No seq.* columns found. Ensure column names follow 'seq.XXXX.XX' format.")
  }

  if (length(seq_cols) < 100) {
    result$warnings <- c(result$warnings,
      sprintf("Only %d protein columns found (expected ~5000-7000 for SomaScan v4.1)", length(seq_cols)))
  }

  # Check for excessive NAs
  if (length(seq_cols) > 0) {
    na_frac <- mean(is.na(as.matrix(expr[, seq_cols])))
    if (na_frac > 0.1) {
      result$warnings <- c(result$warnings,
        sprintf("%.1f%% of values are NA — check data quality", na_frac * 100))
    }
  }

  return(result)
}
