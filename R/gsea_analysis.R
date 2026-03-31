# R/gsea_analysis.R
# ==================
# Gene Set Enrichment Analysis on DNB core proteins.
# Uses fgsea with MSigDB Hallmark and Reactome pathways.
#
# Usage: Rscript R/gsea_analysis.R

library(yaml)
library(fgsea)

# ---- Load configuration ----
config <- yaml::read_yaml("config/config.yaml")
set.seed(config$random_seed)

# ---- Load DNB core proteins ----
core_proteins_path <- file.path(config$paths$results_dnb, "dnb_core_proteins.csv")
if (!file.exists(core_proteins_path)) {
  stop(sprintf("DNB core proteins not found at %s. Run DNB analysis first.", core_proteins_path))
}

core_proteins <- read.csv(core_proteins_path, stringsAsFactors = FALSE)
message(sprintf("Loaded %d DNB core proteins", nrow(core_proteins)))

# ---- Load UniProt mapping ----
uniprot_map_path <- config$paths$somascan_uniprot_map
if (!file.exists(uniprot_map_path)) {
  stop(sprintf("SomaScan-UniProt map not found at %s", uniprot_map_path))
}

uniprot_map <- read.csv(uniprot_map_path, stringsAsFactors = FALSE)
message(sprintf("Loaded UniProt mapping: %d entries", nrow(uniprot_map)))

# Map protein IDs to gene symbols
# Expect columns: SeqId, EntrezGeneSymbol
id_col <- intersect(c("SeqId", "SEQID", "seq_id"), colnames(uniprot_map))[1]
gene_col <- intersect(c("EntrezGeneSymbol", "GeneSymbol", "GENE"), colnames(uniprot_map))[1]

if (is.na(id_col) || is.na(gene_col)) {
  stop("Cannot find expected columns in UniProt mapping file")
}

# Create ranking vector: gene symbol -> frequency score
merged <- merge(core_proteins, uniprot_map, by.x = "protein", by.y = id_col, all.x = TRUE)
gene_ranks <- setNames(merged$frequency, merged[[gene_col]])
gene_ranks <- gene_ranks[!is.na(names(gene_ranks)) & names(gene_ranks) != ""]
gene_ranks <- sort(gene_ranks, decreasing = TRUE)
message(sprintf("Mapped %d proteins to gene symbols for GSEA", length(gene_ranks)))

# ---- Load gene sets ----
load_gmt <- function(path) {
  if (!file.exists(path)) {
    message(sprintf("GMT file not found: %s", path))
    return(list())
  }
  fgsea::gmtPathways(path)
}

hallmark_sets <- load_gmt(config$paths$msigdb_hallmark)
reactome_sets <- load_gmt(config$paths$reactome_pathways)

message(sprintf("Gene sets: %d Hallmark, %d Reactome", length(hallmark_sets), length(reactome_sets)))

# ---- Run fgsea ----
output_dir <- file.path(config$paths$results_dnb, "gsea_results")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

run_gsea <- function(pathways, name) {
  if (length(pathways) == 0 || length(gene_ranks) < 10) {
    message(sprintf("Skipping %s GSEA: insufficient data", name))
    return(NULL)
  }

  result <- tryCatch(
    fgsea(
      pathways = pathways,
      stats = gene_ranks,
      minSize = 15,
      maxSize = 500,
      nPermSimple = 10000
    ),
    error = function(e) {
      message(sprintf("fgsea failed for %s: %s", name, e$message))
      return(NULL)
    }
  )

  if (!is.null(result)) {
    # BH-FDR correction
    result$fdr <- p.adjust(result$pval, method = "BH")
    result <- result[order(result$fdr), ]

    # Write results
    output_path <- file.path(output_dir, sprintf("gsea_%s.csv", tolower(name)))
    # Convert leadingEdge list column to string for CSV
    result$leadingEdge_str <- sapply(result$leadingEdge, function(x) paste(x, collapse = ";"))
    write.csv(result[, c("pathway", "pval", "padj", "fdr", "NES", "size", "leadingEdge_str")],
              output_path, row.names = FALSE)
    message(sprintf("%s GSEA: %d pathways tested, %d significant (FDR < 0.25)",
                    name, nrow(result), sum(result$fdr < 0.25)))
  }

  return(result)
}

hallmark_results <- run_gsea(hallmark_sets, "Hallmark")
reactome_results <- run_gsea(reactome_sets, "Reactome")

message("GSEA analysis complete")
