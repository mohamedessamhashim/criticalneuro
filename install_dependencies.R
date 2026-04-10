#!/usr/bin/env Rscript
# ============================================================
# Install R Dependencies for CriticalNeuroMap
# ============================================================
# Run once: Rscript install_dependencies.R

cat("Installing R dependencies for CriticalNeuroMap...\n\n")

# CRAN packages
cran_pkgs <- c("WGCNA", "lme4", "lmerTest", "ggplot2",
               "dplyr", "tidyr", "yaml", "optparse",
               "BiocManager")

for (pkg in cran_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing %s from CRAN...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  } else {
    cat(sprintf("  %s already installed\n", pkg))
  }
}

# Bioconductor packages
bioc_pkgs <- c("BioTIP", "limma", "sva")
for (pkg in bioc_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing %s from Bioconductor...\n", pkg))
    BiocManager::install(pkg, update = FALSE, ask = FALSE)
  } else {
    cat(sprintf("  %s already installed\n", pkg))
  }
}

cat("\nAll R dependencies installed.\n")
cat("Verify with: Rscript -e 'library(WGCNA); library(BioTIP); library(lme4)'\n")
