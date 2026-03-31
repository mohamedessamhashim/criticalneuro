"""Cross-platform protein harmonisation via UniProt.

Maps both SomaScan SeqIds and Olink assay names to UniProt accessions
to enable cross-platform comparison. Rule 12: cross-platform validation
always uses UniProt accession as the protein identifier.

Usage:
    python src/preprocessing/platform_harmoniser.py --build-overlap
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def build_platform_overlap(
    somascan_map_path: str,
    olink_map_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Build the overlap file between SomaScan and Olink proteins.

    Joins SomaScan and Olink UniProt maps on UniProt accession to
    identify proteins measurable on both platforms.

    Parameters
    ----------
    somascan_map_path : str
        Path to somascan_uniprot_map.csv with columns:
        SeqId, UniProt, EntrezGeneSymbol, TargetFullName.
    olink_map_path : str
        Path to olink_uniprot_map.csv with columns:
        AssayName, UniProt, EntrezGeneSymbol, Panel.
    output_path : str
        Path to write platform_protein_overlap.csv.

    Returns
    -------
    pd.DataFrame
        Overlap DataFrame with columns: UniProt, SeqId, AssayName,
        EntrezGeneSymbol.
    """
    soma_map = pd.read_csv(somascan_map_path)
    olink_map = pd.read_csv(olink_map_path)

    # Standardize UniProt column name
    for col_name in ["UniProt", "Uniprot", "UNIPROT", "uniprot", "UniProt_ID"]:
        if col_name in soma_map.columns:
            soma_map = soma_map.rename(columns={col_name: "UniProt"})
        if col_name in olink_map.columns:
            olink_map = olink_map.rename(columns={col_name: "UniProt"})

    # Clean UniProt IDs — strip whitespace, drop empty
    soma_map["UniProt"] = soma_map["UniProt"].astype(str).str.strip()
    olink_map["UniProt"] = olink_map["UniProt"].astype(str).str.strip()
    soma_map = soma_map[soma_map["UniProt"].str.len() > 0]
    olink_map = olink_map[olink_map["UniProt"].str.len() > 0]
    soma_map = soma_map[soma_map["UniProt"] != "nan"]
    olink_map = olink_map[olink_map["UniProt"] != "nan"]

    # Some UniProt entries may have multiple IDs separated by |
    # Explode to handle multi-mapping
    soma_map["UniProt"] = soma_map["UniProt"].str.split(r"[|;]")
    soma_map = soma_map.explode("UniProt")
    soma_map["UniProt"] = soma_map["UniProt"].str.strip()

    olink_map["UniProt"] = olink_map["UniProt"].str.split(r"[|;]")
    olink_map = olink_map.explode("UniProt")
    olink_map["UniProt"] = olink_map["UniProt"].str.strip()

    # Standardize column names for SeqId and AssayName
    seq_col = None
    for candidate in ["SeqId", "SEQID", "seq_id", "SomaId"]:
        if candidate in soma_map.columns:
            seq_col = candidate
            break
    if seq_col and seq_col != "SeqId":
        soma_map = soma_map.rename(columns={seq_col: "SeqId"})

    assay_col = None
    for candidate in ["AssayName", "ASSAYNAME", "Assay", "OlinkID"]:
        if candidate in olink_map.columns:
            assay_col = candidate
            break
    if assay_col and assay_col != "AssayName":
        olink_map = olink_map.rename(columns={assay_col: "AssayName"})

    # Standardize gene symbol column
    for df in [soma_map, olink_map]:
        for candidate in ["EntrezGeneSymbol", "GeneSymbol", "Gene", "GENE"]:
            if candidate in df.columns and candidate != "EntrezGeneSymbol":
                df.rename(columns={candidate: "EntrezGeneSymbol"}, inplace=True)
                break

    # Inner join on UniProt
    soma_subset = soma_map[["SeqId", "UniProt"]].drop_duplicates()
    olink_subset = olink_map[["AssayName", "UniProt"]].drop_duplicates()

    overlap = soma_subset.merge(olink_subset, on="UniProt", how="inner")

    # Add gene symbol from either source
    if "EntrezGeneSymbol" in soma_map.columns:
        gene_map = soma_map[["UniProt", "EntrezGeneSymbol"]].drop_duplicates()
        overlap = overlap.merge(gene_map, on="UniProt", how="left")
    elif "EntrezGeneSymbol" in olink_map.columns:
        gene_map = olink_map[["UniProt", "EntrezGeneSymbol"]].drop_duplicates()
        overlap = overlap.merge(gene_map, on="UniProt", how="left")

    # Deduplicate by UniProt (keep first mapping if multiple)
    overlap = overlap.drop_duplicates(subset="UniProt")

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(output, index=False)

    logger.info(
        "Platform overlap: %d proteins shared between SomaScan (%d) and Olink (%d)",
        len(overlap),
        soma_map["UniProt"].nunique(),
        olink_map["UniProt"].nunique(),
    )

    return overlap


def get_overlap_proteins(config: dict) -> list[str]:
    """Read pre-built overlap file and return UniProt accession list.

    Parameters
    ----------
    config : dict
        Configuration with paths.platform_protein_overlap.

    Returns
    -------
    list[str]
        UniProt accessions measurable on both platforms.

    Raises
    ------
    FileNotFoundError
        If overlap file does not exist. Instructs user to run
        --build-overlap first.
    """
    overlap_path = Path(config["paths"]["platform_protein_overlap"])

    if not overlap_path.exists():
        raise FileNotFoundError(
            f"Platform overlap file not found at {overlap_path}. "
            "Run: python src/preprocessing/platform_harmoniser.py --build-overlap"
        )

    overlap_df = pd.read_csv(overlap_path)
    proteins = overlap_df["UniProt"].tolist()
    logger.info("Loaded %d overlap proteins from %s", len(proteins), overlap_path)
    return proteins


def map_somascan_to_uniprot(
    df: pd.DataFrame,
    seq_cols: list[str],
    map_path: str,
) -> pd.DataFrame:
    """Add UniProt column to a SomaScan DataFrame.

    Creates a mapping from SeqId column names to UniProt accessions
    and adds a melted/lookup-ready format.

    Parameters
    ----------
    df : pd.DataFrame
        SomaScan DataFrame with seq.* columns.
    seq_cols : list[str]
        List of seq.* column names.
    map_path : str
        Path to somascan_uniprot_map.csv.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with UniProt mapping available.
    """
    soma_map = pd.read_csv(map_path)

    # Standardize column names
    for candidate in ["SeqId", "SEQID", "seq_id", "SomaId"]:
        if candidate in soma_map.columns:
            soma_map = soma_map.rename(columns={candidate: "SeqId"})
            break

    for candidate in ["UniProt", "Uniprot", "UNIPROT"]:
        if candidate in soma_map.columns:
            soma_map = soma_map.rename(columns={candidate: "UniProt"})
            break

    # Create SeqId → UniProt lookup
    lookup = dict(zip(soma_map["SeqId"], soma_map["UniProt"]))

    # Add UniProt mapping as metadata (store as attribute)
    df = df.copy()
    df.attrs["uniprot_map"] = {col: lookup.get(col, None) for col in seq_cols}

    n_mapped = sum(1 for v in df.attrs["uniprot_map"].values() if v is not None)
    logger.info("Mapped %d/%d SomaScan proteins to UniProt", n_mapped, len(seq_cols))

    return df


def map_olink_to_uniprot(
    df: pd.DataFrame,
    npx_cols: list[str],
    map_path: str,
) -> pd.DataFrame:
    """Add UniProt column to an Olink DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Olink DataFrame with NPX_* columns.
    npx_cols : list[str]
        List of NPX_* column names.
    map_path : str
        Path to olink_uniprot_map.csv.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with UniProt mapping available.
    """
    olink_map = pd.read_csv(map_path)

    for candidate in ["AssayName", "ASSAYNAME", "Assay", "OlinkID"]:
        if candidate in olink_map.columns:
            olink_map = olink_map.rename(columns={candidate: "AssayName"})
            break

    for candidate in ["UniProt", "Uniprot", "UNIPROT"]:
        if candidate in olink_map.columns:
            olink_map = olink_map.rename(columns={candidate: "UniProt"})
            break

    lookup = dict(zip(olink_map["AssayName"], olink_map["UniProt"]))

    df = df.copy()
    # Map NPX_PROTEINNAME → strip prefix → lookup
    df.attrs["uniprot_map"] = {}
    for col in npx_cols:
        assay_name = col.replace("NPX_", "")
        df.attrs["uniprot_map"][col] = lookup.get(assay_name, None)

    n_mapped = sum(1 for v in df.attrs["uniprot_map"].values() if v is not None)
    logger.info("Mapped %d/%d Olink proteins to UniProt", n_mapped, len(npx_cols))

    return df


def main():
    """CLI entry point for platform harmonisation."""
    import yaml

    parser = argparse.ArgumentParser(
        description="Cross-platform protein harmonisation"
    )
    parser.add_argument(
        "--build-overlap",
        action="store_true",
        help="Build the platform protein overlap file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.build_overlap:
        build_platform_overlap(
            config["paths"]["somascan_uniprot_map"],
            config["paths"]["olink_uniprot_map"],
            config["paths"]["platform_protein_overlap"],
        )


if __name__ == "__main__":
    main()
