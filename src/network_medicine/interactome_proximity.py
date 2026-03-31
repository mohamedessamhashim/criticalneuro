"""
Interactome proximity analysis of DNB core proteins.
Uses NetMedPy (Aldana et al., Bioinformatics 2025).

Run: python src/network_medicine/interactome_proximity.py

Inputs:
    data/results/dnb/somascan/dnb_core_proteins.csv
    data/results/ppmi/somascan/dnb_core_proteins.csv
    data/reference/interactome_menche2015.tsv  (or .pkl)

Outputs:
    data/results/network_medicine/lcc_results.csv
    data/results/network_medicine/proximity_results.csv
    data/results/network_medicine/cross_disease_proximity.csv
    data/results/network_medicine/interactome_summary.csv
"""

import logging
import pathlib
import pickle
import pandas as pd
import networkx as nx
import netmedpy
from netmedpy.NetMedPy import (_degree_match_null_model,
                                _sample_preserving_degrees)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('network_medicine')

# ── Hardcoded fallback lists (verified from paper) ────────────────────────────
ADNI_DNB_PROTEINS = [
    'HPX', 'CHGA', 'C1QA', 'C1QB', 'C1QC',
    'PARK7', 'F2', 'PRDX6', 'CRYBB2', 'CUL3',
    'CA1', 'PGD', 'HSP90AB1'
]
PPMI_DNB_PROTEINS = ['CRYBB2']

N_ITER       = 1000   # permutations for null model
NULL_MODEL   = 'log_binning'   # degree-log-binning (Guney et al. 2016 standard)
RANDOM_SEED  = 42


def load_interactome():
    """Load human PPI network. Tries pkl first (fast), then tsv."""
    pkl_path = pathlib.Path('data/reference/interactome_ppi.pkl')
    tsv_path = pathlib.Path('data/reference/interactome_menche2015.tsv')

    if pkl_path.exists():
        logger.info(f"Loading interactome from {pkl_path}")
        with open(pkl_path, 'rb') as f:
            G = pickle.load(f)
    elif tsv_path.exists():
        logger.info(f"Loading interactome from {tsv_path}")
        df = pd.read_csv(tsv_path, sep='\t', header=None,
                         names=['protein_a', 'protein_b'])
        G = nx.from_pandas_edgelist(df, 'protein_a', 'protein_b')
        G.remove_edges_from(nx.selfloop_edges(G))
        # Cache as pkl for faster future loads
        with open(pkl_path, 'wb') as f:
            pickle.dump(G, f)
        logger.info(f"Saved interactome pkl to {pkl_path}")
    else:
        raise FileNotFoundError(
            "Interactome not found. Download the human PPI network and save as "
            "data/reference/interactome_ppi.pkl (NetworkX Graph) or "
            "data/reference/interactome_menche2015.tsv (two-column gene-symbol TSV)."
        )

    logger.info(f"Interactome: {G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges")
    return G


def load_proteins():
    """Load DNB core proteins from CSVs. Falls back to hardcoded lists."""
    adni_proteins = ADNI_DNB_PROTEINS.copy()
    ppmi_proteins = PPMI_DNB_PROTEINS.copy()

    _GENE_COLS = ['gene_symbol', 'EntrezGeneSymbol', 'gene', 'symbol', 'protein_id']

    _GENE_COLS = ['gene_symbol', 'EntrezGeneSymbol', 'gene', 'symbol', 'protein_id']

    def _parse_genes_from_csv(csv_path, label):
        """Load gene symbols from CSV, splitting pipe-delimited multi-gene entries."""
        df = pd.read_csv(csv_path)
        col = next((c for c in _GENE_COLS if c in df.columns), None)
        if col is None:
            logger.warning(f"No gene symbol column found in {csv_path} — using hardcoded list")
            return None
        raw = df[col].dropna().tolist()
        # Split pipe-delimited multi-gene entries (e.g. 'C1QA|C1QB|C1QC')
        genes = []
        for entry in raw:
            genes.extend(str(entry).split('|'))
        # Deduplicate while preserving order
        seen = set()
        unique = [g for g in genes if not (g in seen or seen.add(g))]
        logger.info(f"Loaded {len(unique)} {label} proteins from CSV (col={col}, raw={len(raw)})")
        return unique

    adni_csv = pathlib.Path('data/results/dnb/somascan/dnb_core_proteins.csv')
    if adni_csv.exists():
        result = _parse_genes_from_csv(adni_csv, 'ADNI')
        if result is not None:
            adni_proteins = result
    else:
        logger.warning(f"{adni_csv} missing — using hardcoded list")

    ppmi_csv = pathlib.Path('data/results/ppmi/somascan/dnb_core_proteins.csv')
    if ppmi_csv.exists():
        result = _parse_genes_from_csv(ppmi_csv, 'PPMI')
        if result is not None:
            ppmi_proteins = result
    else:
        logger.warning(f"{ppmi_csv} missing — using hardcoded list")

    return adni_proteins, ppmi_proteins


def filter_to_interactome(proteins, G, label=''):
    """Keep only proteins present in interactome nodes. Log dropped proteins."""
    in_net  = [p for p in proteins if p in G.nodes()]
    dropped = [p for p in proteins if p not in G.nodes()]
    if dropped:
        logger.warning(f"[{label}] Proteins not in interactome: {dropped}")
    logger.info(f"[{label}] {len(in_net)}/{len(proteins)} proteins in interactome")
    return in_net


def get_distance_matrix(G):
    """
    Compute or load all-pair shortest path distance matrix.
    This takes 15-30 min on M2 — cached to pkl after first run.
    """
    dist_pkl = pathlib.Path('data/reference/interactome_dist_matrix.pkl')

    if dist_pkl.exists():
        logger.info(f"Loading cached distance matrix from {dist_pkl}")
        with open(dist_pkl, 'rb') as f:
            dist_matrix = pickle.load(f)
    else:
        logger.info("Computing all-pair distance matrix (15-30 min)...")
        dist_matrix = netmedpy.all_pair_distances(G, method='shortest_path')
        with open(dist_pkl, 'wb') as f:
            pickle.dump(dist_matrix, f)
        logger.info(f"Distance matrix saved to {dist_pkl}")

    return dist_matrix


def run_lcc(G, proteins, label='ADNI'):
    """LCC significance test — are proteins more connected than random?"""
    logger.info(f"LCC analysis: {label} ({len(proteins)} proteins, "
                f"n_iter={N_ITER})")
    result = netmedpy.lcc_significance(
        G, proteins,
        n_iter=N_ITER,
        null_model='degree_match'
    )
    logger.info(f"LCC [{label}]: size={result['lcc_size']}, "
                f"expected={result['d_mu']:.1f}±{result['d_sigma']:.1f}, "
                f"Z={result['z_score']:.3f}, p={result['p_val']:.4f}")
    return result


def _mean_pairwise_distance(proteins, dist_matrix):
    """Mean of all pairwise shortest paths within a group, excluding diagonal."""
    total, count, unreachable = 0.0, 0, 0
    for i, a in enumerate(proteins):
        for j, b in enumerate(proteins):
            if i == j:
                continue
            d = dist_matrix.get(a, b)
            if d == float('inf') or d != d:  # inf or NaN
                unreachable += 1
            else:
                total += d
                count += 1
    if unreachable:
        logger.warning(f"{unreachable} unreachable pairs in within-group distance")
    return total / count if count > 0 else float('inf')


def run_within_proximity(G, proteins, dist_matrix, n_iter=N_ITER):
    """
    Within-group mean pairwise distance vs. degree-matched null.
    Replaces the buggy run_proximity(A, A) call where AMSPL collapses to 0.
    """
    logger.info(f"Within-group proximity: {len(proteins)} proteins, "
                f"n_iter={n_iter}")
    d_actual = _mean_pairwise_distance(proteins, dist_matrix)
    logger.info(f"Within-group d_AA (actual) = {d_actual:.4f}")

    # Build degree-matched null distribution using NetMedPy's internal utilities
    bucket = _degree_match_null_model(G)
    null_distances = []
    for _ in range(n_iter):
        rand_set = list(_sample_preserving_degrees(G, proteins, bucket))
        null_distances.append(_mean_pairwise_distance(rand_set, dist_matrix))

    import numpy as np
    d_mu    = float(np.mean(null_distances))
    d_sigma = float(np.std(null_distances))
    z_score = (d_actual - d_mu) / d_sigma if d_sigma > 0 else 0.0
    # One-tailed p: fraction of null sets closer (smaller d) than actual
    p_value = sum(1 for d in null_distances if d <= d_actual) / n_iter

    logger.info(f"Within-group proximity: d_actual={d_actual:.4f}, "
                f"null={d_mu:.4f}±{d_sigma:.4f}, "
                f"Z={z_score:.3f}, p={p_value:.4f}")
    return {
        'd_aa_actual': d_actual,
        'd_mu':        d_mu,
        'd_sigma':     d_sigma,
        'z_score':     z_score,
        'p_value':     p_value,
        'dist':        null_distances,
    }


def run_proximity(G, proteins_a, proteins_b, dist_matrix, label=''):
    """
    Network proximity: are proteins_a closer to proteins_b than random?
    Uses degree-log-binning null model (Guney et al. 2016 standard).
    """
    logger.info(f"Proximity [{label}]: {len(proteins_a)} x {len(proteins_b)} "
                f"proteins, n_iter={N_ITER}, null={NULL_MODEL}")
    result = netmedpy.proximity(
        G,
        proteins_a,
        proteins_b,
        dist_matrix,
        n_iter=N_ITER,
        null_model=NULL_MODEL
    )
    logger.info(f"Proximity [{label}]: Z={result['z_score']:.3f}, "
                f"p={result['p_value_single_tail']:.4f}, "
                f"d_actual={result['raw_amspl']:.4f}, "
                f"d_random={result['d_mu']:.4f}±{result['d_sigma']:.4f}")
    return result


def save_results(lcc_adni, prox_adni, prox_cross,
                 adni_proteins, ppmi_proteins, G):
    """Write all results to CSV. Never overwrite existing CSVs without --force."""
    out = pathlib.Path('data/results/network_medicine')
    out.mkdir(parents=True, exist_ok=True)

    # ── LCC results ──────────────────────────────────────────────────────────
    lcc_df = pd.DataFrame([{
        'cohort':           'ADNI',
        'disease':          'Alzheimer Disease',
        'n_input_proteins': len(ADNI_DNB_PROTEINS),
        'n_in_interactome': len(adni_proteins),
        'lcc_size':         lcc_adni['lcc_size'],
        'lcc_mean':         round(lcc_adni['d_mu'], 2),
        'lcc_std':          round(lcc_adni['d_sigma'], 2),
        'z_score':          round(lcc_adni['z_score'], 3),
        'p_value':          round(lcc_adni['p_val'], 4),
        'significant':      lcc_adni['p_val'] < 0.05,
        'null_model':       'degree_match',
        'n_iter':           N_ITER,
    }])
    lcc_df.to_csv(out / 'lcc_results.csv', index=False)
    logger.info(f"Saved {out / 'lcc_results.csv'}")

    # ── Proximity results ─────────────────────────────────────────────────────
    prox_df = pd.DataFrame([
        {
            'comparison':       'ADNI_self',
            'label':            'ADNI DNB proteins vs. themselves',
            'set_a':            'ADNI DNB core proteins',
            'set_b':            'ADNI DNB core proteins',
            'n_a':              len(adni_proteins),
            'n_b':              len(adni_proteins),
            'amspl_actual':     round(prox_adni['d_aa_actual'], 4),
            'd_random_mean':    round(prox_adni['d_mu'], 4),
            'd_random_std':     round(prox_adni['d_sigma'], 4),
            'z_score':          round(prox_adni['z_score'], 3),
            'p_value':          round(prox_adni['p_value'], 4),
            'significant':      prox_adni['p_value'] < 0.05,
            'null_model':       NULL_MODEL,
            'n_iter':           N_ITER,
            'interpretation':   'DNB proteins closer to each other than random'
                                if prox_adni['z_score'] < -1.5 else
                                'DNB proteins not significantly close',
        },
        {
            'comparison':       'AD_PD_cross',
            'label':            'ADNI DNB (AD) vs. PPMI DNB (PD)',
            'set_a':            'ADNI DNB core proteins',
            'set_b':            'PPMI DNB core proteins (CRYBB2)',
            'n_a':              len(adni_proteins),
            'n_b':              len([p for p in PPMI_DNB_PROTEINS
                                     if p in G.nodes()]),
            'amspl_actual':     round(prox_cross['raw_amspl'], 4),
            'd_random_mean':    round(prox_cross['d_mu'], 4),
            'd_random_std':     round(prox_cross['d_sigma'], 4),
            'z_score':          round(prox_cross['z_score'], 3),
            'p_value':          round(prox_cross['p_value_single_tail'], 4),
            'significant':      prox_cross['p_value_single_tail'] < 0.05,
            'null_model':       NULL_MODEL,
            'n_iter':           N_ITER,
            'interpretation':   'AD and PD DNB proteins closer than random (supports cross-disease module)'
                                if prox_cross['z_score'] < -1.5 else
                                'AD and PD DNB proteins not significantly close',
        }
    ])
    prox_df.to_csv(out / 'proximity_results.csv', index=False)
    logger.info(f"Saved {out / 'proximity_results.csv'}")

    # ── Interactome summary ───────────────────────────────────────────────────
    summary_df = pd.DataFrame([{
        'interactome_nodes':      G.number_of_nodes(),
        'interactome_edges':      G.number_of_edges(),
        'adni_proteins_input':    len(ADNI_DNB_PROTEINS),
        'adni_proteins_in_net':   len(adni_proteins),
        'ppmi_proteins_input':    len(PPMI_DNB_PROTEINS),
        'ppmi_proteins_in_net':   len([p for p in PPMI_DNB_PROTEINS
                                       if p in G.nodes()]),
        'crybb2_in_interactome':  'CRYBB2' in G.nodes(),
    }])
    summary_df.to_csv(out / 'interactome_summary.csv', index=False)
    logger.info(f"Saved {out / 'interactome_summary.csv'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='NetMedPy interactome proximity analysis')
    parser.add_argument('--force', action='store_true',
                        help='Rerun even if output CSVs already exist')
    args = parser.parse_args()

    out_check = pathlib.Path('data/results/network_medicine/proximity_results.csv')
    if out_check.exists() and not args.force:
        logger.info("Results already exist. Use --force to rerun.")
        return

    # Load data
    G = load_interactome()
    adni_raw, ppmi_raw = load_proteins()

    # Filter to interactome
    adni_proteins = filter_to_interactome(adni_raw, G, label='ADNI')
    ppmi_proteins = filter_to_interactome(ppmi_raw, G, label='PPMI')

    if len(adni_proteins) < 1:
        raise ValueError(
            f"Only {len(adni_proteins)} ADNI proteins found in interactome. "
            "Check that the CSV contains a gene symbol column (e.g. EntrezGeneSymbol) "
            "and that symbols match nodes in the interactome."
        )

    # Distance matrix (cached)
    dist_matrix = get_distance_matrix(G)

    # Analysis 1: LCC
    lcc_adni = run_lcc(G, adni_proteins, label='ADNI')

    # Analysis 2: Within-group proximity of ADNI DNB proteins
    # (NOT run_proximity(A, A) — that collapses to AMSPL=0 when A==B)
    prox_adni = run_within_proximity(G, adni_proteins, dist_matrix)

    # Analysis 3: Cross-disease AD → PD proximity
    prox_cross = run_proximity(
        G, adni_proteins, ppmi_proteins, dist_matrix,
        label='AD_PD_cross'
    )

    # Save
    save_results(lcc_adni, prox_adni, prox_cross,
                 adni_proteins, ppmi_proteins, G)

    logger.info("=== Network medicine analysis complete ===")
    logger.info(f"LCC Z = {lcc_adni['z_score']:.3f}, "
                f"p = {lcc_adni['p_val']:.4f}")
    logger.info(f"Within-group proximity Z = {prox_adni['z_score']:.3f}, "
                f"p = {prox_adni['p_value']:.4f}")
    logger.info(f"Cross-disease proximity Z = {prox_cross['z_score']:.3f}, "
                f"p = {prox_cross['p_value_single_tail']:.4f}")


if __name__ == '__main__':
    main()
