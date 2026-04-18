#!/usr/bin/env python3
"""
Figure 4: Core DNB protein network visualization
Output: data/results/figures/figure_04_core_network
Data: data/results/dnb/somascan/wgcna/dnb_core_proteins_wgcna_annotated.csv
      data/results/network_medicine/lcc_results.csv

Network of 7 ADNI DNB proteins with PPI edges. Node size ~ kME, color ~ variance ratio.
LCC edges in bold orange, indirect shortest paths dashed grey.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as mlines
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from src.figures.figure_utils import (
    set_style, save_figure, SINGLE_COL,
)

set_style()
plt.rcParams.update({
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

# Documented PPI edges (LCC)
LCC_EDGES = [('RAC1', 'COPS5'), ('PKM', 'LANCL1')]
# Indirect shortest-path connections
INDIRECT_EDGES = [('PDCD6IP', 'RAC1'), ('PKM', 'RAC1'), ('CSNK1G1', 'PKM')]

# kME range for size scaling
KME_MIN, KME_MAX = 0.611, 0.695
SIZE_MIN, SIZE_MAX = 800, 2000

# Variance ratio range for color scaling
VAR_MIN, VAR_MAX = 0.769, 5.704


def load_data():
    prot_path = Path('data/results/dnb/somascan/wgcna/dnb_core_proteins_wgcna_annotated.csv')
    lcc_path = Path('data/results/network_medicine/lcc_results.csv')
    if not prot_path.exists():
        raise FileNotFoundError(f'{prot_path} not found')
    proteins = pd.read_csv(prot_path)
    lcc = pd.read_csv(lcc_path) if lcc_path.exists() else None
    return proteins, lcc


def _kme_to_size(kme):
    """Linear scaling of kME to scatter node size."""
    t = (kme - KME_MIN) / (KME_MAX - KME_MIN + 1e-9)
    t = np.clip(t, 0, 1)
    return SIZE_MIN + t * (SIZE_MAX - SIZE_MIN)


def make_figure(proteins, lcc):
    fig, ax = plt.subplots(figsize=(6.0, 5.5))

    G = nx.Graph()
    for _, row in proteins.iterrows():
        G.add_node(row['EntrezGeneSymbol'],
                    kME=row['kME'],
                    var_ratio=row['var_ratio'],
                    full_name=row['TargetFullName'])

    # Add edges
    for u, v in LCC_EDGES:
        G.add_edge(u, v, etype='lcc')
    for u, v in INDIRECT_EDGES:
        G.add_edge(u, v, etype='indirect')

    # Layout
    pos = nx.kamada_kawai_layout(G)

    # Nudge nodes away from boundaries (tighter clamp to avoid clipping)
    for node, (px, py) in pos.items():
        pos[node] = (np.clip(px, -0.70, 0.70), np.clip(py, -0.65, 0.75))

    # Node attributes
    nodes = list(G.nodes())
    kme_vals = np.array([G.nodes[n]['kME'] for n in nodes])
    var_vals = np.array([G.nodes[n]['var_ratio'] for n in nodes])
    node_sizes = np.array([_kme_to_size(k) for k in kme_vals])

    norm = mcolors.Normalize(vmin=VAR_MIN, vmax=VAR_MAX)
    cmap = cm.get_cmap('YlOrRd')
    node_colors = [cmap(norm(v)) for v in var_vals]

    # Draw indirect edges first (background)
    indirect_edges = [(u, v) for u, v, d in G.edges(data=True) if d['etype'] == 'indirect']
    nx.draw_networkx_edges(G, pos, edgelist=indirect_edges, ax=ax,
                           style='dashed', edge_color='#999999',
                           width=1.0, alpha=0.5)

    # Draw LCC edges (foreground, bold orange)
    lcc_edges = [(u, v) for u, v, d in G.edges(data=True) if d['etype'] == 'lcc']
    nx.draw_networkx_edges(G, pos, edgelist=lcc_edges, ax=ax,
                           style='solid', edge_color='#E8750A',
                           width=3.0, alpha=0.9)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, edgecolors='#333333',
                           linewidths=0.8)

    # Draw labels with offset to avoid overlap
    label_pos = {}
    for node, (px, py) in pos.items():
        # Offset labels slightly above nodes
        label_pos[node] = (px, py + 0.08)
    nx.draw_networkx_labels(G, label_pos, ax=ax, font_size=9, font_weight='bold')

    # Colorbar for variance ratio
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, aspect=15)
    cbar.set_label('Variance ratio', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Size legend: two circles for kME=0.61 and kME=0.70
    s_small = _kme_to_size(0.61)
    s_large = _kme_to_size(0.70)
    leg_small = ax.scatter([], [], s=s_small, c='#DDDDDD', edgecolors='#333333',
                           linewidths=0.8, label='kME = 0.61')
    leg_large = ax.scatter([], [], s=s_large, c='#DDDDDD', edgecolors='#333333',
                           linewidths=0.8, label='kME = 0.70')
    # Edge type legend entries
    lcc_line = mlines.Line2D([], [], color='#E8750A', linewidth=3,
                             label='LCC edge (direct PPI)')
    ind_line = mlines.Line2D([], [], color='#999999', linewidth=1, linestyle='--',
                             label='Indirect shortest path')
    ax.legend(handles=[leg_small, leg_large, lcc_line, ind_line],
              loc='lower right', fontsize=8, framealpha=0.9,
              edgecolor='#cccccc', fancybox=True)

    # LCC annotation box — top-right corner
    if lcc is not None and len(lcc) > 0:
        row = lcc.iloc[0]
        lcc_text = (f"LCC size = {int(row['lcc_size'])}\n"
                    f"z = {row['z_score']:.3f}, p = {row['p_value']:.3f}")
        ax.text(0.98, 0.98, lcc_text, transform=ax.transAxes,
                fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='#999999', alpha=0.95))

    # Title with dashed grey box
    title_text = ax.set_title('Core DNB Proteins in Interactome', fontsize=12, pad=12)
    title_text.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor='#999999', linestyle='--', linewidth=1.0))

    ax.axis('off')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    fig.tight_layout()
    return fig


def main():
    proteins, lcc = load_data()
    fig = make_figure(proteins, lcc)
    save_figure(fig, 'data/results/figures/figure_04_core_network')
    fig.savefig('figure_04_fixed.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
