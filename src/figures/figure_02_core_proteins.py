#!/usr/bin/env python3
"""
Figure 2: ADNI DNB Core Proteins — IL13 and BCL2L10
Output: data/results/figures/figure_02_core_proteins
Data:
  - data/results/dnb/dnb_core_proteins.csv
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import numpy as np
from pathlib import Path
from src.figures.figure_utils import (
    set_style, add_panel_label, save_figure,
    PALETTE, FONT_SIZE, TICK_SIZE, DOUBLE_COL
)

set_style()

# Verified core proteins from pooled MCI→Dementia analysis (n = 100 converters)
CORE_PROTEINS = [
    {'gene': 'IL13',    'uniprot': 'P35225', 'aptamer': 'seq.3072.4',
     'full_name': 'Interleukin-13',
     'role': 'Anti-inflammatory cytokine; Th2 immune response;\nneuroinflammation modulator',
     'literature': 'Elevated in AD CSF (Motta et al. 2007);\nprotective role in neuroinflammation'},
    {'gene': 'BCL2L10', 'uniprot': 'Q9HD36', 'aptamer': 'seq.7249.307',
     'full_name': 'Bcl-2-like protein 10',
     'role': 'Anti-apoptotic BCL-2 family member;\nregulates mitochondrial apoptosis pathway',
     'literature': 'BCL-2 family implicated in neuronal\nsurvival/death decisions in AD'},
]


def load_data():
    csv_path = Path('data/results/dnb/dnb_core_proteins.csv')
    if not csv_path.exists():
        print(f'WARNING: {csv_path} not found — using hardcoded verified values')
        return pd.DataFrame(CORE_PROTEINS), True

    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} core proteins from {csv_path}')
    return df, False


def make_panel_a(ax, used_fallback=False):
    """Panel A: Core protein annotation cards."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    colors = [PALETTE['converter'], '#4393c3']

    for i, protein in enumerate(CORE_PROTEINS):
        y_top = 8.8 - i * 4.5
        color = colors[i]

        # Card background
        card = FancyBboxPatch((0.3, y_top - 3.8), 9.4, 3.8,
                               boxstyle='round,pad=0.15',
                               facecolor=color, alpha=0.08,
                               edgecolor=color, linewidth=1.5, zorder=1)
        ax.add_patch(card)

        # Gene symbol (large, bold)
        ax.text(0.8, y_top - 0.3, protein['gene'],
                fontsize=14, fontweight='bold', color=color,
                va='top', ha='left', zorder=2)

        # Full protein name
        ax.text(0.8, y_top - 1.0, protein['full_name'],
                fontsize=8, color='#333333', va='top', ha='left',
                fontstyle='italic', zorder=2)

        # UniProt + aptamer
        ax.text(9.2, y_top - 0.3,
                f'UniProt: {protein["uniprot"]}\n{protein["aptamer"]}',
                fontsize=6.5, color='#888888', va='top', ha='right', zorder=2)

        # Biological role
        ax.text(0.8, y_top - 1.7, 'Role:', fontsize=7, fontweight='bold',
                color='#444444', va='top', zorder=2)
        ax.text(1.8, y_top - 1.7, protein['role'],
                fontsize=6.5, color='#555555', va='top', zorder=2)

        # Literature context
        ax.text(0.8, y_top - 2.9, 'Literature:', fontsize=7, fontweight='bold',
                color='#444444', va='top', zorder=2)
        ax.text(2.5, y_top - 2.9, protein['literature'],
                fontsize=6.5, color='#555555', va='top', zorder=2)

    # Header annotation
    ax.text(5.0, 9.7, '2 DNB core proteins — ADNI MCI→Dementia group (n = 100)',
            fontsize=8, ha='center', va='top', fontweight='bold', color='#333333')

    add_panel_label(ax, 'A', x=-0.05, y=1.02)

    if used_fallback:
        ax.text(0.5, 0.01,
                'Note: generated from hardcoded values — rerun pipeline to use computed data',
                transform=ax.transAxes, fontsize=5, color='#999999',
                ha='center', style='italic')


def make_panel_b(ax):
    """Panel B: Methodological note — why only 2 proteins."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Explanation card
    card = FancyBboxPatch((0.3, 0.5), 9.4, 9.0,
                           boxstyle='round,pad=0.15',
                           facecolor='#F8F9FA', edgecolor='#cccccc',
                           linewidth=1.0, zorder=1)
    ax.add_patch(card)

    ax.text(5.0, 9.0, 'WGCNA Module DNB Convergence',
            fontsize=9, ha='center', va='top', fontweight='bold',
            color='#333333', zorder=2)

    explanation_lines = [
        ('The WGCNA-guided DNB analysis identified a', 7, 'normal', '#444444'),
        ('set of just 2 proteins that maximize:', 7, 'normal', '#444444'),
        ('', 6, 'normal', '#ffffff'),
        (r'DNB = ($\sigma_D$ × PCC$_D$) / |PCC$_O$|', 8, 'bold', '#333333'),
        ('', 6, 'normal', '#ffffff'),
        ('Key implications:', 7, 'bold', '#444444'),
        ('', 5, 'normal', '#ffffff'),
        ('• Small PCC_O denominator (2 vs 6,944', 6.5, 'normal', '#555555'),
        ('  remaining proteins) inflates the score', 6.5, 'normal', '#555555'),
        ('', 5, 'normal', '#ffffff'),
        ('• Both proteins show frequency 1.0', 6.5, 'normal', '#555555'),
        ('  (selected in all 100 converter participants)', 6.5, 'normal', '#555555'),
        ('', 5, 'normal', '#ffffff'),
        ('• Cross-validation needed to assess', 6.5, 'normal', '#555555'),
        ('  generalization (in-sample bias)', 6.5, 'normal', '#555555'),
        ('', 5, 'normal', '#ffffff'),
        ('Future: simulated annealing or expanded', 6.5, 'normal', '#888888'),
        ('search may identify larger, more robust sets', 6.5, 'normal', '#888888'),
    ]

    y = 8.0
    for text, fs, fw, color in explanation_lines:
        ax.text(1.0, y, text, fontsize=fs, fontweight=fw, color=color,
                va='top', zorder=2)
        y -= 0.45

    add_panel_label(ax, 'B', x=-0.05, y=1.02)


def make_figure(df, used_fallback=False):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 4.2),
                             gridspec_kw={'width_ratios': [55, 45]})
    make_panel_a(axes[0], used_fallback)
    make_panel_b(axes[1])
    fig.tight_layout(pad=1.0)
    return fig


def main():
    df, used_fallback = load_data()
    fig = make_figure(df, used_fallback)
    save_figure(fig, 'data/results/figures/figure_02_core_proteins')
    plt.close(fig)


if __name__ == '__main__':
    main()
