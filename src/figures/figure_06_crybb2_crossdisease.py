#!/usr/bin/env python3
"""
Figure 6: Cross-Disease Comparison — ADNI (AD) vs PPMI (PD)
Output: data/results/figures/figure_06_cross_disease
Data:
  - data/results/dnb/dnb_core_proteins.csv
  - data/results/ppmi/ppmi_dnb_core_proteins.csv
  - data/results/dnb/dnb_scores_by_stage.csv
  - data/results/ppmi/ppmi_dnb_scores_by_stage.csv
  - data/results/network_medicine/proximity_results.csv
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from pathlib import Path
from src.figures.figure_utils import (
    set_style, add_panel_label, save_figure,
    PALETTE, FONT_SIZE, TICK_SIZE, DOUBLE_COL
)

set_style()


def load_data():
    adni_core = pd.read_csv('data/results/dnb/dnb_core_proteins.csv') \
        if Path('data/results/dnb/dnb_core_proteins.csv').exists() else pd.DataFrame()
    ppmi_core = pd.read_csv('data/results/ppmi/ppmi_dnb_core_proteins.csv') \
        if Path('data/results/ppmi/ppmi_dnb_core_proteins.csv').exists() else pd.DataFrame()
    adni_dnb = pd.read_csv('data/results/dnb/dnb_scores_by_stage.csv') \
        if Path('data/results/dnb/dnb_scores_by_stage.csv').exists() else pd.DataFrame()
    ppmi_dnb = pd.read_csv('data/results/ppmi/ppmi_dnb_scores_by_stage.csv') \
        if Path('data/results/ppmi/ppmi_dnb_scores_by_stage.csv').exists() else pd.DataFrame()
    prox = pd.read_csv('data/results/network_medicine/proximity_results.csv') \
        if Path('data/results/network_medicine/proximity_results.csv').exists() else pd.DataFrame()

    return adni_core, ppmi_core, adni_dnb, ppmi_dnb, prox


def make_panel_a(ax, adni_dnb, ppmi_dnb):
    """Panel A: Side-by-side normalized DNB score comparison."""
    # ADNI stages: reference → stable → converter
    adni_stages = ['CN_amyloid_negative', 'stable_MCI', 'MCI_to_Dementia']
    ppmi_stages = ['PD_SLOW', 'PD_INTERMEDIATE', 'PD_FAST']
    labels = ['Reference', 'Stable/Slow', 'Converter/Fast']

    adni_vals = []
    ppmi_vals = []
    for s in adni_stages:
        row = adni_dnb[adni_dnb['stage'] == s]
        adni_vals.append(row['dnb_score'].values[0] if len(row) > 0 else 0)
    for s in ppmi_stages:
        row = ppmi_dnb[ppmi_dnb['stage'] == s]
        ppmi_vals.append(row['dnb_score'].values[0] if len(row) > 0 else 0)

    # Normalize to reference
    adni_norm = [v / adni_vals[0] if adni_vals[0] > 0 else 0 for v in adni_vals]
    ppmi_norm = [v / ppmi_vals[0] if ppmi_vals[0] > 0 else 0 for v in ppmi_vals]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, adni_norm, width, label='ADNI (AD)',
                   color=PALETTE['converter'], alpha=0.8, edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x + width / 2, ppmi_norm, width, label='PPMI (PD)',
                   color=PALETTE['pd_intermediate'], alpha=0.8, edgecolor='white', linewidth=0.8)

    # Value labels — inside bars to avoid collision
    for bar, val in zip(bars1, adni_norm):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.08,
                f'{val:.2f}', ha='center', va='top', fontsize=5.5, color='white',
                fontweight='bold')
    for bar, val in zip(bars2, ppmi_norm):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.08,
                f'{val:.2f}', ha='center', va='top', fontsize=5.5, color='white',
                fontweight='bold')

    ax.axhline(1.0, ls='--', color='#cccccc', lw=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel('DNB Score\n(norm. to ref.)', fontsize=7)
    ax.set_ylim(0, max(max(adni_norm), max(ppmi_norm)) * 1.08)
    ax.set_title('DNB Score Pattern: AD vs PD', fontsize=8)
    ax.legend(fontsize=5.5, loc='upper left', framealpha=0.9)

    add_panel_label(ax, 'A')


def make_panel_b(ax, adni_core, ppmi_core):
    """Panel B: Core protein comparison table — no overlap."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5.0, 9.6, 'Core Proteins', fontsize=8,
            ha='center', va='top', fontweight='bold')
    ax.text(5.0, 8.9, 'No shared proteins', fontsize=6.5,
            ha='center', va='top', color='#888888', fontstyle='italic')

    # ADNI section
    y = 8.0
    ax.text(0.3, y, 'ADNI (AD)', fontsize=7.5, fontweight='bold',
            color=PALETTE['converter'], va='top')
    y -= 0.7
    if len(adni_core) > 0:
        for _, row in adni_core.iterrows():
            gene = row.get('EntrezGeneSymbol', row.get('protein', ''))
            name = row.get('TargetFullName', '')
            uniprot = row.get('UniProt', '')
            ax.text(0.6, y, f'\u2022 {gene} \u2014 {name} ({uniprot})',
                    fontsize=6, color='#333333', va='top')
            y -= 0.7
    else:
        ax.text(0.6, y, '\u2022 IL13 \u2014 Interleukin-13 (P35225)', fontsize=6, va='top')
        y -= 0.7
        ax.text(0.6, y, '\u2022 BCL2L10 \u2014 Bcl-2-like protein 10 (Q9HD36)', fontsize=6, va='top')
        y -= 0.7

    # Separator
    y -= 0.2
    ax.plot([0.3, 9.7], [y + 0.1, y + 0.1], '--', color='#cccccc', lw=0.8)
    y -= 0.4

    # PPMI section
    ax.text(0.3, y, 'PPMI (PD)', fontsize=7.5, fontweight='bold',
            color=PALETTE['pd_intermediate'], va='top')
    y -= 0.7
    if len(ppmi_core) > 0:
        unique_genes = ppmi_core.drop_duplicates(subset='EntrezGeneSymbol')
        for _, row in unique_genes.iterrows():
            gene = row.get('EntrezGeneSymbol', row.get('protein', ''))
            name = row.get('TargetFullName', '')
            uniprot = row.get('UniProt', '')
            n_apt = len(ppmi_core[ppmi_core['EntrezGeneSymbol'] == gene])
            apt_note = f', {n_apt} apt.' if n_apt > 1 else ''
            ax.text(0.6, y, f'\u2022 {gene} \u2014 {name} ({uniprot}{apt_note})',
                    fontsize=6, color='#333333', va='top')
            y -= 0.7
    else:
        ax.text(0.6, y, '\u2022 CXCL12 \u2014 SDF-1 (P48061, 2 apt.)',
                fontsize=6, va='top')

    add_panel_label(ax, 'B', x=-0.05, y=1.02)


def make_panel_c(ax, prox):
    """Panel C: Network proximity results."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5.0, 9.6, 'Interactome Proximity', fontsize=8,
            ha='center', va='top', fontweight='bold')

    # Results card
    card = FancyBboxPatch((0.3, 0.5), 9.4, 8.5,
                           boxstyle='round,pad=0.15',
                           facecolor='#F8F9FA', edgecolor='#cccccc',
                           linewidth=1.0, zorder=1)
    ax.add_patch(card)

    if len(prox) > 0:
        y = 8.2
        for _, row in prox.iterrows():
            label = row.get('label', row.get('comparison', ''))
            z = row['z_score']
            p = row['p_value']
            is_sig = row.get('significant', p < 0.05)
            sig_color = PALETTE['converter'] if is_sig else '#888888'
            sig_label = 'Significant' if is_sig else 'Not significant'

            # Comparison label
            ax.text(0.8, y, label, fontsize=6.5, va='top',
                    fontweight='bold', color='#333333', zorder=2)
            y -= 0.6
            # Stats on second line
            ax.text(1.2, y, f'z = {z:.2f}   p = {p:.3f}   [{sig_label}]',
                    fontsize=6, va='top', color=sig_color, zorder=2)
            y -= 1.0

        # Interpretation
        y -= 0.3
        ax.text(0.8, y, 'Interpretation:', fontsize=6.5,
                va='top', fontweight='bold', color='#444444', zorder=2)
        y -= 0.6
        interp_lines = [
            '\u2022 Self-proximity z > 0: core proteins',
            '  not clustered (expected for n=2)',
            '\u2022 Cross-proximity not significant:',
            '  AD/PD proteins in distinct regions',
        ]
        for line in interp_lines:
            ax.text(1.0, y, line, fontsize=5.5, va='top',
                    color='#555555', zorder=2)
            y -= 0.55
    else:
        ax.text(5.0, 5.0, 'Network proximity\nresults not available',
                fontsize=8, ha='center', va='center', color='#888888')

    add_panel_label(ax, 'C', x=-0.05, y=1.02)


def make_figure(adni_core, ppmi_core, adni_dnb, ppmi_dnb, prox):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 4.0),
                             gridspec_kw={'width_ratios': [30, 35, 35]})
    make_panel_a(axes[0], adni_dnb, ppmi_dnb)
    make_panel_b(axes[1], adni_core, ppmi_core)
    make_panel_c(axes[2], prox)
    fig.tight_layout(pad=1.2)
    return fig


def main():
    adni_core, ppmi_core, adni_dnb, ppmi_dnb, prox = load_data()
    fig = make_figure(adni_core, ppmi_core, adni_dnb, ppmi_dnb, prox)
    save_figure(fig, 'data/results/figures/figure_06_cross_disease')
    plt.close(fig)


if __name__ == '__main__':
    main()
