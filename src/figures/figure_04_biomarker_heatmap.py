#!/usr/bin/env python3
"""
Figure 4: Biomarker Correlation Heatmap
Output: data/results/figures/figure_04_biomarker_heatmap
Data:
  - data/results/validation/biomarker_correlations.csv
  - data/results/validation/biomarker_correlations_pvalues.csv
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from pathlib import Path
from src.figures.figure_utils import (
    set_style, add_panel_label, save_figure,
    FONT_SIZE, TICK_SIZE, SINGLE_COL
)

set_style()

# Display name mapping
DISPLAY_NAMES = {
    'sdnb_score':        'sDNB Score',
    'ABETA42_40_RATIO':  'A\u03b242/40 Ratio',
    'CDRSB':             'CDR Sum of Boxes',
    'AMYLOID_STATUS':    'Amyloid Status',
    'APOE4':             'APOE4 Genotype',
}

# Hardcoded verified correlation and p-value matrices (fallback)
VARS = ['sdnb_score', 'ABETA42_40_RATIO', 'CDRSB', 'AMYLOID_STATUS', 'APOE4']

HARDCODED_CORR = {
    ('sdnb_score', 'ABETA42_40_RATIO'):  0.083,
    ('sdnb_score', 'CDRSB'):            -0.074,
    ('sdnb_score', 'AMYLOID_STATUS'):   -0.052,
    ('sdnb_score', 'APOE4'):             0.012,
    ('APOE4', 'ABETA42_40_RATIO'):       0.491,
    ('APOE4', 'AMYLOID_STATUS'):         0.469,
    ('ABETA42_40_RATIO', 'CDRSB'):       np.nan,
    ('ABETA42_40_RATIO', 'AMYLOID_STATUS'): np.nan,
    ('CDRSB', 'AMYLOID_STATUS'):         1.0,
    ('CDRSB', 'APOE4'):                  0.254,
}

HARDCODED_PVAL = {
    ('sdnb_score', 'ABETA42_40_RATIO'):  0.624,
    ('sdnb_score', 'CDRSB'):             0.793,
    ('sdnb_score', 'AMYLOID_STATUS'):    0.649,
    ('sdnb_score', 'APOE4'):             0.834,
    ('APOE4', 'ABETA42_40_RATIO'):       0.002,
    ('APOE4', 'AMYLOID_STATUS'):         0.00001,
    ('ABETA42_40_RATIO', 'CDRSB'):       np.nan,
    ('ABETA42_40_RATIO', 'AMYLOID_STATUS'): np.nan,
    ('CDRSB', 'AMYLOID_STATUS'):         0.0001,
    ('CDRSB', 'APOE4'):                  0.360,
}


def _build_matrix_from_hardcoded():
    n = len(VARS)
    corr = np.full((n, n), np.nan)
    pval = np.full((n, n), np.nan)
    idx = {v: i for i, v in enumerate(VARS)}

    for (a, b), r in HARDCODED_CORR.items():
        i, j = idx[a], idx[b]
        corr[i, j] = r
        corr[j, i] = r

    for (a, b), p in HARDCODED_PVAL.items():
        i, j = idx[a], idx[b]
        pval[i, j] = p
        pval[j, i] = p

    return corr, pval


def load_data():
    corr_path = Path('data/results/validation/biomarker_correlations.csv')
    pval_path = Path('data/results/validation/biomarker_correlations_pvalues.csv')

    fallback = False
    if not corr_path.exists() or not pval_path.exists():
        print('WARNING: biomarker correlation CSVs not found — using hardcoded verified values from paper')
        corr_mat, pval_mat = _build_matrix_from_hardcoded()
        return corr_mat, pval_mat, VARS, True

    df_corr = pd.read_csv(corr_path, index_col=0)
    df_pval = pd.read_csv(pval_path, index_col=0)

    # Reorder to match VARS
    present = [v for v in VARS if v in df_corr.columns]
    if len(present) < len(VARS):
        missing = set(VARS) - set(present)
        print(f'WARNING: missing variables {missing} — filling with NaN')

    df_corr = df_corr.reindex(index=present, columns=present)
    df_pval = df_pval.reindex(index=present, columns=present)

    print(f'Loaded biomarker correlations ({len(present)} variables)')
    return df_corr.values, df_pval.values, present, False


def _sig_label(p):
    if np.isnan(p):
        return ''
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.10:
        return '\u2020'
    else:
        return 'ns'


def make_figure(corr_mat, pval_mat, var_names, used_fallback=False):
    n = len(var_names)
    display = [DISPLAY_NAMES.get(v, v) for v in var_names]

    # Set diagonal to NaN
    corr_plot = corr_mat.copy()
    np.fill_diagonal(corr_plot, np.nan)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.2))

    cmap = plt.cm.RdBu_r
    cmap.set_bad('white')

    im = ax.imshow(corr_plot, cmap=cmap, vmin=-1.0, vmax=1.0, aspect='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Spearman \u03c1', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Cell annotations
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = corr_mat[i, j]
            p = pval_mat[i, j]
            if np.isnan(r):
                continue

            text_color = 'white' if abs(r) > 0.5 else 'black'
            sig = _sig_label(p)

            # Special case: CDRSB ↔ AMYLOID_STATUS artefact note
            vi, vj = var_names[i] if i < len(var_names) else '', var_names[j] if j < len(var_names) else ''
            is_artefact = ({vi, vj} == {'CDRSB', 'AMYLOID_STATUS'})

            if is_artefact:
                ax.text(j, i, f'{r:.2f}\nartefact\n(small n)',
                        ha='center', va='center', fontsize=5,
                        color=text_color, style='italic')
            else:
                ax.text(j, i, f'{r:.2f}\n{sig}',
                        ha='center', va='center', fontsize=6,
                        color=text_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display, fontsize=TICK_SIZE, rotation=45, ha='right')
    ax.set_yticklabels(display, fontsize=TICK_SIZE)
    ax.set_title('Spearman correlations: sDNB and established biomarkers',
                 fontsize=8, pad=6)

    # Footnote — placed at -0.30 in axes coords to clear the rotated x-tick labels
    footnote = ('Note: n varies by biomarker availability (sDNB n = 185; '
                'A\u03b242/40 n = 26). All sDNB correlations are non-significant.')
    fig.text(0.5, -0.48, footnote, ha='center', fontsize=6, color='#444444',
             transform=ax.transAxes)

    if used_fallback:
        ax.text(0.5, -0.42,
                'Note: generated from hardcoded values — rerun pipeline to use computed data',
                transform=ax.transAxes, fontsize=5, color='#999999',
                ha='center', style='italic')

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)  # preserve space for rotated tick labels and footnote
    return fig


def main():
    corr_mat, pval_mat, var_names, used_fallback = load_data()
    fig = make_figure(corr_mat, pval_mat, var_names, used_fallback)
    save_figure(fig, 'data/results/figures/figure_04_biomarker_heatmap')
    plt.close(fig)


if __name__ == '__main__':
    main()
