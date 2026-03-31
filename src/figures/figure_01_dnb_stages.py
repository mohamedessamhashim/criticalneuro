#!/usr/bin/env python3
"""
Figure 1: ADNI DNB Score Across Disease Stages
Output: data/results/figures/figure_01_dnb_stages
Data: data/results/dnb/dnb_scores_by_stage.csv
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pathlib import Path
from src.figures.figure_utils import (
    set_style, add_panel_label, format_pvalue, save_figure,
    PALETTE, FONT_SIZE, TICK_SIZE, SINGLE_COL
)

set_style()

# Stage name mapping: CSV value → display label
STAGE_LABELS = {
    'MCI_to_Dementia':     'MCI \u2192 Dementia',
    'CN_amyloid_positive': 'CN Amyloid-positive\u2020',
    'CN_amyloid_negative': 'CN Amyloid-neg (ref)',
    'stable_MCI':          'Stable MCI',
}

STAGE_COLORS = {
    'MCI_to_Dementia':     PALETTE['converter'],
    'CN_amyloid_positive': PALETTE['cn_pos'],
    'CN_amyloid_negative': PALETTE['cn_neg'],
    'stable_MCI':          PALETTE['stable'],
}

# Hardcoded verified values from paper (fallback)
HARDCODED = [
    {'stage': 'MCI_to_Dementia',     'dnb_score': 7.49, 'n_samples': 100},
    {'stage': 'CN_amyloid_positive', 'dnb_score': 6.63, 'n_samples': 6},
    {'stage': 'CN_amyloid_negative', 'dnb_score': 6.35, 'n_samples': 37},
    {'stage': 'stable_MCI',          'dnb_score': 6.18, 'n_samples': 85},
]


def load_data():
    csv_path = Path('data/results/dnb/dnb_scores_by_stage.csv')
    if not csv_path.exists():
        print(f'WARNING: {csv_path} not found — using hardcoded verified values from paper')
        return pd.DataFrame(HARDCODED), True

    df = pd.read_csv(csv_path)
    required = {'stage', 'dnb_score', 'n_samples'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns {missing} in {csv_path}')

    # Drop "other" group
    df = df[df['stage'] != 'other'].copy()
    # Keep only stages we want to display
    df = df[df['stage'].isin(STAGE_LABELS)].copy()
    print(f'Loaded {len(df)} stage rows from {csv_path}')
    return df, False


def make_figure(df, used_fallback=False):
    # Order by DNB score descending (top = highest)
    df = df.sort_values('dnb_score', ascending=True)   # ascending for barh (bottom→top)

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.3, 3.0))

    y_positions = np.arange(len(df))
    colors = [STAGE_COLORS[s] for s in df['stage']]

    bars = ax.barh(y_positions, df['dnb_score'], height=0.6,
                   color=colors, left=5.5, clip_on=False)

    # X-axis: extended to 8.6 to give annotation room inside axes
    ax.set_xlim(5.5, 8.6)
    ax.set_xticks(np.arange(5.5, 8.6, 0.5))
    ax.set_xlabel('DNB Score', fontsize=8)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)

    # Y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [STAGE_LABELS[s] for s in df['stage']],
        fontsize=TICK_SIZE
    )
    ax.tick_params(axis='y', length=0)

    # Score annotations at right end of each bar — appended to score label
    for i, (_, row) in enumerate(df.iterrows()):
        score = row['dnb_score']
        n = int(row['n_samples'])
        is_converter = row['stage'] == 'MCI_to_Dementia'
        fw = 'bold' if is_converter else 'normal'

        # Score label (with +21% inline for converter)
        if is_converter:
            label = f'{score:.2f}  (+21%)'
        else:
            label = f'{score:.2f}'
        ax.text(score + 5.5 + 0.04, i, label,
                va='center', ha='left', fontsize=FONT_SIZE, fontweight=fw,
                clip_on=False)

        # n annotation in grey — one column width to the right
        ax.text(score + 5.5 + 0.55, i, f'n={n}',
                va='center', ha='left', fontsize=5.5, color='#888888',
                clip_on=False)

    # Reference dashed line at CN_amyloid_negative (6.35)
    ref_score = 6.35
    ax.axvline(x=ref_score, color='#888888', linestyle='--', linewidth=0.8)
    # Reference label — placed just to the right of the dashed line, above x-axis
    ax.text(ref_score + 0.03, 3.55, 'ref.',
            fontsize=5.5, color='#888888', style='italic', va='top', ha='left')

    # Footnote
    ax.text(0.0, -0.22, '\u2020n = 6; interpret cautiously',
            transform=ax.transAxes, fontsize=6, color='#666666')

    if used_fallback:
        ax.text(0.5, 0.02,
                'Note: generated from hardcoded values — rerun pipeline to use computed data',
                transform=ax.transAxes, fontsize=5, color='#999999',
                ha='center', style='italic')

    ax.set_title('ADNI DNB Score by Disease Stage', fontsize=9, pad=6)
    fig.subplots_adjust(left=0.26, right=0.70, top=0.88, bottom=0.28)
    return fig


def main():
    df, used_fallback = load_data()
    fig = make_figure(df, used_fallback)
    save_figure(fig, 'data/results/figures/figure_01_dnb_stages')
    plt.close(fig)


if __name__ == '__main__':
    main()
