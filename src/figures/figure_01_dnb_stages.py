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

def load_data():
    csv_path = Path('data/results/dnb/somascan/wgcna/sdnb_scores_wgcna.csv')
    if not csv_path.exists():
        print(f'WARNING: {csv_path} not found — cannot build figure')
        return pd.DataFrame(), True

    df = pd.read_csv(csv_path)
    required = {'TRAJECTORY', 'sdnb_score'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns {missing} in {csv_path}')

    # Group by stage to compute mean sDNB score
    grouped = df.groupby('TRAJECTORY')
    stage_df = grouped['sdnb_score'].agg(['mean', 'count']).reset_index()
    stage_df.rename(columns={'TRAJECTORY': 'stage', 'mean': 'dnb_score', 'count': 'n_samples'}, inplace=True)

    # Drop "other" group
    stage_df = stage_df[stage_df['stage'] != 'other'].copy()
    # Keep only stages we want to display
    stage_df = stage_df[stage_df['stage'].isin(STAGE_LABELS)].copy()
    print(f'Loaded WGCNA patient scores corresponding to {len(stage_df)} stage rows from {csv_path}')
    return stage_df, False


def make_figure(df, used_fallback=False):
    # Order by DNB score descending (top = highest)
    df = df.sort_values('dnb_score', ascending=True)   # ascending for barh (bottom→top)

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.3, 3.0))

    y_positions = np.arange(len(df))
    colors = [STAGE_COLORS[s] for s in df['stage']]

    bars = ax.barh(y_positions, df['dnb_score'], height=0.6,
                   color=colors, clip_on=False)

    # Determine dynamic x-axis bounds based on sDNB scores
    score_min = df['dnb_score'].min()
    score_max = df['dnb_score'].max()
    x_start = max(0, score_min - 0.2)
    x_end = score_max * 1.4

    # X-axis
    ax.set_xlim(x_start, x_end)
    ax.set_xlabel('Patient sDNB Score (WGCNA)', fontsize=8)
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
    ref_row = df[df['stage'] == 'CN_amyloid_negative']
    ref_score = ref_row['dnb_score'].values[0] if not ref_row.empty else score_min

    for i, (_, row) in enumerate(df.iterrows()):
        score = row['dnb_score']
        n = int(row['n_samples'])
        is_converter = row['stage'] == 'MCI_to_Dementia'
        fw = 'bold' if is_converter else 'normal'

        # Score label
        if is_converter and ref_score > 0:
            pct_change = ((score - ref_score) / ref_score) * 100
            label = f'{score:.2f}  (+{pct_change:.0f}%)'
        else:
            label = f'{score:.2f}'

        ax.text(score + (x_end - score_max)*0.02, i, label,
                va='center', ha='left', fontsize=FONT_SIZE, fontweight=fw,
                clip_on=False)

        # n annotation in grey
        ax.text(score + (x_end - score_max)*0.4, i, f'n={n}',
                va='center', ha='left', fontsize=5.5, color='#888888',
                clip_on=False)

    # Reference dashed line at CN_amyloid_negative
    ax.axvline(x=ref_score, color='#888888', linestyle='--', linewidth=0.8)
    # Reference label
    ax.text(ref_score + (x_end - x_start)*0.01, 3.55, 'ref.',
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
