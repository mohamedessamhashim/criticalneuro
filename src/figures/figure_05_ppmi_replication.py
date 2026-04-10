#!/usr/bin/env python3
"""
Figure 5: PPMI Replication — PD DNB Scores
Output: data/results/figures/figure_05_ppmi_replication
Data: data/results/ppmi/ppmi_dnb_scores_by_stage.csv
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as mpl_patches
import numpy as np
import pandas as pd
from pathlib import Path
from src.figures.figure_utils import (
    set_style, add_panel_label, save_figure,
    PALETTE, FONT_SIZE, TICK_SIZE, DOUBLE_COL
)

set_style()

STAGE_LABELS = {
    'PD_INTERMEDIATE': 'PD Intermediate',
    'PD_FAST':         'PD Fast',
    'PD_SLOW':         'PD Slow (reference)',
}

STAGE_COLORS = {
    'PD_INTERMEDIATE': PALETTE['pd_intermediate'],
    'PD_FAST':         PALETTE['pd_fast'],
    'PD_SLOW':         PALETTE['pd_slow'],
}

def load_data():
    csv_path = Path('data/results/dnb/somascan_ppmi/wgcna/sdnb_scores_wgcna.csv')
    if not csv_path.exists():
        print(f'WARNING: {csv_path} not found — cannot build figure')
        return pd.DataFrame(), True

    df = pd.read_csv(csv_path)
    required = {'TRAJECTORY', 'sdnb_score'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns {missing} in {csv_path}')

    grouped = df.groupby('TRAJECTORY')
    stage_df = grouped['sdnb_score'].agg(['mean', 'count']).reset_index()
    stage_df.rename(columns={'TRAJECTORY': 'stage', 'mean': 'dnb_score', 'count': 'n_samples'}, inplace=True)

    stage_df = stage_df[stage_df['stage'] != 'other'].copy()
    stage_df = stage_df[stage_df['stage'].isin(STAGE_LABELS)].copy()
    print(f'Loaded WGCNA patient scores corresponding to {len(stage_df)} PPMI stage rows from {csv_path}')
    return stage_df, False


def make_panel_a(ax, df, used_fallback=False):
    """Horizontal bar chart ordered by DNB score descending."""
    df = df.sort_values('dnb_score', ascending=True)  # ascending for barh

    y_pos = np.arange(len(df))
    colors = [STAGE_COLORS[s] for s in df['stage']]

    ax.barh(y_pos, df['dnb_score'], height=0.6, color=colors)

    score_min = df['dnb_score'].min()
    score_max = df['dnb_score'].max()
    x_start = max(0, score_min - 0.1)
    x_end = score_max * 1.5

    ax.set_xlim(x_start, x_end)
    ax.set_xlabel('Patient sDNB Score (WGCNA)', fontsize=8)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)

    # Y labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([STAGE_LABELS[s] for s in df['stage']], fontsize=TICK_SIZE)
    ax.tick_params(axis='y', length=0)

    # Score + n annotations
    ref_row = df[df['stage'] == 'PD_SLOW']
    ref_score = ref_row['dnb_score'].values[0] if not ref_row.empty else score_min

    for i, (_, row) in enumerate(df.iterrows()):
        score = row['dnb_score']
        n = int(row['n_samples'])
        is_peak = row['stage'] == 'PD_INTERMEDIATE'
        fw = 'bold' if is_peak else 'normal'

        ax.text(score + (x_end - score_max)*0.02, i, f'{score:.3f}', va='center', ha='left',
                fontsize=FONT_SIZE, fontweight=fw)
        ax.text(score + (x_end - score_max)*0.4, i, f'(n = {n})', va='center', ha='left',
                fontsize=6, color='#888888')

    # Reference line at PD_SLOW
    ax.axvline(x=ref_score, color='#888888', linestyle='--', linewidth=0.8)
    ax.text(ref_score + (x_end - x_start)*0.01, 2.55, 'PD Slow\nref.',
            fontsize=6, color='#888888', style='italic', va='top', ha='left')

    # Inverted-U annotation box
    if 'PD_INTERMEDIATE' in df['stage'].values:
        inter_idx = df['stage'].tolist().index('PD_INTERMEDIATE')
        ax.annotate(
            'Tipping-point theory:\npeak at pre-transition\n(intermediate) stage',
            xy=(df[df['stage'] == 'PD_INTERMEDIATE']['dnb_score'].values[0], inter_idx), xycoords='data',
            xytext=(0.95, 0.75), textcoords='axes fraction',
            fontsize=6, color='#444444', ha='right', va='top',
            arrowprops=dict(arrowstyle='->', color='#444444',
                            connectionstyle='arc3,rad=-0.2', lw=0.8),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#cccccc', alpha=0.85)
        )

    ax.set_title('PPMI DNB Score by PD Progression Group', fontsize=8)
    ax.text(0.5, -0.20,
            '\u2021 Other/unclassified (n = 20) excluded; see supplementary',
            transform=ax.transAxes, fontsize=6, ha='center', color='#666666')
    add_panel_label(ax, 'A')


def make_panel_b(ax):
    """Tipping-point theory schematic (illustrative)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Draw inverted-U curve
    x = np.linspace(0.5, 9.5, 300)
    # Inverted U peaking at x=5
    y = 9.0 * np.exp(-0.5 * ((x - 5.0) / 1.8) ** 2) + 0.5
    ax.plot(x, y, color='#666666', linewidth=2.0, zorder=3)

    # Tipping point line
    ax.axvline(x=5.0, color='#888888', linestyle='--', linewidth=0.8)
    ax.text(7.0, 7.5, 'Tipping point /\nBifurcation',
            fontsize=6, color='#666666', va='top')

    # Shaded pre-transition region
    ax.fill_betweenx([0, 10.5], 0, 5.0,
                     color='#FFFDE7', alpha=0.6, zorder=1)
    ax.text(2.5, 1.2,
            'Pre-transition window\n(target for early detection)',
            fontsize=6, color='#666666', ha='center')

    # Three colored dots on the curve
    # PD_SLOW at x=2
    x_slow, y_slow = 2.0, float(9.0 * np.exp(-0.5 * ((2.0 - 5.0) / 1.8) ** 2) + 0.5)
    ax.scatter([x_slow], [y_slow], color=PALETTE['pd_slow'], s=60, zorder=5)
    ax.text(x_slow - 0.3, y_slow - 1.2, 'PD Slow', ha='center', fontsize=7,
            color=PALETTE['pd_slow'])

    # PD_INTERMEDIATE at x=5 (peak)
    x_int, y_int = 5.0, float(9.0 * np.exp(-0.5 * ((5.0 - 5.0) / 1.8) ** 2) + 0.5)
    ax.scatter([x_int], [y_int], color=PALETTE['pd_intermediate'], s=80, zorder=5,
               marker='*')
    ax.text(x_int + 1.2, y_int, '* PD Intermediate', ha='left', fontsize=7,
            color=PALETTE['pd_intermediate'], fontweight='bold')

    # PD_FAST at x=7 (descending)
    x_fast, y_fast = 7.0, float(9.0 * np.exp(-0.5 * ((7.0 - 5.0) / 1.8) ** 2) + 0.5)
    ax.scatter([x_fast], [y_fast], color=PALETTE['pd_fast'], s=60, zorder=5)
    ax.text(x_fast, y_fast - 0.8, 'PD Fast', ha='center', fontsize=7,
            color=PALETTE['pd_fast'])

    # Axes labels (fake arrows as text)
    ax.annotate('', xy=(9.8, 0.3), xytext=(0.3, 0.3),
                arrowprops=dict(arrowstyle='->', color='#444444', lw=1.0))
    ax.text(5.0, -0.1, 'Disease progression \u2192',
            ha='center', fontsize=7, color='#444444')
    ax.annotate('', xy=(0.3, 9.8), xytext=(0.3, 0.3),
                arrowprops=dict(arrowstyle='->', color='#444444', lw=1.0))
    ax.text(-0.2, 5.0, 'DNB Score / Network instability',
            ha='center', fontsize=7, color='#444444', rotation=90, va='center')

    ax.set_title('Tipping-Point Theory Schematic', fontsize=8)
    ax.text(0.5, -0.12,
            'Schematic: DNB theory predicts peak instability before tipping point',
            transform=ax.transAxes, fontsize=5.5, ha='center', color='#666666',
            style='italic')

    add_panel_label(ax, 'B', x=-0.08, y=1.02)


def make_figure(df, used_fallback=False):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5),
                             gridspec_kw={'width_ratios': [55, 45]})
    make_panel_a(axes[0], df, used_fallback)
    make_panel_b(axes[1])
    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(bottom=0.25)
    return fig


def main():
    df, used_fallback = load_data()
    fig = make_figure(df, used_fallback)
    save_figure(fig, 'data/results/figures/figure_05_ppmi_replication')
    plt.close(fig)


if __name__ == '__main__':
    main()
