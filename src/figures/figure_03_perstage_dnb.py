#!/usr/bin/env python3
"""
Figure 3: Per-stage DNB decomposition with PCC_outside
Output: data/results/figures/figure_03_perstage_dnb
Data: data/results/dnb/somascan/wgcna/stage_dnb_scores.csv

(A) Bar chart of PCC_outside across 4 trajectory groups. Monotonic decline
    from CN_amyloid_negative (0.247) -> stable_MCI (0.109) -> MCI_to_Dementia (0.102).
    Dashed trend line through interpretable points. CN_amyloid_positive (n=6) hatched.
(B) Composite DNB scores across same groups. Non-monotonic pattern.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pathlib import Path
from src.figures.figure_utils import (
    set_style, add_panel_label, save_figure,
    DOUBLE_COL, PANEL_HEIGHT,
)

set_style()
plt.rcParams.update({
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

# Ordered stages along clinical trajectory
STAGE_ORDER = ['CN_amyloid_negative', 'CN_amyloid_positive', 'stable_MCI', 'MCI_to_Dementia']

STAGE_LABELS = {
    'CN_amyloid_negative': 'CN A\u03b2\u2212',
    'CN_amyloid_positive': 'CN A\u03b2+*',
    'stable_MCI':          'Stable MCI',
    'MCI_to_Dementia':     'MCI\u2192Dem',
}

# Disease-progression palette
STAGE_COLORS = {
    'CN_amyloid_negative': '#4A90D9',
    'CN_amyloid_positive': '#A8C8E8',
    'stable_MCI':          '#E8A020',
    'MCI_to_Dementia':     '#C0392B',
}

UNRELIABLE_STAGE = 'CN_amyloid_positive'


def load_data():
    csv_path = Path('data/results/dnb/somascan/wgcna/stage_dnb_scores.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f'{csv_path} not found — run DNB pipeline first')
    df = pd.read_csv(csv_path)
    df = df.set_index('stage').loc[STAGE_ORDER].reset_index()
    return df


def _draw_bars(ax, x, values, stages, ylabel, title, annotate_n=True, df=None):
    """Draw bars with hatching for unreliable stage."""
    for i, (xi, val, stage) in enumerate(zip(x, values, stages)):
        color = STAGE_COLORS[stage]
        is_unreliable = (stage == UNRELIABLE_STAGE)
        bar = ax.bar(xi, val, color=color, width=0.6,
                     edgecolor='#333333', linewidth=0.5,
                     alpha=0.4 if is_unreliable else 1.0,
                     hatch='////' if is_unreliable else None)
        # If hatched, redraw edge to be visible
        if is_unreliable:
            ax.bar(xi, val, color='none', width=0.6,
                   edgecolor='#333333', linewidth=0.5)

    # Value annotations
    for i, (val, stage) in enumerate(zip(values, stages)):
        fmt = f'{val:.3f}' if max(values) < 1 else f'{val:.2f}'
        ax.text(i, val + max(values) * 0.03, fmt, ha='center', fontsize=10)
        if annotate_n and df is not None:
            n = int(df.iloc[i]['n_samples'])
            ax.text(i, -max(values) * 0.05, f'n={n}', ha='center',
                    fontsize=8, color='#888888')

    labels = [STAGE_LABELS[s] for s in stages]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_ylim(0, max(values) * 1.3)


def make_figure(df):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, PANEL_HEIGHT + 0.3))
    x = np.arange(len(df))
    stages = df['stage'].tolist()

    # ── Panel A: PCC_outside ─────────────────────────────────────────────
    vals_a = df['mean_pcc_outside'].values
    _draw_bars(ax_a, x, vals_a, stages,
               ylabel='Mean PCC$_{outside}$',
               title='Between-module decorrelation',
               annotate_n=True, df=df)
    add_panel_label(ax_a, 'A')

    # Trend line through interpretable points (skip CN_amyloid_positive at index 1)
    trend_idx = [0, 2, 3]
    trend_x = [x[i] for i in trend_idx]
    trend_y = [vals_a[i] for i in trend_idx]
    ax_a.plot(trend_x, trend_y, 'k--', linewidth=1.2, zorder=5)

    # Annotation arrow for trend
    mid_x = (trend_x[0] + trend_x[1]) / 2
    mid_y = (trend_y[0] + trend_y[1]) / 2
    ax_a.annotate('DNB theory\nprediction',
                  xy=(mid_x, mid_y), xytext=(mid_x + 0.6, mid_y + 0.08),
                  fontsize=8, ha='left', va='bottom',
                  arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor='#cccccc', alpha=0.9))

    # Bold red footnote
    ax_a.text(0.0, -0.20, '*n=6; interpret cautiously',
              transform=ax_a.transAxes, fontsize=9,
              fontweight='bold', color='#C0392B')

    # ── Panel B: Composite DNB score ─────────────────────────────────────
    vals_b = df['dnb_score'].values
    _draw_bars(ax_b, x, vals_b, stages,
               ylabel='Composite DNB score',
               title='DNB composite index',
               annotate_n=False)
    add_panel_label(ax_b, 'B')

    fig.tight_layout(w_pad=3)
    return fig


def main():
    df = load_data()
    fig = make_figure(df)
    save_figure(fig, 'data/results/figures/figure_03_perstage_dnb')
    fig.savefig('figure_03_fixed.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
