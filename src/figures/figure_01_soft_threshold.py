#!/usr/bin/env python3
"""
Figure 1: Soft-thresholding power selection for WGCNA
Output: data/results/figures/figure_01_soft_threshold
Data: data/results/wgcna/soft_threshold_fit.csv

(A) Scale-free topology fit index (R²) vs soft-thresholding power.
    Dashed line at R²=0.80. Power 9 selected (R²=0.876), highlighted red.
(B) Mean connectivity vs power. Power 9 highlighted.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from src.figures.figure_utils import (
    set_style, add_panel_label, save_figure,
    PALETTE, DOUBLE_COL, PANEL_HEIGHT,
)

set_style()
# Publication font overrides
plt.rcParams.update({
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

SELECTED_POWER = 9
R2_THRESHOLD = 0.80


def load_data():
    csv_path = Path('data/results/wgcna/soft_threshold_fit.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f'{csv_path} not found — run WGCNA pipeline first')
    df = pd.read_csv(csv_path)
    return df


def make_figure(df):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, PANEL_HEIGHT))

    powers = df['Power'].values
    r2 = df['SFT.R.sq'].values
    mean_k = df['mean.k.'].values

    sel_mask = df['Power'] == SELECTED_POWER
    sel_r2 = df.loc[sel_mask, 'SFT.R.sq'].values[0]
    sel_k = df.loc[sel_mask, 'mean.k.'].values[0]

    # ── Panel A: R² vs power ─────────────────────────────────────────────
    ax_a.plot(powers, r2, 'o-', color='#333333', markersize=4, zorder=2)
    ax_a.plot(SELECTED_POWER, sel_r2, 'o', color='#D55E00', markersize=7,
              zorder=3, label=f'Power {SELECTED_POWER} (R²={sel_r2:.3f})')
    ax_a.axhline(y=R2_THRESHOLD, color='#888888', linestyle='--', linewidth=0.8)
    # Position label at right terminus of the dashed line
    ax_a.text(1.0, R2_THRESHOLD, f'  R²={R2_THRESHOLD}',
              va='center', fontsize=10, color='#888888',
              transform=ax_a.get_yaxis_transform(), clip_on=False)

    ax_a.set_xlabel('Soft-thresholding power')
    ax_a.set_ylabel('Scale-free topology fit index (R²)')
    ax_a.set_ylim(0, 1.05)
    ax_a.legend(fontsize=8, loc='lower right')
    add_panel_label(ax_a, 'A')

    # ── Panel B: Mean connectivity vs power ──────────────────────────────
    ax_b.plot(powers, mean_k, 'o-', color='#333333', markersize=4, zorder=2)
    ax_b.plot(SELECTED_POWER, sel_k, 'o', color='#D55E00', markersize=7,
              zorder=3, label=f'Power {SELECTED_POWER} (k={sel_k:.1f})')

    ax_b.set_xlabel('Soft-thresholding power')
    ax_b.set_ylabel('Mean connectivity')
    ax_b.legend(fontsize=8, loc='upper right')
    add_panel_label(ax_b, 'B')

    fig.tight_layout(w_pad=3)
    return fig


def main():
    df = load_data()
    fig = make_figure(df)
    save_figure(fig, 'data/results/figures/figure_01_soft_threshold')
    fig.savefig('figure_01_fixed.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
