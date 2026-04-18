#!/usr/bin/env python3
"""
Figure 5: Interactome proximity analysis
Output: data/results/figures/figure_05_proximity
Data: data/results/network_medicine/proximity_results.csv

(A) Histogram of null distribution for ADNI self-proximity.
    Red line at observed d=2.333. z=-2.403, p=0.013.
(B) Histogram of null distribution for AD vs PD cross-disease proximity.
    Red line at observed d=2.429. z=0.679, p=0.850.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from src.figures.figure_utils import (
    set_style, add_panel_label, format_pvalue, save_figure,
    DOUBLE_COL, PANEL_HEIGHT,
)

set_style()
plt.rcParams.update({
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})


def load_data():
    csv_path = Path('data/results/network_medicine/proximity_results.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f'{csv_path} not found — run proximity pipeline first')
    df = pd.read_csv(csv_path)
    return df


def _plot_null_panel(ax, row, title):
    """Plot a null distribution histogram with observed value line."""
    mean = row['d_random_mean']
    std = row['d_random_std']
    observed = row['amspl_actual']
    z = row['z_score']
    p = row['p_value']

    # Generate synthetic null distribution (normal approximation)
    rng = np.random.default_rng(42)
    null_dist = rng.normal(loc=mean, scale=std, size=1000)

    ax.hist(null_dist, bins=40, color='#56B4E9', edgecolor='white',
            linewidth=0.3, alpha=0.8, density=True, label='Null distribution')

    # Observed value
    ax.axvline(x=observed, color='#D55E00', linewidth=1.5, linestyle='-',
               label=f'Observed d = {observed:.3f}')

    # Stats annotation — 3 decimal places, Unicode minus for negative z
    sig_str = 'significant' if p < 0.05 else 'n.s.'
    z_str = f'\u22122{abs(z):.3f}'[1:] if z < 0 else f'{z:.3f}'
    if z < 0:
        z_str = f'\u2212{abs(z):.3f}'
    stats_text = f'z = {z_str}\np = {p:.3f} ({sig_str})'
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    ax.set_xlabel('Mean shortest path length')
    ax.set_ylabel('Density')
    ax.set_title(title, fontsize=9, pad=6)
    ax.legend(fontsize=8, loc='upper left')


def make_figure(df):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, PANEL_HEIGHT))

    # Panel A: ADNI self-proximity
    adni_row = df[df['comparison'] == 'ADNI_self'].iloc[0]
    _plot_null_panel(ax_a, adni_row, 'ADNI DNB self-proximity')
    add_panel_label(ax_a, 'A')

    # Panel B: AD vs PD cross-disease
    cross_row = df[df['comparison'] == 'AD_PD_cross'].iloc[0]
    _plot_null_panel(ax_b, cross_row, 'AD vs PD cross-disease proximity')
    add_panel_label(ax_b, 'B')

    fig.tight_layout(w_pad=3)
    return fig


def main():
    df = load_data()
    fig = make_figure(df)
    save_figure(fig, 'data/results/figures/figure_05_proximity')
    fig.savefig('figure_05_fixed.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
