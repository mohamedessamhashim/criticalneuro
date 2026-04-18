#!/usr/bin/env python3
"""
Figure 2: WGCNA module dendrogram and color assignment
Output: data/results/figures/figure_02_module_overview
Data: data/results/wgcna/wgcna_modules.csv

Bar chart of module sizes (number of proteins per module), colored by
WGCNA module colour name. Grey module shown separately as unassigned.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from src.figures.figure_utils import (
    set_style, save_figure, SINGLE_COL, PANEL_HEIGHT,
)

set_style()
plt.rcParams.update({
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

# Map WGCNA module names to hex colours for plotting
_WGCNA_COLORS = {
    'turquoise': '#40E0D0', 'blue': '#0000FF', 'brown': '#8B4513',
    'yellow': '#FFD700', 'green': '#006400', 'red': '#FF0000',
    'black': '#000000', 'pink': '#FFB6C1', 'magenta': '#FF00FF',
    'purple': '#800080', 'greenyellow': '#ADFF2F', 'tan': '#D2B48C',
    'salmon': '#FA8072', 'cyan': '#00FFFF', 'midnightblue': '#191970',
    'lightcyan': '#E0FFFF', 'lightgreen': '#90EE90', 'lightyellow': '#FFFFE0',
    'royalblue': '#4169E1', 'darkred': '#8B0000', 'darkgreen': '#006400',
    'darkturquoise': '#00CED1', 'darkgrey': '#A9A9A9', 'orange': '#FFA500',
    'grey': '#BEBEBE',
}


def load_data():
    csv_path = Path('data/results/wgcna/wgcna_modules.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f'{csv_path} not found — run WGCNA pipeline first')
    df = pd.read_csv(csv_path)
    return df


def make_figure(df):
    counts = df['module'].value_counts()

    # Separate grey (unassigned) from real modules
    grey_count = counts.pop('grey') if 'grey' in counts.index else 0
    counts = counts.sort_values(ascending=False)

    n_assigned = len(counts)

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 1.5, PANEL_HEIGHT + 0.8))

    # Build y-positions: assigned modules at 0..n-1, then a gap, then grey
    y_assigned = np.arange(n_assigned)
    gap = 0.7
    y_grey = n_assigned + gap

    # Draw assigned module bars
    colors_assigned = [_WGCNA_COLORS.get(mod, '#BEBEBE') for mod in counts.index]
    ax.barh(y_assigned, counts.values, color=colors_assigned,
            edgecolor='#333333', linewidth=0.4)

    # Annotate assigned bars
    for i, val in enumerate(counts.values):
        ax.text(val + counts.max() * 0.01, i, str(val),
                va='center', fontsize=8, color='#333333')

    # Dashed separator line between assigned and grey
    sep_y = n_assigned - 0.5 + gap / 2
    ax.axhline(y=sep_y, color='#999999', linestyle='--', linewidth=0.8)

    # Draw grey bar
    if grey_count > 0:
        ax.barh(y_grey, grey_count, color='#BEBEBE',
                edgecolor='#333333', linewidth=0.4)
        ax.text(grey_count + counts.max() * 0.01, y_grey,
                f'{grey_count}  ',
                va='center', fontsize=8, color='#333333')
        ax.text(grey_count + counts.max() * 0.06, y_grey,
                '(excluded from DNB analysis)',
                va='center', fontsize=7, color='#888888', style='italic')

    # Y-axis labels
    all_y = list(y_assigned) + ([y_grey] if grey_count > 0 else [])
    all_labels = list(counts.index) + (['grey (unassigned)'] if grey_count > 0 else [])
    ax.set_yticks(all_y)
    ax.set_yticklabels(all_labels, fontsize=8)
    ax.invert_yaxis()

    ax.set_xlabel('Number of proteins')
    ax.set_title(f'WGCNA Module Sizes ({len(df):,} proteins total)', fontsize=11, pad=6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig


def main():
    df = load_data()
    fig = make_figure(df)
    save_figure(fig, 'data/results/figures/figure_02_module_overview')
    fig.savefig('figure_02_fixed.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
