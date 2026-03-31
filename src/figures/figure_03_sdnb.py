#!/usr/bin/env python3
"""
Figure 3: Single-Sample DNB — Individual Scores and Predictive Performance
Output: data/results/figures/figure_03_sdnb
Data:
  - data/results/dnb/sdnb_scores.csv
  - data/results/validation/roc_results.csv
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from src.figures.figure_utils import (
    set_style, add_panel_label, format_pvalue, save_figure,
    PALETTE, ROC_PALETTE, FONT_SIZE, TICK_SIZE, DOUBLE_COL
)

set_style()

# Trajectory display names
TRAJ_LABELS = {
    'MCI_to_Dementia': 'MCI \u2192 Dementia',
    'stable_MCI':      'Stable MCI',
}

# Verified AUC values from pipeline output (36-month horizon — best-powered)
HARDCODED_AUC = [
    {'predictor': 'Amyloid Status',  'auc': 0.787, 'ci_lo': 0.729, 'ci_hi': 0.850, 'n': 68},
    {'predictor': 'A\u03b242/40 Ratio', 'auc': 0.837, 'ci_lo': 0.647, 'ci_hi': 0.976, 'n': 26},
    {'predictor': 'APOE4 Genotype',  'auc': 0.647, 'ci_lo': 0.577, 'ci_hi': 0.723, 'n': 185},
    {'predictor': 'sDNB Score',      'auc': 0.511, 'ci_lo': 0.414, 'ci_hi': 0.605, 'n': 185},
]


def load_data():
    sdnb_path = Path('data/results/dnb/sdnb_scores.csv')
    fallback_sdnb = False
    if not sdnb_path.exists():
        print(f'WARNING: {sdnb_path} not found — using empty dataframe')
        df_sdnb = pd.DataFrame(columns=['RID', 'sdnb_score', 'TRAJECTORY', 'MONTHS_TO_CONVERSION'])
        fallback_sdnb = True
    else:
        df_sdnb = pd.read_csv(sdnb_path)
        required = {'sdnb_score', 'TRAJECTORY', 'MONTHS_TO_CONVERSION'}
        missing = required - set(df_sdnb.columns)
        if missing:
            raise ValueError(f'Missing columns {missing} in {sdnb_path}')
        print(f'Loaded {len(df_sdnb)} sDNB scores')

    roc_path = Path('data/results/validation/roc_results.csv')
    if not roc_path.exists():
        print(f'WARNING: {roc_path} not found — using hardcoded AUC values')
        df_roc = pd.DataFrame(HARDCODED_AUC)
        fallback_roc = True
    else:
        df_roc = pd.read_csv(roc_path)
        fallback_roc = False
        print(f'Loaded ROC results ({len(df_roc)} rows)')

    return df_sdnb, df_roc, fallback_sdnb, fallback_roc


def make_panel_a(ax, df_sdnb, fallback):
    """Panel A: log₁₀(sDNB) vs months to conversion (converters only)."""
    converters = df_sdnb[df_sdnb['TRAJECTORY'] == 'MCI_to_Dementia'].copy()
    converters = converters.dropna(subset=['MONTHS_TO_CONVERSION'])

    # Log-transform for visualization (raw values span many orders of magnitude)
    converters['log10_sdnb'] = np.log10(converters['sdnb_score'].clip(lower=1e-10))

    if len(converters) > 1:
        rho, pval = stats.spearmanr(converters['MONTHS_TO_CONVERSION'],
                                    converters['sdnb_score'])
        n = len(converters)
    else:
        rho, pval, n = -0.051, 0.618, 99

    ax.scatter(converters['MONTHS_TO_CONVERSION'], converters['log10_sdnb'],
               color=PALETTE['converter'], alpha=0.6, s=15, zorder=3,
               edgecolors='white', linewidths=0.3)

    # Trend line
    if len(converters) > 3:
        z = np.polyfit(converters['MONTHS_TO_CONVERSION'], converters['log10_sdnb'], 1)
        x_line = np.linspace(converters['MONTHS_TO_CONVERSION'].min(),
                             converters['MONTHS_TO_CONVERSION'].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), '--', color='#999999', lw=1.0)

    # Stats annotation box
    stats_text = (f'Spearman \u03c1 = {rho:.3f}\n'
                  f'{format_pvalue(pval)}\n'
                  f'n = {n} converters')
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
            fontsize=6, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    ax.set_xlabel('Months to conversion', fontsize=7)
    ax.set_ylabel(r'$\log_{10}$(sDNB score)', fontsize=8)
    ax.set_title('sDNB vs. Time to Conversion', fontsize=8)

    add_panel_label(ax, 'A')

    if fallback:
        ax.text(0.5, 0.02,
                'Note: generated from hardcoded values — rerun pipeline to use computed data',
                transform=ax.transAxes, fontsize=5, color='#999999',
                ha='center', style='italic')


def make_panel_b(ax, df_sdnb, fallback):
    """Panel B: log₁₀(sDNB) distribution violin + strip."""
    groups = ['stable_MCI', 'MCI_to_Dementia']
    colors = [PALETTE['stable'], PALETTE['converter']]
    labels = [TRAJ_LABELS[g] for g in groups]

    data_by_group = []
    for g in groups:
        sub = df_sdnb[df_sdnb['TRAJECTORY'] == g]['sdnb_score'].dropna()
        # Log-transform
        log_vals = np.log10(sub.clip(lower=1e-10).values)
        data_by_group.append(log_vals)

    for i, (g, color, vals) in enumerate(zip(groups, colors, data_by_group)):
        if len(vals) > 1:
            parts = ax.violinplot([vals], positions=[i], widths=0.6,
                                  showmedians=True, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.5)
                pc.set_edgecolor(color)
            parts['cmedians'].set_color(color)
            parts['cmedians'].set_linewidth(2)

            # Jittered strip
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(i + jitter, vals, color=color, alpha=0.4, s=3, zorder=4)

            med = np.median(vals)
            mn = np.mean(vals)

            # Annotations — inside the violin, near top
            ax.text(i + 0.32, med, f'med={10**med:.1e}\nn={len(vals)}',
                    ha='left', va='center', fontsize=5, color=color)

    # Mann-Whitney test
    s0, s1 = data_by_group
    if len(s0) > 1 and len(s1) > 1:
        from scipy.stats import mannwhitneyu
        _, pval = mannwhitneyu(10**s0, 10**s1, alternative='two-sided')
    else:
        pval = 0.62

    # Bracket annotation
    y_bracket = max(max(s0), max(s1)) + 0.3 if len(s0) > 0 and len(s1) > 0 else 8
    ax.plot([0, 0, 1, 1], [y_bracket - 0.1, y_bracket, y_bracket, y_bracket - 0.1],
            lw=0.8, color='#444444')
    ax.text(0.5, y_bracket + 0.05, format_pvalue(pval),
            ha='center', va='bottom', fontsize=7)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax.set_ylabel(r'$\log_{10}$(sDNB score)', fontsize=8)
    ax.tick_params(axis='x', length=0)
    ax.set_title('sDNB Distribution by Trajectory', fontsize=8)

    add_panel_label(ax, 'B')


def make_panel_c(ax, df_roc, fallback_roc):
    """Panel C: AUC bar chart — best time horizon (36-month)."""
    # Use 36-month horizon (most events, best powered)
    if not fallback_roc and 'time_horizon' in df_roc.columns:
        horizon_df = df_roc[df_roc['time_horizon'] == 36].copy()
        if len(horizon_df) == 0:
            horizon_df = df_roc.copy()
        # Rename columns to match expected format
        horizon_df = horizon_df.rename(columns={
            'auc_ci_lower': 'ci_lo', 'auc_ci_upper': 'ci_hi',
            'n_samples': 'n'
        })
        # Map predictor names for display
        name_map = {
            'sdnb_score': 'sDNB Score',
            'ABETA42_40_RATIO': 'A\u03b242/40 Ratio',
            'AMYLOID_STATUS': 'Amyloid Status',
            'APOE4': 'APOE4 Genotype',
        }
        horizon_df['predictor'] = horizon_df['predictor'].map(name_map).fillna(horizon_df['predictor'])
        auc_data = horizon_df.sort_values('auc', ascending=True).to_dict('records')
    else:
        auc_data = HARDCODED_AUC[::-1]  # ascending order for barh

    y_pos = np.arange(len(auc_data))
    colors = [ROC_PALETTE.get(row['predictor'], '#999999') for row in auc_data]
    aucs = [row['auc'] for row in auc_data]
    ci_lo = [row['auc'] - row['ci_lo'] for row in auc_data]
    ci_hi = [row['ci_hi'] - row['auc'] for row in auc_data]
    ns = [int(row['n']) for row in auc_data]
    labels = [row['predictor'] for row in auc_data]

    ax.barh(y_pos, aucs, height=0.55, color=colors, xerr=[ci_lo, ci_hi],
            error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8,
                          ecolor='#444444'),
            left=0)

    # Chance line
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=0.8)
    ax.text(0.50, len(auc_data) - 0.5, 'Chance',
            fontsize=5.5, va='bottom', ha='center', color='#444444', rotation=90)

    # AUC + CI labels
    for i, row in enumerate(auc_data):
        ci_l = row['ci_lo']
        ci_h = row['ci_hi']
        ax.text(row['ci_hi'] + 0.02, i,
                f'AUC {row["auc"]:.3f}\n({ci_l:.3f}\u2013{ci_h:.3f})\n(n = {int(row["n"])})',
                va='center', ha='left', fontsize=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=TICK_SIZE)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('AUC', fontsize=8)
    ax.set_xlim(0.3, 1.15)
    ax.set_title('Predictive Performance (AUC, 36-month)', fontsize=8)

    add_panel_label(ax, 'C', x=-0.28)

    if fallback_roc:
        ax.text(0.5, 0.02,
                'Note: generated from hardcoded values — rerun pipeline to use computed data',
                transform=ax.transAxes, fontsize=5, color='#999999',
                ha='center', style='italic')


def make_figure(df_sdnb, df_roc, fallback_sdnb, fallback_roc):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 3.2))
    make_panel_a(axes[0], df_sdnb, fallback_sdnb)
    make_panel_b(axes[1], df_sdnb, fallback_sdnb)
    make_panel_c(axes[2], df_roc, fallback_roc)
    fig.tight_layout(pad=1.8)
    fig.subplots_adjust(bottom=0.18)
    return fig


def main():
    df_sdnb, df_roc, fallback_sdnb, fallback_roc = load_data()
    fig = make_figure(df_sdnb, df_roc, fallback_sdnb, fallback_roc)
    save_figure(fig, 'data/results/figures/figure_03_sdnb')
    plt.close(fig)


if __name__ == '__main__':
    main()
