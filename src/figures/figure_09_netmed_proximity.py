import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from src.figures.figure_utils import set_style, add_panel_label, save_figure, PALETTE

set_style()

def make_figure():
    df = pd.read_csv('data/results/network_medicine/proximity_results.csv')

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.09, 3.0))

    # ── Panel A: Z-score bar chart ─────────────────────────────────────────
    labels = [
        'ADNI DNB proteins\n(self-proximity)',
        'AD DNB \u2192 PD DNB\n(cross-disease)'
    ]
    z_scores = df['z_score'].tolist()
    p_values = df['p_value'].tolist()
    colors = [PALETTE['converter'], PALETTE['highlight']]

    bars = ax_a.barh(labels, z_scores, color=colors, alpha=0.85,
                     height=0.45, edgecolor='white', linewidth=0.5)

    # Significance threshold line
    ax_a.axvline(-1.96, color='#333333', linewidth=1.0, linestyle='--')
    ax_a.text(-1.96, -0.55, 'p < 0.05\nthreshold\n(Z = \u22121.96)',
              ha='center', va='top', fontsize=6, color='#333333', style='italic')

    # Z=0 reference line
    ax_a.axvline(0, color='#999999', linewidth=0.6, linestyle=':')

    # Annotate Z and p on bars
    for i, (z, p) in enumerate(zip(z_scores, p_values)):
        p_str = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
        sig_marker = ' *' if p < 0.05 else ' ns'
        ax_a.text(z - 0.05 if z < 0 else z + 0.05,
                  i,
                  f'Z = {z:.2f}{sig_marker}\n{p_str}',
                  ha='right' if z < 0 else 'left',
                  va='center', fontsize=6.5, fontweight='bold',
                  color='white' if abs(z) > 1.5 else '#333333')

    ax_a.set_xlabel('Proximity Z-score\n(negative = closer than random)')
    ax_a.set_xlim(min(z_scores) - 1.5, max(z_scores) + 1.0)
    ax_a.set_title('Network proximity of DNB proteins\nin human interactome', fontsize=8)
    ax_a.invert_yaxis()
    add_panel_label(ax_a, 'A')

    # ── Panel B: conceptual module diagram ────────────────────────────────
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 7)
    ax_b.axis('off')
    ax_b.set_title('Network medicine interpretation', fontsize=8)

    # Background interactome (grey dots)
    np.random.seed(42)
    bg_x = np.random.uniform(0.3, 9.7, 80)
    bg_y = np.random.uniform(0.3, 6.7, 80)
    ax_b.scatter(bg_x, bg_y, s=8, color='#dddddd', zorder=1)

    # AD module circle
    ad_circle = plt.Circle((2.8, 3.5), 1.5, color='#4393c3', alpha=0.15,
                            zorder=2, linewidth=0)
    ax_b.add_patch(ad_circle)
    ad_border = plt.Circle((2.8, 3.5), 1.5, fill=False, color='#4393c3',
                            linewidth=1.5, zorder=3, linestyle='-')
    ax_b.add_patch(ad_border)
    ax_b.text(2.8, 5.3, 'AD DNB module\n(2 proteins:\nIL13, BCL2L10)', ha='center',
              fontsize=7, color='#4393c3', fontweight='bold')

    # Protein dots in AD module
    ad_dots_x = [2.6, 3.0]
    ad_dots_y = [3.7, 3.3]
    ax_b.scatter(ad_dots_x, ad_dots_y, s=30, color='#4393c3', zorder=4,
                 edgecolors='white', linewidth=0.5)

    # PD module circle (smaller — only CRYBB2)
    pd_circle = plt.Circle((6.8, 3.5), 0.8, color='#CC79A7', alpha=0.15,
                            zorder=2, linewidth=0)
    ax_b.add_patch(pd_circle)
    pd_border = plt.Circle((6.8, 3.5), 0.8, fill=False, color='#CC79A7',
                            linewidth=1.5, zorder=3, linestyle='-')
    ax_b.add_patch(pd_border)
    ax_b.text(6.8, 4.6, 'PD DNB module\n(CXCL12, 2 apt.)', ha='center',
              fontsize=7, color='#CC79A7', fontweight='bold')
    # CXCL12 dot
    ax_b.scatter([6.8], [3.5], s=80, color='#CC79A7', zorder=5,
                 marker='o', edgecolors='white', linewidth=0.5)
    ax_b.text(6.8, 3.1, 'CXCL12', ha='center', fontsize=6.5,
              color='#CC79A7', fontweight='bold')

    # Proximity arrow between modules
    prox_row = df[df['comparison'] == 'AD_PD_cross'].iloc[0]
    z_cross = prox_row['z_score']
    p_cross = prox_row['p_value']
    arrow_color = PALETTE['converter'] if p_cross < 0.05 else '#999999'
    arrow_label = ('Significant proximity\n(p < 0.05)' if p_cross < 0.05
                   else 'Not significant')

    ax_b.annotate('', xy=(5.8, 3.5), xytext=(4.3, 3.5),
                  arrowprops=dict(arrowstyle='<->', color=arrow_color,
                                  lw=2.0))
    ax_b.text(5.05, 3.85,
              f'Z = {z_cross:.2f}\n{arrow_label}',
              ha='center', fontsize=6.5, color=arrow_color, fontweight='bold')

    # Interactome label
    ax_b.text(5.0, 0.2, 'Human protein-protein interaction network\n'
              '(Menche et al. 2015; ~13,000 proteins, ~140,000 interactions)',
              ha='center', fontsize=6, color='#888888', style='italic')

    add_panel_label(ax_b, 'B')
    plt.tight_layout()
    return fig


def main():
    fig = make_figure()
    save_figure(fig, 'data/results/figures/figure_09_netmed_proximity')
    plt.close(fig)


if __name__ == '__main__':
    main()
