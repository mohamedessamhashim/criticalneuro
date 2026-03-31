#!/usr/bin/env python3
"""
Figure 7: Study Summary Overview (Graphical Abstract)
Output: data/results/figures/figure_07_summary
Data: hardcoded (graphical abstract — no pipeline data required)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
from src.figures.figure_utils import (
    set_style, save_figure, PALETTE, FONT_SIZE, DOUBLE_COL
)

set_style()


def _fancy_box(ax, x, y, w, h, text_lines, facecolor, edgecolor,
               fontsize=7, title=None, title_color='#222222'):
    """Draw a rounded box with stacked text lines."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle='round,pad=0.05',
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=1.2, zorder=2)
    ax.add_patch(box)

    if title:
        ax.text(x + w / 2, y + h - 0.10, title,
                ha='center', va='top', fontsize=fontsize + 1,
                fontweight='bold', color=title_color, zorder=3)
        text_y_start = y + h - 0.35  # increased gap: title → first content line
    else:
        text_y_start = y + h - 0.12

    line_spacing = (h - 0.45) / max(len(text_lines), 1) if title else h / max(len(text_lines), 1)
    for i, (line, color, fw, fs) in enumerate(text_lines):
        ax.text(x + w / 2, text_y_start - i * line_spacing,
                line, ha='center', va='top',
                fontsize=fs, color=color, fontweight=fw, zorder=3)


def make_figure():
    # Expanded height and ylim to eliminate compression and title/content overlap
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # ── Section 1: Data ──────────────────────────────────────────────────────
    # ADNI box (upper data box)
    _fancy_box(ax, 0.1, 3.6, 2.0, 2.0,
               title='ADNI',
               text_lines=[
                   ('CSF SomaScan 7k', '#333333', 'normal', 6),
                   ('n = 332', '#333333', 'normal', 6),
                   ('6,946 proteins', '#333333', 'normal', 6),
               ],
               facecolor='#EBF5FB', edgecolor='#4393c3',
               fontsize=7, title_color='#4393c3')

    # PPMI box (lower data box)
    _fancy_box(ax, 0.1, 0.8, 2.0, 2.0,
               title='PPMI',
               text_lines=[
                   ('CSF SomaScan', '#333333', 'normal', 6),
                   ('n = 1,250', '#333333', 'normal', 6),
                   ('4,785 proteins', '#333333', 'normal', 6),
               ],
               facecolor='#F9EBF8', edgecolor='#CC79A7',
               fontsize=7, title_color='#CC79A7')

    # ── Arrow 1 → Section 2 ──────────────────────────────────────────────────
    ax.annotate('', xy=(2.7, 3.0), xytext=(2.15, 3.0),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=2.0))

    # ── Section 2: Method ────────────────────────────────────────────────────
    method_box = FancyBboxPatch((2.75, 1.8), 2.3, 2.4,
                                boxstyle='round,pad=0.05',
                                facecolor='#FDFEFE', edgecolor='#666666',
                                linewidth=1.0, zorder=2)
    ax.add_patch(method_box)
    ax.text(3.90, 4.00, 'DNB Framework', ha='center', va='top',
            fontsize=8, fontweight='bold', color='#333333', zorder=3)
    ax.text(3.90, 3.65,
            r'DNB = ($\sigma_D$ × PCC$_D$) / |PCC$_O$|',
            ha='center', va='top', fontsize=6.5, color='#333333',
            style='italic', zorder=3)
    ax.text(3.90, 3.15,
            'Detects coordinated network\ndestabilization near tipping point',
            ha='center', va='top', fontsize=6, color='#555555', zorder=3)

    # ── Arrow 2 → Section 3 ──────────────────────────────────────────────────
    ax.annotate('', xy=(5.3, 3.0), xytext=(5.1, 3.0),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=2.0))

    # ── Section 3: Results ───────────────────────────────────────────────────
    # AD results box (upper)
    _fancy_box(ax, 5.35, 3.8, 2.25, 1.8,
               title='AD Result',
               text_lines=[
                   ('MCI\u2192Dementia: DNB 7.49', '#D55E00', 'bold', 6),
                   ('2 core proteins', '#333333', 'normal', 6),
                   ('IL13, BCL2L10', '#D55E00', 'normal', 6),
               ],
               facecolor='#FEF9F5', edgecolor='#4393c3',
               fontsize=7, title_color='#4393c3')

    # PD results box (lower)
    _fancy_box(ax, 5.35, 1.8, 2.25, 1.8,
               title='PD Result',
               text_lines=[
                   ('PD Intermediate: 0.565', '#CC79A7', 'bold', 6),
                   ('Inverted-U pattern', '#333333', 'normal', 6),
                   ('Core: CXCL12', '#CC79A7', 'normal', 6),
               ],
               facecolor='#FDF5FC', edgecolor='#CC79A7',
               fontsize=7, title_color='#CC79A7')

    # Findings note box — no CRYBB2, honest assessment
    findings_box = FancyBboxPatch((5.35, 0.5), 2.25, 0.9,
                                boxstyle='round,pad=0.04',
                                facecolor='#FFF9E6', edgecolor='#E69F00',
                                linewidth=1.2, zorder=3)
    ax.add_patch(findings_box)
    ax.text(6.475, 0.95,
            'sDNB AUC \u2248 0.51 (near chance)\nValidation needed',
            ha='center', va='center', fontsize=6.5, fontweight='bold',
            color='#E69F00', zorder=4)

    # ── Arrow 3 → Section 4 ──────────────────────────────────────────────────
    ax.annotate('', xy=(7.85, 3.0), xytext=(7.65, 3.0),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=2.0))

    # ── Section 4: Conclusion ────────────────────────────────────────────────
    conc_box = FancyBboxPatch((7.90, 0.5), 2.0, 5.0,
                              boxstyle='round,pad=0.05',
                              facecolor='#F8F9FA', edgecolor='#444444',
                              linewidth=1.0, zorder=2)
    ax.add_patch(conc_box)
    ax.text(8.90, 5.25, 'Conclusions', ha='center', va='top',
            fontsize=8, fontweight='bold', color='#222222', zorder=3)

    conclusion_lines = [
        'First DNB application',
        'at SomaScan 7k scale',
        '',
        'Disease-specific core',
        'proteins (no overlap)',
        '',
        'sDNB near chance level',
        '(cross-validation needed)',
        '',
        'Longitudinal validation',
        '(Knight-ADRC collaboration)',
    ]
    for i, line in enumerate(conclusion_lines):
        ax.text(8.90, 4.90 - i * 0.38, line,
                ha='center', va='top', fontsize=6,
                color='#333333' if line else '#ffffff', zorder=3)

    fig.tight_layout(pad=0.3)
    return fig


def main():
    fig = make_figure()
    save_figure(fig, 'data/results/figures/figure_07_summary')
    plt.close(fig)


if __name__ == '__main__':
    main()
