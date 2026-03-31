# src/figures/figure_utils.py

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ── Colour palette ──────────────────────────────────────────────────────────
# Colourblind-safe palette (Wong 2011, Nature Methods)
PALETTE = {
    'converter':        '#D55E00',   # vermillion — MCI→Dementia / PD_FAST
    'stable':           '#0072B2',   # blue — Stable MCI / PD_SLOW
    'cn_neg':           '#009E73',   # green — CN amyloid-negative (reference)
    'cn_pos':           '#F0E442',   # yellow — CN amyloid-positive (cautionary)
    'pd_intermediate':  '#CC79A7',   # pink — PD_INTERMEDIATE (tipping point peak)
    'pd_slow':          '#0072B2',   # blue — PD_SLOW reference
    'pd_fast':          '#D55E00',   # vermillion — PD_FAST
    'other':            '#999999',   # grey — unclassified / other
    'highlight':        '#E69F00',   # orange — CRYBB2 / cross-disease highlight
    'neutral':          '#56B4E9',   # sky blue — neutral annotations
}

# Biomarker ROC colours (distinct from trajectory palette)
ROC_PALETTE = {
    'Amyloid Status':    '#1b7837',
    'Aβ42/40 Ratio':     '#762a83',
    'APOE4 Genotype':    '#d6604d',
    'sDNB Score':        '#4393c3',
}

# ── Typography ───────────────────────────────────────────────────────────────
FONT_FAMILY  = 'Arial'        # universally available; Nature journals preference
FONT_SIZE    = 8              # base font size (pts) — journal single-column standard
TITLE_SIZE   = 9
LABEL_SIZE   = 8
TICK_SIZE    = 7
LEGEND_SIZE  = 7
PANEL_SIZE   = 11             # bold panel labels A, B, C

# ── Figure dimensions ────────────────────────────────────────────────────────
# Nature/Radiology: AI single-column = 88mm = 3.46 in; double-column = 180mm = 7.09 in
SINGLE_COL   = 3.46           # inches
DOUBLE_COL   = 7.09           # inches
PANEL_HEIGHT = 2.8            # inches per panel row

# ── Global rcParams ──────────────────────────────────────────────────────────
def set_style():
    mpl.rcParams.update({
        'font.family':        FONT_FAMILY,
        'font.size':          FONT_SIZE,
        'axes.titlesize':     TITLE_SIZE,
        'axes.labelsize':     LABEL_SIZE,
        'xtick.labelsize':    TICK_SIZE,
        'ytick.labelsize':    TICK_SIZE,
        'legend.fontsize':    LEGEND_SIZE,
        'figure.dpi':         300,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.05,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.linewidth':     0.8,
        'xtick.major.width':  0.8,
        'ytick.major.width':  0.8,
        'lines.linewidth':    1.2,
        'patch.linewidth':    0.8,
    })


def add_panel_label(ax, label, x=-0.18, y=1.05):
    """Add bold panel label (A, B, C) to top-left of axis."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=PANEL_SIZE, fontweight='bold',
            va='top', ha='left')


def format_pvalue(p):
    """Return formatted p-value string for annotations."""
    if p < 0.001:
        return 'p < 0.001'
    elif p < 0.01:
        return f'p = {p:.3f}'
    else:
        return f'p = {p:.2f}'


def save_figure(fig, path, formats=('pdf', 'png')):
    """Save figure in both PDF (for submission) and PNG (for preview)."""
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(f'{path}.{fmt}', dpi=300, bbox_inches='tight')
    print(f'Saved: {path}.pdf / .png')
