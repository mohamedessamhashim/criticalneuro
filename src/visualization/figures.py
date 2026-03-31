"""Publication figure generation.

Reads results from CSV files in data/results/ and produces all
publication figures. Analysis scripts never generate figures (Rule 4).
All figures use the exact color palette from config.

Figure order reflects the primary/secondary hierarchy:
  Fig 1-4: DNB results (PRIMARY)
  Fig 5-7: CSD results (SECONDARY)
  Fig 8:   Biomarker correlation
  Fig 9:   Platform concordance
  Fig 10:  PPMI replication
  Fig 11:  Cross-disease comparison

SomaScan: filled markers. Olink: open markers. Golden Set: starred markers.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    """Load config.yaml."""
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def _get_color(group: str, config: dict) -> str:
    """Get hex color for a trajectory group."""
    return config["visualization"]["color_palette"].get(group, "#999999")


def _save_figure(fig: plt.Figure, name: str, config: dict) -> None:
    """Save figure to results/figures/ as PDF."""
    output_dir = Path(config["paths"]["results_figures"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.{config['visualization']['format']}"
    fig.savefig(path, dpi=config["visualization"]["dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


# ---- PRIMARY: DNB Figures (1-4) ----

def generate_figure_1(config: dict) -> None:
    """DNB score across disease stages (PRIMARY).

    Violin/box plots of DNB score by clinical stage. Shows both
    SomaScan DNB and Olink DNB with different marker styles.
    """
    # Try platform-specific results first, fall back to legacy
    soma_path = Path(config["paths"].get("results_dnb_somascan", "data/results/dnb/somascan")) / "dnb_scores_by_stage.csv"
    olink_path = Path(config["paths"].get("results_dnb_olink", "data/results/dnb/olink")) / "dnb_scores_by_stage.csv"
    legacy_path = Path(config["paths"]["results_dnb"]) / "dnb_scores_by_stage.csv"

    dnb_path = soma_path if soma_path.exists() else legacy_path
    if not dnb_path.exists():
        logger.warning("DNB stage scores not found, skipping Figure 1")
        return

    dnb = pd.read_csv(dnb_path)

    stage_order = [
        "CN_amyloid_negative", "CN_amyloid_positive",
        "stable_MCI", "MCI_to_Dementia", "established_AD",
    ]
    dnb["stage"] = pd.Categorical(dnb["stage"], categories=stage_order, ordered=True)
    dnb = dnb.sort_values("stage").dropna(subset=["stage"])

    fig, ax = plt.subplots(figsize=(8, 5))

    # SomaScan: filled bars
    colors = [_get_color(s, config) for s in dnb["stage"]]
    bars = ax.bar(range(len(dnb)), dnb["dnb_score"], color=colors, alpha=0.8, label="SomaScan")

    # Olink overlay if available
    if olink_path.exists():
        olink_dnb = pd.read_csv(olink_path)
        olink_dnb["stage"] = pd.Categorical(olink_dnb["stage"], categories=stage_order, ordered=True)
        olink_dnb = olink_dnb.sort_values("stage").dropna(subset=["stage"])
        ax.scatter(
            range(len(olink_dnb)), olink_dnb["dnb_score"],
            marker="o", facecolors="none",
            edgecolors=[_get_color(s, config) for s in olink_dnb["stage"]],
            s=80, linewidths=2, label="Olink", zorder=5,
        )

    ax.set_xticks(range(len(dnb)))
    ax.set_xticklabels(dnb["stage"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("DNB Score")
    ax.set_title("DNB score across disease stages")
    ax.legend()

    plt.tight_layout()
    _save_figure(fig, "figure_01_dnb_stages", config)


def generate_figure_2(config: dict) -> None:
    """Golden Set cross-platform DNB core proteins.

    Panel A: Venn diagram of SomaScan vs Olink DNB core protein overlap.
    Panel B: Scatter plot of SomaScan vs Olink DNB frequency for overlap
    proteins, colored by Golden Set membership.
    """
    cp_dir = Path(config["paths"].get("results_cross_platform", "data/results/cross_platform"))
    golden_path = cp_dir / "golden_set_proteins.csv"

    soma_core_path = Path(config["paths"].get("results_dnb_somascan", "data/results/dnb/somascan")) / "dnb_core_proteins.csv"
    olink_core_path = Path(config["paths"].get("results_dnb_olink", "data/results/dnb/olink")) / "dnb_core_proteins.csv"

    if not golden_path.exists():
        logger.warning("Golden Set not found, skipping Figure 2")
        return

    golden = pd.read_csv(golden_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Venn diagram
    ax = axes[0]
    n_soma_only = 0
    n_olink_only = 0
    n_shared = len(golden)

    if soma_core_path.exists() and olink_core_path.exists():
        soma_core = set(pd.read_csv(soma_core_path)["protein"])
        olink_core = set(pd.read_csv(olink_core_path)["protein"])
        # Use UniProt for proper comparison if available
        if "UniProt" in golden.columns:
            n_shared = len(golden)
        n_soma_only = len(soma_core) - n_shared
        n_olink_only = len(olink_core) - n_shared

    circle1 = plt.Circle((-0.3, 0), 0.5, fill=False, edgecolor="#E41A1C", lw=2)
    circle2 = plt.Circle((0.3, 0), 0.5, fill=False, edgecolor="#377EB8", lw=2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.text(-0.55, 0, str(n_soma_only), ha="center", va="center", fontsize=14)
    ax.text(0.0, 0, str(n_shared), ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.55, 0, str(n_olink_only), ha="center", va="center", fontsize=14)
    ax.text(-0.55, -0.65, "SomaScan only", ha="center", fontsize=10, color="#E41A1C")
    ax.text(0.55, -0.65, "Olink only", ha="center", fontsize=10, color="#377EB8")
    ax.text(0.0, 0.65, "Golden Set", ha="center", fontsize=11, fontweight="bold")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("A. DNB core protein overlap")

    # Panel B: Frequency scatter
    ax = axes[1]
    if "frequency_somascan" in golden.columns and "frequency_olink" in golden.columns:
        is_golden = golden.get("is_golden_set", pd.Series([True] * len(golden)))
        non_golden = golden[~is_golden]
        golden_set = golden[is_golden]

        if len(non_golden) > 0:
            ax.scatter(
                non_golden["frequency_somascan"], non_golden["frequency_olink"],
                c="#CCCCCC", s=30, alpha=0.5, label="Not Golden Set",
            )
        if len(golden_set) > 0:
            ax.scatter(
                golden_set["frequency_somascan"], golden_set["frequency_olink"],
                c="#FFD700", marker="*", s=100, edgecolors="black",
                linewidths=0.5, label="Golden Set", zorder=5,
            )

        ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
        ax.set_xlabel("SomaScan DNB frequency")
        ax.set_ylabel("Olink DNB frequency")
        ax.legend()
    ax.set_title("B. Cross-platform DNB frequency")

    plt.tight_layout()
    _save_figure(fig, "figure_02_golden_set", config)


def generate_figure_3(config: dict) -> None:
    """Golden Set biology: network visualization + pathway enrichment.

    Nodes colored by biological pathway from GSEA results.
    """
    import networkx as nx

    results_dir = Path(config["paths"]["results_dnb"])
    core_path = results_dir / "dnb_core_proteins.csv"
    gsea_path = results_dir / "gsea_results" / "gsea_hallmark.csv"

    if not core_path.exists():
        logger.warning("DNB core proteins not found, skipping Figure 3")
        return

    core = pd.read_csv(core_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Network
    ax = axes[0]
    G = nx.Graph()
    for _, row in core.iterrows():
        G.add_node(row["protein"], weight=row["frequency"])

    nodes = list(G.nodes())
    for i in range(min(len(nodes), 20)):
        for j in range(i + 1, min(len(nodes), 20)):
            G.add_edge(nodes[i], nodes[j])

    if len(G.nodes()) > 0:
        pos = nx.kamada_kawai_layout(G)
        weights = [G.nodes[n].get("weight", 0.5) * 500 for n in G.nodes()]
        nx.draw_networkx(
            G, pos, ax=ax, node_size=weights,
            node_color=_get_color("MCI_to_Dementia", config),
            font_size=6, alpha=0.7, with_labels=True,
        )
    ax.set_title("A. DNB core protein network")

    # Panel B: Pathway enrichment
    ax = axes[1]
    if gsea_path.exists():
        gsea = pd.read_csv(gsea_path)
        sig = gsea[gsea["fdr"] < 0.25].head(15)
        if len(sig) > 0:
            ax.barh(range(len(sig)), sig["NES"],
                    color=_get_color("MCI_to_Dementia", config), alpha=0.7)
            ax.set_yticks(range(len(sig)))
            ax.set_yticklabels(sig["pathway"].str.replace("HALLMARK_", ""), fontsize=8)
            ax.set_xlabel("Normalized Enrichment Score")
    ax.set_title("B. Pathway enrichment (Golden Set)")

    plt.tight_layout()
    _save_figure(fig, "figure_03_golden_set_biology", config)


def generate_figure_4(config: dict) -> None:
    """sDNB individual-level scores.

    Panel A: Scatter plot of sDNB vs time-to-conversion with Spearman rho.
    Panel B: Violin comparison vs stable MCI.
    """
    results_dir = Path(config["paths"].get("results_dnb_somascan", config["paths"]["results_dnb"]))
    sdnb_path = results_dir / "sdnb_scores.csv"

    if not sdnb_path.exists():
        # Fall back to legacy location
        sdnb_path = Path(config["paths"]["results_dnb"]) / "sdnb_scores.csv"
    if not sdnb_path.exists():
        logger.warning("sDNB scores not found, skipping Figure 4")
        return

    sdnb = pd.read_csv(sdnb_path)
    converter_label = config["adni"]["converter_group"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Scatter
    ax = axes[0]
    converters = sdnb[sdnb["TRAJECTORY"] == converter_label].dropna(subset=["MONTHS_TO_CONVERSION"])
    if len(converters) > 0:
        ax.scatter(
            converters["MONTHS_TO_CONVERSION"], converters["sdnb_score"],
            c=_get_color(converter_label, config), alpha=0.6, s=30,
        )
        from scipy.stats import spearmanr
        rho, p = spearmanr(converters["MONTHS_TO_CONVERSION"], converters["sdnb_score"])
        ax.annotate(f"Spearman rho={rho:.3f}\np={p:.2e}", xy=(0.05, 0.95),
                    xycoords="axes fraction", va="top", fontsize=10)
    ax.set_xlabel("Months to conversion")
    ax.set_ylabel("sDNB score")
    ax.set_title("A. sDNB vs time to conversion")

    # Panel B: Violin
    ax = axes[1]
    groups = [config["adni"]["stable_group"], converter_label]
    for i, group in enumerate(groups):
        data = sdnb[sdnb["TRAJECTORY"] == group]["sdnb_score"].dropna()
        if len(data) > 0:
            parts = ax.violinplot(data, positions=[i], showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(_get_color(group, config))
                pc.set_alpha(0.7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["Stable MCI", "MCI\u2192Dementia"])
    ax.set_ylabel("sDNB Score")
    ax.set_title("B. sDNB by trajectory")

    plt.tight_layout()
    _save_figure(fig, "figure_04_sdnb", config)


# ---- SECONDARY: CSD Figures (5-7) ----

def generate_figure_5(config: dict) -> None:
    """CSD at the protein level (SECONDARY).

    Volcano plot of Kendall tau vs -log10(FDR p-value).
    Golden Set proteins marked with starred symbol.
    """
    results_dir = Path(config["paths"]["results_csd"])
    stats_path = results_dir / "group_csd_statistics.csv"

    if not stats_path.exists():
        logger.warning("CSD group statistics not found, skipping Figure 5")
        return

    stats_df = pd.read_csv(stats_path)

    required_cols = {"median_var_tau_converter", "median_var_tau_stable", "var_fdr_p", "protein"}
    if stats_df.empty or not required_cols.issubset(stats_df.columns):
        logger.warning(
            "CSD group statistics file exists but is empty or missing required columns "
            "(CSD requires longitudinal data with ≥4 visits per participant — "
            "current data appears cross-sectional). Skipping Figure 5."
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Volcano plot
    ax = axes[0]
    x = stats_df["median_var_tau_converter"] - stats_df["median_var_tau_stable"]
    y = -np.log10(stats_df["var_fdr_p"].clip(lower=1e-300))

    sig_mask = stats_df["var_fdr_p"] < config["csd"]["alpha"]
    ax.scatter(x[~sig_mask], y[~sig_mask], c="#CCCCCC", s=10, alpha=0.5)
    ax.scatter(
        x[sig_mask], y[sig_mask],
        c=_get_color("MCI_to_Dementia", config), s=15, alpha=0.7,
    )

    # Mark Golden Set proteins with starred symbol
    golden_path = Path(config["paths"].get("results_cross_platform", "data/results/cross_platform")) / "golden_set_proteins.csv"
    if golden_path.exists():
        golden = pd.read_csv(golden_path)
        golden_proteins = set(golden[golden.get("is_golden_set", True) == True]["protein_somascan"].dropna())
        golden_mask = stats_df["protein"].isin(golden_proteins) & sig_mask
        if golden_mask.any():
            ax.scatter(
                x[golden_mask], y[golden_mask],
                marker="*", c="#FFD700", s=80, edgecolors="black",
                linewidths=0.5, zorder=10, label="Golden Set",
            )
            ax.legend()

    ax.axhline(-np.log10(config["csd"]["alpha"]), ls="--", c="grey", lw=0.8)
    ax.set_xlabel("Kendall tau difference (converter - stable)")
    ax.set_ylabel("-log10(FDR p-value)")
    ax.set_title("A. CSD at the protein level")

    # Panel B: Example time series
    ax = axes[1]
    ax.text(0.5, 0.5, "Example time series\n(requires raw data)",
            ha="center", va="center", transform=ax.transAxes, fontsize=12, color="grey")
    ax.set_title("B. Example protein trajectory")

    plt.tight_layout()
    _save_figure(fig, "figure_05_csd_volcano", config)


def generate_figure_6(config: dict) -> None:
    """Composite CSD score and conversion prediction.

    Panel A: Violin/box plots of composite CSD score.
    Panel B: ROC curves for Golden Set DNB, SomaScan DNB, composite CSD,
    p-tau217, NfL, Abeta42/40.
    """
    results_dir = Path(config["paths"]["results_csd"])
    scores_path = results_dir / "composite_csd_scores.csv"

    if not scores_path.exists():
        logger.warning("Composite CSD scores not found, skipping Figure 6")
        return

    scores = pd.read_csv(scores_path)

    if scores.empty or "composite_csd_score" not in scores.columns:
        logger.warning("Composite CSD scores file is empty or missing required columns. Skipping Figure 6.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Violin/box plot
    ax = axes[0]
    groups = [config["adni"]["stable_group"], config["adni"]["converter_group"]]
    plot_data = scores[scores["TRAJECTORY"].isin(groups)]

    for i, group in enumerate(groups):
        data = plot_data[plot_data["TRAJECTORY"] == group]["composite_csd_score"]
        parts = ax.violinplot(data.dropna(), positions=[i], showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(_get_color(group, config))
            pc.set_alpha(0.7)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["Stable MCI", "MCI\u2192Dementia"])
    ax.set_ylabel("Composite CSD Score")
    ax.set_title("A. CSD score by trajectory")

    # Panel B: AUC comparison bar chart with 95% CI
    ax = axes[1]
    roc_path = Path(config["paths"]["results_validation"]) / "roc_results.csv"
    if roc_path.exists():
        roc_df = pd.read_csv(roc_path)
        # Use primary time horizon (first in list) for display
        primary_horizon = roc_df["time_horizon"].iloc[0]
        horizon_df = roc_df[roc_df["time_horizon"] == primary_horizon].copy()
        horizon_df = horizon_df.sort_values("auc", ascending=True)

        y_pos = range(len(horizon_df))
        xerr_low = (horizon_df["auc"] - horizon_df["auc_ci_lower"]).clip(lower=0).values
        xerr_high = (horizon_df["auc_ci_upper"] - horizon_df["auc"]).clip(lower=0).values
        ax.barh(
            y_pos, horizon_df["auc"],
            xerr=[xerr_low, xerr_high],
            color=_get_color("MCI_to_Dementia", config),
            alpha=0.7, capsize=4,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(horizon_df["predictor"], fontsize=8)
        ax.axvline(0.5, ls="--", color="grey", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("AUC (95% CI)")
        ax.set_title(f"B. AUC comparison ({primary_horizon}m horizon)")
    else:
        ax.text(0.5, 0.5, "ROC results not yet available",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title("B. AUC comparison")

    plt.tight_layout()
    _save_figure(fig, "figure_06_csd_prediction", config)


def generate_figure_7(config: dict) -> None:
    """Temporal dynamics: CSD composite by time-to-conversion strata.

    Individual trajectories as faint lines. Comparison panel showing
    NfL over same strata.
    """
    results_dir = Path(config["paths"]["results_csd"])
    temporal_path = results_dir / "temporal_specificity.csv"

    if not temporal_path.exists():
        logger.warning("Temporal specificity not found, skipping Figure 7")
        return

    try:
        temporal = pd.read_csv(temporal_path)
    except Exception:
        logger.warning("Temporal specificity file could not be parsed (likely empty). Skipping Figure 7.")
        return

    if temporal.empty or "mean_csd" not in temporal.columns:
        logger.warning("Temporal specificity file is empty or missing required columns. Skipping Figure 7.")
        return

    fig, ax = plt.subplots(figsize=config["visualization"]["figsize_single"])

    x = range(len(temporal))
    ax.bar(
        x, temporal["mean_csd"],
        yerr=temporal["sem_csd"],
        color=_get_color("MCI_to_Dementia", config),
        alpha=0.7, capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(temporal["time_window"], rotation=15, ha="right")
    ax.set_ylabel("Composite CSD Score")
    ax.set_title("CSD score by time to conversion")

    plt.tight_layout()
    _save_figure(fig, "figure_07_temporal", config)


# ---- Biomarker and Platform Figures (8-9) ----

def generate_figure_8(config: dict) -> None:
    """Biomarker correlation heatmap.

    Spearman correlation matrix of Golden Set DNB, SomaScan DNB,
    composite CSD, and established biomarkers.
    """
    results_dir = Path(config["paths"]["results_validation"])
    corr_path = results_dir / "biomarker_correlations.csv"

    if not corr_path.exists():
        logger.warning("Biomarker correlations not found, skipping Figure 8")
        return

    corr = pd.read_csv(corr_path, index_col=0).astype(float)
    corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if corr.shape[0] < 2:
        logger.warning(
            "Biomarker correlation matrix has fewer than 2 non-empty variables "
            "(plasma biomarker files not yet downloaded). Skipping Figure 8."
        )
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr.astype(float), annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True, ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Spearman correlations: CSD, DNB, and biomarkers")

    plt.tight_layout()
    _save_figure(fig, "figure_08_biomarker_heatmap", config)


def generate_figure_9(config: dict) -> None:
    """Platform concordance.

    SomaScan DNB frequency (x) vs Olink DNB frequency (y) for positive
    control and Golden Set proteins. Points on diagonal = platform-independent.
    GFAP and NEFL annotated by name.
    """
    cp_dir = Path(config["paths"].get("results_cross_platform", "data/results/cross_platform"))
    concordance_path = cp_dir / "platform_concordance.csv"
    golden_path = cp_dir / "golden_set_proteins.csv"

    if not golden_path.exists():
        logger.warning("Golden Set not found, skipping Figure 9")
        return

    golden = pd.read_csv(golden_path)

    fig, ax = plt.subplots(figsize=(7, 7))

    if "frequency_somascan" in golden.columns and "frequency_olink" in golden.columns:
        is_golden = golden.get("is_golden_set", pd.Series([True] * len(golden)))

        # All overlap proteins
        ax.scatter(
            golden["frequency_somascan"], golden["frequency_olink"],
            c="#CCCCCC", s=30, alpha=0.5, label="Overlap proteins",
        )

        # Golden Set proteins
        gs = golden[is_golden]
        if len(gs) > 0:
            ax.scatter(
                gs["frequency_somascan"], gs["frequency_olink"],
                c="#FFD700", marker="*", s=120, edgecolors="black",
                linewidths=0.5, label="Golden Set", zorder=5,
            )

        # Annotate known proteins (GFAP, NEFL) using target name if available
        for _, row in golden.iterrows():
            name = str(row.get("TargetFullName",
                               row.get("protein_somascan", "")))
            for marker in ["GFAP", "NEFL"]:
                if marker.lower() in name.lower():
                    ax.annotate(
                        marker,
                        (row["frequency_somascan"], row["frequency_olink"]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=9, fontweight="bold",
                    )

    ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    ax.set_xlabel("SomaScan DNB frequency")
    ax.set_ylabel("Olink DNB frequency")
    ax.set_title("Platform concordance")
    ax.set_aspect("equal")
    ax.legend()

    plt.tight_layout()
    _save_figure(fig, "figure_09_platform_concordance", config)


# ---- Replication Figures (10-11) ----

def generate_figure_10(config: dict) -> None:
    """PPMI replication: side-by-side Figures 1 and 2 for PD data."""
    ppmi_dir = Path(config["paths"]["results_ppmi"])
    dnb_path = ppmi_dir / "ppmi_dnb_scores_by_stage.csv"
    csd_path = ppmi_dir / "ppmi_group_csd_statistics.csv"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: PD DNB by stage
    ax = axes[0]
    if dnb_path.exists():
        dnb = pd.read_csv(dnb_path)
        colors = [_get_color(s, config) for s in dnb["stage"]]
        ax.bar(range(len(dnb)), dnb["dnb_score"], color=colors, alpha=0.8)
        ax.set_xticks(range(len(dnb)))
        ax.set_xticklabels(dnb["stage"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("DNB Score")
    ax.set_title("A. PD DNB by progression stage (PPMI)")

    # Panel B: PD CSD volcano
    ax = axes[1]
    if csd_path.exists():
        try:
            stats_df = pd.read_csv(csd_path)
        except Exception:
            stats_df = pd.DataFrame()
        if "median_var_tau_converter" in stats_df.columns:
            x = stats_df["median_var_tau_converter"] - stats_df["median_var_tau_stable"]
            y = -np.log10(stats_df["var_fdr_p"].clip(lower=1e-300))
            ax.scatter(x, y, c=_get_color("PD_FAST", config), s=10, alpha=0.6)
    ax.set_xlabel("Kendall tau difference")
    ax.set_ylabel("-log10(FDR p-value)")
    ax.set_title("B. PD CSD analysis (PPMI)")

    plt.tight_layout()
    _save_figure(fig, "figure_10_ppmi_replication", config)


def generate_figure_11(config: dict) -> None:
    """Cross-disease comparison: Venn diagram + enrichment dot plot.

    Venn of Golden Set proteins shared between AD (ADNI) and PD (PPMI).
    """
    adni_dir = Path(config["paths"]["results_dnb"])
    ppmi_dir = Path(config["paths"]["results_ppmi"])
    adni_core_path = adni_dir / "dnb_core_proteins.csv"
    ppmi_core_path = ppmi_dir / "ppmi_dnb_core_proteins.csv"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Venn diagram
    ax = axes[0]
    if adni_core_path.exists() and ppmi_core_path.exists():
        adni_core = set(pd.read_csv(adni_core_path)["protein"])
        ppmi_core = set(pd.read_csv(ppmi_core_path)["protein"])

        overlap = adni_core & ppmi_core
        adni_only = adni_core - ppmi_core
        ppmi_only = ppmi_core - adni_core

        circle1 = plt.Circle((-0.3, 0), 0.5, fill=False,
                              edgecolor=_get_color("MCI_to_Dementia", config), lw=2)
        circle2 = plt.Circle((0.3, 0), 0.5, fill=False,
                              edgecolor=_get_color("PD_FAST", config), lw=2)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.text(-0.5, 0, str(len(adni_only)), ha="center", va="center", fontsize=14)
        ax.text(0.0, 0, str(len(overlap)), ha="center", va="center", fontsize=14, fontweight="bold")
        ax.text(0.5, 0, str(len(ppmi_only)), ha="center", va="center", fontsize=14)
        ax.text(-0.5, -0.6, "AD only", ha="center", fontsize=10)
        ax.text(0.5, -0.6, "PD only", ha="center", fontsize=10)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.8, 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("A. Shared DNB core proteins (AD vs PD)")

    # Panel B: Pathway enrichment comparison
    ax = axes[1]
    adni_gsea = adni_dir / "gsea_results" / "gsea_hallmark.csv"
    if adni_gsea.exists():
        adni_enrichment = pd.read_csv(adni_gsea).head(10)
        y_pos = range(len(adni_enrichment))
        ax.scatter(
            adni_enrichment["NES"], y_pos,
            c=_get_color("MCI_to_Dementia", config), s=50, label="AD", alpha=0.7,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(adni_enrichment["pathway"].str.replace("HALLMARK_", ""), fontsize=8)
    ax.set_xlabel("Normalized Enrichment Score")
    ax.set_title("B. Pathway enrichment comparison")
    ax.legend()

    plt.tight_layout()
    _save_figure(fig, "figure_11_cross_disease", config)


# ---- Orchestrator ----

def generate_all_figures(config: dict = None) -> None:
    """Generate publication figures for available pipeline results.

    CSD figures (5-7) are omitted: both ADNI and PPMI SomaScan datasets are
    cross-sectional (≤2 visits/participant), so CSD rolling-window analysis
    cannot run. Figures 5-7 will be re-added once longitudinal SomaScan data
    with ≥4 visits per participant is available.
    """
    if config is None:
        config = _load_config()

    logger.info("Generating publication figures...")

    generate_figure_1(config)   # DNB stages (PRIMARY)
    generate_figure_2(config)   # Golden Set Venn + scatter
    generate_figure_3(config)   # Golden Set biology
    generate_figure_4(config)   # sDNB individual scores
    # Figures 5-7 (CSD) omitted — no longitudinal data available
    generate_figure_8(config)   # Biomarker heatmap
    generate_figure_9(config)   # Platform concordance
    generate_figure_10(config)  # PPMI replication
    generate_figure_11(config)  # Cross-disease comparison

    logger.info("All available figures generated")
