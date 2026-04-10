"""Stage 5a: Network medicine — interactome proximity analysis.

Wraps src/network_medicine/interactome_proximity.py,
reading CTS proteins from BioTIP output instead of hardcoded lists.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def run_network_medicine_stage(cfg: dict) -> None:
    """Run interactome proximity analysis on CTS proteins.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration.
    """
    if not cfg.get("network_medicine", {}).get("run", True):
        logger.info("Network medicine disabled in config — skipping.")
        return

    output_dir = Path(cfg["output"]["dir"]) / "network_medicine"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CTS proteins from BioTIP output
    cts_path = Path(cfg["output"]["dir"]) / "biotip" / "biotip_cts_proteins.csv"
    if not cts_path.exists():
        logger.warning("BioTIP CTS proteins not found at %s — skipping network medicine", cts_path)
        return

    cts_df = pd.read_csv(cts_path)
    if len(cts_df) == 0:
        logger.warning("No CTS proteins found — skipping network medicine")
        return

    cts_proteins = cts_df["AptName"].tolist()
    logger.info("Running network medicine on %d CTS proteins", len(cts_proteins))

    # Try to run interactome proximity
    try:
        from src.network_medicine.interactome_proximity import run_proximity_analysis
        run_proximity_analysis(cts_proteins, cfg, output_dir)
    except ImportError:
        logger.warning("Network medicine module not available — skipping")
    except Exception as e:
        logger.warning("Network medicine failed: %s — continuing pipeline", e)
