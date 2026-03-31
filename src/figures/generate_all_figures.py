"""
Master figure generation script.
Run: python src/figures/generate_all_figures.py

Generates all figures in order. Each figure script is independent
and can be run individually. This script runs them all.
"""

import sys
import traceback
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports resolve
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FIGURE_SCRIPTS = [
    'src/figures/figure_01_dnb_stages.py',
    'src/figures/figure_02_core_proteins.py',
    'src/figures/figure_03_sdnb.py',
    'src/figures/figure_04_biomarker_heatmap.py',
    'src/figures/figure_05_ppmi_replication.py',
    'src/figures/figure_06_crybb2_crossdisease.py',
    'src/figures/figure_07_summary.py',
    'src/figures/figure_08_netmed_lcc.py',
    'src/figures/figure_09_netmed_proximity.py',
]


def run_figure(script_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('fig', script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == '__main__':
    failed = []
    for script in FIGURE_SCRIPTS:
        print(f'\n=== Generating {Path(script).stem} ===')
        try:
            run_figure(script)
            print(f'\u2713 {Path(script).stem} complete')
        except Exception as e:
            print(f'\u2717 {Path(script).stem} FAILED: {e}')
            traceback.print_exc()
            failed.append(script)

    print(f'\n=== Summary: {len(FIGURE_SCRIPTS) - len(failed)}/{len(FIGURE_SCRIPTS)} figures generated ===')
    if failed:
        print('Failed:', failed)
        sys.exit(1)
