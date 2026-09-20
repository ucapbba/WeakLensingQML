"""Driver: run the QML/PCl/heracles pipeline for all 9 GLASS grid combinations.

Each (n_arcmin2, footprint) combo is run in its own subprocess (qml_grid_pipeline_worker.py),
since relinking the QML C++ library in-place and reloading it via ctypes multiple times within
one long-lived process caused heap corruption the first time n_pix_mask (baked into the library
at compile time) changed between combos. A fresh process per combo means only one build of the
library is ever loaded at a time. Already-completed combos (an existing results .npz) are skipped,
so this can be safely re-run after fixing an error partway through.

Usage: python3 qml_grid_pipeline.py [--nside N]
"""
import argparse
import os
import subprocess
import sys

root_filepath = '/home/vscode/WeakLensingQML'
worker_path = f'{root_filepath}/examples/qml_grid_pipeline_worker.py'

parser = argparse.ArgumentParser()
parser.add_argument('--nside', type=int, default=64)
args = parser.parse_args()
n_side = args.nside

# Keep the original directory name for the already-published n_side=64 results
results_dir = (f'{root_filepath}/data/qml_grid_results' if n_side == 64
               else f'{root_filepath}/data/qml_grid_results_N{n_side}')

n_arcmin2_values = [1, 10, 20]
footprints = [(55, 125), (65, 115), (75, 105)]

for lo, hi in footprints:
    for n_arcmin2 in n_arcmin2_values:
        combo_tag = f'ngal{n_arcmin2:g}_fp{lo}_{hi}'
        results_path = f'{results_dir}/results_{combo_tag}.npz'
        if os.path.exists(results_path):
            print(f'=== {combo_tag} === already done, skipping')
            continue

        subprocess.run([sys.executable, worker_path, str(n_side), str(n_arcmin2), str(lo), str(hi)], check=True)
