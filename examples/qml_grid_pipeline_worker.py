"""Run the QML/PCl/heracles pipeline for a single (n_side, n_arcmin2, footprint) combo.

Invoked as its own subprocess (one per combo) by examples/qml_grid_pipeline.py, rather than
being loaded in-process for all 9 combos: relinking cpp/build-release/libQMLShearLib.so against
a different n_pix_mask (a compile-time constant sized into the library's Eigen matrix types) and
reloading it via ctypes into a process that still has the previous build mapped caused heap
corruption (`malloc(): unsorted double linked list corrupted`) the first time the footprint (and
therefore n_pix_mask) changed. A fresh process per combo, with a private build per combo, guarantees
only one build of each library is ever loaded at a time.

QML is estimated per ell and, as for TR1 South, in bandpowers formed by binning the per-ell Fisher
matrix and y_ell before inverting.

Usage: python3 qml_grid_pipeline_worker.py <n_side> <n_arcmin2> <lo> <hi>
"""
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy import constants
import healpy as hp
import pymaster as nmt
import heracles
import heracles.healpy

root_filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_filepath)

from python.lib.CppInterface.GeneratePowerSpectra import PowerSpec
from python.lib.CppInterface.GenerateCppConstants import generate_constants_h
from python.lib.CppInterface.CppLib import CppLib

glass_filepath = f'{root_filepath}/data'
tom_bin = 1

n_side = int(sys.argv[1])
n_pix = 12 * n_side * n_side
l_max = 2 * n_side
ells = np.arange(2, l_max + 1)
n_ell = len(ells)
l_max_full = 3 * n_side - 1
n_ell_full = l_max_full - 1

num_maps = 10

# Optionally offload the (expensive) Fisher matrix to a remote machine via cpp/run_remote_fisher.sh,
# e.g. QML_REMOTE_HOST=bradley@192.168.1.105 QML_REMOTE_DIR=qml_remote. Unset = run it locally.
remote_host = os.environ.get('QML_REMOTE_HOST')
remote_dir = os.environ.get('QML_REMOTE_DIR', 'qml_remote')
remote_fisher_script = f'{root_filepath}/cpp/run_remote_fisher.sh'

# Keep the original directory name for the already-published n_side=64 results
results_dir = f'{glass_filepath}/qml_grid_results' if n_side == 64 else f'{glass_filepath}/qml_grid_results_N{n_side}'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{glass_filepath}/theory_cl_vals', exist_ok=True)

intrinsic_gal_ellip = 0.3
area_per_pix = 148_510_661 / n_pix

fiducial_cosmology = {'h': 0.7, 'Omega_c': 0.25, 'Omega_b': 0.05, 'sigma8': 0.75, 'n_s': 0.96,
                      'm_nu': 0.0, 'T_CMB': 2.7255, 'w0': -1.0, 'wa': 0.0}

# Bandpowers, exactly as for TR1 South: uniform bands of width 20 from l=10 up to l_max_full, the last band absorbing
# the remainder so it is never narrower than the others. The per-ell Fisher matrix and y_ell are binned *before*
# inverting, rather than inverting the per-ell Fisher matrix (unstable on small footprints) and binning afterwards.
# The same edges are used for NaMaster and heracles.
bandpower_width = 20
ell_bp_edges = np.r_[np.arange(10, l_max_full + 1 - bandpower_width, bandpower_width), l_max_full + 1]
ell_bp_ini, ell_bp_end = ell_bp_edges[:-1], ell_bp_edges[1:]
n_bands = len(ell_bp_ini)

# 0/1 indicator (not a 1/N average): for a flat bandpower model d(C_ell)/d(C_b) = 1, so F_b = W F W^T and y_b = W y
bin_matrix = np.zeros((n_bands, n_ell_full))
ells_full = np.arange(2, l_max_full + 1)
for b in range(n_bands):
    bin_matrix[b, (ells_full >= ell_bp_ini[b]) & (ells_full < ell_bp_end[b])] = 1.0
bin_matrix_EB = np.block([[bin_matrix, np.zeros_like(bin_matrix)], [np.zeros_like(bin_matrix), bin_matrix]])



def get_theory(n_arcmin2):
    cl_file_path = f'{glass_filepath}/theory_cl_vals/TheorySpectra_N{n_side}_glass_ngal{n_arcmin2:g}.dat'
    if os.path.exists(cl_file_path):
        cl_EE = np.loadtxt(cl_file_path)
        return cl_file_path, cl_EE

    with np.load(f'{glass_filepath}/nz_glass_custom_ngal{n_arcmin2:g}.npz') as npz:
        redshift_range = npz['z']
        redshift_dist = npz['nz']
    pow_spec = PowerSpec(n_side, fiducial_cosmology, redshift_range, redshift_dist, cl_file_path)
    pow_spec.compute_power_spec()
    return cl_file_path, pow_spec.cl_EE


def main(n_arcmin2, lo, hi):
    combo_tag = f'ngal{n_arcmin2:g}_fp{lo}_{hi}'
    print(f'=== {combo_tag} ===', flush=True)
    combo_start = time.time()

    cl_file_path, cl_EE = get_theory(n_arcmin2)

    mask_filepath = f'{glass_filepath}/masks/SkyMask_glass_N{n_side}_fp{lo}_{hi}.fits'
    mask = hp.read_map(mask_filepath) > 0
    footprint_area = mask.sum() * area_per_pix

    map_gamma1_path = f'{glass_filepath}/Map_glass_N{n_side}_{combo_tag}_gamma1.fits'
    map_gamma2_path = f'{glass_filepath}/Map_glass_N{n_side}_{combo_tag}_gamma2.fits'
    map_gamma1 = hp.read_map(map_gamma1_path)
    map_gamma2 = hp.read_map(map_gamma2_path)

    catalog_filepath = f'{glass_filepath}/glass_catalog_N{n_side}_{combo_tag}.fits'
    catalog = heracles.FitsCatalog(catalog_filepath)
    catalog_bin = catalog[f'BIN == {tom_bin}']
    n_gal_bin = catalog_bin.size

    avg_gal_den = n_gal_bin / footprint_area
    cl_noise = intrinsic_gal_ellip ** 2 / (avg_gal_den / (constants.arcminute ** 2))
    num_gal_per_pix = avg_gal_den * area_per_pix
    noise_var = (intrinsic_gal_ellip / np.sqrt(num_gal_per_pix)) ** 2

    # mask and noise_var are compile-time constants, so each combo builds its own private copy of cpp/. That means a
    # process only ever loads one build of each library path (relinking one shared library in place and reloading it
    # corrupted the heap), and the shared cpp/detail/constants.h is never modified
    cpp_dir = f'{glass_filepath}/qml_grid_cpp/N{n_side}_{combo_tag}'
    shutil.copytree(f'{root_filepath}/cpp', cpp_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns('build*'))
    generate_constants_h(n_side, mask, noise_var, num_maps, f'{cpp_dir}/detail/constants.h')
    build_dir = f'{cpp_dir}/build-release'
    subprocess.run(['cmake', '-S', cpp_dir, '-B', build_dir, '-DCMAKE_BUILD_TYPE=Release'], check=True, capture_output=True)
    subprocess.run(['cmake', '--build', build_dir, '--target', 'QMLShearLib', '-j'], check=True, capture_output=True)
    cpp_lib = CppLib(f'{build_dir}/libQMLShearLib.so', mask_filepath, cl_file_path)

    fisher_matrix_out = f'{glass_filepath}/Fisher_matrix_N{n_side}_glass_{combo_tag}.dat'
    start_time = time.time()
    if os.path.exists(fisher_matrix_out):
        # Already computed for this combo (e.g. by an earlier, interrupted run) - reuse it
        print(f'Reusing existing Fisher matrix {fisher_matrix_out}', flush=True)
    elif remote_host:
        # The remote script syncs this combo's private cpp/ copy, rebuilds there, and copies the Fisher matrix back
        subprocess.run(['bash', remote_fisher_script, remote_host, remote_dir,
                        mask_filepath, cl_file_path, fisher_matrix_out, cpp_dir], check=True)
    else:
        cpp_lib.compute_fisher_matrix(fisher_matrix_out)
    print(f'Fisher matrix computation took {(time.time() - start_time):,.1f} seconds', flush=True)

    fisher_matrix_QML = np.loadtxt(fisher_matrix_out)
    fisher_matrix_QML = np.delete(fisher_matrix_QML, np.s_[n_ell_full: 2 * n_ell_full], axis=0)
    fisher_matrix_QML = np.delete(fisher_matrix_QML, np.s_[n_ell_full: 2 * n_ell_full], axis=1)
    fisher_matrix_QML = (fisher_matrix_QML.copy() + fisher_matrix_QML.copy().T) / 2
    inv_fisher_QML = np.linalg.inv(fisher_matrix_QML)
    cov_EE_EE_QML = inv_fisher_QML[0: n_ell, 0: n_ell]
    cov_BB_BB_QML = inv_fisher_QML[n_ell_full: n_ell_full + n_ell, n_ell_full: n_ell_full + n_ell]

    # NaMaster's mode-coupling/covariance workspaces depend on the mask, so (unlike the shared
    # in-process version) they're rebuilt for every combo here - cheap next to the Fisher matrix
    field_mask = nmt.NmtField(mask, None, spin=2)
    bins = nmt.NmtBin.from_nside_linear(n_side, 1)
    wsp = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(field_mask, field_mask, bins)
    covar_wsp = nmt.NmtCovarianceWorkspace.from_fields(field_mask, field_mask, field_mask, field_mask)

    cl_EB = np.zeros_like(cl_EE)
    cl_BB = np.zeros_like(cl_EE)
    covar_22_22 = nmt.gaussian_covariance(
        covar_wsp, 2, 2, 2, 2,
        [cl_EE + cl_noise, cl_EB, cl_EB, cl_BB + cl_noise],
        [cl_EE + cl_noise, cl_EB, cl_EB, cl_BB + cl_noise],
        [cl_EE + cl_noise, cl_EB, cl_EB, cl_BB + cl_noise],
        [cl_EE + cl_noise, cl_EB, cl_EB, cl_BB + cl_noise],
        wsp, wb=wsp,
    ).reshape([n_ell_full, 4, n_ell_full, 4])
    cov_EE_EE_PCl = covar_22_22[:, 0, :, 0][0: n_ell, 0: n_ell]
    cov_BB_BB_PCl = covar_22_22[:, 3, :, 3][0: n_ell, 0: n_ell]

    sigma_ratio_EE = np.sqrt(np.diag(cov_EE_EE_PCl / cov_EE_EE_QML))
    sigma_ratio_BB = np.sqrt(np.diag(cov_BB_BB_PCl / cov_BB_BB_QML))

    y_ell_output = f'{glass_filepath}/y_ell_out_N{n_side}_glass_{combo_tag}.dat'
    cpp_lib.compute_power_spectrum(map_gamma1_path, map_gamma2_path, y_ell_output)
    y_ell_QML = np.loadtxt(y_ell_output)
    y_ell_QML = np.delete(y_ell_QML, np.s_[n_ell_full: 2 * n_ell_full], axis=0)
    cl_QML = inv_fisher_QML @ y_ell_QML
    cl_EE_QML = cl_QML[0: n_ell]
    cl_BB_QML = cl_QML[n_ell_full: n_ell_full + n_ell]

    field_shear = nmt.NmtField(mask, [map_gamma1, map_gamma2])
    cl_EE_PCl, cl_BB_PCl = nmt.compute_full_master(field_shear, field_shear, bins)[[0, 3], 0: n_ell]

    # heracles' own catalogue-to-Cl pipeline, decoupled with its own mixing matrix (built at
    # 2x resolution, per the base notebook) rather than NaMaster's. The shear spectrum keeps its shot noise
    # (debias=False), like QML (F^-1 y) and NaMaster, so all three are compared with signal + noise; the weight
    # spectrum used for the mixing matrices is debiased as usual
    mapper2 = heracles.healpy.HealpixMapper(2 * n_side, 2 * l_max_full)
    fieldsShe2 = {
        'SHE': heracles.Shears(mapper2, 'RA', 'DEC', 'E1', 'E2', 'W', mask='WHT'),
        'WHT': heracles.Weights(mapper2, 'RA', 'DEC', 'W'),
    }
    she_cat = catalog_bin
    she_cat.visibility = mask.astype(float)
    data_she2 = heracles.map_catalogs(fieldsShe2, {tom_bin: she_cat}, parallel=True)
    alms_she2 = heracles.transform(fieldsShe2, data_she2)
    cls_she_unbinned2 = heracles.angular_power_spectra(alms_she2, include=[('SHE', 'SHE')], debias=False)
    cls_wht2 = heracles.angular_power_spectra(alms_she2, include=[('WHT', 'WHT')])
    mixing_mats = heracles.mixing_matrices(fieldsShe2, cls_wht2)
    inv_mixing_mats = heracles.invert_mixing_matrix(mixing_mats)

    she_she_key = ('SHE', 'SHE', tom_bin, tom_bin)
    cls_she_decoupled = heracles.apply_mixing_matrix({she_she_key: cls_she_unbinned2[she_she_key]}, inv_mixing_mats)
    cl_EE_heracles = np.array(cls_she_decoupled[she_she_key][0, 0])[2: l_max + 1]
    cl_BB_heracles = np.array(cls_she_decoupled[she_she_key][1, 1])[2: l_max + 1]

    # --- Bandpowers (same edges for all three estimators) ---
    bins_bp = nmt.NmtBin.from_edges(ell_bp_ini, ell_bp_end)
    ell_bp = bins_bp.get_effective_ells()

    # QML: bin the per-ell Fisher matrix and y_ell before inverting
    inv_fisher_QML_bp = np.linalg.inv(bin_matrix_EB @ fisher_matrix_QML @ bin_matrix_EB.T)
    cl_QML_bp = inv_fisher_QML_bp @ (bin_matrix_EB @ y_ell_QML)
    cl_EE_QML_bp, cl_BB_QML_bp = cl_QML_bp[0: n_bands], cl_QML_bp[n_bands:]
    cov_EE_EE_QML_bp = inv_fisher_QML_bp[0: n_bands, 0: n_bands]
    cov_BB_BB_QML_bp = inv_fisher_QML_bp[n_bands:, n_bands:]

    # NaMaster on the same bandpowers
    wsp_bp = nmt.NmtWorkspace()
    wsp_bp.compute_coupling_matrix(field_mask, field_mask, bins_bp)
    covar_22_22_bp = nmt.gaussian_covariance(
        covar_wsp, 2, 2, 2, 2,
        [cl_EE + cl_noise, cl_EB, cl_EB, cl_BB + cl_noise],
        [cl_EE + cl_noise, cl_EB, cl_EB, cl_BB + cl_noise],
        [cl_EE + cl_noise, cl_EB, cl_EB, cl_BB + cl_noise],
        [cl_EE + cl_noise, cl_EB, cl_EB, cl_BB + cl_noise],
        wsp_bp, wb=wsp_bp,
    ).reshape([n_bands, 4, n_bands, 4])
    cov_EE_EE_PCl_bp = covar_22_22_bp[:, 0, :, 0]
    cov_BB_BB_PCl_bp = covar_22_22_bp[:, 3, :, 3]
    cl_EE_PCl_bp, cl_BB_PCl_bp = nmt.compute_full_master(field_shear, field_shear, bins_bp)[[0, 3]]

    # heracles: binned spectra decoupled with binned mixing matrices, then onto the same bands
    mixing_mats_bp = heracles.mixing_matrices(fieldsShe2, cls_wht2, l1max=l_max_full, l2max=l_max_full, bins=ell_bp_edges)
    cls_she_bp = heracles.apply_mixing_matrix(
        heracles.binned({she_she_key: cls_she_unbinned2[she_she_key]}, ell_bp_edges),
        heracles.invert_mixing_matrix(mixing_mats_bp), lmax=l_max_full)
    cl_EE_heracles_bp = bins_bp.bin_cell(np.array(cls_she_bp[she_she_key][0, 0])[: bins_bp.lmax + 1])
    cl_BB_heracles_bp = bins_bp.bin_cell(np.array(cls_she_bp[she_she_key][1, 1])[: bins_bp.lmax + 1])

    results_path = f'{results_dir}/results_{combo_tag}.npz'
    np.savez(
        results_path,
        n_arcmin2=n_arcmin2, footprint_lo=lo, footprint_hi=hi, f_sky=mask.mean(),
        ells=ells,
        sigma_ratio_EE=sigma_ratio_EE, sigma_ratio_BB=sigma_ratio_BB,
        cl_EE_QML=cl_EE_QML, cl_EE_PCl=cl_EE_PCl, cl_EE_heracles=cl_EE_heracles,
        cl_BB_QML=cl_BB_QML, cl_BB_PCl=cl_BB_PCl, cl_BB_heracles=cl_BB_heracles,
        cl_EE_theory=cl_EE[2: l_max + 1] + cl_noise, cl_noise=cl_noise,
        cov_EE_EE_QML=cov_EE_EE_QML, cov_BB_BB_QML=cov_BB_BB_QML,
        cov_EE_EE_PCl=cov_EE_EE_PCl, cov_BB_BB_PCl=cov_BB_BB_PCl,
        ell_bp=ell_bp, ell_bp_ini=ell_bp_ini, ell_bp_end=ell_bp_end,
        cl_EE_QML_bp=cl_EE_QML_bp, cl_BB_QML_bp=cl_BB_QML_bp,
        cl_EE_PCl_bp=cl_EE_PCl_bp, cl_BB_PCl_bp=cl_BB_PCl_bp,
        cl_EE_heracles_bp=cl_EE_heracles_bp, cl_BB_heracles_bp=cl_BB_heracles_bp,
        cov_EE_EE_QML_bp=cov_EE_EE_QML_bp, cov_BB_BB_QML_bp=cov_BB_BB_QML_bp,
        cov_EE_EE_PCl_bp=cov_EE_EE_PCl_bp, cov_BB_BB_PCl_bp=cov_BB_BB_PCl_bp,
        cl_EE_theory_bp=bins_bp.bin_cell(cl_EE[: bins_bp.lmax + 1]) + cl_noise,
    )
    print(f'Wrote results to {results_path} ({(time.time() - combo_start):,.1f} s total)', flush=True)


if __name__ == '__main__':
    main(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
