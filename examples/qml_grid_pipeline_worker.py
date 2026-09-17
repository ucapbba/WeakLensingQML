"""Run the QML/PCl/heracles pipeline for a single (n_arcmin2, footprint) combo.

Invoked as its own subprocess (one per combo) by examples/qml_grid_pipeline.py, rather than
being loaded in-process for all 9 combos: relinking cpp/build-release/libQMLShearLib.so against
a different n_pix_mask (a compile-time constant sized into the library's Eigen matrix types) and
reloading it via ctypes into a process that still has the previous build mapped caused heap
corruption (`malloc(): unsorted double linked list corrupted`) the first time the footprint (and
therefore n_pix_mask) changed. A fresh process per combo guarantees only one build of the library
is ever loaded at a time.

Usage: python3 qml_grid_pipeline_worker.py <n_arcmin2> <lo> <hi>
"""
import os
import subprocess
import sys
import time

import numpy as np
from scipy import constants
import healpy as hp
import healpy.sphtfunc
import pymaster as nmt
import heracles
import heracles.healpy

root_filepath = '/home/vscode/WeakLensingQML'
sys.path.append(root_filepath)

from python.lib.CppInterface.GeneratePowerSpectra import PowerSpec
from python.lib.CppInterface.GenerateCppConstants import generate_constants_h
from python.lib.CppInterface.CppLib import CppLib

glass_filepath = f'{root_filepath}/data'
tom_bin = 1

n_side = 64
n_pix = 12 * n_side * n_side
l_max = 2 * n_side
ells = np.arange(2, l_max + 1)
n_ell = len(ells)
l_max_full = 3 * n_side - 1
n_ell_full = l_max_full - 1

num_maps = 10
build_dir = f'{root_filepath}/cpp/build-release'
constants_h_path = f'{root_filepath}/cpp/detail/constants.h'
lib_path = f'{build_dir}/libQMLShearLib.so'

results_dir = f'{glass_filepath}/qml_grid_results'
os.makedirs(results_dir, exist_ok=True)

intrinsic_gal_ellip = 0.3
area_per_pix = 148_510_661 / n_pix

fiducial_cosmology = {'h': 0.7, 'Omega_c': 0.25, 'Omega_b': 0.05, 'sigma8': 0.75, 'n_s': 0.96,
                      'm_nu': 0.0, 'T_CMB': 2.7255, 'w0': -1.0, 'wa': 0.0}

# healpy's pixel-window-function download is blocked by SSL interception in this environment;
# point it at the local copy already fetched for examples/glass_fullsky_simulation.ipynb
healpy.sphtfunc.DATAURL = f'file://{root_filepath}/examples/'


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

    # Relink the QML C++ library against this combo's mask/noise_var (both are compile-time
    # constants) - an incremental `make`, not a full CMake reconfigure, since n_side (the only
    # thing here that could affect CMakeLists.txt) never changes
    generate_constants_h(n_side, mask, noise_var, num_maps, constants_h_path)
    subprocess.run(['make', 'QMLShearLib'], cwd=build_dir, check=True)

    cpp_lib = CppLib(lib_path, mask_filepath, cl_file_path)

    fisher_matrix_out = f'{glass_filepath}/Fisher_matrix_N{n_side}_glass_{combo_tag}.dat'
    start_time = time.time()
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
    # 2x resolution, per the base notebook) rather than NaMaster's
    mapper2 = heracles.healpy.HealpixMapper(2 * n_side, 2 * l_max_full)
    fieldsShe2 = {
        'SHE': heracles.Shears(mapper2, 'RA', 'DEC', 'E1', 'E2', 'W', mask='WHT'),
        'WHT': heracles.Weights(mapper2, 'RA', 'DEC', 'W'),
    }
    she_cat = catalog_bin
    she_cat.visibility = mask.astype(float)
    data_she2 = heracles.map_catalogs(fieldsShe2, {tom_bin: she_cat}, parallel=True)
    alms_she2 = heracles.transform(fieldsShe2, data_she2)
    cls_she_unbinned2 = heracles.angular_power_spectra(alms_she2, include=[('SHE', 'SHE'), ('WHT', 'WHT')])
    mixing_mats = heracles.mixing_matrices(fieldsShe2, cls_she_unbinned2)
    inv_mixing_mats = heracles.invert_mixing_matrix(mixing_mats)

    she_she_key = ('SHE', 'SHE', tom_bin, tom_bin)
    cls_she_decoupled = heracles.apply_mixing_matrix({she_she_key: cls_she_unbinned2[she_she_key]}, inv_mixing_mats)
    cl_EE_heracles = np.array(cls_she_decoupled[she_she_key][0, 0])[2: l_max + 1]
    cl_BB_heracles = np.array(cls_she_decoupled[she_she_key][1, 1])[2: l_max + 1]

    results_path = f'{results_dir}/results_{combo_tag}.npz'
    np.savez(
        results_path,
        n_arcmin2=n_arcmin2, footprint_lo=lo, footprint_hi=hi,
        ells=ells,
        sigma_ratio_EE=sigma_ratio_EE, sigma_ratio_BB=sigma_ratio_BB,
        cl_EE_QML=cl_EE_QML, cl_EE_PCl=cl_EE_PCl, cl_EE_heracles=cl_EE_heracles,
        cl_BB_QML=cl_BB_QML, cl_BB_PCl=cl_BB_PCl, cl_BB_heracles=cl_BB_heracles,
        cl_EE_theory=cl_EE[2: l_max + 1] + cl_noise, cl_noise=cl_noise,
    )
    print(f'Wrote results to {results_path} ({(time.time() - combo_start):,.1f} s total)', flush=True)


if __name__ == '__main__':
    main(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
