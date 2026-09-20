"""
Standalone script to (re)generate the summary figures used in the LaTeX report as
standalone PNG files, for both the nside=64 and nside=128 QML grid result sets, plus
the TR1 real-catalogue sky-coverage and galaxy-density figures.

Run with: python examples/export_report_figures.py
"""
import os

import numpy as np
import healpy as hp
import fitsio
from matplotlib import pyplot as plt
import seaborn as sns

root_filepath = '/home/vscode/WeakLensingQML'
out_dir = f'{root_filepath}/examples/report_figures'
os.makedirs(out_dir, exist_ok=True)

plt.rcParams['text.usetex'] = False
sns.set(font_scale=1.1, rc={'text.usetex': False, 'figure.dpi': 150})

footprints = [(55, 125), (65, 115), (75, 105)]
n_arcmin2_values = [1, 10, 20]


def make_grid_figures(n_side, results_dir, tag):
    # --- masks ---
    fig = plt.figure(figsize=(4 * len(footprints), 3.5))
    for i, (lo, hi) in enumerate(footprints):
        mask_path = f'{root_filepath}/data/masks/SkyMask_glass_N{n_side}_fp{lo}_{hi}.fits'
        mask = hp.read_map(mask_path) > 0
        f_sky = mask.sum() / mask.size
        print(f'[N{n_side}] footprint ({lo}, {hi}) deg: f_sky = {(f_sky * 100):.3f} %')
        hp.mollview(mask, cmap='nipy_spectral', cbar=False, title=f'({lo}, {hi}) deg',
                    sub=(1, len(footprints), i + 1), fig=fig)
    fig.savefig(f'{out_dir}/masks_{tag}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- load results ---
    results = {}
    for n_arcmin2 in n_arcmin2_values:
        for lo, hi in footprints:
            combo_tag = f'ngal{n_arcmin2:g}_fp{lo}_{hi}'
            with np.load(f'{results_dir}/results_{combo_tag}.npz') as npz:
                results[(n_arcmin2, (lo, hi))] = {key: npz[key] for key in npz.files}
    print(f'[N{n_side}] loaded {len(results)} result files')

    # --- sigma ratio ---
    fig, axes = plt.subplots(len(n_arcmin2_values), len(footprints),
                              figsize=(4 * len(footprints), 3.2 * len(n_arcmin2_values)),
                              sharex=True, sharey=True, squeeze=False)
    for row, n_arcmin2 in enumerate(n_arcmin2_values):
        for col, (lo, hi) in enumerate(footprints):
            ax = axes[row, col]
            data = results[(n_arcmin2, (lo, hi))]
            ells = data['ells']
            ax.loglog(ells, data['sigma_ratio_EE'], c='cornflowerblue', lw=2, label='EE-EE')
            ax.loglog(ells, data['sigma_ratio_BB'], c='mediumseagreen', lw=2, label='BB-BB')
            if row == 0:
                ax.set_title(f'footprint ({lo}, {hi}) deg')
            if col == 0:
                ax.set_ylabel(f'n = {n_arcmin2:g} /arcmin$^2$\n' + r'$\sigma_{\ell}^{\mathrm{PCl}} / \sigma_{\ell}^{\mathrm{QML}}$')
            if row == len(n_arcmin2_values) - 1:
                ax.set_xlabel(r'$\ell$')
    axes[0, 0].legend()
    fig.suptitle(f'PCl / QML covariance ratio (N_side={n_side})')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/sigma_ratio_{tag}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- EE ---
    fig, axes = plt.subplots(len(n_arcmin2_values), len(footprints),
                              figsize=(4 * len(footprints), 3.2 * len(n_arcmin2_values)),
                              sharex=True, squeeze=False)
    for row, n_arcmin2 in enumerate(n_arcmin2_values):
        for col, (lo, hi) in enumerate(footprints):
            ax = axes[row, col]
            data = results[(n_arcmin2, (lo, hi))]
            ells = data['ells']
            ax.loglog(ells, ells * (ells + 1) * data['cl_EE_QML'] / (2 * np.pi), lw=1, c='cornflowerblue', label='QML')
            ax.loglog(ells, ells * (ells + 1) * data['cl_EE_PCl'] / (2 * np.pi), lw=1, c='mediumseagreen', label='PCl')
            ax.loglog(ells, ells * (ells + 1) * data['cl_EE_heracles'] / (2 * np.pi), lw=1, ls=':', c='darkorange', label='heracles')
            ax.loglog(ells, ells * (ells + 1) * data['cl_EE_theory'] / (2 * np.pi), lw=2, ls='--', c='hotpink', label='Theory')
            if row == 0:
                ax.set_title(f'footprint ({lo}, {hi}) deg')
            if col == 0:
                ax.set_ylabel(f'n = {n_arcmin2:g} /arcmin$^2$\n' + r'$\ell(\ell+1) C_{\ell}^{EE} / 2\pi$')
            if row == len(n_arcmin2_values) - 1:
                ax.set_xlabel(r'$\ell$')
    axes[0, 0].legend()
    fig.suptitle(f'EE power spectrum estimates (N_side={n_side})')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/EE_{tag}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- BB ---
    fig, axes = plt.subplots(len(n_arcmin2_values), len(footprints),
                              figsize=(4 * len(footprints), 3.2 * len(n_arcmin2_values)),
                              sharex=True, squeeze=False)
    for row, n_arcmin2 in enumerate(n_arcmin2_values):
        for col, (lo, hi) in enumerate(footprints):
            ax = axes[row, col]
            data = results[(n_arcmin2, (lo, hi))]
            ells = data['ells']
            cl_noise = float(data['cl_noise'])
            ax.loglog(ells, ells * (ells + 1) * np.abs(data['cl_BB_QML']) / (2 * np.pi), lw=1, c='cornflowerblue', label='QML')
            ax.loglog(ells, ells * (ells + 1) * np.abs(data['cl_BB_PCl']) / (2 * np.pi), lw=1, c='mediumseagreen', label='PCl')
            ax.loglog(ells, ells * (ells + 1) * np.abs(data['cl_BB_heracles']) / (2 * np.pi), lw=1, ls=':', c='darkorange', label='heracles')
            ax.loglog(ells, ells * (ells + 1) * cl_noise / (2 * np.pi), lw=2, ls='--', c='hotpink', label='Theory noise')
            if row == 0:
                ax.set_title(f'footprint ({lo}, {hi}) deg')
            if col == 0:
                ax.set_ylabel(f'n = {n_arcmin2:g} /arcmin$^2$\n' + r'$\ell(\ell+1) |C_{\ell}^{BB}| / 2\pi$')
            if row == len(n_arcmin2_values) - 1:
                ax.set_xlabel(r'$\ell$')
    axes[0, 0].legend()
    fig.suptitle(f'BB power spectrum estimates (N_side={n_side})')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/BB_{tag}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- BB/noise ratio table (for report text, not a figure) ---
    print(f'\n[N{n_side}] BB/noise ratio table:')
    print(f'{"n [/arcmin^2]":>15}{"footprint [deg]":>18}{"BB/noise (l<30)":>18}{"BB/noise (all l)":>18}')
    for n_arcmin2 in n_arcmin2_values:
        for lo, hi in footprints:
            data = results[(n_arcmin2, (lo, hi))]
            ratio = np.abs(data['cl_BB_QML']) / float(data['cl_noise'])
            low_ell = data['ells'] < 30
            print(f'{n_arcmin2:>15g}{f"({lo}, {hi})":>18}{ratio[low_ell].mean():>18.2f}{ratio.mean():>18.2f}')


def read_partial_map(path, nside, *, nest=False):
    """
    Read a HEALPix map in "partial" format from *path* and return it at resolution *nside*.
    The returned NSIDE cannot be larger than the NSIDE of the stored map.
    """
    data, header = fitsio.read(path, header=True)
    nside_in = header['NSIDE']
    fact = (nside_in // nside) ** 2
    if fact == 0:
        raise ValueError(f'requested NSIDE={nside} greater than map NSIDE={nside_in}')
    out = np.zeros(12 * nside ** 2)
    ipix, wht = data['PIXEL'], data['WEIGHT']
    order = header['ORDERING']
    if order == 'RING':
        ipix = hp.ring2nest(nside_in, ipix)
    elif order != 'NESTED':
        raise ValueError(f'unknown pixel ordering {order} in map')
    ipix = ipix // fact
    if not nest:
        ipix = hp.nest2ring(nside, ipix)
    np.add.at(out, ipix, wht / fact)
    return out


def make_tr1_figures():
    tr1_dir = f'{root_filepath}/TR1'
    pipeline_input_dir = f'{tr1_dir}/pipeline_input'
    n_side = 256

    coverage_filepath = f'{pipeline_input_dir}/EUC_LE3_VMPZ-ID_HPEFFECTIVECOVERAGE-CONCAT-20251204T094108.787432Z_0.12.fits'
    she_catalog_filepath = f'{pipeline_input_dir}/EUC_LE3_WLCATALOG-CONCAT-20251218T164508.418188Z_00.00.fits'

    coverage = read_partial_map(coverage_filepath, n_side)
    mask = (coverage > 0)

    f_sky = mask.sum() / mask.size
    area_mask = f_sky * 4 * np.pi  # steradians
    arcmin2_per_steradian = (180 / np.pi) ** 2 * 3600

    print(f'\n[TR1] f_sky of the real survey mask (n_side={n_side}) is {(f_sky * 100):.3f} %')

    fig = plt.figure(figsize=(9, 5))
    hp.mollview(mask, cmap='nipy_spectral', cbar=False, title='TR1 real survey effective-coverage mask', fig=fig)
    fig.savefig(f'{out_dir}/TR1_mask.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    she_data = fitsio.read(she_catalog_filepath, columns=['SHE_WEIGHT', 'TOM_BIN_ID'])
    she_weight_all = she_data['SHE_WEIGHT'].astype(np.float64)
    tom_bin_id = she_data['TOM_BIN_ID']
    tom_bins = np.unique(tom_bin_id)

    def effective_density(weights):
        n_eff = np.sum(weights) ** 2 / np.sum(weights ** 2)
        return n_eff, n_eff / (area_mask * arcmin2_per_steradian)

    bin_labels = [str(b) for b in tom_bins] + ['combined']
    density_per_bin = []
    for b in tom_bins:
        _, density = effective_density(she_weight_all[tom_bin_id == b])
        density_per_bin.append(density)
    _, density_combined = effective_density(she_weight_all)
    density_per_bin.append(density_combined)

    print(f'[TR1] {"Bin":>10} {"density [gal/arcmin^2]":>25}')
    for label, density in zip(bin_labels, density_per_bin):
        print(f'[TR1] {label:>10} {density:25.3f}')

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['cornflowerblue'] * len(tom_bins) + ['darkorange']
    ax.bar(bin_labels, density_per_bin, color=colors)
    ax.set_xlabel('Tomographic bin')
    ax.set_ylabel(r'Effective galaxy density [gal/arcmin$^2$]')
    ax.set_title('TR1 catalogue: effective galaxy density per bin')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/TR1_density_per_bin.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    make_grid_figures(64, f'{root_filepath}/data/qml_grid_results', 'N64')
    make_grid_figures(128, f'{root_filepath}/data/qml_grid_results_N128', 'N128')
    make_tr1_figures()
    print(f'\nAll figures written to {out_dir}')
