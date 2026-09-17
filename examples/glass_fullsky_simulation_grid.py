"""Generate GLASS full-sky catalogues for a grid of galaxy densities and footprint cuts.

Same simulation pipeline as glass_fullsky_simulation.ipynb, but run once for every
combination of n_arcmin2 in (1, 10, 20) and (galactic, ecliptic) colatitude cut in
((55, 125), (65, 115), (75, 105)) - 9 combinations total. The matter/convergence
realisation is computed once and reused across all combinations (footprint and galaxy
density only affect galaxy sampling, not the underlying fields). Runs sequentially in a
single process - no multiprocessing.
"""
import sys

import numpy as np
import healpy as hp
import fitsio

import camb
from cosmology.compat.camb import Cosmology

import glass
import glass.ext.camb

root_filepath = '/home/vscode/WeakLensingQML'
sys.path.append(root_filepath)

rng = np.random.default_rng(seed=42)

# Fiducial cosmology, matching examples/glass_fullsky_simulation.ipynb
fiducial_cosmology = {'h': 0.7, 'Omega_c': 0.25, 'Omega_b': 0.05, 'sigma8': 0.75, 'n_s': 0.96}

pars = camb.set_params(
    H0=100 * fiducial_cosmology['h'],
    omch2=fiducial_cosmology['Omega_c'] * fiducial_cosmology['h'] ** 2,
    ombh2=fiducial_cosmology['Omega_b'] * fiducial_cosmology['h'] ** 2,
    NonLinear=camb.model.NonLinear_both,
)
pars.InitPower.set_params(As=2e-9, ns=fiducial_cosmology['n_s'])
pars.set_matter_power(redshifts=[0.0], kmax=2.0)

# Rescale As so that CAMB's sigma8 matches our fiducial sigma8 (Cl scales as As, sigma8^2 scales as As too)
sigma8_initial = camb.get_results(pars).get_sigma8()[-1]
As_scaled = 2e-9 * (fiducial_cosmology['sigma8'] / sigma8_initial) ** 2
pars.InitPower.set_params(As=As_scaled, ns=fiducial_cosmology['n_s'])

results = camb.get_background(pars)
cosmo = Cosmology(results)

# Resolution and matter shells
nside = lmax = 64

# shells of 200 Mpc in comoving distance spacing, out to z=3
zb = glass.distance_grid(cosmo, 0.0, 3.0, dx=200.0)
shells = glass.linear_windows(zb)

# angular matter power spectra of the shells, from CAMB
cls = glass.ext.camb.matter_cls(pars, lmax, shells)

# healpy's pixel window function download is blocked by SSL interception in this environment;
# point it at the local copy in examples/pixel_window_functions/ instead
import healpy.sphtfunc
healpy.sphtfunc.DATAURL = f'file://{root_filepath}/examples/'

# Matter fields: realised once and reused for every combination below, since neither the
# footprint cut nor the galaxy density affects the underlying matter/convergence fields
fields = glass.lognormal_fields(shells)
cls = glass.discretized_cls(cls, nside=nside, lmax=lmax, ncorr=3)
gls = glass.solve_gaussian_spectra(fields, cls)
matter = list(glass.generate(fields, gls, nside, ncorr=3, rng=rng))

# Convergence/shear per shell (depends only on the matter realisation above)
convergence = glass.MultiPlaneConvergence(cosmo)
shell_lensing = []
for i, delta_i in enumerate(matter):
    convergence.add_window(delta_i, shells[i])
    kappa_i = convergence.kappa
    gamm1_i, gamm2_i = glass.shear_from_convergence(kappa_i)
    shell_lensing.append((kappa_i, gamm1_i, gamm2_i))

sigma_e = 0.3

# Smail-type n(z) shape, matching examples/glass_fullsky_simulation.ipynb (unnormalised;
# scaled per n_arcmin2 below)
z_nz = np.arange(0.0, 3.0, 0.01)
dndz_shape = glass.smail_nz(z_nz, z_mode=0.9, alpha=2.0, beta=1.5)

n_arcmin2_values = [1, 10, 20]
footprints = [(55, 125), (65, 115), (75, 105)]

# Per-n_arcmin2 setup: normalised n(z) and its per-shell partition, saved once each
ngal_by_narcmin2 = {}
for n_arcmin2 in n_arcmin2_values:
    dndz = dndz_shape * n_arcmin2
    ngal_by_narcmin2[n_arcmin2] = glass.partition(z_nz, dndz, shells)

    nz_filepath = f'{root_filepath}/data/nz_glass_custom_ngal{n_arcmin2:g}.npz'
    np.savez(nz_filepath, z=z_nz, nz=dndz)
    print(f'Wrote redshift distribution to {nz_filepath}')

# Per-footprint setup: visibility mask, saved once each
vis_by_footprint = {}
for lo, hi in footprints:
    vis = glass.vmap_galactic_ecliptic(nside, galactic=np.radians((lo, hi)), ecliptic=np.radians((lo, hi)))
    vis_by_footprint[(lo, hi)] = vis

    f_sky = vis.sum() / vis.size
    print(f'f_sky of the ({lo}, {hi}) deg GLASS footprint at N_side={nside} is {(f_sky * 100):.3f} %')

    mask_filepath = f'{root_filepath}/data/masks/SkyMask_glass_N{nside}_fp{lo}_{hi}.fits'
    hp.write_map(mask_filepath, vis.astype(float), overwrite=True, fits_IDL=False, dtype=np.float64)
    print(f'Wrote mask to {mask_filepath}')

# Generate a catalogue (and its gamma1/gamma2 maps) for each n_arcmin2 x footprint
# combination, reusing the shared matter/convergence realisation above
n_pix = 12 * nside * nside
for n_arcmin2 in n_arcmin2_values:
    ngal = ngal_by_narcmin2[n_arcmin2]
    for lo, hi in footprints:
        vis = vis_by_footprint[(lo, hi)]
        tag = f'N{nside}_ngal{n_arcmin2:g}_fp{lo}_{hi}'
        print(f'--- {tag} ---')

        catalog_filepath = f'{root_filepath}/data/glass_catalog_{tag}.fits'
        with glass.write_catalog(catalog_filepath) as out:
            for i, (kappa_i, gamm1_i, gamm2_i) in enumerate(shell_lensing):
                for gal_lon, gal_lat, gal_count in glass.positions_from_delta(
                    ngal[i], matter[i], vis=vis, rng=rng,
                ):
                    gal_eps = glass.ellipticity_intnorm(gal_count, sigma_e, rng=rng, xp=np)
                    gal_she = glass.galaxy_shear(gal_lon, gal_lat, gal_eps, kappa_i, gamm1_i, gamm2_i)

                    out.write(
                        RA=gal_lon,
                        DEC=gal_lat,
                        E1=gal_she.real,
                        E2=gal_she.imag,
                        W=np.ones_like(gal_lon),
                        BIN=np.ones_like(gal_lon, dtype=np.int32),
                    )
        print(f'Wrote catalogue to {catalog_filepath}')

        # Quick validation: pixelise the catalogue into per-pixel mean-shear maps
        sum_w = np.zeros(n_pix)
        sum_w_e1 = np.zeros(n_pix)
        sum_w_e2 = np.zeros(n_pix)
        n_gal = 0

        catalog = fitsio.FITS(catalog_filepath)[1]
        for start in range(0, catalog.get_nrows(), 1_000_000):
            page = catalog[start:start + 1_000_000]
            pix = hp.ang2pix(nside, page['RA'], page['DEC'], lonlat=True)
            np.add.at(sum_w, pix, page['W'])
            np.add.at(sum_w_e1, pix, page['W'] * page['E1'])
            np.add.at(sum_w_e2, pix, page['W'] * page['E2'])
            n_gal += len(page)

        map_gamma1 = np.divide(sum_w_e1, sum_w, out=np.zeros(n_pix), where=sum_w > 0)
        map_gamma2 = np.divide(sum_w_e2, sum_w, out=np.zeros(n_pix), where=sum_w > 0)
        map_gamma1 *= vis
        map_gamma2 *= vis

        footprint_area_arcmin2 = vis.mean() * 4 * np.pi * (180 / np.pi) ** 2 * 3600
        print(f'Simulated {n_gal} galaxies (mean density {n_gal / footprint_area_arcmin2:.3f} /arcmin^2 within footprint)')

        map_gamma1_path = f'{root_filepath}/data/Map_glass_{tag}_gamma1.fits'
        map_gamma2_path = f'{root_filepath}/data/Map_glass_{tag}_gamma2.fits'
        hp.write_map(map_gamma1_path, map_gamma1, overwrite=True, fits_IDL=False, dtype=np.float64)
        hp.write_map(map_gamma2_path, map_gamma2, overwrite=True, fits_IDL=False, dtype=np.float64)
        print(f'Wrote gamma_1 map to {map_gamma1_path}')
        print(f'Wrote gamma_2 map to {map_gamma2_path}')
