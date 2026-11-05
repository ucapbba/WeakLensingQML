"""
Helpers for the DR1 shear notebooks: which tiles belong to which field, per-pixel sums from the tile catalogue, and
a heracles catalogue that streams the Parquet tiles.

The catalogues (dr1_dps_tiles_v6/{metacal,lensmc}_only, see data/README_v6.md) are one Parquet file per tile, ~10^8
galaxies in all, too many to hold in memory at once. Everything here reads one tile at a time.
"""
import glob
import json
import os

import healpy as hp
import numpy as np
import pyarrow.parquet as pq
from heracles.catalog.base import CatalogBase, CatalogPage

# Catalogue columns for each shear method: position, calibrated ellipticities and the calibrated weight (the raw
# SHE_LENSMC_WEIGHT is a 0/1 flag and the raw SHE_METACAL_WEIGHT a placeholder, see the README)
COLUMNS = {
    'metacal': {'ra': 'SHE_METACAL_RA', 'dec': 'SHE_METACAL_DEC', 'e1': 'SHE_METACAL_E1_CORRECTED',
                'e2': 'SHE_METACAL_E2_CORRECTED', 'w': 'SHE_METACAL_WEIGHT_CORRECTED'},
    'lensmc': {'ra': 'SHE_LENSMC_RA', 'dec': 'SHE_LENSMC_DEC', 'e1': 'SHE_LENSMC_E1_CORRECTED',
               'e2': 'SHE_LENSMC_E2_CORRECTED', 'w': 'SHE_LENSMC_WEIGHT_CORRECTED'},
}
PHZ, TOM_BIN = 'PHZ_MODE_1', 'TOM_BIN_ID'

# The South field spans Dec -70 to -10 and the North field Dec +50 to +80, so the sign of Dec separates them
FIELDS = {'south': lambda dec: dec < 0, 'north': lambda dec: dec >= 0}


def field_tiles(tile_dir, field, method, cache_path):
    """Tile files whose galaxies lie in *field*. Each tile is in one field, classified once by its median Dec."""
    if os.path.exists(cache_path):
        dec_tile = json.load(open(cache_path))
    else:
        dec_col = COLUMNS[method]['dec']
        dec_tile = {os.path.basename(f): float(np.median(pq.read_table(f, columns=[dec_col]).column(0).to_numpy()))
                    for f in sorted(glob.glob(f'{tile_dir}/tile_*.parquet'))}
        json.dump(dec_tile, open(cache_path, 'w'))
    return [f'{tile_dir}/{name}' for name, dec in sorted(dec_tile.items()) if FIELDS[field](dec)]


def read_tile(path, method, keys=('ra', 'dec', 'e1', 'e2', 'w')):
    """
    One tile's galaxies with an assigned tomographic bin (TOM_BIN_ID >= 1, as for TR1) and a positive calibrated
    weight, as numpy arrays keyed by 'ra', 'dec', 'e1', 'e2', 'w' (and 'phz' if asked for).
    """
    cols = {k: PHZ if k == 'phz' else COLUMNS[method][k] for k in keys}
    t = pq.read_table(path, columns=list(dict.fromkeys(list(cols.values()) + [TOM_BIN, COLUMNS[method]['w']])))
    keep = (t.column(TOM_BIN).to_numpy() >= 1) & (t.column(COLUMNS[method]['w']).to_numpy() > 0)
    return {k: t.column(c).to_numpy()[keep] for k, c in cols.items()}


def pixel_sums(tiles, method, n_side, z_edges):
    """
    Per-pixel sums over the catalogue at resolution *n_side*: galaxy count, sum w, sum w^2, sum w*e1, sum w*e2 and
    sum w*(e1^2 + e2^2), using the raw catalogue e1 (the notebooks apply the sign convention), plus the weighted
    PHZ_MODE_1 histogram on *z_edges*.
    """
    n_pix = hp.nside2npix(n_side)
    sums = {k: np.zeros(n_pix) for k in ['n', 'sw', 'sw2', 'swe1', 'swe2', 'swee']}
    nz = np.zeros(len(z_edges) - 1)
    for path in tiles:
        g = read_tile(path, method, keys=('ra', 'dec', 'e1', 'e2', 'w', 'phz'))
        pix = hp.ang2pix(n_side, g['ra'], g['dec'], lonlat=True)
        w = g['w'].astype(np.float64)
        e1, e2 = g['e1'].astype(np.float64), g['e2'].astype(np.float64)
        for k, v in [('n', None), ('sw', w), ('sw2', w ** 2), ('swe1', w * e1), ('swe2', w * e2),
                     ('swee', w * (e1 ** 2 + e2 ** 2))]:
            sums[k] += np.bincount(pix, weights=v, minlength=n_pix)
        nz += np.histogram(g['phz'], bins=z_edges, weights=w)[0]
    return sums, nz


class ParquetTileCatalog(CatalogBase):
    """
    heracles catalogue over the DR1 tiles, read one tile per page. Columns are renamed to the TR1 catalogue names
    (SHE_RA, SHE_DEC, SHE_E1_CORRECTED, SHE_E2_CORRECTED, SHE_WEIGHT) so the heracles field definitions are the same
    as in the TR1 South notebook. If *keep_pixels* (a boolean HEALPix map at resolution *n_side*) is given, only
    galaxies in those pixels are returned.
    """

    names_out = {'SHE_RA': 'ra', 'SHE_DEC': 'dec', 'SHE_E1_CORRECTED': 'e1', 'SHE_E2_CORRECTED': 'e2', 'SHE_WEIGHT': 'w'}

    def __init__(self, tiles, method, keep_pixels=None, n_side=None):
        super().__init__()
        self._tiles = list(tiles)
        self._method = method
        self._keep_pixels = keep_pixels
        self._n_side = n_side

    def __copy__(self):
        other = super().__copy__()
        other._tiles, other._method = self._tiles, self._method
        other._keep_pixels, other._n_side = self._keep_pixels, self._n_side
        return other

    def _names(self):
        return list(self.names_out)

    def _size(self, selection):
        return None

    def _join(self, *where):
        return None

    def _pages(self, selection):
        for path in self._tiles:
            g = read_tile(path, self._method)
            if self._keep_pixels is not None:
                keep = self._keep_pixels[hp.ang2pix(self._n_side, g['ra'], g['dec'], lonlat=True)]
                g = {k: v[keep] for k, v in g.items()}
            yield CatalogPage({out: g[k] for out, k in self.names_out.items()})
