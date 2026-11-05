# DR1 cycle 1: shear power spectra, QML vs NaMaster vs heracles

Summary: `report/DR1_cycle1_QML_summary.pdf` (South B-mode excess in both MetaCal and LensMC; North passes).

QML, NaMaster (pseudo-$C_\ell$) and heracles power spectra of the Euclid DR1 shear catalogues, following the TR1 South
analysis (`TR1/TR1_south_power_spectra_and_covariance_estimation.ipynb`): $N_{\rm side}=64$, $\ell_{\max}=150$,
bandpowers of width 20 from $\ell=10$, all tomographic bins combined. Each run is one shear catalogue
(**MetaCal** or **LensMC**) on one field (**South** or **North**; the fields are analysed separately).

⚠︎ The DR1 data are unblinded and restricted to the B-Mode Tiger Team. Source:
`https://webdav.pic.es/pnfs/pic.es/data/astro/cosmohub/disk/euclid/euclid_bmodes/dr1/dps/dr1_dps_tiles_v6/` (see
`data/README_v6.md`).

## Which run is where

| Catalogue | Field | Notebook | Results (`data/`) | Figures (`figures/`) |
|---|---|---|---|---|
| MetaCal | South | `DR1_south_metaCal_power_spectra_and_covariance.ipynb` | `metacal/south/` | `metacal/south/` |
| MetaCal | North | `DR1_north_metaCal_power_spectra_and_covariance.ipynb` | `metacal/north/` | `metacal/north/` |
| LensMC | South | `DR1_south_lensMC_power_spectra_and_covariance.ipynb` | `lensmc/south/` | `lensmc/south/` |
| LensMC | North | `DR1_north_lensMC_power_spectra_and_covariance.ipynb` | `lensmc/north/` | `lensmc/north/` |
| both | both | `DR1_cycle1_comparison.ipynb` (reads the four `results_summary_*.npz`; run last) | – | `report/figures/` |
| MetaCal | both | `DR1_metacal_maps.ipynb` (density, noise and mean-shear maps) | `metacal/metacal_maps_N512.npz` | `metacal/maps/` |

The four power-spectrum notebooks are the same pipeline; only the `field` and `method` settings at the top differ.

## Folder layout

```
DR1/cycle1/
├── README.md                  this file
├── dr1_tools.py               reads the Parquet tiles one at a time (pixel sums, heracles catalogue)
├── DR1_*_power_spectra_and_covariance.ipynb, DR1_metacal_maps.ipynb, DR1_cycle1_comparison.ipynb
├── report/                    DR1_cycle1_QML_summary.tex/.pdf; figures/ from DR1_cycle1_comparison.ipynb
├── data/                      gitignored
│   ├── README_v6.md           the data release README
│   ├── metacal_only/          MetaCal catalogue: one Parquet file per tile + manifest.json (SHA-256 per tile)
│   ├── lensmc_only/           LensMC catalogue, same layout
│   ├── tile_dec_{method}.json median Dec of each tile (used to split South/North)
│   └── {metacal,lensmc}/{south,north}/
│       ├── results_summary_*.npz          all bandpowers, covariances, masks and settings of the run
│       ├── pixel_sums_N64.npz             per-pixel sums from the catalogue (cached, the slow catalogue pass)
│       ├── Fisher_matrix_*_{cut,nocut}_*.dat, y_ell_*  QML outputs (cached; ~20-60 min each to recompute)
│       ├── heracles_bandpowers.npz        heracles with/without the cut, noise kept and removed (cached)
│       ├── masks/, Map_*.fits, TheorySpectra_*.dat
│       └── cpp_{cut,nocut}/               private C++ builds (mask and noise are compile-time constants)
└── figures/{metacal,lensmc}/{south,north}/
    ├── pixels.png, ee.png, bb.png, sigma.png      as in TR1/report_south/TR1_South_QML_summary.pdf
    ├── fig1_precision.png, fig2_bmodes.png, fig3_emodes.png   bandpowers with error bars, noise removed
    └── cut_effect.png                    QML and heracles with and without the noisy pixels
```

Delete a cached file to force that step to be recomputed.

## Pixel cut

DR1 has no coverage map, so pixels are cut on noise directly: an $N_{\rm side}=64$ pixel is kept if its effective
galaxy number $(\sum w)^2/\sum w^2$ is at least half the field median (`neff_min_frac = 0.5`). The `cut` runs use the
kept pixels, the `nocut` runs every pixel with galaxies.
