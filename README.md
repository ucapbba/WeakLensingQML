# WeakLensingQML

Software package that implements the Quadratic Maximum Likelihood (QML) estimator and applies it
to simulated cosmic shear data and compares the results to a Pseudo-Cl implementation.  
This software package is split into three distinct parts:

* A "pre-processing" Python package. This script, located under `./python/PreProcessing.py`, computes
  and saves relevant data files for later processes, such as the fiduciary cosmic shear power spectrum
  used in the analysis, the sky mask, and computing an analytic version of the QML's covariance matrix.

* A C++ executable which forms the bulk of the QML implementation. This implements our conjugate-gradient
  approach for our quadratic estimator, and is parallelized for maximum performance. The code relies
  on the [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) linear algebra package, and
  the [HealPix](https://healpix.sourceforge.io/) spherical harmonic transform library.

* A "post-processing" Python package. This script, located under `./python/PostProcessing.py`, analyses
  the results of the C++ code and compares the QML's estimates with those from the Pseudo-Cl estimator
  and produces an array of plots highlighting the results.


## Examples

We provide an example [Jupyter Notebook](examples/power_spectra_and_covariance_estimation.ipynb) which runs you through
the basic ideas and implementation of our method. The notebook starts out with generating a mask that's applicable for
a space-based Stage-IV galaxy survey, computing an example power spectrum for cosmic shear, computing the Fisher matrix
for this mask & power spectrum, and then finally estimating the power spectrum for a set of cosmic shear maps.
Using this notebook, it should be easy to extend this to whatever way the user sees fit. 

## Building the native C++ library

The notebook uses a compiled shared library from the C++ component. Build it from the repository root with:

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build -j2
```

This produces the shared library used by the notebook at [cpp/build/libQMLShearLib.so](cpp/build/libQMLShearLib.so).


## GPU acceleration (optional)

The dense covariance-matrix build/inversion in `python/lib/PreProcessing/QMLClass.py` (`compute_covariance_matrix`,
`compute_cov_inv_Y`, `compute_Y_cov_inv_Y`) runs on the GPU via [cupy](https://cupy.dev/) if it's installed and a
GPU is available, and falls back to plain NumPy otherwise. No code changes or extra environment variables are
needed either way.

To use the GPU path:

```bash
# Pick the cupy wheel matching your CUDA toolkit version, e.g.:
pip install cupy-cuda12x
```

To verify the covariance-matrix maths is unaffected by which backend is used (safe to run with or without a GPU):

```bash
pip install pytest
pytest python/tests/test_qml_gpu_backend.py -v
```


## Publications

The algorithms and implementation is detailed in Maraio, Hall, and Taylor 2022
which can be found on the arXiv at [https://arxiv.org/abs/2207.10412](https://arxiv.org/abs/2207.10412).

If you use all or part of our code/method then please consider citing the above paper as it demonstrates that this work
is useful to the community, thank you! 

## Contact

If any issues are encountered with this software, feel free to either raise an issue on this GitHub repository or
directly email the main author in the above paper.
