"""
Correctness check for the numpy/cupy backend swap in QMLClass.py.

Builds synthetic (non-physical) inputs of the right shapes, runs the
compute_covariance_matrix / compute_cov_inv_Y / compute_Y_cov_inv_Y methods,
and compares against a hand-written NumPy-only reference that mirrors the
original (pre-GPU-backend) formulas exactly. This does not require a GPU:
on a machine without cupy/a GPU, QMLClass.xp falls back to numpy and this
test still validates that the refactor didn't change the maths.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "PreProcessing"))

import QMLClass  # noqa: E402


class DummyYMatrix:
    def __init__(self, Y):
        self.Y_matrix = Y
        self.Y_matrix_dagger = np.conj(Y).T


@pytest.fixture
def qml_with_reference():
    rng = np.random.default_rng(0)

    n_side = 4
    l_max = 3
    n_pix_mask = 20
    num_m_modes = (l_max + 1) ** 2 - 4

    mask = np.zeros(12 * n_side ** 2, dtype=bool)
    mask[:n_pix_mask] = True
    rng.shuffle(mask)

    Y_full = (rng.normal(size=(2 * n_pix_mask, 2 * num_m_modes)) +
              1j * rng.normal(size=(2 * n_pix_mask, 2 * num_m_modes)))

    qml = QMLClass.QML(n_side=n_side, l_max=l_max, redshift=1.0, mask=mask,
                        Y_matrix=DummyYMatrix(Y_full))
    qml.S_tilde_EE = rng.uniform(0.1, 1.0, size=num_m_modes)
    qml.S_tilde_BB = rng.uniform(0.1, 1.0, size=num_m_modes)

    qml.compute_covariance_matrix()
    qml.compute_cov_inv_Y()
    qml.compute_Y_cov_inv_Y()

    # Reference computation, plain NumPy, mirroring the pre-refactor code
    Y_matrix_QE = Y_full[0:n_pix_mask, 0:num_m_modes]
    Y_matrix_UE = Y_full[n_pix_mask:2 * n_pix_mask, 0:num_m_modes]

    S_matrix_QQ = (Y_matrix_QE * qml.S_tilde_EE @ np.conj(Y_matrix_QE).T).real
    S_matrix_QU = (Y_matrix_QE * qml.S_tilde_EE @ np.conj(Y_matrix_UE).T).real
    S_matrix_UU = (Y_matrix_UE * qml.S_tilde_EE @ np.conj(Y_matrix_UE).T).real

    cov_ref = np.zeros([2 * n_pix_mask, 2 * n_pix_mask], dtype=float)
    cov_ref[0:n_pix_mask, 0:n_pix_mask] = S_matrix_QQ
    cov_ref[0:n_pix_mask, n_pix_mask:2 * n_pix_mask] = S_matrix_QU
    cov_ref[n_pix_mask:2 * n_pix_mask, 0:n_pix_mask] = S_matrix_QU.T
    cov_ref[n_pix_mask:2 * n_pix_mask, n_pix_mask:2 * n_pix_mask] = S_matrix_UU
    np.fill_diagonal(cov_ref, cov_ref.diagonal() + qml.noise_array)

    cov_inv_ref = np.linalg.inv(cov_ref)
    cov_inv_Y_ref = cov_inv_ref @ Y_full
    Y_dagger_cov_inv_Y_ref = np.conj(Y_full).T @ cov_inv_Y_ref

    return qml, cov_ref, cov_inv_ref, cov_inv_Y_ref, Y_dagger_cov_inv_Y_ref


def test_backend_reports_numpy_without_gpu():
    # In any environment without a working GPU + cupy, the module must fall back to numpy
    if not QMLClass._GPU_AVAILABLE:
        assert QMLClass.xp is np


def test_covariance_matches_reference(qml_with_reference):
    qml, cov_ref, *_ = qml_with_reference
    assert isinstance(qml.cov, np.ndarray)
    assert np.allclose(qml.cov, cov_ref)


def test_covariance_inverse_matches_reference(qml_with_reference):
    qml, _, cov_inv_ref, *_ = qml_with_reference
    assert isinstance(qml.cov_inv, np.ndarray)
    assert np.allclose(qml.cov_inv, cov_inv_ref)


def test_cov_inv_y_matches_reference(qml_with_reference):
    qml, _, _, cov_inv_Y_ref, _ = qml_with_reference
    assert isinstance(qml.cov_inv_Y, np.ndarray)
    assert np.allclose(qml.cov_inv_Y, cov_inv_Y_ref)


def test_y_dagger_cov_inv_y_matches_reference(qml_with_reference):
    qml, _, _, _, Y_dagger_cov_inv_Y_ref = qml_with_reference
    assert isinstance(qml.Y_dagger_cov_inv_Y, np.ndarray)
    assert np.allclose(qml.Y_dagger_cov_inv_Y, Y_dagger_cov_inv_Y_ref)
