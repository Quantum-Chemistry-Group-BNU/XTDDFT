"""Test XTDA SO/ST basis transformations, Davidson roots, and dipoles."""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "12")

import numpy as np
import pytest
from pyscf import dft, gto
from scipy.linalg import eigh


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from XTDDFT.XTDDFT.xtda import (
    XTDA,
    _so2st,
    _so2st_matrix,
    _st2so_matrix,
)
from XTDDFT.utils.backend import set_backend


@pytest.mark.slow
def test_xtda_so_st_dense_davidson_and_transition_dipoles():
    """Check SO/ST equivalence, eigenvectors, Davidson roots, and dipoles."""
    set_backend("cpu")
    mol = gto.M(
        atom="""
            C -2.91400807 -0.55823559  0.03052752
            H -2.37721823 -1.66302155  0.09965968
            H -3.96527633 -0.77226144 -0.33224522
            N -2.34030191  0.59563094  0.25221390
            H -2.80790566  1.54696657  0.05511270
        """,
        basis="def2svp",
        unit="A",
        spin=1,
        charge=1,
        symmetry=False,
        verbose=0,
    )
    mf = dft.ROKS(mol, xc="HF")
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged

    td = XTDA(mf, davidson=True, davidson_backend="cpu", so2st=False)
    a_so = td.get_Amat()
    a_st = td.get_Amat_ST()
    so2st = _so2st_matrix(td.nc, td.no, td.nv)
    st2so = _st2so_matrix(td.nc, td.no, td.nv)

    energy_so, vector_so = eigh(a_so)
    energy_st, vector_st = eigh(a_st)
    vector_so_in_st = _so2st(vector_so, td.nc, td.no, td.nv)
    dense_overlap = np.diag(vector_so_in_st.T @ vector_st)

    nroots = 20
    td.kernel(nstates=nroots)
    davidson_energy = np.asarray(td.e)
    davidson_vectors = np.asarray(td.v)
    davidson_overlap = np.diag(davidson_vectors.T @ vector_st[:, :nroots])

    reference_dipoles = np.load(
        ROOT / "tests_tmp" / "transition_dipoles_itrans0_resp.npy"
    )
    calculated_dipoles = td.transition_dipole_array()

    so_to_st_error = np.max(np.abs(so2st @ a_so @ so2st.T - a_st))
    st_to_so_error = np.max(np.abs(st2so @ a_st @ st2so.T - a_so))
    dense_energy_error = np.max(np.abs(energy_so - energy_st))
    davidson_energy_error = np.max(
        np.abs(davidson_energy - energy_st[:nroots])
    )
    dipole_error = np.max(
        np.abs(np.abs(calculated_dipoles) - np.abs(reference_dipoles))
    )

    print("\nXTDA SO/ST dense, Davidson, and transition-dipole comparison")
    print(f"SO -> ST A-matrix maximum error:       {so_to_st_error:.3e}")
    print(f"ST -> SO A-matrix maximum error:       {st_to_so_error:.3e}")
    print(f"Dense eigenvalue maximum error:        {dense_energy_error:.3e}")
    print(f"Dense-vector minimum overlap:          {np.min(np.abs(dense_overlap)):.12f}")
    print(f"Davidson eigenvalue maximum error:     {davidson_energy_error:.3e}")
    print(f"Davidson-vector minimum overlap:       {np.min(np.abs(davidson_overlap)):.12f}")
    print(f"Transition-dipole magnitude max error: {dipole_error:.3e}")

    np.testing.assert_allclose(
        so2st @ a_so @ so2st.T, a_st, atol=1e-10, rtol=1e-10
    )
    np.testing.assert_allclose(
        st2so @ a_st @ st2so.T, a_so, atol=1e-10, rtol=1e-10
    )
    np.testing.assert_allclose(energy_so, energy_st, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        np.abs(dense_overlap), np.ones_like(dense_overlap), atol=1e-8, rtol=1e-8
    )
    assert np.all(td.converged)
    np.testing.assert_allclose(
        davidson_energy, energy_st[:nroots], atol=1e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        np.abs(davidson_overlap),
        np.ones_like(davidson_overlap),
        atol=1e-5,
        rtol=1e-5,
    )
    assert calculated_dipoles.shape == reference_dipoles.shape == (7, 7, 3)
    np.testing.assert_allclose(
        np.abs(calculated_dipoles),
        np.abs(reference_dipoles),
        atol=1e-3,
        rtol=1e-3,
    )
