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

from XTDDFT_dev.XTDDFT.xtda import XTDA, _so2st_matrix
from XTDDFT_dev.utils.backend import set_backend


@pytest.mark.slow
def test_xtda_so_and_st_dense_builders_are_equivalent():
    set_backend("cpu")
    mol = gto.M(
        atom="""
            C -2.91400807 -0.55823559  0.03052752
            H -2.37721823 -1.66302155  0.09965968
            H -3.96527633 -0.77226144 -0.33224522
            N -2.34030191  0.59563094  0.25221390
            H -2.80790566  1.54696657  0.05511270
        """,
        basis="cc-pvdz",
        unit="A",
        spin=1,
        charge=1,
        symmetry=False,
        verbose=0,
    )
    mf = dft.ROKS(mol, xc="b3lyp")
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged

    td = XTDA(mf, davidson=False, so2st=False)
    a_so = td.get_Amat()
    a_st = td.get_Amat_ST()
    transform = _so2st_matrix(td.nc, td.no, td.nv)
    a_so_in_st = transform @ a_so @ transform.T

    energy_so, vector_so = eigh(a_so)
    energy_st, vector_st = eigh(a_st)
    vector_so_in_st = transform @ vector_so
    diagonal_overlap = np.diag(vector_so_in_st.T @ vector_st)
    phases = np.where(diagonal_overlap < 0.0, -1.0, 1.0)
    vector_st_aligned = vector_st * phases

    matrix_error = np.max(np.abs(a_so_in_st - a_st))
    energy_error = np.max(np.abs(energy_so - energy_st))
    minimum_overlap = np.min(np.abs(diagonal_overlap))
    vector_error = np.max(np.abs(vector_so_in_st - vector_st_aligned))

    print("\nXTDA spin-orbital/spin-tensor comparison")
    print(f"A matrix maximum error:               {matrix_error:.3e}")
    print(f"Eigenvalue maximum error:              {energy_error:.3e}")
    print(f"Minimum transformed-vector overlap:    {minimum_overlap:.12f}")
    print(f"Phase-aligned eigenvector max error:    {vector_error:.3e}")

    np.testing.assert_allclose(a_so_in_st, a_st, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(energy_so, energy_st, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        np.abs(diagonal_overlap), np.ones_like(diagonal_overlap),
        atol=1e-8, rtol=1e-8,
    )
    np.testing.assert_allclose(
        vector_so_in_st, vector_st_aligned, atol=1e-8, rtol=1e-8
    )
