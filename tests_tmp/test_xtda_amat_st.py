import sys
from pathlib import Path

import numpy as np
from pyscf import dft, gto


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from XTDDFT_dev.XTDDFT.xtda import XTDA


def test_get_amat_st_matches_spin_orbital_delta_path():
    mol = gto.M(
        atom="""
            O  0.00  0.00  0.00
            H  0.76  0.00  0.58
            H -0.71  0.09  0.63
        """,
        basis="sto-3g",
        charge=1,
        spin=1,
        symmetry=False,
        verbose=0,
    )
    mf = dft.ROKS(mol, xc="b3lyp")
    mf.grids.level = 1
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged

    td = XTDA(mf, davidson=False, davidson_backend="cpu")
    amat_so_delta = td.get_Amat().copy()
    amat_st_delta = td.get_Amat_ST().copy()

    np.testing.assert_allclose(
        amat_st_delta, amat_so_delta, atol=1e-10, rtol=1e-10
    )

