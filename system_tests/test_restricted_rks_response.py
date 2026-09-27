"""Restricted closed-shell response paths agree with the spin-balanced limit."""

from pathlib import Path
import sys

import numpy as np
from pyscf import dft, gto, lib, tdscf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from XTDDFT.XTDDFT.sf_tda_up import SF_TDA_up
from XTDDFT.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT.XTDDFT.xtda import XTDA
from XTDDFT.utils.backend import set_backend


def test_pbe0_rks_closed_shell_response():
    set_backend("cpu")
    lib.num_threads(2)
    mol = gto.M(
        atom="O 0 0 0; H 0 0 0.9572; H 0.9266 0 -0.2396",
        basis="sto-3g", unit="Angstrom", spin=0, verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "PBE0"
    mf.grids.level = 3
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged

    triplet = tdscf.TDA(mf)
    triplet.singlet = False
    triplet.kernel(nstates=3)
    assert all(triplet.converged)

    up = SF_TDA_up(mf, method=1, collinear_samples=20, davidson=False)
    up.kernel(nstates=3)
    assert np.asarray(mf.mo_coeff).ndim == 2
    assert np.asarray(up.ctx.mo_coeff).ndim == 3
    np.testing.assert_allclose(up.e, triplet.e, rtol=0, atol=1e-7)

    up_davidson = SF_TDA_up(mf, method=1, collinear_samples=20, davidson=True)
    up_davidson.kernel(nstates=3)
    assert all(up_davidson.converged)
    np.testing.assert_allclose(up_davidson.e, up.e, rtol=0, atol=1e-7)

    up_alda0 = SF_TDA_up(mf, method=0, davidson=False)
    up_alda0.kernel(nstates=3)
    up_alda0_davidson = SF_TDA_up(mf, method=0, davidson=True)
    up_alda0_davidson.kernel(nstates=3)
    assert all(up_alda0_davidson.converged)
    np.testing.assert_allclose(up_alda0_davidson.e, up_alda0.e, rtol=0, atol=1e-7)

    down = XSF_TDA_down(mf, method=1, SA=0, collinear_samples=20, davidson=False)
    down.kernel(nstates=3, fglobal=0)
    np.testing.assert_allclose(down.e, up.e, rtol=0, atol=1e-7)

    down_alda0 = XSF_TDA_down(mf, method=0, SA=0, davidson=False)
    down_alda0.kernel(nstates=3, fglobal=0)
    np.testing.assert_allclose(down_alda0.e, up_alda0.e, rtol=0, atol=1e-7)

    xtda = XTDA(mf, davidson=False)
    xtda.kernel(nstates=3)
    xtda_davidson = XTDA(mf, davidson=True)
    xtda_davidson.kernel(nstates=3)
    assert all(xtda_davidson.converged)
    np.testing.assert_allclose(xtda_davidson.e, xtda.e, rtol=0, atol=1e-7)
    for energy in triplet.e:
        assert np.min(np.abs(np.linalg.eigvalsh(xtda.A) - energy)) < 1e-7