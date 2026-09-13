"""Test XSF-TDA-down dense/Davidson roots and residuals for O2."""

from pathlib import Path
import sys

import numpy as np
from pyscf import dft, gto, lib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from XTDDFT.XTDDFT.sf_tda_up import SF_TDA_up
from XTDDFT.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT.utils.backend import set_backend


def test_o2_b3lyp_spin_flip_dense_matches_davidson():
    """Check XSF-down methods and the SF-up COL dense/Davidson paths."""
    set_backend("cpu")
    lib.num_threads(2)
    mol = gto.M(
        atom="O 0 0 0; O 0 0 1.2075",
        basis="sto-3g",
        unit="Angstrom",
        spin=2,
        verbose=0,
    )
    mf = dft.ROKS(mol)
    mf.xc = "b3lyp"
    mf.grids.level = 1
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged

    for method in (0, 1, 2):  # ALDA0, MCOL, COL
        kwargs = dict(method=method, SA=3, collinear_samples=20, davidson_backend="cpu")
        dense = XSF_TDA_down(mf, davidson=False, **kwargs)
        dense.kernel(nstates=20)
        davidson = XSF_TDA_down(mf, davidson=True, **kwargs)
        davidson.kernel(nstates=20)

        assert np.all(davidson.converged), f"method={method}"
        np.testing.assert_allclose(dense.e, davidson.e, rtol=0, atol=2e-7)
        residual = dense.A @ np.asarray(davidson.v) - np.asarray(davidson.v) * np.asarray(davidson.e)
        assert np.max(np.abs(residual)) < 2e-6, f"method={method}"

    dense = SF_TDA_up(
        mf, method=2, davidson=False, davidson_backend="cpu"
    )
    dense.kernel(nstates=5)
    davidson = SF_TDA_up(
        mf, method=2, davidson=True, davidson_backend="cpu"
    )
    davidson.kernel(nstates=5)

    assert np.all(davidson.converged), "SF-up method=2"
    vind, _ = davidson.gen_tda_operation_sf()
    np.testing.assert_allclose(
        dense.A,
        np.asarray(vind(np.eye(dense.A.shape[0]))).T,
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(dense.e, davidson.e, rtol=0, atol=2e-7)
    residual = dense.A @ np.asarray(davidson.v) - np.asarray(davidson.v) * np.asarray(davidson.e)
    assert np.max(np.abs(residual)) < 2e-6, "SF-up method=2"
