"""Test XSF-TDA-down dense/Davidson roots and residuals for O2."""

from pathlib import Path
import sys

import numpy as np
import pytest
from pyscf import dft, gto, lib, scf
from pyscf.pbc import gto as pbcgto, scf as pbcscf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from XTDDFT.XTDDFT.sf_tda_up import SF_TDA_up
from XTDDFT.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT.utils.backend import set_backend


def test_rohf_sa0_defaults_to_full_sf_space():
    set_backend("cpu")
    lib.num_threads(2)
    mol = gto.M(
        atom="O 0 0 0; O 0 0 1.2075",
        basis="sto-3g",
        unit="Angstrom",
        spin=2,
        verbose=0,
    )
    mf = scf.ROHF(mol).run(conv_tol=1e-11)
    td = XSF_TDA_down(mf, method=2, SA=0, davidson=False)
    td.kernel(nstates=1)

    assert not td.re
    assert td.A.shape == ((td.nc + td.no) * (td.no + td.nv),) * 2


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


@pytest.mark.parametrize("channel", ["sfd", "sfu"])
@pytest.mark.parametrize("omega", [None, -0.3])
@pytest.mark.parametrize("exxdiv", [None, "ewald"])
def test_pbc_dense_sf_exchange_matches_get_k(channel, omega, exxdiv):
    """Include the Ewald term for full- and short-range exchange kernels."""
    from XTDDFT.XTDDFT.sf_tda_up import add_hf_a_b2a
    from XTDDFT.XTDDFT.xsf_tda_down import add_hf_a_a2b

    lib.num_threads(2)
    cell = pbcgto.M(
        atom="O 2.5 2.5 2.0; O 2.5 2.5 3.2075", a=np.eye(3) * 6,
        basis="sto-3g", spin=2, precision=1e-8, mesh=[17] * 3, verbose=0,
    )
    mf = pbcscf.ROHF(cell, exxdiv=exxdiv).density_fit(auxbasis="weigend")
    eigenvalues, eigenvectors = np.linalg.eigh(mf.get_ovlp())
    coeff = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    # Scaled orbitals require overlap factors, rather than a bare identity shift.
    occupied = coeff[:, :2] * [1, 1.2]
    virtual = coeff[:, 2:5] * [0.8, 1.1, 0.9]
    build = add_hf_a_a2b if channel == "sfd" else add_hf_a_b2a
    actual = build(np.zeros((2, 3, 2, 3)), mf, occupied, virtual,
                   2, 3, hyb=0.4, omega=omega)
    expected = np.empty_like(actual)
    for j in range(2):
        for b in range(3):
            dm = np.outer(virtual[:, b], occupied[:, j])
            k = mf.get_k(cell, dm, hermi=0, omega=omega)
            expected[:, :, j, b] = -0.4 * (virtual.T @ k @ occupied).T
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("exxdiv", [None, "ewald"])
def test_pbc_sf_dense_matches_davidson_action(exxdiv):
    """Check every dense matrix column and random AX for SF-down and SF-up."""
    lib.num_threads(2)
    cell = pbcgto.M(
        atom="O 2.5 2.5 2.0; O 2.5 2.5 3.2075", a=np.eye(3) * 6,
        basis="sto-3g", spin=2, precision=1e-8, mesh=[17] * 3, verbose=0,
    )
    mf = pbcscf.ROHF(cell, exxdiv=exxdiv).density_fit(auxbasis="weigend")
    mf.chkfile = None
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged

    for sa in (0, 1, 2, 3):
        td = XSF_TDA_down(mf, method=2, SA=sa, davidson=False)
        dense = td._prepare_dense_A(fglobal=1.0)
        vind, _ = td.gen_tda_operation_sf(fglobal=1.0)
        np.testing.assert_allclose(dense, vind(np.eye(len(dense))).T, atol=1e-9, rtol=0)
        vectors = np.random.default_rng(27).normal(size=(3, len(dense)))
        np.testing.assert_allclose(vectors @ dense.T, vind(vectors), atol=1e-9, rtol=0)

    td = SF_TDA_up(mf, method=2, davidson=False)
    dense = td.get_Amat()
    vind, _ = td.gen_tda_operation_sf()
    np.testing.assert_allclose(dense, vind(np.eye(len(dense))).T, atol=1e-9, rtol=0)
    vectors = np.random.default_rng(27).normal(size=(3, len(dense)))
    np.testing.assert_allclose(vectors @ dense.T, vind(vectors), atol=1e-9, rtol=0)
