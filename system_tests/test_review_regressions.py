"""Regression checks for public vectors, root selection, and saved units."""
from types import SimpleNamespace

import numpy as np
from pyscf import dft, gto
from pyscf.pbc import gto as pbcgto

from XTDDFT.XTDDFT.xtda import XTDA
from XTDDFT.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT.XTDDFT.nac import nac


def test_ro_dense_so_matrix_returns_st_properties():
    mol = gto.M(atom="C 0 0 0; H 0 .95 .15; H .1 -.75 .65",
                basis="sto-3g", spin=2, verbose=0)
    mf = dft.ROKS(mol, xc="b3lyp")
    mf.grids.level = 1
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    solvers = [XTDA(mf, davidson=False, so2st=value) for value in (True, False)]
    for td in solvers:
        td.kernel(nstates=3)
    st, so = solvers
    np.testing.assert_allclose(st.e, so.e, atol=1e-10)
    overlap = np.diag(st.v.T @ so.v)
    np.testing.assert_allclose(abs(overlap), 1, atol=1e-9)
    so.v *= np.sign(overlap)
    np.testing.assert_allclose(st.transition_dipole_array(),
                               so.transition_dipole_array(), atol=1e-8)
    np.testing.assert_allclose(st.nuc_grad_method(state=2).kernel(),
                               so.nuc_grad_method(state=2).kernel(), atol=1e-8)


def test_pbc_davidson_keeps_negative_and_small_roots():
    td = XSF_TDA_down.__new__(XSF_TDA_down)
    td.mf = SimpleNamespace(cell=pbcgto.Cell())
    td.davidson_backend = "cpu"
    td.davidson_matvec_batch_size = None
    hdiag = np.array([-.2, .0005, .1, .3])
    td.gen_tda_operation_sf = lambda **kwargs: (lambda xs: xs * hdiag, hdiag)
    td.init_guess = lambda nroots, **kwargs: np.eye(4)
    td.davidson_process(2)
    np.testing.assert_allclose(td.e, hdiag[:2], atol=1e-12)
    dense_e, _ = td._diagonalize_dense(np.diag(hdiag), 2)
    np.testing.assert_allclose(dense_e, hdiag[:2], atol=1e-12)


def test_nac_compatibility_saves_hartree(monkeypatch, tmp_path):
    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)
    mf = SimpleNamespace(converged=True, kernel=lambda: None)
    energies = np.array([.125, .25])
    td = SimpleNamespace(e=energies, kernel=lambda **kwargs: (energies * 27.2114, None))
    result = {(0, 1): np.zeros((1, 3))}
    monkeypatch.setattr(nac.dft, "ROKS", lambda mol: mf)
    monkeypatch.setattr(nac, "XTDA", lambda *args, **kwargs: td)
    monkeypatch.setattr(nac, "NAC", lambda *args: SimpleNamespace(kernel=lambda: result))
    monkeypatch.chdir(tmp_path)
    nac.finite_difference_nac(mol, [(0, 1)], "b3lyp")
    with np.load("xtda_nac.npz") as saved:
        np.testing.assert_array_equal(saved["energies_hartree"], energies[:1])
    assert "0.125" in (tmp_path / "xtda_nac.txt").read_text()
