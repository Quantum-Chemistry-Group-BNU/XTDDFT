"""Compare XTDA spin contamination in RO and equivalent U orbital bases."""

import numpy as np
import pytest
from pyscf import dft, gto

from XTDDFT.XTDDFT.xtda import XTDA, _st2so


def test_ro_spin_contamination_matches_uks_orbital_formula():
    mol = gto.M(
        atom="C 0 0 0; H 0 .95 .15; H .1 -.75 .65",
        basis="sto-3g", spin=2, verbose=0,
    )
    mf = dft.ROKS(mol, xc="b3lyp")
    mf.grids.level = 1
    mf.kernel()
    assert mf.converged

    st = XTDA(mf, davidson=False, so2st=True)
    so = XTDA(mf, davidson=False, so2st=False)
    st.kernel(nstates=3)
    with pytest.warns(UserWarning, match="so2st=False: restricted-reference XTDA eigenvectors were converted from SO to ST after dense diagonalization"):
        so.kernel(nstates=3)
    np.testing.assert_allclose(st.e, so.e, atol=1e-10)
    np.testing.assert_allclose(
        np.abs(np.diag(st.v.T @ so.v)), 1, atol=1e-9,
    )
    np.testing.assert_allclose(st.deltaS2(), so.deltaS2(), atol=1e-9)

    davidson = XTDA(mf, davidson=True)
    davidson.kernel(nstates=3)
    np.testing.assert_allclose(davidson.e, st.e, atol=1e-8)
    np.testing.assert_allclose(davidson.deltaS2(), st.deltaS2(), atol=1e-7)
    davidson_so = XTDA(mf, davidson=True, so2st=False)
    with pytest.warns(UserWarning, match="so2st=False: restricted-reference XTDA eigenvectors were converted from SO to ST after the Davidson solve"):
        davidson_so.kernel(nstates=3)
    np.testing.assert_allclose(davidson_so.e, davidson.e, atol=1e-8)
    np.testing.assert_allclose(davidson_so.deltaS2(), davidson.deltaS2(), atol=1e-7)

    u = XTDA(mf.to_uks(), davidson=False)
    u.v = _st2so(st.v, st.nc, st.no, st.nv)
    u.nstates = st.nstates
    np.testing.assert_allclose(st.deltaS2(), u.deltaS2(), atol=1e-9)
    np.testing.assert_allclose(st.analyse(threshold=2), u.deltaS2(), atol=1e-9)

    # Disable the RO-only Delta A so both references solve the same response matrix.
    ro_as_u = XTDA(mf, davidson=False, so2st=True, use_delta_a=False)
    uk = XTDA(mf.to_uks(), davidson=False)
    for td in (ro_as_u, uk):
        td.kernel(nstates=3)
    np.testing.assert_allclose(ro_as_u.e, uk.e, atol=1e-9)
    np.testing.assert_allclose(ro_as_u.deltaS2(), uk.deltaS2(), atol=1e-9)
