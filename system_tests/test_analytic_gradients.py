from pathlib import Path
from copy import deepcopy
from functools import lru_cache
import sys

import numpy as np
import pytest
from pyscf import dft, gto


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT.XTDDFT.sf_tda_up import SF_TDA_up
from XTDDFT.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT.XTDDFT.xtda import XTDA
from XTDDFT.utils import unit


# Small triplet CH2 reference with closed, open, and virtual orbital blocks.
ATOM = (
    ("C", (0.00,  0.00,  0.00)),
    ("H", (0.00,  0.95,  0.15)),
    ("H", (0.10, -0.75,  0.65)),
)
CASES = (
    ("roks", "sc", 0),
    ("uks", "sc", 0),
    ("roks", "sfu", 1),
    ("uks", "sfu", 1),
    ("roks", "sfd", 1),
    ("uks", "sfd", 2),
)


def build_mf(reference, atom=ATOM):
    mol = gto.M(
        atom=atom,
        unit="Angstrom",
        spin=2,
        charge=0,
        basis="sto3g",
        verbose=0,
    )
    mf = dft.ROKS(mol) if reference == "roks" else dft.UKS(mol)
    mf.xc = "b3lyp"
    mf.conv_tol = 1e-11
    mf.grids.level = 4  # Level 3 fails the existing total-gradient tolerance.
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope="module")
def solve_td():
    # The six channels share two references at each of three geometries.
    cached_mf = lru_cache(maxsize=6)(build_mf)

    def solve(reference, channel, method, atom=ATOM):
        geometry = tuple((symbol, tuple(coord)) for symbol, coord in atom)
        # Solvers may mutate their reference: never expose the cached object.
        mf = deepcopy(cached_mf(reference, geometry))
        if channel == "sc":
            td = XTDA(mf, davidson=True)
        elif channel == "sfu":
            td = SF_TDA_up(mf, method=method, davidson=True, collinear_samples=20)
        else:
            td = XSF_TDA_down(mf, method=method, davidson=True, collinear_samples=20)
        td.kernel(nstates=1)
        return td

    yield solve
    cached_mf.cache_clear()


@pytest.mark.parametrize("reference,channel,method", CASES)
def test_analytic_gradient_smoke(reference, channel, method, solve_td):
    td = solve_td(reference, channel, method)
    mo_coeff = np.array(td.mf.mo_coeff, copy=True)
    mo_occ = np.array(td.mf.mo_occ, copy=True)

    gradient = td.nuc_grad_method(state=1).kernel()

    assert gradient.shape == (len(ATOM), 3)
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(td.mf.mo_coeff, mo_coeff)
    np.testing.assert_allclose(td.mf.mo_occ, mo_occ)


def test_uks_sc_analytic_gradient_honors_nontrivial_atmlst(solve_td):
    td = solve_td("uks", "sc", 0)
    gradient_method = td.nuc_grad_method(state=1)

    full = gradient_method.kernel()
    selected = gradient_method.kernel(atmlst=[2, 0])

    assert selected.shape == (2, 3)
    np.testing.assert_allclose(selected, full[[2, 0]], atol=1e-12, rtol=1e-12)


def displaced_atom(atom_index, axis, displacement):
    atom = [(symbol, np.array(coord, dtype=float)) for symbol, coord in ATOM]
    atom[atom_index][1][axis] += displacement
    return atom


@pytest.mark.slow
@pytest.mark.parametrize("reference,channel,method", CASES)
def test_analytic_gradient_matches_one_coordinate_finite_difference(
    reference, channel, method, solve_td
):
    atom_index, axis, step = 1, 1, 1e-3
    td = solve_td(reference, channel, method)
    analytic = td.nuc_grad_method(state=1).kernel()[atom_index, axis]

    td_plus = solve_td(
        reference, channel, method, displaced_atom(atom_index, axis, step)
    )
    td_minus = solve_td(
        reference, channel, method, displaced_atom(atom_index, axis, -step)
    )
    energy_plus = td_plus.mf.e_tot + td_plus.e[0]
    energy_minus = td_minus.mf.e_tot + td_minus.e[0]
    finite_difference = (energy_plus - energy_minus) / (2 * step / unit.bohr)

    assert analytic == pytest.approx(finite_difference, abs=5e-4, rel=5e-3)
