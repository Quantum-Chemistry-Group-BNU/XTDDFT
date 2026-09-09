from pathlib import Path
import sys

import numpy as np
import pytest
from pyscf import dft, gto


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT_dev.XTDDFT.sf_tda_up import SF_TDA_up
from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT_dev.XTDDFT.xtda import XTDA
from XTDDFT_dev.utils import unit


ATOM = (
    ("H", (0.0, 0.934473, -0.588078)),
    ("H", (0.0, -0.934473, -0.588078)),
    ("C", (0.0, 0.0, 0.0)),
    ("O", (0.0, 0.0, 1.221104)),
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
        basis="6-31g",
        verbose=0,
    )
    mf = dft.ROKS(mol) if reference == "roks" else dft.UKS(mol)
    mf.xc = "b3lyp"
    mf.conv_tol = 1e-11
    mf.grids.level = 5
    mf.kernel()
    assert mf.converged
    return mf


def solve_td(reference, channel, method, atom=ATOM):
    mf = build_mf(reference, atom)
    if channel == "sc":
        td = XTDA(mf, davidson=True)
    elif channel == "sfu":
        td = SF_TDA_up(mf, method=method, davidson=True, collinear_samples=20)
    else:
        td = XSF_TDA_down(mf, method=method, davidson=True, collinear_samples=20)
    td.kernel(nstates=1)
    return td


@pytest.mark.parametrize("reference,channel,method", CASES)
def test_analytic_gradient_smoke(reference, channel, method):
    td = solve_td(reference, channel, method)
    mo_coeff = np.array(td.mf.mo_coeff, copy=True)
    mo_occ = np.array(td.mf.mo_occ, copy=True)

    gradient = td.nuc_grad_method(state=1).kernel()

    assert gradient.shape == (4, 3)
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, atol=2e-6)
    np.testing.assert_allclose(td.mf.mo_coeff, mo_coeff)
    np.testing.assert_allclose(td.mf.mo_occ, mo_occ)


def displaced_atom(atom_index, axis, displacement):
    atom = [(symbol, np.array(coord, dtype=float)) for symbol, coord in ATOM]
    atom[atom_index][1][axis] += displacement
    return atom


@pytest.mark.slow
@pytest.mark.parametrize("reference,channel,method", CASES)
def test_analytic_gradient_matches_one_coordinate_finite_difference(
    reference, channel, method
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
