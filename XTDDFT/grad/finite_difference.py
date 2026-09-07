#!/usr/bin/env python
import numpy as np
from typing import List
from pyscf import gto, dft, __config__

from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT_dev.XTDDFT.sf_tda_up import SF_TDA_up
from XTDDFT_dev.XTDDFT.xtda import XTDA
from XTDDFT_dev.utils import unit


def excited_energy(atom, spec, state, mk, xc='b3lyp', method=1, cs=20):
    """excited energy = mf.e_tot + td.e[state-1], unit of td.e is Hartree"""
    kind = mk.lower()
    if kind not in ("usf_up", "usf_down", "usc", "xsf_up", "xsf_down", "xsc"):
        raise ValueError("method_kind must be 'usf_up', 'usf_down'," \
        " 'usc', 'xsf_up', 'xsf_down', 'xsc'")
    
    mol = gto.M(atom=atom, verbose=3, **spec)
    if kind[0] == 'u':
        mf = dft.UKS(mol)
    else:
        mf = dft.ROKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.grids.level = 5
    mf.kernel()

    if 'sf_down' in mk.lower():
        td = XSF_TDA_down(mf, method=method, davidson=True, collinear_samples=cs)
    elif 'sf_up' in mk.lower():
        td = SF_TDA_up(mf, method=method, davidson=True, collinear_samples=cs)
    elif 'sc' in mk.lower():
        td = XTDA(mf, davidson=True)
    else:
        raise ValueError
    # Request +2 extra roots, make davidson itersion more stable
    td.kernel(nstates=max(state, 1) + 2)
    return mf.e_tot + td.e[state-1]


def fd_gradient(atoms, state, mk, xc='b3lyp', charge=0, spin=2,
                 basis='6-31g', h=1e-5, method=1, cs=20):
    """finite difference truncate to second order, (-3E0+4E+ - E++)/(2h), unit of h is Angstrom"""
    assert isinstance(atoms, List) and (len(atoms[0]) == 2)
    print('***** The molecular coordinates of the finite difference input are in angstroms *****')
    natm = len(atoms)
    spec = dict(charge=charge, spin=spin, basis=basis)
    g = np.zeros((natm, 3))
    h_au = h / unit.bohr

    for i in range(natm):
        for d in range(3):
            atoms_ph = [(atom, coord.copy()) for atom, coord in atoms]
            atoms_ph[i][1][d] += h
            Eph = excited_energy(atoms_ph, spec, state, mk, xc, method, cs)
            atoms_mh = [(atom, coord.copy()) for atom, coord in atoms]
            atoms_mh[i][1][d] -= h
            Emh = excited_energy(atoms_mh, spec, state, mk, xc, method, cs)
            g[i, d] = (Eph - Emh) / (2*h_au)  # energy unit is Hartree/bohr
    return g


def fd_gradient_forth(atoms, state, mk, xc='b3lyp', charge=0, spin=2,
                       basis='6-31g', h=1e-5, method=1, cs=20):
    """finite difference truncate to second order, (-3E0+4E+ - E++)/(2h), unit of h is Angstrom"""
    assert isinstance(atoms, List) and (len(atoms[0]) == 2)
    print('***** The molecular coordinates of the finite difference input are in angstroms *****')
    natm = len(atoms)
    spec = dict(charge=charge, spin=spin, basis=basis)
    g = np.zeros((natm, 3))
    h_au = h / unit.bohr

    for i in range(natm):
        for d in range(3):
            atoms_mh = [(atom, coord.copy()) for atom, coord in atoms]
            atoms_mh[i][1][d] -= h
            Emh = excited_energy(atoms_mh, spec, state, mk, xc, method, cs)
            atoms_m2h = [(atom, coord.copy()) for atom, coord in atoms]
            atoms_m2h[i][1][d] -= 2 * h
            Emhh = excited_energy(atoms_m2h, spec, state, mk, xc, method, cs)
            atoms_ph = [(atom, coord.copy()) for atom, coord in atoms]
            atoms_ph[i][1][d] += h
            Eph = excited_energy(atoms_ph, spec, state, mk, xc, method, cs)
            atoms_p2h = [(atom, coord.copy()) for atom, coord in atoms]
            atoms_p2h[i][1][d] += 2 * h
            Ephh = excited_energy(atoms_p2h, spec, state, mk, xc, method, cs)
            g[i, d] = (Emhh - 8*Emh + 8*Eph - Ephh) / (12*h_au)  # energy unit is Hartree/bohr
    return g


def forward_difference(atoms, state, mk, xc='b3lyp', charge=0, spin=2,
                        basis='6-31g', h=1e-5, method=1, cs=20):
    """finite difference truncate to second order, (-3E0+4E+ - E++)/(2h), unit of h is Angstrom"""
    assert isinstance(atoms, List) and (len(atoms[0]) == 2)
    print('***** The molecular coordinates of the finite difference input are in angstroms *****')
    natm = len(atoms)
    spec = dict(charge=charge, spin=spin, basis=basis)
    g = np.zeros((natm, 3))
    h_au = h / unit.bohr

    E0 = excited_energy(atoms, spec, state, mk, xc, method, cs)
    for i in range(natm):
        for d in range(3):
            atoms_h = [(atom, coord.copy()) for atom, coord in atoms]
            atoms_h[i][1][d] += h
            Eh = excited_energy(atoms_h, spec, state, mk, xc, method, cs)
            atoms_2h = [(atom, coord.copy()) for atom, coord in atoms]
            atoms_2h[i][1][d] += 2 * h
            Ehh = excited_energy(atoms_2h, spec, state, mk, xc, method, cs)
            g[i, d] = (-3*E0 + 4*Eh - Ehh) / (2*h_au)  # energy unit is Hartree/bohr
    return g

