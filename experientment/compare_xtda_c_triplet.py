#!/usr/bin/env python
import sys

import numpy as np
from pyscf import dft, gto


ROOT_PARENT = "/home/chang/Desktop/work/NV_center/cluster/XSFTDA"
OLD_ROOT = "/home/chang/Desktop/work/NV_center/cluster/XSFTDA/XTDDFT"
if ROOT_PARENT not in sys.path:
    sys.path.insert(0, ROOT_PARENT)
if OLD_ROOT not in sys.path:
    sys.path.insert(0, OLD_ROOT)

from XTDDFT_dev.XTDDFT.xtda import XTDA as DevXTDA
from XTDDFT_dev.utils.unit import ha2eV
from x2c_hamiltonian.driver.tdm import TDM_SGS
from xtddft.XTDA import X_TDA as OldXTDA


mol = gto.M(atom="C 0 0 0", basis="sto-3g", spin=2, charge=0, verbose=0)
mf = dft.ROKS(mol)
mf.xc = "svwn"
mf.conv_tol = 1.0e-10
mf.max_cycle = 200
mf.kernel()

nstates = 6
print("SCF converged:", mf.converged)
print("SCF energy / Ha:", f"{mf.e_tot:.12f}")
print("spin:", mol.spin)
print("basis:", mol.basis)
print("")

old = OldXTDA(mf)
old_e, _old_v_return = old.kernel(nstates=nstates)
old_e_ha = np.asarray(old.e[:nstates])
old_vec = np.asarray(old.values[:, :nstates])

charges = mol.atom_charges()
coords = mol.atom_coords()
charge_center = np.einsum("z,zr->r", charges, coords) / charges.sum()
with mol.with_common_orig(charge_center):
    ints_ao = mol.intor_symmetric("int1e_r", comp=3)
ints_mo = np.einsum("xpq,pi,qj->xij", ints_ao, mf.mo_coeff, mf.mo_coeff)
n = (old.nc, old.no, old.nv)
sl = (
    slice(0, old.nc),
    slice(old.nc, old.nc + old.no),
    slice(old.nc + old.no, old.nc + old.no + old.nv),
)
old_gs_tdm = []
for istate in range(nstates):
    xl = old.trans_format(old.values[:, istate])
    old_gs_tdm.append(TDM_SGS(mol.spin / 2.0, xl, [np.ones((1,))], ints_mo, n, sl))
old_gs_tdm = np.asarray(old_gs_tdm)
old_osc = (2.0 / 3.0) * old_e_ha * np.einsum("sx,sx->s", old_gs_tdm, old_gs_tdm)

new = DevXTDA(mf, davidson=False)
new_e_ev, new_v = new.kernel(nstates=nstates)
new_e = np.asarray(new.e)[:nstates]
new_vec = np.asarray(new.v)[:, :nstates]
new_gs_tdm = new.transition_dipoles_ground()
new_osc = new.osc_str()

print("")
print("Energy comparison / Ha")
for istate in range(nstates):
    print(
        f"{istate + 1:2d} old={old_e_ha[istate]: .12f} "
        f"new={new_e[istate]: .12f} diff={new_e[istate] - old_e_ha[istate]: .3e}"
    )

print("")
print("Eigenvector comparison in tensor basis; sign adjusted")
for istate in range(nstates):
    dot = float(np.dot(old_vec[:, istate], new_vec[:, istate]))
    sign = 1.0 if dot >= 0 else -1.0
    max_abs = float(np.max(np.abs(old_vec[:, istate] - sign * new_vec[:, istate])))
    print(f"{istate + 1:2d} dot={dot: .12f} max_abs_diff={max_abs:.3e}")

print("")
print("Ground-state transition dipole and oscillator strength comparison; sign adjusted")
for istate in range(nstates):
    sign = 1.0 if np.dot(old_vec[:, istate], new_vec[:, istate]) >= 0 else -1.0
    dip_diff = float(np.max(np.abs(sign * new_gs_tdm[istate] - old_gs_tdm[istate])))
    print(
        f"{istate + 1:2d} dip_max_abs_diff={dip_diff:.3e} "
        f"old_osc={old_osc[istate]: .12e} "
        f"new_osc={new_osc[istate]: .12e} diff={new_osc[istate] - old_osc[istate]: .3e}"
    )

print("")
print("New ground-state transition dipoles / a.u.")
for istate, dip in enumerate(new_gs_tdm, start=1):
    print(f"{istate:2d} {dip[0]: .12e} {dip[1]: .12e} {dip[2]: .12e}")

print("")
print("Old kernel returned energies / eV:", old_e[:nstates])
print("New kernel returned energies / eV:", new_e_ev[:nstates])
print("Max abs energy diff / Ha:", float(np.max(np.abs(new_e - old_e_ha))))
print("Max abs energy diff / eV:", float(np.max(np.abs(new_e * ha2eV - old_e[:nstates]))))
vec_diffs = []
dip_diffs = []
for istate in range(nstates):
    sign = 1.0 if np.dot(old_vec[:, istate], new_vec[:, istate]) >= 0 else -1.0
    vec_diffs.append(np.max(np.abs(old_vec[:, istate] - sign * new_vec[:, istate])))
    dip_diffs.append(np.max(np.abs(old_gs_tdm[istate] - sign * new_gs_tdm[istate])))
print("Max abs vector diff:", float(np.max(vec_diffs)))
print("Max abs sign-adjusted gs TDM diff:", float(np.max(dip_diffs)))
print("Max abs osc diff:", float(np.max(np.abs(new_osc - old_osc))))
