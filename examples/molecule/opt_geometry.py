#!/usr/bin/env python
"""Optimize an excited-state geometry with one of six analytic gradients."""
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OMP_DYNAMIC"] = "False"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
from pathlib import Path

import numpy as np
from pyscf import gto, scf
from pyscf.geomopt import as_pyscf_method, geometric_solver

from XTDDFT.XTDDFT.sf_tda_up import SF_TDA_up
from XTDDFT.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT.XTDDFT.xtda import XTDA
from XTDDFT.utils.backend import asnumpy, get_array_module, set_backend
from XTDDFT.utils.unit import ha2eV


SUPPORTED_METHODS = (
    "xsc",
    "usc",
    "xsf_up",
    "usf_up",
    "xsf_down",
    "usf_down",
)

# ===== Manually edit these parameters on the server =====
method_kind = "xsf_down"
use_gpu = False
xc = "b3lyp"
basis = "cc-pvdz"
spin = 2
charge = 0
state = 1
sf_method = 1  # 1 for multicollinear, 2 for collinear
collinear_samples = 20
verbose = 4
conv_tol = 1e-10
max_cycle = 200
grids_level = 5
maxsteps = 100
trajectory_file = Path(f"ch2o_{method_kind}_opt_traj.xyz")
atom = """
    H   0.000000   0.934473  -0.588078
    H   0.000000  -0.934473  -0.588078
    C   0.000000   0.000000   0.000000
    O   0.000000   0.000000   1.221104
"""
conv_params = {
    "convergence_energy": 1e-6,
    "convergence_grms": 3e-4,
    "convergence_gmax": 4.5e-4,
    "convergence_drms": 1.2e-3,
    "convergence_dmax": 1.8e-3,
}
# ========================================================


def _check_method_kind(kind):
    kind = kind.lower()
    if kind not in SUPPORTED_METHODS:
        raise ValueError(f"method_kind must be one of {SUPPORTED_METHODS}")
    return kind


def build_reference(mol, kind=None, use_gpu=False):
    """Build the ROKS/UKS reference selected by method_kind."""
    kind = method_kind if kind is None else kind
    kind = _check_method_kind(kind)
    mf = scf.UKS(mol) if kind.startswith("u") else scf.ROKS(mol)
    mf.xc = xc
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    mf.grids.level = grids_level
    return mf.to_gpu() if use_gpu else mf


def build_response(mf, kind=None):
    """Build the response solver selected by method_kind."""
    kind = method_kind if kind is None else kind
    kind = _check_method_kind(kind)
    if kind.endswith("sc"):
        return XTDA(mf, davidson=True)
    kwargs = {
        "method": sf_method,
        "davidson": True,
        "collinear_samples": collinear_samples,
    }
    if kind.endswith("sf_up"):
        return SF_TDA_up(mf, **kwargs)
    return XSF_TDA_down(mf, **kwargs)


def solve_excited(mol, with_gradient=True, use_gpu=False):
    """Return the selected excited-state energy and optional gradient."""
    mf = build_reference(mol, use_gpu=use_gpu)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")

    td = build_response(mf)
    td.kernel(nstates=max(state + 2, 3))
    if not np.all(td.converged):
        raise RuntimeError("Excited-state calculation did not converge")

    energy = mf.e_tot + td.e[state - 1]
    gradient = None
    if with_gradient:
        gradient = td.nuc_grad_method(state=state).kernel()
    return mf, td, energy, gradient


def write_xyz(mol, filename):
    coords = mol.atom_coords(unit="Angstrom")
    with filename.open("a") as handle:
        handle.write(f"{mol.natm}\n\n")
        for atom_index, (x, y, z) in enumerate(coords):
            symbol = mol.atom_pure_symbol(atom_index)
            handle.write(f"{symbol:<3s} {x:16.10f} {y:16.10f} {z:16.10f}\n")


def main():
    _check_method_kind(method_kind)
    set_backend("gpu" if use_gpu else "cpu")
    mol = gto.M(
        atom=atom,
        spin=spin,
        charge=charge,
        basis=basis,
        verbose=verbose,
    )

    reference = build_reference(
        mol, use_gpu=use_gpu and method_kind.lower().startswith("u")
    )
    reference.kernel()
    if not reference.converged:
        raise RuntimeError("Reference-state SCF did not converge")
    mol_reference = geometric_solver.optimize(
        reference, maxsteps=maxsteps, **conv_params
    )

    timings = {"scf": [], "excitation": [], "gradient": []}
    trajectory_file.write_text("")

    def energy_gradient(current_mol):
        start = time.perf_counter()
        mf = build_reference(current_mol, use_gpu=use_gpu)
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("SCF did not converge")
        timings["scf"].append(time.perf_counter() - start)

        start = time.perf_counter()
        td = build_response(mf)
        td.kernel(nstates=max(state + 2, 3))
        if not np.all(td.converged):
            raise RuntimeError("Excited-state calculation did not converge")
        energy = mf.e_tot + td.e[state - 1]
        timings["excitation"].append(time.perf_counter() - start)

        start = time.perf_counter()
        gradient = td.nuc_grad_method(state=state).kernel()
        timings["gradient"].append(time.perf_counter() - start)

        gradient_norm = get_array_module(gradient).linalg.norm(gradient)
        print(
            f"E(SCF)={mf.e_tot:.12f}  omega[{state}]={td.e[state - 1]:.12f}  "
            f"E(total)={energy:.12f}  |g|={float(gradient_norm):.6e}"
        )
        write_xyz(current_mol, trajectory_file)
        return float(energy), asnumpy(gradient)

    start = time.perf_counter()
    method = as_pyscf_method(mol_reference, energy_gradient)
    mol_excited = geometric_solver.optimize(
        method, maxsteps=maxsteps, **conv_params
    )
    elapsed = time.perf_counter() - start

    steps = len(timings["gradient"])
    for label in ("scf", "excitation", "gradient"):
        print(f"{label} average use {np.mean(timings[label]):10.2f} s")
    print(f"each step average use {elapsed / steps:10.2f} s")
    print("Reference geometry (Bohr)")
    print(mol_reference.atom_coords())
    print("Excited-state geometry (Bohr)")
    print(mol_excited.atom_coords())

    final_reference = build_reference(mol_reference, use_gpu=use_gpu)
    final_reference.kernel()
    if not final_reference.converged:
        raise RuntimeError("Final reference-state SCF did not converge")
    _, final_td, excited_energy, _ = solve_excited(
        mol_excited, with_gradient=False, use_gpu=use_gpu
    )
    final_td.analyse()

    reference_energy = final_reference.e_tot
    print(
        f"ground energy of reference geometry: {reference_energy:16.6f} Ha "
        f"{reference_energy * ha2eV:16.6f} eV"
    )
    print(
        f"state {state} energy of excited geometry: {excited_energy:16.6f} Ha "
        f"{excited_energy * ha2eV:16.6f} eV"
    )
    print(
        f"adiabatic excitation energy: {excited_energy - reference_energy:16.6f} Ha "
        f"{(excited_energy - reference_energy) * ha2eV:16.6f} eV"
    )


if __name__ == "__main__":
    main()
