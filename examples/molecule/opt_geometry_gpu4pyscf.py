#!/usr/bin/env python
"""Optimize UKS-TDA states with GPU4PySCF for comparison with XTDDFT."""
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OMP_DYNAMIC"] = "False"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pyscf import gto
from pyscf.data.nist import HARTREE2EV as HA2EV
from pyscf.geomopt import geometric_solver


METHODS = ("usc", "usf_up", "usf_down")

# Match examples/molecule/opt_geometry.py.
xc = "b3lyp"
basis = "cc-pvdz"
spin = 2
charge = 0
state = 1
collinear_samples = 20
verbose = 4
conv_tol = 1e-10
max_cycle = 200
grids_level = 5
maxsteps = 100
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


def build_reference(mol):
    from gpu4pyscf.dft import uks

    mf = uks.UKS(mol, xc=xc)
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    mf.grids.level = grids_level
    return mf


def build_response(mf, kind):
    if kind == "usc":
        return mf.TDA().set(nstates=max(state + 2, 3))
    if kind not in METHODS:
        raise ValueError(f"kind must be one of {METHODS}")
    return mf.SFTDA().set(
        extype=0 if kind == "usf_up" else 1,
        collinear="mcol",
        collinear_samples=collinear_samples,
        nstates=max(state + 2, 3),
    )


def run_reference(mol):
    mf = build_reference(mol).run()
    if not mf.converged:
        raise RuntimeError("UKS reference did not converge")
    return geometric_solver.optimize(mf, maxsteps=maxsteps, **conv_params)


def run_excited(mol, kind):
    mf = build_reference(mol).run()
    if not mf.converged:
        raise RuntimeError(f"{kind} UKS reference did not converge")

    td = build_response(mf, kind).run()
    if not bool(td.converged.all().item()):
        raise RuntimeError(f"{kind} excited-state calculation did not converge")

    scanner = td.nuc_grad_method().as_scanner(state=state)
    return geometric_solver.optimize(scanner, maxsteps=maxsteps, **conv_params)


def evaluate_excited(mol, kind):
    mf = build_reference(mol).run()
    if not mf.converged:
        raise RuntimeError(f"final {kind} UKS reference did not converge")
    td = build_response(mf, kind).run()
    if not bool(td.converged.all().item()):
        raise RuntimeError(f"final {kind} calculation did not converge")

    td.analyze()
    if kind.startswith("usf_"):
        print(f"{kind} state {state} <S^2> = {td.spin_square(state=state - 1):.8f}")
    energy = mf.e_tot + td.e[state - 1].item()
    return float(energy), td


def adiabatic_st_gap(singlet_energy, triplet_energy):
    return (singlet_energy - triplet_energy) * HA2EV


def print_geometry(label, mol):
    print(f"\n{label} optimized geometry (Angstrom)")
    for symbol, (x, y, z) in zip(mol.elements, mol.atom_coords(unit="Angstrom")):
        print(f"{symbol:<3s} {x:16.10f} {y:16.10f} {z:16.10f}")


def main():
    # Importing registers TDA/SFTDA methods on GPU UKS objects.
    import gpu4pyscf.tdscf  # noqa: F401

    mol = gto.M(
        atom=atom,
        spin=spin,
        charge=charge,
        basis=basis,
        verbose=verbose,
    )

    mol_reference = run_reference(mol)
    final_reference = build_reference(mol_reference).run()
    if not final_reference.converged:
        raise RuntimeError("final UKS reference did not converge")
    triplet_energy = float(final_reference.e_tot)
    reference_s2 = final_reference.spin_square()[0]
    print_geometry("UKS reference", mol_reference)
    print(f"UKS reference energy = {triplet_energy:.12f} Ha")
    print(f"UKS reference <S^2> = {reference_s2:.8f}")
    mol_reference.tofile("gpu4pyscf_reference.xyz")

    energies = {}
    for kind in METHODS:
        mol_excited = run_excited(mol_reference, kind)
        energy, _ = evaluate_excited(mol_excited, kind)
        energies[kind] = energy
        print_geometry(kind, mol_excited)
        print(f"{kind} state {state} total energy = {energy:.12f} Ha")
        print(
            f"{kind} adiabatic energy relative to UKS reference = "
            f"{(energy - triplet_energy) * HA2EV:.8f} eV"
        )
        mol_excited.tofile(f"gpu4pyscf_{kind}.xyz")

    gap = adiabatic_st_gap(energies["usf_down"], triplet_energy)
    print(f"\nAdiabatic S-T gap (usf_down - UKS reference) = {gap:.8f} eV")


if __name__ == "__main__":
    main()
