#!/usr/bin/env python
"""Compare this project's B3LYP/UTDA transition dipoles with PySCF TDA."""

import sys
from pathlib import Path

import numpy as np
from pyscf import dft, gto, tdscf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from XTDDFT_dev.XTDDFT.xtda import XTDA


def _compare_utda_transition_dipoles():
    mol = gto.M(
        atom="""
            O   0.000  0.000  0.000
            C   1.402  0.031 -0.027
            H   1.786  1.019  0.137
            H   1.827 -0.438  0.914
            H   1.694 -0.617 -0.806
        """,
        basis="sto-3g",
        spin=1,
        symmetry=True,
        verbose=0,
    )
    mf = dft.UKS(mol, xc="b3lyp")
    mf.grids.level = 3
    mf.conv_tol = 1e-11
    mf.max_cycle = 100
    mf.kernel()
    assert mf.converged
    assert mol.groupname == "C1"

    nstates = 4
    utda = XTDA(mf, davidson=False, davidson_backend="cpu")
    utda.kernel(nstates=nstates)

    pyscf_tda = tdscf.TDA(mf)
    (a_aa, a_ab, a_bb), _ = pyscf_tda.get_ab()
    da = int(np.prod(a_aa.shape[:2]))
    db = int(np.prod(a_bb.shape[:2]))
    a_pyscf = np.block([
        [a_aa.reshape(da, da), a_ab.reshape(da, db)],
        [a_ab.transpose(2, 3, 0, 1).reshape(db, da), a_bb.reshape(db, db)],
    ])
    a_pyscf = (a_pyscf + a_pyscf.T) * 0.5
    all_energies, all_vectors = np.linalg.eigh(a_pyscf)
    roots = np.flatnonzero(all_energies > pyscf_tda.positive_eig_threshold)[:nstates]
    energies_utda = np.asarray(utda.e)
    energies_pyscf = all_energies[roots]
    vectors_pyscf = all_vectors[:, roots]

    nocca, nvira = a_aa.shape[:2]
    noccb, nvirb = a_bb.shape[:2]
    pyscf_tda.xy = [
        ((vector[:da].reshape(nocca, nvira), vector[da:].reshape(noccb, nvirb)), (0, 0))
        for vector in vectors_pyscf.T
    ]
    dipoles_utda = utda.transition_dipoles_ground()
    dipoles_pyscf = pyscf_tda.transition_dipole()

    vectors_utda = utda.v.copy()
    utda.v = vectors_pyscf[utda.order]
    dipoles_utda_same_vectors = utda.transition_dipoles_ground()
    utda.v = vectors_utda

    vectors_utda_raw = np.empty_like(utda.v)
    vectors_utda_raw[utda.order] = utda.v
    overlaps = np.einsum("is,is->s", vectors_utda_raw, vectors_pyscf)
    signs = np.where(overlaps < 0, -1.0, 1.0)
    dipoles_pyscf_aligned = dipoles_pyscf * signs[:, None]

    a_pyscf_ordered = a_pyscf[np.ix_(utda.order, utda.order)]
    matrix_diff = np.max(np.abs(utda.A - a_pyscf_ordered))
    energy_diff = np.max(np.abs(energies_utda - energies_pyscf))
    dipole_diff = np.max(np.abs(dipoles_utda - dipoles_pyscf_aligned))
    formula_diff = np.max(np.abs(dipoles_utda_same_vectors - dipoles_pyscf))
    np.testing.assert_allclose(utda.A, a_pyscf_ordered, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(energies_utda, energies_pyscf, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(dipoles_utda, dipoles_pyscf_aligned, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(dipoles_utda_same_vectors, dipoles_pyscf, atol=1e-10, rtol=1e-10)

    return {
        "mol": mol,
        "mf": mf,
        "energies_utda": energies_utda,
        "energies_pyscf": energies_pyscf,
        "dipoles_utda": dipoles_utda,
        "dipoles_pyscf_aligned": dipoles_pyscf_aligned,
        "overlaps": overlaps,
        "matrix_diff": matrix_diff,
        "energy_diff": energy_diff,
        "dipole_diff": dipole_diff,
        "formula_diff": formula_diff,
    }


def test_utda_transition_dipoles_match_pyscf():
    _compare_utda_transition_dipoles()


if __name__ == "__main__":
    result = _compare_utda_transition_dipoles()
    np.set_printoptions(precision=10, suppress=True)
    print(f"molecule: distorted CH3O radical, point group = {result['mol'].groupname}")
    print(f"method: UKS/B3LYP, grid level = {result['mf'].grids.level}")
    print(f"UKS converged: {result['mf'].converged}, energy = {result['mf'].e_tot:.12f} Ha")
    print("project UTDA excitation energies (Ha):")
    print(result["energies_utda"])
    print("PySCF TDA excitation energies (Ha):")
    print(result["energies_pyscf"])
    print("project UTDA transition dipoles (a.u.):")
    print(result["dipoles_utda"])
    print("PySCF TDA transition dipoles, sign-aligned (a.u.):")
    print(result["dipoles_pyscf_aligned"])
    print("state-vector overlaps before sign alignment:")
    print(result["overlaps"])
    print(f"max |A-matrix difference| = {result['matrix_diff']:.3e} Ha")
    print(f"max |energy difference| = {result['energy_diff']:.3e} Ha")
    print(f"max |end-to-end dipole difference| = {result['dipole_diff']:.3e} a.u.")
    print(f"max |same-vector formula difference| = {result['formula_diff']:.3e} a.u.")
    print("PASS: project B3LYP/UTDA and PySCF TDA transition dipoles agree.")
