#!/usr/bin/env python
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import numpy as np
from pyscf import dft, lib
from pyscf.scf import chkfile as mol_chkfile

from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT_dev.utils.backend import set_backend
from XTDDFT_dev.utils.unit import ha2eV

lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))
set_backend("cpu")

# ===== Edit these parameters =====
chk = "ROKS.chk"
results_file = "XSF_mol.npz"
xc = "B3LYP"
threshold = 0.05
# ================================


def resolve_file(filename):
    path = Path(filename).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), SCRIPT_DIR, ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find {filename!r} in cwd, script directory, or repo root")


def scalar(data, key, default):
    return np.asarray(data[key]).item() if key in data else default


chk_path = resolve_file(chk)
results_path = resolve_file(results_file)

with np.load(results_path) as data:
    if "vectors" not in data:
        raise KeyError(f"{results_path} does not contain 'vectors'")
    vectors = np.asarray(data["vectors"])
    if "e_ha" in data:
        energies_ha = np.asarray(data["e_ha"], dtype=float).reshape(-1)
    elif "e_ev" in data:
        energies_ha = np.asarray(data["e_ev"], dtype=float).reshape(-1) / ha2eV
    else:
        raise KeyError(f"{results_path} contains neither 'e_ha' nor 'e_ev'")
    method = int(scalar(data, "method", 1))
    SA = int(scalar(data, "SA", 3))
    remove = bool(scalar(data, "remove", True))
    converged = np.asarray(data["converged"], dtype=bool) if "converged" in data else None

if vectors.ndim != 2 or vectors.shape[1] != energies_ha.size:
    raise ValueError(
        f"Expected vectors shape (dimension, {energies_ha.size}); got {vectors.shape}"
    )

mol, scf_rec = mol_chkfile.load_scf(str(chk_path))
mo_energy = np.asarray(scf_rec["mo_energy"])
mo_coeff = np.asarray(scf_rec["mo_coeff"])
mo_occ = np.asarray(scf_rec["mo_occ"])
if mo_coeff.ndim == 3:
    if mo_coeff.shape[0] != 2 or not np.allclose(mo_coeff[0], mo_coeff[1]):
        raise ValueError("XSF ROKS analysis requires matching alpha/beta orbitals")
    mo_coeff = mo_coeff[0]
    if mo_energy.ndim == 2:
        mo_energy = mo_energy[0]
    if mo_occ.ndim == 2:
        mo_occ = mo_occ[0] + mo_occ[1]

mf = dft.ROKS(mol, xc=xc)
mf.mo_energy = mo_energy
mf.mo_coeff = mo_coeff
mf.mo_occ = mo_occ
mf.e_tot = scf_rec.get("e_tot")
mf.converged = True

xsf = XSF_TDA_down(mf, method=method, SA=SA, davidson=True)
xsf.e = energies_ha
xsf.v = vectors
xsf.nstates = energies_ha.size
xsf.converged = converged
xsf.re = remove
if remove:
    xsf.vects = xsf.get_vect()

print("chk:", chk_path.resolve())
print("results:", results_path.resolve())
print("method:", method, "SA:", SA, "remove:", remove)
print("energies / eV:", energies_ha * ha2eV)
if converged is not None:
    print("converged:", converged)
print("\nXSF analysis (components above threshold =", threshold, ")")
xsf.analyse(threshold=threshold)
