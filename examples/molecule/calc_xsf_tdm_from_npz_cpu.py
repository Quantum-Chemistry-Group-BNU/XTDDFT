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
from XTDDFT_dev.utils.backend import backend_info, set_backend
from XTDDFT_dev.utils.unit import ha2eV

lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))
set_backend("cpu")

chk = "roks_smd_toluene_cc_pvdz.chk"
xsf_file = "XSF.npz"
out_file = "XSF_TDM.npz"

xc = "pbe0"
#auxbasis = "cc-pvdz-jkfit"
method = 1
SA = 3
use_density_fit = True

chk_path = Path(chk)
xsf_path = Path(xsf_file)
if not chk_path.exists():
    raise FileNotFoundError(f"chk file does not exist: {chk_path.resolve()}")
if not xsf_path.exists():
    raise FileNotFoundError(f"XSF file does not exist: {xsf_path.resolve()}")

mol, scf_rec = mol_chkfile.load_scf(str(chk_path))

mf = dft.ROKS(mol, xc=xc)
if use_density_fit:
    mf = mf.density_fit(auxbasis=auxbasis)

mf.mo_energy = np.asarray(scf_rec["mo_energy"])
mf.mo_coeff = np.asarray(scf_rec["mo_coeff"])
mf.mo_occ = np.asarray(scf_rec["mo_occ"])
mf.e_tot = scf_rec.get("e_tot", None)
mf.converged = True

data = np.load(xsf_path)
if "v" in data:
    vectors = np.asarray(data["v"])
elif "vectors" in data:
    vectors = np.asarray(data["vectors"])
else:
    raise KeyError("XSF.npz must contain v or vectors")

if "e_ha" in data:
    energies_ha = np.asarray(data["e_ha"], dtype=float).reshape(-1)
elif "e_ev" in data:
    energies_ha = np.asarray(data["e_ev"], dtype=float).reshape(-1) / ha2eV
else:
    raise KeyError("XSF.npz must contain e_ha or e_ev")

if vectors.ndim != 2:
    raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
if vectors.shape[1] != energies_ha.size:
    raise ValueError(
        "vectors.shape[1] must match the number of energies: "
        f"{vectors.shape[1]} vs {energies_ha.size}"
    )

xsf = XSF_TDA_down(
    mf,
    method=method,
    SA=SA,
    davidson=True,
    davidson_backend="cpu",
)
xsf.v = vectors
xsf.e = energies_ha
xsf.nstates = energies_ha.size
xsf.converged = np.ones(xsf.nstates, dtype=bool)
xsf.re = np.asarray(mf.mo_coeff).ndim != 3
if xsf.re:
    xsf.vects = xsf.get_vect()

tdm = xsf.transition_dipole_matrix()
delta_e_ha = energies_ha[:, None] - energies_ha[None, :]
delta_e_ev = delta_e_ha * ha2eV
dipole2 = np.einsum("ijx,ijx->ij", tdm, tdm)
osc = (2.0 / 3.0) * delta_e_ha * dipole2

positive_osc = np.zeros_like(osc)
mask = delta_e_ha > 0.0
positive_osc[mask] = osc[mask]

np.savez_compressed(
    out_file,
    tdm_au=tdm,
    oscillator_strength=osc,
    positive_transition_oscillator_strength=positive_osc,
    transition_energy_ha=delta_e_ha,
    transition_energy_ev=delta_e_ev,
    energies_ha=energies_ha,
    energies_ev=energies_ha * ha2eV,
    xsf_file=np.asarray(str(xsf_path)),
    chk_file=np.asarray(str(chk_path)),
)

print("backend:", backend_info())
print("Loaded chk:", chk_path.resolve())
print("Loaded XSF:", xsf_path.resolve())
print("Saved:", Path(out_file).resolve())
print("energies / eV:")
print(energies_ha * ha2eV)
print("oscillator strength matrix f(i <- j):")
print(osc)
print("positive-transition oscillator strengths:")
print(positive_osc)
