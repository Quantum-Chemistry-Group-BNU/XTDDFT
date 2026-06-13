#!/usr/bin/env python
import os
import sys
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")
os.environ.setdefault("CUPY_ACCELERATORS", "cub,cutensor")
os.environ.setdefault("GPU4PYSCF_NUMINT_BLOCK_SIZE", "32768")
os.environ.setdefault("GPU4PYSCF_NUMINT_BLOCK_MEM_FRACTION", "0.4")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import cupy as cp
import numpy as np
from pyscf import lib
from pyscf.scf import chkfile as mol_chkfile
from gpu4pyscf import dft as gpubasedft

from XTDDFT_dev.utils.backend import set_backend
from XTDDFT_dev.utils.unit import ha2eV
from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down


cp.cuda.set_allocator(None)
lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))


# ===== Manually edit these parameters on the server =====
chk = "roks_smd_toluene_cc_pvdz.chk"
vectors_file = "xsf_vectors.npy"
energies_file = "xsf_energies_ev.npy"
output_file = "xsf_emission_from_saved_states.npz"

xc = "b3lyp"
auxbasis = "cc-pvdz-jkfit"
use_density_fit = True

method = 1
SA = 3
# ========================================================


set_backend("gpu")

chk_path = Path(chk)
if not chk_path.exists():
    raise FileNotFoundError(f"chk file does not exist in current directory: {chk_path}")
if not Path(vectors_file).exists():
    raise FileNotFoundError(f"vectors .npy file does not exist in current directory: {vectors_file}")
if not Path(energies_file).exists():
    raise FileNotFoundError(f"energies .npy file does not exist in current directory: {energies_file}")

vectors = np.load(vectors_file)
energies_ev = np.load(energies_file)
vectors = np.asarray(vectors)
energies_ev = np.asarray(energies_ev, dtype=float).reshape(-1)

if vectors.ndim != 2:
    raise ValueError(f"vectors must be a 2D array with shape (ndim, nstates); got {vectors.shape}")
if vectors.shape[1] != energies_ev.size:
    raise ValueError(
        "vectors.shape[1] must match the number of energies. "
        f"got vectors.shape={vectors.shape}, energies.shape={energies_ev.shape}"
    )

mol, scf_rec = mol_chkfile.load_scf(str(chk_path))

mf = gpubasedft.ROKS(mol, xc=xc)
if use_density_fit:
    mf = mf.density_fit(auxbasis=auxbasis)
mf = mf.SMD()
mf.with_solvent.method = "SMD"
mf.with_solvent.solvent = "toluene"
mf.with_solvent.lebedev_order = 29

mf.mo_energy = cp.asarray(scf_rec["mo_energy"])
mf.mo_coeff = cp.asarray(scf_rec["mo_coeff"])
mf.mo_occ = cp.asarray(scf_rec["mo_occ"])
mf.e_tot = scf_rec.get("e_tot", None)
mf.converged = True

xsf_method = XSF_TDA_down(
    mf,
    method=method,
    SA=SA,
    davidson=True,
    davidson_backend="cpu",
)
xsf_method.v = vectors
xsf_method.e = energies_ev / ha2eV
xsf_method.nstates = energies_ev.size
xsf_method.converged = np.ones(energies_ev.size, dtype=bool)

tdm = xsf_method.transition_dipole_matrix()
delta_e_ha = xsf_method.e[:, None] - xsf_method.e[None, :]
delta_e_ev = energies_ev[:, None] - energies_ev[None, :]
dipole2 = np.einsum("ijx,ijx->ij", tdm, tdm)

emission_osc = np.zeros_like(delta_e_ha)
emission_mask = delta_e_ha > 0.0
emission_osc[emission_mask] = (2.0 / 3.0) * delta_e_ha[emission_mask] * dipole2[emission_mask]

np.savez_compressed(
    output_file,
    tdm_au=tdm,
    emission_oscillator_strength=emission_osc,
    emission_energy_ev=delta_e_ev,
    energies_ev=energies_ev,
    vectors_file=np.asarray(vectors_file),
    energies_file=np.asarray(energies_file),
)

print("Loaded vectors:", vectors_file, vectors.shape)
print("Loaded energies / eV:", energies_file, energies_ev)
print("Emission oscillator strengths, row=initial high state, col=final low state")
print(emission_osc)
print("Saved emission data:", output_file)
