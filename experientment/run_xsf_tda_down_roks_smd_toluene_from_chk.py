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

from XTDDFT_dev.utils.backend import backend_info, set_backend
from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down


cp.cuda.set_allocator(None)
lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))


# ===== Manually edit these parameters on the server =====
chk = "roks_smd_toluene_cc_pvdz.chk"
xc = "b3lyp"
auxbasis = "cc-pvdz-jkfit"
nstates = 6

use_density_fit = True

method = 1
SA = 3
remove = None
foo = 1.0
d_lda = 0.3
fglobal = None
fit = True
davidson_backend = "cpu"  # CPU Davidson + GPU matrix-vector product.
delta_a_jk_batch_size = 1

run_analyse = True
analyse_threshold = 0.05
save_results = True
save_file = "xsf_tda_down_roks_smd_toluene_cc_pvdz_mcol_sa3_nstates6.npz"
# ========================================================


set_backend("gpu")
chk_path = Path(chk)
if not chk_path.exists():
    raise FileNotFoundError(f"chk file does not exist in current directory: {chk_path}")

print("chk path:", chk_path.resolve())
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

print("backend:", backend_info())
print("CUDA device count:", cp.cuda.runtime.getDeviceCount())
print("mf class:", type(mf))
print("reference: ROKS/SMD")
print("xc:", mf.xc)
print("solvent:", getattr(getattr(mf, "with_solvent", None), "solvent", None))
print("natm:", mol.natm)
print("nelectron:", mol.nelectron)
print("spin:", mol.spin)
print("e_tot:", mf.e_tot)
print("mo_coeff shape:", mf.mo_coeff.shape)
print("mo_occ shape:", mf.mo_occ.shape)
print("method:", method)
print("SA:", SA)
print("nstates:", nstates)

print("=" * 80)
print(f"Running XSF_TDA_down method={method} SA={SA}")
xsf_method = XSF_TDA_down(
    mf,
    method=method,
    SA=SA,
    davidson=True,
    davidson_backend=davidson_backend,
    delta_a_jk_batch_size=delta_a_jk_batch_size,
)
ee, vv = xsf_method.kernel(
    nstates=nstates,
    remove=remove,
    foo=foo,
    d_lda=d_lda,
    fglobal=fglobal,
    fit=fit,
    save=save_results,
    save_file=save_file,
)
cp.cuda.Stream.null.synchronize()

print("converged:", xsf_method.converged)
print("fglobal:", getattr(xsf_method, "fglobal", None))
print("result file:", getattr(xsf_method, "result_file", None))
print("energies / eV:")
print(np.asarray(ee))
print("vectors shape:", vv.shape)

if run_analyse:
    xsf_method.analyse(threshold=analyse_threshold)
