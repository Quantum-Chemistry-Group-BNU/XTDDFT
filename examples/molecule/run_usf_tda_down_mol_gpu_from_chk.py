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

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import cupy as cp
import numpy as np
from gpu4pyscf import dft as gpubasedft
from pyscf import lib
from pyscf.scf import chkfile as mol_chkfile

from XTDDFT_dev.utils.backend import backend_info, set_backend

set_backend("gpu")
from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down


cp.cuda.set_allocator(None)
lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))


def resolve_file(filename):
    path = Path(filename).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), SCRIPT_DIR, ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


def make_gpu_uks_from_chk(chk, xc, use_density_fit=True, auxbasis=None):
    mol, scf_rec = mol_chkfile.load_scf(str(chk))
    mf = gpubasedft.UKS(mol, xc=xc)
    if use_density_fit:
        mf = mf.density_fit(auxbasis=auxbasis)
    mf.mo_energy = cp.asarray(scf_rec["mo_energy"])
    mf.mo_coeff = cp.asarray(scf_rec["mo_coeff"])
    mf.mo_occ = cp.asarray(scf_rec["mo_occ"])
    mf.e_tot = scf_rec.get("e_tot", None)
    mf.converged = True
    if mf.mo_coeff.ndim != 3 or mf.mo_occ.ndim != 2:
        raise ValueError("USF-TDA example expects a UKS checkpoint; use UKS.chk.")
    return mf


# ===== Manually edit these parameters on the server =====
chk = "UKS.chk"
xc = "pbe0"
nstates = 15

use_density_fit = True
auxbasis = None
grid_level = 4

method = 1
SA = 0
remove = False
foo = 1.0
d_lda = 0.3
fglobal = 0.0
fit = False
davidson_backend = "cpu"
davidson_matvec_batch_size = 15
run_analyse = True
analyse_threshold = 0.05
save_results = True
save_file = "USF_mol"
# ========================================================


def main():
    chk_path = resolve_file(chk)
    if not chk_path.exists():
        raise FileNotFoundError(f"chk file does not exist: {chk_path}")

    mf = make_gpu_uks_from_chk(chk_path, xc, use_density_fit, auxbasis)
    mf.grids.level = grid_level

    print("backend:", backend_info())
    print("CUDA device count:", cp.cuda.runtime.getDeviceCount())
    print("chk path:", chk_path.resolve())
    print("mf class:", type(mf))
    print("reference: UKS")
    print("with_df:", type(getattr(mf, "with_df", None)))
    print("e_tot:", mf.e_tot)
    print("mo_coeff shape:", mf.mo_coeff.shape)
    print("mo_occ shape:", mf.mo_occ.shape)
    print("SA:", SA)
    print("remove:", remove)

    usf_method = XSF_TDA_down(
        mf,
        method=method,
        SA=SA,
        davidson=True,
        davidson_backend=davidson_backend,
        davidson_matvec_batch_size=davidson_matvec_batch_size,
    )
    ee, vv = usf_method.kernel(
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

    print("converged:", usf_method.converged)
    print("energies / eV:")
    print(np.asarray(ee))
    print("vectors shape:", vv.shape)

    if run_analyse:
        usf_method.analyse(threshold=analyse_threshold)
    return usf_method, ee, vv


if __name__ == "__main__":
    main()
