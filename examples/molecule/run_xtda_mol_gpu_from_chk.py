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
from XTDDFT_dev.XTDDFT.xtda import XTDA


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


def make_gpu_roks_from_chk(chk, xc, use_density_fit=True, auxbasis=None):
    mol, scf_rec = mol_chkfile.load_scf(str(chk))
    mo_energy = np.asarray(scf_rec["mo_energy"])
    mo_coeff = np.asarray(scf_rec["mo_coeff"])
    mo_occ = np.asarray(scf_rec["mo_occ"])
    if mo_coeff.ndim == 3:
        if mo_coeff.shape[0] != 2 or not np.allclose(mo_coeff[0], mo_coeff[1]):
            raise ValueError("XTDA ROKS example expects restricted orbitals; use ROKS.chk.")
        mo_coeff = mo_coeff[0]
        if mo_energy.ndim == 2:
            mo_energy = mo_energy[0]
        if mo_occ.ndim == 2:
            mo_occ = mo_occ[0] + mo_occ[1]

    mf = gpubasedft.ROKS(mol, xc=xc)
    if use_density_fit:
        mf = mf.density_fit(auxbasis=auxbasis)
    mf.mo_energy = cp.asarray(mo_energy)
    mf.mo_coeff = cp.asarray(mo_coeff)
    mf.mo_occ = cp.asarray(mo_occ)
    mf.e_tot = scf_rec.get("e_tot", None)
    mf.converged = True
    return mf


# ===== Manually edit these parameters on the server =====
chk = "ROKS.chk"
xc = "pbe0"
nstates = 15

use_density_fit = True
auxbasis = None
grid_level = 4

method = 0
davidson = True
davidson_backend = "cpu"
so2st = False
dense_batch_size = 64
davidson_matvec_batch_size = 15
jk_batch_size = 15
analyse_threshold = 0.03
save_results = True
save_file = "XTDA_mol.npz"
# ========================================================


def main():
    chk_path = resolve_file(chk)
    if not chk_path.exists():
        raise FileNotFoundError(f"chk file does not exist: {chk_path}")

    mf = make_gpu_roks_from_chk(chk_path, xc, use_density_fit, auxbasis)
    mf.grids.level = grid_level

    print("backend:", backend_info())
    print("CUDA device count:", cp.cuda.runtime.getDeviceCount())
    print("chk path:", chk_path.resolve())
    print("mf class:", type(mf))
    print("with_df:", type(getattr(mf, "with_df", None)))
    print("e_tot:", mf.e_tot)
    print("mo_coeff shape:", mf.mo_coeff.shape)
    print("mo_occ shape:", mf.mo_occ.shape)

    xtda_method = XTDA(
        mf,
        method=method,
        davidson=davidson,
        davidson_backend=davidson_backend,
        so2st=so2st,
        dense_batch_size=dense_batch_size,
        davidson_matvec_batch_size=davidson_matvec_batch_size,
        jk_batch_size=jk_batch_size,
        jk_block_split=True,
    )
    ee, vv = xtda_method.kernel(nstates=nstates, save=save_results, save_file=save_file)
    xtda_method.analyse(threshold=analyse_threshold)
    cp.cuda.Stream.null.synchronize()

    print("converged:", getattr(xtda_method, "converged", None))
    print("energies / eV:")
    print(np.asarray(ee))
    print("vectors shape:", vv.shape)
    return xtda_method, ee, vv


if __name__ == "__main__":
    main()
