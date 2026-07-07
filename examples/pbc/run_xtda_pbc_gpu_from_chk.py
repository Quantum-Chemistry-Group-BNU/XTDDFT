#!/usr/bin/env python
import os
import sys
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

# SCRIPT_DIR = Path(__file__).resolve().parent
# ROOT = SCRIPT_DIR.parent
# PROJECT_PARENT = ROOT.parent
# if str(PROJECT_PARENT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_PARENT))

import cupy as cp
import numpy as np
from pyscf.pbc.scf import chkfile as pbc_chkfile
from pyscf.pbc import dft as pbcdft

from XTDDFT_dev.utils.backend import backend_info, set_backend
from XTDDFT_dev.XTDDFT.xtda import XTDA


def resolve_grid_file(filename, chk_path):
    path = Path(filename).expanduser()
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"grid file does not exist: {path}")
        return path

    for base in (Path.cwd(), chk_path.parent, SCRIPT_DIR, ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"grid file {filename!r} was not found in cwd={Path.cwd()}, "
        f"chk_dir={chk_path.parent}, script_dir={SCRIPT_DIR}, or repo_root={ROOT}."
    )


def load_becke_grids(mf, cell, filename, level=4):
    data = np.load(filename)
    mf.grids = pbcdft.BeckeGrids(cell)
    mf.grids.level = level
    mf.grids.coords = cp.asarray(data["coords"])
    mf.grids.weights = cp.asarray(data["weights"])
    print(f"Loaded Becke grids from {filename}")
    print("ngrids =", mf.grids.coords.shape[0])
    print("sum weights =", mf.grids.weights.sum())
    return mf.grids


def make_gpu_roks_from_chk(chk, xc, use_density_fit=True):
    cell, scf_rec = pbc_chkfile.load_scf(chk)

    roks_mf = pbcdft.ROKS(cell)
    if use_density_fit:
        roks_mf = roks_mf.density_fit()
    roks_mf.xc = xc
    roks_mf.converged = True

    mf = roks_mf.to_gpu()
    mf.xc = xc
    mf.mo_energy = cp.asarray(scf_rec["mo_energy"])
    mf.mo_coeff = cp.asarray(scf_rec["mo_coeff"])
    mf.mo_occ = cp.asarray(scf_rec["mo_occ"])
    mf.e_tot = scf_rec.get("e_tot", None)
    mf.converged = True

    with_df = getattr(mf, "with_df", None)
    if with_df is not None and hasattr(with_df, "is_gamma_point"):
        with_df.is_gamma_point = True
    return cell, mf


# ===== Manually edit these parameters on the server =====
chk = "roks_from_uks_ccpVDZ.chk"
xc = "pbe0"
nstates = 15

use_density_fit = True  # If GDF is problematic on the machine, try False.

grid_level = 4
grid_file = "becke_grids_ccpvdz_level4.npz"

method = 0  # XTDA currently implements method=0 spin-conserving response.
davidson = True
davidson_backend = "cpu"  # CPU Davidson + GPU matrix-vector product.
so2st = False
davidson_matvec_batch_size = 15
jk_batch_size = 15  # Split Davidson trial-vector JK builds; use None to disable.
dense_batch_size = 64
analyse_threshold = 0.03
save_results = True
save_file = 'XTDA'
# ========================================================


def main():
    set_backend("gpu")
    # chk_path = resolve_existing_file(chk, "chk")
    # grid_path = resolve_grid_file(grid_file, chk_path)

    # print("chk path:", chk_path)
    #print("grid path:", grid_path)

    cell, mf = make_gpu_roks_from_chk(chk, xc, use_density_fit=use_density_fit)

    load_becke_grids(mf, cell, grid_file, level=grid_level)

    print("backend:", backend_info())
    print("CUDA device count:", cp.cuda.runtime.getDeviceCount())
    print("mf class:", type(mf))
    print("with_df:", type(getattr(mf, "with_df", None)))
    print("with_df.is_gamma_point:", getattr(getattr(mf, "with_df", None), "is_gamma_point", None))
    print("cell mesh:", mf.cell.mesh)
    print("e_tot:", mf.e_tot)
    print("mo_coeff shape:", mf.mo_coeff.shape)
    print("mo_occ shape:", mf.mo_occ.shape)

    print("=" * 80)
    print(f"Running XTDA method={method}")
    xtda_method = XTDA(
        mf,
        method=method,
        davidson=davidson,
        davidson_backend=davidson_backend,
        so2st=so2st,
        dense_batch_size=dense_batch_size,
        davidson_matvec_batch_size = davidson_matvec_batch_size,
        jk_batch_size=jk_batch_size,
        jk_block_split=True
    )
    #init_space = np.load("XTDA_tmp.npz")["raw_vectors"]
    ee, vv = xtda_method.kernel(nstates=nstates,
                                #init_space=init_space,
                                save=save_results,
                                save_file=f'{save_file}.npz'
                                )
    xtda_method.analyse(threshold=analyse_threshold)
    cp.cuda.Stream.null.synchronize()

    print("converged:", getattr(xtda_method, "converged", None))
    print("energies / eV:")
    print(np.asarray(ee))
    print("vectors shape:", vv.shape)
    return xtda_method, ee, vv


if __name__ == "__main__":
    main()
