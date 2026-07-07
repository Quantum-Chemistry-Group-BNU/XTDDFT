#!/usr/bin/env python
import os
# ============================================================
# 0. Scratch / temporary directory
# ============================================================

# scratch_dir = "/home/dell/users/clf/scratch/pyscf_tmp"
# os.makedirs(scratch_dir, exist_ok=True)
# 
# os.environ["PYSCF_TMPDIR"] = scratch_dir
# os.environ["TMPDIR"] = scratch_dir
# os.environ["TEMP"] = scratch_dir
# os.environ["TMP"] = scratch_dir

# Optional: force Python tempfile module to use this directory
# import tempfile
# tempfile.tempdir = scratch_dir
import sys
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

import cupy as cp
import numpy as np
from pyscf.pbc.scf import chkfile as pbc_chkfile
from pyscf.pbc import dft as pbcdft

from XTDDFT_dev.utils.backend import backend_info, set_backend
from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down


def save_becke_grids(grids, filename):
    coords = cp.asnumpy(grids.coords) if isinstance(grids.coords, cp.ndarray) else np.asarray(grids.coords)
    weights = cp.asnumpy(grids.weights) if isinstance(grids.weights, cp.ndarray) else np.asarray(grids.weights)
    np.savez_compressed(filename, coords=coords, weights=weights, level=np.array([grids.level]))
    print(f"Saved Becke grids to {filename}")
    print("ngrids =", coords.shape[0])
    print("sum weights =", weights.sum())


def load_or_build_becke_grids(mf, cell, filename, level=4):
    filename = Path(filename).expanduser().resolve()

    print("cwd =", Path.cwd())
    print("grid_file =", filename)
    print("grid exists =", filename.is_file())

    # 不要重新 mf.grids = pbcdft.BeckeGrids(cell)
    # 尽量保留 mf.to_gpu() 后已有的 grids 对象
    if getattr(mf, "grids", None) is None:
        mf.grids = pbcdft.BeckeGrids(cell)

    mf.grids.level = level

    if filename.is_file():
        data = np.load(filename)
        mf.grids.coords = cp.asarray(data["coords"])
        mf.grids.weights = cp.asarray(data["weights"])
        print(f"Loaded Becke grids from {filename}")

    else:
        print(f"Saved grids not found. Building Becke grids at level {level} ...")
        mf.grids.build()

        if not isinstance(mf.grids.coords, cp.ndarray):
            mf.grids.coords = cp.asarray(mf.grids.coords)
        if not isinstance(mf.grids.weights, cp.ndarray):
            mf.grids.weights = cp.asarray(mf.grids.weights)

        save_becke_grids(mf.grids, filename)

    # 避免沿用旧的筛选表
    for key in ("non0tab", "screen_index"):
        if hasattr(mf.grids, key):
            setattr(mf.grids, key, None)

    print("grids class:", type(mf.grids))
    print("coords type:", type(mf.grids.coords))
    print("weights type:", type(mf.grids.weights))
    print("ngrids =", mf.grids.coords.shape[0])
    print("sum weights =", mf.grids.weights.sum())

    # 防止后续 XSF_TDA_down / numint 再次调用 grids.build() 重新生成 grids
    def _reuse_loaded_grids(*args, **kwargs):
        print("Skip mf.grids.build(): using current Becke grids")
        return mf.grids

    mf.grids.build = _reuse_loaded_grids

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
SA = 3

use_density_fit = True  # If GDF is problematic on the machine, try False.

use_saved_grids = True
grid_level = 4
grid_file = "becke_grids_ccpvdz_level4.npz"

remove = None
foo = 1.0
d_lda = 0.3

delta_a_diag_j_batch_size = None # 现在的策略不需要它来主要调节内存了
delta_a_diag_method="df"          # "response", "df", "pbc_df", "auto"
delta_a_diag_df_aux_batch_size=256

davidson_matvec_batch_size = 15
delta_a_jk_batch_size = 15


fglobal = None
fit = True
davidson_backend = "cpu"  # CPU Davidson + GPU matrix-vector product.
save_results = True
save_file = 'XSF'
# ========================================================


def main():
    set_backend("gpu")
    cell, mf = make_gpu_roks_from_chk(chk, xc, use_density_fit=use_density_fit)

    if use_saved_grids:
        load_or_build_becke_grids(mf, cell, grid_file, level=grid_level)

    print("backend:", backend_info())
    print("CUDA device count:", cp.cuda.runtime.getDeviceCount())
    print("mf class:", type(mf))
    print("with_df:", type(getattr(mf, "with_df", None)))
    print("with_df.is_gamma_point:", getattr(getattr(mf, "with_df", None), "is_gamma_point", None))
    print("cell mesh:", mf.cell.mesh)
    print("e_tot:", mf.e_tot)
    print("mo_coeff shape:", mf.mo_coeff.shape)
    print("mo_occ shape:", mf.mo_occ.shape)
    print("SA:", SA)

    last = None
    for method in (1,):
        print("=" * 80)
        print(f"Running XSF_TDA_down method={method}")
        xsf_method = XSF_TDA_down(
            mf, method=method, SA=SA, davidson=True,
            davidson_backend=davidson_backend,
            delta_a_diag_j_batch_size = delta_a_diag_j_batch_size,
            delta_a_diag_method = delta_a_diag_method,
            delta_a_diag_df_aux_batch_size = delta_a_diag_df_aux_batch_size,
            davidson_matvec_batch_size = davidson_matvec_batch_size,
            delta_a_jk_batch_size = delta_a_jk_batch_size,
        )
        #init_space = np.load("XSF_tmp.npz")["vectors"]
        ee, vv = xsf_method.kernel(
            nstates=nstates,
            #init_space=init_space,
            remove=remove,
            foo=foo,
            d_lda=d_lda,
            fglobal=fglobal,
            fit=fit,
            save=save_results,
            save_file = save_file,
            #jk_batch_size=jk_batch_size,

        )
        xsf_method.analyse()
        cp.cuda.Stream.null.synchronize()

        print("converged:", xsf_method.converged)
        print("fglobal:", getattr(xsf_method, "fglobal", None))
        print("energies / eV:")
        print(np.asarray(ee))
        print("vectors shape:", vv.shape)
        last = xsf_method, ee, vv
    return last


if __name__ == "__main__":
    main()
