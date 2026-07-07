#!/usr/bin/env python
import os
import sys
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")


import cupy as cp
import numpy as np
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.scf import chkfile as pbc_chkfile

from XTDDFT_dev.utils.backend import backend_info, set_backend
from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down



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


def make_gpu_uks_from_chk(chk, xc, use_density_fit=True):
    cell, scf_rec = pbc_chkfile.load_scf(chk)

    uks_mf = pbcdft.UKS(cell)
    if use_density_fit:
        uks_mf = uks_mf.density_fit()
    uks_mf.xc = xc
    uks_mf.converged = True

    mf = uks_mf.to_gpu()
    mf.xc = xc
    mf.mo_energy = cp.asarray(scf_rec["mo_energy"])
    mf.mo_coeff = cp.asarray(scf_rec["mo_coeff"])
    mf.mo_occ = cp.asarray(scf_rec["mo_occ"])
    mf.e_tot = scf_rec.get("e_tot", None)
    mf.converged = True

    if mf.mo_coeff.ndim != 3 or mf.mo_occ.ndim != 2:
        raise ValueError(
            "USF-TDA expects a UKS checkpoint with spin-resolved mo_coeff/mo_occ. "
            "Use a UKS chk file, not ROKS/RKS."
        )

    with_df = getattr(mf, "with_df", None)
    if with_df is not None and hasattr(with_df, "is_gamma_point"):
        with_df.is_gamma_point = True
    if use_density_fit and (
        with_df is None or with_df.__class__.__name__.upper() == "FFTDF"
    ):
        raise RuntimeError(
            "use_density_fit=True was requested, but the GPU UKS object did "
            "not get a GDF density-fitting object."
        )
    return cell, mf


# ===== Manually edit these parameters on the server =====
chk = "pbe0_uks_ccpVDZ.chk"
xc = "pbe0"
nstates = 15

use_density_fit = True

grid_level = 4
grid_file = "becke_grids_ccpvdz_level4.npz"

method = 1  # method=0 ALDA0 spin-flip response.
SA = 0      # USF-TDA: no finite-spin Delta A correction.
remove = False
foo = 1.0
d_lda = 0.3
fglobal = 0.0
fit = False
davidson_backend = "cpu"  # CPU Davidson + GPU matrix-vector product.

davidson_matvec_batch_size = 15
run_analyse = True
analyse_threshold = 0.05
save_results = True
save_file = 'USF'
# ========================================================


def main():
    set_backend("gpu")

    cell, mf = make_gpu_uks_from_chk(chk, xc, use_density_fit=use_density_fit)
    load_or_build_becke_grids(mf, cell, grid_file, level=grid_level)

    print("backend:", backend_info())
    print("CUDA device count:", cp.cuda.runtime.getDeviceCount())
    print("mf class:", type(mf))
    print("reference: UKS")
    print("with_df:", type(getattr(mf, "with_df", None)))
    print("with_df.is_gamma_point:", getattr(getattr(mf, "with_df", None), "is_gamma_point", None))
    print("cell mesh:", mf.cell.mesh)
    print("e_tot:", mf.e_tot)
    print("mo_coeff shape:", mf.mo_coeff.shape)
    print("mo_occ shape:", mf.mo_occ.shape)
    print("SA:", SA)
    print("remove:", remove)

    print("=" * 80)
    print(f"Running USF_TDA_down method={method}")
    usf_method = XSF_TDA_down(
        mf,
        method=method,
        SA=SA,
        davidson=True,
        davidson_backend=davidson_backend,
        davidson_matvec_batch_size = davidson_matvec_batch_size,
    )
    #init_space = np.load("USF_tmp.npz")["vectors"]
    ee, vv = usf_method.kernel(
        nstates=nstates,
        #init_space=init_space,
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
    print("result file:", getattr(usf_method, "result_file", None))
    print("energies / eV:")
    print(np.asarray(ee))
    print("vectors shape:", vv.shape)

    if run_analyse:
        usf_method.analyse(threshold=analyse_threshold)
    return usf_method, ee, vv


if __name__ == "__main__":
    main()
