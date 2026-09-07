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
from pyscf.pbc.scf import chkfile as pbc_chkfile
from pyscf.pbc import dft as pbcdft

from XTDDFT_dev.utils.backend import backend_info, set_backend
from XTDDFT_dev.XTDDFT.xtda import XTDA


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
            "This UTDA example expects a UKS checkpoint with spin-resolved "
            "mo_coeff/mo_occ arrays. Use a UKS chk file, not ROKS/RKS."
        )

    with_df = getattr(mf, "with_df", None)
    if with_df is not None and hasattr(with_df, "is_gamma_point"):
        with_df.is_gamma_point = True
    if use_density_fit and (
        with_df is None or with_df.__class__.__name__.upper() == "FFTDF"
    ):
        raise RuntimeError(
            "use_density_fit=True was requested, but the GPU UKS object did "
            "not get a GDF density-fitting object. Check gpu4pyscf/PySCF PBC "
            "density_fit support on this machine."
        )
    return cell, mf


# ===== Manually edit these parameters on the server =====
chk = "pbe0_uks_ccpVDZ.chk"
xc = "pbe0"
nstates = 15

use_density_fit = True  # Use GDF for hybrid PBC response; set False only for FFTDF fallback.

grid_level = 4
grid_file = "becke_grids_ccpvdz_level4.npz"

method = 0  # UKS input makes XTDA run unrestricted TDA.
davidson = True
davidson_backend = "cpu"  # CPU Davidson + GPU matrix-vector product.
so2st = False  # Spin-tensor transformation is not used for UKS amplitudes.
dense_batch_size = 64
davidson_matvec_batch_size = 15
#analyse_threshold = 0.1
run_analyse = True
analyse_delta_s2 = True
save_results = True
save_file = 'UTDA'
# ========================================================


def main():
    set_backend("gpu")

    #print("chk path:", chk_path)
    #print("grid path:", grid_path)

    cell, mf = make_gpu_uks_from_chk(chk, xc, use_density_fit=use_density_fit)

    load_becke_grids(mf, cell, grid_file, level=grid_level)

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

    print("=" * 80)
    print(f"Running UTDA method={method}")
    utda_method = XTDA(
        mf,
        method=method,
        davidson=davidson,
        davidson_backend=davidson_backend,
        so2st=so2st,
        dense_batch_size=dense_batch_size,
        davidson_matvec_batch_size = davidson_matvec_batch_size,
    )
    #init_space = np.load("UTDA_tmp.npz")["raw_vectors"]
    ee, vv = utda_method.kernel(
        nstates=nstates,
        #init_space=init_space,
        save=save_results,
        save_file=save_file,
    )
    cp.cuda.Stream.null.synchronize()

    print("converged:", getattr(utda_method, "converged", None))
    print("result file:", getattr(utda_method, "result_file", None))
    print("energies / eV:")
    print(np.asarray(ee))
    print("vectors shape:", vv.shape)

    if run_analyse:
        utda_method.analyse(
            #threshold=analyse_threshold,
            compute_s2=analyse_delta_s2,
        )
    return utda_method, ee, vv


if __name__ == "__main__":
    main()
