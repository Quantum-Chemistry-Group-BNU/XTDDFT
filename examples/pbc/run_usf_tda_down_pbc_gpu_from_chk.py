#!/usr/bin/env python
import os
import sys
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import cupy as cp
import numpy as np
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.scf import chkfile as pbc_chkfile

from XTDDFT_dev.utils.backend import backend_info, set_backend
from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down


def resolve_existing_file(filename, label):
    path = Path(filename).expanduser()
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
        return path

    for base in (Path.cwd(), SCRIPT_DIR, ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{label} file {filename!r} was not found in cwd={Path.cwd()}, "
        f"script_dir={SCRIPT_DIR}, or repo_root={ROOT}."
    )


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
chk = "hse06_uks_ccpVDZ.chk"
xc = "hse06"
nstates = 8

use_density_fit = True

grid_level = 4
grid_file = "becke_grids_ccpVDZ_level4.npz"

method = 0  # method=0 ALDA0 spin-flip response.
SA = 0      # USF-TDA: no finite-spin Delta A correction.
remove = False
foo = 1.0
d_lda = 0.3
fglobal = 0.0
fit = False
davidson_backend = "cpu"  # CPU Davidson + GPU matrix-vector product.

run_analyse = False
analyse_threshold = 0.05
save_results = True
save_file = None
# ========================================================


def main():
    set_backend("gpu")
    chk_path = resolve_existing_file(chk, "chk")
    grid_path = resolve_grid_file(grid_file, chk_path)

    print("chk path:", chk_path)
    print("grid path:", grid_path)

    cell, mf = make_gpu_uks_from_chk(chk_path, xc, use_density_fit=use_density_fit)
    load_becke_grids(mf, cell, grid_path, level=grid_level)

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
    print("result file:", getattr(usf_method, "result_file", None))
    print("energies / eV:")
    print(np.asarray(ee))
    print("vectors shape:", vv.shape)

    if run_analyse:
        usf_method.analyse(threshold=analyse_threshold)
    return usf_method, ee, vv


if __name__ == "__main__":
    main()
