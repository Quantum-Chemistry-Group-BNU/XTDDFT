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
ROOT = SCRIPT_DIR.parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import cupy as cp
import numpy as np
from pyscf.pbc.scf import chkfile as pbc_chkfile
from pyscf.pbc import dft as pbcdft

from XTDDFT_dev.utils.backend import backend_info, set_backend
from XTDDFT_dev.utils.unit import ha2eV
from XTDDFT_dev.utils.visualize import write_nto_cubes
from XTDDFT_dev.XTDDFT.xtda import XTDA


# ===== Manually edit these parameters on the server =====
chk = "pre_uks_ccpVDZ.chk"
results_file = "utda_method0_spin_conserving_davidson_nstates8_results.npz"
outdir = "utda_method0_spin_conserving_nstates8_nto_ground_to_state0"
prefix = "utda_method0_ground_to_state0"

xc = "pbe0"
use_density_fit = True

method = 0
davidson = True
davidson_backend = "cpu"
so2st = False
dense_batch_size = 64

state_i = None
state_f = 0
nroots = 5
resolution = 0.15
# ========================================================


set_backend("gpu")

chk_path = Path(chk).expanduser()
if not chk_path.is_absolute():
    for base in (Path.cwd(), SCRIPT_DIR, ROOT):
        candidate = base / chk_path
        if candidate.exists():
            chk_path = candidate
            break
if not chk_path.exists():
    raise FileNotFoundError(
        f"chk file {chk!r} was not found in cwd={Path.cwd()}, "
        f"script_dir={SCRIPT_DIR}, or repo_root={ROOT}."
    )

results_path = Path(results_file).expanduser()
if not results_path.is_absolute():
    for base in (Path.cwd(), chk_path.parent, SCRIPT_DIR, ROOT):
        candidate = base / results_path
        if candidate.exists():
            results_path = candidate
            break
if not results_path.exists():
    raise FileNotFoundError(
        f"UTDA results file {results_file!r} was not found in cwd={Path.cwd()}, "
        f"chk_dir={chk_path.parent}, script_dir={SCRIPT_DIR}, or repo_root={ROOT}."
    )

data = np.load(results_path)
if "vectors" not in data:
    raise KeyError(f"{results_path} does not contain saved key 'vectors'")
vectors = np.asarray(data["vectors"])
if "e_ha" in data:
    energies_ha = np.asarray(data["e_ha"], dtype=float).reshape(-1)
elif "e_ev" in data:
    energies_ha = np.asarray(data["e_ev"], dtype=float).reshape(-1) / ha2eV
else:
    raise KeyError(f"{results_path} must contain either 'e_ha' or 'e_ev'")
if vectors.ndim != 2:
    raise ValueError(f"vectors must be 2D; got shape={vectors.shape}")
if vectors.shape[1] != energies_ha.size:
    raise ValueError(
        "vectors.shape[1] must match the number of energies. "
        f"got vectors.shape={vectors.shape}, energies.shape={energies_ha.shape}"
    )

print("backend:", backend_info())
print("chk path:", chk_path.resolve())
print("results path:", results_path.resolve())
print("state_i -> state_f:", state_i, "->", state_f)
print("nroots:", nroots)
print("resolution:", resolution)

cell, scf_rec = pbc_chkfile.load_scf(str(chk_path))

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
        "This UTDA NTO script expects a UKS checkpoint with spin-resolved "
        "mo_coeff/mo_occ arrays. Use a UKS chk file, not ROKS/RKS."
    )

with_df = getattr(mf, "with_df", None)
if with_df is not None and hasattr(with_df, "is_gamma_point"):
    with_df.is_gamma_point = True
if use_density_fit and (
    with_df is None or with_df.__class__.__name__.upper() == "FFTDF"
):
    raise RuntimeError(
        "use_density_fit=True was requested, but the GPU UKS object did not "
        "get a GDF density-fitting object."
    )

utda_method = XTDA(
    mf,
    method=method,
    davidson=davidson,
    davidson_backend=davidson_backend,
    so2st=so2st,
    dense_batch_size=dense_batch_size,
)
utda_method.v = vectors
utda_method.e = energies_ha
utda_method.nstates = energies_ha.size
utda_method.converged = np.ones(energies_ha.size, dtype=bool)

print("cell nao:", cell.nao_nr())
print("spin:", cell.spin)
print("e_tot:", mf.e_tot)
print("energies / eV:", energies_ha * ha2eV)
print("vectors shape:", vectors.shape)

result = write_nto_cubes(
    utda_method,
    state_f=state_f,
    state_i=state_i,
    nroots=nroots,
    outdir=outdir,
    prefix=prefix,
    resolution=resolution,
)

singular_values = result["singular_values"]
weights = singular_values ** 2
if np.sum(weights) > 0:
    weights = weights / np.sum(weights)

print("NTO singular values:", singular_values)
print("NTO normalized weights:", weights)
print("Cube files:")
for item in result["files"]:
    print(" ", item["path"])
