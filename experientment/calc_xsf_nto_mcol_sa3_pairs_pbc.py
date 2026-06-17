#!/usr/bin/env python
import os
import sys
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import numpy as np
from pyscf import lib
from pyscf.pbc.scf import chkfile as pbc_chkfile
from pyscf.pbc import dft as pbcdft

from XTDDFT_dev.utils.backend import backend_info, set_backend
from XTDDFT_dev.utils.unit import ha2eV
from XTDDFT_dev.utils.visualize import write_nto_cubes
from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down


lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))


# ===== Manually edit these parameters on the server =====
chk = "roks_b3lyp_ccpVDZ.chk"
results_file = "xsf_tda_down_mcol_davidson_sa3_nstates8_results.npz"
outdir = "xsf_tda_down_mcol_davidson_sa3_nstates8_nto_pairs"
prefix = "xsf_tda_down_mcol_sa3"

xc = "b3lyp"
use_density_fit = True

method = 1
SA = 3
pairs = [(0, 1), (0, 2), (0, 3), (0, 4)]
nroots = 5
resolution = 0.15
# ========================================================


set_backend("cpu")

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
        f"XSF results file {results_file!r} was not found in cwd={Path.cwd()}, "
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

max_state = max(max(pair) for pair in pairs)
if max_state >= energies_ha.size:
    raise ValueError(
        f"requested pair includes state {max_state}, but only "
        f"{energies_ha.size} states were saved."
    )

remove_saved = bool(np.asarray(data["remove"]).item()) if "remove" in data else None
fglobal_saved = float(np.asarray(data["fglobal"]).item()) if "fglobal" in data else None

print("backend:", backend_info())
print("chk path:", chk_path.resolve())
print("results path:", results_path.resolve())
print("pairs:", pairs)
print("nroots:", nroots)
print("resolution:", resolution)

cell, scf_rec = pbc_chkfile.load_scf(str(chk_path))

mf = pbcdft.ROKS(cell)
if use_density_fit:
    mf = mf.density_fit()
mf.xc = xc
mf.mo_energy = np.asarray(scf_rec["mo_energy"])
mf.mo_coeff = np.asarray(scf_rec["mo_coeff"])
mf.mo_occ = np.asarray(scf_rec["mo_occ"])
mf.e_tot = scf_rec.get("e_tot", None)
mf.converged = True

with_df = getattr(mf, "with_df", None)
if with_df is not None and hasattr(with_df, "is_gamma_point"):
    with_df.is_gamma_point = True

xsf_method = XSF_TDA_down(
    mf,
    method=method,
    SA=SA,
    davidson=True,
    davidson_backend="cpu",
)
xsf_method.v = vectors
xsf_method.e = energies_ha
xsf_method.nstates = energies_ha.size
xsf_method.converged = np.ones(energies_ha.size, dtype=bool)
xsf_method.re = (not xsf_method.type_u) if remove_saved is None else remove_saved
if xsf_method.re:
    xsf_method.vects = xsf_method.get_vect()
if fglobal_saved is not None:
    xsf_method.fglobal = fglobal_saved

print("natm:", cell.natm)
print("nelectron:", cell.nelectron)
print("spin:", cell.spin)
print("mesh:", cell.mesh)
print("e_tot:", mf.e_tot)
print("energies / eV:", energies_ha * ha2eV)
print("vectors shape:", vectors.shape)
print("remove:", xsf_method.re)
print("output root:", Path(outdir).resolve())

for state_i, state_f in pairs:
    pair_label = f"state{state_i}_to_state{state_f}"
    pair_outdir = Path(outdir) / pair_label
    pair_prefix = f"{prefix}_{pair_label}"

    print("=" * 80)
    print("NTO pair:", state_i, "->", state_f)
    print("pair output:", pair_outdir)

    result = write_nto_cubes(
        xsf_method,
        state_f=state_f,
        state_i=state_i,
        nroots=nroots,
        outdir=pair_outdir,
        prefix=pair_prefix,
        resolution=resolution,
    )

    singular_values = result["singular_values"]
    weights = singular_values ** 2
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)

    print("NTO singular values:", singular_values)
    print("NTO normalized weights among printed roots:", weights)
    print("Cube files:")
    for item in result["files"]:
        print(" ", item["path"])
