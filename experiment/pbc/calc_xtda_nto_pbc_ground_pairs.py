#!/usr/bin/env python
import os
import sys
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
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
from XTDDFT_dev.XTDDFT.xtda import XTDA


lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))


# ===== Manually edit these parameters on the server =====
chk = "roks_b3lyp_ccpVDZ.chk"
results_file = "xtda_spin_conserving_davidson_nstates8_results.npz"
outdir = "xtda_spin_conserving_davidson_nstates8_nto_ground_pairs_pbc"
prefix = "xtda_spin_conserving_pbc"

xc = "b3lyp"
use_density_fit = True

method = 0
pairs = [(None, 0), (None, 1), (None, 2), (None, 3)]
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
        f"XTDA results file {results_file!r} was not found in cwd={Path.cwd()}, "
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

max_state = max(state_f for state_i, state_f in pairs)
if max_state >= energies_ha.size:
    raise ValueError(
        f"requested pair includes state {max_state}, but only "
        f"{energies_ha.size} states were saved."
    )

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

xtda_method = XTDA(
    mf,
    method=method,
    davidson=True,
    davidson_backend="cpu",
    so2st=False,
    dense_batch_size=64,
)
xtda_method.v = vectors
xtda_method.e = energies_ha
xtda_method.nstates = energies_ha.size
xtda_method.converged = np.ones(energies_ha.size, dtype=bool)

print("natm:", cell.natm)
print("nelectron:", cell.nelectron)
print("spin:", cell.spin)
print("mesh:", cell.mesh)
print("e_tot:", mf.e_tot)
print("energies / eV:", energies_ha * ha2eV)
print("vectors shape:", vectors.shape)
print("output root:", Path(outdir).resolve())

for state_i, state_f in pairs:
    pair_label = f"ground_to_state{state_f}" if state_i is None else f"state{state_i}_to_state{state_f}"
    pair_outdir = Path(outdir) / pair_label
    pair_prefix = f"{prefix}_{pair_label}"

    print("=" * 80)
    print("NTO pair:", state_i, "->", state_f)
    print("pair output:", pair_outdir)

    result = write_nto_cubes(
        xtda_method,
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
