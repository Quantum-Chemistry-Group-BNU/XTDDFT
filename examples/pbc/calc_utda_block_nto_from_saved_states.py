#!/usr/bin/env python
import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")
os.environ.setdefault("CUPY_ACCELERATORS", "cub,cutensor")
os.environ.setdefault("GPU4PYSCF_NUMINT_BLOCK_SIZE", "32768")
os.environ.setdefault("GPU4PYSCF_NUMINT_BLOCK_MEM_FRACTION", "0.4")
import re
import sys
from pathlib import Path

import numpy as np
from pyscf import dft
from pyscf.scf import chkfile as mol_chkfile
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.scf import chkfile as pbc_chkfile


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT_dev.XTDDFT.xtda import XTDA
from XTDDFT_dev.utils.backend import set_backend
from XTDDFT_dev.utils.unit import ha2eV


# ===== Manually edit these parameters on the server =====
chk = "pre_uks_def2svp.chk"
results_file = "utda.npz"
outdir = "utda_block_nto"
prefix = "utda"

xc = "pbe0"
states = [0, 1, 2]
nroots = 3
resolution = 0.15
component_tol = 1.0e-12
# ========================================================


def find_existing_path(name, bases):
    path = Path(name).expanduser()
    if path.is_absolute():
        return path
    for base in bases:
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


def load_uks_reference(chk_path):
    try:
        cell, scf_rec = pbc_chkfile.load_scf(str(chk_path))
        mf = pbcdft.UKS(cell)
        kind = "pbc"
    except Exception as pbc_error:
        try:
            mol, scf_rec = mol_chkfile.load_scf(str(chk_path))
            mf = dft.UKS(mol)
            kind = "mol"
        except Exception as mol_error:
            raise RuntimeError(
                f"Failed to load {chk_path} as either PBC or molecular UKS chk. "
                f"PBC error: {pbc_error!r}; molecular error: {mol_error!r}"
            ) from mol_error

    mf.xc = xc
    mf.mo_energy = np.asarray(scf_rec["mo_energy"])
    mf.mo_coeff = np.asarray(scf_rec["mo_coeff"])
    mf.mo_occ = np.asarray(scf_rec["mo_occ"])
    mf.e_tot = scf_rec.get("e_tot", None)
    mf.converged = True

    if mf.mo_coeff.ndim != 3 or mf.mo_occ.ndim != 2:
        raise ValueError(
            "UTDA block-NTO analysis expects a UKS checkpoint with spin-resolved "
            "mo_coeff/mo_occ arrays."
        )
    return kind, mf


def load_saved_vectors(results_path):
    data = np.load(results_path)
    if "vectors" not in data:
        raise KeyError(f"{results_path} does not contain saved key 'vectors'")
    vectors = np.asarray(data["vectors"])
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D; got shape={vectors.shape}")
    if "e_ha" in data:
        energies_ha = np.asarray(data["e_ha"], dtype=float).reshape(-1)
    elif "e_ev" in data:
        energies_ha = np.asarray(data["e_ev"], dtype=float).reshape(-1) / ha2eV
    else:
        energies_ha = np.zeros(vectors.shape[1])
    if energies_ha.size != vectors.shape[1]:
        raise ValueError(
            "vectors.shape[1] must match the number of energies. "
            f"got vectors.shape={vectors.shape}, energies.shape={energies_ha.shape}"
        )
    return vectors, energies_ha


def system_from_mf(mf):
    return mf.cell if hasattr(mf, "cell") else mf.mol


def default_cubegen_orbital(system):
    if hasattr(system, "lattice_vectors"):
        try:
            from pyscf.pbc.tools import cubegen
        except ImportError:
            from pyscf.tools import cubegen
    else:
        from pyscf.tools import cubegen
    return cubegen.orbital


def safe_label(text):
    return re.sub(r"[^0-9A-Za-z_+-]+", "", text.replace("(", "").replace(")", ""))


def write_block_nto_cubes(method, blocks, outdir, prefix, resolution=0.15):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    system = system_from_mf(method.mf)
    cubegen_orbital = default_cubegen_orbital(system)
    mo_coeff = np.asarray(method.ctx.mo_coeff)
    nmo_a = mo_coeff[0].shape[1]
    files = []

    for block_name, block in blocks.items():
        label = safe_label(block_name)
        for pair, (hole, particle) in enumerate(zip(block["holes"].T, block["particles"].T), start=1):
            for kind, vector in (("hole", hole), ("particle", particle)):
                for spin, mo, coeff in (
                    ("alpha", mo_coeff[0], vector[:nmo_a]),
                    ("beta", mo_coeff[1], vector[nmo_a:]),
                ):
                    if np.linalg.norm(coeff) <= component_tol:
                        continue
                    outfile = outdir / f"{prefix}_{label}_svd{pair}_{kind}_{spin}.cube"
                    cubegen_orbital(system, str(outfile), mo @ coeff, resolution=resolution)
                    files.append(str(outfile))
    return files


set_backend("cpu")
chk_path = find_existing_path(chk, (Path.cwd(), SCRIPT_DIR, ROOT))
if not chk_path.exists():
    raise FileNotFoundError(
        f"chk file {chk!r} was not found in cwd={Path.cwd()}, script_dir={SCRIPT_DIR}, or repo_root={ROOT}."
    )
results_path = find_existing_path(results_file, (Path.cwd(), chk_path.parent, SCRIPT_DIR, ROOT))
if not results_path.exists():
    raise FileNotFoundError(
        f"UTDA results file {results_file!r} was not found in cwd={Path.cwd()}, "
        f"chk_dir={chk_path.parent}, script_dir={SCRIPT_DIR}, or repo_root={ROOT}."
    )

kind, mf = load_uks_reference(chk_path)
vectors, energies_ha = load_saved_vectors(results_path)

utda_method = XTDA(mf, method=0, davidson=True, davidson_backend="cpu", so2st=False)
utda_method.v = vectors
utda_method.e = energies_ha
utda_method.nstates = energies_ha.size
utda_method.converged = np.ones(energies_ha.size, dtype=bool)

print("reference kind:", kind)
print("chk path:", chk_path.resolve())
print("results path:", results_path.resolve())
print("states:", states)
print("energies / eV:", energies_ha * ha2eV)
print("vectors shape:", vectors.shape)

for state in states:
    blocks = utda_method.block_nto(state=state, nroots=nroots)
    block_total = sum(block["block_weight"] for block in blocks.values())
    state_outdir = Path(outdir) / f"state{state}"
    state_prefix = f"{prefix}_state{state}"

    print()
    print(f"State {state}")
    print("blocks:")
    for name, block in blocks.items():
        rel = block["block_weight"] / block_total if block_total > 0 else 0.0
        weights = block["weights"]
        local_weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        print(f"  {name}: {block['source']} -> {block['target']}")
        print(f"    block weight: {block['block_weight']:.12g}  fraction: {rel:.6f}")
        print(f"    singular values: {block['singular_values']}")
        print(f"    normalized kept weights: {local_weights}")

    files = write_block_nto_cubes(
        utda_method,
        blocks,
        outdir=state_outdir,
        prefix=state_prefix,
        resolution=resolution,
    )
    print("Cube files:")
    for file in files:
        print(" ", file)
