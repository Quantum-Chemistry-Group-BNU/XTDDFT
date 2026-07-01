#!/usr/bin/env python
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

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

from XTDDFT_dev.XTDDFT.sf_tda_up import SF_TDA_up
from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT_dev.XTDDFT.xtda import XTDA
from XTDDFT_dev.utils.backend import set_backend
from XTDDFT_dev.utils.unit import ha2eV


# ===== Manually edit these parameters on the server =====
method_kind = "utda"  # "utda", "xtda", "usf", or "xsf"
chk = "pre_uks_def2svp.chk"
results_file = "utda.npz"
outdir = "top5_block_nto"
prefix = method_kind

xc = "pbe0"
states = [0, 1, 2]
top_ntos = 5
nroots_per_block = 10
resolution = 0.15
component_tol = 1.0e-12

sf_method = 0
xsf_SA = 3
xsf_remove = None
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


def load_reference(chk_path, kind):
    try:
        cell, rec = pbc_chkfile.load_scf(str(chk_path))
        if kind in ("utda", "usf"):
            mf = pbcdft.UKS(cell)
        else:
            mf = pbcdft.ROKS(cell)
        ref = "pbc"
    except Exception as pbc_error:
        try:
            mol, rec = mol_chkfile.load_scf(str(chk_path))
            if kind in ("utda", "usf"):
                mf = dft.UKS(mol)
            else:
                mf = dft.ROKS(mol)
            ref = "mol"
        except Exception as mol_error:
            raise RuntimeError(
                f"Failed to load {chk_path} as PBC or molecular chk. "
                f"PBC error: {pbc_error!r}; molecular error: {mol_error!r}"
            ) from mol_error

    mf.xc = xc
    mf.mo_energy = np.asarray(rec["mo_energy"])
    mf.mo_coeff = np.asarray(rec["mo_coeff"])
    mf.mo_occ = np.asarray(rec["mo_occ"])
    mf.e_tot = rec.get("e_tot", None)
    mf.converged = True

    with_df = getattr(mf, "with_df", None)
    if with_df is not None and hasattr(with_df, "is_gamma_point"):
        with_df.is_gamma_point = True
    return ref, mf


def load_saved_results(path):
    data = np.load(path)
    vectors = np.asarray(data["vectors"])
    if "e_ha" in data:
        energies = np.asarray(data["e_ha"], dtype=float).reshape(-1)
    elif "e_ev" in data:
        energies = np.asarray(data["e_ev"], dtype=float).reshape(-1) / ha2eV
    else:
        energies = np.zeros(vectors.shape[1])
    if vectors.ndim != 2 or vectors.shape[1] != energies.size:
        raise ValueError(f"bad saved shapes: vectors={vectors.shape}, energies={energies.shape}")
    return data, vectors, energies


def safe_label(text):
    return re.sub(r"[^0-9A-Za-z_+-]+", "", text.replace("(", "").replace(")", ""))


def system_from_mf(mf):
    return mf.cell if hasattr(mf, "cell") else mf.mol


def cubegen_orbital(system):
    if hasattr(system, "lattice_vectors"):
        try:
            from pyscf.pbc.tools import cubegen
        except ImportError:
            from pyscf.tools import cubegen
    else:
        from pyscf.tools import cubegen
    return cubegen.orbital


def top_blocks(blocks, top_ntos):
    ranked = []
    total = sum(float(np.real_if_close(block["block_weight"])) for block in blocks.values())
    for block_name, block in blocks.items():
        for pair, weight in enumerate(block["weights"]):
            ranked.append((float(np.real_if_close(weight)), block_name, pair))
    ranked.sort(reverse=True)
    ranked = ranked[:top_ntos]

    filtered = {}
    for _weight, block_name, pair in ranked:
        filtered.setdefault(block_name, []).append(pair)

    out = {}
    for block_name, pairs in filtered.items():
        pairs = np.asarray(pairs, dtype=int)
        block = blocks[block_name]
        out[block_name] = dict(block)
        out[block_name]["singular_values"] = block["singular_values"][pairs]
        out[block_name]["weights"] = block["weights"][pairs]
        out[block_name]["holes"] = block["holes"][:, pairs]
        out[block_name]["particles"] = block["particles"][:, pairs]
        out[block_name]["original_pairs"] = pairs
    return ranked, total, out


def write_cubes(method, blocks, outdir, prefix):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    system = system_from_mf(method.mf)
    write_orbital = cubegen_orbital(system)
    mo_coeff = np.asarray(method.ctx.mo_coeff)
    spin_orbital = mo_coeff.ndim == 3 and next(iter(blocks.values()))["holes"].shape[0] > mo_coeff[0].shape[1]
    files = []

    for block_name, block in blocks.items():
        label = safe_label(block_name)
        original_pairs = block.get("original_pairs", np.arange(block["holes"].shape[1]))
        for pair, hole, particle in zip(original_pairs, block["holes"].T, block["particles"].T):
            for role, vector in (("hole", hole), ("particle", particle)):
                if spin_orbital:
                    nmo_a = mo_coeff[0].shape[1]
                    pieces = (
                        ("alpha", mo_coeff[0], vector[:nmo_a]),
                        ("beta", mo_coeff[1], vector[nmo_a:]),
                    )
                else:
                    mo = mo_coeff[0] if mo_coeff.ndim == 3 else mo_coeff
                    pieces = (("orb", mo, vector),)
                for spin, mo, coeff in pieces:
                    if np.linalg.norm(coeff) <= component_tol:
                        continue
                    outfile = outdir / f"{prefix}_{label}_svd{int(pair) + 1}_{role}_{spin}.cube"
                    write_orbital(system, str(outfile), mo @ coeff, resolution=resolution)
                    files.append(str(outfile))
    return files


set_backend("cpu")
method_kind = method_kind.lower()
chk_path = find_existing_path(chk, (Path.cwd(), SCRIPT_DIR, ROOT))
results_path = find_existing_path(results_file, (Path.cwd(), chk_path.parent, SCRIPT_DIR, ROOT))
if not chk_path.exists():
    raise FileNotFoundError(chk_path)
if not results_path.exists():
    raise FileNotFoundError(results_path)

data, vectors, energies_ha = load_saved_results(results_path)
ref_kind, mf = load_reference(chk_path, method_kind)

if method_kind == "utda":
    td = XTDA(mf, method=0, davidson=True, davidson_backend="cpu", so2st=False)
elif method_kind == "xtda":
    td = XTDA(mf, method=0, davidson=True, davidson_backend="cpu", so2st=False)
elif method_kind == "usf":
    td = SF_TDA_up(mf, method=sf_method, davidson=True, davidson_backend="cpu")
elif method_kind == "xsf":
    td = XSF_TDA_down(mf, method=sf_method, SA=xsf_SA, davidson=True, davidson_backend="cpu")
    td.re = (not td.type_u) if xsf_remove is None else bool(xsf_remove)
    if td.re:
        td.vects = td.get_vect()
else:
    raise ValueError("method_kind must be 'utda', 'xtda', 'usf', or 'xsf'")

td.v = vectors
td.e = energies_ha
td.nstates = energies_ha.size
td.converged = np.ones(energies_ha.size, dtype=bool)

print("method:", method_kind)
print("reference:", ref_kind)
print("chk path:", chk_path.resolve())
print("results path:", results_path.resolve())
print("energies / eV:", energies_ha * ha2eV)
print("vectors shape:", vectors.shape)

for state in states:
    blocks = td.block_nto(state=state, nroots=nroots_per_block)
    ranked, block_total, filtered_blocks = top_blocks(blocks, top_ntos)
    state_outdir = Path(outdir) / f"{method_kind}_state{state}"
    state_prefix = f"{prefix}_state{state}"

    print()
    print(f"State {state}")
    print(f"top {top_ntos} block NTOs by total block weight:")
    for rank, (weight, name, pair) in enumerate(ranked, start=1):
        frac = weight / block_total if block_total > 0 else 0.0
        print(f"  #{rank}: {name} svd{pair + 1} weight={weight:.12g} fraction={frac:.6f}")

    files = write_cubes(td, filtered_blocks, state_outdir, state_prefix)
    print("Cube files:")
    for file in files:
        print(" ", file)
