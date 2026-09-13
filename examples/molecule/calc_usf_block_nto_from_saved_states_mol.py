#!/usr/bin/env python
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

import numpy as np
from pyscf import dft
from pyscf.scf import chkfile as mol_chkfile
from pyscf.tools import cubegen

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT.XTDDFT.sf_tda_up import SF_TDA_up
from XTDDFT.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT.utils.backend import set_backend
from XTDDFT.utils.unit import ha2eV


# ===== Manually edit these parameters on the server =====
method_kind = "usf_down"  # "usf_up" or "usf_down"
chk = "UKS.chk"
results_file = "USF_mol.npz"
outdir = "top5_usf_block_nto_mol"
prefix = method_kind

xc = "pbe0"
states = [0, 1, 2]
top_ntos = 5
resolution = 0.15
component_tol = 1.0e-12
sf_method = 1
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


def load_saved_results(path):
    data = np.load(path)
    vectors = np.asarray(data["vectors"])
    if "e_ha" in data:
        energies = np.asarray(data["e_ha"], dtype=float).reshape(-1)
    elif "e_ev" in data:
        energies = np.asarray(data["e_ev"], dtype=float).reshape(-1) / ha2eV
    else:
        raise KeyError(f"{path} must contain 'e_ha' or 'e_ev'")
    if vectors.ndim != 2 or vectors.shape[1] != energies.size:
        raise ValueError(f"bad saved shapes: vectors={vectors.shape}, energies={energies.shape}")
    return vectors, energies


def load_uks(chk_path):
    mol, rec = mol_chkfile.load_scf(str(chk_path))
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.mo_energy = np.asarray(rec["mo_energy"])
    mf.mo_coeff = np.asarray(rec["mo_coeff"])
    mf.mo_occ = np.asarray(rec["mo_occ"])
    mf.e_tot = rec.get("e_tot", None)
    mf.converged = True
    if mf.mo_coeff.ndim != 3 or mf.mo_occ.ndim != 2:
        raise ValueError("USF NTO requires a UKS checkpoint with spin-resolved orbitals")
    return mf


def usf_down_block_nto(method, state=0, nroots=None):
    """Reference-to-state NTOs for alpha-occ -> beta-vir USF_DOWN."""
    amp = method._spinflip_amplitude_matrix_u(state)
    particles_local, singular_values, holes_h = np.linalg.svd(amp.T, full_matrices=False)
    holes_local = holes_h.conj().T
    block_weight = float(np.real_if_close(np.sum(np.abs(singular_values) ** 2)))
    if nroots is not None:
        keep = min(int(nroots), singular_values.size)
        singular_values = singular_values[:keep]
        particles_local = particles_local[:, :keep]
        holes_local = holes_local[:, :keep]

    mo_coeff = np.asarray(method.ctx.mo_coeff)
    nmo_a, nmo_b = mo_coeff[0].shape[1], mo_coeff[1].shape[1]
    occ_a = np.asarray(method.ctx.occidx_a, dtype=int)
    vir_b = np.asarray(method.ctx.viridx_b, dtype=int)
    holes = np.zeros((nmo_a + nmo_b, singular_values.size), dtype=holes_local.dtype)
    particles = np.zeros((nmo_a + nmo_b, singular_values.size), dtype=particles_local.dtype)
    holes[occ_a, :] = holes_local
    particles[nmo_a + vir_b, :] = particles_local
    return {
        "SF_DOWN": {
            "source": "alpha_occ",
            "target": "beta_vir",
            "singular_values": singular_values,
            "weights": np.abs(singular_values) ** 2,
            "block_weight": block_weight,
            "holes": holes,
            "particles": particles,
        }
    }


def write_cubes(method, block_name, block, state_outdir, state_prefix):
    state_outdir.mkdir(parents=True, exist_ok=True)
    mo_coeff = np.asarray(method.ctx.mo_coeff)
    nmo_a = mo_coeff[0].shape[1]
    files = []

    for pair, (hole, particle) in enumerate(zip(block["holes"].T, block["particles"].T), 1):
        for role, vector in (("hole", hole), ("particle", particle)):
            pieces = (
                ("alpha", mo_coeff[0], vector[:nmo_a]),
                ("beta", mo_coeff[1], vector[nmo_a:]),
            )
            for spin, mo, coeff in pieces:
                if np.linalg.norm(coeff) <= component_tol:
                    continue
                outfile = state_outdir / (
                    f"{state_prefix}_{block_name}_svd{pair}_{role}_{spin}.cube"
                )
                cubegen.orbital(method.mf.mol, str(outfile), mo @ coeff, resolution=resolution)
                files.append(str(outfile))
    return files


def main():
    set_backend("cpu")
    kind = method_kind.lower()
    if kind not in ("usf_up", "usf_down"):
        raise ValueError("method_kind must be 'usf_up' or 'usf_down'")

    chk_path = find_existing_path(chk, (Path.cwd(), SCRIPT_DIR, ROOT))
    results_path = find_existing_path(
        results_file, (Path.cwd(), chk_path.parent, SCRIPT_DIR, ROOT)
    )
    if not chk_path.exists():
        raise FileNotFoundError(chk_path)
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    vectors, energies_ha = load_saved_results(results_path)
    mf = load_uks(chk_path)
    if kind == "usf_up":
        td = SF_TDA_up(mf, method=sf_method, davidson=True, davidson_backend="cpu")
    else:
        td = XSF_TDA_down(
            mf, method=sf_method, SA=0, davidson=True, davidson_backend="cpu"
        )
        td.re = False

    td.v = vectors
    td.e = energies_ha
    td.nstates = energies_ha.size
    td.converged = np.ones(energies_ha.size, dtype=bool)

    print("method:", kind)
    print("reference: molecular UKS")
    print("chk path:", chk_path.resolve())
    print("results path:", results_path.resolve())
    print("energies / eV:", energies_ha * ha2eV)
    print("vectors shape:", vectors.shape)

    for state in states:
        if state < 0 or state >= td.nstates:
            raise IndexError(f"state {state} out of range for {td.nstates} saved states")
        blocks = (
            usf_down_block_nto(td, state=state, nroots=top_ntos)
            if kind == "usf_down"
            else td.block_nto(state=state, nroots=top_ntos)
        )
        block_name, block = next(iter(blocks.items()))
        state_outdir = Path(outdir) / f"{kind}_state{state}"
        state_prefix = f"{prefix}_state{state}"

        print()
        print(f"State {state}: {block['source']} -> {block['target']}")
        for pair, weight in enumerate(block["weights"], 1):
            fraction = weight / block["block_weight"] if block["block_weight"] > 0 else 0.0
            print(f"  #{pair}: weight={weight:.12g} fraction={fraction:.6f}")

        print("Cube files:")
        for filename in write_cubes(td, block_name, block, state_outdir, state_prefix):
            print(" ", filename)


if __name__ == "__main__":
    main()
