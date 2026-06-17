#!/usr/bin/env python
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import numpy as np
from pyscf.scf import chkfile as mol_chkfile
from pyscf.tools import cubegen

from XTDDFT_dev.utils.unit import ha2eV


# ===== Manually edit these parameters on the server =====
chk = "rks_smd_toluene_cc_pvdz.chk"
outdir = "rks_smd_toluene_cc_pvdz_homo_lumo_cubes"
prefix = "rks_smd_toluene_cc_pvdz"
resolution = 0.15
margin = 3.0
occ_tol = 1.0e-8
# ========================================================


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

outdir_path = Path(outdir).expanduser()
outdir_path.mkdir(parents=True, exist_ok=True)

mol, scf_rec = mol_chkfile.load_scf(str(chk_path))
mo_coeff = np.asarray(scf_rec["mo_coeff"])
mo_occ = np.asarray(scf_rec["mo_occ"])
mo_energy = np.asarray(scf_rec["mo_energy"])

if mo_coeff.ndim != 2:
    raise ValueError(f"Expected RKS/RHF mo_coeff to be 2D; got shape={mo_coeff.shape}")
if mo_occ.ndim != 1 or mo_energy.ndim != 1:
    raise ValueError(f"Expected 1D mo_occ/mo_energy; got {mo_occ.shape=} {mo_energy.shape=}")

occupied = np.where(mo_occ > occ_tol)[0]
virtual = np.where(mo_occ <= occ_tol)[0]
if occupied.size == 0:
    raise ValueError("No occupied orbitals found from mo_occ")
if virtual.size == 0:
    raise ValueError("No virtual orbitals found from mo_occ")

homo_idx = int(occupied[np.argmax(mo_energy[occupied])])
lumo_idx = int(virtual[np.argmin(mo_energy[virtual])])

homo_file = outdir_path / f"{prefix}_homo.cube"
lumo_file = outdir_path / f"{prefix}_lumo.cube"

cubegen.orbital(
    mol,
    str(homo_file),
    mo_coeff[:, homo_idx],
    resolution=resolution,
    margin=margin,
)
cubegen.orbital(
    mol,
    str(lumo_file),
    mo_coeff[:, lumo_idx],
    resolution=resolution,
    margin=margin,
)

print("chk path:", chk_path.resolve())
print("natm:", mol.natm)
print("nelectron:", mol.nelectron)
print("HOMO index:", homo_idx)
print("HOMO energy / eV:", mo_energy[homo_idx] * ha2eV)
print("HOMO cube:", homo_file)
print("LUMO index:", lumo_idx)
print("LUMO energy / eV:", mo_energy[lumo_idx] * ha2eV)
print("LUMO cube:", lumo_file)
