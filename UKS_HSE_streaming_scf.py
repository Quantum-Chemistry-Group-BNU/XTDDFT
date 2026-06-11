import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["CUPY_ACCELERATORS"] = "cub,cutensor"
os.environ["GPU4PYSCF_PBC_RSDF_PP_LOC_MEM_FRACTION"] = "0.40"
os.environ["GPU4PYSCF_PBC_RSDF_PP_LOC_GBLKSIZE"] = "16384"
os.environ["GPU4PYSCF_PBC_NUMINT_BLOCK_SIZE"] = "32768"
os.environ["GPU4PYSCF_PBC_NUMINT_BLOCK_MEM_FRACTION"] = "0.4"

import numpy as np
import cupy as cp

cp.cuda.set_allocator(None)

from ase.io import read
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft
from pyscf.pbc import dft
from pyscf.pbc.tools import pyscf_ase
from pyscf.pbc.scf import chkfile
from pyscf.scf import hf as mol_hf
from pyscf.pbc.scf import uhf as pbc_uhf
import gpu4pyscf.pbc.df as gpudf

from XTDDFT_dev.XTDDFT.streaming_pbc_df_jk import install_streaming_df_jk


# ===== User parameters =====
charge = -1
spin = 2
xc = "hse06"

cif_file = "3A2-geo.POSCAR.cif"
guess_chk = "pre_uks_ccpVDZ.chk"
out_chk = "hse06_uks_ccpVDZ_streaming.chk"
grid_file = "becke_grids_ccpVDZ_level4.npz"

cache_dir = Path(__file__).resolve().parent / "df_cderi_cache_hse06_streaming"
cache_tag = "hse06_uks_ccpvdz"
streaming_aux_block_size = 1


def _finalize_without_mp(mf):
    mol_hf.SCF._finalize(mf)
    return mf


def to_numpy(x):
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


# ===== Build cell =====
atoms = read(cif_file, format="cif")

cell = pbcgto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(atoms)
cell.a = np.asarray(atoms.cell)
cell.dimension = 3
cell.basis = "cc-pVDZ"
cell.charge = charge
cell.spin = spin
cell.unit = "Angstrom"
cell.precision = 1e-6
cell.mesh = [1000, 1000, 1000]
cell.max_memory = 16000
cell.verbose = 5
cell.build()


# ===== Check chkfile/cell compatibility =====
chk_cell, _ = chkfile.load_scf(guess_chk)

print("Current nao =", cell.nao_nr())
print("Chkfile nao =", chk_cell.nao_nr())

if chk_cell.nao_nr() != cell.nao_nr():
    raise RuntimeError(
        "chkfile AO count does not match the current cell. "
        "The basis/structure may differ, so project=False cannot be used."
    )


# ===== Build HSE06 UKS-GPU object =====
mf = dft.UKS(cell).to_gpu()
mf.xc = xc
mf.exxdiv = "ewald"

mf.with_df = gpudf.GDF(cell)
mf.with_df.mesh = cell.mesh
mf.with_df.is_gamma_point = True

mf.chkfile = out_chk
mf.level_shift = 0.5
mf.max_cycle = 200
mf.conv_tol = 1e-10
mf.conv_tol_grad = 1e-6
mf.diis_space = 8

data = np.load(grid_file)
mf.grids = pbcdft.BeckeGrids(cell)
mf.grids.level = 4
mf.grids.coords = cp.array(data["coords"])
mf.grids.weights = cp.array(data["weights"])

# Avoid finalize-stage CPU multipole analysis issues.
mf._finalize = lambda: _finalize_without_mp(mf)


# ===== Install HDF5-backed streaming J/K before SCF =====
streaming_cache_paths = install_streaming_df_jk(
    mf,
    cache_dir=cache_dir,
    tag=cache_tag,
    omegas=(0.0, -0.11),
    aux_block_size=streaming_aux_block_size,
    build_cache=True,
    use_gpu=True,
)

print("Streaming DF cache files:")
for omega, path in streaming_cache_paths.items():
    print(f"  omega={omega}: {path}")
print("SCF exxdiv =", mf.exxdiv)


# ===== Initial DM from compatible cc-pVDZ chkfile =====
dm0 = pbc_uhf.init_guess_by_chkfile(
    cell,
    guess_chk,
    project=False,
)

dm0_cpu = to_numpy(dm0)
s_cpu = to_numpy(mf.get_ovlp())

print("N_alpha from guess =", np.einsum("ij,ji->", dm0_cpu[0], s_cpu).real)
print("N_beta  from guess =", np.einsum("ij,ji->", dm0_cpu[1], s_cpu).real)

dm0_gpu = cp.asarray(dm0_cpu)


# ===== Full UKS/HSE06 SCF with streaming DF J/K =====
energy = mf.kernel(dm0=dm0_gpu)

print("SCF converged:", mf.converged)
print("Final HSE06 energy:", float(energy))
print("HSE06 chkfile written to:", mf.chkfile)
