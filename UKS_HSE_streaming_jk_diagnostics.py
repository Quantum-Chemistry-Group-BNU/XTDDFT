import inspect
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["CUPY_ACCELERATORS"] = "cub,cutensor"
os.environ["GPU4PYSCF_PBC_RSDF_PP_LOC_MEM_FRACTION"] = "0.40"
os.environ["GPU4PYSCF_PBC_RSDF_PP_LOC_GBLKSIZE"] = "16384"
os.environ["GPU4PYSCF_PBC_NUMINT_BLOCK_SIZE"] = "32768"
os.environ["GPU4PYSCF_PBC_NUMINT_BLOCK_MEM_FRACTION"] = "0.4"

import h5py
import numpy as np
import cupy as cp

cp.cuda.set_allocator(None)

from ase.io import read
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft
from pyscf.pbc import dft
from pyscf.pbc.tools import pyscf_ase
from pyscf.pbc.scf import chkfile
from pyscf.pbc.scf import uhf as pbc_uhf
import gpu4pyscf.pbc.df as gpudf

import XTDDFT_dev.XTDDFT.streaming_pbc_df_jk as streaming_jk
from XTDDFT_dev.XTDDFT.streaming_pbc_df_jk import install_streaming_df_jk


charge = -1
spin = 2
xc = "hse06"

cif_file = "3A2-geo.POSCAR.cif"
guess_chk = "pre_uks_ccpVDZ.chk"
grid_file = "becke_grids_ccpVDZ_level4.npz"

cache_dir = Path(__file__).resolve().parent / "df_cderi_cache_hse06_streaming"
cache_tag = "hse06_uks_ccpvdz"
streaming_aux_block_size = 1024
kpt_gamma = np.zeros(3)


def to_numpy(x):
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def array_stats(label, ref, test):
    ref_np = to_numpy(ref)
    test_np = to_numpy(test)
    diff = test_np - ref_np
    ref_norm = np.linalg.norm(ref_np.ravel())
    diff_norm = np.linalg.norm(diff.ravel())
    max_abs = np.max(np.abs(diff))
    rel = diff_norm / max(ref_norm, 1e-300)
    print(f"{label}: shape={ref_np.shape}")
    print(f"  ref_norm  = {ref_norm:.16e}")
    print(f"  diff_norm = {diff_norm:.16e}")
    print(f"  max_abs   = {max_abs:.16e}")
    print(f"  rel_norm  = {rel:.16e}")


def scalar(label, value):
    try:
        value = float(value)
    except Exception:
        value = float(to_numpy(value))
    print(f"{label} = {value:.16f}")


def build_cell():
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
    return cell


def build_mf(cell, label):
    mf = dft.UKS(cell).to_gpu()
    mf.xc = xc
    mf.exxdiv = "ewald"
    mf.with_df = gpudf.GDF(cell)
    mf.with_df.mesh = cell.mesh
    mf.with_df.is_gamma_point = True
    mf.chkfile = f"hse06_uks_ccpVDZ_{label}.chk"
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
    return mf


def print_cache_info(paths):
    print("Cache files:")
    for omega, path in paths.items():
        print(f"  omega={omega}: {path}")
        with h5py.File(path, "r") as handle:
            group = handle["pbc_gamma"]
            print(f"    attr omega={group.attrs.get('omega')}")
            print(f"    attr nao={group.attrs.get('nao')}")
            print(f"    cderi_0 shape={group['cderi_0'].shape}")


def print_module_info():
    print("Streaming helper module:")
    print(f"  file = {streaming_jk.__file__}")
    version = getattr(streaming_jk, "STREAMING_PBC_DF_JK_VERSION", "missing")
    print(f"  version = {version}")
    src = inspect.getsource(streaming_jk._prepare_cache_for_omega)
    print("  _prepare_cache_for_omega contains 'cell.omega':", "cell.omega" in src)
    print("  _prepare_cache_for_omega source:")
    for line in src.splitlines():
        print("    " + line)


def main():
    print_module_info()

    cell = build_cell()
    chk_cell, _ = chkfile.load_scf(guess_chk)
    print("Current nao =", cell.nao_nr())
    print("Chkfile nao =", chk_cell.nao_nr())
    if chk_cell.nao_nr() != cell.nao_nr():
        raise RuntimeError("chkfile AO count does not match current cell")

    dm0 = pbc_uhf.init_guess_by_chkfile(cell, guess_chk, project=False)
    dm0_cpu = to_numpy(dm0)
    dm0_gpu = cp.asarray(dm0_cpu)

    ref = build_mf(cell, "diag_ref")
    s_cpu = to_numpy(ref.get_ovlp())
    print("N_alpha from guess =", np.einsum("ij,ji->", dm0_cpu[0], s_cpu).real)
    print("N_beta  from guess =", np.einsum("ij,ji->", dm0_cpu[1], s_cpu).real)
    print("Initial cell.omega =", getattr(cell, "omega", None))
    omega, lr_factor, sr_factor = ref._numint.rsh_and_hybrid_coeff(ref.xc)
    print(f"rsh_and_hybrid_coeff: omega={omega}, lr_factor={lr_factor}, sr_factor={sr_factor}")

    print("\n=== Reference GPU4PySCF full-memory path ===")
    vj_ref = ref.get_j(cell, dm0_gpu, hermi=1, kpt=kpt_gamma, kpts_band=None)
    print("After ref J cell.omega =", getattr(cell, "omega", None))
    vk_sr_ref_noexx = ref.with_df.get_jk(
        dm0_gpu, 1, kpt_gamma, None, with_j=False, with_k=True,
        omega=-omega, exxdiv=None,
    )[1]
    print("After ref K_SR no exxdiv cell.omega =", getattr(cell, "omega", None))
    vk_sr_ref_ewald = ref.get_k(cell, dm0_gpu, hermi=1, kpt=kpt_gamma, kpts_band=None, omega=-omega)
    print("After ref K_SR ewald cell.omega =", getattr(cell, "omega", None))
    veff_ref = ref.get_veff(cell, dm0_gpu)
    scalar("ref.ecoul", veff_ref.ecoul)
    scalar("ref.exc", veff_ref.exc)

    print("\n=== Streaming HDF5 path ===")
    stream = build_mf(cell, "diag_streaming")
    paths = install_streaming_df_jk(
        stream,
        cache_dir=cache_dir,
        tag=cache_tag,
        omegas=(0.0, -0.11),
        aux_block_size=streaming_aux_block_size,
        build_cache=True,
        use_gpu=True,
    )
    print_cache_info(paths)
    print("After install cell.omega =", getattr(cell, "omega", None))

    vj_stream = stream.get_j(cell, dm0_gpu, hermi=1, kpt=kpt_gamma, kpts_band=None)
    print("After streaming J cell.omega =", getattr(cell, "omega", None))
    vk_sr_stream_noexx = stream.with_df.get_jk(
        dm0_gpu, 1, kpt_gamma, None, with_j=False, with_k=True,
        omega=-omega, exxdiv=None,
    )[1]
    print("After streaming K_SR no exxdiv cell.omega =", getattr(cell, "omega", None))
    vk_sr_stream_ewald = stream.get_k(cell, dm0_gpu, hermi=1, kpt=kpt_gamma, kpts_band=None, omega=-omega)
    print("After streaming K_SR ewald cell.omega =", getattr(cell, "omega", None))
    veff_stream = stream.get_veff(cell, dm0_gpu)
    scalar("stream.ecoul", veff_stream.ecoul)
    scalar("stream.exc", veff_stream.exc)

    print("\n=== Matrix comparisons: streaming - reference ===")
    array_stats("J omega=0", vj_ref, vj_stream)
    array_stats("K_SR omega=-0.11 exxdiv=None", vk_sr_ref_noexx, vk_sr_stream_noexx)
    array_stats("K_SR omega=-0.11 exxdiv=ewald", vk_sr_ref_ewald, vk_sr_stream_ewald)
    array_stats("veff", veff_ref, veff_stream)
    array_stats("veff.vj", veff_ref.vj, veff_stream.vj)
    array_stats("veff.vk", veff_ref.vk, veff_stream.vk)

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
