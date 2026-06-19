#!/usr/bin/env python
"""GPU XSF-TDA-down example using a chk file as the SCF initial guess.

这个脚本适合已有一个 PySCF/GPU4PySCF chk 文件、想用它作为下一次 GPU ROKS
自洽场初猜的情况。流程是：

1. 从 chk 文件读取 molecule；
2. 用同一个 chk 文件通过 mf.from_chk() 生成初猜密度矩阵 dm0；
3. 用 GPU4PySCF 重新运行 ROKS SCF；
4. 基于新的 GPU ROKS reference 运行 XSF-TDA-down；
5. 保存激发能和本征矢。

它不是直接把 chk 里的 MO 当作已经收敛的 reference 使用；SCF 仍会重新 kernel。
"""

import os
import sys
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
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
from gpu4pyscf import dft as gpubasedft
from pyscf import lib
from pyscf.scf import chkfile as mol_chkfile

from XTDDFT_dev.utils.backend import asnumpy, backend_info, set_backend

set_backend("gpu")

from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down


cp.cuda.set_allocator(None)
lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))
print("backend:", backend_info())
print("CUDA device count:", cp.cuda.runtime.getDeviceCount())


def reference_spin_square(mf, mol):
    """Return reference <S^2>, multiplicity, and how the value was obtained."""

    if hasattr(mf, "spin_square"):
        s2_ref, mult_ref = mf.spin_square()
        return float(s2_ref), float(mult_ref), "mf.spin_square()"

    s_ref = float(mol.spin) / 2.0
    s2_ref = s_ref * (s_ref + 1.0)
    mult_ref = float(mol.spin + 1)
    spin_source = "ideal value from mol.spin"
    return s2_ref, mult_ref, spin_source


# ===== User parameters =====
guess_chk = "xsf_tda_down_gpu_roks_ref.chk"
xc = "CAM-B3LYP"

nstates = 8
method = 1  # 1 = multicollinear XSF kernel
SA = 3
collinear_samples = 20

use_density_fit = True
max_memory = 4000
grid_level = 3
output_file = "xsf_tda_down_gpu_mcol_from_chk_results.npz"
# ===========================


# 1. 从 chk 文件读取 molecule。
# chk 文件必须和当前计算想用的体系一致；如果你换了基组、坐标或电荷，
# 旧 chk 只能作为参考，不能直接作为可靠初猜。
chk_path = Path(guess_chk).expanduser()
if not chk_path.is_absolute():
    for base in (Path.cwd(), SCRIPT_DIR, ROOT):
        candidate = base / chk_path
        if candidate.exists():
            chk_path = candidate
            break
if not chk_path.exists():
    raise FileNotFoundError(
        f"chk file {guess_chk!r} was not found in cwd={Path.cwd()}, "
        f"script_dir={SCRIPT_DIR}, or repo_root={ROOT}."
    )

mol, _ = mol_chkfile.load_scf(str(chk_path))
print("initial guess chk:", chk_path.resolve())


# 2. 构造 GPU ROKS 对象，并从 chk 文件生成初猜密度矩阵。
mf = gpubasedft.ROKS(mol, xc=xc)
if use_density_fit:
    mf = mf.density_fit()
mf.max_memory = max_memory
mf.grids.level = grid_level
mf.chkfile = "xsf_tda_down_gpu_roks_ref_from_chk.chk"

dm0 = mf.from_chk(str(chk_path))
mf.kernel(dm0=dm0)
if not mf.converged:
    raise RuntimeError("GPU ROKS reference did not converge from chk initial guess.")

mf.mo_energy = cp.asarray(mf.mo_energy)
mf.mo_coeff = cp.asarray(mf.mo_coeff)
mf.mo_occ = cp.asarray(mf.mo_occ)

s2_ref, mult_ref, spin_source = reference_spin_square(mf, mol)
print("\nReference state")
print("------------------------------------------------------------")
print("E_ref   =", mf.e_tot)
print("<S^2>   =", s2_ref)
print("2S+1    =", mult_ref)
print("source  =", spin_source)
print("chkfile =", mf.chkfile)


# 3. 运行 XSF-TDA-down。
xsf = XSF_TDA_down(
    mf,
    method=method,
    SA=SA,
    davidson=True,
    davidson_backend="cpu",
    collinear_samples=collinear_samples,
)

energies_ev, vectors = xsf.kernel(
    nstates=nstates,
    remove=None,
    foo=1.0,
    d_lda=0.3,
    fglobal=None,
    fit=True,
    save=False,
)
cp.cuda.Stream.null.synchronize()


print("\nXSF-TDA-down states")
print("------------------------------------------------------------")
print(" root        E/eV        converged")
print("------------------------------------------------------------")
converged = asnumpy(xsf.converged) if xsf.converged is not None else None
for i, energy in enumerate(np.asarray(energies_ev), start=1):
    flag = bool(converged[i - 1]) if converged is not None and i <= converged.size else None
    print(f"{i:5d}  {energy:14.8f}  {flag}")

try:
    delta_s2 = np.asarray(xsf.deltaS2())
except Exception as err:
    print("\nDelta <S^2> analysis skipped:", err)
    delta_s2 = np.asarray([])
else:
    print("\nDelta <S^2>")
    print("------------------------------------------------------------")
    for i, value in enumerate(delta_s2[: xsf.nstates], start=1):
        print(f"{i:5d}  {float(value):14.8f}")


np.savez_compressed(
    output_file,
    e_ev=np.asarray(energies_ev),
    e_ha=asnumpy(xsf.e),
    vectors=asnumpy(xsf.v),
    converged=asnumpy(xsf.converged) if xsf.converged is not None else np.asarray([]),
    delta_s2=delta_s2,
    mo_energy=asnumpy(mf.mo_energy),
    mo_occ=asnumpy(mf.mo_occ),
    mo_coeff=asnumpy(mf.mo_coeff),
    e_ref=np.asarray(mf.e_tot),
    s2_ref=np.asarray(s2_ref),
    multiplicity_ref=np.asarray(mult_ref),
    spin_square_source=np.asarray(spin_source),
    initial_guess_chk=np.asarray(str(chk_path.resolve())),
    xc=np.asarray(xc),
    method=np.asarray(method),
    SA=np.asarray(SA),
    collinear_samples=np.asarray(collinear_samples),
)

print("\nSaved results")
print("------------------------------------------------------------")
print("file    =", Path(output_file).resolve())
print("vectors =", np.asarray(vectors).shape)
print("Done.")
