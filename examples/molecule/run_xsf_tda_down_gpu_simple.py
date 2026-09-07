#!/usr/bin/env python
"""GPU 上运行 XSF-TDA-down 的最小示例。

这个脚本和 run_xsf_tda_down_cpu_simple.py 做同一件事，但把主要数组和
PySCF reference 对象切到 GPU：

1. 用 GPU4PySCF 在 GPU 上做一个开壳层 ROKS 基态参考态；
2. 基于这个参考态运行 XSF-TDA-down；
3. 使用 method=1，也就是 multicollinear XC kernel；
4. 把激发能和本征矢保存到 npz 文件。保存前会把 CuPy 数组转回 NumPy。

注意：这个脚本需要可用的 CUDA、CuPy 和 gpu4pyscf 环境。
"""

import os
import sys
from pathlib import Path


# CUDA_VISIBLE_DEVICES 控制使用哪张 GPU。
# 下面几个线程变量仍然有用，因为部分 PySCF/GPU4PySCF 准备步骤会调用 CPU 库。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("CUPY_ACCELERATORS", "cub,cutensor")
os.environ.setdefault("GPU4PYSCF_NUMINT_BLOCK_SIZE", "32768")
os.environ.setdefault("GPU4PYSCF_NUMINT_BLOCK_MEM_FRACTION", "0.4")


# 让脚本无论从哪个工作目录运行，都能 import 到本仓库的 XTDDFT_dev 包。
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))


import cupy as cp
import numpy as np
from gpu4pyscf import dft as gpubasedft
from pyscf import gto, lib

from XTDDFT_dev.utils.backend import asnumpy, backend_info, set_backend

# 关键设置：强制 XTDDFT_dev 使用 GPU/CuPy 后端。
# 这句必须在导入 XSF_TDA_down 之前执行，否则模块级 backend 可能已经按默认值初始化。
set_backend("gpu")

from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down


# 禁用 CuPy memory pool 可以让初学者更容易从 nvidia-smi 观察真实显存变化。
# 如果追求性能，可以注释掉这一行，使用 CuPy 默认 memory pool。
cp.cuda.set_allocator(None)
lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))
print("backend:", backend_info())
print("CUDA device count:", cp.cuda.runtime.getDeviceCount())


def reference_spin_square(mf, mol):
    """Return reference <S^2>, multiplicity, and how the value was obtained."""

    if hasattr(mf, "spin_square"):
        s2_ref, mult_ref = mf.spin_square()
        return float(s2_ref), float(mult_ref), "mf.spin_square()"

    # Some GPU4PySCF wrappers, for example DFROKS, do not expose spin_square().
    # In that case use the ideal spin implied by mol.spin = N_alpha - N_beta = 2S.
    s_ref = float(mol.spin) / 2.0
    s2_ref = s_ref * (s_ref + 1.0)
    mult_ref = float(mol.spin + 1)
    spin_source = "ideal value from mol.spin"
    return s2_ref, mult_ref, spin_source


# ===== User parameters =====
xc = "PBE0"
basis = "6-31G"
charge = 0
spin = 2
cycle = 200
unit = "Angstrom"

nstates = 6
method = 1  # 1 = multicollinear XSF kernel
SA = 3
collinear_samples = 20

use_density_fit = True
max_memory = 4000
grid_level = 4
output_file = "xsf_tda_down_gpu_mcol_results.npz"
# ===========================


# 1. 构建分子。这里和 CPU 示例使用同一个小分子，方便比较 CPU/GPU 结果。
mol = gto.M(
    atom="""
    H 0.000000  0.934473 -0.588078
    H 0.000000 -0.934473 -0.588078
    C 0.000000  0.000000  0.000000
    O 0.000000  0.000000  1.221104
    """,
    basis=basis,
    charge=charge,
    spin=spin,
    unit=unit,
    verbose=5,
)


# 2. 运行 GPU ROKS reference。
# gpubasedft.ROKS 会使用 GPU4PySCF 的 DFT 实现。
mf = gpubasedft.ROKS(mol, xc=xc)
if use_density_fit:
    mf = mf.density_fit()
mf.max_memory = max_memory
mf.grids.level = grid_level
mf.max_cycle = cycle
mf.chkfile = "xsf_tda_down_gpu_roks_ref.chk"
mf.conv_tol = 1e-10        # 能量收敛阈值
mf.conv_tol_grad = 1e-6    # orbital gradient / density 相关的收敛阈值


mf.kernel()
if not mf.converged:
    raise RuntimeError("GPU ROKS reference did not converge.")

# 确保后续 XSF-TDA 看到的是 CuPy 数组。
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


# 3. 构造 XSF-TDA-down 计算对象。
# davidson_backend="cpu" 表示 Davidson 子空间迭代在 CPU 上做，
# 但矩阵-向量响应和数组 backend 仍然走 GPU。这通常更稳，也节省 GPU 显存。
xsf = XSF_TDA_down(
    mf,
    method=method,
    SA=SA,
    davidson=True,
    davidson_backend="cpu",
    collinear_samples=collinear_samples,
)

# 4. 开始 XSF-TDA 计算。
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


# 5. 打印每个 root 的激发能和 Davidson 是否收敛。
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


# 6. 保存结果。npz 文件只能可靠保存 NumPy 数组，所以这里统一用 asnumpy() 转回 CPU。
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
    xc=np.asarray(xc),
    basis=np.asarray(basis),
    method=np.asarray(method),
    SA=np.asarray(SA),
    collinear_samples=np.asarray(collinear_samples),
)

print("\nSaved results")
print("------------------------------------------------------------")
print("file    =", Path(output_file).resolve())
print("vectors =", np.asarray(vectors).shape)
print("Done.")
