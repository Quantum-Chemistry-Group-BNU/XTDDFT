#!/usr/bin/env python
"""CPU 上运行 XSF-TDA-down 的最小示例。

这个脚本演示一个完整但尽量简洁的流程：

1. 用 PySCF 在 CPU 上做一个开壳层 ROKS 基态参考态；
2. 基于这个参考态运行 XSF-TDA-down；
3. 使用 method=1，也就是 multicollinear XC kernel；
4. 把激发能和本征矢保存到 npz 文件，方便后续分析或画 NTO。
"""

import os
import sys
from pathlib import Path


# 这些环境变量控制 CPU 并行线程数。
# PySCF、NumPy、MKL/OpenBLAS 都可能开线程；初学时建议先设成一个明确值，
# 否则不同机器上可能因为线程数不同而速度差别很大。
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")


import numpy as np
from pyscf import dft, gto, lib

from XTDDFT_dev.utils.backend import backend_info, set_backend

# 关键设置：强制 XTDDFT_dev 使用 CPU/NumPy 后端。
# 如果不写这一句，utils.backend 的默认模式是 auto；
# 在有 CuPy/CUDA 的服务器上，auto 会自动切到 GPU。
set_backend("cpu")

from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down


# 同步 PySCF 的线程数。这里读取上面设置的 OMP_NUM_THREADS。
lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))
print("backend:", backend_info())


# ===== User parameters =====
# 下面这些是最常改的参数。
# xc/basis 控制 DFT 泛函和 AO 基组；charge/spin/unit 定义分子电荷、自旋和坐标单位。
xc = "PBE0"
basis = "6-31G"
charge = 0
spin = 2
unit = "Angstrom"
cycle = 200
nstates = 6
method = 1  # 1 = multicollinear XSF kernel
SA = 3
collinear_samples = 20

# density fitting 可以显著加速 Coulomb/exchange 相关积分，适合作为默认设置。
use_density_fit = True
max_memory = 4000
grid_level = 3
output_file = "xsf_tda_down_cpu_mcol_results.npz"
# ===========================


# 1. 构建分子。
# PySCF 中 spin 不是 multiplicity，而是 N_alpha - N_beta = 2S。
# 这里 spin=2 对应 triplet reference，即 S=1、multiplicity=3。
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


# 2. 运行 CPU ROKS reference。
# XSF-TDA-down 通常从高自旋开壳层参考态出发，这里用 ROKS 描述 triplet reference。
mf = dft.ROKS(mol)
mf.xc = "B3LYP"
if use_density_fit:
    mf = mf.density_fit()
mf.max_memory = max_memory
mf.grids.level = grid_level
mf.init_guess = 'atom'
mf.max_cycle = cycle
mf.chkfile = "xsf_tda_down_cpu_roks_ref.chk"
mf.conv_tol = 1e-10        # 能量收敛阈值
mf.conv_tol_grad = 1e-6    # orbital gradient / density 相关的收敛阈值
# kernel() 会真正开始 SCF 自洽计算。
mf.kernel()
if not mf.converged:
    raise RuntimeError("ROKS reference did not converge.")

# spin_square() 用来检查参考态自旋。
# 对理想 triplet，<S^2> 约等于 S(S+1)=2。
s2_ref, mult_ref = mf.spin_square()
print("\nReference state")
print("------------------------------------------------------------")
print("E_ref   =", mf.e_tot)
print("<S^2>   =", s2_ref)
print("2S+1    =", mult_ref)
print("chkfile =", mf.chkfile)


# 3. 构造 XSF-TDA-down 计算对象。
# davidson=True 表示用 Davidson 迭代求最低的若干个本征值/本征矢，
# 比显式构造并对角化完整矩阵更适合稍大的体系。
# davidson_backend="cpu" 要求 Davidson 迭代本身也在 CPU 上执行。
xsf = XSF_TDA_down(
    mf,
    method=method,
    SA=SA,
    davidson=True,
    davidson_backend="cpu",
    collinear_samples=collinear_samples,
)

# 4. 开始 XSF-TDA 计算。
# 返回值 energies_ev 是 eV 单位的激发能；vectors 是对应的本征矢。
# save=False 是因为下面会用 np.savez_compressed 手动保存更多元数据。
energies_ev, vectors = xsf.kernel(
    nstates=nstates,
    remove=None,
    foo=1.0,
    d_lda=0.3,
    fglobal=None,
    fit=True,
    save=False,
)

# 5. 打印每个 root 的激发能和 Davidson 是否收敛。
print("\nXSF-TDA-down states")
print("------------------------------------------------------------")
print(" root        E/eV        converged")
print("------------------------------------------------------------")
converged = np.asarray(xsf.converged) if xsf.converged is not None else None
for i, energy in enumerate(np.asarray(energies_ev), start=1):
    flag = bool(converged[i - 1]) if converged is not None and i <= converged.size else None
    print(f"{i:5d}  {energy:14.8f}  {flag}")

# deltaS2() 给出激发态相对参考态的自旋信息，用于辅助判断自旋性质。
# 这个分析依赖当前对象中的轨道/矢量结构；如果用户改成别的 reference 导致不适用，
# 这里跳过分析，但仍然保存本征矢。
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

# 6. 保存结果。
# 最重要的是：
#   e_ev/e_ha: 激发能，分别用 eV 和 Hartree 表示；
#   vectors: XSF-TDA 本征矢，后续做 NTO、TDM 或其他分析时会用到；
#   mo_energy/mo_occ/mo_coeff: SCF 轨道信息，方便复现实验或离线分析。
np.savez_compressed(
    output_file,
    e_ev=np.asarray(energies_ev),
    e_ha=np.asarray(xsf.e),
    vectors=np.asarray(xsf.v),
    converged=np.asarray(xsf.converged) if xsf.converged is not None else np.asarray([]),
    delta_s2=delta_s2,
    mo_energy=np.asarray(mf.mo_energy),
    mo_occ=np.asarray(mf.mo_occ),
    mo_coeff=np.asarray(mf.mo_coeff),
    e_ref=np.asarray(mf.e_tot),
    s2_ref=np.asarray(s2_ref),
    multiplicity_ref=np.asarray(mult_ref),
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
