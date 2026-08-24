#!/usr/bin/env python
"""在 ROKS 参考态上运行 SOC 态相互作用（SOC-SI）的最小示例。

流程：
1. PySCF sfx2c + ROKS 做开壳层基态；
2. XSF-TDA-down (|S->)、XTDA (|So>)、SF-TDA-up (|S+>) 三个自旋块；
3. sf-X2C + SOMF 构造自旋轨道耦合 Vso；
4. SI_driver 做 SOC 态相互作用，输出 SOC 能级。
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import numpy as np
from pyscf import dft, gto, lib, scf

from XTDDFT_dev.utils.backend import set_backend

set_backend("cpu")

from XTDDFT_dev.XTDDFT.soc.soc_si import SOCSI

lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))

# ===== User parameters =====
xc = "bhandhlyp"
basis = "cc-pVDZ"
atom = "As 0 0 0"
charge = 0
spin = 3  # 2S = 3，即 S=3/2 开壳层
nstates = (20, 20, 20)  # (|S->, |So>, |S+>) 各取 20 个态
backend = "auto"  # "auto"/"cpu"/"gpu"：SOC 计算（Vso/SI）使用的后端
output_file = "soc_si_results.npz"
# ===========================

mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=3)
mol.build()

mf = scf.sfx2c(dft.ROKS(mol))
mf.xc = xc
mf.max_cycle = 200
mf.run()
assert mf.converged

mysoc = SOCSI(mf, nstates=nstates, backend=backend)
eso, vso = mysoc.kernel(printnum=20)

np.savez_compressed(
    output_file,
    eso=eso,
    vso=vso,
    hso=mysoc.mysi.hso,
    heff=mysoc.mysi.heff,
)
print("file     =", os.path.abspath(output_file))
print("E_so(eV) =", np.asarray(eso) * 27.2113834)
