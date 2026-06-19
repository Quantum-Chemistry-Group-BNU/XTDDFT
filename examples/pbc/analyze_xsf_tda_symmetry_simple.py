#!/usr/bin/env python
"""最简单的 PBC XSF-TDA 对称性后处理脚本。

默认读取当前目录下三个文件：

- roks_from_uks_ccpVDZ.chk
- XSF.npz
- becke_grids_ccpvdz_level4.npz

这个脚本只针对 XSF_TDA_down，不做 XTDA/UTDA/SF_TDA_up 的自动判断。
分析结果是有限超胞点群下的不可约表示权重，不是完整空间群分析。
"""

import os
import sys
from pathlib import Path


# 线程数设置。按机器情况可以自己改。
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")


# 让脚本可以从仓库外/仓库内运行时都 import 到 XTDDFT_dev。
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))


import numpy as np
from pyscf import lib
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.scf import chkfile as pbc_chkfile

from XTDDFT_dev.utils.backend import backend_info, set_backend
from XTDDFT_dev.utils.unit import ha2eV
from XTDDFT_dev.utils.symmetry import analyze_excited_state_symmetry
from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down


lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))
set_backend("cpu")


# =============================================================================
# 1. 用户只需要先改这里
# =============================================================================

chk_file = "roks_from_uks_ccpVDZ.chk"
xsf_file = "XSF.npz"
grid_file = "becke_grids_ccpvdz_level4.npz"

# 这里不会重新做 SCF，但重建 mf 对象时需要给出 xc。
xc = "b3lyp"
use_density_fit = True

# 如果 XSF.npz 里没有保存 method/SA，就用这里的默认值。
# method=1 是 multicollinear，method=0 是 ALDA0。
xsf_method = 1
SA = 3

# 对称性参数。
# point_group=None 表示自动识别；如果自动识别降群，可以手动设成 "C3v"、"D3d" 等。
point_group = None
# 读取 PySCF chk 时，这里的长度单位跟 cell.lattice_vectors()/atom_coords() 一致，通常是 Bohr。
symmetry_tol = 1.0e-3

# energy_tol 控制哪些 root 被认为是近简并子空间。
energy_tol = 1.0e-5

# 每次处理多少个 grid 点。这个参数控制内存。
# 对 3000 个 AO，20000 个 grid 点大约需要几 GB 级别内存；内存紧张就改成 5000 或 10000。
grid_block_size = 20000

# 投影后端：
#   "spglib_ao_permutation": 使用 spglib 周期性操作 R,t 做原子置换 + AO shell 旋转，推荐 PBC 缺陷超胞使用。
#   "ao_permutation": 使用原子置换 + AO shell 旋转，不需要读取/积分 grid，适合严格点群几何。
#   "grid": 使用保存好的 Becke grid 做数值投影，较慢但对轻微几何误差更宽容。
projection_backend = "spglib_ao_permutation"

# active-subspace 对称性分析参数。
# 程序会从 XSF amplitude 中筛选重要组态，只对相关轨道子空间做投影。
analysis_mode = "active"
amplitude_weight_cutoff = 1.0e-6
cumulative_weight_cutoff = 0.995
min_analyzed_weight = 0.95
reference_mode = "open_shell"


# =============================================================================
# 2. 检查三个输入文件是否存在
# =============================================================================

chk_path = Path(chk_file)
xsf_path = Path(xsf_file)
grid_path = Path(grid_file)

if not chk_path.exists():
    raise FileNotFoundError(f"Cannot find SCF chk file: {chk_path.resolve()}")

if not xsf_path.exists():
    raise FileNotFoundError(f"Cannot find XSF result npz file: {xsf_path.resolve()}")

if projection_backend == "grid" and not grid_path.exists():
    raise FileNotFoundError(f"Cannot find saved grid npz file: {grid_path.resolve()}")


# =============================================================================
# 3. 读取保存好的 XSF-TDA 激发能和本征矢
# =============================================================================

xsf_data = np.load(xsf_path)

if "vectors" not in xsf_data:
    raise KeyError("XSF.npz must contain key 'vectors'.")

vectors = np.asarray(xsf_data["vectors"])
if vectors.ndim != 2:
    raise ValueError(f"vectors must be a 2D array, got shape={vectors.shape}")

if "e_ha" in xsf_data:
    energies_ha = np.asarray(xsf_data["e_ha"], dtype=float).reshape(-1)
elif "e_ev" in xsf_data:
    energies_ha = np.asarray(xsf_data["e_ev"], dtype=float).reshape(-1) / ha2eV
else:
    raise KeyError("XSF.npz must contain either 'e_ha' or 'e_ev'.")

if vectors.shape[1] != energies_ha.size:
    raise ValueError(
        "vectors.shape[1] must equal number of energies. "
        f"vectors.shape={vectors.shape}, energies.shape={energies_ha.shape}"
    )


# =============================================================================
# 4. 如果使用 grid 后端，读取已经保存好的 Becke grids
# =============================================================================

grid_coords = None
grid_weights = None

if projection_backend == "grid":
    grid_data = np.load(grid_path)

    if "coords" not in grid_data or "weights" not in grid_data:
        raise KeyError("grid npz must contain keys 'coords' and 'weights'.")

    grid_coords = np.asarray(grid_data["coords"])
    grid_weights = np.asarray(grid_data["weights"])

    if grid_coords.ndim != 2 or grid_coords.shape[1] != 3:
        raise ValueError(f"grid coords must have shape (ngrids, 3), got {grid_coords.shape}")

    if grid_weights.ndim != 1 or grid_weights.shape[0] != grid_coords.shape[0]:
        raise ValueError(
            "grid weights must be 1D and have the same length as coords. "
            f"coords={grid_coords.shape}, weights={grid_weights.shape}"
        )


# =============================================================================
# 5. 从 chk 文件重建 PBC ROKS mean-field 对象
# =============================================================================

cell, scf_rec = pbc_chkfile.load_scf(str(chk_path))

mf = pbcdft.ROKS(cell)
if use_density_fit:
    mf = mf.density_fit()

mf.xc = xc
mf.mo_energy = np.asarray(scf_rec["mo_energy"])
mf.mo_coeff = np.asarray(scf_rec["mo_coeff"])
mf.mo_occ = np.asarray(scf_rec["mo_occ"])
mf.e_tot = scf_rec.get("e_tot", None)
mf.converged = True

# 对 Gamma 点 PBC 路径做显式标记。
with_df = getattr(mf, "with_df", None)
if with_df is not None and hasattr(with_df, "is_gamma_point"):
    with_df.is_gamma_point = True


# =============================================================================
# 6. 重建 XSF_TDA_down 对象
# =============================================================================

if "method" in xsf_data:
    saved_method = int(np.asarray(xsf_data["method"]).item())
else:
    saved_method = xsf_method

if "SA" in xsf_data:
    saved_SA = int(np.asarray(xsf_data["SA"]).item())
else:
    saved_SA = SA

xsf = XSF_TDA_down(
    mf,
    method=saved_method,
    SA=saved_SA,
    davidson=True,
    davidson_backend="cpu",
)

# 对 restricted XSF_TDA_down，保存结果时可能 remove 了一个 OO trace 分量。
# 如果 npz 里保存了 remove，就必须沿用保存时的设置。
if "remove" in xsf_data:
    xsf.re = bool(np.asarray(xsf_data["remove"]).item())
else:
    xsf.re = not xsf.type_u

if xsf.re:
    xsf.vects = xsf.get_vect()

if "fglobal" in xsf_data:
    xsf.fglobal = float(np.asarray(xsf_data["fglobal"]).item())

# 把保存好的激发态结果塞回对象。
xsf.v = vectors
xsf.e = energies_ha
xsf.nstates = energies_ha.size
xsf.converged = np.ones(energies_ha.size, dtype=bool)


# =============================================================================
# 7. 打印输入信息，确认读入的是正确文件
# =============================================================================

print("backend:", backend_info())
print("chk file:", chk_path.resolve())
print("XSF file:", xsf_path.resolve())
print("grid file:", grid_path.resolve() if projection_backend == "grid" else "not used")
print("finite-supercell point-group analysis: yes")
print("cell natm:", cell.natm)
print("cell nelectron:", cell.nelectron)
print("cell spin:", cell.spin)
print("cell mesh:", cell.mesh)
print("mf e_tot:", mf.e_tot)
print("mo_coeff shape:", np.asarray(mf.mo_coeff).shape)
print("mo_occ shape:", np.asarray(mf.mo_occ).shape)
print("XSF method:", saved_method)
print("XSF SA:", saved_SA)
print("XSF remove:", xsf.re)
print("energies / eV:", energies_ha * ha2eV)
print("vectors shape:", vectors.shape)
if projection_backend == "grid":
    print("grid coords shape:", grid_coords.shape)
    print("grid weights shape:", grid_weights.shape)
print("point_group override:", point_group)
print("symmetry_tol:", symmetry_tol)
print("energy_tol:", energy_tol)
print("grid_block_size:", grid_block_size)
print("projection_backend:", projection_backend)
print("analysis_mode:", analysis_mode)
print("amplitude_weight_cutoff:", amplitude_weight_cutoff)
print("cumulative_weight_cutoff:", cumulative_weight_cutoff)
print("min_analyzed_weight:", min_analyzed_weight)
print("reference_mode:", reference_mode)
print("=" * 80)


# =============================================================================
# 8. 做基态和激发态对称性分析
# =============================================================================

report = analyze_excited_state_symmetry(
    xsf,
    energy_tol=energy_tol,
    symmetry_tol=symmetry_tol,
    point_group=point_group,
    grid_coords=grid_coords,
    grid_weights=grid_weights,
    grid_block_size=grid_block_size,
    projection_backend=projection_backend,
    analysis_mode=analysis_mode,
    reference_mode=reference_mode,
    amplitude_weight_cutoff=amplitude_weight_cutoff,
    cumulative_weight_cutoff=cumulative_weight_cutoff,
    min_analyzed_weight=min_analyzed_weight,
)

print(report.format_text())
