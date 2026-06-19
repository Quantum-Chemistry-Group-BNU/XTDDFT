#!/usr/bin/env python
"""从 PBC chk 文件和保存好的 XTDA/XSF-TDA 本征矢分析态的点群不可约表示。

这个脚本是离线后处理脚本：不会重新跑 SCF，也不会重新求解 TDA。
它只做三件事：

1. 读取已有的 Gamma 点 PBC SCF chk 文件；
2. 读取已有的 XTDA 或 XSF-TDA 结果 npz 文件；
3. 重建方法对象，并调用 analyze_excited_state_symmetry 输出不可约表示权重。

注意：这里做的是“有限超胞点群分析”，不是完整空间群或非 Gamma k 点 little group 分析。
"""

import os
import sys
from pathlib import Path


# ===== 基础运行环境 =====
# 这些线程数会影响 PySCF / NumPy / BLAS 的并行行为。
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")


# 当前脚本在 experiment/pbc 下，ROOT 是仓库根目录 XTDDFT_dev。
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
from XTDDFT_dev.XTDDFT.symmetry import analyze_excited_state_symmetry
from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT_dev.XTDDFT.xtda import XTDA


lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))


# =============================================================================
# User parameters
# =============================================================================

# method_kind:
#   "auto": 尝试从 npz 的 class_name 或文件名判断；
#   "xsf":  结果来自 XSF_TDA_down；
#   "xtda": 结果来自 XTDA。
method_kind = "auto"

# SCF chk 文件和保存好的激发态结果 npz。
chk = "roks_b3lyp_ccpVDZ.chk"
results_file = "xtda_spin_conserving_davidson_nstates8_results.npz"

# 重建 PySCF mean-field 对象时需要的泛函设置。
# 这里不会重新做 SCF，只是给 mf.xc 和 response 对象使用。
xc = "b3lyp"
use_density_fit = True

# XSF_TDA_down 专用参数。如果 npz 里保存了 method/SA，会优先使用 npz 里的值。
xsf_method = 1
SA = 3

# XTDA 专用参数。如果 npz 里保存了 method，会优先使用 npz 里的值。
xtda_method = 0
so2st = False
dense_batch_size = 64

# 对称性分析参数。
# point_group = None 时由 libmsym 自动识别有限超胞点群。
# 如果结构有轻微数值噪声导致自动识别降群，可以手动指定，例如 "C3v"、"D3d"。
point_group = None
# 读取 PySCF chk 时，这里的长度单位跟 cell.lattice_vectors()/atom_coords() 一致，通常是 Bohr。
symmetry_tol = 1.0e-3
energy_tol = 1.0e-5
grid_level = 4
grid_block_size = 20000
projection_backend = "spglib_ao_permutation"
analysis_mode = "active"
amplitude_weight_cutoff = 1.0e-6
cumulative_weight_cutoff = 0.995
min_analyzed_weight = 0.95
reference_mode = "open_shell"

# 如果想把报告保存成文件，填入文件名；否则设为 None，只打印到屏幕。
output_file = None


# =============================================================================
# 1. 选择 CPU 后端做离线后处理
# =============================================================================

set_backend("cpu")


# =============================================================================
# 2. 找到 chk 文件
# =============================================================================

chk_path = Path(chk).expanduser()
if not chk_path.is_absolute():
    # 依次在当前目录、脚本目录、仓库根目录中寻找。
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


# =============================================================================
# 3. 找到保存激发态本征矢的 npz 文件
# =============================================================================

results_path = Path(results_file).expanduser()
if not results_path.is_absolute():
    for base in (Path.cwd(), chk_path.parent, SCRIPT_DIR, ROOT):
        candidate = base / results_path
        if candidate.exists():
            results_path = candidate
            break

if not results_path.exists():
    raise FileNotFoundError(
        f"results file {results_file!r} was not found in cwd={Path.cwd()}, "
        f"chk_dir={chk_path.parent}, script_dir={SCRIPT_DIR}, or repo_root={ROOT}."
    )


# =============================================================================
# 4. 读取 npz 里的激发能和本征矢
# =============================================================================

data = np.load(results_path)

if "vectors" not in data:
    raise KeyError(f"{results_path} does not contain saved key 'vectors'")

vectors = np.asarray(data["vectors"])
if vectors.ndim != 2:
    raise ValueError(f"vectors must be 2D; got shape={vectors.shape}")

if "e_ha" in data:
    energies_ha = np.asarray(data["e_ha"], dtype=float).reshape(-1)
elif "e_ev" in data:
    energies_ha = np.asarray(data["e_ev"], dtype=float).reshape(-1) / ha2eV
else:
    raise KeyError(f"{results_path} must contain either 'e_ha' or 'e_ev'")

if vectors.shape[1] != energies_ha.size:
    raise ValueError(
        "vectors.shape[1] must match the number of energies. "
        f"got vectors.shape={vectors.shape}, energies.shape={energies_ha.shape}"
    )


# =============================================================================
# 5. 判断结果来自 XSF_TDA_down 还是 XTDA
# =============================================================================

kind = method_kind.lower()

if kind == "auto":
    kind = None

    if "class_name" in data:
        class_name = str(np.asarray(data["class_name"]).item()).lower()
        if "xsf" in class_name:
            kind = "xsf"
        elif "xtda" in class_name:
            kind = "xtda"

    if kind is None:
        filename = results_path.name.lower()
        if "xsf" in filename:
            kind = "xsf"
        elif "xtda" in filename or "x_tda" in filename:
            kind = "xtda"

    if kind is None:
        raise ValueError(
            "Could not infer method_kind from result metadata or filename. "
            "Set method_kind = 'xsf' or 'xtda' in the user parameters."
        )

if kind not in ("xsf", "xtda"):
    raise ValueError("method_kind must be 'auto', 'xsf', or 'xtda'.")


# =============================================================================
# 6. 从 chk 文件重建 PBC ROKS mean-field 对象
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

# XTDDFT 目前只支持 Gamma 点 PBC 路径。这里显式标记 density fitting 为 Gamma。
with_df = getattr(mf, "with_df", None)
if with_df is not None and hasattr(with_df, "is_gamma_point"):
    with_df.is_gamma_point = True


# =============================================================================
# 7. 重建 XSF_TDA_down 或 XTDA 对象，并把保存的本征矢塞回去
# =============================================================================

if kind == "xsf":
    # 如果 npz 里保存了 method/SA，优先使用保存值；否则使用用户参数。
    if "method" in data:
        saved_method = int(np.asarray(data["method"]).item())
    else:
        saved_method = xsf_method

    if "SA" in data:
        saved_sa = int(np.asarray(data["SA"]).item())
    else:
        saved_sa = SA

    td_obj = XSF_TDA_down(
        mf,
        method=saved_method,
        SA=saved_sa,
        davidson=True,
        davidson_backend="cpu",
    )

    # XSF restricted 路径可能删除了一个 OO trace 分量；保存结果里有 remove 时必须恢复同样设置。
    if "remove" in data:
        td_obj.re = bool(np.asarray(data["remove"]).item())
    else:
        td_obj.re = not td_obj.type_u

    if td_obj.re:
        td_obj.vects = td_obj.get_vect()

    if "fglobal" in data:
        td_obj.fglobal = float(np.asarray(data["fglobal"]).item())

elif kind == "xtda":
    if "method" in data:
        saved_method = int(np.asarray(data["method"]).item())
    else:
        saved_method = xtda_method

    td_obj = XTDA(
        mf,
        method=saved_method,
        davidson=True,
        davidson_backend="cpu",
        so2st=so2st,
        dense_batch_size=dense_batch_size,
    )


# 这些属性是对称性分析真正需要的结果数据。
td_obj.v = vectors
td_obj.e = energies_ha
td_obj.nstates = energies_ha.size
td_obj.converged = np.ones(energies_ha.size, dtype=bool)


# =============================================================================
# 8. 打印输入信息，方便确认文件和参数没有选错
# =============================================================================

print("backend:", backend_info())
print("chk path:", chk_path.resolve())
print("results path:", results_path.resolve())
print("method kind:", kind)
print("finite-supercell point-group analysis: yes")
print("natm:", cell.natm)
print("nelectron:", cell.nelectron)
print("spin:", cell.spin)
print("mesh:", cell.mesh)
print("e_tot:", mf.e_tot)
print("mo_coeff shape:", np.asarray(mf.mo_coeff).shape)
print("mo_occ shape:", np.asarray(mf.mo_occ).shape)
print("energies / eV:", energies_ha * ha2eV)
print("vectors shape:", vectors.shape)
print("point_group override:", point_group)
print("symmetry_tol:", symmetry_tol)
print("energy_tol:", energy_tol)
print("grid_level:", grid_level)
print("grid_block_size:", grid_block_size)
print("projection_backend:", projection_backend)
print("analysis_mode:", analysis_mode)
print("amplitude_weight_cutoff:", amplitude_weight_cutoff)
print("cumulative_weight_cutoff:", cumulative_weight_cutoff)
print("min_analyzed_weight:", min_analyzed_weight)
print("reference_mode:", reference_mode)
print("=" * 80)


# =============================================================================
# 9. 真正执行对称性分析
# =============================================================================

report = analyze_excited_state_symmetry(
    td_obj,
    energy_tol=energy_tol,
    symmetry_tol=symmetry_tol,
    point_group=point_group,
    grid_level=grid_level,
    grid_block_size=grid_block_size,
    projection_backend=projection_backend,
    analysis_mode=analysis_mode,
    reference_mode=reference_mode,
    amplitude_weight_cutoff=amplitude_weight_cutoff,
    cumulative_weight_cutoff=cumulative_weight_cutoff,
    min_analyzed_weight=min_analyzed_weight,
)

text = report.format_text()
print(text)


# =============================================================================
# 10. 如果需要，把报告写入文本文件
# =============================================================================

if output_file is not None:
    out = Path(output_file).expanduser()
    out.write_text(text + "\n", encoding="utf-8")
    print("=" * 80)
    print("Wrote report:", out.resolve())
