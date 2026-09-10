import copy
from pathlib import Path

import numpy as np
from pyscf import dft, gto, lib
from pyscf.lib import logger

from ..base import _is_gpu_mf, _is_pbc_mf
from ..grad import _reference_family
from ..xtda import XTDA
from ...utils.backend import _asnumpy
from ...utils.unit import bohr


def _state_pairs(pairs):
    pairs = np.asarray(pairs)
    if pairs.shape == (2,):
        pairs = pairs[None, :]
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or not pairs.size
        or pairs.dtype.kind not in "iu"
        or np.any(pairs < 0)
        or np.any(np.all(pairs == 0, axis=1))
    ):
        raise ValueError(
            "pairs must contain physical states >= 0 and cannot be (0, 0)"
        )
    return pairs


def signed_permutation(overlap, label, threshold=0.5):
    """由重叠矩阵确定唯一的换序和正负号。

    对足够小的位移，每个参考轨道（或参考激发态）应只与一个位移后的
    轨道（或激发态）有较大重叠。矩阵中该元素的位置给出对应关系，符号
    给出应乘的相位。若某行或某列不能唯一匹配，说明步长过大、求解根数
    不足，或存在不能用简单正负号处理的简并子空间旋转，此时停止计算比
    猜测配对更可靠。
    """
    overlap = np.asarray(overlap)
    if np.iscomplexobj(overlap):
        raise NotImplementedError(
            "Signed-permutation alignment requires real vectors"
        )
    selected = np.abs(overlap) > threshold
    if (
        overlap.ndim != 2
        or overlap.shape[0] != overlap.shape[1]
        or not np.isfinite(overlap).all()
        or not np.all(selected.sum(axis=0) == 1)
        or not np.all(selected.sum(axis=1) == 1)
    ):
        raise ValueError(
            f"{label} overlap cannot be mapped to a signed permutation. "
            "Try a smaller step or more XTDA roots; degenerate rotations "
            f"require a different alignment.\n{overlap}"
        )
    return np.where(selected, np.sign(overlap), 0.0)


def mo_overlap(reference_mf, displaced_mf):
    """计算跨构型 MO 重叠 ``<phi_p(R0)|phi_q(R)>``。

    两个构型的 AO 基函数也随核坐标移动，因此不能直接比较 MO 系数。
    这里先由 PySCF 计算跨构型 AO 重叠 ``S_AO(R0,R)``，再变换为
    ``C(R0).T @ S_AO(R0,R) @ C(R)``。
    """
    ao_overlap = gto.intor_cross(
        "int1e_ovlp", reference_mf.mol, displaced_mf.mol
    )
    return reference_mf.mo_coeff.T @ ao_overlap @ displaced_mf.mo_coeff


def align_mos(reference_mf, displaced_mf, threshold=0.5):
    """把位移构型的 MO 换序、换号，使其与参考构型逐一对应。

    对齐后第 ``p`` 列仍表示参考构型中的第 ``p`` 个轨道。随后建立的
    XTDA 组态基也就与参考构型具有相同顺序，位移前后的振幅分量才可以
    直接相减。
    """
    permutation = signed_permutation(
        mo_overlap(reference_mf, displaced_mf), "MO", threshold
    )
    order = np.argmax(np.abs(permutation), axis=1)
    phase = permutation[np.arange(len(order)), order]
    if not np.array_equal(reference_mf.mo_occ, displaced_mf.mo_occ[order]):
        raise ValueError("Displaced SCF changed the reference orbital occupations")

    aligned_mf = copy.copy(displaced_mf)
    aligned_mf.mo_coeff = displaced_mf.mo_coeff[:, order] * phase
    aligned_mf.mo_energy = displaced_mf.mo_energy[order]
    aligned_mf.mo_occ = displaced_mf.mo_occ[order]
    return aligned_mf


def align_amplitudes(reference_amplitudes, displaced_amplitudes, threshold=0.5):
    """把位移构型的 XTDA 根换序、换号到参考态顺序。

    MO 对齐后，两边振幅位于同一个组态基中，因此态重叠可写为
    ``X(R0).T @ X(R)``。返回值的第 ``J`` 列始终对应参考构型的第
    ``J`` 个激发态。
    """
    permutation = signed_permutation(
        reference_amplitudes.T @ displaced_amplitudes,
        "XTDA state",
        threshold,
    )
    return displaced_amplitudes @ permutation.T


def displaced_mol(mol, atom, axis, step):
    """Copy a molecule and move one Cartesian coordinate in angstrom."""
    coordinates = mol.atom_coords(unit="Angstrom").copy()
    coordinates[atom, axis] += step
    moved = mol.copy()
    moved.set_geom_(coordinates, unit="Angstrom")
    return moved


def _check_solution(td, nstates):
    if not td.mf.converged:
        raise RuntimeError("ROKS did not converge")
    if td.e is None or td.v is None:
        raise RuntimeError("Run td.kernel() before calculating NACs")
    if td.v.shape[1] < nstates or len(td.e) < nstates:
        raise ValueError(f"NAC requires at least {nstates} XTDA roots")
    if td.converged is not None and not np.all(_asnumpy(td.converged)):
        raise RuntimeError("XTDA did not converge")


class NAC(lib.StreamObject):
    """Finite-difference NAC driver following the gradient driver style.

    Example::

        td.kernel(nstates=4)
        nac = td.nac_method(pairs=[(0, 1), (1, 2)])
        vectors = nac.kernel()  # dict: (I, J) -> (natom, 3), bohr^-1

    State 0 is the ground state and states 1, 2, ... are XTDA excited states.
    All available reference roots are tracked, including unused buffer roots.
    ``kernel`` always calculates all atoms and does not write files.
    """

    def __init__(self, td, pairs=(0, 1), step=1e-4, align_threshold=0.5):
        supported = (
            isinstance(td, XTDA)
            and td.method == 0
            and not _is_pbc_mf(td.mf)
            and not _is_gpu_mf(td.mf)
            and _reference_family(td.mf) == "roks"
            and td.davidson_backend == "cpu"
        )
        if not supported:
            raise NotImplementedError(
                "NAC currently supports molecular CPU ROKS/XTDA only"
            )
        self.base = td
        self.mol = td.mf.mol
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.pairs = _state_pairs(pairs)
        self.step = step
        self.align_threshold = align_threshold
        self.de = None

    def _solve_displaced(self, mol, reference_amplitudes):
        reference_mf = self.base.mf
        # 每个位移点都复用参考 SCF 的泛函、收敛阈值、积分网格等设置。
        # 先深拷贝再 reset，避免新的几何和缓存反过来修改参考构型 R0。
        displaced_mf = copy.deepcopy(reference_mf).reset(mol)
        displaced_mf.chkfile = None
        displaced_mf.kernel()
        if not displaced_mf.converged:
            raise RuntimeError("Displaced ROKS did not converge")

        # 先对齐 MO，再用对齐后的 MO 建立 XTDA 组态基。
        aligned_mf = align_mos(
            reference_mf, displaced_mf, self.align_threshold
        )

        # 位移点沿用参考 XTDA 的 Davidson/dense 等求解设置。
        xtda_options = {
            name: getattr(self.base, name)
            for name in (
                "method",
                "davidson",
                "davidson_backend",
                "so2st",
                "dense_batch_size",
                "jk_batch_size",
                "jk_block_split",
                "use_delta_a",
                "davidson_matvec_batch_size",
            )
        }
        displaced_td = XTDA(
            aligned_mf, df_cache=self.base.df_cache_config, **xtda_options
        )
        # 跟踪参考计算已有的全部根，包括用户为稳定 Davidson 求解而额外
        # 请求的 buffer roots；根数过少时，目标态可能在位移后跑出子空间。
        displaced_td.kernel(nstates=reference_amplitudes.shape[1])
        _check_solution(displaced_td, reference_amplitudes.shape[1])
        # MO 对齐只统一了组态基，XTDA 本征态本身仍需再换序和换号。
        displaced_amplitudes = align_amplitudes(
            reference_amplitudes,
            _asnumpy(displaced_td.v),
            self.align_threshold,
        )
        return aligned_mf, displaced_amplitudes

    def kernel(self):
        """Return ``{(state_i, state_j): NAC}`` for all atoms."""
        self.de = None
        pairs = _state_pairs(self.pairs)
        if not np.isfinite(self.step) or self.step <= 0:
            raise ValueError("step must be positive and finite (angstrom)")
        if (
            not np.isfinite(self.align_threshold)
            or not 0 < self.align_threshold < 1
        ):
            raise ValueError("align_threshold must lie between zero and one")
        reference_td = self.base
        reference_mf = reference_td.mf
        # Physical state N maps to XTDA root N-1.
        _check_solution(reference_td, int(pairs.max()))

        # 物理态 I>0 对应 reference_amplitudes[:, I-1]；态 0 是基态。
        reference_amplitudes = _asnumpy(reference_td.v)
        if np.iscomplexobj(reference_amplitudes) or np.iscomplexobj(
            reference_mf.mo_coeff
        ):
            raise NotImplementedError(
                "NAC currently requires real orbitals and amplitudes"
            )
        # PDF 式 (6)：gamma^{IJ} 固定取参考构型 R0 的值。
        # 对激发态对，左侧振幅 X_I 也固定在 R0，右侧 X_J 做差分。
        gamma = np.asarray(
            [
                reference_td.transition_density_matrix(state_i, state_j)
                for state_i, state_j in pairs
            ]
        )
        # gamma.shape = (npair, nmo, nmo)。
        # nac[k, A, alpha] 对应第 k 个态对在原子 A、方向 alpha 的 NAC。
        nac = np.zeros((len(pairs), self.mol.natm, 3))
        # self.step 的单位为 angstrom；utils.unit.bohr 是 1 bohr 的 angstrom
        # 数值，所以 dR 是以 bohr 表示的半步长 h，与原程序的 dR 相同。
        dR = self.step / bohr

        # 对每个核坐标 R_{A alpha} 独立求解 R0+h 和 R0-h。
        # ponytail: 非简并情形用符号置换即可；简并子空间需要更一般的
        # 子空间对齐，目前遇到歧义直接报错，避免静默地产生错误 NAC。
        for atom in range(self.mol.natm):
            for axis in range(3):
                logger.info(self, "NAC atom %d, %s", atom + 1, "XYZ"[axis])
                plus_mol = displaced_mol(self.mol, atom, axis, self.step)
                minus_mol = displaced_mol(self.mol, atom, axis, -self.step)
                plus_mf, plus_amplitudes = self._solve_displaced(
                    plus_mol, reference_amplitudes
                )
                minus_mf, minus_amplitudes = self._solve_displaced(
                    minus_mol, reference_amplitudes
                )

                # PDF 式 (7)：
                # <phi_p|d phi_q/dR> ~=
                #   [<phi_p(R0)|phi_q(R0+h)> -
                #    <phi_p(R0)|phi_q(R0-h)>] / (2h)。
                orbital_derivative = (
                    mo_overlap(reference_mf, plus_mf)
                    - mo_overlap(reference_mf, minus_mf)
                ) / (2.0 * dR)
                # 对实、归一化且相位连续的轨道，归一化条件的导数给出
                # <phi_p|d phi_p/dR> = 0；显式置零可去除数值噪声。
                np.fill_diagonal(orbital_derivative, 0.0)

                # PDF 式 (8)：dX_J/dR ~= [X_J(R0+h)-X_J(R0-h)]/(2h)。
                amplitude_derivative = (
                    plus_amplitudes - minus_amplitudes
                ) / (2.0 * dR)

                # PDF 式 (6) 第一项：sum_pq gamma_pq^{IJ}
                #                              <phi_p|d phi_q/dR>。
                orbital_term = np.einsum(
                    "kpq,pq->k", gamma, orbital_derivative
                )
                # PDF 式 (6) 第二项只存在于两个激发态之间。
                # X_I.T dX_J exists only when both physical states are
                # excited.  The ground determinant has no XTDA amplitude.
                amplitude_term = np.zeros(len(pairs))
                excited_pairs = np.all(pairs > 0, axis=1)
                amplitude_term[excited_pairs] = np.einsum(
                    "dk,dk->k",
                    reference_amplitudes[:, pairs[excited_pairs, 0] - 1],
                    amplitude_derivative[:, pairs[excited_pairs, 1] - 1],
                )
                nac[:, atom, axis] = orbital_term + amplitude_term

        self.de = {tuple(pair): nac[index] for index, pair in enumerate(pairs)}
        return self.de


def finite_difference_nac(mol, pairs, xc, step=1e-4, save=True):
    """Compatibility entry point using physical states (0=ground, 1=S1, ...)."""
    pairs = _state_pairs(pairs)
    mf = dft.ROKS(mol)
    mf.xc = xc
    mf.max_cycle = 400
    mf.conv_tol = 1e-11
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("ROKS did not converge")

    td = XTDA(mf, davidson=True, davidson_backend="cpu")
    energies, _ = td.kernel(nstates=int(pairs.max()))
    nac_driver = NAC(td, pairs, step)
    result = nac_driver.kernel()
    if save:
        _write_results(mol, pairs, energies, result, xc, step)
    return result


def _write_results(mol, pairs, energies, result, xc, step):
    nac = np.asarray([result[tuple(pair)] for pair in pairs])
    symbols = np.asarray([mol.atom_symbol(atom) for atom in range(mol.natm)])
    with Path("xtda_nac.txt").open("w", encoding="utf-8") as output:
        print(
            "XTDA central-finite-difference NAC / bohr^-1; state 0 is ground",
            file=output,
        )
        print(f"Step = {step:.8e} angstrom", file=output)
        print(
            "Excitation energies / hartree (physical states 1, 2, ...):",
            energies,
            file=output,
        )
        for pair_index, (state_i, state_j) in enumerate(pairs):
            print(f"\nState pair {state_i} {state_j}", file=output)
            for atom, symbol in enumerate(symbols):
                x, y, z = nac[pair_index, atom]
                values = " ".join(f"{value:18.10e}" for value in (x, y, z))
                print(f"{atom + 1:4d} {symbol:3s} {values}", file=output)
    np.savez_compressed(
        "xtda_nac.npz",
        state_pairs=pairs,
        nac_per_bohr=nac,
        energies_hartree=energies,
        symbols=symbols,
        coordinates_angstrom=mol.atom_coords(unit="Angstrom"),
        dx_angstrom=step,
        xc=xc,
    )
