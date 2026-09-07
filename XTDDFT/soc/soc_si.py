"""SOC 态相互作用高层驱动。

依次运行本地三个自旋块方法（XSF-TDA-down 给出 |S->、XTDA 给出 |So>、
SF-TDA-up 给出 |S+>），用 sf-X2C + SOMF 构造自旋轨道耦合哈密顿量，
最后交给 SI_driver 做态相互作用。跃迁偶极矩阵直接复用三个方法类自带
的 transition_dipole_matrix / transition_dipoles_ground 实现。
"""

import numpy as np

from ..base import _is_pbc_mf, _reject_k_method
from ..sf_tda_up import SF_TDA_up
from ...utils.backend import _asnumpy, require_cupy, resolve_backend
from ...utils.unit import BDF_c
from ..xtda import XTDA
from ..xsf_tda_down import XSF_TDA_down
from .si_driver import SI_driver
from .x2c_somf import get_soDKH1_somf

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger(__name__)


class SOCSI:
    """自旋轨道耦合态相互作用（SOC-SI）驱动。

    Args:
        mf: ROKS/RHF 参考态（分子体系，PBC 与 UKS 暂不支持）。
        nstates: (n_sm, n_so, n_sp)，|S->、|So>、|S+> 三个自旋块分别要
            多少个激发态。S=1/2 时 |S-> 块不存在，n_sm 会被忽略。
        xsf_method / sf_method: 传给 XSF_TDA_down / SF_TDA_up 的 method。
        davidson: 三个方法是否用 Davidson 求解。
        davidson_backend: "cpu" 或 "gpu"。
        ngs: 是否包含基态（1）或不包含（0）。
        cal_osc: 是否计算跃迁偶极矩和振子强度。
        iop: "x2c" 或 "bp"，传给 get_soDKH1_somf。
        include_mf2e / mf2e_impl / nproc / use_1c: SOMF 双电子项参数。
        c: 光速，默认 BDF 值 137.0359895。
        xsf_kwargs / xtda_kwargs / sf_kwargs: 传给各方法类的额外参数。
        backend: "auto"/"cpu"/"gpu"，SOC 计算（Vso/SI）使用的硬件后端。
    """

    def __init__(self, mf, nstates=(20, 20, 20), xsf_method=0, sf_method=0,
                 davidson=True, davidson_backend="cpu", ngs=1, cal_osc=True,
                 iop="x2c", include_mf2e=True, mf2e_impl="auto", nproc=1,
                 use_1c=True, c=None, xsf_kwargs=None, xtda_kwargs=None,
                 sf_kwargs=None, backend="auto"):
        _reject_k_method(mf)
        if _is_pbc_mf(mf):
            raise NotImplementedError("SOC-SI 目前只支持分子体系")
        if mf.mo_coeff.ndim != 2:
            raise NotImplementedError("SOC-SI 需要 ROKS/RHF 参考态（mo_coeff 为二维）")
        self.mf = mf
        self.mol = mf.mol
        self.S = self.mol.spin / 2.0
        if self.S < 0.5:
            raise NotImplementedError("SOC-SI 需要开壳层参考态（S >= 1/2）")
        self.nstates = tuple(nstates)
        self.xsf_method = xsf_method
        self.sf_method = sf_method
        self.davidson = davidson
        self.davidson_backend = davidson_backend
        self.ngs = ngs
        self.cal_osc = bool(cal_osc)
        self.iop = iop
        self.include_mf2e = include_mf2e
        self.mf2e_impl = mf2e_impl
        self.nproc = nproc
        self.use_1c = use_1c
        self.c = c
        self.xsf_kwargs = xsf_kwargs or {}
        self.xtda_kwargs = xtda_kwargs or {}
        self.sf_kwargs = sf_kwargs or {}
        self.backend = resolve_backend(backend)
        self.xp = require_cupy() if self.backend == "gpu" else np

    def _collect_states(self, n_sm, n_so, n_sp):
        """把三个方法类的本征向量组装成 SI_driver 的 state dict。"""
        states = {"|S->": [], "|So>": [], "|S+>": []}
        if self.xsf is not None:
            vects = self.xp.asarray(_asnumpy(self.xsf.vects))
            v = self.xp.asarray(_asnumpy(self.xsf.v))[:, :n_sm]
            no = self.xsf.no
            dim3 = (self.xsf.nc * self.xsf.nv
                    + self.xsf.nc * self.xsf.no
                    + self.xsf.no * self.xsf.nv)
            for i in range(n_sm):
                xo = (vects @ v[dim3:, i]).reshape(no, no)
                xo_diag = self.xp.einsum("ii->i", xo)
                xo_non = xo - self.xp.einsum("i,ij->ij", xo_diag, self.xp.eye(no))
                states["|S->"].append((
                    float(_asnumpy(self.xsf.e[i])),
                    self.xp.concatenate([v[:dim3, i], xo_non.reshape(-1), xo_diag]),
                ))
        v_so = self.xp.asarray(_asnumpy(self.xtda.v))[:, :n_so]
        for i in range(n_so):
            states["|So>"].append((float(_asnumpy(self.xtda.e[i])), v_so[:, i]))
        v_sp = self.xp.asarray(_asnumpy(self.sf.v))[:, :n_sp]
        for i in range(n_sp):
            states["|S+>"].append((float(_asnumpy(self.sf.e[i])), v_sp[:, i]))
        return states

    def _collect_tdm_blocks(self, n_sm, n_so, n_sp):
        """复用本地方法类的跃迁偶极矩阵，组装 SI_driver 需要的分块。"""
        gs_so = self.xp.asarray(_asnumpy(self.xtda.transition_dipoles_ground()))[:n_so]
        so = self.xp.asarray(_asnumpy(self.xtda.transition_dipole_matrix()))[:n_so, :n_so]
        sm = None
        if self.xsf is not None:
            sm = self.xp.asarray(_asnumpy(self.xsf.transition_dipole_matrix()))[:n_sm, :n_sm]
        sp = self.xp.asarray(_asnumpy(self.sf.transition_dipole_matrix()))[:n_sp, :n_sp]
        return {"gs_so": gs_so, "so": so, "sm": sm, "sp": sp}

    def kernel(self, printnum=40):
        """运行三个自旋块 TDA、构造 Vso 并做 SOC 态相互作用。"""
        n_sm, n_so, n_sp = self.nstates
        if self.S == 0.5:
            n_sm = 0

        self.xsf = None
        if n_sm:
            self.xsf = XSF_TDA_down(
                self.mf, self.xsf_method, davidson=self.davidson,
                davidson_backend=self.davidson_backend, **self.xsf_kwargs,
            )
            self.xsf.kernel(nstates=n_sm, remove=1)
            n_sm = self.xsf.v.shape[1]

        self.xtda = XTDA(
            self.mf, davidson=self.davidson,
            davidson_backend=self.davidson_backend, **self.xtda_kwargs,
        )
        self.xtda.kernel(nstates=n_so)
        n_so = self.xtda.v.shape[1]

        self.sf = SF_TDA_up(
            self.mf, self.sf_method, davidson=self.davidson,
            davidson_backend=self.davidson_backend, **self.sf_kwargs,
        )
        self.sf.kernel(nstates=n_sp)
        n_sp = self.sf.v.shape[1]

        self.states = self._collect_states(n_sm, n_so, n_sp)

        c = self.c if self.c is not None else BDF_c
        self.Vso = get_soDKH1_somf(
            self.mf, self.mol, c, iop=self.iop,
            include_mf2e=self.include_mf2e, mf2e_impl=self.mf2e_impl,
            nproc=self.nproc, use_1c=self.use_1c, backend=self.backend,
        )
        mo = self.xp.asarray(_asnumpy(self.mf.mo_coeff))
        self.Vso_mo = self.xp.einsum("nij,ik,jl->nkl", self.Vso, mo, mo)

        tdm_blocks = None
        if self.cal_osc:
            tdm_blocks = self._collect_tdm_blocks(n_sm, n_so, n_sp)

        self.mysi = SI_driver(
            mf=self.mf, S=self.S, Vso=self.Vso_mo, ngs=self.ngs,
            states=self.states, cal_osc=self.cal_osc, tdm_blocks=tdm_blocks,
            backend=self.backend,
        )
        self.eso, self.vso = self.mysi.kernel(printnum=printnum)
        return self.eso, self.vso
