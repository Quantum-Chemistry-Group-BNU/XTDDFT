import numpy as np

from .base import (
    XTDDFT_base,
    _as_cpu_ctx,
    _as_cpu_mf,
    _build_initial_guess_from_gaps,
    _get_hf_mo_fock,
    _get_mo_fock,
    _get_ovlp,
    _molecular_dipole_integrals,
    _molecular_ground_dipole,
    _prepare_davidson_init_space,
    _run_davidson,
    _system,
)
from ..utils.hxc_part import gen_response_tda
from ..utils.backend import _asarray, _asnumpy, backend, contract, get_array_module, require_cupy, set_backend, xp
from ..utils.unit import ha2eV

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger(__name__)


def _order_pyscf2my(nc, no, nv):
    """Convert alpha|beta TDA order to CVa|OVa|COb|CVb order."""
    order = np.indices(((nc + no) * nv + nc * (no + nv),)).squeeze()
    for oi in range(nc):
        for noi in range(no):
            order = np.insert(
                order,
                (nc + no) * nv + no * oi + noi,
                (nc + no) * nv + oi * (no + nv) + noi,
            )
            order = np.delete(order, (nc + no) * nv + oi * (no + nv) + noi + 1)
    return order


def _so2st(eigvec, nc, no, nv):
    """Convert CVa|OVa|COb|CVb vectors to CV(0)|CO(0)|OV(0)|CV(1)."""
    cva = eigvec[:nc * nv]
    ova = eigvec[nc * nv:(nc + no) * nv]
    cob = eigvec[(nc + no) * nv:(nc + no) * nv + nc * no]
    cvb = eigvec[(nc + no) * nv + nc * no:]
    cv0 = np.sqrt(0.5) * (cva + cvb)
    cv1 = np.sqrt(0.5) * (cvb - cva)
    return np.vstack([cv0, cob, ova, cv1])


def _so2st_matrix(nc, no, nv):
    dim = 2 * nc * nv + nc * no + no * nv
    return _so2st(np.eye(dim), nc, no, nv)


def _st2so(eigvec, nc, no, nv):
    """Convert CV(0)|CO(0)|OV(0)|CV(1) to CVa|OVa|COb|CVb order."""
    cv0 = eigvec[:nc * nv]
    co0 = eigvec[nc * nv:nc * (no + nv)]
    ov0 = eigvec[nc * (no + nv):nc * (no + nv) + no * nv]
    cv1 = eigvec[nc * (no + nv) + no * nv:]
    cva = (cv0 - cv1) / np.sqrt(2.0)
    cvb = (cv0 + cv1) / np.sqrt(2.0)
    return np.vstack((cva, ov0, co0, cvb))


def _st2so_matrix(nc, no, nv):
    dim = 2 * nc * nv + nc * no + no * nv
    return _st2so(np.eye(dim), nc, no, nv)




class XTDA(XTDDFT_base):
    """Spin-conserving open-shell TDA.

    Davidson still uses PySCF alpha|beta working vectors internally.  For
    restricted open-shell references, stored eigenvectors and post-processing
    use the spin-tensor order CV(0)|CO(0)|OV(0)|CV(1).  Unrestricted references
    keep the spin-orbital order CVa|OVa|COb|CVb.
    """

    def __init__(self, mf, method=0, davidson=True, davidson_backend="cpu",
                 so2st=False, dense_batch_size=64, jk_batch_size=None,
                 jk_block_split=False, use_delta_a=True, df_cache=None,
                 davidson_matvec_batch_size=None):
        if method != 0:
            raise NotImplementedError("XTDA currently implements method=0 spin-conserving response only.")
        davidson_backend = davidson_backend.lower()
        if davidson_backend not in ("cpu", "gpu", "auto"):
            raise ValueError("davidson_backend must be 'cpu', 'gpu', or 'auto'")
        super().__init__(mf, method, davidson=davidson, df_cache=df_cache)
        logger.info("XTDA spin-conserving TDA response")
        self.davidson_backend = "cpu" if davidson_backend == "auto" else davidson_backend
        self.so2st = so2st
        self.dense_batch_size = dense_batch_size
        if jk_batch_size is not None and jk_batch_size < 1:
            raise ValueError("jk_batch_size must be a positive integer or None")
        self.jk_batch_size = jk_batch_size
        if davidson_matvec_batch_size is not None and davidson_matvec_batch_size < 1:
            raise ValueError("davidson_matvec_batch_size must be a positive integer or None")
        self.davidson_matvec_batch_size = davidson_matvec_batch_size
        self.jk_block_split = jk_block_split
        self.use_delta_a = bool(use_delta_a)
        self.order = _order_pyscf2my(self.nc, self.no, self.nv)
        self._raw_v = None
        self.type_u = _asnumpy(self.mf.mo_coeff).ndim == 3
        self.dS2 = None

    def _result_method_label(self):
        return "spin_conserving"

    @property
    def _is_restricted_open_shell(self):
        return _asnumpy(self.mf.mo_coeff).ndim != 3 and self.no > 0

    def _split_vectors(self, zs, ctx):
        nocca, nvira = ctx.nocc_a, ctx.nvir_a
        noccb, nvirb = ctx.nocc_b, ctx.nvir_b
        # PySCF 工作顺序 alpha|beta；x 为试探向量编号，i/a 为占据/虚轨道指标。
        # za[:, :nc, :]=CVa，za[:, nc:, :]=OVa；
        # zb[:, :, :no]=COb，zb[:, :, no:]=CVb。
        # beta 展平后每个 C 的 O/V 列交错；self.order 才把 COb、CVb 分别聚拢。
        za = zs[:, :nocca * nvira].reshape(zs.shape[0], nocca, nvira)
        zb = zs[:, nocca * nvira:].reshape(zs.shape[0], noccb, nvirb)
        return za, zb

    def _tag_transition_dm(self, dms, mo1a, mo1b, orboa, orbob):
        if not backend.is_gpu:
            return dms
        try:
            from gpu4pyscf.lib.cupy_helper import tag_array
        except Exception:
            return dms
        return tag_array(dms, mo1=[mo1a, mo1b], occ_coeff=[orboa, orbob])

    def _hdiag_from_fock(self, ctx, focka_mo, fockb_mo,
                         focka_hf=None, fockb_hf=None, spin=None):
        diag_a = focka_mo.diagonal()
        diag_b = fockb_mo.diagonal()
        e_ia_a = diag_a[ctx.viridx_a] - diag_a[ctx.occidx_a, None]
        e_ia_b = diag_b[ctx.viridx_b] - diag_b[ctx.occidx_b, None]

        if (
            focka_hf is not None
            and fockb_hf is not None
            and ctx.no > 0
            and ctx.nc > 0
            and ctx.nv > 0
            and spin is not None
            and abs(spin) > 1e-14
        ):
            si = 0.5 * spin
            factor_a = 0.5 * (1 - xp.sqrt((si + 1) / si) + 1 / (2 * si))
            factor_b = 0.5 * (-1 + xp.sqrt((si + 1) / si) + 1 / (2 * si))
            delta_f = fockb_hf - focka_hf
            diag_delta = delta_f.diagonal()
            core_delta = diag_delta[:ctx.nc]
            virt_delta = diag_delta[ctx.nocc_a:]

            # self-block diagonals; CVa-CVb terms are off-diagonal couplings.
            e_ia_a[:ctx.nc, :] += (
                factor_a * virt_delta + factor_b * core_delta[:, None]
            )
            e_ia_b[:, -ctx.nv:] += (
                factor_b * virt_delta + factor_a * core_delta[:, None]
            )
        return xp.hstack([e_ia_a.reshape(-1), e_ia_b.reshape(-1)])

    def _apply_response_in_batches(self, vresp, dms):
        batch_size = self.jk_batch_size
        if batch_size is None:
            return vresp(dms)

        if dms.ndim == 4 and dms.shape[0] == 2:
            ntrial = dms.shape[1]
            if ntrial <= batch_size:
                return vresp(dms)
            blocks = []
            for start in range(0, ntrial, batch_size):
                stop = min(start + batch_size, ntrial)
                blocks.append(vresp(dms[:, start:stop]))
            return get_array_module(blocks[0]).concatenate(blocks, axis=1)

        ntrial = dms.shape[0]
        if ntrial <= batch_size:
            return vresp(dms)
        blocks = []
        for start in range(0, ntrial, batch_size):
            stop = min(start + batch_size, ntrial)
            blocks.append(vresp(dms[start:stop]))
        return get_array_module(blocks[0]).concatenate(blocks, axis=0)

    def _apply_response_blocks(self, vresp, dms_blocks):
        response = None
        for dms_block in dms_blocks:
            block_response = self._apply_response_in_batches(vresp, dms_block)
            response = block_response if response is None else response + block_response
        return response

    def _should_use_delta_a(self, mf, ctx, enabled=None):
        enabled = self.use_delta_a if enabled is None else bool(enabled)
        return (
            enabled
            and _asnumpy(mf.mo_coeff).ndim != 3
            and ctx.no > 0
            and _system(mf).spin != 0
        )

    def _apply_davidson_matvec_batch(self, vind, zs, batch_size=None):
        if batch_size is None:
            batch_size = getattr(self, "davidson_matvec_batch_size", None)
        if batch_size is None or getattr(zs, "ndim", 1) == 1 or zs.shape[0] <= batch_size:
            return vind(zs)

        blocks = []
        for start in range(0, zs.shape[0], batch_size):
            blocks.append(vind(zs[start:start + batch_size]))
        return get_array_module(blocks[0]).concatenate(blocks, axis=0)

    def gen_tda_operation(self, mf=None, ctx=None, use_delta_a=None):
        mf = self.mf if mf is None else mf
        ctx = self.ctx if ctx is None else ctx

        mo_coeff = _asarray(ctx.mo_coeff)
        mo_occ = _asarray(ctx.mo_occ)
        orboa = mo_coeff[0][:, ctx.occidx_a]
        orbva = mo_coeff[0][:, ctx.viridx_a]
        orbob = mo_coeff[1][:, ctx.occidx_b]
        orbvb = mo_coeff[1][:, ctx.viridx_b]

        focka_mo, fockb_mo = _get_mo_fock(mf, mo_coeff, mo_occ)
        vresp = gen_response_tda(mf, hermi=0, ctx=ctx)

        # 仅 use_delta_a=True 的非零自旋 RO 开壳层启用额外修正；UHF/UKS 不启用。
        # S_i=mol.spin/2；HF Fock 在当前参考密度上构造，不另做一次 HF 自洽计算。
        apply_delta_a = self._should_use_delta_a(mf, ctx, enabled=use_delta_a)
        if apply_delta_a:
            # Si
            si = 0.5 * _system(mf).spin
            if abs(si) < 1e-14:  # 如果是0，那么
                apply_delta_a = False
            else:
                focka_hf, fockb_hf = _get_hf_mo_fock(mf, mo_coeff, mo_occ)
                # the coefficient \delta A for the Spin-orbital basis
                factor_a = 0.5 * (1 - xp.sqrt((si + 1) / si) + 1 / (2 * si))
                factor_b = 0.5 * (-1 + xp.sqrt((si + 1) / si) + 1 / (2 * si))
                factor_ab = 0.5 / (2 * si)
        hdiag = self._hdiag_from_fock(
            ctx, focka_mo, fockb_mo,
            focka_hf=focka_hf if apply_delta_a else None,
            fockb_hf=fockb_hf if apply_delta_a else None,
            spin=_system(mf).spin if apply_delta_a else None,
        )

        def vind(zs0):
            zs = xp.asarray(zs0)
            if zs.ndim == 1:
                zs = zs.reshape(1, -1)
            nz = zs.shape[0]
            za, zb = self._split_vectors(zs, ctx)

            def response_dm(za_part, zb_part):   # transform to AO
                mo1a_part = contract("xov,pv->xpo", za_part, orbva)
                dmsa_part = contract("xpo,qo->xpq", mo1a_part, orboa.conj())
                mo1b_part = contract("xov,pv->xpo", zb_part, orbvb)
                dmsb_part = contract("xpo,qo->xpq", mo1b_part, orbob.conj())
                dms_part = xp.asarray([dmsa_part, dmsb_part])
                return self._tag_transition_dm(
                    dms_part, mo1a_part, mo1b_part, orboa, orbob
                )

            # 只拆分响应的输入密度；每块仍投影到全部输出通道并相加，保留交叉耦合。
            if self.jk_block_split:
                def response_blocks():
                    za_part = xp.zeros_like(za)
                    zb_part = xp.zeros_like(zb)
                    # 输入 CVa：C_alpha -> V_alpha。
                    za_part[:, :ctx.nc, :] = za[:, :ctx.nc, :]
                    yield response_dm(za_part, zb_part)

                    za_part = xp.zeros_like(za)
                    zb_part = xp.zeros_like(zb)
                    # 输入 OVa：O_alpha -> V_alpha。
                    za_part[:, ctx.nc:, :] = za[:, ctx.nc:, :]
                    yield response_dm(za_part, zb_part)

                    za_part = xp.zeros_like(za)
                    zb_part = xp.zeros_like(zb)
                    # 输入 COb：C_beta -> O_beta。
                    zb_part[:, :, :ctx.no] = zb[:, :, :ctx.no]
                    yield response_dm(za_part, zb_part)

                    za_part = xp.zeros_like(za)
                    zb_part = xp.zeros_like(zb)
                    # 输入 CVb：C_beta -> V_beta。
                    zb_part[:, :, ctx.no:] = zb[:, :, ctx.no:]
                    yield response_dm(za_part, zb_part)

                v1ao = self._apply_response_blocks(vresp, response_blocks())
            else:
                v1ao = self._apply_response_in_batches(vresp, response_dm(za, zb))
            # ab 是 alpha 保自旋激发与 beta 保自旋激发的耦合，不是自旋翻转激发。
            v1ao = xp.asarray(v1ao)
            v1a = contract("xpq,qo->xpo", v1ao[0], orboa)
            v1a = contract("xpo,pv->xov", v1a, orbva.conj())
            v1b = contract("xpq,qo->xpo", v1ao[1], orbob)
            v1b = contract("xpo,pv->xov", v1b, orbvb.conj())

            # 常规 Fock 项 delta_ij*F_ab - F_ij*delta_ab：
            # alpha 覆盖 {CVa,OVa}x{CVa,OVa}；beta 覆盖 {COb,CVb}x{COb,CVb}。
            # 非对角 Fock 元素也包含 CVa<->OVa、COb<->CVb 的一电子耦合。
            v1a += contract("xib,ab->xia", za, focka_mo[ctx.viridx_a[:, None], ctx.viridx_a])
            v1a -= contract("xja,ij->xia", za, focka_mo[ctx.occidx_a[:, None], ctx.occidx_a])
            v1b += contract("xib,ab->xia", zb, fockb_mo[ctx.viridx_b[:, None], ctx.viridx_b])
            v1b -= contract("xja,ij->xia", zb, fockb_mo[ctx.occidx_b[:, None], ctx.occidx_b])

            # 以下仅添加 Delta A：只改 CVa/CVb 的 2x2 通道子矩阵，
            # CO、OV 及其耦合已由上方处理。A[P,Q] 表示输出 P、输入 Q。
            if apply_delta_a:
                nc, no, nv = ctx.nc, ctx.no, ctx.nv
                cv_a = za[:, :nc, :]
                cv_b = zb[:, :, -nv:]
                # Delta A[CVa,CVa]
                v1a[:, :nc, :] += (
                    factor_a * (
                        contract("xib,ab->xia", cv_a, fockb_hf[ctx.nocc_a:, ctx.nocc_a:])
                        - contract("xib,ab->xia", cv_a, focka_hf[ctx.nocc_a:, ctx.nocc_a:])
                    )
                    + factor_b * (
                        contract("xja,ij->xia", cv_a, fockb_hf[:nc, :nc])
                        - contract("xja,ij->xia", cv_a, focka_hf[:nc, :nc])
                    )
                )
                # Delta A[CVa,CVb]
                cv_coupling = (
                    contract("xib,ab->xia", cv_b, fockb_hf[ctx.nocc_a:, ctx.nocc_a:])
                    - contract("xib,ab->xia", cv_b, focka_hf[ctx.nocc_a:, ctx.nocc_a:])
                    + contract("xja,ij->xia", cv_b, fockb_hf[:nc, :nc])
                    - contract("xja,ij->xia", cv_b, focka_hf[:nc, :nc])
                )
                v1a[:, :nc, :] -= factor_ab * cv_coupling

                # Delta A[CVb,CVa]
                v1b[:, :, -nv:] -= factor_ab * (
                    contract("xib,ab->xia", cv_a, fockb_hf[ctx.nocc_a:, ctx.nocc_a:])
                    - contract("xib,ab->xia", cv_a, focka_hf[ctx.nocc_a:, ctx.nocc_a:])
                    + contract("xja,ij->xia", cv_a, fockb_hf[:nc, :nc])
                    - contract("xja,ij->xia", cv_a, focka_hf[:nc, :nc])
                )
                # Delta A[CVb,CVb]
                v1b[:, :, -nv:] += (
                    factor_b * (
                        contract("xib,ab->xia", cv_b, fockb_hf[ctx.nocc_a:, ctx.nocc_a:])
                        - contract("xib,ab->xia", cv_b, focka_hf[ctx.nocc_a:, ctx.nocc_a:])
                    )
                    + factor_a * (
                        contract("xja,ij->xia", cv_b, fockb_hf[:nc, :nc])
                        - contract("xja,ij->xia", cv_b, focka_hf[:nc, :nc])
                    )
                )
            return xp.hstack([v1a.reshape(nz, -1), v1b.reshape(nz, -1)])

        return vind, hdiag

    def init_guess(self, nstates, hdiag=None):
        if hdiag is None:
            _, hdiag = self.gen_tda_operation()
        return _build_initial_guess_from_gaps(hdiag, nstates)

    def davidson_process(self, nstates, init_space=None):
        vind, hdiag = self.gen_tda_operation()
        matvec_batch_size = self.davidson_matvec_batch_size
        if matvec_batch_size is not None:
            base_vind = vind
            vind = lambda zs: self._apply_davidson_matvec_batch(
                base_vind, zs, matvec_batch_size
            )
        nroots = min(nstates, int(hdiag.size))
        x0 = _prepare_davidson_init_space(self.init_guess(nroots, hdiag=hdiag), init_space)
        converged, e, x1 = _run_davidson(
            self.mf, self.davidson_backend,
            vind, hdiag, x0, nroots,
        )
        self.converged = converged
        self.e = xp.asarray(e)
        raw_v = xp.asarray(_asnumpy(x1)).T
        self._raw_v = raw_v
        # 先重排为 CVa|OVa|COb|CVb；RO 再转张量基，U 参考保留轨道基。
        # 当前实现由 type_u 决定是否转换，并不由构造参数 so2st 控制。
        ordered_v = raw_v[self.order]
        if self.type_u:
            self.v = ordered_v
        else:
            self.v = xp.asarray(_so2st(_asnumpy(ordered_v), self.nc, self.no, self.nv))
        logger.info("XTDA Davidson converged: {}", converged)
        return self.e, self.v

    def _get_dense_integrals(self):
        """Return Fock and Hxc integrals shared by the two dense builders.
           Only use CPU"""
        from pyscf import ao2mo
        from .base import (
            _df_ao2mo_pbc, _is_pbc_mf, _iter_block_data, _response_max_memory,
        )

        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)
        mode = backend.mode
        set_backend("cpu")
        try:
            mo_coeff = np.asarray(ctx.mo_coeff)
            mo_occ = np.asarray(ctx.mo_occ)
            if np.iscomplexobj(mo_coeff):
                raise NotImplementedError("Dense XTDA requires real molecular or Gamma orbitals")
            occ = (ctx.occidx_a, ctx.occidx_b)
            vir = (ctx.viridx_a, ctx.viridx_b)
            orbo = [mo_coeff[s][:, occ[s]] for s in range(2)]
            orbv = [mo_coeff[s][:, vir[s]] for s in range(2)]

            def eri(coeffs, omega=None):   # (uv|wx)
                shape = tuple(c.shape[1] for c in coeffs)
                if 0 in shape:
                    return np.zeros(shape)
                if _is_pbc_mf(mf):
                    from pyscf.pbc.df import aft, df
                    from .base import _get_gamma_kpt

                    # Match GDF.get_jk's AFT integrator for long-range exchange.
                    if (omega is not None and omega > 0
                            and isinstance(mf.with_df, df.GDF)
                            and mf.cell.dimension >= 2
                            and mf.cell.low_dim_ft_type != "inf_vacuum"):
                        lr_df = aft.AFTDF(mf.cell, mf.with_df.kpts)
                        cutoff = aft.estimate_ke_cutoff_for_omega(mf.cell, omega)
                        lr_df.mesh = mf.cell.cutoff_to_mesh(cutoff)
                        with lr_df.range_coulomb(omega) as rsh_df:
                            value = rsh_df.ao2mo(
                                coeffs, _get_gamma_kpt(mf), compact=False
                            )
                    else:
                        value = _df_ao2mo_pbc(mf, coeffs, omega=omega, compact=False)
                elif getattr(mf, "with_df", None) is not None:
                    if omega is None:
                        value = mf.with_df.ao2mo(coeffs, compact=False)
                    else:
                        with mf.with_df.range_coulomb(omega) as rsh_df:
                            value = rsh_df.ao2mo(coeffs, compact=False)
                elif omega is None:
                    value = ao2mo.general(mf.mol, coeffs, compact=False)
                else:
                    with mf.mol.with_range_coulomb(omega):
                        value = ao2mo.general(mf.mol, coeffs, compact=False)
                return np.asarray(value).reshape(shape)

            # ao2mo omits the exchange G=0 correction included by get_k.
            # exxdiv correction
            exx_shift = 0.0
            if _is_pbc_mf(mf) and getattr(mf, "exxdiv", None) is not None:
                from pyscf.pbc.tools import pbc
                from .base import _get_gamma_kpt

                if mf.exxdiv != "ewald":
                    raise NotImplementedError("Dense XTDA supports exxdiv=None or 'ewald'")
                kpt = _get_gamma_kpt(mf)
                exx_shift = self.hyb * pbc.madelung(mf.cell, kpt)
                if self.omega != 0 and self.alpha != self.hyb:
                    exx_shift += (self.alpha - self.hyb) * pbc.madelung(
                        mf.cell, kpt, omega=self.omega
                    )

            # For spin-conserving transitions i_s -> a_s and j_t -> b_t,
            # this builds the J + exact-exchange part of the response kernel:
            #
            #   K^{st}_{ia,jb} = (i_s a_s|j_t b_t)
            #       - delta_st [hyb (i_s j_s|a_s b_s)
            #       + (alpha-hyb) (i_s j_s|a_s b_s)_LR(omega)
            #       + exx_shift <i_s|j_s><a_s|b_s>].
            #
            # (pq|rs) uses chemists' notation.  Exact exchange is same-spin
            # only; transpose maps ao2mo order (i,j,a,b) to (i,a,j,b).
            # The semilocal f_xc and one-electron Fock terms are added below.
            # Blocks: (s,t) = (alpha,alpha), (alpha,beta), (beta,beta).
            kernels = []
            for s, t in ((0, 0), (0, 1), (1, 1)):
                k = eri((orbo[s], orbv[s], orbo[t], orbv[t]))
                if s == t:
                    coeffs = (orbo[s], orbo[s], orbv[s], orbv[s])
                    if self.hyb != 0:
                        k -= self.hyb * eri(coeffs).transpose(0, 2, 1, 3)
                    if self.omega != 0 and self.alpha != self.hyb:
                        k -= (self.alpha - self.hyb) * eri(
                            coeffs, self.omega
                        ).transpose(0, 2, 1, 3)
                    if exx_shift != 0:
                        overlap = np.asarray(_get_ovlp(mf))
                        k -= exx_shift * np.einsum(
                            "ij,ab->iajb",
                            orbo[s].T @ overlap @ orbo[s],
                            orbv[s].T @ overlap @ orbv[s],
                        )
                kernels.append(k)

            if self.mfxctype is not None and self.xctype != "HF":
                ni = mf._numint
                dm = np.asarray([
                    (mo_coeff[s] * mo_occ[s]) @ mo_coeff[s].T for s in range(2)
                ])
                make_rho = ni._gen_rho_evaluator(
                    _system(mf), dm, hermi=1, with_lapl=False
                )[0]
                max_memory = _response_max_memory(mf, None)
                deriv = 0 if self.xctype == "LDA" else 1
                for ao, mask, weight, coords in _iter_block_data(
                    mf, ni, deriv, max_memory, force_cpu=True
                ):
                    rho = [make_rho(s, ao, mask, self.xctype) for s in range(2)]
                    fxc = ni.eval_xc_eff(
                        mf.xc, rho, deriv=2, xctype=self.xctype
                    )[2]
                    pairs = []
                    for s in range(2):
                        if self.xctype == "LDA":
                            pair = np.einsum(
                                "ri,ra->ria", ao @ orbo[s], ao @ orbv[s]
                            )[None]
                        else:
                            ro = np.einsum("xrp,pi->xri", ao, orbo[s])
                            rv = np.einsum("xrp,pa->xra", ao, orbv[s])
                            pair = np.einsum("xri,ra->xria", ro, rv[0])
                            pair[1:4] += np.einsum("ri,xra->xria", ro[0], rv[1:4])
                            if self.xctype == "MGGA":
                                tau = 0.5 * np.einsum("xri,xra->ria", ro[1:4], rv[1:4])
                                pair = np.concatenate((pair, tau[None]), axis=0)
                        pairs.append(pair)
                    for k, (s, t) in zip(kernels, ((0, 0), (0, 1), (1, 1))):
                        k += np.einsum(
                            "xria,xyr,yrjb,r->iajb",
                            pairs[s], fxc[s, :, t, :], pairs[t], weight,
                            optimize=True,
                        )

            focka, fockb = _get_mo_fock(mf, mo_coeff, mo_occ)
            delta_f = None
            if self._should_use_delta_a(mf, ctx):
                fha, fhb = _get_hf_mo_fock(mf, mo_coeff, mo_occ)
                delta_f = np.asarray(fhb - fha)
            return ctx, focka, fockb, kernels, delta_f
        finally:
            set_backend(mode)

    def get_Amat(self):
        """Build SO A; self.so2st optionally converts the completed RO matrix.
        """
        ctx, focka, fockb, (aa, ab, bb), delta_f = self._get_dense_integrals()
        nc, no, nv = ctx.nc, ctx.no, ctx.nv
        na, nb = ctx.nocc_a * ctx.nvir_a, ctx.nocc_b * ctx.nvir_b
        # Build the alpha-alpha excitation block A^{aa,aa}.
        aa += (
            np.einsum("ij,ab->iajb", np.eye(ctx.nocc_a),
                      focka[np.ix_(ctx.viridx_a, ctx.viridx_a)])
            - np.einsum("ab,ji->iajb", np.eye(ctx.nvir_a),
                        focka[np.ix_(ctx.occidx_a, ctx.occidx_a)])
        )
        # Build the beta-beta excitation block A^{bb,bb}.
        bb += (
            np.einsum("ij,ab->iajb", np.eye(ctx.nocc_b),
                      fockb[np.ix_(ctx.viridx_b, ctx.viridx_b)])
            - np.einsum("ab,ji->iajb", np.eye(ctx.nvir_b),
                        fockb[np.ix_(ctx.occidx_b, ctx.occidx_b)])
        )

        # Add Delta A to the CVa-CVa, CVb-CVb, and CVa-CVb subblocks.
        if delta_f is not None:
            si = 0.5 * _system(self.mf).spin
            q = np.sqrt((si + 1) / si)
            factor_a = 0.5 * (1 - q + 1 / (2 * si))
            factor_b = 0.5 * (-1 + q + 1 / (2 * si))
            dv = np.einsum("ij,ab->iajb", np.eye(nc), delta_f[nc + no:, nc + no:])
            dc = np.einsum("ab,ji->iajb", np.eye(nv), delta_f[:nc, :nc])
            aa[:nc, :, :nc, :] += factor_a * dv + factor_b * dc
            bb[:, no:, :, no:] += factor_b * dv + factor_a * dc
            # Table II, including the 2013 erratum.
            ab[:nc, :, :, no:] -= (dv + dc) / (4 * si)

        # Assemble the full alpha|beta spin-orbital response matrix.
        amat = np.block([
            [aa.reshape(na, na), ab.reshape(na, nb)],
            [ab.reshape(na, nb).T, bb.reshape(nb, nb)],
        ])
        order = _order_pyscf2my(nc, no, nv)
        amat = amat[np.ix_(order, order)]
        if self.so2st and not self.type_u:
            transform = _so2st_matrix(nc, no, nv)
            amat = transform @ amat @ transform.T
        self.A = amat
        logger.info("XTDA dense A dimension: {}", self.A.shape[0])
        return self.A

    def get_Amat_ST(self):
        """Build CV(0)|CO(0)|OV(0)|CV(1) blocks directly from integrals.

        batch_size is retained for call compatibility; no matvec expansion is used.
        """
        if self.type_u:
            raise ValueError("get_Amat_ST requires a restricted reference")
        ctx, fa, fb, (aa, ab, bb), delta_f = self._get_dense_integrals()
        nc, no, nv = ctx.nc, ctx.no, ctx.nv
        ba = ab.transpose(2, 3, 0, 1)
        c, o, v = slice(0, nc), slice(nc, nc + no), slice(nc + no, None)
        ic, io, iv = np.eye(nc), np.eye(no), np.eye(nv)
        dim1 = nc * nv
        dim2 = dim1 + nc * no
        dim3 = dim2 + no * nv
        amat = np.zeros((dim3 + dim1, dim3 + dim1))
        # X-TDA uses A(S_i=infinity) + Delta A, not the finite-S S-TDA A.
        fmean = 0.5 * (fa + fb)
        fspin = 0.5 * (fb - fa)

        # CV(0)-CV(0)
        amat[:dim1, :dim1] = (
            0.5 * (aa[:nc, :, :nc, :] + bb[:, no:, :, no:]
                   + ab[:nc, :, :, no:] + ba[:, no:, :nc, :])
            + np.einsum("ij,ab->iajb", ic, fmean[v, v])
            - np.einsum("ab,ji->iajb", iv, fmean[c, c])
        ).reshape(dim1, dim1)
        # CV(0)-CO(0)
        amat[:dim1, dim1:dim2] = (
            ab[:nc, :, :, :no] + bb[:, no:, :, :no]
            + np.einsum("ij,av->iajv", ic, fb[v, o])
        ).reshape(dim1, nc * no) / np.sqrt(2)
        # CV(0)-OV(0)
        amat[:dim1, dim2:dim3] = (
            aa[:nc, :, nc:, :] + ba[:, no:, nc:, :]
            - np.einsum("ab,vi->iavb", iv, fa[o, c])
        ).reshape(dim1, no * nv) / np.sqrt(2)
        # CV(0)-CV(1)
        amat[:dim1, dim3:] = (
            0.5 * (-aa[:nc, :, :nc, :] + bb[:, no:, :, no:]
                   + ab[:nc, :, :, no:] - ba[:, no:, :nc, :])
            + np.einsum("ij,ab->iajb", ic, fspin[v, v])
            - np.einsum("ab,ji->iajb", iv, fspin[c, c])
        ).reshape(dim1, dim1)
        # CO(0)-CO(0)
        amat[dim1:dim2, dim1:dim2] = (
            bb[:, :no, :, :no]
            + np.einsum("ij,uv->iujv", ic, fb[o, o])
            - np.einsum("uv,ji->iujv", io, fb[c, c])
        ).reshape(nc * no, nc * no)
        # CO(0)-OV(0)
        amat[dim1:dim2, dim2:dim3] = ba[:, :no, nc:, :].reshape(nc * no, no * nv)
        # CO(0)-CV(1)
        amat[dim1:dim2, dim3:] = (
            -ba[:, :no, :nc, :] + bb[:, :no, :, no:]
            + np.einsum("ij,ub->iujb", ic, fb[o, v])
        ).reshape(nc * no, dim1) / np.sqrt(2)
        # OV(0)-OV(0)
        amat[dim2:dim3, dim2:dim3] = (
            aa[nc:, :, nc:, :]
            + np.einsum("uv,ab->uavb", io, fa[v, v])
            - np.einsum("ab,vu->uavb", iv, fa[o, o])
        ).reshape(no * nv, no * nv)
        # OV(0)-CV(1)
        amat[dim2:dim3, dim3:] = (
            -aa[nc:, :, :nc, :] + ab[nc:, :, :, no:]
            + np.einsum("ab,ju->uajb", iv, fa[c, o])
        ).reshape(no * nv, dim1) / np.sqrt(2)
        # CV(1)-CV(1)
        amat[dim3:, dim3:] = (
            0.5 * (aa[:nc, :, :nc, :] + bb[:, no:, :, no:]
                   - ab[:nc, :, :, no:] - ba[:, no:, :nc, :])
            + np.einsum("ij,ab->iajb", ic, fmean[v, v])
            - np.einsum("ab,ji->iajb", iv, fmean[c, c])
        ).reshape(dim1, dim1)

        if delta_f is not None:
            si = 0.5 * _system(self.mf).spin
            q = np.sqrt((si + 1) / si)
            fs = 0.5 * delta_f
            dv = np.einsum("ij,ab->iajb", ic, fs[v, v])
            dc = np.einsum("ab,ji->iajb", iv, fs[c, c])
            # Table II with the corrected CV(0)-CV(1) sign.
            amat[:dim1, dim3:] += (q - 1) * (dv - dc).reshape(dim1, dim1)
            amat[dim3:, dim3:] += ((dv + dc) / si).reshape(dim1, dim1)

        bounds = (0, dim1, dim2, dim3, dim3 + dim1)
        for i in range(4):
            for j in range(i + 1, 4):
                rows = slice(bounds[i], bounds[i + 1])
                cols = slice(bounds[j], bounds[j + 1])
                amat[cols, rows] = amat[rows, cols].T
        self.A = amat
        logger.info("XTDA spin-tensor A dimension: {}", self.A.shape[0])
        return self.A

    def _diagonalize_dense(self, amat, nstates):
        amat = np.asarray(_asnumpy(amat))
        if self.davidson_backend == "gpu":
            cp = require_cupy()
            e, v = cp.linalg.eigh(cp.asarray(amat))
            self.e = e[:nstates]
            self.v = v[:, :nstates]
        else:
            e, v = np.linalg.eigh(amat)
            self.e = e[:nstates]
            self.v = v[:, :nstates]
        return self.e, self.v

    def _split_analysis_vectors(self, data=None):
        # CVα | OVα | COβ | CVβ  (UKS)  or CV(0) | CO(0) | OV(0) | CV(1) (ROKS)
        data = _asnumpy(self.v if data is None else data)
        dim1 = self.nc * self.nv
        if self.type_u:
            dim2 = dim1 + self.no * self.nv
            dim3 = dim2 + self.nc * self.no
            return (
                data[:dim1].T.reshape(data.shape[1], self.nc, self.nv),
                data[dim1:dim2].T.reshape(data.shape[1], self.no, self.nv),
                data[dim2:dim3].T.reshape(data.shape[1], self.nc, self.no),
                data[dim3:].T.reshape(data.shape[1], self.nc, self.nv),
            )
        dim2 = dim1 + self.nc * self.no
        dim3 = dim2 + self.no * self.nv
        return (
            data[:dim1].T.reshape(data.shape[1], self.nc, self.nv),
            data[dim1:dim2].T.reshape(data.shape[1], self.nc, self.no),
            data[dim2:dim3].T.reshape(data.shape[1], self.no, self.nv),
            data[dim3:].T.reshape(data.shape[1], self.nc, self.nv),
        )

    def deltaS2(self):
        # refer to 10.1063/1.3573374, Eq A9
        if not self.type_u:
            _cv0, _co0, _ov0, cv1 = self._split_analysis_vectors()
            return 2.0 * np.einsum("nij,nij->n", cv1, cv1)

        cv_a, ov_a, co_b, cv_b = self._split_analysis_vectors()

        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)
        mo_coeff = _asnumpy(ctx.mo_coeff)
        mooa = mo_coeff[0][:, _asnumpy(ctx.occidx_a)]
        moob = mo_coeff[1][:, _asnumpy(ctx.occidx_b)]
        mova = mo_coeff[0][:, _asnumpy(ctx.viridx_a)]
        movb = mo_coeff[1][:, _asnumpy(ctx.viridx_b)]
        s = _asnumpy(_get_ovlp(mf))

        sccba = moob.conj().T @ s @ mooa
        sccab = mooa.conj().T @ s @ moob
        svcab = mova.conj().T @ s @ moob
        svcba = movb.conj().T @ s @ mooa
        svvab = mova.conj().T @ s @ movb

        nc, no = self.nc, self.no
        ds2 = (
            contract("nia,nja,ki,jk->n", cv_a, cv_a, sccba[:, :nc], sccba.T[:nc, :])
            + contract("nia,nja,ki,jk->n", ov_a, ov_a, sccba[:, nc:], sccba.T[nc:, :])
            + contract("nia,nja,ki,jk->n", ov_a, cv_a, sccba[:, nc:], sccba.T[:nc, :])
            + contract("nia,nja,ki,jk->n", cv_a, ov_a, sccba[:, :nc], sccba.T[nc:, :])
            - contract("nia,nib,ak,kb->n", cv_a, cv_a, svcab, svcab.T)
            - contract("nia,nib,ak,kb->n", ov_a, ov_a, svcab, svcab.T)
            + contract("nia,nja,ki,jk->n", cv_b, cv_b, sccab, sccab.T)
            + contract("nia,nja,ki,jk->n", co_b, co_b, sccab, sccab.T)
            - contract("nia,nib,ak,kb->n", co_b, co_b, svcba[:no, :], svcba.T[:, :no])
            - contract("nia,nib,ak,kb->n", cv_b, cv_b, svcba[no:, :], svcba.T[:, no:])
            - contract("nia,nib,ak,kb->n", co_b, cv_b, svcba[:no, :], svcba.T[:, no:])
            - contract("nia,nib,ak,kb->n", cv_b, co_b, svcba[no:, :], svcba.T[:, :no])
            - 2.0 * contract("nia,njb,ji,ab->n", cv_a, cv_b, sccba[:, :nc], svvab[:, no:])
            - 2.0 * contract("nia,njb,ji,ab->n", cv_a, co_b, sccba[:, :nc], svvab[:, :no])
            - 2.0 * contract("nia,njb,ji,ab->n", ov_a, cv_b, sccba[:, nc:], svvab[:, no:])
            - 2.0 * contract("nia,njb,ji,ab->n", ov_a, co_b, sccba[:, nc:], svvab[:, :no])
        )
        return np.real_if_close(ds2)

    def _dipole_mo_blocks(self):
        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)
        dip_ao = _molecular_dipole_integrals(mf)
        mo_coeff = _asnumpy(ctx.mo_coeff)
        ints_aa = np.asarray(contract("xpq,pi,qj->xij", dip_ao, mo_coeff[0].conj(), mo_coeff[0]))
        ints_bb = np.asarray(contract("xpq,pi,qj->xij", dip_ao, mo_coeff[1].conj(), mo_coeff[1]))
        return ints_aa, ints_bb, ctx, mf

    def _checked_state_index(self, state):
        nstates = min(self.nstates, _asnumpy(self.v).shape[1])
        state = int(state)
        if state < 0:
            state += nstates
        if state < 0 or state >= nstates:
            raise IndexError(f"state index {state} out of range for {nstates} states")
        return state

    def _state_blocks(self, state):
        state = self._checked_state_index(state)
        return self._split_analysis_vectors(_asnumpy(self.v[:, state:state + 1]))

    def _transition_density_matrix_restricted_ground(self, state):
        # Already compared and verified against the NTO_XTDDFT.pdf & xtda+soc derivation.pdf  by Longfei Chang 26.09.08
        cv0, co0, ov0, _cv1 = [block[0] for block in self._state_blocks(state)]
        nmo = self.nc + self.no + self.nv
        gamma = np.zeros((nmo, nmo), dtype=np.result_type(cv0, co0, ov0))
        c = slice(0, self.nc)
        o = slice(self.nc, self.nc + self.no)
        v = slice(self.nc + self.no, nmo)
        gamma[v, c] += np.sqrt(2.0) * cv0.conj().T
        gamma[o, c] += co0.conj().T
        gamma[v, o] += ov0.conj().T
        return gamma

    def _transition_density_matrix_restricted_excited(self, state_f, state_i):
        # Already compared and verified against the NTO_XTDDFT.pdf & xtda+soc derivation.pdf  by Longfei Chang 26.09.08
        cv0_f, co0_f, ov0_f, cv1_f = [block[0].conj() for block in self._state_blocks(state_f)]
        cv0_i, co0_i, ov0_i, cv1_i = [block[0] for block in self._state_blocks(state_i)]
        nmo = self.nc + self.no + self.nv
        gamma = np.zeros(
            (nmo, nmo),
            dtype=np.result_type(cv0_f, co0_f, ov0_f, cv1_f, cv0_i, co0_i, ov0_i, cv1_i),
        )
        c = slice(0, self.nc)
        o = slice(self.nc, self.nc + self.no)
        v = slice(self.nc + self.no, nmo)
        si = _system(self.mf).spin / 2.0
        eta = np.sqrt((si + 1.0) / (2.0 * si)) if abs(si) > 1.0e-14 else 0.0

        gamma[v, v] += contract("ia,ib->ab", cv0_f, cv0_i)
        gamma[c, c] -= contract("ia,ja->ji", cv0_f, cv0_i)
        gamma[o, o] += contract("iu,iv->uv", co0_f, co0_i)
        gamma[c, c] -= contract("iu,ju->ji", co0_f, co0_i)
        gamma[v, v] += contract("ua,ub->ab", ov0_f, ov0_i)
        gamma[o, o] -= contract("va,ua->uv", ov0_f, ov0_i)
        gamma[v, v] += contract("ia,ib->ab", cv1_f, cv1_i)
        gamma[c, c] -= contract("ia,ja->ji", cv1_f, cv1_i)

        factor = 1.0 / np.sqrt(2.0)
        gamma[v, o] += factor * contract("ia,iu->au", cv0_f, co0_i)
        gamma[o, v] += factor * contract("iu,ia->ua", co0_f, cv0_i)
        gamma[o, c] -= factor * contract("ia,ua->ui", cv0_f, ov0_i)
        gamma[c, o] -= factor * contract("ua,ia->iu", ov0_f, cv0_i)
        gamma[o, v] += eta * contract("iu,ib->ub", co0_f, cv1_i)
        gamma[v, o] += eta * contract("ib,iu->bu", cv1_f, co0_i)
        gamma[c, o] += eta * contract("ua,ia->iu", ov0_f, cv1_i)
        gamma[o, c] += eta * contract("ia,ua->ui", cv1_f, ov0_i)
        return gamma

    def _unrestricted_ground_amplitudes(self, state):
        # Already compared and verified against the NTO_XTDDFT.pdf   by Longfei Chang 26.09.08
        cv_a, ov_a, co_b, cv_b = [block[0] for block in self._state_blocks(state)]
        dtype = np.result_type(cv_a, ov_a, co_b, cv_b)
        amp_a = np.zeros((self.nc + self.no, self.nv), dtype=dtype)
        amp_b = np.zeros((self.nc, self.no + self.nv), dtype=dtype)
        amp_a[:self.nc, :] = cv_a
        amp_a[self.nc:, :] = ov_a
        amp_b[:, :self.no] = co_b
        amp_b[:, self.no:] = cv_b
        return amp_a, amp_b

    def _transition_density_matrix_unrestricted_ground(self, state):
        # Already compared and verified against the NTO_XTDDFT.pdf   by Longfei Chang 26.09.08
        amp_a, amp_b = self._unrestricted_ground_amplitudes(state)
        ctx = self.ctx
        occ_a = _asnumpy(ctx.occidx_a).astype(int)
        vir_a = _asnumpy(ctx.viridx_a).astype(int)
        occ_b = _asnumpy(ctx.occidx_b).astype(int)
        vir_b = _asnumpy(ctx.viridx_b).astype(int)
        mo_coeff = _asnumpy(ctx.mo_coeff)
        nmo_a = int(mo_coeff[0].shape[1])
        nmo_b = int(mo_coeff[1].shape[1])
        gamma = np.zeros((nmo_a + nmo_b, nmo_a + nmo_b), dtype=np.result_type(amp_a, amp_b))
        gamma[np.ix_(vir_a, occ_a)] += amp_a.conj().T
        beta = nmo_a
        gamma[np.ix_(beta + vir_b, beta + occ_b)] += amp_b.conj().T
        return gamma

    def _transition_density_matrix_unrestricted_excited(self, state_f, state_i):
        # Already compared and verified against the NTO_XTDDFT.pdf   by Longfei Chang 26.09.08
        # have vertified against pyscf by Longfei Chang 26.09.08
        amp_a_f, amp_b_f = self._unrestricted_ground_amplitudes(state_f)
        amp_a_i, amp_b_i = self._unrestricted_ground_amplitudes(state_i)
        ctx = self.ctx
        occ_a = _asnumpy(ctx.occidx_a).astype(int)
        vir_a = _asnumpy(ctx.viridx_a).astype(int)
        occ_b = _asnumpy(ctx.occidx_b).astype(int)
        vir_b = _asnumpy(ctx.viridx_b).astype(int)
        mo_coeff = _asnumpy(ctx.mo_coeff)
        nmo_a = int(mo_coeff[0].shape[1])
        nmo_b = int(mo_coeff[1].shape[1])
        gamma = np.zeros(
            (nmo_a + nmo_b, nmo_a + nmo_b),
            dtype=np.result_type(amp_a_f, amp_b_f, amp_a_i, amp_b_i),
        )
        gamma[np.ix_(occ_a, occ_a)] -= contract("ia,ja->ji", amp_a_f.conj(), amp_a_i)
        gamma[np.ix_(vir_a, vir_a)] += contract("ia,ib->ab", amp_a_f.conj(), amp_a_i)
        beta = nmo_a
        gamma[np.ix_(beta + occ_b, beta + occ_b)] -= contract("ia,ja->ji", amp_b_f.conj(), amp_b_i)
        gamma[np.ix_(beta + vir_b, beta + vir_b)] += contract("ia,ib->ab", amp_b_f.conj(), amp_b_i)
        return gamma

    def _physical_excited_state_index(self, state):
        state = int(state)
        if state < 1:
            raise ValueError("excited-state labels start at 1; state 0 is the ground state")
        return self._checked_state_index(state - 1)

    def transition_density_matrix(self, state_f=1, state_i=0):
        """Spin-free transition density matrix for physical states.

        State 0 is the ROHF/UKS reference determinant; labels 1, 2, ... are
        XTDA excited states. Restricted references use C|O|V spatial-MO order;
        unrestricted references use alpha|beta spin-MO order.
        """
        if state_f is None or state_i is None:
            raise ValueError("use physical state 0 for the ground state")
        state_f = int(state_f)
        state_i = int(state_i)
        if state_f == 0 and state_i == 0:
            raise ValueError("at least one state must be an excited state")

        if self.type_u:
            ground = self._transition_density_matrix_unrestricted_ground
            excited = self._transition_density_matrix_unrestricted_excited
        else:
            ground = self._transition_density_matrix_restricted_ground
            excited = self._transition_density_matrix_restricted_excited

        if state_f == 0:
            return ground(self._physical_excited_state_index(state_i)).conj().T
        if state_i == 0:
            return ground(self._physical_excited_state_index(state_f))
        return excited(
            self._physical_excited_state_index(state_f),
            self._physical_excited_state_index(state_i),
        )

    def nto(self, state_f=1, state_i=0, nroots=None):
        """Natural transition orbitals using physical state labels."""
        gamma = self.transition_density_matrix(state_f=state_f, state_i=state_i)
        particles, singular_values, holes_h = np.linalg.svd(gamma, full_matrices=False)
        holes = holes_h.conj().T
        if nroots is not None:
            nroots = min(int(nroots), singular_values.size)
            singular_values = singular_values[:nroots]
            holes = holes[:, :nroots]
            particles = particles[:, :nroots]
        return singular_values, holes, particles

    def _embedded_block_svd(self, matrix, source_indices, target_indices, source, target, nmo, nroots):
        matrix = np.asarray(matrix)
        particles_local, singular_values, holes_h = np.linalg.svd(matrix, full_matrices=False)
        holes_local = holes_h.conj().T
        block_weight = float(np.real_if_close(np.sum(np.abs(singular_values) ** 2)))
        if nroots is not None:
            keep = min(int(nroots), singular_values.size)
            singular_values = singular_values[:keep]
            particles_local = particles_local[:, :keep]
            holes_local = holes_local[:, :keep]

        holes = np.zeros((nmo, singular_values.size), dtype=holes_local.dtype)
        particles = np.zeros((nmo, singular_values.size), dtype=particles_local.dtype)
        holes[source_indices, :] = holes_local
        particles[target_indices, :] = particles_local
        return {
            "source": source,
            "target": target,
            "singular_values": singular_values,
            "weights": np.abs(singular_values) ** 2,
            "block_weight": block_weight,
            "holes": holes,
            "particles": particles,
        }

    def _block_nto_restricted(self, state, nroots):
        cv0, co0, ov0, cv1 = [block[0] for block in self._state_blocks(state)]
        nmo = self.nc + self.no + self.nv
        c = slice(0, self.nc)
        o = slice(self.nc, self.nc + self.no)
        v = slice(self.nc + self.no, nmo)
        specs = {
            "CV(0)": (cv0.T, c, v, "C", "V"),
            "CO(0)": (co0.T, c, o, "C", "O"),
            "OV(0)": (ov0.T, o, v, "O", "V"),
            "CV(1)": (cv1.T, c, v, "C", "V"),
        }
        return {
            name: self._embedded_block_svd(matrix, source_idx, target_idx, source, target, nmo, nroots)
            for name, (matrix, source_idx, target_idx, source, target) in specs.items()
        }

    def _block_nto_unrestricted(self, state, nroots):
        cv_a, ov_a, co_b, cv_b = [block[0] for block in self._state_blocks(state)]
        ctx = self.ctx
        mo_coeff = _asnumpy(ctx.mo_coeff)
        nmo_a = int(mo_coeff[0].shape[1])
        nmo_b = int(mo_coeff[1].shape[1])
        occ_a = _asnumpy(ctx.occidx_a).astype(int)
        vir_a = _asnumpy(ctx.viridx_a).astype(int)
        occ_b = _asnumpy(ctx.occidx_b).astype(int)
        vir_b = _asnumpy(ctx.viridx_b).astype(int)
        beta = nmo_a
        nmo = nmo_a + nmo_b
        specs = {
            "CVa": (cv_a.T, occ_a[:self.nc], vir_a, "alpha_C", "alpha_V"),
            "OVa": (ov_a.T, occ_a[self.nc:self.nc + self.no], vir_a, "alpha_O", "alpha_V"),
            "COb": (co_b.T, beta + occ_b, beta + vir_b[:self.no], "beta_C", "beta_O"),
            "CVb": (cv_b.T, beta + occ_b, beta + vir_b[self.no:self.no + self.nv], "beta_C", "beta_V"),
        }
        return {
            name: self._embedded_block_svd(matrix, source_idx, target_idx, source, target, nmo, nroots)
            for name, (matrix, source_idx, target_idx, source, target) in specs.items()
        }

    def block_nto(self, state=0, nroots=None):
        """SVD channels for XTDA spin-tensor blocks or unrestricted UTDA blocks."""
        if self.type_u:
            return self._block_nto_unrestricted(state, nroots)
        return self._block_nto_restricted(state, nroots)

    def transition_dipoles_ground(self):
        """Ground-state to excited-state transition dipoles in a.u."""
        ints_aa, ints_bb, ctx, _mf = self._dipole_mo_blocks()
        nstates = min(self.nstates, _asnumpy(self.v).shape[1])

        if not self.type_u:
            c = slice(0, self.nc)
            o = slice(self.nc, self.nc + self.no)
            v = slice(self.nc + self.no, self.nc + self.no + self.nv)
            cv0, co0, ov0, _cv1 = self._split_analysis_vectors(_asnumpy(self.v)[:, :nstates])
            tdm = np.zeros((nstates, 3))
            for istate in range(nstates):
                tdm[istate] = (
                    np.sqrt(2.0) * contract("xai,ia->x", ints_aa[:, v, c], cv0[istate])
                    + contract("xiu,iu->x", ints_aa[:, c, o], co0[istate])
                    + contract("xua,ua->x", ints_aa[:, o, v], ov0[istate])
                )
            return tdm
        # have vertified against pyscf by Longfei Chang 26.09.08 for UTDA
        occ_a = _asnumpy(ctx.occidx_a)
        vir_a = _asnumpy(ctx.viridx_a)
        occ_b = _asnumpy(ctx.occidx_b)
        vir_b = _asnumpy(ctx.viridx_b)

        r_ov_a = ints_aa[:, occ_a][:, :, vir_a]
        r_ov_b = ints_bb[:, occ_b][:, :, vir_b]
        r_cv_a = r_ov_a[:, :self.nc, :].reshape(3, -1)
        r_ov_a = r_ov_a[:, self.nc:, :].reshape(3, -1)
        r_co_b = r_ov_b[:, :, :self.no].reshape(3, -1)
        r_cv_b = r_ov_b[:, :, self.no:].reshape(3, -1)
        dip_ordered = np.hstack([r_cv_a, r_ov_a, r_co_b, r_cv_b])

        vectors = _asnumpy(self.v)
        nstates = min(nstates, vectors.shape[1])
        return np.einsum("xd,ds->sx", dip_ordered, vectors[:, :nstates], optimize=True)

    def osc_str(self):
        trans_dip = self.transition_dipoles_ground()
        energies = _asnumpy(self.e)[:trans_dip.shape[0]]
        return (2.0 / 3.0) * energies * np.einsum("sx,sx->s", trans_dip, trans_dip)

    def _full_spin_conserving_amplitudes(self, vectors):
        cv_a, ov_a, co_b, cv_b = self._split_analysis_vectors(vectors)
        amps_a = []
        amps_b = []
        for istate in range(cv_a.shape[0]):
            xa = np.zeros((self.nc + self.no, self.nv))
            xb = np.zeros((self.nc, self.no + self.nv))
            xa[:self.nc, :] = cv_a[istate]
            xa[self.nc:, :] = ov_a[istate]
            xb[:, :self.no] = co_b[istate]
            xb[:, self.no:] = cv_b[istate]
            amps_a.append(xa)
            amps_b.append(xb)
        return amps_a, amps_b

    def _transition_dipole_matrix_restricted(self, include_ground_dipole=False):
        # Already compared and verified against the NTO_XTDDFT.pdf & xtda+soc derivation.pdf  by Longfei Chang 26.09.08
        ints_mo, _ints_bb, _ctx, mf = self._dipole_mo_blocks()
        vectors = _asnumpy(self.v)
        nstates = min(self.nstates, vectors.shape[1])
        c = slice(0, self.nc)
        o = slice(self.nc, self.nc + self.no)
        v = slice(self.nc + self.no, self.nc + self.no + self.nv)
        rcc = ints_mo[:, c, c]
        roo = ints_mo[:, o, o]
        rvv = ints_mo[:, v, v]
        rco = ints_mo[:, c, o]
        roc = ints_mo[:, o, c]
        rov = ints_mo[:, o, v]
        rvo = ints_mo[:, v, o]

        cv0, co0, ov0, cv1 = self._split_analysis_vectors(vectors[:, :nstates])
        cv0_f = cv0.conj()
        co0_f = co0.conj()
        ov0_f = ov0.conj()
        cv1_f = cv1.conj()
        si = _system(self.mf).spin / 2.0
        eta = np.sqrt((si + 1.0) / (2.0 * si)) if abs(si) > 1.0e-14 else 0.0
        factor = 1.0 / np.sqrt(2.0)

        tdm = (
            contract("sia,xab,tib->stx", cv0_f, rvv, cv0)
            - contract("sia,xji,tja->stx", cv0_f, rcc, cv0)
            + contract("siu,xuv,tiv->stx", co0_f, roo, co0)
            - contract("siu,xji,tju->stx", co0_f, rcc, co0)
            + contract("sua,xab,tub->stx", ov0_f, rvv, ov0)
            - contract("sva,xuv,tua->stx", ov0_f, roo, ov0)
            + contract("sia,xab,tib->stx", cv1_f, rvv, cv1)
            - contract("sia,xji,tja->stx", cv1_f, rcc, cv1)
            + factor * contract("sia,xau,tiu->stx", cv0_f, rvo, co0)
            + factor * contract("siu,xua,tia->stx", co0_f, rov, cv0)
            - factor * contract("sia,xui,tua->stx", cv0_f, roc, ov0)
            - factor * contract("sua,xiu,tia->stx", ov0_f, rco, cv0)
            + eta * contract("siu,xub,tib->stx", co0_f, rov, cv1)
            + eta * contract("sib,xbu,tiu->stx", cv1_f, rvo, co0)
            + eta * contract("sua,xiu,tia->stx", ov0_f, rco, cv1)
            + eta * contract("sia,xui,tua->stx", cv1_f, roc, ov0)
        )

        if include_ground_dipole:
            gs = _molecular_ground_dipole(mf)
            for i in range(nstates):
                tdm[i, i] += gs
        return np.asarray(tdm)

    def transition_dipole_matrix(self, include_ground_dipole=False):
        """Excited-state to excited-state transition dipole matrix in a.u."""
        if not self.type_u:
            return self._transition_dipole_matrix_restricted(include_ground_dipole=include_ground_dipole)

        # have vertified against pyscf for UKS
        ints_aa, ints_bb, ctx, mf = self._dipole_mo_blocks()
        occ_a = _asnumpy(ctx.occidx_a)
        vir_a = _asnumpy(ctx.viridx_a)
        occ_b = _asnumpy(ctx.occidx_b)
        vir_b = _asnumpy(ctx.viridx_b)
        r_oo_a = ints_aa[:, occ_a][:, :, occ_a]
        r_vv_a = ints_aa[:, vir_a][:, :, vir_a]
        r_oo_b = ints_bb[:, occ_b][:, :, occ_b]
        r_vv_b = ints_bb[:, vir_b][:, :, vir_b]

        vectors = _asnumpy(self.v)
        nstates = min(self.nstates, vectors.shape[1])
        amps_a, amps_b = self._full_spin_conserving_amplitudes(vectors[:, :nstates])
        tdm = np.zeros((nstates, nstates, 3))
        for i, (a0, b0) in enumerate(zip(amps_a, amps_b)):
            for j, (a1, b1) in enumerate(zip(amps_a, amps_b)):
                tdm[i, j] = (
                    np.einsum("ia,xab,ib->x", a0, r_vv_a, a1, optimize=True)
                    - np.einsum("ia,xji,ja->x", a0, r_oo_a, a1, optimize=True)
                    + np.einsum("ia,xab,ib->x", b0, r_vv_b, b1, optimize=True)
                    - np.einsum("ia,xji,ja->x", b0, r_oo_b, b1, optimize=True)
                )
        if include_ground_dipole:
            gs = _molecular_ground_dipole(mf)
            for i in range(nstates):
                tdm[i, i] += gs
        return tdm

    def transition_dipole_array(self, nstates=7, include_ground_dipole=True):
        """Return transition dipoles in the ground/excited-state array layout, just for debug."""
        nstates = int(nstates)
        if nstates < 1:
            raise ValueError("nstates must be a positive integer")

        tdm = np.asarray(
            self.transition_dipole_matrix(
                include_ground_dipole=include_ground_dipole
            )
        )
        ground_tdm = np.asarray(self.transition_dipoles_ground())
        n_excited = min(nstates - 1, tdm.shape[0], ground_tdm.shape[0])
        result = np.zeros(
            (nstates, nstates, 3),
            dtype=np.result_type(tdm, ground_tdm),
        )
        result[0, 1:n_excited + 1, :] = ground_tdm[:n_excited]
        upper = np.triu(np.ones((n_excited, n_excited), dtype=bool), k=1)
        result[1:n_excited + 1, 1:n_excited + 1, :] = (
            tdm[:n_excited, :n_excited, :] * upper[:, :, None]
        )
        return result

    def calculate_TDM(self, include_ground_dipole=True):
        gs_tdm = self.transition_dipoles_ground()
        gs_osc = self.osc_str()
        print("Ground state to Excited state transition dipole moments(a.u.)")
        print("State      X        Y        Z      OSC.")
        for i, dip in enumerate(gs_tdm):
            print(
                f" {i + 1:2d}     |GS>    "
                f"{dip[0]:>8.4f} {dip[1]:>8.4f} {dip[2]:>8.4f}  {gs_osc[i]:>8.4f} "
            )

        tdm = self.transition_dipole_matrix(include_ground_dipole=include_ground_dipole)
        energies = _asnumpy(self.e)[:tdm.shape[0]]
        osc = (2.0 / 3.0) * (
            (energies[:, None] - energies[None, :]) * np.einsum("ijx,ijx->ij", tdm, tdm)
        )
        print("\nExcited state to Excited state transition dipole moments(a.u.)")
        print("StateL StateR      X        Y        Z      f(L<-R)")
        for i in range(tdm.shape[0]):
            for j in range(tdm.shape[1]):
                print(
                    f" {i + 1:2d}     {j + 1:2d}    "
                    f"{tdm[i, j, 0]:>8.4f} {tdm[i, j, 1]:>8.4f} {tdm[i, j, 2]:>8.4f}  "
                    f"{osc[i, j]:>8.4f} "
                )
        return {"gs_tdm": gs_tdm, "gs_osc": gs_osc, "tdm": tdm, "osc": osc}

    analyze_TDM = calculate_TDM

    def kernel(self, nstates=1, save=False, save_file=None, init_space=None):
        self.nstates = nstates
        if self.davidson:
            self.davidson_process(nstates, init_space=init_space)
        else:
            self.get_Amat()
            self._diagonalize_dense(self.A, nstates)
        if save:
            self.save_results(save_file)
        return _asnumpy(self.e[:nstates] * ha2eV), self.v[:, :nstates]

    def analyse(
        self,
        threshold=0.1,
        compute_s2=True,
        analyze_symmetry=False,
        point_group=None,
        symmetry_tol=1.0e-3,
        energy_tol=1.0e-5,
        projection_backend="auto",
        symmetry_kwargs=None,
    ):
        energies = _asnumpy(self.e) * ha2eV
        block0, block1, block2, block3 = self._split_analysis_vectors()
        self.dS2 = _asnumpy(self.deltaS2()) if compute_s2 else None
        symmetry_labels = None
        if analyze_symmetry:
            from ..utils.symmetry import analyze_state_symmetry_labels

            kwargs = {} if symmetry_kwargs is None else dict(symmetry_kwargs)
            symmetry_labels, self.symmetry_report = analyze_state_symmetry_labels(
                self,
                point_group=point_group,
                symmetry_tol=symmetry_tol,
                energy_tol=energy_tol,
                projection_backend=projection_backend,
                active_roots=range(min(self.nstates, block0.shape[0])),
                **kwargs,
            )
        for istate in range(min(self.nstates, block0.shape[0])):
            ds2_text = f"{self.dS2[istate]:8.4f}" if self.dS2 is not None else "     n/a"
            irrep_text = ""
            if symmetry_labels is not None and istate < len(symmetry_labels):
                irrep_text = f"    irrep:{symmetry_labels[istate]}"
            print(
                f"D{istate + 1}    w:{energies[istate]:10.4f} eV"
                f"    d<S^2>:{ds2_text}"
                f"{irrep_text}"
            )
            if self.type_u:
                labels = (
                    ("CV(aa)", block0[istate], 1, self.nc + self.no + 1),
                    ("OV(aa)", block1[istate], self.nc + 1, self.nc + self.no + 1),
                    ("CO(bb)", block2[istate], 1, self.nc + 1),
                    ("CV(bb)", block3[istate], 1, self.nc + self.no + 1),
                )
            else:
                labels = (
                    ("CV(0)", block0[istate], 1, self.nc + self.no + 1),
                    ("CO(0)", block1[istate], 1, self.nc + 1),
                    ("OV(0)", block2[istate], self.nc + 1, self.nc + self.no + 1),
                    ("CV(1)", block3[istate], 1, self.nc + self.no + 1),
                )
            for label, block, occ_offset, vir_offset in labels:
                for occ, vir in zip(*np.where(abs(block) > threshold)):
                    amp = block[occ, vir]
                    print(
                        f"    {label} {occ + occ_offset:3d} -> {vir + vir_offset:3d}    "
                        f"c_i: {amp:8.5f}    Per: {100 * amp**2:5.2f}%"
                    )
            print(" ")
        return self.dS2
