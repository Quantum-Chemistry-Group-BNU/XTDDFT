from types import SimpleNamespace

import math
import numpy as np
from pyscf import ao2mo, scf, lib, dft
from pyscf.dft import numint2c
from pyscf.dft.gen_grid import NBINS
from pyscf.dft.numint import _dot_ao_ao_sparse,_scale_ao_sparse,_tau_dot_sparse
from pyscf.pbc import scf as pbc_scf
from pyscf.pbc.dft import numint2c as pbc_numint2c

from XTDDFT.base import (
    XTDDFT_base,
    _build_initial_guess_from_gaps,
    _ensure_gamma_df,
    _get_gamma_kpt,
    _get_k,
    _get_hcore,
    _get_veff,
    AldA0,
    _is_gpu_mf,
    _is_pbc_mf,
    _make_spinflip_problem,
    _make_spinflip_vind,
    _run_davidson,
    _spinflip_gaps,
    mf_info,
    _as_cpu_mf,
    _as_cpu_ctx,
    _system,
    _is_ks_mf,
    _make_reference_dm,
    _response_max_memory,
    cache_xc_kernel_sf_mc,
)
from utils.backend import _asnumpy, contract, xp
from utils.unit import ha2eV

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger(__name__)

def add_hf_a_a2b(a_a2b, mf, orbo_a, orbv_b, nocc_a, nvir_b, hyb=1, omega=None):
    # 考虑SF_TDA_UP时，仅有CV激发，因此，只需要考虑这个空间; K矩阵中含有的精确交换部分
    if abs(hyb) < 1e-14 or nocc_a == 0 or nvir_b == 0:
        return a_a2b

    if _is_pbc_mf(mf):
        kpt = _get_gamma_kpt(mf)
        if omega is None or abs(omega) < 1e-14:
            eri_mo = mf.with_df.ao2mo([orbo_a, orbo_a, orbv_b, orbv_b], kpt, compact=False)
        else:
            with mf.with_df.range_coulomb(omega) as rsh_df:
                eri_mo = rsh_df.ao2mo([orbo_a, orbo_a, orbv_b, orbv_b], kpt, compact=False)
    else:
        if omega is not None and abs(omega) >= 1e-14:
            raise NotImplementedError("Range-separated molecular HF exchange is not implemented in SF_TDA_up.")
        eri_mo = ao2mo.general(mf.mol, [orbo_a, orbo_a, orbv_b, orbv_b], compact=False)

    eri_mo = np.asarray(eri_mo).reshape(nocc_a,nocc_a,nvir_b,nvir_b)
    a_a2b -= contract('ijba->iajb', eri_mo) * hyb
    return a_a2b

def construct_xc_a2b(ao, orbo_a, orbv_b, fxc_ab):
    rho_o_a = contract('rp,pi->ri', ao, orbo_a)
    rho_v_b = contract('rp,pi->ri', ao, orbv_b)
    rho_ov_a2b = contract('ri,ra->ria', rho_o_a, rho_v_b)
    w_ov = contract('ria,r->ria', rho_ov_a2b, fxc_ab)
    iajb = contract('ria,rjb->iajb', rho_ov_a2b, w_ov)
    return iajb

def _pair_hessian_block_a2b(occ_fock_a, vir_fock_b, tensor_block):
    nocc_a = occ_fock_a.shape[0]
    nvir_b = vir_fock_b.shape[0]
    return (
        contract('ij,ab->iajb', np.eye(nocc_a), vir_fock_b)
        - contract('ji,ab->iajb', occ_fock_a, np.eye(nvir_b))
        + tensor_block.reshape(nocc_a, nvir_b, nocc_a, nvir_b)
    )

def _convert_a2b_to_cv_co_ov_oo(amat, nc, no, nv):
    """Convert flattened (C,O)x(O,V) a->b order to CV|CO|OV|OO order."""
    nvir_b = no + nv
    idx = []
    idx.extend(i * nvir_b + a for i in range(nc) for a in range(no, nvir_b))  # CV
    idx.extend(i * nvir_b + a for i in range(nc) for a in range(no))  # CO
    idx.extend(i * nvir_b + a for i in range(nc, nc + no) for a in range(no, nvir_b))   # OV
    idx.extend(i * nvir_b + a for i in range(nc, nc + no) for a in range(no))  # OO

    ordered = np.asarray(amat)[np.ix_(idx, idx)]
    return ordered

def _make_rohf_reference_mf(mf):
    if _is_pbc_mf(mf):
        hf = pbc_scf.ROHF(mf.cell)
        hf.kpt = _get_gamma_kpt(mf)
    else:
        hf = scf.ROHF(mf.mol)
        if getattr(mf, "with_x2c", None) is not None and hasattr(hf, "x2c"):
            hf = hf.x2c()

    for name in ("with_df", "exxdiv", "max_memory", "verbose", "stdout"):
        if hasattr(mf, name):
            try:
                setattr(hf, name, getattr(mf, name))
            except AttributeError:
                pass
    return hf

def _ao2mo_full_gamma(mf, mo_coeff):
    mo_coeff = _asnumpy(mo_coeff)
    nmo = mo_coeff.shape[1]
    if _is_pbc_mf(mf):
        _ensure_gamma_df(mf)
        kpt = _get_gamma_kpt(mf)
        try:
            eri = mf.with_df.ao2mo([mo_coeff] * 4, kpt, compact=False)
        except TypeError:
            eri = mf.with_df.ao2mo([mo_coeff] * 4, [kpt] * 4, compact=False)
    else:
        eri = ao2mo.general(mf.mol, [mo_coeff] * 4, compact=False)
    return _asnumpy(eri).reshape(nmo, nmo, nmo, nmo)

def _as_spin_potential_cpu(vhf):
    vhf = _asnumpy(vhf)
    if vhf.ndim == 3 and vhf.shape[0] == 1:
        vhf = vhf[0]
    if vhf.ndim == 4 and vhf.shape[1] == 1:
        vhf = vhf[:, 0]
    if vhf.ndim == 2:
        vhf = np.stack([vhf, vhf])
    return vhf

def _get_mo_fock_cpu(mf, mo_coeff, mo_occ):
    mo_coeff = _asnumpy(mo_coeff)
    mo_occ = _asnumpy(mo_occ)
    dm = np.asarray([
        (mo_coeff[s] * mo_occ[s]) @ mo_coeff[s].conj().T
        for s in range(2)
    ])
    vhf = _as_spin_potential_cpu(_get_veff(mf, dm))
    h1e = _asnumpy(_get_hcore(mf))
    focka_mo = mo_coeff[0].conj().T @ (h1e + vhf[0]) @ mo_coeff[0]
    fockb_mo = mo_coeff[1].conj().T @ (h1e + vhf[1]) @ mo_coeff[1]
    return focka_mo, fockb_mo

def _iter_block_data_cpu(mf, ni, ao_deriv, max_memory):
    if _is_pbc_mf(mf):
        for ao, ao_k2, mask, weight, coords in ni.block_loop(
            mf.cell, mf.grids, mf.cell.nao_nr(), ao_deriv,
            _get_gamma_kpt(mf), None, max_memory,
        ):
            yield ao, mask, weight, coords
    else:
        for ao, mask, weight, coords in ni.block_loop(
            mf.mol, mf.grids, mf.mol.nao_nr(), ao_deriv, max_memory
        ):
            yield ao, mask, weight, coords

class XSF_TDA_down(XTDDFT_base): # just for ROKS
    def __init__(self, mf, method, davidson=True, SA = None, davidson_backend="cpu"):
        """SA=0: SF-TDA
           SA=1: only add diagonal block for dA
           SA=2: add all dA except for OO block
           SA=3: full dA
        """
        davidson_backend = davidson_backend.lower()
        if davidson_backend not in ("cpu", "gpu", "auto"):
            raise ValueError("davidson_backend must be 'cpu', 'gpu', or 'auto'")
        super().__init__(mf, method, davidson=davidson)
        self.isf = -1
        self.type_u = True
        self.davidson_backend = "cpu" if davidson_backend == "auto" else davidson_backend
        if SA is None:
            self.SA = 3
        else:
            self.SA = SA
        spin_mf = mf if hasattr(mf, "spin_square") else _as_cpu_mf(mf)
        _,dsp1 = spin_mf.spin_square()
        self.ground_s = (dsp1-1)/2
    
    def get_Amat_ALDA0(self):
        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)
        ni = mf._numint if hasattr(mf, "_numint") else self.ni

        a_a2b = np.zeros((ctx.nocc_a, ctx.nvir_b, ctx.nocc_a, ctx.nvir_b))
        if self.mfxctype is not None:  # 构建K矩阵
            if self.hyb != 0:
                a_a2b = add_hf_a_a2b(a_a2b, mf, ctx.orbo_a, ctx.orbv_b, ctx.nocc_a, ctx.nvir_b, self.hyb)
            # 范围分离泛函的处理
            if abs(self.omega) > 1e-10:
                a_a2b = add_hf_a_a2b(
                    a_a2b, mf, ctx.orbo_a, ctx.orbv_b, ctx.nocc_a, ctx.nvir_b,
                    self.alpha - self.hyb, omega=self.omega
                )
            dm0 = _make_reference_dm(mf, ctx.mo_occ)
            make_rho = ni._gen_rho_evaluator(_system(mf), dm0, hermi=0, with_lapl=False)[0]
            max_memory = _response_max_memory(mf, None)

            if self.xctype == 'LDA' and not getattr(self, "collinear", False):
                ao_deriv = 0
                for ao, mask, weight, coords in _iter_block_data_cpu(mf, ni, ao_deriv, max_memory):
                    rho0a = make_rho(0, ao, mask, self.xctype)
                    rho0b = make_rho(1, ao, mask, self.xctype)
                    rho = (rho0a, rho0b)
                    fxc_ab = AldA0(ni, mf, rho, weight, self.xctype, omega=self.omega)
                    a_a2b += construct_xc_a2b(ao, ctx.orbo_a, ctx.orbv_b, fxc_ab)

            elif self.xctype == 'GGA' and not getattr(self, "collinear", False):  # 进行简化
                ao_deriv = 0
                for ao, mask, weight, coords in _iter_block_data_cpu(mf, ni, ao_deriv, max_memory):
                    # 这里只需要 density，不需要 gradient
                    rho0a = make_rho(0, ao, mask, 'LDA')
                    rho0b = make_rho(1, ao, mask, 'LDA')
                    # 为 GGA eval_xc_eff 构造 shape = (4, ngrids) 的输入
                    rha = np.zeros((4, rho0a.size))
                    rhb = np.zeros((4, rho0b.size))
                    rha[0] = rho0a
                    rhb[0] = rho0b
                    fxc_ab = AldA0(ni, mf, (rha, rhb), weight, self.xctype, omega=self.omega)
                    a_a2b += construct_xc_a2b(ao, ctx.orbo_a, ctx.orbv_b, fxc_ab)

        else:
            a_a2b = add_hf_a_a2b(a_a2b, mf, ctx.orbo_a, ctx.orbv_b, ctx.nocc_a, ctx.nvir_b, hyb=1)

        focka_mo, fockb_mo = _get_mo_fock_cpu(mf, ctx.mo_coeff, ctx.mo_occ)

        nc = self.nc
        no = self.no
        nv = self.nv
        dim = (nc+no)*(nv+no)
        logger.info(f'The dims of A matrix: {dim}')
        amat = _pair_hessian_block_a2b(
            focka_mo[:ctx.nocc_a, :ctx.nocc_a],
            fockb_mo[ctx.nocc_b:, ctx.nocc_b:],
            a_a2b,
        ).reshape(dim, dim)
        Amat = _convert_a2b_to_cv_co_ov_oo(amat, nc, no, nv)
        del a_a2b
        self.sf_tda_A = np.asarray(Amat)
        return self.sf_tda_A
    
    def get_Amat_MCOL(self, collinear_samples=30):
        r'''A and B matrices for TDDFT response function.

        A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
        B[i,a,j,b] = (ia||jb)

        Spin symmetry is not considered in the returned A, B lists.
        List A has two items: (A_baba, A_abab).
        List B has two items: (B_baab, B_abba).
        '''
        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)

        a_a2b = np.zeros((ctx.nocc_a, ctx.nvir_b, ctx.nocc_a, ctx.nvir_b))

        if self.mfxctype is not None:
            ni0 = mf._numint
            ni = pbc_numint2c.NumInt2C() if _is_pbc_mf(mf) else numint2c.NumInt2C()
            ni.collinear = 'mcol'
            ni.collinear_samples = collinear_samples

            if self.hyb != 0:
                a_a2b = add_hf_a_a2b(a_a2b, mf, ctx.orbo_a, ctx.orbv_b, ctx.nocc_a, ctx.nvir_b, self.hyb)
            if abs(self.omega) > 1e-10:
                a_a2b = add_hf_a_a2b(a_a2b, mf, ctx.orbo_a, ctx.orbv_b, ctx.nocc_a, ctx.nvir_b,
                                     self.alpha - self.hyb, omega=self.omega)

            max_memory = _response_max_memory(mf, None)

            if collinear_samples >= 0:
                # it should be optimized, which is the disadvantage of mc approach.
                fxc = cache_xc_kernel_sf_mc(
                    ni, mf, _system(mf), mf.grids, mf.xc,
                    ctx.mo_coeff, ctx.mo_occ, deriv=2, spin=1,
                    max_memory=max_memory,
                )[2]
                p0, p1 = 0, 0
                if self.xctype == 'LDA':
                    ao_deriv = 0
                    for ao, mask, weight, coords in _iter_block_data_cpu(mf, ni0, ao_deriv, max_memory):
                        p0 = p1
                        p1 += weight.shape[0]
                        wfxc = fxc[0, 0][..., p0:p1] * weight
                        rho_o_a = contract('rp,pi->ri', ao, ctx.orbo_a)
                        rho_v_b = contract('rp,pi->ri', ao, ctx.orbv_b)
                        rho_ov_a2b = contract('ri,ra->ria', rho_o_a, rho_v_b)
                        w_ov = contract('ria,r->ria', rho_ov_a2b, wfxc * 2.0)
                        iajb = contract('ria,rjb->iajb', rho_ov_a2b, w_ov)
                        a_a2b += iajb
                elif self.xctype == 'GGA':
                    ao_deriv = 1
                    for ao, mask, weight, coords in _iter_block_data_cpu(mf, ni, ao_deriv, max_memory):
                        p0 = p1
                        p1 += weight.shape[0]
                        wfxc = fxc[..., p0:p1] * weight
                        rho_o_a = contract('xrp,pi->xri', ao, ctx.orbo_a)
                        rho_v_b = contract('xrp,pi->xri', ao, ctx.orbv_b)
                        rho_ov_a2b = contract('xri,ra->xria', rho_o_a, rho_v_b[0])
                        rho_ov_a2b[1:4] += contract('ri,xra->xria', rho_o_a[0], rho_v_b[1:4])
                        w_ov = contract('xyr,xria->yria', wfxc * 2.0, rho_ov_a2b)
                        iajb = contract('yria,yrjb->iajb', w_ov, rho_ov_a2b)
                        a_a2b += iajb
                elif self.xctype == 'MGGA':
                    ao_deriv = 1
                    for ao, mask, weight, coords in _iter_block_data_cpu(mf, ni, ao_deriv, max_memory):
                        p0 = p1
                        p1 += weight.shape[0]
                        wfxc = fxc[..., p0:p1] * weight
                        rho_o_a = contract('xrp,pi->xri', ao, ctx.orbo_a)
                        rho_v_b = contract('xrp,pi->xri', ao, ctx.orbv_b)
                        rho_ov_a2b = contract('xri,ra->xria', rho_o_a, rho_v_b[0])
                        rho_ov_a2b[1:4] += contract('ri,xra->xria', rho_o_a[0], rho_v_b[1:4])
                        tau_ov_a2b = contract('xri,xra->ria', rho_o_a[1:4], rho_v_b[1:4]) * 0.5
                        rho_ov_a2b = np.vstack([rho_ov_a2b, tau_ov_a2b[np.newaxis]])
                        w_ov = contract('xyr,xria->yria', wfxc * 2.0, rho_ov_a2b)
                        iajb = contract('yria,yrjb->iajb', w_ov, rho_ov_a2b)
                        a_a2b += iajb
                elif self.xctype == 'HF':
                    pass
                elif self.xctype == 'NLC':
                    raise NotImplementedError('NLC functional is not supported here.')
                else:
                    raise NotImplementedError(f'Unsupported xctype: {self.xctype}')
        else:
            a_a2b = add_hf_a_a2b(a_a2b, mf, ctx.orbo_a, ctx.orbv_b, ctx.nocc_a, ctx.nvir_b, hyb=1)

        focka_mo, fockb_mo = _get_mo_fock_cpu(mf, ctx.mo_coeff, ctx.mo_occ)
        dim = (self.nc + self.no) * (self.nv + self.no)
        amat = _pair_hessian_block_a2b(
            focka_mo[:ctx.nocc_a, :ctx.nocc_a],
            fockb_mo[ctx.nocc_b:, ctx.nocc_b:],
            a_a2b,
        ).reshape(dim, dim)
        Amat = _convert_a2b_to_cv_co_ov_oo(amat, self.nc, self.no, self.nv)
        self.sf_tda_A = np.asarray(Amat)
        return self.sf_tda_A

    def get_Amat(self, SA=None, foo=1.0, fglobal=1.0, d_lda=None, fit=True):
        if self.method == 0:
            self.get_Amat_ALDA0()
        elif self.method == 1:
            self.get_Amat_MCOL()
        else:
            raise NotImplementedError(f"Unsupported method={self.method!r}.")

        if SA is None:
            SA = self.SA

        # Delta A is only defined for restricted open-shell references.
        if _asnumpy(self.mf.mo_coeff).ndim == 3:
            self.A = self.sf_tda_A
            return self.A

        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)

        nc, no, nv = ctx.nc, ctx.no, ctx.nv
        dim1 = nc * nv
        dim2 = dim1 + nc * no
        dim3 = dim2 + no * nv
        si = 1.0e10 if SA == 0 else no / 2
        if SA == 0:
            logger.info("Perform SF-TDA calculating without finite-spin Delta A.")
        elif abs(2 * si - 1) < 1e-12:
            raise ValueError(
                "Table-3 Delta A contains 2*S_i-1 denominators and is singular for S_i=1/2. "
                "Use SA=0 for doublet references."
            )

        Amat = np.zeros_like(_asnumpy(self.sf_tda_A))
        hf = _make_rohf_reference_mf(mf)
        fockA_hf, fockB_hf = _get_mo_fock_cpu(hf, ctx.mo_coeff, ctx.mo_occ)
        fockS = (fockB_hf - fockA_hf) * 0.5
        eri = _ao2mo_full_gamma(mf, ctx.mo_coeff[0])

        iden_C = np.identity(nc)
        iden_O = np.identity(no)
        iden_V = np.identity(nv)

        fockS_C = fockS[:nc, :nc]
        fockS_V = fockS[nc + no:, nc + no:]
        fockS_CV = fockS[:nc, nc + no:]

        # Delta A blocks follow Table 3 for spin-down excitations.
        Amat[:dim1, :dim1] += (
            contract('ij,ab->iajb', iden_C, fockS_V).reshape(nc * nv, nc * nv)
            + contract('ji,ab->iajb', fockS_C, iden_V).reshape(nc * nv, nc * nv)
        ) / si
        Amat[dim1:dim2, dim1:dim2] += (
            2.0 * contract('ji,uv->iujv', fockS_C, iden_O)
            - contract('uijv->iujv', eri[nc:nc + no, :nc, :nc, nc:nc + no])
        ).reshape(nc * no, nc * no) / (2 * si - 1)
        Amat[dim2:dim3, dim2:dim3] += (
            2.0 * contract('uv,ab->uavb', iden_O, fockS_V)
            - contract('auvb->uavb', eri[nc + no:, nc:nc + no, nc:nc + no, nc + no:])
        ).reshape(no * nv, no * nv) / (2 * si - 1)

        if SA > 1:
            cv_scale = np.sqrt(1 + 1 / (2 * si)) - 1
            tmp_CV_CO = cv_scale * (
                contract('ij,av->iajv', iden_C, fockB_hf[nc + no:, nc:nc + no])
                - contract('avji->iajv', eri[nc + no:, nc:nc + no, :nc, :nc])
            ).reshape(nc * nv, nc * no)
            Amat[:dim1, dim1:dim2] += tmp_CV_CO
            Amat[dim1:dim2, :dim1] += tmp_CV_CO.T

            tmp_CV_OV = cv_scale * (
                -contract('vi,ab->iavb', fockA_hf[nc:nc + no, :nc], iden_V)
                - contract('abvi->iavb', eri[nc + no:, nc + no:, nc:nc + no, :nc])
            ).reshape(nc * nv, no * nv)
            Amat[:dim1, dim2:dim3] += tmp_CV_OV
            Amat[dim2:dim3, :dim1] += tmp_CV_OV.T

            tmp_CO_OV = (
                contract('uivb->iuvb', eri[nc:nc + no, :nc, nc:nc + no, nc + no:])
                - contract('ubvi->iuvb', eri[nc:nc + no, nc + no:, nc:nc + no, :nc])
            ).reshape(nc * no, no * nv) / (2 * si - 1)
            Amat[dim1:dim2, dim2:dim3] += tmp_CO_OV
            Amat[dim2:dim3, dim1:dim2] += tmp_CO_OV.T

        if SA > 2:
            factor = np.sqrt((2 * si + 1) / (2 * si - 1))
            tmp_CV_OO = (
                -(factor - 1) * contract(
                    'avwi->iawv', eri[nc + no:, nc:nc + no, nc:nc + no, :nc]
                ).reshape(nc * nv, no * no)
                + (factor / si) * contract('ia,wv->iawv', fockS_CV, iden_O).reshape(nc * nv, no * no)
            )
            Amat[:dim1, dim3:] += foo * tmp_CV_OO
            Amat[dim3:, :dim1] += foo * tmp_CV_OO.T

            tmp_CO_OO = (
                (np.sqrt(2 * si / (2 * si - 1)) - 1) * (
                    -contract('iw,uv->iuwv', fockA_hf[:nc, nc:nc + no], iden_O).reshape(nc * no, no * no)
                    - contract('uvwi->iuwv', eri[nc:nc + no, nc:nc + no, nc:nc + no, :nc]).reshape(nc * no, no * no)
                )
                + contract('iu,wv->iuwv', fockB_hf[:nc, nc:nc + no], iden_O).reshape(nc * no, no * no)
                / np.sqrt(2 * si * (2 * si - 1))
            )
            Amat[dim1:dim2, dim3:] += foo * tmp_CO_OO
            Amat[dim3:, dim1:dim2] += foo * tmp_CO_OO.T

            tmp_OV_OO = (
                (np.sqrt(2 * si / (2 * si - 1)) - 1) * (
                    contract('wu,av->uawv', iden_O, fockB_hf[nc + no:, nc:nc + no]).reshape(no * nv, no * no)
                    - contract('avwu->uawv', eri[nc + no:, nc:nc + no, nc:nc + no, nc:nc + no]).reshape(no * nv, no * no)
                )
                - contract('au,wv->uawv', fockA_hf[nc + no:, nc:nc + no], iden_O).reshape(no * nv, no * no)
                / np.sqrt(2 * si * (2 * si - 1))
            )
            Amat[dim2:dim3, dim3:] += foo * tmp_OV_OO
            Amat[dim3:, dim2:dim3] += foo * tmp_OV_OO.T

        A = _asnumpy(self.sf_tda_A) + fglobal * Amat

        self.A = np.asarray(A)
        return self.A

    def _default_fglobal(self, d_lda=0.3, fit=True):
        if abs(self.omega) < 1e-14:
            cx = self.hyb
        else:
            cx = self.hyb + (self.alpha - self.hyb) * math.erf(self.omega)
        fglobal = (1 - d_lda) * cx + d_lda
        if self.method == 1 and fit:
            fglobal *= 4 * (cx - 0.5) ** 2
        return fglobal

    def get_vect(self): # construct Vmat N*(N-1)
        tmp_v = np.zeros((self.no-1,self.no)) #(self.no-1,self.no)
        for i in range(1,self.no): # 1->v1 2->v2 3->v3 ...
            factor = 1/np.sqrt((self.no-i+1)*(self.no-i))
            tmp = [self.no-i] + [-1]*(self.no-i) #(N-i,-1,-1,-1 ...)
            tmp_v[i-1][i-1:] = np.array(tmp)*factor
        self.vect = tmp_v.T # N(N-1)
        #print('v ',v)
        vects = np.eye(self.no*self.no)
        vects = vects[:,:-1] # no*no*(no*no-1)
        index = [0]
        for i in range(1,self.no):
            index.append(i*(self.no+1))
        #print('index ',index)
        for i in range(self.vect.shape[1]):
            vects[0::self.no+1, index[i]] = self.vect[:,i]
        #print(vect)
        return vects

    def remove(self):
        # remove sf=si state
        dim3 = self.nc*self.nv + self.nc*self.no + self.no*self.nv
        A0 = _asnumpy(self.A)
        dim = A0.shape[0]
        self.vects = self.get_vect() # (no*no,no*no-1)
        A = np.zeros((dim-1,dim-1))
        A[:dim3,:dim3] = A0[:dim3,:dim3]
        A[:dim3,dim3:] = A0[:dim3,dim3:] @ self.vects
        A[dim3:,:dim3] = self.vects.T @ A0[dim3:,:dim3]
        A[dim3:,dim3:] = self.vects.T @ A0[dim3:,dim3:] @ self.vects
        return A

    def frozen_A(self, frozen):
        f = 1 if not isinstance(frozen, int) or frozen == 0 else frozen
        A = _asnumpy(self.A)
        minus_cv = A[f * self.nv:, f * self.nv:]
        dim = minus_cv.shape[0]
        kept = np.r_[0:(self.nc - f) * self.nv, (self.nc - f) * self.nv + f * self.no:dim]
        return minus_cv[np.ix_(kept, kept)]

    def _diagonalize_dense(self, amat, nstates):
        mat = xp.asarray(amat)
        e, v = xp.linalg.eigh(mat)
        self.e = xp.asarray(e[:nstates])
        self.v = xp.asarray(v[:, :nstates])
        return self.e, self.v
    
    def gen_tda_operation_sf(self)

    def kernel(self, nstates=1, remove=None, frozen=None, foo=1.0, d_lda=0.3, fglobal=None, fit=True):
        self.re = (_asnumpy(self.mf.mo_coeff).ndim != 3) if remove is None else bool(remove)
        nov = (self.nc + self.no) * (self.no + self.nv)
        self.nstates = min(nstates, nov)
        if fglobal is None:
            fglobal = self._default_fglobal(d_lda=d_lda, fit=fit)
        self.fglobal = fglobal

        self.A = self.get_Amat(foo=foo, fglobal=fglobal)
        if self.re:
            logger.info(f"fglobal: {fglobal}")
            self.A = self.remove()
        elif frozen is not None:
            self.A = self.frozen_A(frozen)

        self._diagonalize_dense(self.A, self.nstates)
        return _asnumpy(self.e[:self.nstates] * ha2eV), self.v[:, :self.nstates]
