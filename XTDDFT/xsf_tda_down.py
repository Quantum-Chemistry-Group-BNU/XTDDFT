import functools
from types import SimpleNamespace

import numpy as np
from pyscf import ao2mo, scf, lib, dft
from pyscf.dft import numint,numint2c,xc_deriv
from pyscf.dft.gen_grid import NBINS
from pyscf.dft.numint import _dot_ao_ao_sparse,_scale_ao_sparse,_tau_dot_sparse
from pyscf.pbc.dft import numint as pbc_numint
from pyscf.pbc.dft import numint2c as pbc_numint2c
from opt_einsum import contract

from XTDDFT.base import (
    XTDDFT_base,
    _build_initial_guess_from_gaps,
    _ensure_gamma_df,
    _get_gamma_kpt,
    _get_k,
    _get_mo_fock,
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
    _iter_block_data,
    _iter_ao_blocks,
    _xc_ao_deriv,
    cache_xc_kernel_sf,
    cache_xc_kernel_sf_mc,
    _cache_xc_kernel_sf_gpu_mol,
    _cache_xc_kernel_sf_gpu_pbc,
    _cache_xc_kernel_sf_mc_gpu_mol
)
from utils.backend import _asnumpy, backend, require_cupy, set_backend, xp
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
    a_a2b -= contract('ijba->iajb', eri_mo, optimize=True) * hyb
    return a_a2b

def construct_xc_a2b(ao, orbo_a, orbv_b, fxc_ab):
    rho_o_a = contract('rp,pi->ri', ao, orbo_a, optimize=True)
    rho_v_b = contract('rp,pi->ri', ao, orbv_b, optimize=True)
    rho_ov_a2b = contract('ri,ra->ria', rho_o_a, rho_v_b, optimize=True)
    w_ov = contract('ria,r->ria', rho_ov_a2b, fxc_ab, optimize=True)
    iajb = contract('ria,rjb->iajb', rho_ov_a2b, w_ov, optimize=True)
    return iajb

def _pair_hessian_block_a2b(occ_fock_a, vir_fock_b, tensor_block):
    nocc_a = occ_fock_a.shape[0]
    nvir_b = vir_fock_b.shape[0]
    return (
        contract('ij,ab->iajb', np.eye(nocc_a), vir_fock_b, optimize=True)
        - contract('ji,ab->iajb', occ_fock_a, np.eye(nvir_b), optimize=True)
        + tensor_block.reshape(nocc_a, nvir_b, nocc_a, nvir_b)
    )

def _convert_a2b_to_cv_co_ov_oo(amat, nc, no, nv):
    """Convert flattened (C,O)x(O,V) a->b order to CV|CO|OV|OO order."""
    nvir_b = no + nv
    idx = []
    idx.extend(i * nvir_b + a for i in range(nc) for a in range(no, nvir_b))
    idx.extend(i * nvir_b + a for i in range(nc) for a in range(no))
    idx.extend(i * nvir_b + a for i in range(nc, nc + no) for a in range(no, nvir_b))
    idx.extend(i * nvir_b + a for i in range(nc, nc + no) for a in range(no))

    ordered = np.asarray(amat)[np.ix_(idx, idx)]
    return np.triu(ordered) + np.triu(ordered, k=1).T

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
        _,dsp1 = mf.spin_square()
        self.ground_s = (dsp1-1)/2
    
    def get_Amat_ALDA0(self):
        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)

        mode = backend.mode
        set_backend("cpu")

        try:
            a_a2b = np.zeros((ctx.nocc_a, ctx.nvir_b, ctx.nocc_a, ctx.nvir_b))
            if self.mfxctype is not None:
                if self.hyb != 0:
                    a_a2b = add_hf_a_a2b(a_a2b, mf, ctx.orbo_a, ctx.orbv_b, ctx.nocc_a, ctx.nvir_b, self.hyb)
                # 范围分离泛函的处理
                if abs(self.omega) > 1e-10:
                    a_a2b = add_hf_a_a2b(
                        a_a2b, mf, ctx.orbo_a, ctx.orbv_b, ctx.nocc_a, ctx.nvir_b,
                        self.alpha - self.hyb, omega=self.omega
                    )
                dm0 = _make_reference_dm(mf, ctx.mo_occ)
                make_rho = self.ni._gen_rho_evaluator(self.mf.mol, dm0, hermi=0, with_lapl=False)[0]
                max_memory = _response_max_memory(mf, None)
            
                if self.xctype == 'LDA' and not getattr(self, "collinear", False):
                    ao_deriv = 0
                    for ao, mask, weight, coords in _iter_block_data(mf, self.ni, ao_deriv, max_memory):
                        rho0a = make_rho(0, ao, mask, self.xctype)
                        rho0b = make_rho(1, ao, mask, self.xctype)
                        rho = (rho0a, rho0b)
                        vxc = self.ni.eval_xc_eff(mf.xc, rho, deriv=1, omega=self.omega, xctype=self.xctype)[1]
                        vxc_a = vxc[0, 0] * weight
                        vxc_b = vxc[1, 0] * weight
                        fxc_ab = (vxc_a - vxc_b) / (rho0a - rho0b + 1e-9)
                        a_a2b += construct_xc_a2b(ao, ctx.orbo_a, ctx.orbv_b, fxc_ab)
            
                elif self.xctype == 'GGA' and not getattr(self, "collinear", False):  # 进行简化
                    ao_deriv = 0
                    for ao, mask, weight, coords in _iter_block_data(mf, self.ni, ao_deriv, max_memory):
                        # 这里只需要 density，不需要 gradient
                        rho0a = make_rho(0, ao, mask, 'LDA')
                        rho0b = make_rho(1, ao, mask, 'LDA')
                        # 为 GGA eval_xc_eff 构造 shape = (4, ngrids) 的输入
                        rha = np.zeros((4, rho0a.size))
                        rhb = np.zeros((4, rho0b.size))
                        rha[0] = rho0a
                        rhb[0] = rho0b
                        vxc = self.ni.eval_xc_eff(mf.xc, (rha, rhb), deriv=1, omega=self.omega, xctype=self.xctype)[1]
                        vxc_a = vxc[0, 0] * weight
                        vxc_b = vxc[1, 0] * weight
                        fxc_ab = (vxc_a - vxc_b) / (rho0a - rho0b + 1e-9)
                        a_a2b += construct_xc_a2b(ao, ctx.orbo_b, ctx.orbv_a, fxc_ab)
            
            else:
                a_a2b = add_hf_a_a2b(a_a2b, mf, ctx.orbo_a, ctx.orbv_b, ctx.nocc_a, ctx.nvir_b, hyb=1)
            
            focka_mo, fockb_mo = (_asnumpy(x) for x in _get_mo_fock(mf, ctx.mo_coeff, ctx.mo_occ))
            
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
        finally:
            set_backend(mode)
        self.sf_tda_A = xp.asarray(Amat)
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
        
        mode = backend.mode
        set_backend("cpu")
        try:
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
                        for ao, mask, weight, coords in _iter_block_data(mf, ni0, ao_deriv, max_memory):
                            p0 = p1
                            p1 += weight.shape[0]
                            wfxc = fxc[0, 0][..., p0:p1] * weight
                            rho_o_a = contract('rp,pi->ri', ao, ctx.orbo_a, optimize=True)
                            rho_v_b = contract('rp,pi->ri', ao, ctx.orbv_b, optimize=True)
                            rho_ov_a2b = contract('ri,ra->ria', rho_o_a, rho_v_b, optimize=True)
                            w_ov = contract('ria,r->ria', rho_ov_a2b, wfxc * 2.0, optimize=True)
                            iajb = contract('ria,rjb->iajb', rho_ov_a2b, w_ov, optimize=True)
                            a_a2b += iajb
                    elif self.xctype == 'GGA':
                        ao_deriv = 1
                        for ao, mask, weight, coords in _iter_block_data(mf, ni, ao_deriv, max_memory):
                            p0 = p1
                            p1 += weight.shape[0]
                            wfxc = fxc[..., p0:p1] * weight
                            rho_o_a = contract('xrp,pi->xri', ao, ctx.orbo_a, optimize=True)
                            rho_v_b = contract('xrp,pi->xri', ao, ctx.orbv_b, optimize=True)
                            rho_ov_a2b = contract('xri,ra->xria', rho_o_a, rho_v_b[0], optimize=True)
                            rho_ov_a2b[1:4] += contract('ri,xra->xria', rho_o_a[0], rho_v_b[1:4], optimize=True)
                            w_ov = contract('xyr,xria->yria', wfxc * 2.0, rho_ov_a2b, optimize=True)
                            iajb = contract('yria,yrjb->iajb', w_ov, rho_ov_a2b, optimize=True)
                            a_a2b += iajb
                    elif self.xctype == 'MGGA':
                        ao_deriv = 1
                        for ao, mask, weight, coords in _iter_block_data(mf, ni, ao_deriv, max_memory):
                            p0 = p1
                            p1 += weight.shape[0]
                            wfxc = fxc[..., p0:p1] * weight
                            rho_o_a = contract('xrp,pi->xri', ao, ctx.orbo_a, optimize=True)
                            rho_v_b = contract('xrp,pi->xri', ao, ctx.orbv_b, optimize=True)
                            rho_ov_a2b = contract('xri,ra->xria', rho_o_a, rho_v_b[0], optimize=True)
                            rho_ov_a2b[1:4] += contract('ri,xra->xria', rho_o_a[0], rho_v_b[1:4], optimize=True)
                            tau_ov_a2b = contract('xri,xra->ria', rho_o_a[1:4], rho_v_b[1:4], optimize=True) * 0.5
                            rho_ov_a2b = np.vstack([rho_ov_a2b, tau_ov_a2b[np.newaxis]])
                            w_ov = contract('xyr,xria->yria', wfxc * 2.0, rho_ov_a2b, optimize=True)
                            iajb = contract('yria,yrjb->iajb', w_ov, rho_ov_a2b, optimize=True)
                            a_a2b += iajb
                    elif self.xctype == 'HF':
                        pass
                    elif self.xctype == 'NLC':
                        raise NotImplementedError('NLC functional is not supported here.')
                    else:
                        raise NotImplementedError(f'Unsupported xctype: {self.xctype}')
            else:
                a_a2b = add_hf_a_a2b(a_a2b, mf, ctx.orbo_a, ctx.orbv_b, ctx.nocc_a, ctx.nvir_b, hyb=1)
            
            focka_mo, fockb_mo = (_asnumpy(x) for x in _get_mo_fock(mf, ctx.mo_coeff, ctx.mo_occ))
            dim = (self.nc + self.no) * (self.nv + self.no)
            amat = _pair_hessian_block_a2b(
                focka_mo[:ctx.nocc_a, :ctx.nocc_a],
                fockb_mo[ctx.nocc_b:, ctx.nocc_b:],
                a_a2b,
            ).reshape(dim, dim)
            Amat = _convert_a2b_to_cv_co_ov_oo(amat, self.nc, self.no, self.nv)
        finally:
            set_backend(mode)
        self.A = xp.asarray(Amat)
        return self.A
