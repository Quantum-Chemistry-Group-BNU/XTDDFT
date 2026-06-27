import math
from types import SimpleNamespace
import numpy as np
from pyscf import ao2mo
from pyscf.dft import numint2c
from pyscf.pbc.dft import numint2c as pbc_numint2c
from ..utils.hxc_part import (
    cache_xc_kernel_sf_mc,
    gen_response_sf,
    gen_response_sf_mc,
    AldA0
) 
from .base import (
    XTDDFT_base,
    _ao2mo_full_gamma,
    _build_initial_guess_from_gaps,
    _df_ao2mo_pbc,
    _get_gamma_kpt,
    _get_hf_mo_fock,
    _get_j,
    _get_jk,
    _get_mo_fock,
    _get_ovlp,
    _is_pbc_mf,
    _iter_block_data,
    _make_rohf_reference_mf,
    _molecular_dipole_integrals,
    _run_davidson,
    _spinflip_gaps,
    _as_cpu_mf,
    _as_cpu_ctx,
    _system,
    _make_reference_dm,
    _response_max_memory,
)
from ..utils.backend import _asarray, _asnumpy, backend, contract, xp
from ..utils.unit import ha2eV

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger(__name__)

def add_hf_a_a2b(a_a2b, mf, orbo_a, orbv_b, nocc_a, nvir_b, hyb=1, omega=None):
    #K矩阵中含有的精确交换部分
    if abs(hyb) < 1e-14 or nocc_a == 0 or nvir_b == 0:
        return a_a2b

    if _is_pbc_mf(mf):
        eri_mo = _df_ao2mo_pbc(
            mf, [orbo_a, orbo_a, orbv_b, orbv_b],
            omega=omega, compact=False,
        )
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

class XSF_TDA_down(XTDDFT_base): # just for ROKS
    def __init__(self, mf, method, davidson=True, SA = None, davidson_backend="cpu",
                 collinear_samples=60, delta_a_jk_batch_size=None,
                 delta_a_diag_j_batch_size=None, df_cache=None):
        """SA=0: SF-TDA
           SA=1: only add diagonal block for dA
           SA=2: add all dA except for OO block
           SA=3: full dA
        """
        davidson_backend = davidson_backend.lower()
        if davidson_backend not in ("cpu", "gpu", "auto"):
            raise ValueError("davidson_backend must be 'cpu', 'gpu', or 'auto'")
        super().__init__(mf, method, davidson=davidson, df_cache=df_cache)
        logger.info("XSF_TDA_down method=0 ALDA0, method=1 multicollinear")
        self.isf = -1
        self.type_u = _asnumpy(self.mf.mo_coeff).ndim == 3
        self.davidson_backend = "cpu" if davidson_backend == "auto" else davidson_backend
        self.collinear_samples = collinear_samples
        if delta_a_jk_batch_size is not None and delta_a_jk_batch_size < 1:
            raise ValueError("delta_a_jk_batch_size must be a positive integer or None")
        self.delta_a_jk_batch_size = delta_a_jk_batch_size
        if delta_a_diag_j_batch_size is not None and delta_a_diag_j_batch_size < 1:
            raise ValueError
        self.delta_a_diag_j_batch_size = delta_a_diag_j_batch_size
        self.SA = (0 if self.type_u else 3) if SA is None else SA
        spin_mf = _as_cpu_mf(mf)
        _,dsp1 = spin_mf.spin_square()
        self.ground_s = (dsp1-1)/2

    def _result_method_label(self):
        return {0: "ALDA0", 1: "MCOL"}.get(self.method, f"method{self.method}")
    
    def get_Amat_ALDA0(self):
        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)
        ni = getattr(mf, "_numint", None)

        a_a2b = np.zeros((ctx.nocc_a, ctx.nvir_b, ctx.nocc_a, ctx.nvir_b))
        if self.mfxctype is not None:  # 构建K矩阵
            if ni is None:
                ni = self.ni
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
                for ao, mask, weight, coords in _iter_block_data(mf, ni, ao_deriv, max_memory, force_cpu=True):
                    rho0a = make_rho(0, ao, mask, self.xctype)
                    rho0b = make_rho(1, ao, mask, self.xctype)
                    rho = (rho0a, rho0b)
                    fxc_ab = AldA0(ni, mf, rho, weight, self.xctype, omega=self.omega)
                    a_a2b += construct_xc_a2b(ao, ctx.orbo_a, ctx.orbv_b, fxc_ab)

            elif self.xctype == 'GGA' and not getattr(self, "collinear", False):  # 进行简化
                ao_deriv = 0
                for ao, mask, weight, coords in _iter_block_data(mf, ni, ao_deriv, max_memory, force_cpu=True):
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

        focka_mo, fockb_mo = _get_mo_fock(mf, ctx.mo_coeff, ctx.mo_occ, force_cpu=True)

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
                    for ao, mask, weight, coords in _iter_block_data(mf, ni0, ao_deriv, max_memory, force_cpu=True):
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
                    for ao, mask, weight, coords in _iter_block_data(mf, ni, ao_deriv, max_memory, force_cpu=True):
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
                    for ao, mask, weight, coords in _iter_block_data(mf, ni, ao_deriv, max_memory, force_cpu=True):
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

        focka_mo, fockb_mo = _get_mo_fock(mf, ctx.mo_coeff, ctx.mo_occ, force_cpu=True)
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
            self.get_Amat_MCOL(self.collinear_samples)
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
        fockA_hf, fockB_hf = _get_mo_fock(hf, ctx.mo_coeff, ctx.mo_occ, force_cpu=True)
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

    def _prepare_dense_A(self, foo=1.0, fglobal=1.0, frozen=None):
        self.A = self.get_Amat(foo=foo, fglobal=fglobal)
        if self.re:
            logger.info(f"fglobal: {fglobal}")
            self.A = self.remove()
        elif frozen is not None:
            self.A = self.frozen_A(frozen)
        return self.A

    def _compress_removed_hdiag(self, hdiag):
        nc, no, nv = self.nc, self.no, self.nv
        dim1 = nc * nv
        dim2 = dim1 + nc * no
        dim3 = dim2 + no * nv
        oo = hdiag[dim3:]
        new_oo = contract("x,xy,xy->y", oo, self.vects.conj(), self.vects)
        new_hdiag = xp.zeros(int(hdiag.size) - 1, dtype=hdiag.dtype)
        new_hdiag[:dim3] = hdiag[:dim3]
        new_hdiag[dim3:] = new_oo
        return new_hdiag

    def _response_j_batch_size(self, total):
        if total <= 0:
            return 1
        if not backend.is_gpu:
            return min(64, total)
        try:
            free_mem, _ = xp.cuda.runtime.memGetInfo()
        except Exception:
            return min(64, total)

        nao = int(self.mo_coeff.shape[-2])
        itemsize = xp.dtype(self.mo_coeff.dtype).itemsize
        response_dim = max(1, int(self.nocc_a) * int(self.nvir_b))
        bytes_per_trial = itemsize * (2 * nao * nao + response_dim)
        fixed_bytes = itemsize * (2 * response_dim)
        budget = max(0, int(0.60 * free_mem) - fixed_bytes)
        raw = max(1, int(budget / max(bytes_per_trial, 1)))
        if raw >= 32:
            raw = max(32, (raw // 32) * 32)
        return min(total, raw)

    def _response_j_diagonals(self, batch_size=None):
        mo_coeff = _asarray(self.mo_coeff)
        nc, no, nv = self.nc, self.no, self.nv
        orbca = mo_coeff[0][:, self.occidx_a[:nc]]
        orboa_open = mo_coeff[0][:, self.occidx_a[nc:nc + no]]
        orbbo = mo_coeff[1][:, self.viridx_b[:no]]
        orbvv = mo_coeff[1][:, self.viridx_b[no:]]

        total_co = nc * no
        total_ov = no * nv
        total = total_co + total_ov
        if batch_size is None:
            batch_size = self._response_j_batch_size(total)

        co_j = xp.zeros(total_co, dtype=mo_coeff.dtype)
        ov_j = xp.zeros(total_ov, dtype=mo_coeff.dtype)

        def build_dm(indices, nrow, ncol, orbv, orbo):
            nb = indices.size
            trial = xp.zeros((nb, nrow * ncol), dtype=mo_coeff.dtype)
            trial[xp.arange(nb), indices] = 1
            trial = trial.reshape(nb, nrow, ncol)
            return contract("xov,qv,po->xpq", trial, orbv.conj(), orbo)

        for p0 in range(0, total, batch_size):
            p1 = min(p0 + batch_size, total)
            dms = []
            parts = []

            co0, co1 = p0, min(p1, total_co)
            if co0 < co1:
                idx = xp.arange(co0, co1)
                dms.append(build_dm(idx, nc, no, orbbo, orbca))
                parts.append(("co", idx, int(idx.size)))

            ov0, ov1 = max(p0, total_co), p1
            if ov0 < ov1:
                idx = xp.arange(ov0 - total_co, ov1 - total_co)
                dms.append(build_dm(idx, no, nv, orbvv, orboa_open))
                parts.append(("ov", idx, int(idx.size)))

            vj = _get_j(self.mf, xp.concatenate(dms, axis=0), hermi=0)
            q0 = 0
            for name, idx, nb in parts:
                vj_part = vj[q0:q0 + nb]
                q0 += nb
                if name == "co":
                    block = contract("xpq,pi,qu->xiu", vj_part, orbca.conj(), orbbo).reshape(nb, total_co)
                    co_j[idx] = block[xp.arange(nb), idx]
                else:
                    block = contract("xpq,pu,qa->xua", vj_part, orboa_open.conj(), orbvv).reshape(nb, total_ov)
                    ov_j[idx] = block[xp.arange(nb), idx]

        return co_j.reshape(nc, no), ov_j.reshape(no, nv)

    def _build_preconditioner_hdiag(self, fockA, fockB, fglobal=1.0,
                                    fockA_hf=None, fockB_hf=None):
        nc, no = self.nc, self.no
        si = no / 2.0
        diag_a = fockA.diagonal()
        diag_b = fockB.diagonal()
        hdiag = diag_b[self.nocc_b:, None].T - diag_a[:self.nocc_a, None]

        use_delta_a = (
            self.SA > 0
            and not self.type_u
            and fockA_hf is not None
            and fockB_hf is not None
        )
        if use_delta_a:
            fockS = (fockB_hf - fockA_hf) * 0.5
            diag_s = fockS.diagonal()
            hdiag[:nc, no:] += fglobal * (
                diag_s[nc + no:] + diag_s[:nc, None]
            ) / si
            co_j, ov_j = self._response_j_diagonals(batch_size=self.delta_a_diag_j_batch_size)
            hdiag[:nc, :no] += fglobal * (
                2.0 * diag_s[:nc, None] - co_j
            ) / (2 * si - 1)
            hdiag[nc:, no:] += fglobal * (
                2.0 * diag_s[nc + no:] - ov_j
            ) / (2 * si - 1)

        return xp.hstack([
            hdiag[:nc, no:].reshape(-1),
            hdiag[:nc, :no].reshape(-1),
            hdiag[nc:, no:].reshape(-1),
            hdiag[nc:, :no].reshape(-1),
        ])

    def _split_block_vectors(self, data, expand_oo=True):
        data = xp.asarray(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        nc, no, nv = self.nc, self.no, self.nv
        dim1 = nc * nv
        dim2 = dim1 + nc * no
        dim3 = dim2 + no * nv

        cv = data[:, :dim1].reshape(data.shape[0], nc, nv)
        co = data[:, dim1:dim2].reshape(data.shape[0], nc, no)
        ov = data[:, dim2:dim3].reshape(data.shape[0], no, nv)
        oo = data[:, dim3:]
        if self.re and expand_oo:
            oo = contract("xy,ny->nx", self.vects, oo)
        oo = oo.reshape(data.shape[0], no, no)
        return cv, co, ov, oo

    def _join_block_vectors(self, cv, co, ov, oo, compress_oo=None):
        if compress_oo is None:
            compress_oo = self.re
        oo = oo.reshape(oo.shape[0], -1)
        if compress_oo:
            oo = contract("xy,nx->ny", self.vects, oo)
        return xp.hstack([
            cv.reshape(cv.shape[0], -1),
            co.reshape(co.shape[0], -1),
            ov.reshape(ov.shape[0], -1),
            oo,
        ])

    def gen_response_sf_delta_A(self, hermi=0, max_memory=None):
        del max_memory

        def vind(dm1):
            return _get_jk(self.mf, dm1, hermi=hermi, batch=True)

        return vind

    def _apply_delta_a_jk_response(self, vresp_hf, dm_hf):
        batch_size = self.delta_a_jk_batch_size
        if batch_size is None or dm_hf.shape[0] <= batch_size:
            return vresp_hf(dm_hf)

        vj_blocks = []
        vk_blocks = []
        for start in range(0, dm_hf.shape[0], batch_size):
            stop = min(start + batch_size, dm_hf.shape[0])
            vj_block, vk_block = vresp_hf(dm_hf[start:stop])
            vj_blocks.append(vj_block)
            vk_blocks.append(vk_block)
        return xp.concatenate(vj_blocks, axis=0), xp.concatenate(vk_blocks, axis=0)

    def _apply_delta_a_jk_response_blocks(self, vresp_hf, dm_blocks):
        return [
            self._apply_delta_a_jk_response(vresp_hf, dm_block)
            for dm_block in dm_blocks
        ]

    def gen_tda_operation_sf(self, foo=1.0, fglobal=1.0):
        nc, no, nv = self.nc, self.no, self.nv
        nvir = no + nv
        nocc = nc + no
        si = no / 2.0
        if self.SA > 0 and _asnumpy(self.mf.mo_coeff).ndim != 3 and abs(2 * si - 1) < 1e-12:
            raise ValueError(
                "Table-3 Delta A contains 2*S_i-1 denominators and is singular for S_i=1/2. "
                "Use SA=0 for doublet references."
            )

        mo_coeff = _asarray(self.mo_coeff)
        mo_occ = _asarray(self.mo_occ)
        orboa = mo_coeff[0][:, self.occidx_a]
        orbvb = mo_coeff[1][:, self.viridx_b]
        fockA, fockB = self._get_fock_mo()

        vresp = (
            gen_response_sf_mc(
                self.mf, hermi=0, collinear_samples=self.collinear_samples,
                ctx=self.ctx,
            )
            if self.method == 1
            else gen_response_sf(self.mf, hermi=0, ctx=self.ctx)
        )

        use_delta_a = self.SA > 0 and _asnumpy(self.mf.mo_coeff).ndim != 3
        if use_delta_a:
            vresp_hf = self.gen_response_sf_delta_A(hermi=0)
            fockA_hf, fockB_hf = _get_hf_mo_fock(self.mf, mo_coeff, mo_occ)
            fockS_hf = (fockB_hf - fockA_hf) * 0.5
            factor1 = xp.sqrt((2 * si + 1) / (2 * si)) - 1
            factor2 = xp.sqrt((2 * si + 1) / (2 * si - 1))
            factor3 = xp.sqrt((2 * si) / (2 * si - 1)) - 1
            factor4 = 1 / xp.sqrt(2 * si * (2 * si - 1))
            iden_O = xp.eye(no)
            fs_cc = fockS_hf[:nc, :nc]
            fs_vv = fockS_hf[nc + no:, nc + no:]
            fs_cv = fockS_hf[:nc, nc + no:]

        hdiag = self._build_preconditioner_hdiag(
            fockA, fockB, fglobal=fglobal,
            fockA_hf=fockA_hf if use_delta_a else None,
            fockB_hf=fockB_hf if use_delta_a else None,
        )
        if self.re:
            self.vects = xp.asarray(self.get_vect())
            hdiag = self._compress_removed_hdiag(hdiag)

        orbca = orboa[:, :nc]
        orboa_open = orboa[:, nc:nc + no]
        orbbo = orbvb[:, :no]
        orbvv = orbvb[:, no:]
        fa_cc = fockA[:nc, :nc]
        fa_co = fockA[:nc, nc:nc + no]
        fa_oc = fockA[nc:nc + no, :nc]
        fa_oo = fockA[nc:nc + no, nc:nc + no]
        fb_oo = fockB[nc:nc + no, nc:nc + no]
        fb_ov = fockB[nc:nc + no, nc + no:]
        fb_vo = fockB[nc + no:, nc:nc + no]
        fb_vv = fockB[nc + no:, nc + no:]

        def project_response_blocks(v1ao):
            return (
                contract("xpq,pi,qa->xia", v1ao, orbca.conj(), orbvv),
                contract("xpq,pi,qu->xiu", v1ao, orbca.conj(), orbbo),
                contract("xpq,pu,qa->xua", v1ao, orboa_open.conj(), orbvv),
                contract("xpq,pu,qv->xuv", v1ao, orboa_open.conj(), orbbo),
            )

        def vind(zs0):
            zs0 = xp.asarray(zs0)
            cv, co, ov, oo = self._split_block_vectors(zs0)

            dmov = (
                contract("xia,qa,pi->xpq", cv, orbvv.conj(), orbca)
                + contract("xiu,qu,pi->xpq", co, orbbo.conj(), orbca)
                + contract("xua,qa,pu->xpq", ov, orbvv.conj(), orboa_open)
                + contract("xuv,qv,pu->xpq", oo, orbbo.conj(), orboa_open)
            ) # 转成AO基的vector
            v1ao = vresp(dmov)  # 获得ao基下的响应函数。也就是K矩阵（包括交换相关和精确HF部分）
            vs_cv, vs_co, vs_ov, vs_oo = project_response_blocks(v1ao)  # 转成MO基下的结果
            # 分别加上前面的fock矩阵
            vs_cv += (
                contract("xiu,ua->xia", co, fb_ov)
                + contract("xib,ba->xia", cv, fb_vv)
                - contract("ij,xja->xia", fa_cc, cv)
                - contract("iu,xua->xia", fa_co, ov)
            )
            vs_co += (
                contract("xiv,vu->xiu", co, fb_oo)
                + contract("xia,au->xiu", cv, fb_vo)
                - contract("ij,xju->xiu", fa_cc, co)
                - contract("iv,xvu->xiu", fa_co, oo)
            )
            vs_ov += (
                contract("xuv,va->xua", oo, fb_ov)
                + contract("xub,ba->xua", ov, fb_vv)
                - contract("ui,xia->xua", fa_oc, cv)
                - contract("uv,xva->xua", fa_oo, ov)
            )
            vs_oo += (
                contract("xuw,wv->xuv", oo, fb_oo)
                + contract("xua,av->xuv", ov, fb_vo)
                - contract("ui,xiv->xuv", fa_oc, co)
                - contract("uw,xwv->xuv", fa_oo, oo)
            )

            if use_delta_a:
                dcv = xp.zeros_like(cv)
                dco = xp.zeros_like(co)
                dov = xp.zeros_like(ov)
                doo = xp.zeros_like(oo)

                cv1_mo = contract("xia,qa,pi->xpq", cv, orbvv.conj(), orbca)
                co1_mo = contract("xiu,qu,pi->xpq", co, orbbo.conj(), orbca)
                ov1_mo = contract("xua,qa,pu->xpq", ov, orbvv.conj(), orboa_open)
                oo1_mo = contract("xuv,qv,pu->xpq", oo, orbbo.conj(), orboa_open)

                (
                    (_v1_cv_j, v1_cv_k),
                    (v1_co_j, v1_co_k),
                    (v1_ov_j, v1_ov_k),
                    (_v1_oo_j, v1_oo_k),
                ) = self._apply_delta_a_jk_response_blocks(
                    vresp_hf, (cv1_mo, co1_mo, ov1_mo, oo1_mo)
                )

                cv_co_j, co_co_j, ov_co_j, oo_co_j = project_response_blocks(v1_co_j)
                cv_ov_j, co_ov_j, ov_ov_j, oo_ov_j = project_response_blocks(v1_ov_j)
                cv_cv_k, co_cv_k, ov_cv_k, oo_cv_k = project_response_blocks(v1_cv_k)
                cv_co_k, co_co_k, ov_co_k, oo_co_k = project_response_blocks(v1_co_k)
                cv_ov_k, co_ov_k, ov_ov_k, oo_ov_k = project_response_blocks(v1_ov_k)
                cv_oo_k, co_oo_k, ov_oo_k, oo_oo_k = project_response_blocks(v1_oo_k)

                dcv += (
                    contract("ab,xib->xia", fs_vv, cv)
                    + contract("ji,xja->xia", fs_cc, cv)
                ) / si
                dco += -co_co_j / (2 * si - 1) + (
                    2.0 * contract("ji,xju->xiu", fs_cc, co)
                ) / (2 * si - 1)
                dov += -ov_ov_j / (2 * si - 1) + (
                    2.0 * contract("ab,xub->xua", fs_vv, ov)
                ) / (2 * si - 1)

                if self.SA > 1:
                    dcv += factor1 * (
                        -cv_co_k
                        + contract("av,xiv->xia", fockB_hf[nc + no:, nc:nc + no], co)
                    )
                    dco += factor1 * (
                        -co_cv_k
                        + contract("av,xja->xjv", fockB_hf[nc + no:, nc:nc + no], cv)
                    )
                    dcv += factor1 * (
                        -cv_ov_k
                        - contract("vi,xva->xia", fockA_hf[nc:nc + no, :nc], ov)
                    )
                    dov += factor1 * (
                        -ov_cv_k
                        - contract("vi,xib->xvb", fockA_hf[nc:nc + no, :nc], cv)
                    )
                    dco += (co_ov_j - co_ov_k) / (2 * si - 1)
                    dov += (ov_co_j - ov_co_k) / (2 * si - 1)

                if self.SA > 2:
                    dcv += foo * (
                        -(factor2 - 1) * cv_oo_k
                        + (factor2 / si) * contract("ia,xvv->xia", fs_cv, oo)
                    )
                    doo += foo * (
                        -(factor2 - 1) * oo_cv_k
                        + (factor2 / si) * contract("vw,ia,xia->xvw", iden_O, fs_cv, cv)
                    )
                    dco += foo * (
                        factor3 * (
                            -co_oo_k
                            - contract("iw,xwu->xiu", fockA_hf[:nc, nc:nc + no], oo)
                        )
                        + factor4 * contract("vw,iu,xvw->xiu", iden_O, fockB_hf[:nc, nc:nc + no], oo)
                    )
                    doo += foo * (
                        factor3 * (
                            -oo_co_k
                            - contract("iw,xiv->xwv", fockA_hf[:nc, nc:nc + no], co)
                        )
                        + factor4 * contract("vw,iu,xiu->xvw", iden_O, fockB_hf[:nc, nc:nc + no], co)
                    )
                    dov += foo * (
                        factor3 * (
                            -ov_oo_k
                            + contract("av,xuv->xua", fockB_hf[nc + no:, nc:nc + no], oo)
                        )
                        - factor4 * contract("vw,au,xvw->xua", iden_O, fockA_hf[nc + no:, nc:nc + no], oo)
                    )
                    doo += foo * (
                        factor3 * (
                            -oo_ov_k
                            + contract("av,xwa->xwv", fockB_hf[nc + no:, nc:nc + no], ov)
                        )
                        - factor4 * contract("vw,au,xua->xwv", iden_O, fockA_hf[nc + no:, nc:nc + no], ov)
                    )
                vs_cv += fglobal * dcv
                vs_co += fglobal * dco
                vs_ov += fglobal * dov
                vs_oo += fglobal * doo

            return self._join_block_vectors(vs_cv, vs_co, vs_ov, vs_oo)

        return vind, hdiag

    def init_guess(self, nstates, hdiag=None):
        if hdiag is None:
            _, hdiag = self.gen_tda_operation_sf()
        return _build_initial_guess_from_gaps(hdiag, nstates)

    def davidson_process(self, nstates, foo=1.0, fglobal=1.0):
        vind, hdiag = self.gen_tda_operation_sf(
            foo=foo, fglobal=fglobal
        )
        nroots = min(nstates, int(hdiag.size))
        x0 = self.init_guess(nroots, hdiag=hdiag)
        converged, e, x1 = _run_davidson(
            self.mf, self.davidson_backend,
            vind, hdiag, x0, nroots,
            positive_eig_threshold=1.0e-3 if _is_pbc_mf(self.mf) else None,
        )
        self.converged = converged
        self.e = xp.asarray(e)
        self.v = xp.asarray(_asnumpy(x1)).T
        logger.info('XSF_TDA_down Davidson converged: {}', converged)
        return self.e, self.v

    def _split_analysis_vectors(self, value):
        value = np.asarray(value)
        nc, no, nv = self.nc, self.no, self.nv
        dim1 = nc * nv
        dim2 = dim1 + nc * no
        dim3 = dim2 + no * nv
        x_cv = value[:dim1].reshape(nc, nv)
        x_co = value[dim1:dim2].reshape(nc, no)
        x_ov = value[dim2:dim3].reshape(no, nv)
        if self.re:
            vects = np.asarray(_asnumpy(self.vects))
            x_oo = (vects @ value[dim3:].reshape(-1, 1)).reshape(no, no)
        else:
            x_oo = value[dim3:].reshape(no, no)
        return x_cv, x_co, x_ov, x_oo

    def _deltaS2_U_overlaps(self):
        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)
        mo_coeff = _asnumpy(ctx.mo_coeff)
        occidx_a = _asnumpy(ctx.occidx_a)
        occidx_b = _asnumpy(ctx.occidx_b)
        viridx_b = _asnumpy(ctx.viridx_b)

        mooa = mo_coeff[0][:, occidx_a]
        moob = mo_coeff[1][:, occidx_b]
        movb = mo_coeff[1][:, viridx_b]
        ovlp = _asnumpy(_get_ovlp(mf))

        sab_oo = contract("pi,pq,qj->ij", mooa.conj(), ovlp, moob)
        sba_oo = sab_oo.conj().T
        sba_vo = contract("pa,pq,qi->ai", movb.conj(), ovlp, mooa)
        return sba_oo, sba_vo

    def _deltaS2_U_from_overlaps(self, nstate, sba_oo, sba_vo):
        x_cv, x_co, x_ov, x_oo = self._split_analysis_vectors(_asnumpy(self.v[:, nstate]))
        x_ba = np.concatenate([np.hstack([x_co, x_cv]), np.hstack([x_oo, x_ov])], axis=0).T
        sba_vo_overlap = contract("ai,ai->", x_ba.conj(), sba_vo)
        ds2 = (
            contract("ai,aj,jk,ki", x_ba.conj(), x_ba, sba_oo.T.conj(), sba_oo)
            - contract("ai,bi,kb,ak", x_ba.conj(), x_ba, sba_vo.T.conj(), sba_vo)
            + abs(sba_vo_overlap) ** 2
        )
        return np.real_if_close(ds2)

    def deltaS2_U(self, nstate):
        return self._deltaS2_U_from_overlaps(nstate, *self._deltaS2_U_overlaps())

    def deltaS2(self):
        ds2 = []
        if self.type_u:
            sba_oo, sba_vo = self._deltaS2_U_overlaps()
            for nstate in range(self.nstates):
                ds2.append(self._deltaS2_U_from_overlaps(nstate, sba_oo, sba_vo) - self.no + 1.0)  # U
            return np.asarray(np.real_if_close(ds2), dtype=float)

        for nstate in range(self.nstates):
            value = _asnumpy(self.v[:, nstate])
            x_cv, _, _, x_oo = self._split_analysis_vectors(value)
            if self.SA == 0 and not self.type_u:
                dp_ab = np.sum(x_cv * x_cv) - np.sum(x_oo * x_oo) + np.sum(np.diag(x_oo)) ** 2
                ds2.append(-2.0 * self.ground_s + 1.0 + dp_ab)  # RO
            else:
                ds2.append(np.nan)
        return np.asarray(np.real_if_close(ds2), dtype=float)

    def _dipole_mo_integrals(self):
        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)
        dip_ao = _molecular_dipole_integrals(mf)
        mo_coeff = _asnumpy(ctx.mo_coeff)
        if self.type_u:
            ints_aa = contract("xpq,pi,qj->xij", dip_ao, mo_coeff[0].conj(), mo_coeff[0])
            ints_bb = contract("xpq,pi,qj->xij", dip_ao, mo_coeff[1].conj(), mo_coeff[1])
            return np.asarray(ints_aa), np.asarray(ints_bb), ctx

        mo_order = np.concatenate([_asnumpy(ctx.occidx_a), _asnumpy(ctx.viridx_a)])
        mo = mo_coeff[0][:, mo_order]
        ints_mo = contract("xpq,pi,qj->xij", dip_ao, mo.conj(), mo)
        return np.asarray(ints_mo), ctx

    def _checked_state_index(self, state):
        nstates = min(self.nstates, _asnumpy(self.v).shape[1])
        state = int(state)
        if state < 0:
            state += nstates
        if state < 0 or state >= nstates:
            raise IndexError(f"state index {state} out of range for {nstates} states")
        return state

    def _state_analysis_blocks(self, state):
        state = self._checked_state_index(state)
        return self._split_analysis_vectors(_asnumpy(self.v[:, state]))

    def _spinflip_amplitude_matrix_u(self, state):
        cv, co, ov, oo = self._state_analysis_blocks(state)
        dtype = np.result_type(cv, co, ov, oo)
        amps = np.zeros((self.nc + self.no, self.no + self.nv), dtype=dtype)
        amps[:self.nc, self.no:] = cv
        amps[:self.nc, :self.no] = co
        amps[self.nc:, self.no:] = ov
        amps[self.nc:, :self.no] = oo
        return amps

    def _transition_density_matrix_u(self, state_f, state_i):
        ctx = self.ctx
        occidx_a = _asnumpy(getattr(ctx, "occidx_a", np.arange(self.nc + self.no))).astype(int)
        viridx_b = _asnumpy(getattr(ctx, "viridx_b", np.arange(self.no + self.nv))).astype(int)
        mo_coeff = getattr(ctx, "mo_coeff", None)
        if mo_coeff is not None:
            mo_coeff = _asnumpy(mo_coeff)
            nmo_a = int(mo_coeff[0].shape[1])
            nmo_b = int(mo_coeff[1].shape[1])
        else:
            nmo_a = int(occidx_a.max() + 1) if occidx_a.size else self.nc + self.no
            nmo_b = int(viridx_b.max() + 1) if viridx_b.size else self.no + self.nv

        amp_f = self._spinflip_amplitude_matrix_u(state_f)
        amp_i = self._spinflip_amplitude_matrix_u(state_i)
        gamma = np.zeros((nmo_a + nmo_b, nmo_a + nmo_b), dtype=np.result_type(amp_f, amp_i))
        gamma[np.ix_(occidx_a, occidx_a)] -= contract("ia,ja->ij", amp_f.conj(), amp_i)
        beta = nmo_a
        gamma[np.ix_(beta + viridx_b, beta + viridx_b)] += contract("ia,ib->ab", amp_f.conj(), amp_i)
        return gamma

    def _transition_density_matrix_r(self, state_f, state_i):
        cv_f, co_f, ov_f, oo_f = self._state_analysis_blocks(state_f)
        cv_i, co_i, ov_i, oo_i = self._state_analysis_blocks(state_i)
        nmo = self.nc + self.no + self.nv
        gamma = np.zeros(
            (nmo, nmo),
            dtype=np.result_type(cv_f, co_f, ov_f, oo_f, cv_i, co_i, ov_i, oo_i),
        )
        c = slice(0, self.nc)
        o = slice(self.nc, self.nc + self.no)
        v = slice(self.nc + self.no, nmo)
        si = self.ground_s
        if self.SA == 0:
            factor1, factor2, factor3 = 1.0, 1.0, 0.0
        else:
            factor1 = math.sqrt((2 * si + 1) / (2 * si))
            factor2 = math.sqrt((2 * si) / (2 * si - 1))
            factor3 = 1.0 / math.sqrt(2 * si * (2 * si - 1))

        cv_f = cv_f.conj()
        co_f = co_f.conj()
        ov_f = ov_f.conj()
        oo_f = oo_f.conj()
        tr_oo_f = np.trace(oo_f)
        tr_oo_i = np.trace(oo_i)

        gamma[v, v] += contract("ia,ib->ab", cv_f, cv_i)
        gamma[c, c] -= contract("ia,ja->ij", cv_f, cv_i)
        gamma[v, o] += factor1 * contract("ia,iv->av", cv_f, co_i)
        gamma[v, o] += factor1 * contract("iu,ib->bu", co_f, cv_i)
        gamma[c, o] -= factor1 * contract("ia,va->iv", cv_f, ov_i)
        gamma[c, o] -= factor1 * contract("ua,ja->ju", ov_f, cv_i)
        gamma[o, o] += contract("iu,iv->uv", co_f, co_i)
        gamma[c, c] -= contract("iu,ju->ij", co_f, co_i)
        gamma[c, o] -= factor2 * contract("iu,vu->iv", co_f, oo_i)
        gamma[c, o] += factor3 * co_f * tr_oo_i
        gamma[c, o] -= factor2 * contract("ut,jt->ju", oo_f, co_i)
        gamma[c, o] += factor3 * tr_oo_f * co_i
        gamma[v, v] += contract("ua,ub->ab", ov_f, ov_i)
        gamma[o, o] -= contract("ua,va->uv", ov_f, ov_i)
        gamma[v, o] += factor2 * contract("ua,uw->aw", ov_f, oo_i)
        gamma[o, v] -= factor3 * ov_f * tr_oo_i
        gamma[v, o] += factor2 * contract("ut,ub->bt", oo_f, ov_i)
        gamma[o, v] -= factor3 * tr_oo_f * ov_i
        gamma[o, o] += contract("ut,uv->tv", oo_f, oo_i)
        gamma[o, o] -= contract("ut,wt->uw", oo_f, oo_i)
        return gamma

    def transition_density_matrix(self, state_f=1, state_i=0):
        """Spin-free transition density matrix for state_f <- state_i.

        Restricted references return the MO order C|O|V.  Unrestricted
        references return a block-diagonal spin-MO matrix in alpha|beta order.
        """
        if self.type_u:
            return self._transition_density_matrix_u(state_f, state_i)
        return self._transition_density_matrix_r(state_f, state_i)

    def nto(self, state_f=1, state_i=0, nroots=None):
        """Natural transition orbitals from the spin-free transition density.

        Returns singular values, hole NTOs, and particle NTOs.  The columns of
        the NTO matrices are ordered by descending singular value.
        """
        gamma = self.transition_density_matrix(state_f, state_i)
        particles, singular_values, holes_h = np.linalg.svd(gamma, full_matrices=False)
        holes = holes_h.conj().T
        if nroots is not None:
            nroots = min(int(nroots), singular_values.size)
            singular_values = singular_values[:nroots]
            holes = holes[:, :nroots]
            particles = particles[:, :nroots]
        return singular_values, holes, particles

    def block_nto(self, state=0, nroots=None):
        """SVD channels of each restricted XSF-TDA spin-adapted block.

        The returned vectors are embedded in the restricted C|O|V MO order so
        they can be written as cube orbitals.  They are block-resolved
        contracted spin-adapted configurations, not ordinary reference-to-state
        NTOs.
        """
        if self.type_u:
            raise NotImplementedError("block_nto is implemented for restricted XSF-TDA blocks only")

        cv, co, ov, oo = self._state_analysis_blocks(state)
        nmo = self.nc + self.no + self.nv
        slices = {
            "C": slice(0, self.nc),
            "O": slice(self.nc, self.nc + self.no),
            "V": slice(self.nc + self.no, nmo),
        }
        specs = {
            "CV": (cv.T, "C", "V"),
            "CO": (co.T, "C", "O"),
            "OV": (ov.T, "O", "V"),
            "OO": (oo, "O", "O"),
        }

        result = {}
        for name, (matrix, source, target) in specs.items():
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
            holes[slices[source], :] = holes_local
            particles[slices[target], :] = particles_local
            block = {
                "source": source,
                "target": target,
                "singular_values": singular_values,
                "weights": np.abs(singular_values) ** 2,
                "block_weight": block_weight,
                "holes": holes,
                "particles": particles,
            }
            if name == "OO":
                block["trace_overlap"] = np.sum(holes_local.conj() * particles_local, axis=0)
            result[name] = block
        return result

    def _transition_dipole_matrix_u(self):
        ints_aa, ints_bb, ctx = self._dipole_mo_integrals()
        occ_a = _asnumpy(ctx.occidx_a)
        vir_b = _asnumpy(ctx.viridx_b)
        r_oo_a = ints_aa[:, occ_a][:, :, occ_a]
        r_vv_b = ints_bb[:, vir_b][:, :, vir_b]

        nstates = min(self.nstates, _asnumpy(self.v).shape[1])
        amps = []
        for istate in range(nstates):
            cv, co, ov, oo = self._split_analysis_vectors(_asnumpy(self.v[:, istate]))
            c = np.zeros((self.nc + self.no, self.no + self.nv))
            c[:self.nc, self.no:] = cv
            c[:self.nc, :self.no] = co
            c[self.nc:, self.no:] = ov
            c[self.nc:, :self.no] = oo
            amps.append(c)

        tdm = np.zeros((nstates, nstates, 3))
        for i, c0 in enumerate(amps):
            for j, c1 in enumerate(amps):
                tdm[i, j] = (
                    contract("ia,xab,ib->x", c0, r_vv_b, c1)
                    - contract("ia,xij,ja->x", c0, r_oo_a, c1)
                )
        return tdm

    def _transition_dipole_matrix_r(self):
        ints_mo, _ctx = self._dipole_mo_integrals()
        nc, no, nv = self.nc, self.no, self.nv
        c = slice(0, nc)
        o = slice(nc, nc + no)
        v = slice(nc + no, nc + no + nv)
        si = self.ground_s
        if self.SA == 0:
            factor1, factor2, factor3 = 1.0, 1.0, 0.0
        else:
            factor1 = math.sqrt((2 * si + 1) / (2 * si))
            factor2 = math.sqrt((2 * si) / (2 * si - 1))
            factor3 = 1.0 / math.sqrt(2 * si * (2 * si - 1))

        nstates = min(self.nstates, _asnumpy(self.v).shape[1])
        amps = [
            self._split_analysis_vectors(_asnumpy(self.v[:, istate]))
            for istate in range(nstates)
        ]
        tdm = np.zeros((nstates, nstates, 3))
        for i, (cv0, co0, ov0, oo0) in enumerate(amps):
            for j, (cv1, co1, ov1, oo1) in enumerate(amps):
                rij = np.zeros(3)
                rij += contract("ia,xab,ib->x", cv0, ints_mo[:, v, v], cv1)
                rij -= contract("ia,xij,ja->x", cv0, ints_mo[:, c, c], cv1)
                rij += factor1 * contract("ia,xav,iv->x", cv0, ints_mo[:, v, o], co1)
                rij += factor1 * contract("iu,xbu,ib->x", co0, ints_mo[:, v, o], cv1)
                rij -= factor1 * contract("ia,xiv,va->x", cv0, ints_mo[:, c, o], ov1)
                rij -= factor1 * contract("ua,xju,ja->x", ov0, ints_mo[:, c, o], cv1)
                rij += contract("iu,xuv,iv->x", co0, ints_mo[:, o, o], co1)
                rij -= contract("iu,xij,ju->x", co0, ints_mo[:, c, c], co1)
                rij -= factor2 * contract("iu,xiv,vu->x", co0, ints_mo[:, c, o], oo1)
                rij += factor3 * contract("iu,xiu,vv->x", co0, ints_mo[:, c, o], oo1)
                rij -= factor2 * contract("ut,xju,jt->x", oo0, ints_mo[:, c, o], co1)
                rij += factor3 * contract("uu,xjv,jv->x", oo0, ints_mo[:, c, o], co1)
                rij += contract("ua,xab,ub->x", ov0, ints_mo[:, v, v], ov1)
                rij -= contract("ua,xuv,va->x", ov0, ints_mo[:, o, o], ov1)
                rij += factor2 * contract("ua,xaw,uw->x", ov0, ints_mo[:, v, o], oo1)
                rij -= factor3 * contract("ua,xua,vv->x", ov0, ints_mo[:, o, v], oo1)
                rij += factor2 * contract("ut,xbt,ub->x", oo0, ints_mo[:, v, o], ov1)
                rij -= factor3 * contract("uu,xvb,vb->x", oo0, ints_mo[:, o, v], ov1)
                rij += contract("ut,xtv,uv->x", oo0, ints_mo[:, o, o], oo1)
                rij -= contract("ut,xuw,wt->x", oo0, ints_mo[:, o, o], oo1)
                tdm[i, j] = rij
        return tdm

    def transition_dipole_matrix(self):
        """Excited-state to excited-state transition dipole matrix in a.u."""
        return self._transition_dipole_matrix_u() if self.type_u else self._transition_dipole_matrix_r()

    def calculate_TDM(self):
        tdm = self.transition_dipole_matrix()
        energies = _asnumpy(self.e)[:tdm.shape[0]]
        osc = (2.0 / 3.0) * (
            (energies[:, None] - energies[None, :]) * np.einsum("ijx,ijx->ij", tdm, tdm)
        )
        print("Excited state to Excited state transition dipole moments(a.u.)")
        print("StateL StateR      X        Y        Z      f(L<-R)")
        for i in range(tdm.shape[0]):
            for j in range(tdm.shape[1]):
                print(
                    f" {i + 1:2d}     {j + 1:2d}    "
                    f"{tdm[i, j, 0]:>8.4f} {tdm[i, j, 1]:>8.4f} {tdm[i, j, 2]:>8.4f}  "
                    f"{osc[i, j]:>8.4f} "
                )
        return tdm, osc

    analyze_TDM = calculate_TDM

    def analyse(
        self,
        threshold=0.05,
        analyze_symmetry=False,
        point_group=None,
        symmetry_tol=1.0e-3,
        energy_tol=1.0e-5,
        projection_backend="auto",
        symmetry_kwargs=None,
    ):
        energies = _asnumpy(self.e) * ha2eV
        self.dS2 = self.deltaS2()
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
                active_roots=range(self.nstates),
                **kwargs,
            )
        for nstate in range(self.nstates):
            x_cv, x_co, x_ov, x_oo = self._split_analysis_vectors(_asnumpy(self.v[:, nstate]))
            irrep_text = ""
            if symmetry_labels is not None and nstate < len(symmetry_labels):
                irrep_text = f" irrep={symmetry_labels[nstate]}"
            if np.isfinite(self.dS2[nstate]):
                print(
                    f"Excited state {nstate + 1} {energies[nstate]:10.5f} eV "
                    f"D<S^2>={self.dS2[nstate]:8.4f}{irrep_text}"
                )
            else:
                print(
                    f"Excited state {nstate + 1} {energies[nstate]:10.5f} eV "
                    f"D<S^2>=     n/a{irrep_text}"
                )

            for label, arr, occ_offset, vir_offset in (
                ("CV(ab)", x_cv, 1, self.nc + self.no + 1),
                ("CO(ab)", x_co, 1, self.nc + 1),
                ("OV(ab)", x_ov, self.nc + 1, self.nc + self.no + 1),
                ("OO(ab)", x_oo, self.nc + 1, self.nc + 1),
            ):
                for occ, vir in zip(*np.where(abs(arr) > threshold)):
                    amp = arr[occ, vir]
                    print(
                        f"{100 * amp**2:5.2f}% {label} "
                        f"{occ + occ_offset}a -> {vir + vir_offset}b {amp:10.5f}"
                    )
            print("========================================")
        return self.dS2

    def kernel(self, nstates=1, remove=None, frozen=None, foo=1.0, d_lda=0.3,
               fglobal=None, fit=True, save=False, save_file=None):
        self.re = (_asnumpy(self.mf.mo_coeff).ndim != 3) if remove is None else bool(remove)
        nov = (self.nc + self.no) * (self.no + self.nv)
        effective_dim = nov - 1 if self.re else nov
        self.nstates = min(nstates, effective_dim)
        if fglobal is None:
            fglobal = self._default_fglobal(d_lda=d_lda, fit=fit)
        self.fglobal = fglobal

        if self.davidson:
            if frozen is not None:
                raise NotImplementedError("frozen orbital truncation is only implemented for dense XSF_TDA_down.")
            self.davidson_process(self.nstates, foo=foo, fglobal=fglobal)
        else:
            self._prepare_dense_A(foo=foo, fglobal=fglobal, frozen=frozen)
            self._diagonalize_dense(self.A, self.nstates)
        if save:
            self.save_results(save_file, fglobal=self.fglobal, remove=self.re)
        return _asnumpy(self.e[:self.nstates] * ha2eV), self.v[:, :self.nstates]
