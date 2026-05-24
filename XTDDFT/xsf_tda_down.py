import math
from types import SimpleNamespace
import numpy as np
from pyscf import ao2mo
from pyscf.dft import numint2c
from pyscf.pbc.dft import numint2c as pbc_numint2c
from XTDDFT.hxc_part import (
    cache_xc_kernel_sf_mc,
    gen_response_sf,
    gen_response_sf_mc,
    AldA0
) 
from XTDDFT.base import (
    XTDDFT_base,
    _ao2mo_full_gamma,
    _build_initial_guess_from_gaps,
    _get_gamma_kpt,
    _get_hf_mo_fock,
    _get_jk,
    _get_mo_fock,
    _get_ovlp,
    _is_pbc_mf,
    _iter_block_data,
    _make_rohf_reference_mf,
    _run_davidson,
    _spinflip_gaps,
    _as_cpu_mf,
    _as_cpu_ctx,
    _system,
    _make_reference_dm,
    _response_max_memory,
)
from utils.backend import _asarray, _asnumpy, contract, xp
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

class XSF_TDA_down(XTDDFT_base): # just for ROKS
    def __init__(self, mf, method, davidson=True, SA = None, davidson_backend="cpu",
                 collinear_samples=60):
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
        self.type_u = _asnumpy(self.mf.mo_coeff).ndim == 3
        self.davidson_backend = "cpu" if davidson_backend == "auto" else davidson_backend
        self.collinear_samples = collinear_samples
        self.SA = (0 if self.type_u else 3) if SA is None else SA
        spin_mf = mf if hasattr(mf, "spin_square") else _as_cpu_mf(mf)
        _,dsp1 = spin_mf.spin_square()
        self.ground_s = (dsp1-1)/2
    
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
        nvir = no + nv
        oo = xp.zeros(no * no, dtype=hdiag.dtype)
        for i in range(no):
            oo[i * no:(i + 1) * no] = hdiag[nc * nvir + i * nvir:nc * nvir + no + i * nvir]
        new_oo = contract("x,xy->y", oo, self.vects)
        new_hdiag = xp.zeros(int(hdiag.size) - 1, dtype=hdiag.dtype)
        new_hdiag[:nc * nvir] = hdiag[:nc * nvir]
        for i in range(no - 1):
            new_hdiag[nc * nvir + i * nvir:nc * nvir + no + i * nvir] = new_oo[i * no:(i + 1) * no]
            new_hdiag[nc * nvir + no + i * nvir:nc * nvir + no + nv + i * nvir] = hdiag[nc * nvir + no + i * nvir:nc * nvir + no + nv + i * nvir]
        new_hdiag[nc * nvir + (no - 1) * nvir:nc * nvir + (no - 1) * nvir + no - 1] = new_oo[(no - 1) * no:]
        new_hdiag[nc * nvir + (no - 1) * nvir + no - 1:] = hdiag[nc * nvir + (no - 1) * nvir + no:]
        return new_hdiag

    def _expand_removed_vectors(self, zs0):
        nc, no, nv = self.nc, self.no, self.nv
        nvir = no + nv
        oo = xp.zeros((zs0.shape[0], no * no - 1), dtype=zs0.dtype)
        for i in range(no - 1):
            oo[:, i * no:(i + 1) * no] = zs0[:, nc * nvir + i * nvir:nc * nvir + no + i * nvir]
        oo[:, (no - 1) * no:] = zs0[:, nc * nvir + (no - 1) * nvir:nc * nvir + (no - 1) * nvir + no - 1]
        new_oo = contract("xy,ny->nx", self.vects, oo)
        full = xp.zeros((zs0.shape[0], zs0.shape[1] + 1), dtype=zs0.dtype)
        full[:, :nc * nvir] = zs0[:, :nc * nvir]
        for i in range(no - 1):
            full[:, nc * nvir + i * nvir:nc * nvir + no + i * nvir] = new_oo[:, i * no:(i + 1) * no]
            full[:, nc * nvir + no + i * nvir:nc * nvir + no + nv + i * nvir] = zs0[:, nc * nvir + no + i * nvir:nc * nvir + no + nv + i * nvir]
        full[:, nc * nvir + (no - 1) * nvir:nc * nvir + no + (no - 1) * nvir] = new_oo[:, -no:]
        full[:, nc * nvir + (no - 1) * nvir + no:] = zs0[:, nc * nvir + (no - 1) * nvir + no - 1:]
        return full

    def _compress_removed_vectors(self, hx):
        nc, no, nv = self.nc, self.no, self.nv
        nvir = no + nv
        out = xp.zeros((hx.shape[0], hx.shape[1] - 1), dtype=hx.dtype)
        out[:, :nc * nvir] = hx[:, :nc * nvir]
        oo = xp.zeros((hx.shape[0], no * no), dtype=hx.dtype)
        for i in range(no):
            oo[:, i * no:(i + 1) * no] = hx[:, nc * nvir + i * nvir:nc * nvir + no + i * nvir]
        new_oo = contract("xy,nx->ny", self.vects, oo)
        for i in range(no - 1):
            out[:, nc * nvir + i * nvir:nc * nvir + no + i * nvir] = new_oo[:, i * no:(i + 1) * no]
            out[:, nc * nvir + no + i * nvir:nc * nvir + no + nv + i * nvir] = hx[:, nc * nvir + no + i * nvir:nc * nvir + no + nv + i * nvir]
        out[:, nc * nvir + (no - 1) * nvir:nc * nvir + (no - 1) * nvir + no - 1] = new_oo[:, (no - 1) * no:]
        out[:, nc * nvir + (no - 1) * nvir + no - 1:] = hx[:, nc * nvir + (no - 1) * nvir + no:]
        return out

    def _order_davidson_vectors(self):
        data = _asnumpy(self.v)
        nroot = data.shape[1]
        cv = np.zeros((nroot, self.nc, self.nv))
        co = np.zeros((nroot, self.nc, self.no))
        ov = np.zeros((nroot, self.no, self.nv))
        oo = np.zeros((nroot, self.no * self.no - 1 if self.re else self.no * self.no))
        nvir = self.no + self.nv
        passed = self.nc * nvir
        for state in range(nroot):
            vec = data[:, state]
            for i in range(self.nc):
                co[state, i] = vec[i * nvir:i * nvir + self.no]
                cv[state, i] = vec[i * nvir + self.no:i * nvir + self.no + self.nv]
            if self.re:
                for i in range(self.no - 1):
                    oo[state, i * self.no:(i + 1) * self.no] = vec[passed + i * nvir:passed + i * nvir + self.no]
                    ov[state, i] = vec[passed + i * nvir + self.no:passed + i * nvir + self.no + self.nv]
                oo[state, (self.no - 1) * self.no:] = vec[passed + (self.no - 1) * nvir:passed + (self.no - 1) * nvir + self.no - 1]
                ov[state, self.no - 1] = vec[passed + (self.no - 1) * nvir + self.no - 1:]
            else:
                for i in range(self.no):
                    oo[state, i * self.no:(i + 1) * self.no] = vec[passed + i * nvir:passed + i * nvir + self.no]
                    ov[state, i] = vec[passed + i * nvir + self.no:passed + i * nvir + self.no + self.nv]
        ordered = np.hstack([
            cv.reshape(nroot, -1),
            co.reshape(nroot, -1),
            ov.reshape(nroot, -1),
            oo.reshape(nroot, -1),
        ])
        return xp.asarray(ordered.T)

    def gen_response_sf_delta_A(self, hermi=0, max_memory=None):
        del max_memory

        def vind(dm1):
            return _get_jk(self.mf, dm1, hermi=hermi, batch=True)

        return vind

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
        hdiag = xp.asarray(_spinflip_gaps(self.ctx, self.isf))
        if self.re:
            self.vects = xp.asarray(self.get_vect())
            hdiag = self._compress_removed_hdiag(hdiag)

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
            factor1 = xp.sqrt((2 * si + 1) / (2 * si)) - 1
            factor2 = xp.sqrt((2 * si + 1) / (2 * si - 1))
            factor3 = xp.sqrt((2 * si) / (2 * si - 1)) - 1
            factor4 = 1 / xp.sqrt(2 * si * (2 * si - 1))
            iden_O = xp.eye(no)

        def vind(zs0):
            zs0 = xp.asarray(zs0)
            full_zs0 = self._expand_removed_vectors(zs0) if self.re else zs0.copy()
            zs = full_zs0.reshape(-1, nocc, nvir)

            dmov = contract("xov,qv,po->xpq", zs, orbvb.conj(), orboa)
            v1ao = vresp(dmov)
            vs = contract("xpq,po,qv->xov", v1ao, orboa.conj(), orbvb)
            vs += contract("ab,xib->xia", fockB[self.nocc_b:, self.nocc_b:], zs)
            vs -= contract("ij,xja->xia", fockA[:self.nocc_a, :self.nocc_a], zs)

            if use_delta_a:
                vs_dA = xp.zeros_like(vs)
                cv1 = zs[:, :nc, no:]
                co1 = zs[:, :nc, :no]
                ov1 = zs[:, nc:, no:]
                oo1 = zs[:, nc:, :no]

                cv1_mo = contract("xov,qv,po->xpq", cv1, orbvb[:, no:].conj(), orboa[:, :nc])
                co1_mo = contract("xov,qv,po->xpq", co1, orbvb[:, :no].conj(), orboa[:, :nc])
                ov1_mo = contract("xov,qv,po->xpq", ov1, orbvb[:, no:].conj(), orboa[:, nc:nc + no])
                oo1_mo = contract("xov,qv,po->xpq", oo1, orbvb[:, :no].conj(), orboa[:, nc:nc + no])

                _, v1_cv_k = vresp_hf(cv1_mo)
                v1_co_j, v1_co_k = vresp_hf(co1_mo)
                v1_ov_j, v1_ov_k = vresp_hf(ov1_mo)
                _, v1_oo_k = vresp_hf(oo1_mo)

                v1_co_j = contract("xpq,po,qv->xov", v1_co_j, orboa.conj(), orbvb)
                v1_ov_j = contract("xpq,po,qv->xov", v1_ov_j, orboa.conj(), orbvb)
                v1_cv_k = contract("xpq,po,qv->xov", v1_cv_k, orboa.conj(), orbvb)
                v1_co_k = contract("xpq,po,qv->xov", v1_co_k, orboa.conj(), orbvb)
                v1_ov_k = contract("xpq,po,qv->xov", v1_ov_k, orboa.conj(), orbvb)
                v1_oo_k = contract("xpq,po,qv->xov", v1_oo_k, orboa.conj(), orbvb)

                vs_dA[:, :nc, no:] += (
                    contract("ab,xib->xia", fockB_hf[nc + no:, nc + no:], zs[:, :nc, no:])
                    - contract("ab,xib->xia", fockA_hf[nc + no:, nc + no:], zs[:, :nc, no:])
                    + contract("ji,xja->xia", fockB_hf[:nc, :nc], zs[:, :nc, no:])
                    - contract("ji,xja->xia", fockA_hf[:nc, :nc], zs[:, :nc, no:])
                ) / (2 * si)
                vs_dA[:, :nc, :no] += -v1_co_j[:, :nc, :no] / (2 * si - 1) + (
                    contract("ji,xju->xiu", fockB_hf[:nc, :nc], zs[:, :nc, :no])
                    - contract("ji,xju->xiu", fockA_hf[:nc, :nc], zs[:, :nc, :no])
                ) / (2 * si - 1)
                vs_dA[:, nc:, no:] += -v1_ov_j[:, nc:, no:] / (2 * si - 1) + (
                    contract("ab,xub->xua", fockB_hf[nc + no:, nc + no:], zs[:, nc:, no:])
                    - contract("ab,xub->xua", fockA_hf[nc + no:, nc + no:], zs[:, nc:, no:])
                ) / (2 * si - 1)

                if self.SA > 1:
                    vs_dA[:, :nc, no:] += factor1 * (
                        -v1_co_k[:, :nc, no:]
                        + contract("av,xiv->xia", fockB_hf[nc + no:, nc:nc + no], zs[:, :nc, :no])
                    )
                    vs_dA[:, :nc, :no] += factor1 * (
                        -v1_cv_k[:, :nc, :no]
                        + contract("av,xja->xjv", fockB_hf[nc + no:, nc:nc + no], zs[:, :nc, no:])
                    )
                    vs_dA[:, :nc, no:] += factor1 * (
                        -v1_ov_k[:, :nc, no:]
                        - contract("vi,xva->xia", fockA_hf[nc:nc + no, :nc], zs[:, nc:, no:])
                    )
                    vs_dA[:, nc:, no:] += factor1 * (
                        -v1_cv_k[:, nc:, no:]
                        - contract("vi,xib->xvb", fockA_hf[nc:nc + no, :nc], zs[:, :nc, no:])
                    )
                    vs_dA[:, :nc, :no] += (v1_ov_j[:, :nc, :no] - v1_ov_k[:, :nc, :no]) / (2 * si - 1)
                    vs_dA[:, nc:, no:] += (v1_co_j[:, nc:, no:] - v1_co_k[:, nc:, no:]) / (2 * si - 1)

                if self.SA > 2:
                    vs_dA[:, :nc, no:] += foo * (
                        -(factor2 - 1) * v1_oo_k[:, :nc, no:]
                        + (factor2 / (2 * si)) * (
                            contract("ia,xvv->xia", fockB_hf[:nc, nc + no:], zs[:, nc:, :no])
                            - contract("ia,xvv->xia", fockA_hf[:nc, nc + no:], zs[:, nc:, :no])
                        )
                    )
                    vs_dA[:, nc:, :no] += foo * (
                        -(factor2 - 1) * v1_cv_k[:, nc:, :no]
                        + (factor2 / (2 * si)) * (
                            contract("vw,ia,xia->xvw", iden_O, fockB_hf[:nc, nc + no:], zs[:, :nc, no:])
                            - contract("vw,ia,xia->xvw", iden_O, fockA_hf[:nc, nc + no:], zs[:, :nc, no:])
                        )
                    )
                    vs_dA[:, :nc, :no] += foo * (
                        factor3 * (
                            -v1_oo_k[:, :nc, :no]
                            - contract("iw,xwu->xiu", fockA_hf[:nc, nc:nc + no], zs[:, nc:, :no])
                        )
                        + factor4 * contract("vw,iu,xvw->xiu", iden_O, fockB_hf[:nc, nc:nc + no], zs[:, nc:, :no])
                    )
                    vs_dA[:, nc:, :no] += foo * (
                        factor3 * (
                            -v1_co_k[:, nc:, :no]
                            - contract("iw,xiv->xwv", fockA_hf[:nc, nc:nc + no], zs[:, :nc, :no])
                        )
                        + factor4 * contract("vw,iu,xiu->xvw", iden_O, fockB_hf[:nc, nc:nc + no], zs[:, :nc, :no])
                    )
                    vs_dA[:, nc:, no:] += foo * (
                        factor3 * (
                            -v1_oo_k[:, nc:, no:]
                            + contract("av,xuv->xua", fockB_hf[nc + no:, nc:nc + no], zs[:, nc:, :no])
                        )
                        - factor4 * contract("vw,au,xvw->xua", iden_O, fockA_hf[nc + no:, nc:nc + no], zs[:, nc:, :no])
                    )
                    vs_dA[:, nc:, :no] += foo * (
                        factor3 * (
                            -v1_ov_k[:, nc:, :no]
                            + contract("av,xwa->xwv", fockB_hf[nc + no:, nc:nc + no], zs[:, nc:, no:])
                        )
                        - factor4 * contract("vw,au,xua->xwv", iden_O, fockA_hf[nc + no:, nc:nc + no], zs[:, nc:, no:])
                    )
                vs += fglobal * vs_dA

            hx = vs.reshape(zs.shape[0], -1)
            return self._compress_removed_vectors(hx) if self.re else hx

        return vind, hdiag

    def init_guess(self, nstates, hdiag=None):
        if hdiag is None:
            hdiag = _spinflip_gaps(self.ctx, self.isf)
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
        self.v = self._order_davidson_vectors()
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

    def deltaS2_U(self, nstate):
        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)
        mo_coeff = _asnumpy(ctx.mo_coeff)
        occidx_a = _asnumpy(ctx.occidx_a)
        occidx_b = _asnumpy(ctx.occidx_b)
        viridx_a = _asnumpy(ctx.viridx_a)
        viridx_b = _asnumpy(ctx.viridx_b)

        mooa = mo_coeff[0][:, occidx_a]
        moob = mo_coeff[1][:, occidx_b]
        mova = mo_coeff[0][:, viridx_a]
        movb = mo_coeff[1][:, viridx_b]
        ovlp = _asnumpy(_get_ovlp(mf))

        sab_oo = mooa.conj().T @ ovlp @ moob
        sba_oo = sab_oo.conj().T
        sba_vo = movb.conj().T @ ovlp @ mooa

        x_cv, x_co, x_ov, x_oo = self._split_analysis_vectors(_asnumpy(self.v[:, nstate]))
        x_ba = np.concatenate([np.hstack([x_co, x_cv]), np.hstack([x_oo, x_ov])], axis=0).T
        ds2 = (
            np.einsum("ai,aj,jk,ki", x_ba.conj(), x_ba, sba_oo.T.conj(), sba_oo)
            - np.einsum("ai,bi,kb,ak", x_ba.conj(), x_ba, sba_vo.T.conj(), sba_vo)
            + np.einsum("ai,bj,jb,ai", x_ba.conj(), x_ba, sba_vo.T.conj(), sba_vo)
        )
        return np.real_if_close(ds2)

    def deltaS2(self):
        ds2 = []
        for nstate in range(self.nstates):
            value = _asnumpy(self.v[:, nstate])
            x_cv, _, _, x_oo = self._split_analysis_vectors(value)
            if self.SA == 0 and not self.type_u:
                dp_ab = np.sum(x_cv * x_cv) - np.sum(x_oo * x_oo) + np.sum(np.diag(x_oo)) ** 2
                ds2.append(-2.0 * self.ground_s + 1.0 + dp_ab)
            elif self.type_u:
                ds2.append(self.deltaS2_U(nstate) - self.no + 1.0)
            else:
                ds2.append(np.nan)
        return np.asarray(np.real_if_close(ds2), dtype=float)

    def analyse(self, threshold=0.05):
        energies = _asnumpy(self.e) * ha2eV
        self.dS2 = self.deltaS2()
        for nstate in range(self.nstates):
            x_cv, x_co, x_ov, x_oo = self._split_analysis_vectors(_asnumpy(self.v[:, nstate]))
            if np.isfinite(self.dS2[nstate]):
                print(f"Excited state {nstate + 1} {energies[nstate]:10.5f} eV D<S^2>={self.dS2[nstate]:8.4f}")
            else:
                print(f"Excited state {nstate + 1} {energies[nstate]:10.5f} eV D<S^2>=     n/a")

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

    def kernel(self, nstates=1, remove=None, frozen=None, foo=1.0, d_lda=0.3, fglobal=None, fit=True):
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
        return _asnumpy(self.e[:self.nstates] * ha2eV), self.v[:, :self.nstates]
