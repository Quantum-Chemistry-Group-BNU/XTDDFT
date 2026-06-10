import numpy as np
from pyscf import ao2mo
from pyscf.dft import numint2c
from pyscf.pbc.dft import numint2c as pbc_numint2c
from .hxc_part import (
    cache_xc_kernel_sf_mc,
    gen_response_sf,
    gen_response_sf_mc,
    AldA0,
) 
from .base import (
    XTDDFT_base,
    _build_initial_guess_from_gaps,
    _df_ao2mo_pbc,
    _get_gamma_kpt,
    _get_mo_fock,
    _is_pbc_mf,
    _iter_block_data,
    _make_spinflip_problem,
    _make_spinflip_vind,
    _molecular_dipole_integrals,
    _run_davidson,
    _spinflip_gaps,
    _as_cpu_mf,
    _as_cpu_ctx,
    _system,
    _make_reference_dm,
    _response_max_memory,
)
from ..utils.backend import _asnumpy, backend, contract, require_cupy, set_backend, xp
from ..utils.unit import ha2eV

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger(__name__)

def add_hf_a_b2a(a_b2a, mf, orbo_b, orbv_a, nc, nv, hyb=1, omega=None):
    # 考虑SF_TDA_UP时，仅有CV激发，因此，只需要考虑这个空间; K矩阵中含有的精确交换部分
    if abs(hyb) < 1e-14 or nc == 0 or nv == 0:
        return a_b2a

    if _is_pbc_mf(mf):
        eri_mo = _df_ao2mo_pbc(
            mf, [orbo_b, orbo_b, orbv_a, orbv_a],
            omega=omega, compact=False,
        )
    else:
        if omega is not None and abs(omega) >= 1e-14:
            raise NotImplementedError("Range-separated molecular HF exchange is not implemented in SF_TDA_up.")
        eri_mo = ao2mo.general(mf.mol, [orbo_b, orbo_b, orbv_a, orbv_a], compact=False)

    eri_mo = np.asarray(eri_mo).reshape(nc, nc, nv, nv)
    a_b2a -= contract('ijba->iajb', eri_mo) * hyb
    return a_b2a

def construct_xc_b2a(ao, orbo_b, orbv_a, fxc_ab):
    rho_v_a = contract('rp,pi->ri', ao, orbv_a)
    rho_o_b = contract('rp,pi->ri', ao, orbo_b)
    rho_ov_b2a = contract('ri,ra->ria', rho_o_b, rho_v_a)
    w_ov = contract('ria,r->ria', rho_ov_b2a, fxc_ab)
    iajb = contract('ria,rjb->iajb', rho_ov_b2a, w_ov)
    return iajb

def _pair_hessian_block_b2a(occ_fock_b, vir_fock_a, tensor_block):
    nocc_b = occ_fock_b.shape[0]
    nvir_a = vir_fock_a.shape[0]
    return (
        contract('ij,ab->iajb', np.eye(nocc_b), vir_fock_a)
        - contract('ji,ab->iajb', occ_fock_b, np.eye(nvir_a))
        + tensor_block.reshape(nocc_b, nvir_a, nocc_b, nvir_a)
    ).reshape(nocc_b * nvir_a, nocc_b * nvir_a)

class SF_TDA_up(XTDDFT_base): # just for ROKS
    def __init__(self, mf, method, davidson=True, davidson_backend="cpu", df_cache=None):
        davidson_backend = davidson_backend.lower()
        if davidson_backend not in ("cpu", "gpu", "auto"):
            raise ValueError("davidson_backend must be 'cpu', 'gpu', or 'auto'")
        super().__init__(mf, method, davidson=davidson, df_cache=df_cache)
        logger.info("SF_TDA_up method=0 ALDA0, method=1 multicollinear")
        self.isf = 1
        self.davidson_backend = "cpu" if davidson_backend == "auto" else davidson_backend

    def _result_method_label(self):
        return {0: "ALDA0", 1: "MCOL"}.get(self.method, f"method{self.method}")

    def get_Amat_ALDA0(self):
        # Dense Amat is always built with CPU PySCF/NumPy.
        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)

        mode = backend.mode
        set_backend("cpu")
        try:
            a_b2a = np.zeros((ctx.nc, ctx.nv, ctx.nc, ctx.nv))

            if self.mfxctype is not None:
                ni = mf._numint
                if self.hyb != 0:
                    a_b2a = add_hf_a_b2a(
                        a_b2a, mf, ctx.orbo_b, ctx.orbv_a, ctx.nc, ctx.nv, self.hyb
                    )
                if abs(self.omega) > 1e-10:
                    a_b2a = add_hf_a_b2a(
                        a_b2a, mf, ctx.orbo_b, ctx.orbv_a, ctx.nc, ctx.nv,
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
                        a_b2a += construct_xc_b2a(ao, ctx.orbo_b, ctx.orbv_a, fxc_ab)

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
                        a_b2a += construct_xc_b2a(ao, ctx.orbo_b, ctx.orbv_a, fxc_ab)

            else:
                a_b2a = add_hf_a_b2a(a_b2a, mf, ctx.orbo_b, ctx.orbv_a, ctx.nc, ctx.nv, hyb=1)

            focka_mo, fockb_mo = _get_mo_fock(mf, ctx.mo_coeff, ctx.mo_occ, force_cpu=True)
            amat = _pair_hessian_block_b2a(
                fockb_mo[:ctx.nc, :ctx.nc],
                focka_mo[ctx.nocc_a:, ctx.nocc_a:],
                a_b2a,
            )
        finally:
            set_backend(mode)

        self.A = np.asarray(amat)
        return self.A
    
    def get_Amat_MCOL(self, collinear_samples=30):
        # Dense Amat is always built with CPU PySCF/NumPy.
        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)

        mode = backend.mode
        set_backend("cpu")
        try:
            a_b2a = np.zeros((ctx.nc, ctx.nv, ctx.nc, ctx.nv))

            if self.mfxctype is not None:
                ni0 = mf._numint
                ni = pbc_numint2c.NumInt2C() if _is_pbc_mf(mf) else numint2c.NumInt2C()
                ni.collinear = 'mcol'
                ni.collinear_samples = collinear_samples

                if self.hyb != 0:
                    a_b2a = add_hf_a_b2a(
                        a_b2a, mf, ctx.orbo_b, ctx.orbv_a, ctx.nc, ctx.nv, self.hyb
                    )
                if abs(self.omega) > 1e-10:
                    a_b2a = add_hf_a_b2a(
                        a_b2a, mf, ctx.orbo_b, ctx.orbv_a, ctx.nc, ctx.nv,
                        self.alpha - self.hyb, omega=self.omega
                    )

                max_memory = _response_max_memory(mf, None)

                if collinear_samples >= 0:
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
                            rho_o_b = contract('rp,pi->ri', ao, ctx.orbo_b)
                            rho_v_a = contract('rp,pi->ri', ao, ctx.orbv_a)
                            rho_ov_b2a = contract('ri,ra->ria', rho_o_b, rho_v_a)
                            w_ov = contract('ria,r->ria', rho_ov_b2a, wfxc * 2.0)
                            iajb = contract('ria,rjb->iajb', rho_ov_b2a, w_ov)
                            a_b2a += iajb
                    elif self.xctype == 'GGA':
                        ao_deriv = 1
                        for ao, mask, weight, coords in _iter_block_data(mf, ni, ao_deriv, max_memory, force_cpu=True):
                            p0 = p1
                            p1 += weight.shape[0]
                            wfxc = fxc[..., p0:p1] * weight
                            rho_o_b = contract('xrp,pi->xri', ao, ctx.orbo_b)
                            rho_v_a = contract('xrp,pi->xri', ao, ctx.orbv_a)
                            rho_ov_b2a = contract('xri,ra->xria', rho_o_b, rho_v_a[0])
                            rho_ov_b2a[1:4] += contract('ri,xra->xria', rho_o_b[0], rho_v_a[1:4])
                            w_ov = contract('xyr,xria->yria', wfxc * 2.0, rho_ov_b2a)
                            iajb = contract('xria,xrjb->iajb', w_ov, rho_ov_b2a)
                            a_b2a += iajb
                    elif self.xctype == 'MGGA':
                        ao_deriv = 1
                        for ao, mask, weight, coords in _iter_block_data(mf, ni, ao_deriv, max_memory, force_cpu=True):
                            p0 = p1
                            p1 += weight.shape[0]
                            wfxc = fxc[..., p0:p1] * weight
                            rho_ob = contract('xrp,pi->xri', ao, ctx.orbo_b)
                            rho_va = contract('xrp,pi->xri', ao, ctx.orbv_a)
                            rho_ov_b2a = contract('xri,ra->xria', rho_ob, rho_va[0])
                            rho_ov_b2a[1:4] += contract('ri,xra->xria', rho_ob[0], rho_va[1:4])
                            tau_ov_b2a = contract('xri,xra->ria', rho_ob[1:4], rho_va[1:4]) * 0.5
                            rho_ov_b2a = np.vstack([rho_ov_b2a, tau_ov_b2a[np.newaxis]])
                            w_ov = contract('xyr,xria->yria', wfxc * 2.0, rho_ov_b2a)
                            iajb = contract('xria,xrjb->iajb', w_ov, rho_ov_b2a)
                            a_b2a += iajb
                    elif self.xctype == 'HF':
                        pass
                    elif self.xctype == 'NLC':
                        raise NotImplementedError('NLC functional is not supported here.')
                    else:
                        raise NotImplementedError(f'Unsupported xctype: {self.xctype}')
            else:
                a_b2a = add_hf_a_b2a(a_b2a, mf, ctx.orbo_b, ctx.orbv_a, ctx.nc, ctx.nv, hyb=1)

            focka_mo, fockb_mo = _get_mo_fock(mf, ctx.mo_coeff, ctx.mo_occ, force_cpu=True)
            amat = _pair_hessian_block_b2a(
                fockb_mo[:ctx.nc, :ctx.nc],
                focka_mo[ctx.nocc_a:, ctx.nocc_a:],
                a_b2a,
            )
        finally:
            set_backend(mode)

        self.A = np.asarray(amat)
        return self.A
    
    def get_Amat(self):
        if self.method == 1:  # multicollinear
            self.get_Amat_MCOL()
        else:
            self.get_Amat_ALDA0()
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
    
    def gen_tda_operation_sf(self):
        if self.method == 1:
            vresp = gen_response_sf_mc(self.mf,hermi=0,collinear_samples=50,ctx=self.ctx)
        else:
            vresp = gen_response_sf(self.mf,hermi=0,ctx=self.ctx)
        problem = _make_spinflip_problem(self.ctx, self._get_fock_mo(), self.isf)
        return _make_spinflip_vind(problem, vresp), problem.hdiag
    
    def init_guess(self, nstates):
        return _build_initial_guess_from_gaps(_spinflip_gaps(self.ctx, self.isf), nstates)
    
    def davidson_process(self, nstates):
        vind, hdiag = self.gen_tda_operation_sf()
        nroots = min(nstates, int(hdiag.size))
        x0 = self.init_guess(nroots)

        converged, e, x1 = _run_davidson(
            self.mf, self.davidson_backend,
            vind, hdiag, x0, nroots,
        )
        self.converged = converged
        self.e = xp.asarray(e)
        self.v = xp.asarray(_asnumpy(x1)).T
        logger.info('SF_TDA_up Davidson converged: {}', converged)
        return self.e, self.v

    def analyse(self):
        for nstate in range(self.nstates):
            value = _asnumpy(self.v[:, nstate])
            x_cv = value[:self.nc * self.nv].reshape(self.nc, self.nv)
            print(f'Excited state {nstate+1} {float(_asnumpy(self.e)[nstate])*ha2eV:10.5f} eV')
            for occ, vir in zip(*np.where(abs(x_cv) > 0.1)):
                vir_label = vir + 1 + self.nc + self.no
                print(f'{100*x_cv[occ, vir]**2:3.0f}% CV(ab) '
                      f'{occ+1}a -> {vir_label}b {x_cv[occ, vir]:10.5f}')
            print(' ')

    def transition_dipole_matrix(self):
        """Excited-state to excited-state transition dipole matrix in a.u."""
        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)
        dip_ao = _molecular_dipole_integrals(mf)
        mo_coeff = _asnumpy(ctx.mo_coeff)
        ints_aa = contract("xpq,pi,qj->xij", dip_ao, mo_coeff[0].conj(), mo_coeff[0])
        ints_bb = contract("xpq,pi,qj->xij", dip_ao, mo_coeff[1].conj(), mo_coeff[1])
        vir_a = _asnumpy(ctx.viridx_a)
        occ_b = _asnumpy(ctx.occidx_b)
        r_vv_a = np.asarray(ints_aa[:, vir_a][:, :, vir_a])
        r_oo_b = np.asarray(ints_bb[:, occ_b][:, :, occ_b])

        vectors = _asnumpy(self.v)
        nstates = min(self.nstates, vectors.shape[1])
        amps = [
            vectors[:, istate].reshape(self.nc, self.nv)
            for istate in range(nstates)
        ]
        tdm = np.zeros((nstates, nstates, 3))
        for i, c0 in enumerate(amps):
            for j, c1 in enumerate(amps):
                tdm[i, j] = (
                    np.einsum("ia,xab,ib->x", c0, r_vv_a, c1, optimize=True)
                    - np.einsum("ia,xij,ja->x", c0, r_oo_b, c1, optimize=True)
                )
        return tdm

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

    def kernel(self, nstates=1, save=False, save_file=None):
        self.nstates = nstates
        if self.davidson:
            self.davidson_process(nstates)
        else:
            self.get_Amat()
            self._diagonalize_dense(self.A, nstates)
        if save:
            self.save_results(save_file)
        return _asnumpy(self.e[:nstates] * ha2eV), self.v[:, :nstates]
