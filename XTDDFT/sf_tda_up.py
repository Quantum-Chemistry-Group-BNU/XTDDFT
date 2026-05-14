import numpy as np
from pyscf import ao2mo, lib
from opt_einsum import contract

from XTDDFT.base import (
    XTDDFT_base,
    _build_sf_context,
    _get_gamma_kpt,
    _get_mo_fock,
    _is_pbc_mf,
)
from utils.backend import _asnumpy, backend, set_backend, xp
from utils.unit import ha2eV

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger(__name__)


def _as_cpu_mf(mf):
    return mf.to_cpu() if hasattr(mf, "to_cpu") else mf


def _as_cpu_ctx(mf):
    mode = backend.mode
    set_backend("cpu")
    try:
        ctx = _build_sf_context(mf)
    finally:
        set_backend(mode)
    for key, value in vars(ctx).items():
        if hasattr(value, "get"):
            setattr(ctx, key, value.get())
        elif key not in ("mf", "cell", "mol"):
            setattr(ctx, key, np.asarray(value))
    return ctx


def _system(mf):
    return mf.cell if _is_pbc_mf(mf) else mf.mol


def add_hf_mol(a_b2a, mf, orbo_b, orbv_a, nc, nv, hyb=1, omega=None):
    # 考虑SF_TDA_UP时，仅有CV激发，因此，只需要考虑这个空间; K矩阵中含有的精确交换部分
    if abs(hyb) < 1e-14 or nc == 0 or nv == 0:
        return a_b2a

    if _is_pbc_mf(mf):
        kpt = _get_gamma_kpt(mf)
        if omega is None or abs(omega) < 1e-14:
            eri_mo = mf.with_df.ao2mo([orbo_b, orbo_b, orbv_a, orbv_a], kpt, compact=False)
        else:
            with mf.with_df.range_coulomb(omega) as rsh_df:
                eri_mo = rsh_df.ao2mo([orbo_b, orbo_b, orbv_a, orbv_a], kpt, compact=False)
    else:
        if omega is not None and abs(omega) >= 1e-14:
            raise NotImplementedError("Range-separated molecular HF exchange is not implemented in SF_TDA_up.")
        eri_mo = ao2mo.general(mf.mol, [orbo_b, orbo_b, orbv_a, orbv_a], compact=False)

    eri_mo = np.asarray(eri_mo).reshape(nc, nc, nv, nv)
    a_b2a -= contract('ijba->iajb', eri_mo) * hyb
    return a_b2a


def construct_xc(ao, orbo_b, orbv_a, fxc_ab):
    rho_v_a = contract('rp,pi->ri', ao, orbv_a)
    rho_o_b = contract('rp,pi->ri', ao, orbo_b)
    rho_ov_b2a = contract('ri,ra->ria', rho_o_b, rho_v_a)
    w_ov = contract('ria,r->ria', rho_ov_b2a, fxc_ab)
    iajb = contract('ria,rjb->iajb', rho_ov_b2a, w_ov)
    return iajb


def _make_reference_dm(mf, mo_occ):
    dm0 = mf.make_rdm1()
    if np.asarray(mf.mo_coeff).ndim == 2:
        dm0.mo_coeff = (mf.mo_coeff, mf.mo_coeff)
        dm0.mo_occ = mo_occ
    return dm0


def _block_loop(mf, ni, ao_deriv, max_memory):
    if _is_pbc_mf(mf):
        return ni.block_loop(
            mf.cell, mf.grids, mf.cell.nao_nr(), ao_deriv,
            _get_gamma_kpt(mf), None, max_memory
        )
    return ni.block_loop(mf.mol, mf.grids, mf.mol.nao_nr(), ao_deriv, max_memory)


def _iter_block_data(mf, ni, ao_deriv, max_memory):
    if _is_pbc_mf(mf):
        for ao, ao_k2, mask, weight, coords in _block_loop(mf, ni, ao_deriv, max_memory):
            yield ao, mask, weight, coords
    else:
        yield from _block_loop(mf, ni, ao_deriv, max_memory)


def _pair_hessian_block(occ_fock, vir_fock, tensor_block):
    nocc = occ_fock.shape[0]
    nvir = vir_fock.shape[0]
    return (
        contract('ij,ab->iajb', np.eye(nocc), vir_fock)
        - contract('ji,ab->iajb', occ_fock, np.eye(nvir))
        + tensor_block.reshape(nocc, nvir, nocc, nvir)
    ).reshape(nocc * nvir, nocc * nvir)


class SF_TDA_up(XTDDFT_base): # just for ROKS
    isf = 1

    def get_Amat(self):
        # Dense Amat is built with PySCF/NumPy.  GPU inputs are converted to CPU
        # for construction, then converted back to the active backend at return.
        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf)

        mode = backend.mode
        set_backend("cpu")
        try:
            a_b2a = np.zeros((ctx.nc, ctx.nv, ctx.nc, ctx.nv))

            try:
                xctype = mf.xc
            except AttributeError:
                xctype = None

            if xctype is not None:
                ni = mf._numint
                ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
                if getattr(mf, "nlc", None) or ni.libxc.is_nlc(mf.xc):
                    logger.warning(
                        'NLC functional found in DFT object. Its second '
                        'derivative is not available and is not included in '
                        'the response function.'
                    )

                omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, _system(mf).spin)
                if hyb != 0:
                    a_b2a = add_hf_mol(
                        a_b2a, mf, ctx.orbo_b, ctx.orbv_a, ctx.nc, ctx.nv, hyb
                    )
                if abs(omega) > 1e-10:
                    a_b2a = add_hf_mol(
                        a_b2a, mf, ctx.orbo_b, ctx.orbv_a, ctx.nc, ctx.nv,
                        alpha - hyb, omega=omega
                    )

                xctype = ni._xc_type(mf.xc)
                dm0 = _make_reference_dm(mf, ctx.mo_occ)
                make_rho = ni._gen_rho_evaluator(_system(mf), dm0, hermi=0, with_lapl=False)[0]
                mem_now = lib.current_memory()[0]
                max_memory = max(2000, mf.max_memory * .8 - mem_now)

            if xctype == 'LDA' and not getattr(self, "collinear", False):
                ao_deriv = 0
                for ao, mask, weight, coords in _iter_block_data(mf, ni, ao_deriv, max_memory):
                    rho0a = make_rho(0, ao, mask, xctype)
                    rho0b = make_rho(1, ao, mask, xctype)
                    rho = (rho0a, rho0b)
                    vxc = ni.eval_xc_eff(mf.xc, rho, deriv=1, omega=omega, xctype=xctype)[1]
                    vxc_a = vxc[0, 0] * weight
                    vxc_b = vxc[1, 0] * weight
                    fxc_ab = (vxc_a - vxc_b) / (rho0a - rho0b + 1e-9)
                    a_b2a += construct_xc(ao, ctx.orbo_b, ctx.orbv_a, fxc_ab)

            elif xctype == 'GGA' and not getattr(self, "collinear", False):  # 进行简化
                ao_deriv = 0
                for ao, mask, weight, coords in _iter_block_data(mf, ni, ao_deriv, max_memory):
                    # 这里只需要 density，不需要 gradient
                    rho0a = make_rho(0, ao, mask, 'LDA')
                    rho0b = make_rho(1, ao, mask, 'LDA')
                    # 为 GGA eval_xc_eff 构造 shape = (4, ngrids) 的输入
                    rha = np.zeros((4, rho0a.size))
                    rhb = np.zeros((4, rho0b.size))
                    rha[0] = rho0a
                    rhb[0] = rho0b
                    vxc = ni.eval_xc_eff(
                        mf.xc, (rha, rhb), deriv=1, omega=omega, xctype=xctype
                    )[1]
                    vxc_a = vxc[0, 0] * weight
                    vxc_b = vxc[1, 0] * weight
                    fxc_ab = (vxc_a - vxc_b) / (rho0a - rho0b + 1e-9)
                    a_b2a += construct_xc(ao, ctx.orbo_b, ctx.orbv_a, fxc_ab)

            elif xctype is None:
                a_b2a = add_hf_mol(a_b2a, mf, ctx.orbo_b, ctx.orbv_a, ctx.nc, ctx.nv, hyb=1)

            focka_mo, fockb_mo = (
                _asnumpy(x) for x in _get_mo_fock(mf, ctx.mo_coeff, ctx.mo_occ)
            )
            amat = _pair_hessian_block(
                fockb_mo[:ctx.nc, :ctx.nc],
                focka_mo[ctx.nocc_a:, ctx.nocc_a:],
                a_b2a,
            )
        finally:
            set_backend(mode)

        self.A = xp.asarray(amat)
        return self.A

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

