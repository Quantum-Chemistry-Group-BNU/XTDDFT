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
    _as_cpu_sf_context,
    _build_initial_guess_from_gaps,
    _build_sf_context,
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
)
from utils.backend import _asnumpy, backend, require_cupy, set_backend, xp
from utils.unit import ha2eV

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger(__name__)


def _as_cpu_mf(mf):
    return mf.to_cpu() if hasattr(mf, "to_cpu") else mf


def _as_cpu_ctx(mf, ctx=None):
    mode = backend.mode
    set_backend("cpu")
    try:
        ctx = _build_sf_context(mf) if ctx is None else _as_cpu_sf_context(ctx, mf)
    finally:
        set_backend(mode)
    return ctx


def _system(mf):
    return mf.cell if _is_pbc_mf(mf) else mf.mol


def _is_ks_mf(mf):
    return hasattr(mf, "_numint") and hasattr(mf, "xc")


def _response_max_memory(mf, max_memory):
    if max_memory is not None:
        return max_memory
    mem_now = lib.current_memory()[0]
    return max(2000, mf.max_memory * .8 - mem_now)


def _xc_response_params(mf, ni, xc_code=None):
    xc_code = mf.xc if xc_code is None else xc_code
    xctype = ni._xc_type(xc_code)
    if xctype == "HF":
        return xctype, True, 0.0, 0.0, 1.0
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc_code, _system(mf).spin)
    return xctype, ni.libxc.is_hybrid_xc(xc_code), omega, alpha, hyb


def _xc_ao_deriv(xctype, with_lapl=False):
    if xctype == "GGA":
        return 1
    if xctype == "MGGA":
        return 2 if with_lapl else 1
    return 0


def _alda0_rho_and_denom(array_api, rho_a, rho_b, xctype):
    if xctype == "LDA":
        return array_api.stack([rho_a, rho_b]), rho_a - rho_b
    rha = array_api.zeros_like(rho_a)
    rhb = array_api.zeros_like(rho_b)
    rha[0] = rho_a[0]
    rhb[0] = rho_b[0]
    return array_api.stack([rha, rhb]), rho_a[0] - rho_b[0]


def _add_hybrid_k(mf, v1, dm1, hybrid, hyb, omega, alpha, hermi):
    if not hybrid:
        return v1
    vk = _get_k(mf, dm1, hermi=hermi) * hyb
    if abs(omega) > 1e-10:
        vk += _get_k(mf, dm1, hermi=hermi, omega=omega) * (alpha - hyb)
    return v1 - vk


def _make_gpu_response_vind(mf, xctype, hybrid, omega, alpha, hyb, hermi, apply_xc):
    cp = require_cupy()

    def vind(dm1):
        dm1 = cp.asarray(dm1)
        v1 = cp.zeros_like(dm1) if xctype == "HF" else apply_xc(dm1)
        return _add_hybrid_k(mf, v1, dm1, hybrid, hyb, omega, alpha, hermi)

    return vind


def _make_response_vxc(ni, mf, dm1, hermi, vxc, max_memory):
    return nr_uks_fxc_sf_tda(
        ni, mf, mf.grids, mf.xc, None, dm1, 0, hermi,
        vxc=vxc, max_memory=max_memory
    )


def add_hf_a_b2a(a_b2a, mf, orbo_b, orbv_a, nc, nv, hyb=1, omega=None):
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
    a_b2a -= contract('ijba->iajb', eri_mo, optimize=True) * hyb
    return a_b2a


def construct_xc(ao, orbo_b, orbv_a, fxc_ab):
    rho_v_a = contract('rp,pi->ri', ao, orbv_a, optimize=True)
    rho_o_b = contract('rp,pi->ri', ao, orbo_b, optimize=True)
    rho_ov_b2a = contract('ri,ra->ria', rho_o_b, rho_v_a, optimize=True)
    w_ov = contract('ria,r->ria', rho_ov_b2a, fxc_ab, optimize=True)
    iajb = contract('ria,rjb->iajb', rho_ov_b2a, w_ov, optimize=True)
    return iajb


def _make_reference_dm(mf, mo_occ):
    dm0 = mf.make_rdm1()
    if np.asarray(mf.mo_coeff).ndim == 2:
        dm0.mo_coeff = (mf.mo_coeff, mf.mo_coeff)
        dm0.mo_occ = mo_occ
    return dm0

def _iter_ao_blocks(mf, ni, ao_deriv, max_memory=None, mol=None, nao=None, kpt=None):
    if _is_gpu_mf(mf):
        if _is_pbc_mf(mf):
            _ensure_gamma_df(mf)
            kpt = _get_gamma_kpt(mf) if kpt is None else kpt
            for ao_ks, weight, coords in ni.block_loop(mf.cell, mf.grids, ao_deriv, kpt):
                yield SimpleNamespace(
                    ao=ao_ks[0], ao_ks=ao_ks, mask=None,
                    weight=weight, coords=coords, idx=None,
                )
        else:
            mol = mf.mol if mol is None else mol
            nao = mol.nao_nr() if nao is None else nao
            for ao, idx, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv):
                yield SimpleNamespace(
                    ao=ao, ao_ks=None, mask=None,
                    weight=weight, coords=coords, idx=idx,
                )
        return

    if _is_pbc_mf(mf):
        kpt = _get_gamma_kpt(mf) if kpt is None else kpt
        for ao, ao_k2, mask, weight, coords in ni.block_loop(
            mf.cell, mf.grids, mf.cell.nao_nr(), ao_deriv,
            kpt, None, max_memory,
        ):
            yield SimpleNamespace(
                ao=ao, ao_ks=(ao, ao_k2), mask=mask,
                weight=weight, coords=coords, idx=None,
            )
    else:
        for ao, mask, weight, coords in ni.block_loop(
            mf.mol, mf.grids, mf.mol.nao_nr(), ao_deriv, max_memory
        ):
            yield SimpleNamespace(
                ao=ao, ao_ks=None, mask=mask,
                weight=weight, coords=coords, idx=None,
            )


def _iter_block_data(mf, ni, ao_deriv, max_memory):
    for block in _iter_ao_blocks(mf, ni, ao_deriv, max_memory):
        yield block.ao, block.mask, block.weight, block.coords


def _make_rho_evaluator(ni, mf, dms, hermi, grids):
    if _is_pbc_mf(mf):
        return ni._gen_rho_evaluator(_system(mf), dms, hermi, False)[:2]
    return ni._gen_rho_evaluator(_system(mf), dms, hermi, False, grids)[:2]


def _pair_hessian_block(occ_fock, vir_fock, tensor_block):
    nocc = occ_fock.shape[0]
    nvir = vir_fock.shape[0]
    return (
        contract('ij,ab->iajb', np.eye(nocc), vir_fock, optimize=True)
        - contract('ji,ab->iajb', occ_fock, np.eye(nvir), optimize=True)
        + tensor_block.reshape(nocc, nvir, nocc, nvir)
    ).reshape(nocc * nvir, nocc * nvir)

def AldA0(ni, mf, rho, weight, xctype):
    vxc= ni.eval_xc_eff(mf.xc, rho, deriv=1, xctype=xctype)[1] # 
    vxc_a = vxc[0,0]*weight
    vxc_b = vxc[1,0]*weight
    fxc_ab = (vxc_a-vxc_b)/(np.array(rho[0])-np.array(rho[1])+1e-9)
    return fxc_ab

def cache_xc_kernel_sf(mf, mo_coeff, mo_occ, spin=1,max_memory=2000): # for ALDA0
    '''Compute the fxc_sf, which can be used in SF-TDDFT/TDA
    '''
    MGGA_DENSITY_LAPL = False
    with_lapl = MGGA_DENSITY_LAPL
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    if mo_coeff is None or mo_occ is None:
        _, mo_occ, mo_coeff = mf_info(mf)

    ao_deriv = _xc_ao_deriv(xctype, MGGA_DENSITY_LAPL)

    assert mo_coeff[0].ndim == 2
    assert spin == 1

    nao = mo_coeff[0].shape[0]
    dm0 = mf.make_rdm1()
    if np.array(mf.mo_coeff).ndim==2:
        dm0.mo_coeff = (mf.mo_coeff, mf.mo_coeff)
        dm0.mo_occ = mo_occ
    make_rho = ni._gen_rho_evaluator(_system(mf), dm0, hermi=0, with_lapl=False)[0]

    fxc_abs = []
    for ao, mask, weight, coords in _iter_block_data(mf, ni, ao_deriv, max_memory):
        rhoa = make_rho(0, ao, mask, xctype)
        rhob = make_rho(1, ao, mask, xctype)
        if xctype == 'LDA':
            rho = (rhoa,rhob)
        else: # GGA
            rha = np.zeros_like(rhoa)
            rhb = np.zeros_like(rhob)
            rha[0] = rhoa[0]
            rhb[0] = rhob[0]
            rho = (rha,rhb)
            rhoa = rhoa[0]
            rhob = rhob[0]
        fxc_ab = AldA0(ni, mf, rho, weight, xctype)
        fxc_abs += list(fxc_ab)
    fxc_abs = np.asarray(fxc_abs)
    return fxc_abs

def __mcfun_fn_eval_xc(ni, xc_code, xctype, rho, deriv):
    evfk = ni.eval_xc_eff(xc_code, rho, deriv=deriv, xctype=xctype)
    for order in range(1, deriv+1):
        if evfk[order] is not None:
            evfk[order] = xc_deriv.ud2ts(evfk[order])
    return evfk

# This function can be merged with pyscf.dft.numint2c.mcfun_eval_xc_adapter()
# This function should be a class function in the Numint2c class.
def mcfun_eval_xc_adapter_sf(ni, xc_code):
    '''Wrapper to generate the eval_xc function required by mcfun

    Kwargs:
        dim: int
            eval_xc_eff_sf is for mc collinear sf tddft/ tda case.add().
    '''

    try:
        import mcfun
    except ImportError:
        raise ImportError('This feature requires mcfun library.\n'
                          'Try install mcfun with `pip install mcfun`')

    xctype = ni._xc_type(xc_code)
    fn_eval_xc = functools.partial(__mcfun_fn_eval_xc, ni, xc_code, xctype)
    nproc = lib.num_threads()

    def eval_xc_eff(xc_code, rho, deriv=1, omega=None, xctype=None,
                verbose=None):
        return mcfun.eval_xc_eff_sf(
            fn_eval_xc, rho, deriv,
            collinear_samples=ni.collinear_samples, workers=nproc)
    return eval_xc_eff

def cache_xc_kernel_sf_mc(self, mf, mol, grids, xc_code, mo_coeff, mo_occ, deriv=2,spin=1,max_memory=2000):
    '''Compute the fxc_sf, which can be used in SF-TDDFT/TDA
    '''
    MGGA_DENSITY_LAPL = False
    xctype = self._xc_type(xc_code)
    ao_deriv = _xc_ao_deriv(xctype, MGGA_DENSITY_LAPL)
    with_lapl = MGGA_DENSITY_LAPL

    assert mo_coeff[0].ndim == 2
    assert spin == 1

    nao = mo_coeff[0].shape[0]
    rhoa = []
    rhob = []

    ni_eval = pbc_numint.NumInt() if _is_pbc_mf(mf) else numint.NumInt()
    ni_block = self if _is_pbc_mf(mf) else ni_eval
    for ao, mask, weight, coords in _iter_block_data(mf, ni_block, ao_deriv, max_memory):
        rhoa.append(ni_eval.eval_rho2(mol, ao, mo_coeff[0], mo_occ[0], mask, xctype, with_lapl))
        rhob.append(ni_eval.eval_rho2(mol, ao, mo_coeff[1], mo_occ[1], mask, xctype, with_lapl))
    rho_ab = (np.hstack(rhoa), np.hstack(rhob))
    rho_ab = np.asarray(rho_ab)
    rho_tmz = np.zeros_like(rho_ab)+1e-11
    rho_tmz[0] += rho_ab[0]+rho_ab[1]
    rho_tmz[1] += rho_ab[0]-rho_ab[1]
    eval_xc = mcfun_eval_xc_adapter_sf(self,xc_code)
    fxc_sf = eval_xc(xc_code, rho_tmz, deriv=2, xctype=xctype)
    return fxc_sf

def _cpu_xc_weighted_density(xctype, rho1sf, kernel_block, weight, method):
    if method == "alda0":
        if xctype == "LDA":
            return rho1sf * kernel_block
        rhosf = np.zeros_like(rho1sf)
        rhosf[0] += rho1sf[0]
        return rhosf * kernel_block
    if xctype == "LDA":
        return rho1sf * kernel_block[0, 0] * 2.0 * weight
    return contract("bg,abg->ag", rho1sf, kernel_block * 2.0, optimize=True) * weight


def _nr_uks_fxc_sf_tda_cpu_common(ni, mf, grids, xc_code, dms, hermi=0,
                                  kernel=None, method="alda0", max_memory=2000):
    if isinstance(dms, np.ndarray):
        dtype = dms.dtype
    else:
        dtype = np.result_type(*dms)
    if hermi != 1 and dtype != np.double:
        raise NotImplementedError('complex density matrix')

    xctype = ni._xc_type(xc_code)

    nao = dms.shape[-1]
    make_rhosf, nset = _make_rho_evaluator(ni, mf, dms, hermi, grids)

    def block_loop(ao_deriv):
        p1 = 0
        for ao, mask, weight, coords in _iter_block_data(mf, ni, ao_deriv, max_memory):
            p0, p1 = p1, p1 + weight.size
            kernel_block = kernel[p0:p1] if method == "alda0" else kernel[..., p0:p1]
            for i in range(nset):
                rho1sf = make_rhosf(i, ao, mask, xctype)
                wv = _cpu_xc_weighted_density(xctype, rho1sf, kernel_block, weight, method)
                yield i, ao, mask, wv

    ao_loc = _system(mf).ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))
    pair_mask = _system(mf).get_overlap_cond() < -np.log(ni.cutoff)
    vmat = np.zeros((nset,nao,nao))
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        for i, ao, mask, wv in block_loop(ao_deriv):
            _dot_ao_ao_sparse(ao, ao, wv, nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[i])
    elif xctype == 'GGA':
        ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5
            aow = _scale_ao_sparse(ao, wv, mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[i])

        # [(\nabla mu) nu + mu (\nabla nu)] * fxc_jb = ((\nabla mu) nu f_jb) + h.c.
        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(nset,nao,nao)

    elif xctype == 'MGGA':
        # assert not MGGA_DENSITY_LAPL
        ao_deriv = 1
        v1 = np.zeros_like(vmat)
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5
            wv[4] *= .5
            aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[i])
            _tau_dot_sparse(ao, ao, wv[4], nbins, mask, pair_mask, ao_loc, out=v1[i])

        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(nset,nao,nao)
        vmat += v1

    if isinstance(dms, np.ndarray) and dms.ndim == 2:
        vmat = vmat[:,0]
    if vmat.dtype != dtype:
        vmat = np.asarray(vmat, dtype=dtype)
    return vmat

def nr_uks_fxc_sf_tda(ni, mf, grids, xc_code, dm0, dms, relativity=0, hermi=0,
                      vxc=None, extype=0, max_memory=2000, verbose=None):
    del dm0, relativity, extype, verbose
    return _nr_uks_fxc_sf_tda_cpu_common(
        ni, mf, grids, xc_code, dms, hermi=hermi,
        kernel=vxc, method="alda0", max_memory=max_memory,
    )


def nr_uks_fxc_sf_tda_mc(ni, mf, grids, xc_code, dm0, dms, relativity=0, hermi=0,
                         rho0=None, vxc=None, fxc=None, extype=0,
                         max_memory=2000, verbose=None):
    del dm0, relativity, rho0, vxc, extype, verbose
    return _nr_uks_fxc_sf_tda_cpu_common(
        ni, mf, grids, xc_code, dms, hermi=hermi,
        kernel=fxc, method="mcol", max_memory=max_memory,
    )

def gen_response_sf(mf,hermi=0,max_memory=None,ctx=None):
    if ctx is None:
        _, mo_occ, mo_coeff = mf_info(mf)
    else:
        mo_occ, mo_coeff = ctx.mo_occ, ctx.mo_coeff
    if _is_gpu_mf(mf):
        return _gen_response_sf_gpu(mf, mo_coeff, mo_occ, hermi=hermi,
                                    max_memory=max_memory)

    if _is_ks_mf(mf):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        xctype, hybrid, omega, alpha, hyb = _xc_response_params(mf, ni)
        max_memory = _response_max_memory(mf, max_memory)
        vxc = cache_xc_kernel_sf(mf, mo_coeff, mo_occ,1,max_memory) # XC kerkel

        def vind(dm1):
            v1 = _make_response_vxc(ni, mf, dm1, hermi, vxc, max_memory)
            return _add_hybrid_k(mf, v1, dm1, hybrid, hyb, omega, alpha, hermi)
    else: # in HF case
        def vind(dm1):
            return -_get_k(mf, dm1, hermi=hermi)
    return vind


def _cache_xc_kernel_sf_gpu_mol(mf, mo_coeff, mo_occ, max_memory=2000):
    del max_memory
    cp = require_cupy()
    from gpu4pyscf.dft.numint import eval_rho2

    ni = mf._numint
    mol = mf.mol
    xctype = ni._xc_type(mf.xc)
    if xctype == "HF":
        return None
    if xctype not in ("LDA", "GGA", "MGGA"):
        raise NotImplementedError(f"GPU ALDA0 response is not implemented for {xctype}.")
    ao_deriv = _xc_ao_deriv(xctype)

    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)
    opt = getattr(ni, "gdftopt", None)
    if opt is None or mol not in [opt.mol, getattr(opt, "_sorted_mol", None)]:
        ni.build(mol, mf.grids.coords)
        opt = ni.gdftopt
    sorted_mol = opt._sorted_mol
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])

    rhoa = []
    rhob = []
    for block in _iter_ao_blocks(mf, ni, ao_deriv, mol=sorted_mol, nao=mol.nao_nr()):
        rhoa.append(eval_rho2(sorted_mol, block.ao, mo_coeff[0, block.idx, :], mo_occ[0], None, xctype, False))
        rhob.append(eval_rho2(sorted_mol, block.ao, mo_coeff[1, block.idx, :], mo_occ[1], None, xctype, False))

    rho_a = cp.hstack(rhoa)
    rho_b = cp.hstack(rhob)
    rho, denom = _alda0_rho_and_denom(cp, rho_a, rho_b, xctype)
    vxc = ni.eval_xc_eff(mf.xc, rho, deriv=1, xctype=xctype, spin=1)[1]
    # gpu4pyscf.tdscf._uhf_resp_sf.nr_uks_fxc_sf multiplies the spin-flip
    # kernel by 2.0 internally for the xx/yy channels.
    fxc_ab = (vxc[0, 0] - vxc[1, 0]) / (2.0 * (denom + 1e-9))
    return cp.pad(fxc_ab[None, None], ((0, 3), (0, 3), (0, 0)))


def _cache_xc_kernel_sf_gpu_pbc(mf, mo_coeff, mo_occ, max_memory=2000):
    del max_memory
    cp = require_cupy()
    _ensure_gamma_df(mf)
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    if xctype == "HF":
        return None
    if xctype not in ("LDA", "GGA", "MGGA"):
        raise NotImplementedError(f"GPU Gamma PBC ALDA0 response is not implemented for {xctype}.")
    ao_deriv = _xc_ao_deriv(xctype)

    dm = cp.asarray([
        (mo_coeff[s] * mo_occ[s]) @ mo_coeff[s].conj().T
        for s in range(2)
    ])
    fxc_abs = []
    for block in _iter_ao_blocks(mf, ni, ao_deriv):
        rho_a = ni.eval_rho(mf.cell, block.ao_ks, dm[0][None], xctype=xctype, hermi=0)
        rho_b = ni.eval_rho(mf.cell, block.ao_ks, dm[1][None], xctype=xctype, hermi=0)
        rho, denom = _alda0_rho_and_denom(cp, rho_a, rho_b, xctype)
        vxc = ni.eval_xc_eff(mf.xc, rho, deriv=1, xctype=xctype)[1]
        fxc_abs.append((vxc[0, 0] - vxc[1, 0]) * block.weight / (denom + 1e-9))
    return cp.hstack(fxc_abs) if fxc_abs else cp.asarray([])


def _cache_xc_kernel_sf_mc_gpu_mol(mf, mo_coeff, mo_occ, collinear_samples, max_memory=2000):
    del max_memory
    cp = require_cupy()
    from gpu4pyscf.dft.numint import eval_rho2
    from gpu4pyscf.tdscf._uhf_resp_sf import mcfun_eval_xc_adapter_sf

    ni = mf._numint
    mol = mf.mol
    xctype = ni._xc_type(mf.xc)
    ao_deriv = _xc_ao_deriv(xctype)

    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)
    opt = getattr(ni, "gdftopt", None)
    if opt is None or mol not in [opt.mol, getattr(opt, "_sorted_mol", None)]:
        ni.build(mol, mf.grids.coords)
        opt = ni.gdftopt
    sorted_mol = opt._sorted_mol
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])

    rhoa = []
    rhob = []
    for block in _iter_ao_blocks(mf, ni, ao_deriv, mol=sorted_mol, nao=mol.nao_nr()):
        rhoa.append(eval_rho2(sorted_mol, block.ao, mo_coeff[0, block.idx, :], mo_occ[0], None, xctype, False))
        rhob.append(eval_rho2(sorted_mol, block.ao, mo_coeff[1, block.idx, :], mo_occ[1], None, xctype, False))

    rho_ab = (cp.hstack(rhoa), cp.hstack(rhob))
    rho_z = cp.asarray([rho_ab[0] + rho_ab[1], rho_ab[0] - rho_ab[1]])
    eval_xc_eff = mcfun_eval_xc_adapter_sf(ni, mf.xc, collinear_samples)
    return eval_xc_eff(mf.xc, rho_z, deriv=2, xctype=xctype)[2]


def _gen_response_sf_mc_gpu_mol(mf, mo_coeff, mo_occ, hermi=0, collinear_samples=60, max_memory=None):
    from gpu4pyscf.tdscf._uhf_resp_sf import nr_uks_fxc_sf

    ni = mf._numint
    max_memory = _response_max_memory(mf, max_memory)
    xctype, hybrid, omega, alpha, hyb = _xc_response_params(mf, ni)
    fxc = None if xctype == "HF" else _cache_xc_kernel_sf_mc_gpu_mol(
        mf, mo_coeff, mo_occ, collinear_samples, max_memory
    )

    def apply_xc(dm1):
        return nr_uks_fxc_sf(
            ni, mf.mol, mf.grids, mf.xc, None, dm1,
            0, hermi, None, None, fxc,
        )

    return _make_gpu_response_vind(mf, xctype, hybrid, omega, alpha, hyb, hermi, apply_xc)


def _cache_xc_kernel_sf_mc_gpu_pbc(mf, mo_coeff, mo_occ, collinear_samples, max_memory=2000):
    del max_memory
    cp = require_cupy()
    _ensure_gamma_df(mf)

    ni = mf._numint
    ni_mc = pbc_numint2c.NumInt2C()
    ni_mc.collinear = "mcol"
    ni_mc.collinear_samples = collinear_samples
    ni_mc.libxc.test_deriv_order(mf.xc, 2, raise_error=True)

    xctype = ni_mc._xc_type(mf.xc)
    ao_deriv = _xc_ao_deriv(xctype)

    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)
    dm = cp.asarray([
        (mo_coeff[s] * mo_occ[s]) @ mo_coeff[s].conj().T
        for s in range(2)
    ])

    rhoa = []
    rhob = []
    for block in _iter_ao_blocks(mf, ni, ao_deriv):
        rhoa.append(_asnumpy(ni.eval_rho(mf.cell, block.ao_ks, dm[0][None], xctype=xctype, hermi=0)))
        rhob.append(_asnumpy(ni.eval_rho(mf.cell, block.ao_ks, dm[1][None], xctype=xctype, hermi=0)))

    rho_ab = np.asarray((np.hstack(rhoa), np.hstack(rhob)))
    rho_z = np.zeros_like(rho_ab) + 1e-11
    rho_z[0] += rho_ab[0] + rho_ab[1]
    rho_z[1] += rho_ab[0] - rho_ab[1]
    eval_xc = mcfun_eval_xc_adapter_sf(ni_mc, mf.xc)
    return eval_xc(mf.xc, rho_z, deriv=2, xctype=xctype)[2]


def _gpu_pbc_xc_weighted_density(xctype, rho1sf, kernel_block, weight, method):
    if method == "alda0":
        if xctype == "LDA":
            return rho1sf * kernel_block
        rhosf = rho1sf * 0
        rhosf[0] = rho1sf[0]
        return rhosf * kernel_block
    if xctype == "LDA":
        return rho1sf * kernel_block[0, 0] * (2.0 * weight)
    return contract("bg,abg->ag", rho1sf, kernel_block * 2.0, optimize=True) * weight


def _nr_uks_fxc_sf_tda_gpu_pbc_common(ni, mf, dm1, hermi=0, kernel=None, method="alda0"):
    cp = require_cupy()
    from gpu4pyscf.pbc.dft.numint import _scale_ao, _tau_dot

    _ensure_gamma_df(mf)
    xctype = ni._xc_type(mf.xc)
    if xctype not in ("LDA", "GGA", "MGGA"):
        raise NotImplementedError(f"GPU Gamma PBC {method} response is not implemented for {xctype}.")
    ao_deriv = _xc_ao_deriv(xctype)

    dm1 = cp.asarray(dm1)
    single = dm1.ndim == 2
    dms = dm1.reshape(1, dm1.shape[-2], dm1.shape[-1]) if single else dm1
    vmat = cp.zeros_like(dms)
    vtau = cp.zeros_like(dms) if xctype == "MGGA" else None
    p0 = p1 = 0
    for block in _iter_ao_blocks(mf, ni, ao_deriv):
        weight = cp.asarray(block.weight)
        p0, p1 = p1, p1 + int(weight.size)
        kernel_block = cp.asarray(
            kernel[p0:p1] if method == "alda0" else kernel[..., p0:p1]
        )
        for i in range(int(dms.shape[0])):
            rho1sf = ni.eval_rho(mf.cell, block.ao_ks, dms[i][None], xctype=xctype, hermi=hermi)
            wv = _gpu_pbc_xc_weighted_density(xctype, rho1sf, kernel_block, weight, method)
            if xctype == "LDA":
                aow = _scale_ao(block.ao, wv)
                vmat[i] += block.ao.conj().T.dot(aow)
            elif xctype == "GGA":
                wv[0] *= .5
                aow = _scale_ao(block.ao[:4], wv[:4])
                vmat[i] += block.ao[0].conj().T.dot(aow)
            elif xctype == "MGGA":
                wv[[0, 4]] *= .5
                aow = _scale_ao(block.ao[:4], wv[:4])
                vmat[i] += block.ao[0].conj().T.dot(aow)
                vtau[i] += _tau_dot(block.ao, block.ao, wv[4])

    if xctype in ("GGA", "MGGA"):
        vmat = vmat + vmat.conj().swapaxes(-2, -1)
    if vtau is not None:
        vmat += vtau
    return vmat[0] if single else vmat


def _nr_uks_fxc_sf_tda_mc_gpu_pbc(ni, mf, dm1, hermi=0, fxc=None):
    return _nr_uks_fxc_sf_tda_gpu_pbc_common(
        ni, mf, dm1, hermi=hermi, kernel=fxc, method="mcol"
    )


def _gen_response_sf_mc_gpu_pbc(mf, mo_coeff, mo_occ, hermi=0, collinear_samples=60, max_memory=None):
    _ensure_gamma_df(mf)

    ni = mf._numint
    max_memory = _response_max_memory(mf, max_memory)
    xctype, hybrid, omega, alpha, hyb = _xc_response_params(mf, ni)
    if xctype != "HF":
        fxc = _cache_xc_kernel_sf_mc_gpu_pbc(
            mf, mo_coeff, mo_occ, collinear_samples, max_memory
        )
    else:
        fxc = None

    def apply_xc(dm1):
        return _nr_uks_fxc_sf_tda_mc_gpu_pbc(ni, mf, dm1, hermi=hermi, fxc=fxc)

    return _make_gpu_response_vind(mf, xctype, hybrid, omega, alpha, hyb, hermi, apply_xc)


def _nr_uks_fxc_sf_tda_gpu_pbc(ni, mf, dm1, hermi=0, vxc=None):
    return _nr_uks_fxc_sf_tda_gpu_pbc_common(
        ni, mf, dm1, hermi=hermi, kernel=vxc, method="alda0"
    )


def _gen_response_sf_gpu(mf, mo_coeff, mo_occ, hermi=0, max_memory=None):
    ni = mf._numint if _is_ks_mf(mf) else None
    max_memory = _response_max_memory(mf, max_memory)

    if _is_ks_mf(mf):
        xctype, hybrid, omega, alpha, hyb = _xc_response_params(mf, ni)
        if xctype != "HF":
            vxc = (
                _cache_xc_kernel_sf_gpu_pbc(mf, mo_coeff, mo_occ, max_memory)
                if _is_pbc_mf(mf)
                else _cache_xc_kernel_sf_gpu_mol(mf, mo_coeff, mo_occ, max_memory)
            )
        else:
            vxc = None

        if _is_pbc_mf(mf):
            def apply_xc(dm1):
                return _nr_uks_fxc_sf_tda_gpu_pbc(ni, mf, dm1, hermi=hermi, vxc=vxc)
        else:
            def apply_xc(dm1):
                from gpu4pyscf.tdscf._uhf_resp_sf import nr_uks_fxc_sf
                return nr_uks_fxc_sf(ni, mf.mol, mf.grids, mf.xc, None, dm1, 0, hermi, None, None, vxc)
        return _make_gpu_response_vind(mf, xctype, hybrid, omega, alpha, hyb, hermi, apply_xc)
    else:
        cp = require_cupy()

        def vind(dm1):
            return -_get_k(mf, cp.asarray(dm1), hermi=hermi)

        return vind

# code from pyscf-forge to construct multicollinear functional
def gen_response_sf_mc(mf, mo_coeff=None, mo_occ=None, hermi=0,
                       collinear_samples=60, max_memory=None,ctx=None):
    '''Generate a function to compute the product of Spin Flip UKS response function
    and UKS density matrices.
    '''
    if ctx is not None:
        mo_coeff, mo_occ = ctx.mo_coeff, ctx.mo_occ
    elif mo_coeff is None or mo_occ is None:
        _, mo_occ, mo_coeff = mf_info(mf)
    if _is_gpu_mf(mf):
        if _is_pbc_mf(mf):
            return _gen_response_sf_mc_gpu_pbc(
                mf, mo_coeff, mo_occ, hermi=hermi,
                collinear_samples=collinear_samples, max_memory=max_memory,
            )
        return _gen_response_sf_mc_gpu_mol(
            mf, mo_coeff, mo_occ, hermi=hermi,
            collinear_samples=collinear_samples, max_memory=max_memory,
        )
    #assert isinstance(mf, (uhf.UHF))
    #if mo_coeff is None: mo_coeff = mf.mo_coeff
    #if mo_occ is None: mo_occ = mf.mo_occ
    ni = pbc_numint2c.NumInt2C() if _is_pbc_mf(mf) else numint2c.NumInt2C()
    ni.collinear = 'mcol'
    ni.collinear_samples = collinear_samples
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    if mf.nlc or ni.libxc.is_nlc(mf.xc):
        logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                    'deriviative is not available. Its contribution is '
                    'not included in the response function.')
    xctype, hybrid, omega, alpha, hyb = _xc_response_params(mf, ni)

    # mf can be pbc.dft.UKS object with multigrid
    if (not hybrid and
        'MultiGridFFTDF' == getattr(mf, 'with_df', None).__class__.__name__):
        raise NotImplementedError("Spin Flip TDDFT doesn't support pbc calculations.")

    fxc = cache_xc_kernel_sf_mc(ni, mf, _system(mf), mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]
    #print('fxs')
    #print(fxc)
    dm0 = None

    max_memory = _response_max_memory(mf, max_memory)

    def vind(dm1):
        in2 = pbc_numint.NumInt() if _is_pbc_mf(mf) else numint.NumInt()
        v1 = nr_uks_fxc_sf_tda_mc(in2, mf, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                               None, None, fxc, max_memory=max_memory)
        return _add_hybrid_k(mf, v1, dm1, hybrid, hyb, omega, alpha, hermi)
    return vind

class SF_TDA_up(XTDDFT_base): # just for ROKS
    def __init__(self, mf, method, davidson=True, davidson_backend="cpu"):
        davidson_backend = davidson_backend.lower()
        if davidson_backend not in ("cpu", "gpu", "auto"):
            raise ValueError("davidson_backend must be 'cpu', 'gpu', or 'auto'")
        super().__init__(mf, method, davidson=davidson)
        self.isf = 1
        self.davidson_backend = "cpu" if davidson_backend == "auto" else davidson_backend

    def get_Amat_ALDA0(self):
        # Dense Amat is built with PySCF/NumPy.  GPU inputs are converted to CPU
        # for construction, then converted back to the active backend at return.
        mf = _as_cpu_mf(self.mf)
        ctx = _as_cpu_ctx(mf, self.ctx)

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
                    a_b2a = add_hf_a_b2a(
                        a_b2a, mf, ctx.orbo_b, ctx.orbv_a, ctx.nc, ctx.nv, hyb
                    )
                if abs(omega) > 1e-10:
                    a_b2a = add_hf_a_b2a(
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
                a_b2a = add_hf_a_b2a(a_b2a, mf, ctx.orbo_b, ctx.orbv_a, ctx.nc, ctx.nv, hyb=1)

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
    
    def get_Amat_MCOL(self, collinear_samples=30):
        a_b2a = np.zeros((self.nocc_b,self.nvir_a,self.nocc_b,self.nvir_a))
        
        if isinstance(self.mf, dft.KohnShamDFT):
            ni0 = self.mf._numint
            ni = numint2c.NumInt2C()
            ni.collinear = 'mcol'
            ni.collinear_samples = collinear_samples
            ni.libxc.test_deriv_order(self.mf.xc, 2, raise_error=True)
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.mf.xc, _system(self.mf).spin)
            
            if hyb != 0:
                a_b2a = add_hf_a_b2a(a_b2a, self.mf, self.orbo_b, self.orbv_a, self.nc, self.nv, hyb)
            if omega != 0:
                k_fac = alpha - hyb
                if k_fac != 0:
                    with _system(self.mf).with_range_coulomb(omega):
                        a_b2a = add_hf_a_b2a(k_fac)
            # If only HF exchange is requested
            if collinear_samples < 0:
                return a_b2a
            xctype = ni._xc_type(self.mf.xc)
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, self.mf.max_memory * 0.8 - mem_now)
            
            # This function is assumed to be available from your original file
            fxc = cache_xc_kernel_sf_mc(
                ni, self.mf, self.mf.grids, self.mf.xc, self.mo_coeff, self.mo_occ, deriv=2, spin=1)[2]
            p0, p1 = 0, 0
            if xctype == 'LDA':
                ao_deriv = 0
                for ao, mask, weight, coords in _iter_block_data(self.mf, ni0, ao_deriv, max_memory):
                    p0 = p1
                    p1 += weight.shape[0]
                    wfxc = fxc[0, 0][..., p0:p1] * weight
                    rho_o_b = contract('rp,pi->ri', ao, self.orbo_b, optimize=True)
                    rho_v_a = contract('rp,pi->ri', ao, self.orbv_a, optimize=True)
                    rho_ov_b2a = contract('ri,ra->ria', rho_o_b, rho_v_a, optimize=True)
                    w_ov = contract('ria,r->ria', rho_ov_b2a, wfxc * 2.0, optimize=True)
                    iajb = contract('ria,rjb->iajb', rho_ov_b2a, w_ov, optimize=True)
                    a_b2a += iajb
            elif xctype == 'GGA':
                ao_deriv = 1
                for ao, mask, weight, coords in _iter_block_data(self.mf, ni0, ao_deriv, max_memory):
                    p0 = p1
                    p1 += weight.shape[0]
                    wfxc = fxc[..., p0:p1] * weight
                    rho_o_b = contract('xrp,pi->xri', ao, self.orbo_b, optimize=True)
                    rho_v_a = contract('xrp,pi->xri', ao, self.orbv_a, optimize=True)
                    rho_ov_b2a = contract('xri,ra->xria', rho_o_b, rho_v_a[0], optimize=True)
                    rho_ov_b2a[1:4] += contract('ri,xra->xria', rho_o_b[0], rho_v_a[1:4], optimize=True)
                    w_ov = contract('xyr,xria->yria', wfxc * 2.0, rho_ov_b2a, optimize=True)
                    iajb = contract('xria,xrjb->iajb', w_ov, rho_ov_b2a, optimize=True)
                    a_b2a += iajb
            elif xctype == 'MGGA':
                ao_deriv = 1
                for ao, mask, weight, coords in _iter_block_data(self.mf, ni0, ao_deriv, max_memory):
                    p0 = p1
                    p1 += weight.shape[0]
                    wfxc = fxc[..., p0:p1] * weight
                    rho_ob = contract('xrp,pi->xri', ao, self.orbo_b, optimize=True)
                    rho_va = contract('xrp,pi->xri', ao, self.orbv_a, optimize=True)
                    rho_ov_b2a = contract('xri,ra->xria', rho_ob, rho_va[0], optimize=True)
                    rho_ov_b2a[1:4] += contract('ri,xra->xria', rho_ob[0], rho_va[1:4], optimize=True)
                    tau_ov_b2a = contract('xri,xra->ria', rho_ob[1:4], rho_va[1:4], optimize=True) * 0.5
                    rho_ov_b2a = np.vstack([rho_ov_b2a, tau_ov_b2a[np.newaxis]])
                    w_ov = contract('xyr,xria->yria', wfxc * 2.0, rho_ov_b2a, optimize=True)
                    iajb = contract('xria,xrjb->iajb', w_ov, rho_ov_b2a, optimize=True)
                    a_b2a += iajb
            elif xctype == 'HF':
                pass
            elif xctype == 'NLC':
                raise NotImplementedError('NLC functional is not supported here.')
            else:
                raise NotImplementedError(f'Unsupported xctype: {xctype}')
        else:
            # Pure HF case
            a_b2a = add_hf_a_b2a(1.0)
        focka_mo, fockb_mo = (
                _asnumpy(x) for x in _get_mo_fock(self.mf, self.mo_coeff, self.mo_occ)
            )
        amat = _pair_hessian_block(
                fockb_mo[:self.nc, :self.nc],
                focka_mo[self.nocc_a:, self.nocc_a:],
                a_b2a,
            )
        self.A = xp.asarray(amat)
        return self.A
    
    def get_Amat(self):
        if self.method == 1:  # multicollinear
            self.get_Amat_MCOL()
        else:
            self.get_Amat_ALDA0()
        return self.A
    
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
