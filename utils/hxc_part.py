import numpy as np
import functools
from pyscf.dft import numint, xc_deriv
from pyscf import lib
from pyscf.dft import numint,numint2c
from pyscf.dft.gen_grid import NBINS
from pyscf.dft.numint import _dot_ao_ao_sparse,_scale_ao_sparse,_tau_dot_sparse
from pyscf.pbc.dft import numint as pbc_numint
from pyscf.pbc.dft import numint2c as pbc_numint2c

from ..XTDDFT.base import (
    _get_gamma_kpt,
    _ensure_gamma_df,
    _as_cpu_mf,
    _get_j,
    _get_jk,
    _get_k,
    _is_gpu_mf,
    _is_pbc_mf,
    _iter_ao_blocks,
    _iter_block_data,
    mf_info,
    _system,
    _is_ks_mf,
    _response_max_memory,
)
from .backend import _asarray, _asnumpy, backend, contract, require_cupy, set_backend, xp

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger(__name__)

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

def AldA0(ni, mf, rho, weight, xctype, omega=None):
    vxc = ni.eval_xc_eff(mf.xc, rho, deriv=1, omega=omega, xctype=xctype)[1]
    vxc_a = vxc[0, 0] * weight
    vxc_b = vxc[1, 0] * weight
    if xctype == "LDA":
        denom = np.asarray(rho[0]) - np.asarray(rho[1])
    else:
        denom = np.asarray(rho[0][0]) - np.asarray(rho[1][0])
    return (vxc_a - vxc_b) / (denom + 1e-9)

def __mcfun_fn_eval_xc(ni, xc_code, xctype, rho, deriv):
    evfk = ni.eval_xc_eff(xc_code, rho, deriv=deriv, xctype=xctype)
    for order in range(1, deriv + 1):
        if evfk[order] is not None:
            evfk[order] = xc_deriv.ud2ts(evfk[order])
    return evfk

def mcfun_eval_xc_adapter_sf(ni, xc_code):
    try:
        import mcfun
    except ImportError:
        raise ImportError(
            "This feature requires mcfun library.\n"
            "Try install mcfun with `pip install mcfun`"
        )

    xctype = ni._xc_type(xc_code)
    fn_eval_xc = functools.partial(__mcfun_fn_eval_xc, ni, xc_code, xctype)
    nproc = lib.num_threads()

    def eval_xc_eff(xc_code, rho, deriv=1, omega=None, xctype=None, verbose=None):
        del xc_code, omega, xctype, verbose
        return mcfun.eval_xc_eff_sf(
            fn_eval_xc, rho, deriv,
            collinear_samples=ni.collinear_samples,
            workers=nproc,
        )
    return eval_xc_eff

def _xc_response_params(mf, ni, xc_code=None):
    xc_code = mf.xc if xc_code is None else xc_code
    xctype = ni._xc_type(xc_code)
    if xctype == "HF":
        return xctype, True, 0.0, 0.0, 1.0
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc_code, _system(mf).spin)
    return xctype, ni.libxc.is_hybrid_xc(xc_code), omega, alpha, hyb

def _use_short_range_k(omega, alpha, hyb):
    return abs(omega) > 1e-10 and abs(alpha) < 1e-12 and abs(hyb) > 1e-12


def _hybrid_k(mf, dm1, hyb, omega, alpha, hermi):
    if _use_short_range_k(omega, alpha, hyb):
        return _get_k(mf, dm1, hermi=hermi, omega=-omega) * hyb
    vk = _get_k(mf, dm1, hermi=hermi) * hyb
    if abs(omega) > 1e-10:
        vk += _get_k(mf, dm1, hermi=hermi, omega=omega) * (alpha - hyb)
    return vk


def _add_hybrid_k(mf, v1, dm1, hybrid, hyb, omega, alpha, hermi):
    if not hybrid:
        return v1
    return v1 - _hybrid_k(mf, dm1, hyb, omega, alpha, hermi)

def _add_spin_conserving_jk(mf, v1, dm1, hybrid, hyb, omega, alpha, hermi, with_j=True):
    dm1 = _asarray(dm1)
    if dm1.ndim == 4 and dm1.shape[0] == 2:
        nset = int(dm1.shape[1])
        if with_j:
            vj = _asarray(_get_j(mf, dm1[0] + dm1[1], hermi=hermi))
            if vj.ndim == 2:
                vj = vj.reshape(1, *vj.shape)
            coul = vj.reshape(nset, *dm1.shape[-2:])
        else:
            coul = 0

        if hybrid:
            flat_dm = dm1.reshape(2 * nset, *dm1.shape[-2:])
            vk = _asarray(_hybrid_k(mf, flat_dm, hyb, omega, alpha, hermi))
            if vk.ndim == 2:
                vk = vk.reshape(1, *vk.shape)
            vk = vk.reshape(2, nset, *dm1.shape[-2:])
        else:
            vk = 0
        return v1 + coul[None] - vk

    if not with_j:
        return _add_hybrid_k(mf, v1, dm1, hybrid, hyb, omega, alpha, hermi)

    if hybrid:
        if _use_short_range_k(omega, alpha, hyb):
            vj = _get_j(mf, dm1, hermi=hermi)
            vk = _hybrid_k(mf, dm1, hyb, omega, alpha, hermi)
        else:
            vj, vk = _get_jk(mf, dm1, hermi=hermi)
            vk = _asarray(vk) * hyb
            if abs(omega) > 1e-10:
                vk += _get_k(mf, dm1, hermi=hermi, omega=omega) * (alpha - hyb)
    else:
        vj = _get_j(mf, dm1, hermi=hermi)
        vk = 0

    vj = _asarray(vj)
    coul = vj if vj.ndim == 2 else vj[0] + vj[1]
    return v1 + coul - vk

def _make_gpu_response_vind(mf, xctype, hybrid, omega, alpha, hyb, hermi, apply_xc):
    cp = require_cupy()

    def vind(dm1):
        dm1 = cp.asarray(dm1)
        v1 = cp.zeros_like(dm1) if xctype == "HF" else apply_xc(dm1)
        return _add_hybrid_k(mf, v1, dm1, hybrid, hyb, omega, alpha, hermi)

    return vind

def _make_rho_evaluator(ni, mf, dms, hermi, grids):
    if _is_pbc_mf(mf):
        return ni._gen_rho_evaluator(_system(mf), dms, hermi, False)[:2]
    return ni._gen_rho_evaluator(_system(mf), dms, hermi, False, grids)[:2]

def _cpu_xc_weighted_density(xctype, rho1sf, kernel_block, weight, method):
    if method == "alda0":
        if xctype == "LDA":
            return rho1sf * kernel_block
        rhosf = np.zeros_like(rho1sf)
        rhosf[0] += rho1sf[0]
        return rhosf * kernel_block
    if xctype == "LDA":  # 这个是为MCOL准备的
        return rho1sf * kernel_block[0, 0] * 2.0 * weight
    return contract("bg,abg->ag", rho1sf, kernel_block * 2.0) * weight

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

def _make_response_vxc(ni, mf, dm1, hermi, vxc, max_memory):
    return nr_uks_fxc_sf_tda(
        ni, mf, mf.grids, mf.xc, None, dm1, 0, hermi,
        vxc=vxc, max_memory=max_memory
    )

def _cache_xc_kernel_tda_gpu_pbc(mf, mo_coeff, mo_occ, max_memory=2000):
    del max_memory
    cp = require_cupy()
    _ensure_gamma_df(mf)
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    if xctype == "HF":
        return None
    if xctype not in ("LDA", "GGA", "MGGA"):
        raise NotImplementedError(f"GPU Gamma PBC TDA response is not implemented for {xctype}.")

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
        rhoa.append(ni.eval_rho(mf.cell, block.ao_ks, dm[0][None], xctype=xctype, hermi=1))
        rhob.append(ni.eval_rho(mf.cell, block.ao_ks, dm[1][None], xctype=xctype, hermi=1))

    rho = cp.stack([cp.hstack(rhoa), cp.hstack(rhob)], axis=0)
    return ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]


def _gpu_pbc_tda_weighted_density(xctype, rho1a, rho1b, fxc_block, weight):
    if xctype == "LDA":
        rho1a = rho1a.reshape(1, -1)
        rho1b = rho1b.reshape(1, -1)
        wv = rho1a * fxc_block[0, 0] + rho1b * fxc_block[1, 0]
    else:
        wv = contract("xg,xbyg->byg", rho1a, fxc_block[0])
        wv += contract("xg,xbyg->byg", rho1b, fxc_block[1])
    return wv * weight


def _nr_uks_fxc_tda_gpu_pbc(ni, mf, dm1, hermi=0, fxc=None):
    cp = require_cupy()
    from gpu4pyscf.pbc.dft.numint import _scale_ao, _tau_dot

    _ensure_gamma_df(mf)
    xctype = ni._xc_type(mf.xc)
    if xctype not in ("LDA", "GGA", "MGGA"):
        raise NotImplementedError(f"GPU Gamma PBC TDA response is not implemented for {xctype}.")
    ao_deriv = _xc_ao_deriv(xctype)

    dm1 = cp.asarray(dm1)
    dma, dmb = dm1
    single = dma.ndim == 2
    if single:
        dma = dma.reshape(1, dma.shape[-2], dma.shape[-1])
        dmb = dmb.reshape(1, dmb.shape[-2], dmb.shape[-1])

    nset = int(dma.shape[0])
    nao = int(dma.shape[-1])
    dtype = dm1.dtype
    vmat = cp.zeros((2, nset, nao, nao), dtype=dtype)
    vtau = cp.zeros_like(vmat) if xctype == "MGGA" else None

    p0 = p1 = 0
    for block in _iter_ao_blocks(mf, ni, ao_deriv):
        weight = cp.asarray(block.weight)
        p0, p1 = p1, p1 + int(weight.size)
        fxc_block = cp.asarray(fxc[..., p0:p1])

        for i in range(nset):
            rho1a = ni.eval_rho(mf.cell, block.ao_ks, dma[i][None], xctype=xctype, hermi=hermi)
            rho1b = ni.eval_rho(mf.cell, block.ao_ks, dmb[i][None], xctype=xctype, hermi=hermi)
            wv = _gpu_pbc_tda_weighted_density(xctype, rho1a, rho1b, fxc_block, weight)

            if xctype == "LDA":
                aow = _scale_ao(block.ao, wv[0, 0])
                vmat[0, i] += block.ao.conj().T.dot(aow)
                aow = _scale_ao(block.ao, wv[1, 0])
                vmat[1, i] += block.ao.conj().T.dot(aow)
            elif xctype == "GGA":
                wv[:, 0] *= .5
                aow = _scale_ao(block.ao[:4], wv[0, :4])
                vmat[0, i] += block.ao[0].conj().T.dot(aow)
                aow = _scale_ao(block.ao[:4], wv[1, :4])
                vmat[1, i] += block.ao[0].conj().T.dot(aow)
            elif xctype == "MGGA":
                wv[:, [0, 4]] *= .5
                aow = _scale_ao(block.ao[:4], wv[0, :4])
                vmat[0, i] += block.ao[0].conj().T.dot(aow)
                vtau[0, i] += _tau_dot(block.ao, block.ao, wv[0, 4])
                aow = _scale_ao(block.ao[:4], wv[1, :4])
                vmat[1, i] += block.ao[0].conj().T.dot(aow)
                vtau[1, i] += _tau_dot(block.ao, block.ao, wv[1, 4])

    if xctype in ("GGA", "MGGA"):
        vmat = vmat + vmat.conj().swapaxes(-2, -1)
    if vtau is not None:
        vmat += vtau
    return vmat[:, 0] if single else vmat


def _gen_response_tda_gpu_pbc(mf, mo_coeff, mo_occ, hermi=0,
                              max_memory=None, with_j=True):
    cp = require_cupy()
    _ensure_gamma_df(mf)
    ni = mf._numint if _is_ks_mf(mf) else None
    max_memory = _response_max_memory(mf, max_memory)

    if _is_ks_mf(mf):
        xctype, hybrid, omega, alpha, hyb = _xc_response_params(mf, ni)
        fxc = None if xctype == "HF" else _cache_xc_kernel_tda_gpu_pbc(
            mf, mo_coeff, mo_occ, max_memory
        )

        def vind(dm1):
            dm1 = cp.asarray(dm1)
            v1 = cp.zeros_like(dm1) if xctype == "HF" else _nr_uks_fxc_tda_gpu_pbc(
                ni, mf, dm1, hermi=hermi, fxc=fxc
            )
            return _add_spin_conserving_jk(
                mf, v1, dm1, hybrid, hyb, omega, alpha, hermi,
                with_j=with_j,
            )
        return vind

    def vind(dm1):
        dm1 = cp.asarray(dm1)
        v1 = cp.zeros_like(dm1)
        return _add_spin_conserving_jk(
            mf, v1, dm1, True, 1.0, 0.0, 0.0, hermi,
            with_j=with_j,
        )
    return vind

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

def gen_response_tda(mf, mo_coeff=None, mo_occ=None, hermi=0,
                     max_memory=None, with_j=True, ctx=None):
    """Spin-conserving UKS/ROKS TDA response for molecules and Gamma PBC."""
    if ctx is not None:
        mo_coeff, mo_occ = ctx.mo_coeff, ctx.mo_occ
    elif mo_coeff is None or mo_occ is None:
        _, mo_occ, mo_coeff = mf_info(mf)

    if _is_gpu_mf(mf) and _is_pbc_mf(mf):
        return _gen_response_tda_gpu_pbc(
            mf, mo_coeff, mo_occ, hermi=hermi,
            max_memory=max_memory, with_j=with_j,
        )

    if _is_ks_mf(mf):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        xctype, hybrid, omega, alpha, hyb = _xc_response_params(mf, ni)
        max_memory = _response_max_memory(mf, max_memory)

        if xctype == "HF":
            def vind(dm1):
                dm1 = _asarray(dm1)
                v1 = xp.zeros_like(dm1)
                return _add_spin_conserving_jk(
                    mf, v1, dm1, hybrid, hyb, omega, alpha, hermi,
                    with_j=with_j,
                )
            return vind

        system = _system(mf)
        if _is_pbc_mf(mf):
            kpt = _get_gamma_kpt(mf)
            kpts = np.asarray([kpt])
            rho0, vxc, fxc = ni.cache_xc_kernel(
                system, mf.grids, mf.xc, mo_coeff, mo_occ, 1,
                kpt=kpt, max_memory=max_memory,
            )
        else:
            rho0, vxc, fxc = ni.cache_xc_kernel(
                system, mf.grids, mf.xc, mo_coeff, mo_occ, 1,
            )
        dm0 = None

        def vind(dm1):
            dm1 = _asarray(dm1)
            if _is_pbc_mf(mf):
                v1 = ni.nr_uks_fxc(
                    system, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                    rho0, vxc, fxc, kpts=kpts,
                    max_memory=max_memory,
                )
            else:
                v1 = ni.nr_uks_fxc(
                    system, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                    rho0, vxc, fxc, max_memory=max_memory,
                )
            return _add_spin_conserving_jk(
                mf, v1, dm1, hybrid, hyb, omega, alpha, hermi,
                with_j=with_j,
            )
        return vind

    if with_j:
        def vind(dm1):
            dm1 = _asarray(dm1)
            v1 = xp.zeros_like(dm1)
            return _add_spin_conserving_jk(
                mf, v1, dm1, True, 1.0, 0.0, 0.0, hermi,
                with_j=True,
            )
        return vind

    def vind(dm1):
        dm1 = _asarray(dm1)
        return -_get_k(mf, dm1, hermi=hermi)
    return vind

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


def _gpu_pbc_xc_weighted_density(xctype, rho1sf, kernel_block, weight, method):
    if method == "alda0":
        if xctype == "LDA":
            return rho1sf * kernel_block
        rhosf = rho1sf * 0
        rhosf[0] = rho1sf[0]
        return rhosf * kernel_block
    if xctype == "LDA":
        return rho1sf * kernel_block[0, 0] * (2.0 * weight)
    return contract("bg,abg->ag", rho1sf, kernel_block * 2.0) * weight


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
        logger.warning(mf, 'NLC functional found in DFT object.  Its second '
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
