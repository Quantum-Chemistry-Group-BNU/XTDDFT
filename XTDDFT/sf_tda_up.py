import functools
import numpy as np
from pyscf import ao2mo, scf, lib, dft
from pyscf.dft import numint,numint2c,xc_deriv
from pyscf.dft.gen_grid import NBINS
from pyscf.dft.numint import _dot_ao_ao_sparse,_scale_ao_sparse,_tau_dot_sparse
from opt_einsum import contract

from XTDDFT.base import (
    XTDDFT_base,
    _build_sf_context,
    _ensure_gamma_df,
    _get_gamma_kpt,
    _get_mo_fock,
    _is_pbc_mf,
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


def _is_gpu_mf(mf):
    return backend.is_gpu or mf.__class__.__module__.startswith("gpu4pyscf")


def _is_ks_mf(mf):
    return hasattr(mf, "_numint") and hasattr(mf, "xc")


def _get_k(mf, dm, hermi=0, omega=None):
    kwargs = {"hermi": hermi}
    if omega is not None and abs(omega) > 1e-14:
        kwargs["omega"] = omega
    if _is_pbc_mf(mf):
        _ensure_gamma_df(mf)
        try:
            return mf.get_k(mf.cell, dm, kpt=_get_gamma_kpt(mf), **kwargs)
        except TypeError:
            return mf.get_k(mf.cell, dm, **kwargs)
    return mf.get_k(mf.mol, dm, **kwargs)


def _make_response_vxc(ni, mf, dm1, hermi, vxc, max_memory):
    return nr_uks_fxc_sf_tda(
        ni, mf, mf.grids, mf.xc, None, dm1, 0, hermi,
        vxc=vxc, max_memory=max_memory
    )


def _spinflip_up_gaps(ctx):
    return (ctx.mo_energy[0][ctx.viridx_a, None] - ctx.mo_energy[1][ctx.occidx_b]).T.ravel()


def _build_initial_guess_from_gaps(gaps, nstates):
    gaps = xp.asarray(gaps)
    nov = int(gaps.size)
    nroots = min(nstates, nov)
    if nroots < 1:
        raise ValueError("No spin-flip-up excitation space is available.")
    e_threshold = xp.sort(gaps)[nroots - 1] + 1e-5
    idx = xp.where(gaps <= e_threshold)[0]
    x0 = xp.zeros((int(idx.size), nov))
    x0[xp.arange(int(idx.size)), idx] = 1.0
    return x0


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

def cache_xc_kernel_sf(mf, mo_coeff, mo_occ, spin=1,max_memory=2000,isf=-1): # for ALDA0
    '''Compute the fxc_sf, which can be used in SF-TDDFT/TDA
    '''
    MGGA_DENSITY_LAPL = False
    with_lapl = MGGA_DENSITY_LAPL
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    mo_energy,mo_occ,mo_coeff = mf_info(mf)

    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 2 if MGGA_DENSITY_LAPL else 1
    else: 
        ao_deriv = 0 # LDA

    assert mo_coeff[0].ndim == 2
    assert spin == 1

    nao = mo_coeff[0].shape[0]
    dm0 = mf.make_rdm1()
    if np.array(mf.mo_coeff).ndim==2:
        dm0.mo_coeff = (mf.mo_coeff, mf.mo_coeff)
        dm0.mo_occ = mo_occ
    make_rho = ni._gen_rho_evaluator(_system(mf), dm0, hermi=0, with_lapl=False)[0]

    fxc_abs = []
    #if isf == -1: # 
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
    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 2 if MGGA_DENSITY_LAPL else 1
    else:
        ao_deriv = 0
    with_lapl = MGGA_DENSITY_LAPL

    assert mo_coeff[0].ndim == 2
    assert spin == 1

    nao = mo_coeff[0].shape[0]
    rhoa = []
    rhob = []

    ni = numint.NumInt()
    for ao, mask, weight, coords in _iter_block_data(mf, ni, ao_deriv, max_memory):
        rhoa.append(ni.eval_rho2(mol, ao, mo_coeff[0], mo_occ[0], mask, xctype, with_lapl))
        rhob.append(ni.eval_rho2(mol, ao, mo_coeff[1], mo_occ[1], mask, xctype, with_lapl))
    rho_ab = (np.hstack(rhoa), np.hstack(rhob))
    rho_ab = np.asarray(rho_ab)
    rho_tmz = np.zeros_like(rho_ab)+1e-11
    rho_tmz[0] += rho_ab[0]+rho_ab[1]
    rho_tmz[1] += rho_ab[0]-rho_ab[1]
    eval_xc = mcfun_eval_xc_adapter_sf(self,xc_code)
    fxc_sf = eval_xc(xc_code, rho_tmz, deriv=2, xctype=xctype)
    return fxc_sf

def nr_uks_fxc_sf_tda(ni, mf, grids, xc_code, dm0, dms, relativity=0, hermi=0,vxc=None, extype=0, max_memory=2000, verbose=None):
    if isinstance(dms, np.ndarray):
        dtype = dms.dtype
    else:
        dtype = np.result_type(*dms)
    if hermi != 1 and dtype != np.double:
        raise NotImplementedError('complex density matrix')

    xctype = ni._xc_type(xc_code)

    nao = dms.shape[-1]
    make_rhosf, nset = ni._gen_rho_evaluator(_system(mf), dms, hermi, False, grids)[:2]

    def block_loop(ao_deriv):
        p1 = 0
        for ao, mask, weight, coords in _iter_block_data(mf, ni, ao_deriv, max_memory):
            p0, p1 = p1, p1 + weight.size
            _vxc = vxc[p0:p1]
            for i in range(nset):
                rho1sf = make_rhosf(i, ao, mask, xctype)
                if xctype == 'LDA':
                    wv = rho1sf * _vxc 
                else:
                    rhosf = np.zeros_like(rho1sf)
                    rhosf[0] += rho1sf[0]
                    wv = rhosf * _vxc
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

def nr_uks_fxc_sf_tda_mc(ni, mf, grids, xc_code, dm0, dms, relativity=0, hermi=0,rho0=None,
                      vxc=None, fxc=None, extype=0, max_memory=2000, verbose=None):
    if isinstance(dms, np.ndarray):
        dtype = dms.dtype
    else:
        dtype = np.result_type(*dms)
    if hermi != 1 and dtype != np.double:
        raise NotImplementedError('complex density matrix')

    xctype = ni._xc_type(xc_code)

    nao = dms.shape[-1]
    make_rhosf, nset = ni._gen_rho_evaluator(_system(mf), dms, hermi, False, grids)[:2]

    def block_loop(ao_deriv):
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(_system(mf), grids, nao, ao_deriv, max_memory=max_memory):
            p0, p1 = p1, p1 + weight.size
            _fxc = fxc[...,p0:p1]
            for i in range(nset):
                rho1sf = make_rhosf(i, ao, mask, xctype)
                if xctype == 'LDA':
                    # *2.0 becausue kernel xx,yy parts.
                    wv = rho1sf * _fxc[0,0]*2.0 *weight
                else:
                    # *2.0 becausue kernel xx,yy parts.
                    wv = contract('bg,abg->ag', rho1sf, _fxc * 2.0, optimize=True) * weight
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

def gen_response_sf(mf,hermi=0,max_memory=None,method=0):
    mo_energy,mo_occ,mo_coeff = mf_info(mf)
    del mo_energy
    if _is_gpu_mf(mf):
        return _gen_response_sf_gpu(mf, mo_coeff, mo_occ, hermi=hermi,
                                    max_memory=max_memory, method=method)

    if _is_ks_mf(mf):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, _system(mf).spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)
        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)
        if method == 0:
            vxc = cache_xc_kernel_sf(mf, mo_coeff, mo_occ,1,max_memory,isf=-1) # XC kerkel 
        dm0 = None

        def vind(dm1):
            if method == 0:
                v1 = _make_response_vxc(ni, mf, dm1, hermi, vxc, max_memory)
            else:
                v1 = np.zeros_like(dm1)
            
            if not hybrid:
                # No with_j because = 0 in spin flip part.
                pass
            else:
                vk = _get_k(mf, dm1, hermi=hermi)
                vk *= hyb
                if omega > 1e-10:  # For range separated Coulomb
                    vk += _get_k(mf, dm1, hermi=hermi, omega=omega) * (alpha-hyb)
                v1 -= vk
            return v1
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
    if xctype == "LDA":
        ao_deriv = 0
    elif xctype in ("GGA", "MGGA"):
        ao_deriv = 1
    else:
        raise NotImplementedError(f"GPU ALDA0 response is not implemented for {xctype}.")

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
    for ao_mask, idx, weight, coords in ni.block_loop(sorted_mol, mf.grids, mol.nao_nr(), ao_deriv):
        del weight, coords
        rhoa.append(eval_rho2(sorted_mol, ao_mask, mo_coeff[0, idx, :], mo_occ[0], None, xctype, False))
        rhob.append(eval_rho2(sorted_mol, ao_mask, mo_coeff[1, idx, :], mo_occ[1], None, xctype, False))

    rho_a = cp.hstack(rhoa)
    rho_b = cp.hstack(rhob)
    if xctype == "LDA":
        rho = cp.stack([rho_a, rho_b])
        denom = rho_a - rho_b
    else:
        rha = cp.zeros_like(rho_a)
        rhb = cp.zeros_like(rho_b)
        rha[0] = rho_a[0]
        rhb[0] = rho_b[0]
        rho = cp.stack([rha, rhb])
        denom = rho_a[0] - rho_b[0]
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
    if xctype == "LDA":
        ao_deriv = 0
    elif xctype in ("GGA", "MGGA"):
        ao_deriv = 1
    else:
        raise NotImplementedError(f"GPU Gamma PBC ALDA0 response is not implemented for {xctype}.")

    dm = cp.asarray([
        (mo_coeff[s] * mo_occ[s]) @ mo_coeff[s].conj().T
        for s in range(2)
    ])
    fxc_abs = []
    for ao_ks, weight, coords in ni.block_loop(mf.cell, mf.grids, ao_deriv, _get_gamma_kpt(mf)):
        del coords
        rho_a = ni.eval_rho(mf.cell, ao_ks, dm[0][None], xctype=xctype, hermi=0)
        rho_b = ni.eval_rho(mf.cell, ao_ks, dm[1][None], xctype=xctype, hermi=0)
        if xctype == "LDA":
            rho = cp.stack([rho_a, rho_b])
            denom = rho_a - rho_b
        else:
            rha = cp.zeros_like(rho_a)
            rhb = cp.zeros_like(rho_b)
            rha[0] = rho_a[0]
            rhb[0] = rho_b[0]
            rho = cp.stack([rha, rhb])
            denom = rho_a[0] - rho_b[0]
        vxc = ni.eval_xc_eff(mf.xc, rho, deriv=1, xctype=xctype)[1]
        fxc_abs.append((vxc[0, 0] - vxc[1, 0]) * weight / (denom + 1e-9))
    return cp.hstack(fxc_abs) if fxc_abs else cp.asarray([])


def _nr_uks_fxc_sf_tda_gpu_pbc(ni, mf, dm1, hermi=0, vxc=None):
    cp = require_cupy()
    from gpu4pyscf.pbc.dft.numint import _scale_ao, _tau_dot

    _ensure_gamma_df(mf)
    xctype = ni._xc_type(mf.xc)
    if xctype == "LDA":
        ao_deriv = 0
    elif xctype in ("GGA", "MGGA"):
        ao_deriv = 1
    else:
        raise NotImplementedError(f"GPU Gamma PBC ALDA0 response is not implemented for {xctype}.")

    dm1 = cp.asarray(dm1)
    single = dm1.ndim == 2
    dms = dm1.reshape(1, dm1.shape[-2], dm1.shape[-1]) if single else dm1
    vmat = cp.zeros_like(dms)
    vtau = cp.zeros_like(dms) if xctype == "MGGA" else None
    p0 = p1 = 0
    for ao_ks, weight, coords in ni.block_loop(mf.cell, mf.grids, ao_deriv, _get_gamma_kpt(mf)):
        del coords
        ao = ao_ks[0]
        p0, p1 = p1, p1 + int(weight.size)
        fxc_block = cp.asarray(vxc[p0:p1])
        for i in range(int(dms.shape[0])):
            rho1sf = ni.eval_rho(mf.cell, ao_ks, dms[i][None], xctype=xctype, hermi=hermi)
            if xctype == "LDA":
                wv = rho1sf * fxc_block
                aow = _scale_ao(ao, wv)
                vmat[i] += ao.conj().T.dot(aow)
            elif xctype == "GGA":
                rhosf = cp.zeros_like(rho1sf)
                rhosf[0] = rho1sf[0]
                wv = rhosf * fxc_block
                wv[0] *= .5
                aow = _scale_ao(ao[:4], wv[:4])
                vmat[i] += ao[0].conj().T.dot(aow)
            elif xctype == "MGGA":
                rhosf = cp.zeros_like(rho1sf)
                rhosf[0] = rho1sf[0]
                wv = rhosf * fxc_block
                wv[[0, 4]] *= .5
                aow = _scale_ao(ao[:4], wv[:4])
                vmat[i] += ao[0].conj().T.dot(aow)
                vtau[i] += _tau_dot(ao, ao, wv[4])
    if xctype in ("GGA", "MGGA"):
        vmat = vmat + vmat.conj().swapaxes(-2, -1)
    if vtau is not None:
        vmat += vtau
    return vmat[0] if single else vmat


def _gen_response_sf_gpu(mf, mo_coeff, mo_occ, hermi=0, max_memory=None, method=0):
    cp = require_cupy()
    if method != 0:
        raise NotImplementedError("GPU Davidson currently supports method=0 (ALDA0/HF) only.")

    ni = mf._numint if _is_ks_mf(mf) else None
    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory * .8 - mem_now)

    if _is_ks_mf(mf):
        xctype = ni._xc_type(mf.xc)
        hybrid = xctype == "HF" or ni.libxc.is_hybrid_xc(mf.xc)
        if xctype == "HF":
            omega, alpha, hyb, vxc = 0.0, 0.0, 1.0, None
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, _system(mf).spin)
            vxc = (
                _cache_xc_kernel_sf_gpu_pbc(mf, mo_coeff, mo_occ, max_memory)
                if _is_pbc_mf(mf)
                else _cache_xc_kernel_sf_gpu_mol(mf, mo_coeff, mo_occ, max_memory)
            )

        def vind(dm1):
            dm1 = cp.asarray(dm1)
            if xctype == "HF":
                v1 = cp.zeros_like(dm1)
            elif _is_pbc_mf(mf):
                v1 = _nr_uks_fxc_sf_tda_gpu_pbc(ni, mf, dm1, hermi=hermi, vxc=vxc)
            else:
                from gpu4pyscf.tdscf._uhf_resp_sf import nr_uks_fxc_sf
                v1 = nr_uks_fxc_sf(ni, mf.mol, mf.grids, mf.xc, None, dm1, 0, hermi, None, None, vxc)

            if hybrid:
                vk = _get_k(mf, dm1, hermi=hermi) * hyb
                if abs(omega) > 1e-10:
                    vk += _get_k(mf, dm1, hermi=hermi, omega=omega) * (alpha - hyb)
                v1 -= vk
            return v1
    else:
        def vind(dm1):
            return -_get_k(mf, cp.asarray(dm1), hermi=hermi)

    return vind

# code from pyscf-forge to construct multicollinear functional
def gen_response_sf_mc(mf, mo_coeff=None, mo_occ=None, hermi=0, collinear_samples=60, max_memory=None,method=1):
    '''Generate a function to compute the product of Spin Flip UKS response function
    and UKS density matrices.
    '''
    mo_energy,mo_occ,mo_coeff = mf_info(mf)
    #assert isinstance(mf, (uhf.UHF))
    #if mo_coeff is None: mo_coeff = mf.mo_coeff
    #if mo_occ is None: mo_occ = mf.mo_occ
    ni = numint2c.NumInt2C()
    ni.collinear = 'mcol'
    ni.collinear_samples = collinear_samples
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    if mf.nlc or ni.libxc.is_nlc(mf.xc):
        logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                    'deriviative is not available. Its contribution is '
                    'not included in the response function.')
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, _system(mf).spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)

    # mf can be pbc.dft.UKS object with multigrid
    if (not hybrid and
        'MultiGridFFTDF' == getattr(mf, 'with_df', None).__class__.__name__):
        raise NotImplementedError("Spin Flip TDDFT doesn't support pbc calculations.")

    fxc = cache_xc_kernel_sf_mc(ni, _system(mf), mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]
    #print('fxs')
    #print(fxc)
    dm0 = None

    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

    def vind(dm1):
        in2 = numint.NumInt()
        v1 = nr_uks_fxc_sf_tda_mc(in2, _system(mf), mf.grids, mf.xc, dm0, dm1, 0, hermi,
                               None, None, fxc, max_memory=max_memory)
        if not hybrid:
            # No with_j because = 0 in spin flip part.
            pass
        else:
            vk = mf.get_k(_system(mf), dm1, hermi=hermi)
            vk *= hyb
            if omega > 1e-10:  # For range separated Coulomb
                vk += mf.get_k(_system(mf), dm1, hermi, omega) * (alpha-hyb)
            v1 -= vk
        return v1
    return vind

class SF_TDA_up(XTDDFT_base): # just for ROKS
    isf = 1

    def __init__(self, mf, method, davidson=True, davidson_backend="cpu"):
        davidson_backend = davidson_backend.lower()
        if davidson_backend not in ("cpu", "gpu", "auto"):
            raise ValueError("davidson_backend must be 'cpu', 'gpu', or 'auto'")
        super().__init__(mf, method, davidson=davidson)
        self.davidson_backend = "cpu" if davidson_backend == "auto" else davidson_backend

    def get_Amat_ALDA0(self):
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
        hdiag = _spinflip_up_gaps(self.ctx)
        ndim = (self.nocc_b, self.nvir_a)
        orbo, orbv = self.orbo_b, self.orbv_a
        focka_mo, fockb_mo = self._get_fock_mo()
        occ_block = fockb_mo[:self.nocc_b, :self.nocc_b]
        vir_block = focka_mo[self.nocc_a:, self.nocc_a:]
        if self.method == 1:
            vresp = gen_response_sf_mc(self.mf,hermi=0,collinear_samples=50)
        else:
            vresp = gen_response_sf(self.mf,hermi=0)
            
        def vind(zs0): # vector-matrix product for indexed operations
            zs = xp.asarray(zs0).reshape(-1, *ndim)
            dmov = contract('xov,qv,po->xpq', zs, orbv.conj(), orbo, optimize=True)
            v1ao = xp.asarray(vresp(dmov))
            vs = contract('xpq,po,qv->xov', v1ao, orbo.conj(), orbv, optimize=True)
            vs += contract('ab,xib->xia', vir_block, zs, optimize=True)
            vs -= contract('ij,xja->xia', occ_block, zs, optimize=True)
            return vs.reshape(zs.shape[0], -1)
        return vind, hdiag
    
    def init_guess(self, nstates):
        return _build_initial_guess_from_gaps(_spinflip_up_gaps(self.ctx), nstates)
    
    def davidson_process(self, nstates):
        vind, hdiag = self.gen_tda_operation_sf()
        nroots = min(nstates, int(hdiag.size))
        x0 = self.init_guess(nroots)

        if self.davidson_backend == "gpu":
            return self._gpu_davidson_process(vind, hdiag, x0, nroots)

        def aop(xs):
            return _asnumpy(vind(xp.asarray(xs)))

        converged, e, x1 = lib.davidson1(
            aop, _asnumpy(x0), _asnumpy(hdiag),
            tol=1e-7, lindep=1e-14,
            nroots=nroots, max_cycle=3000
        )
        self.converged = converged
        self.e = xp.asarray(e)
        self.v = xp.asarray(np.asarray(x1).T)
        logger.info('SF_TDA_up Davidson converged: {}', converged)
        return self.e, self.v

    def _gpu_davidson_process(self, vind, hdiag, x0, nroots):
        if not _is_gpu_mf(self.mf):
            raise RuntimeError("davidson_backend='gpu' requires the gpu backend/gpu4pyscf path.")

        cp = require_cupy()
        from gpu4pyscf.tdscf._lr_eig import eigh as lr_eigh

        hdiag = cp.asarray(hdiag)
        x0 = cp.asarray(x0)

        def precond(dx, e, *args):
            denom = hdiag - e
            threshold = 1.0e-8
            small = abs(denom) < threshold
            denom = cp.where(small, cp.where(denom >= 0, threshold, -threshold), denom)
            return dx / denom

        def pick(w, v, nroots, envs):
            del envs
            idx = cp.argsort(w)[:nroots]
            return w[idx], v[:, idx], idx

        result = lr_eigh(
            vind, x0, precond,
            tol_residual=1e-7, lindep=1e-14,
            nroots=nroots, pick=pick, max_cycle=3000
        )
        converged, e, x1 = result[:3]
        self.converged = converged
        self.e = cp.asarray(e)
        self.v = cp.asarray(x1).T
        logger.info('SF_TDA_up GPU Davidson converged: {}', converged)
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