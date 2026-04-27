"""GPU spin-flip TDA for Gamma-point PBC gpu4pyscf references.

The SCF object is expected to be a gpu4pyscf PBC object, for example::

    from gpu4pyscf.pbc import dft, df as gpudf
    mf = dft.UKS(cell).to_gpu()
    mf.xc = "pbe"
    mf.with_df = gpudf.GDF(cell)

This module keeps the public API of :mod:`SF_TDA_pbc` but the Davidson
matrix-vector path is evaluated with CuPy arrays and gpu4pyscf PBC JK/numint
kernels.  The dense explicit matrix path is intentionally delegated to the CPU
implementation because gpu4pyscf PBC ``GDF.ao2mo`` is not implemented.
"""

import functools
from types import SimpleNamespace

import cupy as cp
import numpy as np
import scipy
from pyscf import lib
from pyscf.dft import xc_deriv
from pyscf.pbc.dft import numint2c as pbc_numint2c
from pyscf.pbc.lib.kpts_helper import is_zero

from gpu4pyscf.pbc.dft.numint import _scale_ao, _tau_dot

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging

    class _LoggerFallback:
        def info(self, msg, *args):
            logging.getLogger(__name__).info(msg.format(*args) if args else msg)

        def warning(self, msg, *args):
            logging.getLogger(__name__).warning(msg.format(*args) if args else msg)

    logger = _LoggerFallback()

ha2eV = 27.2113834


def _asnumpy(x):
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else np.asarray(x)


def _asarray(x):
    return x if isinstance(x, cp.ndarray) else cp.asarray(x)


def _get_gamma_kpt(mf):
    """Return the Gamma k-point and reject general k-point calculations."""
    if hasattr(mf, "kpts"):
        kpts = np.asarray(mf.kpts).reshape(-1, 3)
        if len(kpts) != 1 or np.linalg.norm(kpts[0]) > 1e-9:
            raise NotImplementedError("SF_TDA_pbc_GPU currently supports Gamma-point PBC only.")
        return kpts[0]
    kpt = np.asarray(getattr(mf, "kpt", np.zeros(3))).reshape(3)
    if np.linalg.norm(kpt) > 1e-9:
        raise NotImplementedError("SF_TDA_pbc_GPU currently supports Gamma-point PBC only.")
    return kpt


def _get_hcore_gamma(mf):
    _ensure_gamma_df(mf)
    try:
        return mf.get_hcore(mf.cell, kpt=_get_gamma_kpt(mf))
    except TypeError:
        return mf.get_hcore()


def _get_veff_gamma(mf, dm):
    _ensure_gamma_df(mf)
    try:
        return mf.get_veff(mf.cell, dm, kpt=_get_gamma_kpt(mf))
    except TypeError:
        return mf.get_veff(mf.cell, dm)


def _ensure_gamma_df(mf):
    """Select gpu4pyscf's real Gamma-point GDF path before CDERI is built."""
    _get_gamma_kpt(mf)
    with_df = getattr(mf, "with_df", None)
    if with_df is None or not hasattr(with_df, "is_gamma_point"):
        return
    if getattr(with_df, "_cderi", None) is None:
        with_df.is_gamma_point = True
    elif not with_df.is_gamma_point:
        logger.warning(
            "mf.with_df was already built with is_gamma_point=False; "
            "resetting it with is_gamma_point=True for batched Gamma-point exchange."
        )
        with_df.reset(mf.cell)
        with_df.is_gamma_point = True


def _get_k_gamma(mf, dm, hermi=0, omega=None):
    dm = _asarray(dm)
    _ensure_gamma_df(mf)
    if dm.ndim == 3 and not getattr(getattr(mf, "with_df", None), "is_gamma_point", False):
        return cp.stack([_get_k_gamma(mf, x, hermi=hermi, omega=omega) for x in dm])
    kwargs = {"hermi": hermi, "kpt": _get_gamma_kpt(mf)}
    if omega is not None and abs(omega) > 1e-14:
        kwargs["omega"] = omega
    return mf.get_k(mf.cell, dm, **kwargs)


def _as_spin_potential(vhf):
    vhf = _asarray(vhf)
    if vhf.ndim == 2:
        vhf = cp.asarray([vhf, vhf])
    return vhf


def SF_TDA(mf, isf=-1, davidson=True, method=0):
    """Return the spin-up or spin-down GPU SF-TDA solver selected by ``isf``."""
    print("method=0 (default) ALDA0, method=1 multicollinear, method=2 collinear/no-XC")
    if isf == -1:
        return SF_TDA_down(mf, method, davidson)
    if isf == 1:
        return SF_TDA_up(mf, method, davidson)
    raise ValueError(f"Unsupported isf={isf!r}; expected -1 or 1.")


def mf_info(mf):
    """Normalize UKS/ROKS orbital data into a two-spin CuPy representation."""
    mo_coeff0 = _asarray(mf.mo_coeff)
    mo_occ0 = _asarray(mf.mo_occ)
    mo_energy0 = _asarray(mf.mo_energy)
    if mo_coeff0.ndim == 3:
        return mo_energy0, mo_occ0, mo_coeff0

    mo_energy = cp.stack([mo_energy0, mo_energy0])
    mo_coeff = cp.stack([mo_coeff0, mo_coeff0])
    mo_occ = cp.zeros((2, mo_coeff0.shape[1]))
    mo_occ[0, cp.where(mo_occ0 >= 1)[0]] = 1
    mo_occ[1, cp.where(mo_occ0 >= 2)[0]] = 1
    return mo_energy, mo_occ, mo_coeff


def _build_spin_orbital_spaces(mo_coeff, mo_occ):
    occidx_a = cp.where(mo_occ[0] == 1)[0]
    viridx_a = cp.where(mo_occ[0] == 0)[0]
    occidx_b = cp.where(mo_occ[1] == 1)[0]
    viridx_b = cp.where(mo_occ[1] == 0)[0]

    orbo_a = mo_coeff[0][:, occidx_a]
    orbv_a = mo_coeff[0][:, viridx_a]
    orbo_b = mo_coeff[1][:, occidx_b]
    orbv_b = mo_coeff[1][:, viridx_b]

    nc = int(len(occidx_b))
    nv = int(len(viridx_a))
    no = int(len(occidx_a) - len(occidx_b))
    return SimpleNamespace(
        occidx_a=occidx_a, viridx_a=viridx_a,
        occidx_b=occidx_b, viridx_b=viridx_b,
        orbo_a=orbo_a, orbv_a=orbv_a,
        orbo_b=orbo_b, orbv_b=orbv_b,
        nc=nc, nv=nv, no=no,
        nocc_a=int(len(occidx_a)), nvir_a=int(len(viridx_a)),
        nocc_b=int(len(occidx_b)), nvir_b=int(len(viridx_b)),
        nao=int(mo_coeff[0].shape[0]),
    )


def _build_sf_context(mf, mo_energy=None, mo_occ=None, mo_coeff=None):
    if mo_energy is None or mo_occ is None or mo_coeff is None:
        mo_energy, mo_occ, mo_coeff = mf_info(mf)
    spaces = _build_spin_orbital_spaces(mo_coeff, mo_occ)
    ctx = {"mf": mf, "cell": mf.cell, "mo_energy": mo_energy,
           "mo_occ": mo_occ, "mo_coeff": mo_coeff}
    ctx.update(vars(spaces))
    return SimpleNamespace(**ctx)


def _make_spin_dm(mo_coeff, mo_occ):
    return cp.asarray([
        (mo_coeff[s] * mo_occ[s]) @ mo_coeff[s].conj().T
        for s in range(2)
    ])


def _get_mo_fock(mf, mo_coeff, mo_occ=None):
    dm = mf.make_rdm1()
    if mo_occ is not None:
        dm = _make_spin_dm(_asarray(mo_coeff), _asarray(mo_occ))
    vhf = _as_spin_potential(_get_veff_gamma(mf, dm))
    h1e = _asarray(_get_hcore_gamma(mf))
    focka_mo = mo_coeff[0].conj().T @ (h1e + vhf[0]) @ mo_coeff[0]
    fockb_mo = mo_coeff[1].conj().T @ (h1e + vhf[1]) @ mo_coeff[1]
    return focka_mo, fockb_mo


def _spinflip_gaps(ctx, isf):
    if isf == 1:
        return (ctx.mo_energy[0][ctx.viridx_a, None]
                - ctx.mo_energy[1][ctx.occidx_b]).T.ravel()
    if isf == -1:
        return (ctx.mo_energy[1][ctx.viridx_b, None]
                - ctx.mo_energy[0][ctx.occidx_a]).T.ravel()
    raise ValueError(f"Unsupported isf={isf!r}; expected -1 or 1.")


def _build_initial_guess_from_gaps(gaps, nstates):
    nov = int(gaps.size)
    nroots = min(nstates, nov)
    e_threshold = cp.sort(gaps)[nroots - 1] + 1e-5
    idx = cp.where(gaps <= e_threshold)[0]
    x0 = cp.zeros((int(idx.size), nov))
    x0[cp.arange(int(idx.size)), idx] = 1.0
    return x0


def _lda_reduced_density(rho):
    rho_lda = cp.zeros_like(rho)
    rho_lda[0] = rho[0]
    return rho_lda


def cache_xc_kernel_sf(mf, mo_coeff, mo_occ, spin=1, max_memory=2000, isf=-1):
    """Cache scalar ALDA0 spin-flip kernel values on GPU grids."""
    del isf, max_memory
    assert spin == 1
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    if xctype == "HF":
        return cp.asarray([])
    if xctype == "LDA":
        ao_deriv = 0
    elif xctype in ("GGA", "MGGA"):
        ao_deriv = 1
    else:
        raise NotImplementedError(f"GPU ALDA0 kernel for {xctype}")

    dm = _make_spin_dm(_asarray(mo_coeff), _asarray(mo_occ))
    fxc_abs = []
    for ao_ks, weight, coords in ni.block_loop(mf.cell, mf.grids, ao_deriv, _get_gamma_kpt(mf)):
        del coords
        rho_a = ni.eval_rho(mf.cell, ao_ks, dm[0][None], xctype=xctype, hermi=0)
        rho_b = ni.eval_rho(mf.cell, ao_ks, dm[1][None], xctype=xctype, hermi=0)
        if xctype == "LDA":
            rho = cp.stack([rho_a, rho_b])
            denom = rho_a - rho_b
        else:
            rho = cp.stack([_lda_reduced_density(rho_a), _lda_reduced_density(rho_b)])
            denom = rho_a[0] - rho_b[0]
        vxc = ni.eval_xc_eff(mf.xc, rho, deriv=1, xctype=xctype)[1]
        fxc_ab = (vxc[0, 0] - vxc[1, 0]) * weight / (denom + 1e-9)
        fxc_abs.append(fxc_ab)
    return cp.hstack(fxc_abs) if fxc_abs else cp.asarray([])


def nr_uks_fxc_sf_tda(ni, cell, grids, xc_code, dm0, dms, relativity=0, hermi=0,
                      vxc=None, extype=0, kpt=None, max_memory=2000, verbose=None):
    """Apply the scalar ALDA0 spin-flip XC kernel to AO perturbation densities."""
    del dm0, relativity, extype, max_memory, verbose
    xctype = ni._xc_type(xc_code)
    if xctype == "LDA":
        ao_deriv = 0
    elif xctype in ("GGA", "MGGA"):
        ao_deriv = 1
    else:
        raise NotImplementedError(f"GPU ALDA0 response for {xctype}")

    dms = _asarray(dms)
    single = dms.ndim == 2
    dms2 = dms.reshape(1, dms.shape[-2], dms.shape[-1]) if single else dms
    nset = int(dms2.shape[0])
    nao = int(dms2.shape[-1])
    dtype = dms2.dtype
    kpt = _get_gamma_kpt(SimpleNamespace(kpt=np.zeros(3))) if kpt is None else np.asarray(kpt)
    if not is_zero(kpt):
        raise NotImplementedError("Only Gamma-point response is implemented.")

    vmat = cp.zeros((nset, nao, nao), dtype=dtype)
    p0 = p1 = 0
    for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpt):
        del coords
        ao = ao_ks[0]
        p0, p1 = p1, p1 + int(weight.size)
        fxc_block = _asarray(vxc[p0:p1])
        for i in range(nset):
            rho1sf = ni.eval_rho(cell, ao_ks, dms2[i][None], xctype=xctype, hermi=hermi)
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
                vmat[i] += _tau_dot(ao, ao, wv[4])

    if xctype != "LDA":
        vmat = vmat + vmat.conj().swapaxes(-2, -1)
    if single:
        return vmat[0]
    return vmat


def __mcfun_fn_eval_xc(ni, xc_code, xctype, rho, deriv):
    """Evaluate XC derivatives and convert them to tensor-spin form."""
    evfk = ni.eval_xc_eff(xc_code, rho, deriv=deriv, xctype=xctype)
    evfk = list(evfk)
    for order in range(1, deriv + 1):
        if evfk[order] is not None:
            evfk[order] = xc_deriv.ud2ts(evfk[order])
    return evfk


def mcfun_eval_xc_adapter_sf(ni, xc_code):
    """Wrap mcfun.eval_xc_eff_sf for the multicollinear spin-flip kernel."""
    try:
        import mcfun
    except ImportError as exc:
        raise ImportError(
            "method=1 requires the mcfun library. Install it with `pip install mcfun`."
        ) from exc

    xctype = ni._xc_type(xc_code)
    fn_eval_xc = functools.partial(__mcfun_fn_eval_xc, ni, xc_code, xctype)
    nproc = lib.num_threads()

    def eval_xc_eff(xc_code, rho, deriv=1, omega=None, xctype=None, verbose=None):
        del xc_code, omega, xctype, verbose
        return mcfun.eval_xc_eff_sf(
            fn_eval_xc, rho, deriv,
            collinear_samples=ni.collinear_samples, workers=nproc
        )

    return eval_xc_eff


def cache_xc_kernel_sf_mc(mf, mo_coeff, mo_occ, collinear_samples=60,
                          spin=1, kpt=None, max_memory=2000):
    """Cache the multicollinear spin-flip XC kernel from GPU grid densities."""
    assert spin == 1
    cell = mf.cell
    xc_code = mf.xc
    ni = mf._numint
    ni_mc = pbc_numint2c.NumInt2C()
    ni_mc.collinear = "mcol"
    ni_mc.collinear_samples = collinear_samples
    ni_mc.libxc.test_deriv_order(xc_code, 2, raise_error=True)

    xctype = ni_mc._xc_type(xc_code)
    if xctype == "GGA":
        ao_deriv = 1
    elif xctype == "MGGA":
        ao_deriv = 1
    else:
        ao_deriv = 0

    if kpt is None:
        kpt = _get_gamma_kpt(mf)

    dm = _make_spin_dm(_asarray(mo_coeff), _asarray(mo_occ))
    rhoa = []
    rhob = []
    for ao_ks, weight, coords in ni.block_loop(cell, mf.grids, ao_deriv, kpt):
        del weight, coords
        rhoa.append(_asnumpy(ni.eval_rho(cell, ao_ks, dm[0][None], xctype=xctype, hermi=0)))
        rhob.append(_asnumpy(ni.eval_rho(cell, ao_ks, dm[1][None], xctype=xctype, hermi=0)))

    rho_ab = np.asarray((np.hstack(rhoa), np.hstack(rhob)))
    rho_tmz = np.zeros_like(rho_ab) + 1e-11
    rho_tmz[0] += rho_ab[0] + rho_ab[1]
    rho_tmz[1] += rho_ab[0] - rho_ab[1]

    eval_xc = mcfun_eval_xc_adapter_sf(ni_mc, xc_code)
    return eval_xc(xc_code, rho_tmz, deriv=2, xctype=xctype)


def nr_uks_fxc_sf_tda_mc(ni, cell, grids, xc_code, dms, hermi=0,
                         fxc=None, kpt=None, max_memory=2000):
    """Apply the multicollinear spin-flip XC kernel on GPU."""
    del max_memory
    xctype = ni._xc_type(xc_code)
    if xctype == "LDA":
        ao_deriv = 0
    elif xctype in ("GGA", "MGGA"):
        ao_deriv = 1
    else:
        raise NotImplementedError(f"GPU multicollinear response for {xctype}")

    dms = _asarray(dms)
    single = dms.ndim == 2
    dms2 = dms.reshape(1, dms.shape[-2], dms.shape[-1]) if single else dms
    nset = int(dms2.shape[0])
    nao = int(dms2.shape[-1])
    dtype = dms2.dtype
    kpt = _get_gamma_kpt(SimpleNamespace(kpt=np.zeros(3))) if kpt is None else np.asarray(kpt)
    if not is_zero(kpt):
        raise NotImplementedError("Only Gamma-point response is implemented.")

    vmat = cp.zeros((nset, nao, nao), dtype=dtype)
    vtau = cp.zeros_like(vmat) if xctype == "MGGA" else None
    p0 = p1 = 0
    for ao_ks, weight, coords in ni.block_loop(cell, grids, ao_deriv, kpt):
        del coords
        ao = ao_ks[0]
        weight = _asarray(weight)
        p0, p1 = p1, p1 + int(weight.size)
        fxc_block = _asarray(fxc[..., p0:p1])
        for i in range(nset):
            rho1sf = ni.eval_rho(cell, ao_ks, dms2[i][None], xctype=xctype, hermi=hermi)
            if xctype == "LDA":
                wv = rho1sf * fxc_block[0, 0] * (2.0 * weight)
                aow = _scale_ao(ao, wv)
                vmat[i] += ao.conj().T.dot(aow)
            elif xctype == "GGA":
                wv = cp.einsum("bg,abg->ag", rho1sf, fxc_block * 2.0) * weight
                wv[0] *= .5
                aow = _scale_ao(ao[:4], wv[:4])
                vmat[i] += ao[0].conj().T.dot(aow)
            elif xctype == "MGGA":
                wv = cp.einsum("bg,abg->ag", rho1sf, fxc_block * 2.0) * weight
                wv[[0, 4]] *= .5
                aow = _scale_ao(ao[:4], wv[:4])
                vmat[i] += ao[0].conj().T.dot(aow)
                vtau[i] += _tau_dot(ao, ao, wv[4])

    if xctype in ("GGA", "MGGA"):
        vmat = vmat + vmat.conj().swapaxes(-2, -1)
    if vtau is not None:
        vmat += vtau
    if single:
        return vmat[0]
    return vmat


def gen_response_sf(mf, hermi=0, max_memory=None, method=0, collinear_samples=60):
    """Generate GPU response function for Gamma-point spin-flip TDA."""
    mo_energy, mo_occ, mo_coeff = mf_info(mf)
    del mo_energy
    cell = mf.cell
    kpt = _get_gamma_kpt(mf)
    ni = mf._numint

    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory * .8 - mem_now)

    xc_code = getattr(mf, "xc", "HF")
    xctype = ni._xc_type(xc_code)
    hybrid = xctype == "HF" or ni.libxc.is_hybrid_xc(xc_code)
    if xctype != "HF":
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc_code, cell.spin)
        if method == 0:
            vxc = cache_xc_kernel_sf(mf, mo_coeff, mo_occ, 1, max_memory, isf=-1)
            fxc = None
        elif method == 1:
            fxc = cache_xc_kernel_sf_mc(
                mf, mo_coeff, mo_occ, collinear_samples=collinear_samples,
                spin=1, kpt=kpt, max_memory=max_memory
            )[2]
            vxc = None
        else:
            vxc = fxc = None
    else:
        omega, alpha, hyb, vxc, fxc = 0.0, 0.0, 1.0, None, None

    def vind(dm1):
        dm1 = _asarray(dm1)
        if method == 0 and xctype != "HF":
            v1 = nr_uks_fxc_sf_tda(
                ni, cell, mf.grids, xc_code, None, dm1, 0, hermi,
                vxc=vxc, kpt=kpt, max_memory=max_memory
            )
        elif method == 1 and xctype != "HF":
            v1 = nr_uks_fxc_sf_tda_mc(
                ni, cell, mf.grids, xc_code, dm1, hermi=hermi,
                fxc=fxc, kpt=kpt, max_memory=max_memory
            )
        else:
            v1 = cp.zeros_like(dm1)

        if hybrid:
            if abs(omega) < 1e-10:
                vk = _get_k_gamma(mf, dm1, hermi=hermi) * hyb
            else:
                vk = _get_k_gamma(mf, dm1, hermi=hermi) * hyb
                vk += _get_k_gamma(mf, dm1, hermi=hermi, omega=omega) * (alpha - hyb)
            v1 -= vk
        return v1

    return vind


def gen_tda_operation_sf(mf, isf, method):
    """Build the GPU matrix-vector product and diagonal preconditioner."""
    ctx = _build_sf_context(mf)
    focka_mo, fockb_mo = _get_mo_fock(mf, ctx.mo_coeff, ctx.mo_occ)

    if isf == -1:
        ndim = (ctx.nocc_a, ctx.nvir_b)
        orbo, orbv = ctx.orbo_a, ctx.orbv_b
        occ_block = focka_mo[:ctx.nocc_a, :ctx.nocc_a]
        vir_block = fockb_mo[ctx.nc:, ctx.nc:]
    elif isf == 1:
        ndim = (ctx.nocc_b, ctx.nvir_a)
        orbo, orbv = ctx.orbo_b, ctx.orbv_a
        occ_block = fockb_mo[:ctx.nocc_b, :ctx.nocc_b]
        vir_block = focka_mo[ctx.nocc_a:, ctx.nocc_a:]
    else:
        raise ValueError(f"Unsupported isf={isf!r}; expected -1 or 1.")

    hdiag = _spinflip_gaps(ctx, isf)
    vresp = gen_response_sf(mf, hermi=0, method=method)

    # def vind(zs0):
    #     zs = _asarray(zs0).reshape(-1, *ndim)
    #     print(zs.shape, orbv.shape, orbo.shape)
    #     dmov = cp.einsum("xov,qv,po->xpq", zs, orbv.conj(), orbo)
    #     v1ao = vresp(dmov)
    #     vs = cp.einsum("xpq,po,qv->xov", v1ao, orbo.conj(), orbv)
    #     vs += cp.einsum("ab,xib->xia", vir_block, zs)
    #     vs -= cp.einsum("ij,xja->xia", occ_block, zs)
    #     return vs.reshape(zs.shape[0], -1)
    def vind(zs0):
        zs = cp.asarray(zs0).reshape(-1, *ndim)

        # X_ov -> AO transition density D_pq
        # 原来：dmov = cp.einsum("xov,qv,po->xpq", zs, orbv.conj(), orbo)
        tmp = cp.matmul(zs, orbv.conj().T)     # (x, o, q)
        dmov = cp.matmul(orbo, tmp)            # (x, p, q)

        v1ao = vresp(dmov)

        # AO response -> ov response
        # 原来：vs = cp.einsum("xpq,po,qv->xov", v1ao, orbo.conj(), orbv)
        tmp = cp.matmul(v1ao, orbv)            # (x, p, v)
        vs = cp.matmul(orbo.conj().T, tmp)     # (x, o, v)

        vs += cp.einsum("ab,xib->xia", vir_block, zs, optimize=True)
        vs -= cp.einsum("ij,xja->xia", occ_block, zs, optimize=True)

        return vs.reshape(zs.shape[0], -1)

    return vind, hdiag


def _reorder_down_vectors(ctx, v, remove=False):
    nstates = v.shape[1]
    nc, nv, no = ctx.nc, ctx.nv, ctx.no
    nvir = no + nv
    passed = nc * nvir
    cv = cp.zeros((nstates, nc, nv))
    co = cp.zeros((nstates, nc, no))
    ov = cp.zeros((nstates, no, nv))
    oo = cp.zeros((nstates, no * no - 1)) if remove else cp.zeros((nstates, no, no))

    for state in range(nstates):
        tmp = v[:, state]
        for i in range(nc):
            co[state, i, :] = tmp[i*nvir:i*nvir + no]
            cv[state, i, :] = tmp[i*nvir + no:i*nvir + no + nv]
        if remove:
            for i in range(no - 1):
                oo[state, i*no:(i + 1)*no] = tmp[passed + i*nvir:passed + i*nvir + no]
                ov[state, i, :] = tmp[passed + i*nvir + no:passed + i*nvir + no + nv]
            oo[state, (no - 1)*no:] = tmp[passed + (no - 1)*nvir:passed + (no - 1)*nvir + no - 1]
            ov[state, no - 1, :] = tmp[passed + (no - 1)*nvir + no - 1:]
        else:
            for i in range(no):
                oo[state, i, :] = tmp[passed + i*nvir:passed + i*nvir + no]
                ov[state, i, :] = tmp[passed + i*nvir + no:passed + i*nvir + no + nv]

    return cp.hstack([
        cv.reshape(nstates, -1),
        co.reshape(nstates, -1),
        ov.reshape(nstates, -1),
        oo.reshape(nstates, -1),
    ]).T


def init_guess(mf, nstates, isf=-1):
    return _build_initial_guess_from_gaps(_spinflip_gaps(_build_sf_context(mf), isf), nstates)


def davidson_process(mf, nstates, method, isf=-1):
    vind, hdiag = gen_tda_operation_sf(mf, isf, method)
    ndim = int(hdiag.size)
    nroots = min(nstates, ndim)
    x0 = init_guess(mf, nroots, isf)
    converged, e, x1 = lib.davidson1(
        lambda xs: _asnumpy(vind(cp.asarray(xs))),
        _asnumpy(x0),
        _asnumpy(hdiag),
        tol=1e-7, lindep=1e-14, nroots=nroots, max_cycle=3000,
    )
    v = cp.asarray(np.asarray(x1).T)
    if isf == -1:
        v = _reorder_down_vectors(_build_sf_context(mf), v)
    logger.info("SF-TDA GPU converged: {}", converged)
    return cp.asarray(e), v


class _SpinFlipTDABase:
    isf = None

    def __init__(self, mf, method, davidson=True):
        self.mf = mf
        self.cell = mf.cell
        self.method = method
        self.davidson = davidson
        self.ctx = _build_sf_context(mf)
        self.__dict__.update(vars(self.ctx))
        self._fock_mo = None
        self.A = None
        self.e = None
        self.v = None
        self.nstates = None
        self.converged = None

    def _get_fock_mo(self):
        if self._fock_mo is None:
            self._fock_mo = _get_mo_fock(self.mf, self.mo_coeff, self.mo_occ)
        return self._fock_mo

    def get_Amat(self):
        """Build dense A through the CPU PBC implementation.

        gpu4pyscf PBC GDF currently does not expose ``ao2mo``.  The Davidson
        path above is the GPU implementation; dense explicit matrices are kept
        as a compatibility path for debugging small systems.
        """
        from .SF_TDA_pbc import SF_TDA as CPU_SF_TDA

        cpu_mf = self.mf.to_cpu() if hasattr(self.mf, "to_cpu") else self.mf
        self.A = cp.asarray(CPU_SF_TDA(cpu_mf, isf=self.isf, davidson=False, method=self.method).get_Amat())
        return self.A

    def _postprocess_davidson_vectors(self, v):
        return v

    def davidson_process(self, nstates):
        vind, hdiag = gen_tda_operation_sf(self.mf, isf=self.isf, method=self.method)
        x0 = _build_initial_guess_from_gaps(_spinflip_gaps(self.ctx, self.isf), nstates)
        converged, e, x1 = lib.davidson1(
            lambda xs: _asnumpy(vind(cp.asarray(xs))),
            _asnumpy(x0),
            _asnumpy(hdiag),
            tol=1e-7, lindep=1e-14, nroots=nstates, max_cycle=3000,
        )
        self.converged = converged
        self.e = cp.asarray(e)
        self.v = self._postprocess_davidson_vectors(cp.asarray(np.asarray(x1).T))
        print("Converged ", converged)
        return self.e, self.v

    def kernel(self, nstates=1):
        self.nstates = nstates
        if self.davidson:
            self.davidson_process(nstates)
        else:
            self.get_Amat()
            e, v = scipy.linalg.eigh(_asnumpy(self.A))
            self.e, self.v = cp.asarray(e), cp.asarray(v)
        return _asnumpy(self.e[:nstates] * ha2eV), self.v[:, :nstates]


class SF_TDA_up(_SpinFlipTDABase):
    isf = 1

    def _postprocess_davidson_vectors(self, v):
        return v

    def analyse(self):
        for nstate in range(self.nstates):
            value = self.v[:, nstate]
            x_cv = value[:self.nc * self.nv].reshape(self.nc, self.nv)
            print(f"Excited state {nstate+1} {float(self.e[nstate].get())*ha2eV:10.5f} eV")
            for occ, vir in zip(*cp.where(abs(x_cv) > 0.1)):
                occ_i, vir_i = int(occ), int(vir)
                amp = float(x_cv[occ_i, vir_i].get())
                print(f"{100*amp**2:3.0f}% CV(ab) {occ_i+1}a -> {vir_i+1+self.nc+self.no}b {amp:10.5f}")


class SF_TDA_down(_SpinFlipTDABase):
    isf = -1

    def _postprocess_davidson_vectors(self, v):
        return _reorder_down_vectors(self.ctx, v)

    def analyse(self):
        nc, nv, no = self.nc, self.nv, self.no
        for nstate in range(self.nstates):
            value = self.v[:, nstate]
            x_cv = value[:nc*nv].reshape(nc, nv)
            x_co = value[nc*nv:nc*nv+nc*no].reshape(nc, no)
            x_ov = value[nc*nv+nc*no:nc*nv+nc*no+no*nv].reshape(no, nv)
            x_oo = value[nc*nv+nc*no+no*nv:].reshape(no, no)
            dp_ab = cp.sum(x_cv*x_cv) - cp.sum(x_oo*x_oo) + cp.sum(cp.diag(x_oo))**2
            print(f"Excited state {nstate+1} {float(self.e[nstate].get())*ha2eV:10.5f} eV D<S^2>={float((-no+1+dp_ab).get()):5.2f}")
            for label, arr, off_o, off_v in (
                ("CV(ab)", x_cv, 1, self.nc + self.no + 1),
                ("CO(ab)", x_co, 1, self.nc + 1),
                ("OV(ab)", x_ov, nc + 1, self.nc + self.no + 1),
                ("OO(ab)", x_oo, nc + 1, self.nc + 1),
            ):
                for o, v in zip(*cp.where(abs(arr) > 0.1)):
                    oi, vi = int(o), int(v)
                    amp = float(arr[oi, vi].get())
                    print(f"{100*amp**2:3.0f}% {label} {oi+off_o}a -> {vi+off_v}b {amp:10.5f}")


if __name__ == "__main__":
    print("Import SF_TDA_pbc_GPU and pass a Gamma-point gpu4pyscf PBC mean-field object.")
