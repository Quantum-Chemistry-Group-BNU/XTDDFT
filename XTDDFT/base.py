from types import SimpleNamespace
import functools
import numpy as np
from pyscf import lib
from pyscf.dft import numint, xc_deriv
from pyscf.pbc.dft import numint as pbc_numint

from utils.backend import backend, contract, require_cupy, xp, _asarray, _asnumpy, set_backend
from utils.unit import ha2eV

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger(__name__)

def _is_k_method(mf):
    return mf.__class__.__name__.upper().startswith("K")

def _reject_k_method(mf):
    if _is_k_method(mf):
        raise NotImplementedError(
            f"{mf.__class__.__name__} is not supported. "
            "Use molecular methods or Gamma-point PBC methods such as RKS, UKS, or ROKS."
        )

def mf_info(mf):
    """Normalize UKS/ROKS orbital data into a two-spin representation."""
    _reject_k_method(mf)
    mo_coeff0 = _asarray(mf.mo_coeff)
    mo_occ0 = _asarray(mf.mo_occ)
    mo_energy0 = _asarray(mf.mo_energy)
    if mo_coeff0.ndim == 3:
        return mo_energy0, mo_occ0, mo_coeff0

    mo_energy = xp.stack([mo_energy0, mo_energy0])
    mo_coeff = xp.stack([mo_coeff0, mo_coeff0])
    mo_occ = xp.zeros((2, mo_coeff0.shape[1]))
    mo_occ[0, xp.where(mo_occ0 >= 1)[0]] = 1
    mo_occ[1, xp.where(mo_occ0 >= 2)[0]] = 1
    return mo_energy, mo_occ, mo_coeff

def _build_spin_orbital_spaces(mo_coeff, mo_occ):
    occidx_a = xp.where(mo_occ[0] == 1)[0]
    viridx_a = xp.where(mo_occ[0] == 0)[0]
    occidx_b = xp.where(mo_occ[1] == 1)[0]
    viridx_b = xp.where(mo_occ[1] == 0)[0]

    orbo_a = mo_coeff[0][:, occidx_a]
    orbv_a = mo_coeff[0][:, viridx_a]
    orbo_b = mo_coeff[1][:, occidx_b]
    orbv_b = mo_coeff[1][:, viridx_b]

    nc = int(len(occidx_b))
    nv = int(len(viridx_a))
    no = int(len(occidx_a) - len(occidx_b))
    nmo_a = nc + nv + no
    return SimpleNamespace(
        occidx_a=occidx_a, viridx_a=viridx_a,
        occidx_b=occidx_b, viridx_b=viridx_b,
        orbo_a=orbo_a, orbv_a=orbv_a,
        orbo_b=orbo_b, orbv_b=orbv_b,
        nc=nc, nv=nv, no=no, nmo_a=nmo_a,
        nocc_a=int(len(occidx_a)), nvir_a=int(len(viridx_a)),
        nocc_b=int(len(occidx_b)), nvir_b=int(len(viridx_b)),
        nao=int(mo_coeff[0].shape[0]),
    )

def _build_sf_context(mf, mo_energy=None, mo_occ=None, mo_coeff=None):
    if mo_energy is None or mo_occ is None or mo_coeff is None:
        mo_energy, mo_occ, mo_coeff = mf_info(mf)
    spaces = _build_spin_orbital_spaces(mo_coeff, mo_occ)
    cell = getattr(mf, "cell", None)
    mol = None if cell is not None else getattr(mf, "mol", None)
    ctx = {"mf": mf, "cell": cell, "mol": mol, "mo_energy": mo_energy,
           "mo_occ": mo_occ, "mo_coeff": mo_coeff}
    ctx.update(vars(spaces))
    return SimpleNamespace(**ctx)

def _as_cpu_sf_context(ctx, mf=None):
    """Copy an existing spin-flip context to NumPy without recomputing mf_info."""
    out = {}
    for key, value in vars(ctx).items():
        if key == "mf":
            out[key] = mf if mf is not None else value
        elif key == "cell" and mf is not None:
            out[key] = getattr(mf, "cell", None)
        elif key == "mol" and mf is not None:
            out[key] = None if getattr(mf, "cell", None) is not None else getattr(mf, "mol", None)
        elif hasattr(value, "get"):
            out[key] = value.get()
        elif hasattr(value, "shape"):
            out[key] = np.asarray(value)
        else:
            out[key] = value
    return SimpleNamespace(**out)

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

def _make_spin_dm(mo_coeff, mo_occ):
    return xp.asarray([
        (mo_coeff[s] * mo_occ[s]) @ mo_coeff[s].conj().T
        for s in range(2)
    ])

def _get_gamma_kpt(mf):
    """Return the Gamma k-point and reject general k-point calculations."""
    _reject_k_method(mf)
    if hasattr(mf, "kpts"):
        kpts = np.asarray(mf.kpts).reshape(-1, 3)
        if len(kpts) != 1 or np.linalg.norm(kpts[0]) > 1e-9:
            raise NotImplementedError("XTDDFT currently supports molecular and Gamma-point PBC calculations only.")
        return kpts[0]
    kpt = np.asarray(getattr(mf, "kpt", np.zeros(3))).reshape(3)
    if np.linalg.norm(kpt) > 1e-9:
        raise NotImplementedError("XTDDFT currently supports molecular and Gamma-point PBC calculations only.")
    return kpt

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

def _get_veff_gamma(mf, dm):
    _ensure_gamma_df(mf)
    try:
        return mf.get_veff(mf.cell, dm, kpt=_get_gamma_kpt(mf))
    except TypeError:
        try:
            return mf.get_veff(mf.cell, dm, kpts=np.asarray([_get_gamma_kpt(mf)]))
        except TypeError:
            return mf.get_veff(mf.cell, dm)

def _get_veff_mol(mf, dm):
    mol = getattr(mf, "mol", None)
    try:
        return mf.get_veff(mol, dm)
    except TypeError:
        return mf.get_veff(dm=dm)

def _drop_gamma_axis(mat):
    mat = _asarray(mat)
    if mat.ndim == 3 and mat.shape[0] == 1:
        return mat[0]
    if mat.ndim == 4 and mat.shape[1] == 1:
        return mat[:, 0]
    return mat

def _as_spin_potential(vhf):
    vhf = _drop_gamma_axis(vhf)
    if vhf.ndim == 2:
        vhf = xp.stack([vhf, vhf])
    return vhf

def _get_hcore_gamma(mf):
    _ensure_gamma_df(mf)
    try:
        return mf.get_hcore(mf.cell, kpt=_get_gamma_kpt(mf))
    except TypeError:
        try:
            return mf.get_hcore(mf.cell, kpts=np.asarray([_get_gamma_kpt(mf)]))
        except TypeError:
            return mf.get_hcore()

def _get_hcore_mol(mf):
    mol = getattr(mf, "mol", None)
    try:
        return mf.get_hcore(mol)
    except TypeError:
        return mf.get_hcore()

def _is_pbc_mf(mf):
    return getattr(mf, "cell", None) is not None

def _is_gpu_mf(mf):
    return backend.is_gpu or mf.__class__.__module__.startswith("gpu4pyscf")

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

def _is_unrestricted_mf(mf):
    cls_name = mf.__class__.__name__.upper()
    if "UKS" in cls_name or "UHF" in cls_name or "ROKS" in cls_name or "ROHF" in cls_name: # 
        return True
    mo_coeff = getattr(mf, "mo_coeff", None)
    if isinstance(mo_coeff, (tuple, list)):
        return len(mo_coeff) == 2
    if mo_coeff is None:
        return False
    return _asarray(mo_coeff).ndim >= 3

def _make_fock_dm(mf, mo_coeff, mo_occ):
    dm_spin = _make_spin_dm(mo_coeff, mo_occ)
    if _is_unrestricted_mf(mf):
        return dm_spin
    dm = dm_spin[0] + dm_spin[1]
    return dm

def _make_reference_dm(mf, mo_occ):
    dm0 = mf.make_rdm1()
    if np.asarray(mf.mo_coeff).ndim == 2:
        dm0.mo_coeff = (mf.mo_coeff, mf.mo_coeff)
        dm0.mo_occ = mo_occ
    return dm0

def _response_max_memory(mf, max_memory):
    if max_memory is not None:
        return max_memory
    mem_now = lib.current_memory()[0]
    return max(2000, mf.max_memory * .8 - mem_now)

def _get_hcore(mf):
    h1e = _get_hcore_gamma(mf) if _is_pbc_mf(mf) else _get_hcore_mol(mf)
    return _drop_gamma_axis(h1e)

def _get_veff(mf, dm):
    return _get_veff_gamma(mf, dm) if _is_pbc_mf(mf) else _get_veff_mol(mf, dm)

def _get_mo_fock(mf, mo_coeff, mo_occ=None, force_cpu=False):
    _reject_k_method(mf)
    if force_cpu:
        mo_coeff = np.asarray(_asnumpy(mo_coeff))
        if mo_occ is None:
            dm = mf.make_rdm1()
        else:
            mo_occ = np.asarray(_asnumpy(mo_occ))
            dm = np.asarray([
                (mo_coeff[s] * mo_occ[s]) @ mo_coeff[s].conj().T
                for s in range(2)
            ])
        vhf = np.asarray(_asnumpy(_drop_gamma_axis(_get_veff(mf, dm))))
        if vhf.ndim == 2:
            vhf = np.stack([vhf, vhf])
        h1e = np.asarray(_asnumpy(_get_hcore(mf)))
        focka_mo = mo_coeff[0].conj().T @ (h1e + vhf[0]) @ mo_coeff[0]
        fockb_mo = mo_coeff[1].conj().T @ (h1e + vhf[1]) @ mo_coeff[1]
        return focka_mo, fockb_mo

    mo_coeff = _asarray(mo_coeff)
    if mo_occ is None:
        dm = mf.make_rdm1()
    else:
        dm = _make_fock_dm(mf, mo_coeff, _asarray(mo_occ))
    vhf = _as_spin_potential(_get_veff(mf, dm))
    h1e = _asarray(_get_hcore(mf))
    focka_mo = mo_coeff[0].conj().T @ (h1e + vhf[0]) @ mo_coeff[0]
    fockb_mo = mo_coeff[1].conj().T @ (h1e + vhf[1]) @ mo_coeff[1]
    return focka_mo, fockb_mo

def _xc_ao_deriv(xctype, with_lapl=False):
    if xctype == "GGA":
        return 1
    if xctype == "MGGA":
        return 2 if with_lapl else 1
    return 0

def _iter_ao_blocks(mf, ni, ao_deriv, max_memory=None, mol=None, nao=None, kpt=None, force_cpu=False):
    if not force_cpu and _is_gpu_mf(mf):
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


def _iter_block_data(mf, ni, ao_deriv, max_memory, force_cpu=False):
    for block in _iter_ao_blocks(mf, ni, ao_deriv, max_memory, force_cpu=force_cpu):
        yield block.ao, block.mask, block.weight, block.coords

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

def _spinflip_gaps(ctx, isf):
    if isf == 1:
        return (ctx.mo_energy[0][ctx.viridx_a, None]
                - ctx.mo_energy[1][ctx.occidx_b]).T.ravel()
    if isf == -1:
        return (ctx.mo_energy[1][ctx.viridx_b, None]
                - ctx.mo_energy[0][ctx.occidx_a]).T.ravel()
    raise ValueError(f"Unsupported isf={isf!r}; expected -1 or 1.")

def _build_initial_guess_from_gaps(gaps, nstates):
    gaps = xp.asarray(gaps)
    nov = int(gaps.size)
    nroots = min(nstates, nov)
    if nroots < 1:
        raise ValueError("No spin-flip excitation space is available.")
    e_threshold = xp.sort(gaps)[nroots - 1] + 1e-5
    idx = xp.where(gaps <= e_threshold)[0]
    x0 = xp.zeros((int(idx.size), nov))
    x0[xp.arange(int(idx.size)), idx] = 1.0
    return x0

def _make_spinflip_problem(ctx, fock_mo, isf):
    focka_mo, fockb_mo = fock_mo
    if isf == 1:
        return SimpleNamespace(
            hdiag=_spinflip_gaps(ctx, isf),
            ndim=(ctx.nocc_b, ctx.nvir_a),
            orbo=ctx.orbo_b,
            orbv=ctx.orbv_a,
            occ_block=fockb_mo[:ctx.nocc_b, :ctx.nocc_b],
            vir_block=focka_mo[ctx.nocc_a:, ctx.nocc_a:],
        )
    if isf == -1:
        return SimpleNamespace(
            hdiag=_spinflip_gaps(ctx, isf),
            ndim=(ctx.nocc_a, ctx.nvir_b),
            orbo=ctx.orbo_a,
            orbv=ctx.orbv_b,
            occ_block=focka_mo[:ctx.nocc_a, :ctx.nocc_a],
            vir_block=fockb_mo[ctx.nocc_b:, ctx.nocc_b:],
        )
    raise ValueError(f"Unsupported isf={isf!r}; expected -1 or 1.")

def _make_spinflip_vind(problem, vresp):
    def vind(zs0):
        zs = xp.asarray(zs0).reshape(-1, *problem.ndim)
        # 转成AO下的transition density matrix， 这里是对系数做变换
        dmov = contract(
            'xov,qv,po->xpq',
            zs, problem.orbv.conj(), problem.orbo,
        )
        v1ao = xp.asarray(vresp(dmov))
        vs = contract(
            'xpq,po,qv->xov',
            v1ao, problem.orbo.conj(), problem.orbv,
        )  # 转化为分子轨道，这里是对轨道做变换
        vs += contract('ab,xib->xia', problem.vir_block, zs)
        vs -= contract('ij,xja->xia', problem.occ_block, zs)
        return vs.reshape(zs.shape[0], -1)
    return vind

def _cpu_davidson(vind, hdiag, x0, nroots):
    def aop(xs):
        return _asnumpy(vind(xp.asarray(xs)))

    return lib.davidson1(
        aop, _asnumpy(x0), _asnumpy(hdiag),
        tol=1e-7, lindep=1e-14,
        nroots=nroots, max_cycle=3000,
    )

def _gpu_davidson(vind, hdiag, x0, nroots):
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

    return lr_eigh(
        vind, x0, precond,
        tol_residual=1e-7, lindep=1e-14,
        nroots=nroots, pick=pick, max_cycle=3000,
    )[:3]

def _run_davidson(mf, davidson_backend, vind, hdiag, x0, nroots):
    if davidson_backend == "gpu":
        if not _is_gpu_mf(mf):
            raise RuntimeError("davidson_backend='gpu' requires the gpu backend/gpu4pyscf path.")
        return _gpu_davidson(vind, hdiag, x0, nroots)
    return _cpu_davidson(vind, hdiag, x0, nroots)

class XTDDFT_base:
    def __init__(self, mf, method, davidson=True):
        logger.info('method=0 (default) ALDA0, method=1 multicollinear, method=2 collinear')
        self.mf = backend.cast(mf)
        self.method = method
        self.davidson = davidson
        self.ctx = _build_sf_context(self.mf)
        self.__dict__.update(vars(self.ctx))
        self._fock_mo = None
        self.A = None
        self.e = None
        self.v = None
        self.nstates = None
        self.converged = None
        # 在这里先给出泛函参数
        try: # dft
            self.mfxctype = self.mf.xc
            self.ni = self.mf._numint
            self.ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
            if getattr(mf, "nlc", None) or self.ni.libxc.is_nlc(mf.xc):
                logger.warning(
                        'NLC functional found in DFT object. Its second '
                        'derivative is not available and is not included in '
                        'the response function.'
                )
            self.omega, self.alpha, self.hyb = self.ni.rsh_and_hybrid_coeff(mf.xc, _system(mf).spin)
            logger.info(f'Omega:{self.omega}, alpha:{self.alpha}, hyb:{self.hyb}')
            self.xctype = self.ni._xc_type(mf.xc)
        except: # HF
            self.mfxctype = None
            self.omega = 0
            self.hyb = 1.0
            self.xctype = None

    def _get_fock_mo(self):
        if self._fock_mo is None:
            self._fock_mo = _get_mo_fock(self.mf, self.mo_coeff, self.mo_occ)
        return self._fock_mo

    def get_Amat(self):  # 完整的A矩阵
        raise NotImplementedError
    
    def _init_guess(self, nstates):
        raise NotImplementedError
    
    def davidson_process(self, nstates):
        raise NotImplementedError
