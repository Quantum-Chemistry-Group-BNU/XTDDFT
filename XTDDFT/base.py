from types import SimpleNamespace
import numpy as np
from pyscf import ao2mo, lib, scf
from pyscf.dft import numint, xc_deriv
from pyscf.pbc import scf as pbc_scf
from pyscf.pbc.dft import numint as pbc_numint

from ..utils.backend import backend, contract, require_cupy, xp, _asarray, _asnumpy, set_backend
from ..utils.unit import ha2eV

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

def _make_rohf_reference_mf(mf):
    if _is_pbc_mf(mf):
        hf = pbc_scf.ROHF(mf.cell)
        hf.kpt = _get_gamma_kpt(mf)
    else:
        hf = scf.ROHF(mf.mol)
        if getattr(mf, "with_x2c", None) is not None and hasattr(hf, "x2c"):
            hf = hf.x2c()

    for name in ("with_df", "exxdiv", "max_memory", "verbose", "stdout"):
        if hasattr(mf, name):
            try:
                setattr(hf, name, getattr(mf, name))
            except AttributeError:
                pass
    return hf

def _ao2mo_full_gamma(mf, mo_coeff):
    mo_coeff = _asnumpy(mo_coeff)
    nmo = mo_coeff.shape[1]
    if _is_pbc_mf(mf):
        _ensure_gamma_df(mf)
        kpt = _get_gamma_kpt(mf)
        try:
            eri = mf.with_df.ao2mo([mo_coeff] * 4, kpt, compact=False)
        except TypeError:
            eri = mf.with_df.ao2mo([mo_coeff] * 4, [kpt] * 4, compact=False)
    else:
        eri = ao2mo.general(mf.mol, [mo_coeff] * 4, compact=False)
    return _asnumpy(eri).reshape(nmo, nmo, nmo, nmo)

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

def _get_k(mf, dm, hermi=0, omega=None):
    kwargs = {"hermi": hermi}
    if omega is not None and abs(omega) > 1e-14:
        kwargs["omega"] = omega
    if _is_pbc_mf(mf):
        _ensure_gamma_df(mf)
        dm = _asarray(dm)
        try:
            return mf.get_k(mf.cell, dm, kpt=_get_gamma_kpt(mf), **kwargs)
        except (TypeError, AssertionError) as err:
            try:
                return mf.get_k(mf.cell, dm, **kwargs)
            except (TypeError, AssertionError):
                if dm.ndim == 3:
                    return xp.stack([
                        _get_k(mf, dm_i, hermi=hermi, omega=omega)
                        for dm_i in dm
                    ])
                raise err
    return mf.get_k(mf.mol, dm, **kwargs)

def _get_j(mf, dm, hermi=0):
    dm = _asarray(dm)
    if _is_pbc_mf(mf):
        _ensure_gamma_df(mf)
        if dm.ndim == 4:
            return xp.stack([
                _get_j(mf, dm[:, i], hermi=hermi)
                for i in range(dm.shape[1])
            ], axis=1)
        with_df = getattr(mf, "with_df", None)
        gamma_df = bool(getattr(with_df, "is_gamma_point", False))
        if gamma_df:
            out = with_df.get_jk(
                dm, hermi=hermi, kpts=None, with_j=True, with_k=False,
                exxdiv=getattr(mf, "exxdiv", None),
            )
            return out[0] if isinstance(out, tuple) else out
        try:
            return mf.get_j(mf.cell, dm, hermi=hermi, kpt=_get_gamma_kpt(mf))
        except TypeError:
            return mf.get_j(mf.cell, dm, hermi=hermi)
    return mf.get_j(mf.mol, dm, hermi=hermi)

def _get_jk(mf, dm, hermi=0, batch=False):
    dm = _asarray(dm)
    if _is_pbc_mf(mf):
        _ensure_gamma_df(mf)
        if dm.ndim == 4:
            vjs, vks = [], []
            for i in range(dm.shape[1]):
                vj, vk = _get_jk(mf, dm[:, i], hermi=hermi, batch=False)
                vjs.append(vj)
                vks.append(vk)
            return xp.stack(vjs, axis=1), xp.stack(vks, axis=1)
        with_df = getattr(mf, "with_df", None)
        gamma_df = bool(getattr(with_df, "is_gamma_point", False))
        if batch and dm.ndim == 3 and not gamma_df:
            vjs, vks = [], []
            for dm_i in dm:
                vj, vk = _get_jk(mf, dm_i, hermi=hermi, batch=False)
                vjs.append(vj)
                vks.append(vk)
            return xp.stack(vjs), xp.stack(vks)
        if dm.ndim == 3 and dm.shape[0] != 2 and not batch:
            return _get_jk(mf, dm, hermi=hermi, batch=True)
        if gamma_df:
            return with_df.get_jk(
                dm, hermi=hermi, kpts=None, with_j=True, with_k=True,
                exxdiv=getattr(mf, "exxdiv", None),
            )
        try:
            return mf.get_jk(mf.cell, dm, hermi=hermi, kpt=_get_gamma_kpt(mf))
        except TypeError:
            return mf.get_jk(mf.cell, dm, hermi=hermi)
    return mf.get_jk(mf.mol, dm, hermi=hermi)

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

def _get_ovlp(mf):
    if _is_pbc_mf(mf):
        try:
            return _drop_gamma_axis(mf.get_ovlp(mf.cell, kpt=_get_gamma_kpt(mf)))
        except TypeError:
            try:
                return _drop_gamma_axis(mf.get_ovlp(mf.cell))
            except TypeError:
                return _drop_gamma_axis(mf.get_ovlp())
    try:
        return mf.get_ovlp(mf.mol)
    except TypeError:
        return mf.get_ovlp()

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

def _get_hf_mo_fock(mf, mo_coeff, mo_occ):
    mo_coeff = _asarray(mo_coeff)
    mo_occ = _asarray(mo_occ)
    dm = xp.asarray([
        (mo_coeff[s] * mo_occ[s]) @ mo_coeff[s].conj().T
        for s in range(2)
    ])
    vj, vk = _get_jk(mf, dm, hermi=1)
    vj, vk = _asarray(vj), _asarray(vk)
    coul = vj if vj.ndim == 2 else vj[0] + vj[1]
    if vk.ndim == 2:
        vk = xp.stack([vk, vk])
    vhf = coul - vk
    h1e = _asarray(_get_hcore(mf))
    return (
        mo_coeff[0].conj().T @ (h1e + vhf[0]) @ mo_coeff[0],
        mo_coeff[1].conj().T @ (h1e + vhf[1]) @ mo_coeff[1],
    )

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

def _cpu_davidson(vind, hdiag, x0, nroots, positive_eig_threshold=None):
    def aop(xs):
        return _asnumpy(vind(xp.asarray(xs)))

    pick = None
    if positive_eig_threshold is not None:
        def pick(w, v, nroots, envs):
            del envs
            idx = np.where(w > positive_eig_threshold)[0]
            return w[idx], v[:, idx], idx

    return lib.davidson1(
        aop, _asnumpy(x0), _asnumpy(hdiag),
        tol=1e-7, lindep=1e-14,
        nroots=nroots, max_cycle=3000, pick=pick,
        verbose=5
    )

def _gpu_davidson(vind, hdiag, x0, nroots, positive_eig_threshold=None):
    cp = require_cupy()
    from gpu4pyscf.tdscf._lr_eig import eigh as lr_eigh

    hdiag = cp.asarray(hdiag)
    x0 = cp.asarray(x0)

    def precond(dx, e, *args):
        e = cp.asarray(e)
        denom = hdiag - e[:, None] if e.ndim > 0 and dx.ndim > 1 else hdiag - e
        threshold = 1.0e-8
        small = abs(denom) < threshold
        denom = cp.where(small, cp.where(denom >= 0, threshold, -threshold), denom)
        return dx / denom

    def pick(w, v, nroots, envs):
        del envs
        if positive_eig_threshold is None:
            idx = cp.argsort(w)[:nroots]
        else:
            idx = cp.where(w > positive_eig_threshold)[0]
        return w[idx], v[:, idx], idx

    return lr_eigh(
        vind, x0, precond,
        tol_residual=1e-7, lindep=1e-14,
        nroots=nroots, pick=pick, max_cycle=3000, verbose=5
    )[:3]

def _run_davidson(mf, davidson_backend, vind, hdiag, x0, nroots,
                  positive_eig_threshold=None):
    if davidson_backend == "gpu":
        if not _is_gpu_mf(mf):
            raise RuntimeError("davidson_backend='gpu' requires the gpu backend/gpu4pyscf path.")
        return _gpu_davidson(
            vind, hdiag, x0, nroots,
            positive_eig_threshold=positive_eig_threshold,
        )
    return _cpu_davidson(
        vind, hdiag, x0, nroots,
        positive_eig_threshold=positive_eig_threshold,
    )

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
