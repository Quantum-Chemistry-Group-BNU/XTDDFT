from types import SimpleNamespace
import numpy as np
from utils.backend import backend, xp, _asarray, _asnumpy
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
    cell = getattr(mf, "cell", None)
    mol = None if cell is not None else getattr(mf, "mol", None)
    ctx = {"mf": mf, "cell": cell, "mol": mol, "mo_energy": mo_energy,
           "mo_occ": mo_occ, "mo_coeff": mo_coeff}
    ctx.update(vars(spaces))
    return SimpleNamespace(**ctx)

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

def _get_hcore(mf):
    h1e = _get_hcore_gamma(mf) if _is_pbc_mf(mf) else _get_hcore_mol(mf)
    return _drop_gamma_axis(h1e)

def _get_veff(mf, dm):
    return _get_veff_gamma(mf, dm) if _is_pbc_mf(mf) else _get_veff_mol(mf, dm)

def _get_mo_fock(mf, mo_coeff, mo_occ=None):
    _reject_k_method(mf)
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

class XTDDFT_base:
    def __init__(self, mf, method, davidson=True):
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
    
    def kernel(self, nstates=1):
        self.nstates = nstates
        if self.davidson:
            self.davidson_process(nstates)
        else:
            
            self.get_Amat()
            e, v = xp.linalg.eigh(xp.asarray(_asnumpy(self.A)))
            self.e, self.v = xp.asarray(e), xp.asarray(v)
        return _asnumpy(self.e[:nstates] * ha2eV), self.v[:, :nstates]
