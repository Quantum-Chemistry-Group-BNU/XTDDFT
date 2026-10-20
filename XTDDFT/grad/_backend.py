from types import ModuleType, SimpleNamespace

import numpy as np
from pyscf.grad.rhf import grad_nuc

from ...utils.backend import require_cupy
from ..base import _is_pbc_mf


def is_gpu_mf(mf) -> bool:
    return any(cls.__module__.startswith("gpu4pyscf") for cls in type(mf).__mro__)


def array_module(mf) -> ModuleType:
    return require_cupy() if is_gpu_mf(mf) else np


def asarray(mf, value):
    return array_module(mf).asarray(value)


def validate_gradient_backend(mf) -> None:
    if _is_pbc_mf(mf):
        raise NotImplementedError("Analytic gradients support molecular calculations only")
    if getattr(mf, "with_df", None) is not None:
        from . import _reference_family

        family = _reference_family(mf)
        if family not in ("uks", "roks"):
            raise NotImplementedError(
                "Density fitting analytic gradients support UKS and ROKS references"
            )
        if getattr(mf, "only_dfj", False):
            raise NotImplementedError(
                "DF gradients require RI-JK (only_dfj=False), not RI-J"
            )
    if is_gpu_mf(mf):
        if mf._numint._xc_type(mf.xc).upper() == "MGGA":
            raise NotImplementedError("GPU analytic gradients do not support MGGA")


def df_jk_energies_per_atom(mf, dm_list, j_factor=None, k_factor=None,
                           omega=None, hermi=0, sum_results=False, verbose=None):
    """GPU4PySCF's density-pair derivative, including auxiliary-basis response.

    Use the same interface as gpu4pyscf.df.grad.tduks_sf (1.8.1). Preserve
    nonsymmetric transition densities: in particular, [X, X.T] is not [X, X].
    The backend includes both three-center and metric derivatives.
    """
    from gpu4pyscf.df.grad.tdrhf import Gradients

    grad = SimpleNamespace(base=SimpleNamespace(_scf=mf), auxbasis_response=True)
    if isinstance(dm_list, (list, tuple)) and len(dm_list) > 1:
        n = len(dm_list)
        if any(f is not None and len(f) != n for f in (j_factor, k_factor)):
            raise ValueError("J/K factor counts must match the density-pair count")
        # A mixed J/K batch builds exchange intermediates even for pure-J
        # pairs. Isolate pairs to select the native J-only path and bound the
        # number of simultaneously resident naux * rank1 * rank2 tensors.
        # Keep list semantics: ndarray inputs have different symmetry factors.
        results = []
        total = None
        for i, pair in enumerate(dm_list):
            jf = None if j_factor is None else j_factor[i:i+1]
            kf = None if k_factor is None else k_factor[i:i+1]
            if jf is not None and jf[0] == 0 and kf is not None and kf[0] != 0:
                jf = None
            result = Gradients.jk_energies_per_atom(
                grad, [pair], jf, kf,
                omega=0.0 if omega is None else omega,
                hermi=hermi if isinstance(hermi, int) else hermi[i:i+1],
                sum_results=sum_results, verbose=verbose,
            )
            if sum_results:
                if total is None:
                    total = result
                else:
                    total += result
            else:
                results.append(result)
        return total if sum_results else np.concatenate(results, axis=0)
    return Gradients.jk_energies_per_atom(
        grad, dm_list, j_factor, k_factor,
        omega=0.0 if omega is None else omega, hermi=hermi,
        sum_results=sum_results, verbose=verbose,
    )


def ucphf_module(mf):
    if is_gpu_mf(mf):
        from gpu4pyscf.scf import ucphf
    else:
        from pyscf.scf import ucphf
    return ucphf


def nuclear_gradient(mf, atmlst=None):
    return asarray(mf, grad_nuc(mf.mol, atmlst))


def cpu_df_get_jk(mf_grad, dm, hyb, alpha, omega, hermi=0, with_j=True):
    """DF AO derivatives with hybrid-scaled exchange and its auxiliary response.

    NumPy arithmetic does not scale PySCF's .aux metadata automatically.
    Scale it explicitly before combining full- and long-range exchange.
    """
    from pyscf import lib

    mf_grad.auxbasis_response = True
    with_k = hyb != 0 or alpha != 0
    vj, vk = mf_grad.get_jk(dm=dm, hermi=hermi, with_j=with_j, with_k=with_k)
    if with_k:
        kaux = vk.aux * hyb
        vk = np.asarray(vk) * hyb
        if omega != 0:
            vk_lr = mf_grad.get_jk(dm=dm, hermi=hermi, with_j=False, omega=omega)[1]
            vk += np.asarray(vk_lr) * (alpha - hyb)
            kaux += vk_lr.aux * (alpha - hyb)
        vk = lib.tag_array(vk, aux=kaux)
    return vj, vk


def cpu_df_td_aux(vj, vk, response_scale):
    """Auxiliary-center derivatives for the existing spin-resolved density ordering.

    Per spin: [P, D] for SF; [P, D, X, X.T] for SC.
    D is the response density stored by the caller (normalization differs).
    response_scale converts the stored response density to the relaxed one.
    PySCF .aux[i,j] differentiates one auxiliary side; cross terms need both.
    """
    nblock = len(vj) // 2
    ground = np.array([0, nblock])
    response = ground + 1
    ja = vj.aux
    aux = ja[np.ix_(ground, ground)].sum(axis=(0, 1))
    aux += response_scale * (ja[np.ix_(ground, response)].sum(axis=(0, 1))
                             + ja[np.ix_(response, ground)].sum(axis=(0, 1)))
    if vk is not None:
        ka = vk.aux
        aux -= ka[ground, ground].sum(axis=0)
        aux -= response_scale * (ka[ground, response] + ka[response, ground]).sum(axis=0)
    if nblock == 4:
        x, xt = ground + 2, ground + 3
        aux += 4 * ja[np.ix_(x, x)].sum(axis=(0, 1))
        if vk is not None:
            aux -= (ka[x, xt] + ka[xt, x]).sum(axis=0)
    return aux
