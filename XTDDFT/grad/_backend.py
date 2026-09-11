from types import ModuleType

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
        raise NotImplementedError(
            "Analytic gradients currently require direct integrals; density fitting changes the formula"
        )
    if is_gpu_mf(mf):
        if mf._numint._xc_type(mf.xc).upper() == "MGGA":
            raise NotImplementedError("GPU analytic gradients do not support MGGA")
        omega = mf._numint.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)[0]
        if omega != 0:
            raise NotImplementedError(
                "GPU analytic gradients do not support range-separated hybrids"
            )


def ucphf_module(mf):
    if is_gpu_mf(mf):
        from gpu4pyscf.scf import ucphf
    else:
        from pyscf.scf import ucphf
    return ucphf


def nuclear_gradient(mf, atmlst=None):
    return asarray(mf, grad_nuc(mf.mol, atmlst))
