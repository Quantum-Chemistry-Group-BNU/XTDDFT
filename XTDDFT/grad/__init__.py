from numbers import Integral


def _reference_family(mf):
    names = {cls.__name__.upper() for cls in type(mf).__mro__}

    if "ROKS" in names:
        return "roks"
    if "UKS" in names:
        return "uks"

    raise NotImplementedError(
        "Analytic gradients require a molecular ROKS or UKS reference"
    )


def nuc_grad_method(td, state=1):
    from ..base import _is_gpu_mf, _is_pbc_mf
    from ..sf_tda_up import SF_TDA_up
    from ..xsf_tda_down import XSF_TDA_down
    from ..xtda import XTDA

    if isinstance(state, bool) or not isinstance(state, Integral) or state < 1:
        raise ValueError("state must be a one-based positive integer")

    state = int(state)

    if _is_pbc_mf(td.mf):
        raise NotImplementedError(
            "Analytic nuclear gradients currently support molecular calculations only"
        )
    if _is_gpu_mf(td.mf):
        raise NotImplementedError(
            "Analytic nuclear gradients currently support the CPU backend only"
        )

    if getattr(td, "e", None) is None or getattr(td, "v", None) is None:
        raise RuntimeError(
            "Run td.kernel() before calculating gradients"
        )

    if state > td.v.shape[1]:
        raise ValueError(
            f"state={state} requested, but only {td.v.shape[1]} states are available"
        )

    supported = (
        isinstance(td, XTDA) and td.method == 0
        or isinstance(td, SF_TDA_up) and td.method == 1
        or isinstance(td, XSF_TDA_down) and td.method in (1, 2)
    )
    if not supported:
        raise NotImplementedError(
            f"Analytic gradients are not implemented for "
            f"{type(td).__name__} method={getattr(td, 'method', None)!r}"
        )

    reference = _reference_family(td.mf)
    is_uks = reference == "uks"

    if isinstance(td, XTDA):
        if is_uks:
            from .gradient_uks_sc import SC_gradient
        else:
            from .gradient_roks_sc import SC_gradient
        return SC_gradient(td, state=state)

    if isinstance(td, SF_TDA_up):
        if is_uks:
            from .gradient_uks_sfu import SFU_gradient
        else:
            from .gradient_roks_sfu import SFU_gradient
        return SFU_gradient(td, method=td.method, state=state)

    if isinstance(td, XSF_TDA_down):
        if is_uks:
            from .gradient_uks_sfd import SFD_gradient
        else:
            from .gradient_roks_sfd import SFD_gradient
        return SFD_gradient(td, method=td.method, state=state)

    raise NotImplementedError(
        f"Gradient is not implemented for {type(td).__name__}"
    )
