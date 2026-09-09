from pathlib import Path
from types import SimpleNamespace
import importlib
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT_dev.XTDDFT import grad
from XTDDFT_dev.XTDDFT.sf_tda_up import SF_TDA_up
from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT_dev.XTDDFT.xtda import XTDA


class ROKS:
    cell = None
    mo_coeff = np.zeros((2, 2))


class UKS:
    cell = None
    mo_coeff = np.zeros((2, 2, 2))


def solved(solver_cls, mf_cls, method):
    td = solver_cls.__new__(solver_cls)
    td.mf = mf_cls()
    td.mol = SimpleNamespace()
    td.method = method
    td.e = np.zeros(2)
    td.v = np.zeros((4, 2))
    return td


@pytest.mark.parametrize(
    "solver_cls,mf_cls,method,module_tail,class_name",
    [
        (XTDA, ROKS, 0, "gradient_roks_sc", "SC_gradient"),
        (XTDA, UKS, 0, "gradient_uks_sc", "SC_gradient"),
        (SF_TDA_up, ROKS, 1, "gradient_roks_sfu", "SFU_gradient"),
        (SF_TDA_up, UKS, 1, "gradient_uks_sfu", "SFU_gradient"),
        (XSF_TDA_down, ROKS, 1, "gradient_roks_sfd", "SFD_gradient"),
        (XSF_TDA_down, ROKS, 2, "gradient_roks_sfd", "SFD_gradient"),
        (XSF_TDA_down, UKS, 1, "gradient_uks_sfd", "SFD_gradient"),
        (XSF_TDA_down, UKS, 2, "gradient_uks_sfd", "SFD_gradient"),
    ],
)
def test_dispatches_supported_gradient_route(
    monkeypatch, solver_cls, mf_cls, method, module_tail, class_name
):
    module = importlib.import_module(f"XTDDFT_dev.XTDDFT.grad.{module_tail}")

    class FakeGradient:
        def __init__(self, td, method=None, state=1):
            self.base = td
            self.method = method
            self.state = state

    monkeypatch.setattr(module, class_name, FakeGradient)
    td = solved(solver_cls, mf_cls, method)

    result = grad.nuc_grad_method(td, state=2)

    assert result.base is td
    assert result.state == 2


@pytest.mark.parametrize("state", [0, -1, True, 1.0, "1", None])
def test_rejects_non_positive_or_non_integer_state(state):
    with pytest.raises(ValueError, match="positive integer"):
        grad.nuc_grad_method(solved(XTDA, ROKS, 0), state=state)


@pytest.mark.parametrize("missing", ["e", "v"])
def test_requires_completed_td_kernel(missing):
    td = solved(XTDA, ROKS, 0)
    setattr(td, missing, None)

    with pytest.raises(RuntimeError, match=r"td\.kernel"):
        grad.nuc_grad_method(td)


def test_rejects_state_beyond_available_roots():
    with pytest.raises(ValueError, match="only 2 states"):
        grad.nuc_grad_method(solved(XTDA, ROKS, 0), state=3)


@pytest.mark.parametrize(
    "solver_cls,method",
    [(XTDA, 1), (SF_TDA_up, 0), (SF_TDA_up, 2), (XSF_TDA_down, 0)],
)
def test_rejects_unsupported_solver_method(solver_cls, method):
    with pytest.raises(NotImplementedError, match=f"method={method}"):
        grad.nuc_grad_method(solved(solver_cls, ROKS, method))


def test_rejects_non_roks_or_uks_reference():
    class RKS:
        cell = None
        mo_coeff = np.zeros((2, 2))

    with pytest.raises(NotImplementedError, match="ROKS or UKS"):
        grad.nuc_grad_method(solved(XTDA, RKS, 0))


def test_rejects_unknown_solver():
    class UnknownSolver:
        pass

    with pytest.raises(NotImplementedError, match="UnknownSolver"):
        grad.nuc_grad_method(solved(UnknownSolver, ROKS, 0))


def test_rejects_periodic_reference():
    td = solved(XTDA, ROKS, 0)
    td.mf.cell = object()

    with pytest.raises(NotImplementedError, match="molecular"):
        grad.nuc_grad_method(td)


def test_rejects_gpu_reference():
    class GPUUKS:
        cell = None
        mo_coeff = np.zeros((2, 2, 2))

    GPUUKS.__module__ = "gpu4pyscf.fake"

    with pytest.raises(NotImplementedError, match="CPU backend"):
        grad.nuc_grad_method(solved(XTDA, GPUUKS, 0))


def test_base_method_delegates_to_dispatcher(monkeypatch):
    td = solved(XTDA, ROKS, 0)
    sentinel = object()
    monkeypatch.setattr(grad, "nuc_grad_method", lambda obj, state: sentinel)

    assert td.nuc_grad_method(state=2) is sentinel
