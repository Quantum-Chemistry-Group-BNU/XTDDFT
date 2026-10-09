from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT.XTDDFT import sf_tda_up, xtda
from XTDDFT.XTDDFT.base import XTDDFT_base


def test_gradient_module_imports_preserve_thread_environment():
    keys = (
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    modules = (
        "gradient_roks_sc",
        "gradient_uks_sc",
        "gradient_roks_sfu",
        "gradient_uks_sfu",
        "gradient_roks_sfd",
        "gradient_uks_sfd",
    )
    code = (
        "import importlib, importlib.util, os, sys; "
        f"spec=importlib.util.spec_from_file_location('XTDDFT', {str(ROOT / '__init__.py')!r}, "
        f"submodule_search_locations=[{str(ROOT)!r}]); "
        "package=importlib.util.module_from_spec(spec); "
        "sys.modules['XTDDFT']=package; spec.loader.exec_module(package); "
        f"keys={keys!r}; modules={modules!r}; "
        "[importlib.import_module('XTDDFT.XTDDFT.grad.' + name) for name in modules]; "
        "assert all(os.environ[key] == 'sentinel' for key in keys)"
    )
    env = os.environ.copy()
    env.update({key: "sentinel" for key in keys})

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_PARENT,
        env=env,
        check=True,
    )


def test_st2so_restores_spin_orbital_block_order():
    nc, no, nv = 2, 1, 3
    cva = np.arange(1, 7, dtype=float)[:, None]
    ova = np.arange(11, 14, dtype=float)[:, None]
    cob = np.arange(21, 23, dtype=float)[:, None]
    cvb = np.arange(31, 37, dtype=float)[:, None]
    spin_orbital = np.vstack((cva, ova, cob, cvb))
    spin_tensor = np.vstack(
        ((cva + cvb) / np.sqrt(2), cob, ova, (cvb - cva) / np.sqrt(2))
    )

    np.testing.assert_allclose(xtda._so2st(spin_orbital, nc, no, nv), spin_tensor)
    actual = xtda._st2so(spin_tensor, nc, no, nv)

    np.testing.assert_allclose(actual, spin_orbital)


def test_dense_mcol_uses_solver_collinear_samples():
    td = sf_tda_up.SF_TDA_up.__new__(sf_tda_up.SF_TDA_up)
    td.method = 1
    td.collinear_samples = 24
    td.A = None
    seen = []
    td.get_Amat_MCOL = lambda samples=30: seen.append(samples)

    td.get_Amat()

    assert seen == [24]


def test_davidson_mcol_uses_solver_collinear_samples(monkeypatch):
    td = sf_tda_up.SF_TDA_up.__new__(sf_tda_up.SF_TDA_up)
    td.method = 1
    td.collinear_samples = 24
    td.mf = object()
    td.ctx = object()
    td.isf = 1
    td._get_fock_mo = lambda: object()
    seen = []

    def fake_response(_mf, **kwargs):
        seen.append(kwargs["collinear_samples"])
        return object()

    monkeypatch.setattr(sf_tda_up, "gen_response_sf_mc", fake_response)
    monkeypatch.setattr(
        sf_tda_up,
        "_make_spinflip_problem",
        lambda *_args: SimpleNamespace(hdiag=np.zeros(1)),
    )
    monkeypatch.setattr(sf_tda_up, "_make_spinflip_vind", lambda *_args: object())

    td.gen_tda_operation_sf()

    assert seen == [24]


@pytest.mark.parametrize("samples", [0, -1, 1.5, True, None])
def test_mcol_rejects_invalid_collinear_samples(monkeypatch, samples):
    monkeypatch.setattr(XTDDFT_base, "__init__", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="positive integer"):
        sf_tda_up.SF_TDA_up(object(), method=1, collinear_samples=samples)


def _gradient_reference(reference="UKS", gpu=True, density_fit=True, only_dfj=False):
    """No CUDA needed: only the reference type and DF configuration are inspected."""
    cls = type(reference, (), {"__module__": "gpu4pyscf.dft" if gpu else "pyscf.dft"})
    mf = cls()
    mf.with_df = object() if density_fit else None
    mf.only_dfj = only_dfj
    mf.xc = "b3lyp"
    mf._numint = SimpleNamespace(_xc_type=lambda _xc: "GGA")
    return mf


@pytest.mark.parametrize("reference", ["UKS", "ROKS"])
def test_gpu_df_gradient_is_accepted(reference):
    from XTDDFT.XTDDFT.grad._backend import validate_gradient_backend

    validate_gradient_backend(_gradient_reference(reference=reference))


@pytest.mark.parametrize("kwargs", [
    {"only_dfj": True},
    {"gpu": False, "only_dfj": True},
])
def test_unsupported_df_gradient_is_rejected(kwargs):
    from XTDDFT.XTDDFT.grad._backend import validate_gradient_backend

    with pytest.raises(NotImplementedError, match="density fitting|DF|RI-J"):
        validate_gradient_backend(_gradient_reference(**kwargs))


@pytest.mark.parametrize("module_name,class_name", [
    ("gradient_uks_sfd", "SFD_gradient"),
    ("gradient_uks_sfu", "SFU_gradient"),
])
def test_uks_sf_gradient_keeps_excitation_sampling(module_name, class_name):
    from importlib import import_module

    cls = getattr(import_module("XTDDFT.XTDDFT.grad." + module_name), class_name)
    td = SimpleNamespace(mf=_gradient_reference(density_fit=False), mol=None,
                         v=np.zeros((1, 1)), collinear_samples=24)
    gradient = cls(td, method=1)
    assert gradient.base.collinear_samples == 24
    assert td.collinear_samples == 24


@pytest.mark.parametrize("module_name", ["gradient_uks_sfd", "gradient_uks_sfu",
                                         "gradient_roks_sfd", "gradient_roks_sfu",
                                         "gradient_roks_sc"])
@pytest.mark.parametrize("omega", [None, 0.3])
def test_sf_df_derivative_dispatch_preserves_nonsymmetric_pairs(monkeypatch, module_name, omega):
    """Catches accidental direct-integral routing or transpose/symmetry loss without CUDA."""
    from importlib import import_module
    from types import ModuleType

    module = import_module("XTDDFT.XTDDFT.grad." + module_name)
    mf = _gradient_reference(reference="ROKS" if "roks" in module_name else "UKS")
    transition = np.array([[0., 2.], [-1., 0.]])
    pairs = [[transition, transition.T]]
    expected = np.array([[1., -2., 3.]])

    def df_derivative(grad, dm_list, j_factor=None, k_factor=None, **kwargs):
        assert grad.base._scf is mf
        assert grad.auxbasis_response is True
        np.testing.assert_array_equal(dm_list[0][0], transition)
        np.testing.assert_array_equal(dm_list[0][1], transition.T)
        assert j_factor == [0.]
        assert k_factor == [0.4]
        assert kwargs["omega"] == (0. if omega is None else omega)
        assert kwargs["sum_results"] is True
        return expected.copy()

    # Only the CUDA boundary is replaced. The production dispatch runs unchanged.
    for name in ("gpu4pyscf", "gpu4pyscf.df", "gpu4pyscf.df.grad",
                 "gpu4pyscf.df.grad.tdrhf"):
        stub = ModuleType(name)
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, name, stub)
    sys.modules["gpu4pyscf.df.grad.tdrhf"].Gradients = SimpleNamespace(
        jk_energies_per_atom=df_derivative)
    actual = module.jk_energies_per_atom(
        mf, pairs, [0.], [0.4], omega=omega, sum_results=True)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("reference", ["UKS", "ROKS"])
def test_cpu_df_gradient_is_accepted(reference):
    from XTDDFT.XTDDFT.grad._backend import validate_gradient_backend
    validate_gradient_backend(_gradient_reference(gpu=False, reference=reference))
