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

from XTDDFT_dev.XTDDFT import sf_tda_up, xtda
from XTDDFT_dev.XTDDFT.base import XTDDFT_base


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
        "import importlib, os; "
        f"keys={keys!r}; modules={modules!r}; "
        "[importlib.import_module('XTDDFT_dev.XTDDFT.grad.' + name) for name in modules]; "
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
