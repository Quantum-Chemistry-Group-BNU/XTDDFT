import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from XTDDFT.utils import hxc_part


@pytest.mark.parametrize("has_sf_kernel", [True, False])
def test_mc_gpu_response_imports_optional_module(monkeypatch, has_sf_kernel):
    """Exercise both API branches without requiring GPU4PySCF or a GPU."""
    response = ModuleType("gpu4pyscf.tdscf._uhf_resp_sf")
    if has_sf_kernel:
        response.nr_uks_fxc_sf = lambda *args: args[5] * args[-1]
    tdscf = ModuleType("gpu4pyscf.tdscf")
    tdscf._uhf_resp_sf = response
    package = ModuleType("gpu4pyscf")
    package.tdscf = tdscf
    for module in (package, tdscf, response):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    monkeypatch.setattr(hxc_part, "_response_max_memory", lambda *args: 2000)
    monkeypatch.setattr(hxc_part, "_xc_response_params", lambda *args: ("LDA", False, 0, 0, 0))
    monkeypatch.setattr(hxc_part, "_cache_xc_kernel_sf_mc_gpu_mol", lambda *args: 3.0)
    monkeypatch.setattr(hxc_part, "require_cupy", lambda: np)
    ni = SimpleNamespace(nr_rks_fxc=lambda *args: args[4] * args[-1])
    mf = SimpleNamespace(_numint=ni, mol=None, grids=None, xc="LDA")

    vind = hxc_part._gen_response_sf_mc_gpu_mol(mf, None, None)

    np.testing.assert_array_equal(vind(np.ones((2, 2))),
                                  np.full((2, 2), 3.0 if has_sf_kernel else 6.0))
