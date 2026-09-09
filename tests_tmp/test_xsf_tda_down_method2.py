from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT_dev.XTDDFT import xsf_tda_down
from XTDDFT_dev.utils import hxc_part


class XsfTdaDownMethod2Test(unittest.TestCase):
    def test_collinear_response_skips_xc_and_keeps_hybrid_exchange(self):
        mf = SimpleNamespace(
            xc="PBE0",
            _numint=SimpleNamespace(
                libxc=SimpleNamespace(test_deriv_order=lambda *args, **kwargs: None)
            ),
        )
        ctx = SimpleNamespace(
            mo_occ=np.zeros((2, 1)),
            mo_coeff=np.ones((2, 1, 1)),
        )
        dm1 = np.eye(2)
        exchange = np.full((2, 2), 0.25)

        with (
            patch.object(hxc_part, "_is_gpu_mf", return_value=False),
            patch.object(hxc_part, "_is_ks_mf", return_value=True),
            patch.object(
                hxc_part, "_xc_response_params",
                return_value=("GGA", True, 0.0, 0.0, 0.25),
            ),
            patch.object(hxc_part, "cache_xc_kernel_sf") as cache_xc,
            patch.object(hxc_part, "_hybrid_k", return_value=exchange),
        ):
            vind = hxc_part.gen_response_sf(
                mf, max_memory=1, ctx=ctx, with_xc=False,
            )
            result = vind(dm1)

        cache_xc.assert_not_called()
        np.testing.assert_allclose(result, -exchange)

    def test_hf_response_is_independent_of_with_xc(self):
        mf = object()
        ctx = SimpleNamespace(
            mo_occ=np.zeros((2, 1)),
            mo_coeff=np.ones((2, 1, 1)),
        )
        dm1 = np.eye(2)
        exchange = np.full((2, 2), 0.5)

        with (
            patch.object(hxc_part, "_is_gpu_mf", return_value=False),
            patch.object(hxc_part, "_is_ks_mf", return_value=False),
            patch.object(hxc_part, "_get_k", return_value=exchange),
        ):
            alda0 = hxc_part.gen_response_sf(mf, ctx=ctx, with_xc=True)(dm1)
            collinear = hxc_part.gen_response_sf(mf, ctx=ctx, with_xc=False)(dm1)

        np.testing.assert_allclose(alda0, collinear)
        np.testing.assert_allclose(collinear, -exchange)

    def test_method2_requires_davidson(self):
        mf = SimpleNamespace(mo_coeff=np.eye(2))

        def base_init(method_obj, mf_obj, method, davidson=True, df_cache=None):
            method_obj.mf = mf_obj
            method_obj.method = method
            method_obj.davidson = davidson

        with (
            patch.object(xsf_tda_down.XTDDFT_base, "__init__", base_init),
            patch.object(
                xsf_tda_down, "_as_cpu_mf",
                return_value=SimpleNamespace(spin_square=lambda: (0.0, 3.0)),
            ),
        ):
            method = xsf_tda_down.XSF_TDA_down(mf, method=2, SA=0)
            self.assertEqual(method._result_method_label(), "COL")
            with self.assertRaisesRegex(NotImplementedError, "Davidson"):
                xsf_tda_down.XSF_TDA_down(mf, method=2, davidson=False, SA=0)

    def test_method2_davidson_dispatches_without_xc(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.method = 2
        method.SA = 0
        method.type_u = True
        method.re = False
        method.nc = 1
        method.no = 0
        method.nv = 1
        method.nocc_a = 1
        method.nocc_b = 1
        method.mo_coeff = np.stack([np.eye(2), np.eye(2)])
        method.mo_occ = np.array([[1.0, 0.0], [1.0, 0.0]])
        method.occidx_a = np.array([0])
        method.viridx_b = np.array([1])
        method.ctx = object()
        method.mf = SimpleNamespace(mo_coeff=method.mo_coeff)
        method._get_fock_mo = lambda: (np.eye(2), np.eye(2))

        with patch.object(
            xsf_tda_down, "gen_response_sf", return_value=lambda dm1: dm1,
        ) as gen_response:
            method.gen_tda_operation_sf()

        self.assertFalse(gen_response.call_args.kwargs["with_xc"])


if __name__ == "__main__":
    unittest.main()
