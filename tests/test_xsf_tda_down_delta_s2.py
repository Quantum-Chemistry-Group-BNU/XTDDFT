from pathlib import Path
import inspect
from types import SimpleNamespace
import unittest

import numpy as np

import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT_dev.XTDDFT import xsf_tda_down


class XsfTdaDownDeltaS2Test(unittest.TestCase):
    def make_method(self, nstates=3):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.mf = object()
        method.ctx = object()
        method.type_u = True
        method.SA = 0
        method.re = False
        method.nc = 1
        method.no = 1
        method.nv = 2
        method.nstates = nstates
        method.v = np.arange(6 * nstates, dtype=float).reshape(6, nstates) / 10.0
        return method

    def test_uks_delta_s2_reuses_cpu_overlap_context(self):
        method = self.make_method()

        mo_coeff = np.stack([np.eye(4), np.eye(4)])
        ctx = SimpleNamespace(
            mo_coeff=mo_coeff,
            occidx_a=np.array([0, 1]),
            occidx_b=np.array([0]),
            viridx_a=np.array([2, 3]),
            viridx_b=np.array([1, 2, 3]),
        )
        calls = {"mf": 0, "ctx": 0, "ovlp": 0}

        def as_cpu_mf(mf):
            calls["mf"] += 1
            return mf

        def as_cpu_ctx(mf, old_ctx):
            calls["ctx"] += 1
            return ctx

        def get_ovlp(mf):
            calls["ovlp"] += 1
            return np.eye(4)

        old_as_cpu_mf = xsf_tda_down._as_cpu_mf
        old_as_cpu_ctx = xsf_tda_down._as_cpu_ctx
        old_get_ovlp = xsf_tda_down._get_ovlp
        try:
            xsf_tda_down._as_cpu_mf = as_cpu_mf
            xsf_tda_down._as_cpu_ctx = as_cpu_ctx
            xsf_tda_down._get_ovlp = get_ovlp

            ds2 = method.deltaS2()
        finally:
            xsf_tda_down._as_cpu_mf = old_as_cpu_mf
            xsf_tda_down._as_cpu_ctx = old_as_cpu_ctx
            xsf_tda_down._get_ovlp = old_get_ovlp

        self.assertEqual(ds2.shape, (3,))
        self.assertTrue(np.all(np.isfinite(ds2)))
        self.assertEqual(calls, {"mf": 1, "ctx": 1, "ovlp": 1})

    def test_delta_s2_optimized_contraction_matches_direct_formula(self):
        method = self.make_method(nstates=1)
        x_cv, x_co, x_ov, x_oo = method._split_analysis_vectors(method.v[:, 0])
        x_ba = np.concatenate([np.hstack([x_co, x_cv]), np.hstack([x_oo, x_ov])], axis=0).T
        sba_oo = np.array([[0.2 + 0.1j, -0.3 + 0.4j]])
        sba_vo = np.array([
            [0.5 - 0.2j, 0.1 + 0.3j],
            [-0.4 + 0.6j, 0.2 - 0.1j],
            [0.7 + 0.1j, -0.5 + 0.2j],
        ])

        expected = (
            xsf_tda_down.contract("ai,aj,jk,ki", x_ba.conj(), x_ba, sba_oo.T.conj(), sba_oo)
            - xsf_tda_down.contract("ai,bi,kb,ak", x_ba.conj(), x_ba, sba_vo.T.conj(), sba_vo)
            + xsf_tda_down.contract("ai,ai->", x_ba.conj(), sba_vo)
            * xsf_tda_down.contract("ai,ai->", x_ba, sba_vo.conj())
        )
        actual = method._deltaS2_U_from_overlaps(0, sba_oo, sba_vo)

        self.assertTrue(np.allclose(actual, expected))

    def test_delta_s2_u_helpers_use_wrapped_contract(self):
        source = (
            inspect.getsource(xsf_tda_down.XSF_TDA_down._deltaS2_U_overlaps)
            + inspect.getsource(xsf_tda_down.XSF_TDA_down._deltaS2_U_from_overlaps)
        )

        self.assertNotIn("np.einsum", source)
        self.assertNotIn("np.vdot", source)
        self.assertNotIn(" @ ", source)
        self.assertIn("contract(", source)


if __name__ == "__main__":
    unittest.main()
