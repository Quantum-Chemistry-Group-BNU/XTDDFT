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

    def test_uks_transition_dipole_uses_wrapped_contract(self):
        source = inspect.getsource(xsf_tda_down.XSF_TDA_down._transition_dipole_matrix_u)

        self.assertNotIn("np.einsum", source)
        self.assertIn("contract(", source)

    def test_restricted_transition_density_contracts_to_transition_dipole(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.type_u = False
        method.re = False
        method.SA = 3
        method.nc = 1
        method.no = 2
        method.nv = 1
        method.ground_s = 1.0
        method.nstates = 2
        dim = (method.nc + method.no) * (method.no + method.nv)
        method.v = np.arange(1, dim * method.nstates + 1, dtype=float).reshape(dim, method.nstates) / 10.0

        nmo = method.nc + method.no + method.nv
        ints_mo = np.arange(1, 3 * nmo * nmo + 1, dtype=float).reshape(3, nmo, nmo) / 17.0
        method._dipole_mo_integrals = lambda: (ints_mo, None)

        gamma = method.transition_density_matrix(0, 1)
        tdm = method._transition_dipole_matrix_r()
        contracted = xsf_tda_down.contract("mn,xmn->x", gamma, ints_mo)

        self.assertEqual(gamma.shape, (nmo, nmo))
        self.assertTrue(np.allclose(contracted, tdm[0, 1]))

    def test_unrestricted_transition_density_contracts_to_transition_dipole(self):
        method = self.make_method(nstates=2)
        method.ctx = SimpleNamespace(
            mo_coeff=np.zeros((2, 4, 4)),
            occidx_a=np.array([0, 1]),
            viridx_b=np.array([1, 2, 3]),
        )
        ints_aa = np.arange(1, 3 * 4 * 4 + 1, dtype=float).reshape(3, 4, 4) / 13.0
        ints_bb = np.arange(101, 101 + 3 * 4 * 4, dtype=float).reshape(3, 4, 4) / 19.0
        method._dipole_mo_integrals = lambda: (ints_aa, ints_bb, method.ctx)

        gamma = method.transition_density_matrix(0, 1)
        tdm = method._transition_dipole_matrix_u()
        spin_block_integrals = np.zeros((3, 8, 8))
        spin_block_integrals[:, :4, :4] = ints_aa
        spin_block_integrals[:, 4:, 4:] = ints_bb
        contracted = xsf_tda_down.contract("mn,xmn->x", gamma, spin_block_integrals)

        self.assertEqual(gamma.shape, (8, 8))
        self.assertTrue(np.allclose(contracted, tdm[0, 1]))

    def test_nto_returns_singular_values(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.type_u = False
        method.re = False
        method.SA = 0
        method.nc = 1
        method.no = 1
        method.nv = 2
        method.ground_s = 0.5
        method.nstates = 2
        dim = (method.nc + method.no) * (method.no + method.nv)
        method.v = np.arange(1, dim * method.nstates + 1, dtype=float).reshape(dim, method.nstates) / 10.0

        singular_values, holes, particles = method.nto(0, 1, nroots=2)
        gamma = method.transition_density_matrix(0, 1)
        expected = np.linalg.svd(gamma, compute_uv=False)[:2]

        self.assertTrue(np.allclose(singular_values, expected))
        self.assertEqual(holes.shape, (gamma.shape[1], 2))
        self.assertEqual(particles.shape, (gamma.shape[0], 2))

    def test_block_nto_returns_spin_adapted_block_svds(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.type_u = False
        method.re = False
        method.nc = 2
        method.no = 2
        method.nv = 3
        method.nstates = 1
        dim = (method.nc + method.no) * (method.no + method.nv)
        method.v = np.arange(1, dim + 1, dtype=float).reshape(dim, 1) / 10.0

        result = method.block_nto(0, nroots=1)
        cv, co, ov, oo = method._split_analysis_vectors(method.v[:, 0])
        expected_matrices = {
            "CV": cv.T,
            "CO": co.T,
            "OV": ov.T,
            "OO": oo,
        }

        self.assertEqual(set(result), {"CV", "CO", "OV", "OO"})
        nmo = method.nc + method.no + method.nv
        for name, matrix in expected_matrices.items():
            block = result[name]
            expected_s = np.linalg.svd(matrix, compute_uv=False)
            self.assertTrue(np.allclose(block["singular_values"], expected_s[:1]))
            self.assertTrue(np.allclose(block["weights"], expected_s[:1] ** 2))
            self.assertAlmostEqual(block["block_weight"], np.sum(matrix * matrix))
            self.assertEqual(block["holes"].shape, (nmo, 1))
            self.assertEqual(block["particles"].shape, (nmo, 1))

        self.assertEqual(result["CV"]["source"], "C")
        self.assertEqual(result["CV"]["target"], "V")
        self.assertEqual(result["CO"]["source"], "C")
        self.assertEqual(result["CO"]["target"], "O")
        self.assertEqual(result["OV"]["source"], "O")
        self.assertEqual(result["OV"]["target"], "V")
        self.assertEqual(result["OO"]["source"], "O")
        self.assertEqual(result["OO"]["target"], "O")
        self.assertIn("trace_overlap", result["OO"])
        self.assertEqual(result["OO"]["trace_overlap"].shape, (1,))

    def test_delta_a_jk_response_batches_density_matrices(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.delta_a_jk_batch_size = 2
        dm_hf = np.arange(5 * 2 * 2, dtype=float).reshape(5, 2, 2)
        batch_sizes = []

        def vresp_hf(batch):
            batch_sizes.append(batch.shape[0])
            return batch + 10.0, batch + 20.0

        v1_j, v1_k = method._apply_delta_a_jk_response(vresp_hf, dm_hf)

        self.assertEqual(batch_sizes, [2, 2, 1])
        self.assertTrue(np.allclose(v1_j, dm_hf + 10.0))
        self.assertTrue(np.allclose(v1_k, dm_hf + 20.0))


if __name__ == "__main__":
    unittest.main()
