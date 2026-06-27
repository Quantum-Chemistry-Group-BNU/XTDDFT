from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT_dev.XTDDFT import sf_tda_up


class SfTdaUpNtoTest(unittest.TestCase):
    def make_method(self):
        method = sf_tda_up.SF_TDA_up.__new__(sf_tda_up.SF_TDA_up)
        method.nc = 2
        method.no = 1
        method.nv = 2
        method.nstates = 3
        nmo = method.nc + method.no + method.nv
        method.mf = SimpleNamespace(mol=object())
        method.ctx = SimpleNamespace(
            mo_coeff=np.stack([np.eye(nmo), np.eye(nmo)]),
            occidx_b=np.array([0, 1]),
            viridx_a=np.array([3, 4]),
        )
        rng = np.random.default_rng(31)
        method.v = rng.normal(size=(method.nc * method.nv, method.nstates))
        method.e = np.arange(1, method.nstates + 1, dtype=float)
        return method, rng

    def test_transition_density_contracts_to_transition_dipole_matrix(self):
        method, rng = self.make_method()
        nmo = method.nc + method.no + method.nv
        dip_ao = rng.normal(size=(3, nmo, nmo))

        ints_spin = np.zeros((3, 2 * nmo, 2 * nmo))
        ints_spin[:, :nmo, :nmo] = dip_ao
        ints_spin[:, nmo:, nmo:] = dip_ao

        expected = np.zeros((method.nstates, method.nstates, 3))
        for i in range(method.nstates):
            for j in range(method.nstates):
                gamma = method.transition_density_matrix(i, j)
                expected[i, j] = sf_tda_up.contract("mn,xmn->x", gamma, ints_spin)

        with mock.patch.object(sf_tda_up, "_molecular_dipole_integrals", return_value=dip_ao):
            tdm = method.transition_dipole_matrix()

        self.assertTrue(np.allclose(tdm, expected))

    def test_ground_state_transition_density_is_zero(self):
        method, _rng = self.make_method()
        nmo = method.nc + method.no + method.nv

        gamma = method.transition_density_matrix(0, None)

        self.assertEqual(gamma.shape, (2 * nmo, 2 * nmo))
        self.assertTrue(np.allclose(gamma, 0.0))

    def test_nto_returns_singular_values_from_transition_density(self):
        method, _rng = self.make_method()

        singular_values, holes, particles = method.nto(2, 1, nroots=3)
        expected = np.linalg.svd(method.transition_density_matrix(2, 1), compute_uv=False)[:3]

        self.assertTrue(np.allclose(singular_values, expected))
        self.assertEqual(holes.shape, (10, 3))
        self.assertEqual(particles.shape, (10, 3))


    def test_block_nto_returns_spinflip_cv_svd(self):
        method, _rng = self.make_method()

        result = method.block_nto(0, nroots=1)
        amp = method._state_amplitude(0)
        expected_s = np.linalg.svd(amp.T, compute_uv=False)

        self.assertEqual(set(result), {"CV"})
        block = result["CV"]
        self.assertEqual(block["source"], "beta_occ")
        self.assertEqual(block["target"], "alpha_vir")
        self.assertTrue(np.allclose(block["singular_values"], expected_s[:1]))
        self.assertTrue(np.allclose(block["weights"], expected_s[:1] ** 2))
        self.assertAlmostEqual(block["block_weight"], np.sum(np.abs(amp) ** 2))
        self.assertEqual(block["holes"].shape, (10, 1))
        self.assertEqual(block["particles"].shape, (10, 1))

if __name__ == "__main__":
    unittest.main()
