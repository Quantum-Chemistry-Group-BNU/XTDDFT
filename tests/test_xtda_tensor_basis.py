from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT_dev.XTDDFT import xtda


class XtdaTensorBasisTest(unittest.TestCase):
    def test_so2st_returns_cv0_co0_ov0_cv1_order(self):
        nc, no, nv = 1, 1, 2
        # spin-orbital order: CVa | OVa | COb | CVb
        cva = np.array([[1.0], [2.0]])
        ova = np.array([[3.0], [4.0]])
        cob = np.array([[5.0]])
        cvb = np.array([[7.0], [11.0]])
        so = np.vstack([cva, ova, cob, cvb])

        st = xtda._so2st(so, nc, no, nv)
        expected = np.vstack([
            (cva + cvb) / np.sqrt(2.0),
            cob,
            ova,
            (cva - cvb) / np.sqrt(2.0),
        ])

        self.assertTrue(np.allclose(st, expected))

    def test_restricted_ground_dipoles_use_tensor_basis(self):
        method = xtda.XTDA.__new__(xtda.XTDA)
        method.type_u = False
        method.nstates = 1
        method.nc = 1
        method.no = 1
        method.nv = 1
        method.v = np.array([[2.0], [3.0], [5.0], [7.0]])  # CV0 | CO0 | OV0 | CV1
        ints = np.zeros((3, 3, 3))
        ints[:, 2, 0] = np.array([11.0, 13.0, 17.0])  # V,C
        ints[:, 0, 1] = np.array([19.0, 23.0, 29.0])  # C,O
        ints[:, 1, 2] = np.array([31.0, 37.0, 41.0])  # O,V
        method._dipole_mo_blocks = lambda: (
            ints,
            ints,
            SimpleNamespace(
                occidx_a=np.array([0, 1]),
                viridx_a=np.array([2]),
                occidx_b=np.array([0]),
                viridx_b=np.array([1, 2]),
            ),
            None,
        )

        dip = method.transition_dipoles_ground()
        expected = (
            np.sqrt(2.0) * ints[:, 2, 0] * 2.0
            + ints[:, 0, 1] * 3.0
            + ints[:, 1, 2] * 5.0
        )

        self.assertEqual(dip.shape, (1, 3))
        self.assertTrue(np.allclose(dip[0], expected))

    def test_restricted_delta_s2_uses_cv1_tensor_component(self):
        method = xtda.XTDA.__new__(xtda.XTDA)
        method.type_u = False
        method.nc = 1
        method.no = 1
        method.nv = 2
        method.v = np.zeros((2 * method.nc * method.nv + method.nc * method.no + method.no * method.nv, 1))
        cv1 = np.array([[2.0, 3.0]])
        method.v[-method.nc * method.nv:, 0] = cv1.reshape(-1)

        self.assertTrue(np.allclose(method.deltaS2(), [2.0 * np.sum(cv1 * cv1)]))

    def test_unrestricted_ground_dipoles_keep_spin_orbital_basis(self):
        method = xtda.XTDA.__new__(xtda.XTDA)
        method.type_u = True
        method.nstates = 1
        method.nc = 1
        method.no = 1
        method.nv = 1
        method.v = np.array([[2.0], [3.0], [5.0], [7.0]])  # CVa | OVa | COb | CVb
        ints_aa = np.zeros((3, 3, 3))
        ints_bb = np.zeros((3, 3, 3))
        ints_aa[:, 0, 2] = np.array([11.0, 13.0, 17.0])  # CVa
        ints_aa[:, 1, 2] = np.array([19.0, 23.0, 29.0])  # OVa
        ints_bb[:, 0, 1] = np.array([31.0, 37.0, 41.0])  # COb
        ints_bb[:, 0, 2] = np.array([43.0, 47.0, 53.0])  # CVb
        method._dipole_mo_blocks = lambda: (
            ints_aa,
            ints_bb,
            SimpleNamespace(
                occidx_a=np.array([0, 1]),
                viridx_a=np.array([2]),
                occidx_b=np.array([0]),
                viridx_b=np.array([1, 2]),
            ),
            None,
        )

        dip = method.transition_dipoles_ground()
        expected = (
            ints_aa[:, 0, 2] * 2.0
            + ints_aa[:, 1, 2] * 3.0
            + ints_bb[:, 0, 1] * 5.0
            + ints_bb[:, 0, 2] * 7.0
        )

        self.assertTrue(np.allclose(dip[0], expected))

    def test_restricted_ground_transition_density_uses_tensor_blocks(self):
        method = xtda.XTDA.__new__(xtda.XTDA)
        method.type_u = False
        method.nstates = 1
        method.nc = 1
        method.no = 1
        method.nv = 1
        method.v = np.array([[2.0], [3.0], [5.0], [7.0]])  # CV0 | CO0 | OV0 | CV1

        gamma = method.transition_density_matrix(0, None)
        expected = np.zeros((3, 3))
        expected[2, 0] = np.sqrt(2.0) * 2.0
        expected[1, 0] = 3.0
        expected[2, 1] = 5.0

        self.assertTrue(np.allclose(gamma, expected), f"\ngamma=\n{gamma}\nexpected=\n{expected}")

    def test_restricted_transition_density_contracts_to_excited_dipoles(self):
        method = xtda.XTDA.__new__(xtda.XTDA)
        method.type_u = False
        method.nstates = 2
        method.nc = 1
        method.no = 1
        method.nv = 1
        method.mf = SimpleNamespace(mol=SimpleNamespace(spin=2))
        method.v = np.array([
            [2.0, 11.0],
            [3.0, 13.0],
            [5.0, 17.0],
            [7.0, 19.0],
        ])
        ints = np.arange(1, 3 * 3 * 3 + 1, dtype=float).reshape(3, 3, 3) / 17.0
        method._dipole_mo_blocks = lambda: (ints, ints, None, object())

        gamma = method.transition_density_matrix(1, 0)
        contracted = xtda.contract("mn,xmn->x", gamma, ints)
        tdm = method.transition_dipole_matrix()

        self.assertTrue(np.allclose(contracted, tdm[1, 0]))

    def test_restricted_transition_dipole_matrix_uses_direct_blocks(self):
        method = xtda.XTDA.__new__(xtda.XTDA)
        method.type_u = False
        method.nstates = 3
        method.nc = 2
        method.no = 2
        method.nv = 2
        method.mf = SimpleNamespace(mol=SimpleNamespace(spin=2))
        rng = np.random.default_rng(12)
        dim = 2 * method.nc * method.nv + method.nc * method.no + method.no * method.nv
        method.v = rng.normal(size=(dim, method.nstates))
        ints = rng.normal(size=(3, method.nc + method.no + method.nv, method.nc + method.no + method.nv))
        method._dipole_mo_blocks = lambda: (ints, ints, None, object())

        expected = np.zeros((method.nstates, method.nstates, 3))
        for i in range(method.nstates):
            for j in range(method.nstates):
                gamma = method.transition_density_matrix(i, j)
                expected[i, j] = xtda.contract("mn,xmn->x", gamma, ints)

        def fail_transition_density_matrix(*_args, **_kwargs):
            raise AssertionError("transition_dipole_matrix should not build full transition densities")

        method.transition_density_matrix = fail_transition_density_matrix
        tdm = method.transition_dipole_matrix()

        self.assertTrue(np.allclose(tdm, expected))

    def test_restricted_excited_transition_density_places_reverse_cross_blocks(self):
        method = xtda.XTDA.__new__(xtda.XTDA)
        method.type_u = False
        method.nstates = 2
        method.nc = 1
        method.no = 1
        method.nv = 1
        method.mf = SimpleNamespace(mol=SimpleNamespace(spin=2))
        method.v = np.array([
            [0.0, 2.0],   # CV(0)
            [3.0, 11.0],  # CO(0)
            [5.0, 13.0],  # OV(0)
            [7.0, 0.0],   # CV(1)
        ])

        gamma = method.transition_density_matrix(0, 1)
        expected = np.zeros((3, 3))
        expected[0, 0] = -3.0 * 11.0
        expected[1, 1] = 3.0 * 11.0 - 5.0 * 13.0
        expected[2, 2] = 5.0 * 13.0
        expected[1, 2] = 3.0 * 2.0 / np.sqrt(2.0)
        expected[0, 1] = -5.0 * 2.0 / np.sqrt(2.0)
        expected[2, 1] = 7.0 * 11.0
        expected[1, 0] = 7.0 * 13.0

        self.assertTrue(np.allclose(gamma, expected), f"\ngamma=\n{gamma}\nexpected=\n{expected}")

    def test_unrestricted_ground_transition_density_keeps_spin_blocks(self):
        method = xtda.XTDA.__new__(xtda.XTDA)
        method.type_u = True
        method.nstates = 1
        method.nc = 1
        method.no = 1
        method.nv = 1
        method.ctx = SimpleNamespace(
            mo_coeff=np.zeros((2, 3, 3)),
            occidx_a=np.array([0, 1]),
            viridx_a=np.array([2]),
            occidx_b=np.array([0]),
            viridx_b=np.array([1, 2]),
        )
        method.v = np.array([[2.0], [3.0], [5.0], [7.0]])  # CVa | OVa | COb | CVb

        gamma = method.transition_density_matrix(0, None)
        expected = np.zeros((6, 6))
        expected[2, 0] = 2.0
        expected[2, 1] = 3.0
        expected[3 + 1, 3 + 0] = 5.0
        expected[3 + 2, 3 + 0] = 7.0

        self.assertTrue(np.allclose(gamma, expected))

    def test_unrestricted_excited_transition_density_contracts_to_utda_dipoles(self):
        method = xtda.XTDA.__new__(xtda.XTDA)
        method.type_u = True
        method.nstates = 3
        method.nc = 1
        method.no = 1
        method.nv = 2
        method.ctx = SimpleNamespace(
            mo_coeff=np.zeros((2, 4, 4)),
            occidx_a=np.array([0, 1]),
            viridx_a=np.array([2, 3]),
            occidx_b=np.array([0]),
            viridx_b=np.array([1, 2, 3]),
        )
        rng = np.random.default_rng(21)
        dim = 2 * method.nc * method.nv + method.no * method.nv + method.nc * method.no
        method.v = rng.normal(size=(dim, method.nstates))
        ints_aa = rng.normal(size=(3, 4, 4))
        ints_bb = rng.normal(size=(3, 4, 4))
        method._dipole_mo_blocks = lambda: (ints_aa, ints_bb, method.ctx, object())

        ints_spin = np.zeros((3, 8, 8))
        ints_spin[:, :4, :4] = ints_aa
        ints_spin[:, 4:, 4:] = ints_bb
        expected = np.zeros((method.nstates, method.nstates, 3))
        for i in range(method.nstates):
            for j in range(method.nstates):
                gamma = method.transition_density_matrix(i, j)
                expected[i, j] = xtda.contract("mn,xmn->x", gamma, ints_spin)

        tdm = method.transition_dipole_matrix()

        self.assertTrue(np.allclose(tdm, expected))

    def test_nto_returns_singular_values_from_transition_density(self):
        method = xtda.XTDA.__new__(xtda.XTDA)
        method.type_u = False
        method.nstates = 1
        method.nc = 1
        method.no = 1
        method.nv = 1
        method.v = np.array([[2.0], [3.0], [5.0], [7.0]])

        singular_values, holes, particles = method.nto(0, None, nroots=2)
        expected = np.linalg.svd(method.transition_density_matrix(0, None), compute_uv=False)[:2]

        self.assertTrue(np.allclose(singular_values, expected))
        self.assertEqual(holes.shape, (3, 2))
        self.assertEqual(particles.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
