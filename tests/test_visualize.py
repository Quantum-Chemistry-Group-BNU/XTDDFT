from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))


class VisualizeNtoCubesTest(unittest.TestCase):
    def test_restricted_nto_cubes_transform_mo_to_ao_coefficients(self):
        from XTDDFT_dev.utils import visualize

        mol = object()
        mo_coeff = np.arange(1, 13, dtype=float).reshape(3, 4)
        method = SimpleNamespace(
            type_u=False,
            mf=SimpleNamespace(mol=mol),
            ctx=SimpleNamespace(
                mo_coeff=np.stack([mo_coeff, mo_coeff]),
                occidx_a=np.array([0, 1]),
                viridx_a=np.array([2, 3]),
            ),
        )
        singular_values = np.array([0.8])
        holes = np.array([[1.0], [2.0], [3.0], [4.0]])
        particles = np.array([[0.5], [0.0], [1.5], [2.0]])
        method.nto = lambda state_f, state_i, nroots=None: (singular_values, holes, particles)

        calls = []

        def fake_orbital(system, outfile, coeff, **kwargs):
            calls.append((system, Path(outfile).name, np.asarray(coeff), kwargs))

        with tempfile.TemporaryDirectory() as tmpdir:
            result = visualize.write_nto_cubes(
                method,
                state_f=2,
                state_i=1,
                nroots=1,
                outdir=tmpdir,
                prefix="demo",
                cubegen_orbital=fake_orbital,
                nx=10,
                ny=11,
                nz=12,
            )

        expected_hole = mo_coeff @ holes[:, 0]
        expected_particle = mo_coeff @ particles[:, 0]
        self.assertTrue(np.allclose(result["singular_values"], singular_values))
        self.assertEqual([call[1] for call in calls], ["demo_nto1_hole.cube", "demo_nto1_particle.cube"])
        self.assertIs(calls[0][0], mol)
        self.assertTrue(np.allclose(calls[0][2], expected_hole))
        self.assertTrue(np.allclose(calls[1][2], expected_particle))
        self.assertEqual(calls[0][3]["nx"], 10)
        self.assertEqual(calls[0][3]["ny"], 11)
        self.assertEqual(calls[0][3]["nz"], 12)

    def test_restricted_ground_state_prefix_and_arguments(self):
        from XTDDFT_dev.utils import visualize

        mol = object()
        mo_coeff = np.eye(2)
        method = SimpleNamespace(
            type_u=False,
            mf=SimpleNamespace(mol=mol),
            ctx=SimpleNamespace(
                mo_coeff=np.stack([mo_coeff, mo_coeff]),
                occidx_a=np.array([0]),
                viridx_a=np.array([1]),
            ),
        )
        seen_args = []

        def fake_nto(state_f, state_i, nroots=None):
            seen_args.append((state_f, state_i, nroots))
            return np.array([1.0]), np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])

        method.nto = fake_nto
        calls = []

        def fake_orbital(system, outfile, coeff, **kwargs):
            calls.append(Path(outfile).name)

        with tempfile.TemporaryDirectory() as tmpdir:
            visualize.write_nto_cubes(
                method,
                state_f=3,
                state_i=None,
                nroots=1,
                outdir=tmpdir,
                cubegen_orbital=fake_orbital,
            )

        self.assertEqual(seen_args, [(3, None, 1)])
        self.assertEqual(
            calls,
            [
                "stateF3_stateIground_nto1_hole.cube",
                "stateF3_stateIground_nto1_particle.cube",
            ],
        )

    def test_unrestricted_nto_cubes_write_nonzero_spin_components(self):
        from XTDDFT_dev.utils import visualize

        mol = object()
        mo_a = np.arange(1, 13, dtype=float).reshape(3, 4)
        mo_b = np.arange(21, 33, dtype=float).reshape(3, 4)
        method = SimpleNamespace(
            type_u=True,
            mf=SimpleNamespace(mol=mol),
            ctx=SimpleNamespace(mo_coeff=np.stack([mo_a, mo_b])),
        )
        singular_values = np.array([1.2])
        holes = np.zeros((8, 1))
        particles = np.zeros((8, 1))
        holes[:4, 0] = np.array([1.0, -1.0, 0.0, 2.0])
        particles[4:, 0] = np.array([0.5, 1.5, -0.5, 0.0])
        method.nto = lambda state_f, state_i, nroots=None: (singular_values, holes, particles)

        calls = []

        def fake_orbital(system, outfile, coeff, **kwargs):
            calls.append((Path(outfile).name, np.asarray(coeff)))

        with tempfile.TemporaryDirectory() as tmpdir:
            result = visualize.write_nto_cubes(
                method,
                outdir=tmpdir,
                prefix="u",
                cubegen_orbital=fake_orbital,
            )

        self.assertEqual([call[0] for call in calls], ["u_nto1_hole_alpha.cube", "u_nto1_particle_beta.cube"])
        self.assertTrue(np.allclose(calls[0][1], mo_a @ holes[:4, 0]))
        self.assertTrue(np.allclose(calls[1][1], mo_b @ particles[4:, 0]))
        self.assertEqual(len(result["files"]), 2)


if __name__ == "__main__":
    unittest.main()
