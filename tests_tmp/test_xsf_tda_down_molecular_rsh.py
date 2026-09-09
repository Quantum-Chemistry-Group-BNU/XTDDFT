from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np
from pyscf import ao2mo, gto


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT_dev.XTDDFT.xsf_tda_down import add_hf_a_a2b


class MolecularRangeSeparatedExchangeTest(unittest.TestCase):
    def test_matches_pyscf_range_separated_ao2mo(self):
        mol = gto.M(
            atom="H 0 0 0; F 0 0 0.9",
            basis="sto-3g",
            spin=0,
            verbose=0,
        )
        coeff = np.eye(mol.nao_nr())
        orbo = coeff[:, :2]
        orbv = coeff[:, 2:]
        omega = 0.33
        scale = 0.65

        with mol.with_range_coulomb(omega):
            eri = ao2mo.general(
                mol, [orbo, orbo, orbv, orbv], compact=False,
            ).reshape(2, 2, orbv.shape[1], orbv.shape[1])
        expected = -np.einsum("ijba->iajb", eri) * scale

        result = add_hf_a_a2b(
            np.zeros_like(expected), SimpleNamespace(mol=mol),
            orbo, orbv, 2, orbv.shape[1], hyb=scale, omega=omega,
        )

        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(mol.omega, 0.0)


if __name__ == "__main__":
    unittest.main()
