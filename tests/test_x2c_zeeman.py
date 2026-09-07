import unittest

import numpy as np
from pyscf import gto, scf

from XTDDFT_dev.XTDDFT.soc import get_zeeman


class X2CZeemanTest(unittest.TestCase):
    def test_carbon_sto3g_nonzero_and_symmetric(self):
        mol = gto.M(
            atom="C 0 0 0",
            basis="sto-3g",
            spin=0,
            verbose=0,
        )
        mf = scf.sfx2c(scf.RHF(mol)).run()
        self.assertTrue(mf.converged)

        h10, h11 = get_zeeman(
            mf,
            mol,
            c=137.0359895,
            origin=np.zeros(3),
            backend="cpu",
        )

        self.assertEqual(h10.shape, (3, mol.nao, mol.nao))
        self.assertEqual(h11.shape, (3, 3, mol.nao, mol.nao))
        self.assertGreater(np.linalg.norm(h10), 1e-6)
        self.assertGreater(np.linalg.norm(h11), 1e-6)
        self.assertTrue(np.allclose(h10 + h10.swapaxes(-1, -2), 0.0))
        self.assertTrue(np.allclose(h11 - h11.swapaxes(-1, -2), 0.0))


if __name__ == "__main__":
    unittest.main()
