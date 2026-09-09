import unittest
from types import SimpleNamespace

import numpy as np
from pyscf import dft, gto, scf

from XTDDFT_dev.utils.backend import set_backend
from XTDDFT_dev.XTDDFT.soc.soc_si import SOCSI


class SocSiTest(unittest.TestCase):
    def test_si_driver_small_matrix(self):
        from XTDDFT_dev.XTDDFT.soc.si_driver import SI_driver

        mol = SimpleNamespace(spin=1, nao=1, nelectron=1)
        mf = SimpleNamespace(mol=mol)
        driver = SI_driver(
            mf=mf,
            S=0.5,
            Vso=np.zeros((3, 1, 1)),
            states={},
            cal_osc=False,
            backend="cpu",
        )

        self.assertEqual(driver.make_heff().shape, (2, 2))

    def test_soc_si_end_to_end(self):
        set_backend("cpu")  # TDA 层保持 CPU；SOC 后端由 backend 参数单独控制
        mol = gto.M(atom="N 0 0 0", basis="6-31G", spin=3, verbose=0)
        mol.build()
        mf = scf.sfx2c(dft.ROKS(mol))
        mf.xc = "bhandhlyp"
        mf.max_cycle = 200
        mf.run()
        self.assertTrue(mf.converged)

        mysoc = SOCSI(mf, nstates=(2, 2, 2), backend="cpu")
        eso, vso = mysoc.kernel(printnum=2)

        dim = mysoc.mysi.dim_hso
        self.assertEqual(vso.shape, (dim, dim))
        self.assertTrue(np.all(np.diff(eso) >= -1e-10))
        self.assertTrue(np.allclose(mysoc.mysi.heff, mysoc.mysi.heff.T.conj()))
        self.assertEqual(mysoc.mysi.dmso.shape, (dim, dim, 3))


if __name__ == "__main__":
    unittest.main()
