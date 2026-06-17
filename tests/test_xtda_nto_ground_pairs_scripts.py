from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MOL_SCRIPT = ROOT / "experientment" / "calc_xtda_nto_mol_ground_pairs.py"
PBC_SCRIPT = ROOT / "experientment" / "calc_xtda_nto_pbc_ground_pairs.py"


class XtdaNtoGroundPairsScriptsTest(unittest.TestCase):
    def test_molecular_script_has_requested_defaults(self):
        source = MOL_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('chk = "roks_b3lyp_ccpVDZ.chk"', source)
        self.assertIn('results_file = "xtda_spin_conserving_davidson_nstates8_results.npz"', source)
        self.assertIn("pairs = [(None, 0), (None, 1), (None, 2), (None, 3)]", source)
        self.assertIn("from pyscf.scf import chkfile as mol_chkfile", source)
        self.assertIn("from pyscf import dft, lib", source)
        self.assertIn("from XTDDFT_dev.XTDDFT.xtda import XTDA", source)
        self.assertIn("if mo_coeff.ndim == 3:", source)
        self.assertIn("mo_coeff = mo_coeff[0]", source)
        self.assertIn("mo_occ = mo_occ[0] + mo_occ[1]", source)
        self.assertIn("mo_coeff shape:", source)
        self.assertIn("write_nto_cubes", source)
        self.assertIn("state_i=state_i", source)
        self.assertIn("state_f=state_f", source)
        self.assertNotIn("def ", source)

    def test_pbc_script_has_requested_defaults(self):
        source = PBC_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('chk = "roks_b3lyp_ccpVDZ.chk"', source)
        self.assertIn('results_file = "xtda_spin_conserving_davidson_nstates8_results.npz"', source)
        self.assertIn("pairs = [(None, 0), (None, 1), (None, 2), (None, 3)]", source)
        self.assertIn("from pyscf.pbc.scf import chkfile as pbc_chkfile", source)
        self.assertIn("from pyscf.pbc import dft as pbcdft", source)
        self.assertIn("from XTDDFT_dev.XTDDFT.xtda import XTDA", source)
        self.assertIn("write_nto_cubes", source)
        self.assertIn("state_i=state_i", source)
        self.assertIn("state_f=state_f", source)
        self.assertNotIn("def ", source)


if __name__ == "__main__":
    unittest.main()
