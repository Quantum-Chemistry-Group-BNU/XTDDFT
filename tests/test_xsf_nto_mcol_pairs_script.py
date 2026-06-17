from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiment" / "pbc" / "calc_xsf_nto_mcol_sa3_pairs_pbc.py"


class XsfNtoMcolPairsScriptTest(unittest.TestCase):
    def test_script_has_requested_pairs_and_defaults(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('chk = "roks_b3lyp_ccpVDZ.chk"', source)
        self.assertIn('results_file = "xsf_tda_down_mcol_davidson_sa3_nstates8_results.npz"', source)
        self.assertIn("pairs = [(0, 1), (0, 2), (0, 3), (0, 4)]", source)
        self.assertIn("SA = 3", source)
        self.assertIn("method = 1", source)
        self.assertIn("from pyscf.pbc.scf import chkfile as pbc_chkfile", source)
        self.assertIn("from pyscf.pbc import dft as pbcdft", source)
        self.assertIn("cell, scf_rec = pbc_chkfile.load_scf", source)
        self.assertIn("mf = pbcdft.ROKS(cell)", source)
        self.assertIn("mesh:", source)
        self.assertIn("write_nto_cubes", source)
        self.assertIn("weights = weights / np.sum(weights)", source)
        self.assertIn("NTO normalized weights among printed roots", source)
        self.assertNotIn("nroots=None", source)
        self.assertNotIn("total_strength", source)
        self.assertNotIn("captured_weight", source)
        self.assertIn("for state_i, state_f in pairs:", source)
        self.assertNotIn("def ", source)


if __name__ == "__main__":
    unittest.main()
