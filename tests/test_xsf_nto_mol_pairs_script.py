from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experientment" / "calc_xsf_nto_mol_sa3_pairs.py"


class XsfNtoMolPairsScriptTest(unittest.TestCase):
    def test_script_has_requested_pairs_and_defaults(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('chk = "scf.chk"', source)
        self.assertIn('results_file = "xsf_tda_down_roks_cc_pvdz_mcol_sa3_nstates6.npz"', source)
        self.assertIn("pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]", source)
        self.assertIn("SA = 3", source)
        self.assertIn("method = 1", source)
        self.assertIn("if mo_coeff.ndim == 3:", source)
        self.assertIn("mo_coeff = mo_coeff[0]", source)
        self.assertIn("mo_occ = mo_occ[0] + mo_occ[1]", source)
        self.assertIn("mo_coeff shape:", source)
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
