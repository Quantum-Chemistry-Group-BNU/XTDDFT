from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiment" / "molecule" / "calc_xsf_nto_from_saved_states_mol.py"


class XsfNtoScriptTest(unittest.TestCase):
    def test_script_has_requested_defaults(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('results_file = "xsf_tda_down_roks_smd_toluene_cc_pvdz_mcol_sa3_nstates6.npz"', source)
        self.assertIn("state_i = 0", source)
        self.assertIn("state_f = 1", source)
        self.assertIn("nroots = 5", source)
        self.assertIn("resolution = 0.15", source)
        self.assertIn("write_nto_cubes", source)


if __name__ == "__main__":
    unittest.main()
