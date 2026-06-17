from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiment" / "pbc" / "calc_utda_nto_from_saved_states.py"


class UtdaNtoScriptTest(unittest.TestCase):
    def test_script_loads_utda_saved_vectors_and_writes_nto_cubes(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('chk = "pre_uks_ccpVDZ.chk"', source)
        self.assertIn('results_file = "utda_method0_spin_conserving_davidson_nstates8_results.npz"', source)
        self.assertIn("state_i = None", source)
        self.assertIn("state_f = 0", source)
        self.assertIn("nroots = 5", source)
        self.assertIn("write_nto_cubes", source)
        self.assertIn("utda_method.v = vectors", source)
        self.assertNotIn("def ", source)


if __name__ == "__main__":
    unittest.main()
