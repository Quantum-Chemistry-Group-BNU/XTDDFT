from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experientment" / "calc_homo_lumo_cubes_from_chk.py"


class HomoLumoCubeScriptTest(unittest.TestCase):
    def test_script_has_requested_defaults_and_cube_writer(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('chk = "rks_smd_toluene_cc_pvdz.chk"', source)
        self.assertIn("resolution = 0.15", source)
        self.assertIn("homo_idx", source)
        self.assertIn("lumo_idx", source)
        self.assertIn("cubegen.orbital", source)
        self.assertIn("homo.cube", source)
        self.assertIn("lumo.cube", source)


if __name__ == "__main__":
    unittest.main()
