from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EMISSION_SCRIPT = (
    ROOT / "examples" / "molecule" / "calc_xsf_emission_from_saved_states.py"
)


class RoksSmdXsfScriptTest(unittest.TestCase):
    def test_emission_script_loads_saved_vectors_and_energies(self):
        source = EMISSION_SCRIPT.read_text()
        self.assertIn('vectors_file = "xsf_vectors.npy"', source)
        self.assertIn('energies_file = "xsf_energies_ev.npy"', source)
        self.assertIn("vectors = np.load(vectors_file)", source)
        self.assertIn("energies_ev = np.load(energies_file)", source)
        self.assertIn("xsf_method.v = vectors", source)
        self.assertIn("xsf_method.e = energies_ev / ha2eV", source)
        self.assertIn("tdm = xsf_method.transition_dipole_matrix()", source)
        self.assertIn("emission_osc = np.zeros_like(delta_e_ha)", source)
        self.assertIn("emission_mask = delta_e_ha > 0.0", source)
        self.assertIn("np.savez_compressed(", source)
        self.assertNotIn("def ", source)
        self.assertNotIn("resolve_existing_file", source)


if __name__ == "__main__":
    unittest.main()
