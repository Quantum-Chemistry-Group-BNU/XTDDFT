from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "experientment" / "run_xsf_tda_down_roks_smd_toluene_from_chk.py"
EMISSION_SCRIPT = ROOT / "experientment" / "calc_xsf_emission_from_saved_states.py"


class RoksSmdXsfScriptTest(unittest.TestCase):
    def test_script_uses_molecular_roks_smd_xsf_defaults(self):
        source = RUN_SCRIPT.read_text()
        self.assertIn('chk = "roks_smd_toluene_cc_pvdz.chk"', source)
        self.assertIn("from pyscf.scf import chkfile as mol_chkfile", source)
        self.assertIn("from gpu4pyscf import dft as gpubasedft", source)
        self.assertNotIn("pyscf.pbc.scf", source)
        self.assertNotIn("WINDOWS_DESKTOP", source)
        self.assertNotIn("resolve_existing_file", source)
        self.assertNotIn("def ", source)
        self.assertIn("chk_path = Path(chk)", source)
        self.assertIn("mol, scf_rec = mol_chkfile.load_scf(str(chk_path))", source)
        self.assertIn("nstates = 6", source)
        self.assertIn("method = 1", source)
        self.assertIn("SA = 3", source)
        self.assertIn("delta_a_jk_batch_size = 1", source)
        self.assertIn("delta_a_jk_batch_size=delta_a_jk_batch_size", source)
        self.assertIn("save_results = True", source)
        self.assertNotIn("calculate_TDM()", source)
        self.assertIn("mf = mf.SMD()", source)
        self.assertIn('mf.with_solvent.solvent = "toluene"', source)

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
