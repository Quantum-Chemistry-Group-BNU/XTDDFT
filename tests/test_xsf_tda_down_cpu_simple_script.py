from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiment" / "molecule" / "run_xsf_tda_down_cpu_simple.py"


class XsfTdaDownCpuSimpleScriptTest(unittest.TestCase):
    def test_script_is_cpu_mcol_example_and_saves_vectors(self):
        self.assertTrue(SCRIPT.exists(), f"missing example script: {SCRIPT}")
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("from pyscf import dft, gto, lib", source)
        self.assertIn(
            "from XTDDFT_dev.utils.backend import backend_info, set_backend", source
        )
        self.assertIn("from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down", source)
        self.assertNotIn("cupy", source)
        self.assertNotIn("gpu4pyscf", source)
        self.assertNotIn('set_backend("gpu")', source)
        self.assertIn('set_backend("cpu")', source)
        self.assertLess(
            source.index('set_backend("cpu")'),
            source.index("from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down"),
        )
        self.assertIn("print(\"backend:\", backend_info())", source)
        self.assertIn("mf = dft.ROKS(mol)", source)
        self.assertIn('mf.xc = "CAM-B3LYP"', source)
        self.assertIn("method = 1", source)
        self.assertIn("SA = 3", source)
        self.assertIn("davidson_backend=\"cpu\"", source)
        self.assertIn("collinear_samples=collinear_samples", source)
        self.assertIn("np.savez_compressed(", source)
        self.assertIn("vectors=np.asarray(xsf.v)", source)
        self.assertIn("e_ev=np.asarray(energies_ev)", source)
        self.assertIn("e_ha=np.asarray(xsf.e)", source)
        self.assertIn('output_file = "xsf_tda_down_cpu_mcol_results.npz"', source)


if __name__ == "__main__":
    unittest.main()
