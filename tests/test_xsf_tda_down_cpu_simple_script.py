from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiment" / "molecule" / "run_xsf_tda_down_cpu_simple.py"
GPU_SCRIPT = ROOT / "experiment" / "molecule" / "run_xsf_tda_down_gpu_simple.py"
GPU_READ_SCRIPT = (
    ROOT / "experiment" / "molecule" / "run_xsf_tda_down_gpu_simple_read.py"
)


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

    def test_gpu_script_is_gpu_mcol_example_and_saves_vectors(self):
        self.assertTrue(GPU_SCRIPT.exists(), f"missing example script: {GPU_SCRIPT}")
        source = GPU_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("import cupy as cp", source)
        self.assertIn("from gpu4pyscf import dft as gpubasedft", source)
        self.assertIn(
            "from XTDDFT_dev.utils.backend import asnumpy, backend_info, set_backend",
            source,
        )
        self.assertIn("from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down", source)
        self.assertNotIn("from pyscf import dft, gto, lib", source)
        self.assertNotIn('set_backend("cpu")', source)
        self.assertIn('set_backend("gpu")', source)
        self.assertLess(
            source.index('set_backend("gpu")'),
            source.index("from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down"),
        )
        self.assertIn("mf = gpubasedft.ROKS(mol, xc=xc)", source)
        self.assertIn("mf.mo_energy = cp.asarray(mf.mo_energy)", source)
        self.assertIn("def reference_spin_square(mf, mol):", source)
        self.assertIn('spin_source = "ideal value from mol.spin"', source)
        self.assertIn(
            "s2_ref, mult_ref, spin_source = reference_spin_square(mf, mol)", source
        )
        self.assertIn('print("source  =", spin_source)', source)
        self.assertIn("davidson_backend=\"cpu\"", source)
        self.assertIn("collinear_samples=collinear_samples", source)
        self.assertIn("cp.cuda.Stream.null.synchronize()", source)
        self.assertIn("np.savez_compressed(", source)
        self.assertIn("vectors=asnumpy(xsf.v)", source)
        self.assertIn("e_ev=np.asarray(energies_ev)", source)
        self.assertIn("e_ha=asnumpy(xsf.e)", source)
        self.assertIn('output_file = "xsf_tda_down_gpu_mcol_results.npz"', source)

    def test_gpu_read_script_uses_chk_as_initial_guess(self):
        self.assertTrue(
            GPU_READ_SCRIPT.exists(), f"missing example script: {GPU_READ_SCRIPT}"
        )
        source = GPU_READ_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("import cupy as cp", source)
        self.assertIn("from gpu4pyscf import dft as gpubasedft", source)
        self.assertIn("from pyscf.scf import chkfile as mol_chkfile", source)
        self.assertIn(
            "from XTDDFT_dev.utils.backend import asnumpy, backend_info, set_backend",
            source,
        )
        self.assertIn('set_backend("gpu")', source)
        self.assertIn('guess_chk = "xsf_tda_down_gpu_roks_ref.chk"', source)
        self.assertIn("mol, _ = mol_chkfile.load_scf(str(chk_path))", source)
        self.assertIn("dm0 = mf.from_chk(str(chk_path))", source)
        self.assertIn("mf.kernel(dm0=dm0)", source)
        self.assertNotIn("mf.mo_energy = cp.asarray(scf_rec", source)
        self.assertNotIn("mf.converged = True", source)
        self.assertIn("mf = gpubasedft.ROKS(mol, xc=xc)", source)
        self.assertIn("def reference_spin_square(mf, mol):", source)
        self.assertIn("davidson_backend=\"cpu\"", source)
        self.assertIn("collinear_samples=collinear_samples", source)
        self.assertIn("np.savez_compressed(", source)
        self.assertIn("vectors=asnumpy(xsf.v)", source)
        self.assertIn("initial_guess_chk=np.asarray(str(chk_path.resolve()))", source)
        self.assertIn(
            'output_file = "xsf_tda_down_gpu_mcol_from_chk_results.npz"', source
        )


if __name__ == "__main__":
    unittest.main()
