# Examples

Install the project first with `pip install -e .`, then run scripts from the
repository root.

## Self-contained examples

- `molecule/run_x2c_zeeman.py`: smallest X2C Zeeman smoke example. It builds
  only one-electron operators and does not run an SCF calculation.
- `molecule/run_soc_si.py`: small N/6-31G end-to-end SOC-SI example.
- `molecule/run_xsf_tda_down_cpu_simple.py`: complete CPU XSF-TDA workflow.
- `molecule/run_xsf_tda_down_gpu_simple.py`: GPU counterpart; requires CUDA,
  CuPy and GPU4PySCF.

## File-dependent examples

- `*_from_chk.py` requires the checkpoint named in the user-parameter block.
- Saved-state analysis scripts require both the listed `.chk` and
  `.npz`/`.npy` result files.
- PBC run scripts require a PBC checkpoint and, where configured, a saved Becke
  grid `.npz`; they also require a working GPU4PySCF PBC environment.
- `*_chk_to_molden.py` and cube-generation scripts require command-line or
  in-file checkpoint paths.

Missing input files should be supplied by the user; the examples do not
download or fabricate calculation data.

## Lightweight validation

```bash
python -m compileall -q examples
python -m pytest -q tests/test_soc_si.py tests/test_x2c_zeeman.py
```
