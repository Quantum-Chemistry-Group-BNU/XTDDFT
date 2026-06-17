#!/usr/bin/env python
"""Minimal CPU XSF-TDA-down example with the multicollinear kernel."""

import os
import sys
from pathlib import Path


# CPU thread controls. Tune these for your machine.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))


import numpy as np
from pyscf import dft, gto, lib

from XTDDFT_dev.utils.backend import backend_info, set_backend

set_backend("cpu")

from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down


lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))
print("backend:", backend_info())


# ===== User parameters =====
xc = "CAM-B3LYP"
basis = "6-31G"
charge = 0
spin = 2
unit = "Angstrom"

nstates = 8
method = 1  # 1 = multicollinear XSF kernel
SA = 3
collinear_samples = 20

use_density_fit = True
max_memory = 4000
grid_level = 3
output_file = "xsf_tda_down_cpu_mcol_results.npz"
# ===========================


mol = gto.M(
    atom="""
    H 0.000000  0.934473 -0.588078
    H 0.000000 -0.934473 -0.588078
    C 0.000000  0.000000  0.000000
    O 0.000000  0.000000  1.221104
    """,
    basis=basis,
    charge=charge,
    spin=spin,
    unit=unit,
    verbose=5,
)


mf = dft.ROKS(mol)
mf.xc = "CAM-B3LYP"
if use_density_fit:
    mf = mf.density_fit()
mf.max_memory = max_memory
mf.grids.level = grid_level
mf.chkfile = "xsf_tda_down_cpu_roks_ref.chk"

mf.kernel()
if not mf.converged:
    raise RuntimeError("ROKS reference did not converge.")

s2_ref, mult_ref = mf.spin_square()
print("\nReference state")
print("------------------------------------------------------------")
print("E_ref   =", mf.e_tot)
print("<S^2>   =", s2_ref)
print("2S+1    =", mult_ref)
print("chkfile =", mf.chkfile)


xsf = XSF_TDA_down(
    mf,
    method=method,
    SA=SA,
    davidson=True,
    davidson_backend="cpu",
    collinear_samples=collinear_samples,
)

energies_ev, vectors = xsf.kernel(
    nstates=nstates,
    remove=None,
    foo=1.0,
    d_lda=0.3,
    fglobal=None,
    fit=True,
    save=False,
)

print("\nXSF-TDA-down states")
print("------------------------------------------------------------")
print(" root        E/eV        converged")
print("------------------------------------------------------------")
converged = np.asarray(xsf.converged) if xsf.converged is not None else None
for i, energy in enumerate(np.asarray(energies_ev), start=1):
    flag = bool(converged[i - 1]) if converged is not None and i <= converged.size else None
    print(f"{i:5d}  {energy:14.8f}  {flag}")

try:
    delta_s2 = np.asarray(xsf.deltaS2())
except Exception as err:
    print("\nDelta <S^2> analysis skipped:", err)
    delta_s2 = np.asarray([])
else:
    print("\nDelta <S^2>")
    print("------------------------------------------------------------")
    for i, value in enumerate(delta_s2[: xsf.nstates], start=1):
        print(f"{i:5d}  {float(value):14.8f}")

np.savez_compressed(
    output_file,
    e_ev=np.asarray(energies_ev),
    e_ha=np.asarray(xsf.e),
    vectors=np.asarray(xsf.v),
    converged=np.asarray(xsf.converged) if xsf.converged is not None else np.asarray([]),
    delta_s2=delta_s2,
    mo_energy=np.asarray(mf.mo_energy),
    mo_occ=np.asarray(mf.mo_occ),
    mo_coeff=np.asarray(mf.mo_coeff),
    e_ref=np.asarray(mf.e_tot),
    s2_ref=np.asarray(s2_ref),
    multiplicity_ref=np.asarray(mult_ref),
    xc=np.asarray(xc),
    basis=np.asarray(basis),
    method=np.asarray(method),
    SA=np.asarray(SA),
    collinear_samples=np.asarray(collinear_samples),
)

print("\nSaved results")
print("------------------------------------------------------------")
print("file    =", Path(output_file).resolve())
print("vectors =", np.asarray(vectors).shape)
print("Done.")
