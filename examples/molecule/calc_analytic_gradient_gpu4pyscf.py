#!/usr/bin/env python
"""Compare GPU4PySCF UKS-TDA analytic-gradient timings with and without DF."""
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OMP_DYNAMIC"] = "False"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

from time import perf_counter
import numpy as np
import cupy as cp
import gpu4pyscf
import gpu4pyscf.tdscf  # Registers TDA and SFTDA on UKS.
from gpu4pyscf.dft import uks
from pyscf import gto


# ===== Manually edit these parameters on the server =====
method_kind = "usc"  # "usc", "usf_down", "usf_up"; UKS reference, TDA only
xc = "b3lyp"
basis = "cc-pvtz"
spin = 2
charge = 0
cs = 20  # multicollinear samples for SF-TDA
states = [1]  # 1-based excited-state indices
verbose = 4
df_options = (False, True)  # Use (False,) or (True,) for separate processes.
auxbasis = None  # GPU4PySCF default; alternatively e.g. "def2-universal-jkfit"
grids_level = 5
scf_conv_tol = 1e-12
td_conv_tol = 1e-9
cphf_conv_tol = 1e-8
max_cycle = 200
atom = '''
    C   0.00   0.00   0.00
    H   0.00   0.95   0.15
    H   0.10  -0.75   0.65
'''
# ========================================================


def sync_gpu():
    cp.cuda.runtime.deviceSynchronize()


def run_case(with_df):
    label = "DF" if with_df else "non-DF"
    print(f"\nGPU4PySCF {method_kind}, {label}", flush=True)
    mol = gto.M(atom=atom, spin=spin, charge=charge, basis=basis,
                verbose=verbose)
    sync_gpu()
    time0 = perf_counter()
    mf = uks.UKS(mol, xc=xc)
    if with_df:
        mf = mf.density_fit(auxbasis=auxbasis)
    mf.conv_tol = scf_conv_tol
    mf.max_cycle = max_cycle
    mf.grids.level = grids_level
    mf.kernel()
    sync_gpu()
    time1 = perf_counter()
    if not mf.converged:
        raise RuntimeError(f"{label}: UKS did not converge")
    print(f"number of basis {mol.nao_nr()}")
    print(f"SCF: {time1 - time0:.3f} s")

    td = mf.TDA() if method_kind == "usc" else mf.SFTDA()
    if method_kind != "usc":
        td.extype = 0 if method_kind == "usf_up" else 1
        td.collinear = "mcol"
        td.collinear_samples = cs
    td.conv_tol = td_conv_tol
    td.max_cycle = max_cycle
    td.kernel(nstates=max(states) + 2)
    sync_gpu()
    time2 = perf_counter()
    if len(td.e) < max(states) or not bool(cp.asarray(td.converged).all().item()):
        raise RuntimeError(f"{label}: requested TDA roots did not converge")
    print(f"excited states: {time2 - time1:.3f} s")

    gradients = {}
    grad_times = {}
    for state in states:
        sync_gpu()
        start = perf_counter()
        grad = td.nuc_grad_method()
        grad.cphf_conv_tol = cphf_conv_tol
        grad.cphf_max_cycle = max_cycle
        if with_df:
            grad.auxbasis_response = True
        gradient = grad.kernel(state=state)
        sync_gpu()
        grad_times[state] = perf_counter() - start
        gradients[state] = cp.asnumpy(cp.asarray(gradient))
        print(f"Gradient state {state}: {grad_times[state]:.3f} s")
        print("analytic (Eh/Bohr):")
        print(np.array2string(gradients[state], precision=10))
    times = {"SCF": time1 - time0, "Excited states": time2 - time1,
             **{f"Gradient state {s}": grad_times[s] for s in states}}
    # Sum timed stages, excluding gradient printing and result transfers.
    times["Total (timed stages)"] = sum(times.values())
    print(f"Total (timed stages): {times['Total (timed stages)']:.3f} s")
    return times, gradients


def main():
    if method_kind not in ("usc", "usf_down", "usf_up"):
        raise ValueError("method_kind must be usc, usf_down or usf_up")
    if not states or any(s < 1 for s in states) or len(set(states)) != len(states):
        raise ValueError("states must contain distinct positive state indices")
    print(f"GPU4PySCF {gpu4pyscf.__version__}; CuPy {cp.__version__}")
    # Initialize CUDA before timing. First-use library/kernel overhead can remain;
    # repeat in fresh processes with df_options=(False,) and (True,) for benchmarks.
    cp.zeros(1)
    sync_gpu()
    results = {df: run_case(df) for df in df_options}
    if False in results and True in results:
        regular, fitted = results[False][0], results[True][0]
        print("\nWall time comparison (s); speedup = non-DF / DF")
        print(f"{'Stage':<24} {'non-DF':>12} {'DF':>12} {'Speedup':>10}")
        for stage in regular:
            print(f"{stage:<24} {regular[stage]:12.3f} {fitted[stage]:12.3f} "
                  f"{regular[stage] / fitted[stage]:10.3f}")
        for state in states:
            diff = results[True][1][state] - results[False][1][state]
            print(f"State {state}: max |DF - non-DF gradient| = "
                  f"{np.max(np.abs(diff)):.6e} Eh/Bohr (same root index)")


if __name__ == "__main__":
    main()
